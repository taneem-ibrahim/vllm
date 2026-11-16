# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the decisions backends of `vllm bench serve`.

`vllm-systemone` and `openai-decisions` send one state and a set of questions
per request and read one non-streamed response. Both server routes reject
unknown fields and name their token usage differently from the pooling
routes, so these tests drive the real request functions, and the real CLI,
against a local server and validate what was sent with the server's own
request models.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import vllm.benchmarks.serve as serve_module
from vllm.benchmarks.lib.decisions import (
    DEFAULT_OPENAI_DECISIONS_QUESTIONS,
    DEFAULT_SYSTEMONE_QUESTIONS,
    OPENAI_DECISIONS_BACKEND,
    SYSTEMONE_BACKEND,
    build_decision_questions,
)
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

# These tests never touch torch, so skip the GPU cleanup fixture.
pytestmark = pytest.mark.skip_global_cleanup

MODEL = "Qwen/Qwen3-0.6B"
STATE = "My card was charged twice for one order."
ENDPOINTS = {
    SYSTEMONE_BACKEND: "/v1/systemone",
    OPENAI_DECISIONS_BACKEND: "/v1/decisions",
}
INPUT_FIELDS = {SYSTEMONE_BACKEND: "state", OPENAI_DECISIONS_BACKEND: "input"}
DEFAULT_QUESTIONS: dict[str, Any] = {
    SYSTEMONE_BACKEND: DEFAULT_SYSTEMONE_QUESTIONS,
    OPENAI_DECISIONS_BACKEND: DEFAULT_OPENAI_DECISIONS_QUESTIONS,
}
BACKENDS = list(ENDPOINTS)
RUST_BACKENDS_DIR = (
    Path(__file__).resolve().parents[2] / "rust" / "src" / "bench" / "src" / "backends"
)


def decision_response(input_tokens: int) -> dict[str, Any]:
    return {
        "answers": {},
        "usage": {"input_tokens": input_tokens, "output_tokens": 3},
    }


async def start_server(
    backend: str, *, status: int = 200, input_tokens: int = 42
) -> tuple[TestServer, list[dict[str, Any]]]:
    """Start a server for `backend`'s route that records what it receives."""
    seen: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append({"payload": await request.json(), "headers": request.headers})
        if status != 200:
            return web.Response(status=status, reason="Bad Request")
        return web.json_response(decision_response(input_tokens))

    app = web.Application()
    app.router.add_post(ENDPOINTS[backend], handler)
    server = TestServer(app)
    await server.start_server()
    return server, seen


async def _send(
    backend: str, *, status: int = 200, **request_kwargs
) -> tuple[RequestFuncOutput, dict[str, Any]]:
    server, seen = await start_server(backend, status=status)
    try:
        request_input = RequestFuncInput(
            **{
                "prompt": STATE,
                "api_url": str(server.make_url(ENDPOINTS[backend])),
                "prompt_len": 9,
                "output_len": 0,
                "model": MODEL,
                **request_kwargs,
            }
        )
        async with aiohttp.ClientSession() as session:
            output = await ASYNC_REQUEST_FUNCS[backend](request_input, session)
    finally:
        await server.close()
    return output, seen[0]


def send(backend: str, **kwargs) -> tuple[RequestFuncOutput, dict[str, Any]]:
    return asyncio.run(_send(backend, **kwargs))


def validate_request(backend: str, payload: dict[str, Any]) -> None:
    """Parse `payload` the way the server route does, questions included."""
    if backend == SYSTEMONE_BACKEND:
        from vllm.entrypoints.generate.structured_decisions.protocol import (
            StructuredDecisionRequest,
        )
        from vllm.entrypoints.generate.structured_decisions.serving import (
            parse_questions,
        )
        from vllm.entrypoints.openai.decisions.strategies import DecisionLimits

        parse_questions(
            StructuredDecisionRequest.model_validate(payload),
            DecisionLimits(max_questions=64, max_options=26),
        )
    else:
        from vllm.entrypoints.openai.decisions.adapters import make_read_question
        from vllm.entrypoints.openai.decisions.protocol import DecisionRequest

        request = DecisionRequest.model_validate(payload)
        for index, question in enumerate(request.questions):
            make_read_question(index, question)


@pytest.mark.parametrize("backend", BACKENDS)
def test_default_request_is_accepted_by_the_server_schema(backend: str):
    """Both routes forbid unknown fields, so a default run must send only
    what the route's request model accepts, `--ignore-eos` included."""
    questions = build_decision_questions(backend)
    output, seen = send(
        backend,
        extra_body={"questions": questions},
        ignore_eos=True,
        model_name="served-alias",
        request_id="bench-7",
        extra_headers={"x-team": "perf"},
    )

    assert output.success
    # The routes are behind validate_json_request, which 415s a missing type.
    assert seen["headers"]["Content-Type"] == "application/json"
    assert seen["headers"]["x-request-id"] == "bench-7"
    assert seen["headers"]["x-team"] == "perf"
    assert seen["payload"] == {
        # --served-model-name wins over --model, as on the other backends.
        "model": "served-alias",
        INPUT_FIELDS[backend]: STATE,
        "questions": questions,
    }
    validate_request(backend, seen["payload"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_output_counts_server_tokens_and_questions(backend: str):
    """Decision usage reports `input_tokens`, not the `prompt_tokens` of the
    pooling routes, and throughput is counted per question."""
    questions = build_decision_questions(backend, num_questions=5)
    output, _ = send(backend, extra_body={"questions": questions})

    assert output.success
    assert output.prompt_len == 42
    assert output.num_input_sequences == 5
    assert output.latency > 0
    assert output.ttft == output.latency


@pytest.mark.parametrize("backend", BACKENDS)
def test_error_response_is_reported_as_a_failure(backend: str):
    output, _ = send(backend, status=400)

    assert not output.success
    assert output.error == "Bad Request"


@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_prompt_is_rejected(backend: str):
    """A decision reads one state, which `--random-batch-size` cannot give."""
    with pytest.raises(ValueError, match="one text prompt per request"):
        send(backend, prompt=["first state", "second state"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_questions", [1, 3, 8])
def test_num_questions_resizes_to_distinct_valid_questions(
    backend: str, num_questions: int
):
    """Repeated questions must stay distinct: `/v1/systemone` keys them by id."""
    questions = build_decision_questions(backend, num_questions=num_questions)

    assert len(questions) == num_questions
    if isinstance(questions, dict):
        assert list(questions)[:3] == list(DEFAULT_SYSTEMONE_QUESTIONS)[:num_questions]
    else:
        names = [question["name"] for question in questions]
        assert len(set(names)) == num_questions
    validate_request(
        backend,
        {"model": MODEL, INPUT_FIELDS[backend]: STATE, "questions": questions},
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_questions_file_is_sent_as_written(backend: str, tmp_path: Path):
    questions = build_decision_questions(backend, num_questions=2)
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(questions))

    assert build_decision_questions(backend, str(path)) == questions
    assert len(build_decision_questions(backend, str(path), 5)) == 5


@pytest.mark.parametrize(
    ("backend", "content"),
    [
        (SYSTEMONE_BACKEND, DEFAULT_OPENAI_DECISIONS_QUESTIONS),
        (OPENAI_DECISIONS_BACKEND, DEFAULT_SYSTEMONE_QUESTIONS),
        (SYSTEMONE_BACKEND, {}),
        (OPENAI_DECISIONS_BACKEND, []),
        (OPENAI_DECISIONS_BACKEND, ["not a question"]),
    ],
)
def test_questions_file_in_the_wrong_format_is_rejected(
    backend: str, content: Any, tmp_path: Path
):
    """The two routes take different shapes, so a file written for the other
    backend should fail before any request is sent."""
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(content))

    with pytest.raises(ValueError, match="--decision-questions-path must hold"):
        build_decision_questions(backend, str(path))


def test_repeating_onto_an_existing_question_id_is_rejected(tmp_path: Path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"q": {"type": "noul"}, "q_2": {"type": "noul"}}))

    with pytest.raises(ValueError, match="duplicate question ids"):
        build_decision_questions(SYSTEMONE_BACKEND, str(path), 3)


def test_non_positive_num_questions_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        build_decision_questions(SYSTEMONE_BACKEND, num_questions=0)


@pytest.mark.parametrize(
    ("backend", "filename"),
    [
        (SYSTEMONE_BACKEND, "default_systemone_questions.json"),
        (OPENAI_DECISIONS_BACKEND, "default_openai_decisions_questions.json"),
    ],
)
def test_default_questions_match_the_rust_client(backend: str, filename: str):
    """`vllm-bench` must send the same default request as `vllm bench serve`."""
    rust_questions = json.loads((RUST_BACKENDS_DIR / filename).read_text())

    assert rust_questions == DEFAULT_QUESTIONS[backend]
    if backend == SYSTEMONE_BACKEND:
        assert list(rust_questions) == list(DEFAULT_SYSTEMONE_QUESTIONS)


def _cli_args(argv: list[str]):
    parser = FlexibleArgumentParser()
    serve_module.add_cli_args(parser)
    return parser.parse_args(argv)


def _write_states(path: Path, num_states: int) -> None:
    lines = [json.dumps({"prompt": f"{STATE} #{i}"}) for i in range(num_states)]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("backend", BACKENDS)
def test_cli_run_reports_question_throughput(backend: str, tmp_path: Path):
    """End to end through the CLI: every request carries the questions, and
    the result counts questions as inputs and tokens from the server usage."""
    dataset_path = tmp_path / "states.jsonl"
    _write_states(dataset_path, num_states=4)

    async def run() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        server, seen = await start_server(backend, input_tokens=100)
        try:
            args = _cli_args(
                [
                    f"--backend={backend}",
                    f"--base-url={server.make_url('')}".rstrip("/"),
                    f"--endpoint={ENDPOINTS[backend]}",
                    f"--model={MODEL}",
                    "--dataset-name=custom",
                    f"--dataset-path={dataset_path}",
                    "--skip-tokenizer-init",
                    "--num-prompts=4",
                    "--decision-num-questions=6",
                    "--ready-check-timeout-sec=5",
                    "--disable-tqdm",
                ]
            )
            return await serve_module.main_async(args), seen
        finally:
            await server.close()

    result, seen = asyncio.run(run())

    # One readiness request, then one request per state.
    assert len(seen) == 5
    for request in seen:
        assert len(request["payload"]["questions"]) == 6
        validate_request(backend, request["payload"])
    assert result["completed"] == 4
    assert result["total_input_sequences"] == 4 * 6
    assert result["total_input_tokens"] == 4 * 100
    assert result["input_sequence_throughput"] > 0
    assert "mean_e2el_ms" in result


def test_cli_rejects_decision_flags_on_other_backends(tmp_path: Path):
    """The flags only shape decision requests, so they must not be silently
    ignored on a backend that never sends questions."""
    args = _cli_args(
        [
            "--backend=openai-embeddings",
            "--endpoint=/v1/embeddings",
            f"--model={MODEL}",
            "--dataset-name=random",
            "--skip-tokenizer-init",
            "--decision-num-questions=4",
        ]
    )

    with pytest.raises(ValueError, match="require a decisions backend"):
        asyncio.run(serve_module.main_async(args))
