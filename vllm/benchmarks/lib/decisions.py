# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Questions sent by the decisions backends of `vllm bench serve`."""

import json
from typing import Any

SYSTEMONE_BACKEND = "vllm-systemone"
OPENAI_DECISIONS_BACKEND = "openai-decisions"
DECISION_BACKENDS = {SYSTEMONE_BACKEND, OPENAI_DECISIONS_BACKEND}

# `/v1/systemone` takes a mapping of question id to question, and
# `/v1/decisions` takes a list of questions.
DecisionQuestions = dict[str, Any] | list[dict[str, Any]]

_TOPIC_INSTRUCTIONS = "Which topic best describes the text?"
_TOPICS = ("technology", "finance", "health", "other")
_ACTIONABLE_INSTRUCTIONS = "Does the text ask the reader to do something?"
_CLARITY_INSTRUCTIONS = "How clearly is the text written?"
_CLARITY_LEVELS = ("unclear", "mixed", "clear")

# One question of each type, so that a default run exercises every read.
# `rust/src/bench/src/backends/default_*_questions.json` hold the same
# questions for `vllm-bench`.
DEFAULT_SYSTEMONE_QUESTIONS: dict[str, Any] = {
    "topic": {
        "type": "choice",
        "instructions": _TOPIC_INSTRUCTIONS,
        "criteria": dict.fromkeys(_TOPICS),
    },
    "actionable": {"type": "noul", "instructions": _ACTIONABLE_INSTRUCTIONS},
    "clarity": {
        "type": "score",
        "instructions": _CLARITY_INSTRUCTIONS,
        "criteria": list(_CLARITY_LEVELS),
    },
}

DEFAULT_OPENAI_DECISIONS_QUESTIONS: list[dict[str, Any]] = [
    {
        "type": "choice",
        "name": "topic",
        "instructions": _TOPIC_INSTRUCTIONS,
        "choices": [{"value": topic} for topic in _TOPICS],
    },
    {
        "type": "predicate",
        "name": "actionable",
        "instructions": _ACTIONABLE_INSTRUCTIONS,
    },
    {
        "type": "score",
        "name": "clarity",
        "instructions": _CLARITY_INSTRUCTIONS,
        "levels": [{"label": level} for level in _CLARITY_LEVELS],
    },
]


def _question_specs(questions: DecisionQuestions) -> list[Any]:
    return list(questions.values() if isinstance(questions, dict) else questions)


def _load_questions(backend: str, path: str) -> DecisionQuestions:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)

    if backend == SYSTEMONE_BACKEND:
        expected, valid = "an object of question id to question", dict
    else:
        expected, valid = "an array of questions", list
    if (
        not isinstance(questions, valid)
        or not questions
        or not all(isinstance(q, dict) for q in _question_specs(questions))
    ):
        raise ValueError(
            f"--decision-questions-path must hold {expected} for "
            f"--backend {backend}, with at least one question."
        )
    return questions


def _cycle_id(question_id: str, cycle: int) -> str:
    return question_id if cycle == 0 else f"{question_id}_{cycle + 1}"


def _resize_questions(
    questions: DecisionQuestions, num_questions: int
) -> DecisionQuestions:
    """Take the first `num_questions`, repeating the questions when there are
    fewer. A repeated question keeps its content and gets a new id or name."""
    if isinstance(questions, dict):
        items = list(questions.items())
        by_id: dict[str, Any] = {}
        for i in range(num_questions):
            question_id, question = items[i % len(items)]
            by_id[_cycle_id(question_id, i // len(items))] = question
        if len(by_id) != num_questions:
            raise ValueError(
                "Repeating the questions to reach --decision-num-questions "
                "produced duplicate question ids; rename the questions."
            )
        return by_id

    resized: list[dict[str, Any]] = []
    for i in range(num_questions):
        question = questions[i % len(questions)]
        cycle = i // len(questions)
        if cycle and isinstance(question.get("name"), str):
            question = {**question, "name": _cycle_id(question["name"], cycle)}
        resized.append(question)
    return resized


def build_decision_questions(
    backend: str,
    path: str | None = None,
    num_questions: int | None = None,
) -> DecisionQuestions:
    """Build the `questions` field that every request of a run sends.

    Args:
        backend: `vllm-systemone` or `openai-decisions`.
        path: A JSON file holding the questions in the backend's request
            format. The built-in questions are used when unset.
        num_questions: The number of questions per request. All questions are
            sent when unset.

    Returns:
        The questions, in the request format of `backend`.

    Raises:
        ValueError: If the file does not hold questions for `backend`, or
            `num_questions` is not positive.

    """
    questions: DecisionQuestions
    if path is not None:
        questions = _load_questions(backend, path)
    elif backend == SYSTEMONE_BACKEND:
        questions = DEFAULT_SYSTEMONE_QUESTIONS
    else:
        questions = DEFAULT_OPENAI_DECISIONS_QUESTIONS

    if num_questions is None:
        return questions
    if num_questions < 1:
        raise ValueError("--decision-num-questions must be at least 1")
    return _resize_questions(questions, num_questions)
