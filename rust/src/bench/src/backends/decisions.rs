// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Decisions backends: one non-streaming HTTP POST that asks a set of typed
//! questions about a state and reads a probability for every allowed answer.
//!
//! Supported variants:
//! - `vllm-systemone`: vLLM `/v1/systemone` (`state` and an object of question id to question)
//! - `openai-decisions`: OpenAI `/v1/decisions` (`input` and an array of questions)
//!
//! The questions of a run are built once from the CLI flags and travel in
//! `extra_body`, mirroring Python's `vllm/benchmarks/lib/decisions.py`.

use serde_json::{Map, Value};

use crate::backends::{RequestFuncInput, RequestFuncOutput, send_json_request};
use crate::cli::BackendKind;
use crate::error::{BenchError, Result};

/// One question of each type, so that a default run exercises every read.
/// `tests/benchmarks/test_decisions_request_func.py` keeps these files equal
/// to the defaults of Python `vllm bench serve`.
const DEFAULT_SYSTEMONE_QUESTIONS: &str = include_str!("default_systemone_questions.json");
const DEFAULT_OPENAI_DECISIONS_QUESTIONS: &str =
    include_str!("default_openai_decisions_questions.json");

/// Response from the decisions endpoints (minimal fields for usage extraction).
#[derive(serde::Deserialize)]
struct DecisionResponse {
    usage: Option<DecisionUsage>,
}

#[derive(serde::Deserialize)]
struct DecisionUsage {
    input_tokens: Option<u64>,
}

fn input_tokens(body: &[u8]) -> Option<u64> {
    serde_json::from_slice::<DecisionResponse>(body).ok()?.usage?.input_tokens
}

/// The `questions` of a request, in the shape its endpoint takes.
#[derive(Debug, Clone, PartialEq)]
enum Questions {
    /// `/v1/systemone`: question id to question, asked in this order.
    ById(Map<String, Value>),
    /// `/v1/decisions`: questions in order, each optionally named.
    InOrder(Vec<Value>),
}

impl Questions {
    fn len(&self) -> usize {
        match self {
            Self::ById(questions) => questions.len(),
            Self::InOrder(questions) => questions.len(),
        }
    }

    /// Take the first `num_questions`, repeating the questions when there are
    /// fewer. A repeated question keeps its content and gets a new id or name.
    fn resize(&self, num_questions: usize) -> Result<Self> {
        let cycle_id = |id: &str, cycle: usize| match cycle {
            0 => id.to_string(),
            _ => format!("{id}_{}", cycle + 1),
        };
        let len = self.len();

        match self {
            Self::ById(questions) => {
                let mut resized = Map::with_capacity(num_questions);
                for (i, (id, question)) in questions.iter().cycle().take(num_questions).enumerate()
                {
                    resized.insert(cycle_id(id, i / len), question.clone());
                }
                if resized.len() != num_questions {
                    return Err(BenchError::Config(
                        "Repeating the questions to reach --decision-num-questions \
                         produced duplicate question ids; rename the questions."
                            .into(),
                    ));
                }
                Ok(Self::ById(resized))
            }
            Self::InOrder(questions) => Ok(Self::InOrder(
                questions
                    .iter()
                    .cycle()
                    .take(num_questions)
                    .enumerate()
                    .map(|(i, question)| {
                        let mut question = question.clone();
                        let cycle = i / len;
                        if cycle > 0
                            && let Some(Value::String(name)) = question.get_mut("name")
                        {
                            *name = cycle_id(name, cycle);
                        }
                        question
                    })
                    .collect(),
            )),
        }
    }
}

impl From<Questions> for Value {
    fn from(questions: Questions) -> Self {
        match questions {
            Questions::ById(questions) => Value::Object(questions),
            Questions::InOrder(questions) => Value::Array(questions),
        }
    }
}

/// The decisions API that requests are sent to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionsBackend {
    Systemone,
    OpenaiDecisions,
}

impl DecisionsBackend {
    pub fn from_kind(kind: BackendKind) -> Option<Self> {
        match kind {
            BackendKind::VllmSystemone => Some(Self::Systemone),
            BackendKind::OpenaiDecisions => Some(Self::OpenaiDecisions),
            _ => None,
        }
    }

    fn kind(self) -> BackendKind {
        match self {
            Self::Systemone => BackendKind::VllmSystemone,
            Self::OpenaiDecisions => BackendKind::OpenaiDecisions,
        }
    }

    /// The request field that carries the state the questions are about.
    fn state_field(self) -> &'static str {
        match self {
            Self::Systemone => "state",
            Self::OpenaiDecisions => "input",
        }
    }

    /// Parse the questions of this backend, rejecting the other backend's
    /// shape so that a mismatched file fails before any request is sent.
    fn parse_questions(self, questions: Value) -> Result<Questions> {
        match (self, questions) {
            (Self::Systemone, Value::Object(questions))
                if !questions.is_empty() && questions.values().all(Value::is_object) =>
            {
                Ok(Questions::ById(questions))
            }
            (Self::OpenaiDecisions, Value::Array(questions))
                if !questions.is_empty() && questions.iter().all(Value::is_object) =>
            {
                Ok(Questions::InOrder(questions))
            }
            _ => {
                let expected = match self {
                    Self::Systemone => "an object of question id to question",
                    Self::OpenaiDecisions => "an array of questions",
                };
                Err(BenchError::Config(format!(
                    "--decision-questions-path must hold {expected} for --backend {}, \
                     with at least one question.",
                    self.kind()
                )))
            }
        }
    }

    fn load_questions(self, path: &str) -> Result<Questions> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            BenchError::Config(format!(
                "Failed to read --decision-questions-path '{path}': {e}"
            ))
        })?;
        let questions = serde_json::from_str(&content).map_err(|e| {
            BenchError::Config(format!(
                "Invalid JSON in --decision-questions-path '{path}': {e}"
            ))
        })?;
        self.parse_questions(questions)
    }

    fn default_questions(self) -> Questions {
        let questions = match self {
            Self::Systemone => DEFAULT_SYSTEMONE_QUESTIONS,
            Self::OpenaiDecisions => DEFAULT_OPENAI_DECISIONS_QUESTIONS,
        };
        let questions = serde_json::from_str(questions).expect("built-in questions are JSON");
        self.parse_questions(questions).expect("built-in questions match their backend")
    }

    /// Build the `questions` field that every request of a run sends.
    ///
    /// `path` is a JSON file holding the questions in this backend's request
    /// format; the built-in questions are used when unset. `num_questions` is
    /// the number of questions per request; all questions are sent when unset.
    pub fn build_questions(
        self,
        path: Option<&str>,
        num_questions: Option<usize>,
    ) -> Result<Value> {
        let questions = match path {
            Some(path) => self.load_questions(path)?,
            None => self.default_questions(),
        };
        match num_questions {
            None => Ok(questions.into()),
            Some(0) => Err(BenchError::Config(
                "--decision-num-questions must be at least 1".into(),
            )),
            Some(num_questions) => Ok(questions.resize(num_questions)?.into()),
        }
    }

    pub async fn send_request(
        &self,
        input: &RequestFuncInput,
        client: &reqwest::Client,
    ) -> Result<RequestFuncOutput> {
        let payload = self.build_payload(input);

        // Preserve client-side prompt_len as fallback if server doesn't report usage.
        let output = RequestFuncOutput {
            prompt_len: input.prompt_len,
            // Each question counts as one input, as each text of an embeddings batch does.
            num_input_sequences: match payload.get("questions") {
                Some(Value::Object(questions)) => questions.len(),
                Some(Value::Array(questions)) => questions.len(),
                _ => 0,
            },
            ..Default::default()
        };

        Ok(send_json_request(input, client, &payload, output, input_tokens).await)
    }

    fn build_payload(&self, input: &RequestFuncInput) -> Value {
        let model = input.model_name.as_deref().unwrap_or(&input.model);

        let mut payload = Map::new();
        payload.insert("model".into(), model.into());
        payload.insert(self.state_field().into(), input.prompt.as_ref().into());

        // Both endpoints reject unknown fields, so `ignore_eos` is not sent.
        if let Some(Value::Object(extra)) = input.extra_body.as_ref() {
            payload.extend(extra.clone());
        }

        Value::Object(payload)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;

    const BACKENDS: [DecisionsBackend; 2] = [
        DecisionsBackend::Systemone,
        DecisionsBackend::OpenaiDecisions,
    ];

    /// Serve `body` as one JSON response over a raw socket. Returns the
    /// endpoint URL and a handle that yields the request body.
    fn spawn_json_server(body: &'static str) -> (String, std::thread::JoinHandle<Value>) {
        use std::io::{BufRead, BufReader, Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let request = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut content_length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap();
                }
                if line == "\r\n" {
                    break;
                }
            }
            let mut request = vec![0; content_length];
            reader.read_exact(&mut request).unwrap();
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                         content-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .unwrap();
            serde_json::from_slice(&request).unwrap()
        });
        (format!("http://{addr}/"), request)
    }

    /// Decision usage reports `input_tokens`, not the `prompt_tokens` of the
    /// pooling routes, and throughput is counted per question. The request
    /// carries nothing but the fields the routes accept: both forbid extras.
    #[tokio::test]
    async fn test_request_sends_questions_and_reads_decision_usage() {
        for backend in BACKENDS {
            let questions = backend.build_questions(None, Some(5)).unwrap();
            let (api_url, request) = spawn_json_server(
                r#"{"answers":{},"usage":{"input_tokens":128,"output_tokens":5}}"#,
            );
            let input = RequestFuncInput {
                api_url,
                prompt: Arc::from("the state"),
                prompt_len: 100,
                model: "base-model".to_string(),
                model_name: Some("served-alias".to_string()),
                ignore_eos: true,
                extra_body: Some(json!({"questions": questions.clone()})),
                ..Default::default()
            };

            let output = backend.send_request(&input, &reqwest::Client::new()).await.unwrap();

            assert!(output.success, "{}", output.error);
            assert_eq!((output.prompt_len, output.num_input_sequences), (128, 5));
            assert_eq!(output.ttft, output.latency);
            assert_eq!(
                request.join().unwrap(),
                json!({
                    "model": "served-alias",
                    backend.state_field(): "the state",
                    "questions": questions,
                })
            );
        }
    }

    #[test]
    fn test_default_questions_keep_file_order() {
        let systemone = DecisionsBackend::Systemone.build_questions(None, None).unwrap();
        let ids: Vec<&String> = systemone.as_object().unwrap().keys().collect();
        assert_eq!(ids, ["topic", "actionable", "clarity"]);

        let openai = DecisionsBackend::OpenaiDecisions.build_questions(None, None).unwrap();
        let types: Vec<&Value> = openai.as_array().unwrap().iter().map(|q| &q["type"]).collect();
        assert_eq!(types, ["choice", "predicate", "score"]);
    }

    /// Repeated questions must stay distinct: `/v1/systemone` keys them by id.
    #[test]
    fn test_num_questions_truncates_and_repeats_under_new_ids() {
        let ids = |num_questions| -> Vec<String> {
            let questions =
                DecisionsBackend::Systemone.build_questions(None, Some(num_questions)).unwrap();
            questions.as_object().unwrap().keys().cloned().collect()
        };
        assert_eq!(ids(2), ["topic", "actionable"]);
        assert_eq!(
            ids(7),
            [
                "topic",
                "actionable",
                "clarity",
                "topic_2",
                "actionable_2",
                "clarity_2",
                "topic_3"
            ]
        );

        let questions = DecisionsBackend::OpenaiDecisions.build_questions(None, Some(5)).unwrap();
        let names: Vec<&Value> = questions.as_array().unwrap().iter().map(|q| &q["name"]).collect();
        assert_eq!(
            names,
            ["topic", "actionable", "clarity", "topic_2", "actionable_2"]
        );
        // A repeated question differs from its original in name only.
        assert_eq!(questions[3]["choices"], questions[0]["choices"]);
    }

    #[test]
    fn test_repeating_onto_an_existing_question_id_is_rejected() {
        let questions = DecisionsBackend::Systemone
            .parse_questions(json!({"q": {"type": "noul"}, "q_2": {"type": "noul"}}))
            .unwrap();
        let err = questions.resize(3).unwrap_err();
        assert!(format!("{err}").contains("duplicate question ids"), "{err}");
    }

    /// The two routes take different shapes, so questions written for the
    /// other backend must fail before any request is sent.
    #[test]
    fn test_questions_in_the_wrong_format_are_rejected() {
        let systemone = DecisionsBackend::Systemone.build_questions(None, None).unwrap();
        let openai = DecisionsBackend::OpenaiDecisions.build_questions(None, None).unwrap();
        for (backend, questions) in [
            (DecisionsBackend::Systemone, openai),
            (DecisionsBackend::OpenaiDecisions, systemone),
            (DecisionsBackend::Systemone, json!({})),
            (DecisionsBackend::OpenaiDecisions, json!([])),
            (DecisionsBackend::OpenaiDecisions, json!(["not a question"])),
        ] {
            let err = backend.parse_questions(questions).unwrap_err();
            assert!(
                format!("{err}").contains("--decision-questions-path must hold"),
                "{err}"
            );
        }
    }

    #[test]
    fn test_zero_num_questions_is_rejected() {
        for backend in BACKENDS {
            let err = backend.build_questions(None, Some(0)).unwrap_err();
            assert!(format!("{err}").contains("at least 1"), "{err}");
        }
    }
}
