//! The gateway's view of the inference engine.
//!
//! One trait, two implementations: the daemon wires the real engine host
//! behind it; conformance tests use [`testing::FakeBackend`]. Generation is
//! push-based: the backend feeds [`TokenEvent`]s into the job's channel and
//! the HTTP layer turns them into SSE (OpenAI) or NDJSON (Ollama).

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::types::ChatMessage;
use crate::ApiError;

/// What the client wants generated.
#[derive(Debug, Clone)]
pub enum PromptInput {
    /// Raw completion text (`/v1/completions`, `/api/generate`).
    Raw(String),
    /// A chat to be rendered through the model's template
    /// (`/v1/chat/completions`, `/api/chat`).
    Chat(Vec<ChatMessage>),
}

/// Generation controls common to both dialects.
#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_tokens: u32,
    /// `<= 0` means greedy.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: Option<u32>,
    /// Client-supplied stop strings (matched against accumulated output).
    pub stop: Vec<String>,
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            seed: None,
            stop: Vec::new(),
        }
    }
}

/// Terminal statistics for a finished generation.
///
/// The M7 timing/speculative fields (docs/perf.md §1) are wall-clock
/// measurements the daemon copies from the engine's `GenerationStats`; `0`
/// uniformly means "not measured" (the pre-instrumentation engine path, or
/// a backend that cannot time itself). The Ollama dialect scales the
/// millisecond fields to the nanosecond `*_duration` fields real Ollama
/// emits; OpenAI `usage` stays counts-only per the OpenAI schema.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DoneStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// "stop" | "length" | "abort" — pre-mapped to dialect wording upstream.
    pub finish: FinishKind,
    /// Wall-clock MILLISECONDS spent prefilling the prompt. `0` = not
    /// measured.
    #[serde(default)]
    pub prefill_ms: u64,
    /// Wall-clock MILLISECONDS spent in the decode loop. `0` = not
    /// measured.
    #[serde(default)]
    pub decode_ms: u64,
    /// MILLISECONDS from generation start to the first emitted piece (time
    /// to first token). `0` = not measured.
    #[serde(default)]
    pub ttft_ms: u64,
    /// Tokens proposed by the speculative draft model (COUNT; `0` until the
    /// M7 `--speculative` engine work lands, and always `0` without a
    /// draft — docs/perf.md §5).
    #[serde(default)]
    pub drafted: u32,
    /// Draft tokens the target model accepted (COUNT; `<= drafted`).
    #[serde(default)]
    pub accepted: u32,
}

impl Default for DoneStats {
    /// All-zero counts and timing with `finish: Stop`. Exists so
    /// construction sites can spell only what they measured
    /// (`..DoneStats::default()`) as the struct grows across milestones;
    /// `finish` MUST always be overwritten — `Stop` here is a spelling
    /// convenience, not a semantic default.
    fn default() -> Self {
        DoneStats {
            prompt_tokens: 0,
            completion_tokens: 0,
            finish: FinishKind::Stop,
            prefill_ms: 0,
            decode_ms: 0,
            ttft_ms: 0,
            drafted: 0,
            accepted: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishKind {
    Stop,
    Length,
    Abort,
}

/// Events streamed from the backend for one generation job.
#[derive(Debug, Clone)]
pub enum TokenEvent {
    /// One rendered token piece.
    Token(String),
    /// Generation finished normally.
    Done(DoneStats),
    /// Generation failed after streaming may have begun.
    Error(String),
}

/// Which public API surface a generation arrived through. Recorded by the
/// daemon's in-memory request log at generation completion (M8 metrics,
/// docs/product.md §1) — the gateway handler is the only place the dialect
/// is known, the daemon backend the only place every generation funnels
/// through, so the job carries it between them. Never used for routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApiDialect {
    /// `/v1/chat/completions` and `/v1/completions`.
    Openai,
    /// `/api/generate` and `/api/chat`.
    Ollama,
}

/// One queued generation.
#[derive(Debug)]
pub struct GenerateJob {
    /// Client-requested model name; backends reject mismatches with
    /// [`ApiError::ModelNotLoaded`].
    pub model: String,
    pub prompt: PromptInput,
    pub params: GenParams,
    /// Which dialect surface the request came in on (metrics request log).
    pub dialect: ApiDialect,
    /// The backend must always terminate the stream with `Done` or `Error`.
    pub tx: mpsc::Sender<TokenEvent>,
}

/// Summary of a model known to the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub name: String,
    /// Size on disk in bytes (0 when unknown).
    pub size_bytes: u64,
    pub loaded: bool,
    /// GGUF metadata worth surfacing (`/api/show`): architecture, quant, etc.
    pub details: std::collections::BTreeMap<String, String>,
}

/// Progress events for a model download (`/api/pull`, internal load).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum PullEvent {
    Downloading { completed: u64, total: u64 },
    Done,
    Error { message: String },
}

/// The daemon-side contract the HTTP layer talks to.
pub trait EngineBackend: Send + Sync + 'static {
    /// Models this node can serve (cached on disk), flagged when loaded.
    fn models(&self) -> Vec<ModelSummary>;
    /// Enqueue a generation. Must return quickly; events flow via `job.tx`.
    fn generate(&self, job: GenerateJob) -> Result<(), ApiError>;
    /// Download a model into the cache (no load). Must return quickly;
    /// progress flows via `tx`, always terminated by `Done` or `Error`.
    fn pull(&self, model: String, tx: mpsc::Sender<PullEvent>) -> Result<(), ApiError>;
}

/// Test double used by the API conformance suites.
pub mod testing {
    use super::*;

    /// Deterministic wall-clock figures the fake reports in every
    /// [`DoneStats`] (milliseconds), so dialect conformance tests can assert
    /// the exact nanosecond scaling of Ollama's `*_duration` fields.
    pub const FAKE_PREFILL_MS: u64 = 12;
    /// See [`FAKE_PREFILL_MS`].
    pub const FAKE_DECODE_MS: u64 = 34;
    /// See [`FAKE_PREFILL_MS`].
    pub const FAKE_TTFT_MS: u64 = 5;

    /// The stats shape every fake generation terminates with: real token
    /// counts, the deterministic fake timing, zero speculative counters.
    fn fake_stats(completion_tokens: u32, finish: FinishKind) -> DoneStats {
        DoneStats {
            prompt_tokens: 3,
            completion_tokens,
            finish,
            prefill_ms: FAKE_PREFILL_MS,
            decode_ms: FAKE_DECODE_MS,
            ttft_ms: FAKE_TTFT_MS,
            ..DoneStats::default()
        }
    }

    /// Streams each whitespace-separated word of `script` as one token,
    /// honoring `max_tokens` and stop strings, for any requested model name
    /// contained in `models`.
    pub struct FakeBackend {
        pub script: String,
        pub model_names: Vec<String>,
    }

    impl FakeBackend {
        pub fn new(script: &str, models: &[&str]) -> Self {
            FakeBackend {
                script: script.to_string(),
                model_names: models.iter().map(|s| s.to_string()).collect(),
            }
        }
    }

    impl EngineBackend for FakeBackend {
        fn models(&self) -> Vec<ModelSummary> {
            self.model_names
                .iter()
                .map(|name| ModelSummary {
                    name: name.clone(),
                    size_bytes: 1024,
                    loaded: true,
                    details: [("general.architecture".to_string(), "fake".to_string())]
                        .into_iter()
                        .collect(),
                })
                .collect()
        }

        fn pull(&self, _model: String, tx: mpsc::Sender<PullEvent>) -> Result<(), ApiError> {
            tokio::spawn(async move {
                let _ = tx
                    .send(PullEvent::Downloading {
                        completed: 512,
                        total: 1024,
                    })
                    .await;
                let _ = tx
                    .send(PullEvent::Downloading {
                        completed: 1024,
                        total: 1024,
                    })
                    .await;
                let _ = tx.send(PullEvent::Done).await;
            });
            Ok(())
        }

        fn generate(&self, job: GenerateJob) -> Result<(), ApiError> {
            if !self.model_names.contains(&job.model) {
                return Err(ApiError::ModelNotLoaded(job.model));
            }
            let script = self.script.clone();
            tokio::spawn(async move {
                let mut sent = 0u32;
                let mut accumulated = String::new();
                for word in script.split_whitespace() {
                    if sent >= job.params.max_tokens {
                        let _ = job
                            .tx
                            .send(TokenEvent::Done(fake_stats(sent, FinishKind::Length)))
                            .await;
                        return;
                    }
                    let piece = if sent == 0 {
                        word.to_string()
                    } else {
                        format!(" {word}")
                    };
                    accumulated.push_str(&piece);
                    if job.params.stop.iter().any(|s| accumulated.contains(s)) {
                        let _ = job
                            .tx
                            .send(TokenEvent::Done(fake_stats(sent, FinishKind::Stop)))
                            .await;
                        return;
                    }
                    if job.tx.send(TokenEvent::Token(piece)).await.is_err() {
                        return; // client went away
                    }
                    sent += 1;
                }
                let _ = job
                    .tx
                    .send(TokenEvent::Done(fake_stats(sent, FinishKind::Stop)))
                    .await;
            });
            Ok(())
        }
    }
}
