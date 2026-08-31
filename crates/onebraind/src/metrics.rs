//! The in-memory request log behind `GET /api/internal/metrics`'s
//! `requests[]` (M8, docs/product.md §1): a head-only ring buffer of the
//! last [`REQUEST_LOG_CAPACITY`] finished generations.
//!
//! Privacy is enforced BY CONSTRUCTION (§10: content stays on the
//! machines): [`RequestEntry`] has no field that could carry prompt or
//! completion text, and the only writer is [`RequestLog::observe`]'s relay,
//! which copies exclusively from [`DoneStats`] — counts and wall-clock
//! timing — plus the model name and dialect. Nothing here is ever
//! persisted; a daemon restart starts an empty log.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{SystemTime, UNIX_EPOCH};

use onebrain_api::backend::{ApiDialect, DoneStats, FinishKind, GenerateJob, TokenEvent};
use serde::Serialize;

/// How many finished generations the ring retains (contract: "last 50").
pub const REQUEST_LOG_CAPACITY: usize = 50;

/// Relay channel capacity between the engine's event stream and the
/// client's original channel. Matches the dialect handlers' own channel
/// size, so the relay adds buffer without changing the engine's
/// backpressure behavior in kind.
const RELAY_CAPACITY: usize = 64;

/// One finished generation, head-only. Counts and timings come verbatim
/// from the terminal [`DoneStats`]; `0` timing uniformly means "not
/// measured" (docs/perf.md §1). NO prompt or completion text, ever — the
/// struct cannot express it.
#[derive(Debug, Clone, Serialize)]
pub struct RequestEntry {
    /// Log-local id (`req-1`, `req-2`, …), monotonic per daemon life.
    pub id: String,
    /// Which public API surface the request arrived through.
    pub dialect: ApiDialect,
    /// The model name the client requested.
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub ttft_ms: u64,
    /// Speculative counters (docs/perf.md §5); `0` without a draft model.
    pub drafted: u32,
    pub accepted: u32,
    /// "stop" | "length" | "abort" — the dialect-neutral finish reason.
    pub finish: FinishKind,
    /// Unix seconds at generation completion.
    pub timestamp_unix: u64,
}

/// The ring buffer. Shared between the daemon backend (writer, one relay
/// task per generation) and the metrics endpoint (reader).
#[derive(Debug, Default)]
pub struct RequestLog {
    entries: StdMutex<VecDeque<RequestEntry>>,
    next_id: AtomicU64,
}

impl RequestLog {
    pub fn new() -> Arc<RequestLog> {
        Arc::new(RequestLog::default())
    }

    /// Wrap `job` so its terminal [`TokenEvent::Done`] is recorded here on
    /// the way to the client — the single choke point every public-API
    /// generation passes through (`DaemonBackend::generate`).
    ///
    /// The relay preserves the engine's client-disconnect semantics: the
    /// moment forwarding to the original channel fails (client gone), the
    /// relay drops its receiver, so the engine host's disconnect sweep
    /// observes a closed channel exactly as it did without the relay. A
    /// generation reaped that way records nothing — only completions the
    /// engine actually finished appear in the log.
    ///
    /// Must be called on a tokio runtime (spawns the relay task).
    pub fn observe(self: &Arc<Self>, job: GenerateJob) -> GenerateJob {
        let GenerateJob {
            model,
            prompt,
            params,
            dialect,
            tx: client_tx,
        } = job;
        let (relay_tx, mut relay_rx) = tokio::sync::mpsc::channel::<TokenEvent>(RELAY_CAPACITY);
        let log = Arc::clone(self);
        let logged_model = model.clone();
        tokio::spawn(async move {
            loop {
                let event = tokio::select! {
                    // Biased recv-first: a Done already buffered when the
                    // client vanishes is still recorded below before the
                    // closed-channel arm can win the race.
                    biased;
                    e = relay_rx.recv() => match e {
                        Some(e) => e,
                        None => return,
                    },
                    // Client gone while the engine is silent (queued, or
                    // mid-prefill): dropping relay_rx here flips the
                    // engine-side sender's `is_closed()` immediately, so
                    // the disconnect sweep reaps the sequence at the next
                    // step boundary instead of after the next event.
                    _ = client_tx.closed() => return,
                };
                // Record BEFORE forwarding: the entry exists by the time
                // the client sees its `done`, and a client that vanishes
                // between finish and delivery still leaves the completed
                // generation on the books.
                if let TokenEvent::Done(stats) = &event {
                    log.record(dialect, &logged_model, stats);
                }
                if client_tx.send(event).await.is_err() {
                    // Client gone: close our end so the engine sees the
                    // disconnect (and reaps the sequence) as before.
                    return;
                }
            }
        });
        GenerateJob {
            model,
            prompt,
            params,
            dialect,
            tx: relay_tx,
        }
    }

    /// Append one finished generation, evicting the oldest beyond capacity.
    fn record(&self, dialect: ApiDialect, model: &str, stats: &DoneStats) {
        let id = format!("req-{}", self.next_id.fetch_add(1, Ordering::Relaxed) + 1);
        let entry = RequestEntry {
            id,
            dialect,
            model: model.to_string(),
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.completion_tokens,
            prefill_ms: stats.prefill_ms,
            decode_ms: stats.decode_ms,
            ttft_ms: stats.ttft_ms,
            drafted: stats.drafted,
            accepted: stats.accepted,
            finish: stats.finish,
            timestamp_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };
        let mut entries = self.entries.lock().expect("request log poisoned");
        if entries.len() == REQUEST_LOG_CAPACITY {
            entries.pop_front();
        }
        entries.push_back(entry);
    }

    /// The retained entries, newest first (the dashboard's request table
    /// reads top-down).
    pub fn snapshot(&self) -> Vec<RequestEntry> {
        self.entries
            .lock()
            .expect("request log poisoned")
            .iter()
            .rev()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onebrain_api::backend::{GenParams, PromptInput};

    fn stats(completion: u32) -> DoneStats {
        DoneStats {
            prompt_tokens: 7,
            completion_tokens: completion,
            finish: FinishKind::Stop,
            prefill_ms: 12,
            decode_ms: 34,
            ttft_ms: 5,
            drafted: 3,
            accepted: 2,
        }
    }

    #[test]
    fn ring_keeps_the_last_fifty_newest_first() {
        let log = RequestLog::new();
        for i in 0..(REQUEST_LOG_CAPACITY as u32 + 5) {
            log.record(ApiDialect::Openai, "m", &stats(i));
        }
        let snap = log.snapshot();
        assert_eq!(snap.len(), REQUEST_LOG_CAPACITY);
        // Newest first; the 5 oldest were evicted.
        assert_eq!(snap[0].completion_tokens, REQUEST_LOG_CAPACITY as u32 + 4);
        assert_eq!(snap.last().unwrap().completion_tokens, 5);
        assert_eq!(snap[0].id, format!("req-{}", REQUEST_LOG_CAPACITY + 5));
    }

    /// The privacy contract (docs/product.md §1, §10): a generation whose
    /// prompt carries a sentinel leaves an entry with real stats and NO
    /// trace of the sentinel anywhere in the serialized log — the entry
    /// type cannot carry text, and this asserts the whole pipeline honors
    /// that.
    #[tokio::test]
    async fn observed_generation_records_stats_and_never_prompt_text() {
        const SENTINEL: &str = "TOP-SECRET-PROMPT-TEXT";
        let log = RequestLog::new();
        let (client_tx, mut client_rx) = tokio::sync::mpsc::channel(8);
        let wrapped = log.observe(GenerateJob {
            model: "tinystories-260k".into(),
            prompt: PromptInput::Raw(SENTINEL.into()),
            params: GenParams::default(),
            dialect: ApiDialect::Ollama,
            tx: client_tx,
        });
        // The engine's view: stream a piece, then the terminal Done.
        wrapped
            .tx
            .send(TokenEvent::Token("hello".into()))
            .await
            .unwrap();
        wrapped.tx.send(TokenEvent::Done(stats(9))).await.unwrap();
        drop(wrapped);
        // The client still receives the full stream through the relay.
        assert!(matches!(
            client_rx.recv().await,
            Some(TokenEvent::Token(t)) if t == "hello"
        ));
        let Some(TokenEvent::Done(done)) = client_rx.recv().await else {
            panic!("relay must forward the terminal Done");
        };
        assert_eq!(done.completion_tokens, 9);
        assert!(client_rx.recv().await.is_none(), "relay closes after use");

        let snap = log.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].model, "tinystories-260k");
        assert_eq!(snap[0].prompt_tokens, 7);
        assert_eq!(snap[0].prefill_ms, 12);
        assert!(snap[0].timestamp_unix > 0);
        let json = serde_json::to_string(&snap).unwrap();
        assert!(
            !json.contains(SENTINEL),
            "prompt text must never reach the request log: {json}"
        );
        assert!(json.contains("\"dialect\":\"ollama\""), "{json}");
        assert!(json.contains("\"finish\":\"stop\""), "{json}");
    }

    /// A vanished client ends the relay (the engine must observe the
    /// closed channel to reap the sequence) without poisoning the log.
    #[tokio::test]
    async fn relay_closes_when_the_client_disconnects() {
        let log = RequestLog::new();
        let (client_tx, client_rx) = tokio::sync::mpsc::channel(1);
        let wrapped = log.observe(GenerateJob {
            model: "m".into(),
            prompt: PromptInput::Raw("hi".into()),
            params: GenParams::default(),
            dialect: ApiDialect::Openai,
            tx: client_tx,
        });
        drop(client_rx); // client gone
        wrapped
            .tx
            .send(TokenEvent::Token("x".into()))
            .await
            .expect("relay buffer accepts the first piece");
        // The relay notices the dead client and drops its receiver; the
        // engine-side sender then observes `closed` like before.
        wrapped.tx.closed().await;
        assert!(log.snapshot().is_empty(), "no Done, no entry");
    }

    /// A client that disconnects while the engine is still silent (queued,
    /// or mid-prefill — no event forwarded yet) must flip the engine-side
    /// sender to closed WITHOUT waiting for the next event, or the
    /// disconnect sweep can never reap the sequence at the next step
    /// boundary (docs/perf.md §6).
    #[tokio::test]
    async fn relay_closes_on_disconnect_before_any_event() {
        let log = RequestLog::new();
        let (client_tx, client_rx) = tokio::sync::mpsc::channel(1);
        let wrapped = log.observe(GenerateJob {
            model: "m".into(),
            prompt: PromptInput::Raw("hi".into()),
            params: GenParams::default(),
            dialect: ApiDialect::Openai,
            tx: client_tx,
        });
        drop(client_rx); // client gone; the engine has emitted NOTHING
        tokio::time::timeout(std::time::Duration::from_secs(5), wrapped.tx.closed())
            .await
            .expect("relay must observe the disconnect without an event to forward");
        assert!(log.snapshot().is_empty(), "no Done, no entry");
    }
}
