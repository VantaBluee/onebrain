//! The engine-owning OS thread and the [`EngineBackend`] handle over it.
//!
//! One `std::thread` owns the loaded [`Model`] and its single [`Session`].
//! Since M7 (docs/perf.md §6) the session is created with a unified KV
//! cache and `n_seq_max = [perf] max_concurrent_requests`, and the serving
//! loop is a MULTI-SEQUENCE STEP LOOP: up to `n_seq_max` generations run
//! concurrently, each decode step batching one token per active sequence,
//! with prompt prefills chunk-interleaved FCFS between steps. The HTTP side
//! talks to it through [`EngineHost`] (cheap clonable sender) and
//! [`DaemonBackend`], the [`EngineBackend`] implementation the gateway
//! routes into.
//!
//! # Prefix/KV reuse (docs/perf.md §4)
//!
//! With `[perf] kv_reuse` (default on), a sequence slot's KV and token
//! history SURVIVE a completed generation. The next fresh request is
//! matched against the retained slots by longest common token prefix: at
//! 64+ shared tokens the divergent suffix is `seq_rm`'d and only the
//! remainder is prefilled — greedy output stays byte-identical to a cold
//! run because the retained prefix KV is exactly what a cold prefill of
//! the same tokens produces. Retained state resets on model swap, epoch
//! teardown, decode failure, and never applies to M5 retry resumes (those
//! stay full re-prefill by contract). Retained KV is cache: admission
//! evicts it oldest-first whenever the unified pool needs the room.
//!
//! # Speculative decoding (docs/perf.md §5)
//!
//! A load may carry a [`DraftRequest`]: a second, SOLO-loaded draft model
//! (excluded from the single-model invariant, unloaded before the target).
//! While exactly one greedy generation is active, each step drafts up to
//! K=8 tokens on the draft session and verifies them in ONE target decode
//! with per-position logits; the longest prefix where the target's own
//! greedy choice equals the draft is accepted and emitted (their verifying
//! decode succeeded — confirm-before-send, one batch earlier), rejected
//! positions are rolled back with a real `seq_rm`, and the draft KV is
//! resynced to the accepted stream. Greedy output is byte-identical to the
//! non-speculative path by construction: every emitted token is the
//! target's own greedy choice.
//!
//! # Generation supervision (M5, docs/resilience.md)
//!
//! Since M5 the gateway's jobs are not sent to this thread directly: they
//! flow through the daemon's [`crate::supervisor`] task, which wraps each
//! one in a [`SupervisedGenerate`] carrying an `outcome` channel. On an
//! [`EngineError::Decode`] while the loaded model is DISTRIBUTED, the host
//! sends nothing terminal on `job.tx`; it reports
//! [`GenOutcome::Interrupted`] (prompt tokens, generated tokens, pieces
//! already streamed) so the supervisor can tear the failed epoch down,
//! re-plan, and retry transparently into the same client stream. In the
//! multi-sequence world a distributed decode failure interrupts EVERY
//! active sequence at once — they share the failed batch and the torn
//! remote KV — and the supervisor re-prefills each affected sequence after
//! one re-plan (docs/perf.md task rule: retry may serialize; correctness
//! first). Solo-model decode failures keep the pre-M5 behavior: a terminal
//! [`TokenEvent::Error`] on `job.tx` — queued behind any confirmed pieces
//! still parked on a full client channel, exactly like a finish (see
//! [`fail_seq`]): no terminal ever overtakes or drops confirmed output.
//!
//! # Ordering with model-replacing messages
//!
//! `Load`/`LoadDistributed`/`Unload`/`ServeShard` act as a BARRIER: the
//! host stops reading its channel, lets every already-accepted generation
//! (active and queued) finish on the current model, then performs the swap.
//! Messages behind the barrier stay in the channel, so the pre-M7 FIFO
//! semantics are preserved exactly. Status queries never queue behind any
//! of this: [`EngineHost::loaded_model`] answers from a shared cache the
//! host updates at load/unload boundaries (docs/perf.md §6 "status
//! honesty").

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use onebrain_api::backend::{
    DoneStats, EmbedJob, EmbedResult, EngineBackend, FinishKind, GenerateJob, ModelSummary,
    PromptInput, PullEvent, TokenEvent,
};
use onebrain_api::ApiError;
use onebrain_engine::rpc::RemoteServer;
use onebrain_engine::{
    Batch, Model, ModelParams, PoolingType, Sampler, SamplerParams, SeqId, Session, SessionParams,
    Token,
};
use onebrain_models::registry::{ModelRef, Resolved};
use onebrain_models::{cache, download};
use onebrain_proto::plan::Epoch;
use serde::Serialize;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, oneshot};

use crate::supervisor::{SupervisorMsg, SupervisorTx};

/// Summary of the currently loaded model, in the wire shape the internal
/// status/load endpoints emit (`{"name","size_bytes","n_layer","n_ctx"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LoadedModel {
    pub name: String,
    pub size_bytes: u64,
    pub n_layer: i32,
    pub n_ctx: u32,
}

/// Pre-terminal progress the host forwards to the `/api/internal/load`
/// NDJSON stream. The terminal ready/error line comes from the `resp`
/// oneshot instead.
#[derive(Debug, Clone, Copy)]
pub enum LoadProgress {
    Downloading { completed: u64, total: u64 },
    Loading,
}

/// The `[perf]` levers the engine host consumes (docs/perf.md §3/§6),
/// resolved from [`crate::config::PerfSection`] by the runtime. `n_batch`
/// is not a config knob (it stays at the pre-M7 512) but lives here so
/// tests can shrink it to force multi-chunk prefills on tiny prompts.
#[derive(Debug, Clone, Copy)]
pub struct HostPerf {
    /// `[perf] max_concurrent_requests`: sequences per session (min 1).
    pub max_concurrent: u32,
    /// Logical batch size (tokens per decode call).
    pub n_batch: u32,
    /// `[perf] n_ubatch`: physical micro-batch (the engine caps it at
    /// `n_batch`).
    pub n_ubatch: u32,
    /// `[perf] prefill_overlap`: pipelined RPC prefill (docs/perf.md §3).
    /// Applied process-wide before each distributed context is created;
    /// `false` restores the sequential M3 path.
    pub prefill_overlap: bool,
    /// `[perf] kv_reuse`: keep each sequence slot's KV + token history after
    /// a completed generation and reuse the longest common prefix on later
    /// requests (docs/perf.md §4). `false` restores the reset-per-request
    /// behavior exactly.
    pub kv_reuse: bool,
}

impl Default for HostPerf {
    fn default() -> Self {
        HostPerf {
            max_concurrent: 4,
            n_batch: 512,
            n_ubatch: 512,
            prefill_overlap: true,
            kv_reuse: true,
        }
    }
}

/// The runtime-togglable subset of [`HostPerf`] (docs/perf.md §10), shared
/// between the internal API (`POST /api/internal/perf`) and the host
/// thread. `onebrain bench --cluster` flips these to construct the M3
/// baseline (`prefill_overlap = false` + `kv_reuse = false`) WITHOUT a
/// daemon restart. The host thread re-reads them at every model (re)load —
/// deliberately the same "contexts created afterwards" semantics as
/// `onebrain_engine::rpc::set_pipeline_overlap`, and because a reload drops
/// the session, retained KV-reuse state can never straddle a `kv_reuse`
/// flip.
#[derive(Debug)]
struct PerfToggles {
    prefill_overlap: AtomicBool,
    kv_reuse: AtomicBool,
}

/// Draft-model half of a load request (docs/perf.md §5): resolved and
/// loaded SOLO on this node AFTER the target model, vocabulary-checked
/// against it, and used for speculative decoding. The draft slot is
/// explicitly excluded from the single-model invariant; it is unloaded
/// BEFORE the target on every swap/teardown.
#[derive(Debug, Clone)]
pub struct DraftRequest {
    /// Registry id, `hf:…`, or local path of the draft model.
    pub reference: String,
    /// Model cache root for resolving/downloading the draft.
    pub cache_root: PathBuf,
}

/// One generation under daemon supervision (M5, docs/resilience.md
/// "Failure lifecycle"): the job itself, an optional retry prefix, and the
/// channel the host reports the attempt's outcome on.
#[derive(Debug)]
pub struct SupervisedGenerate {
    pub job: GenerateJob,
    /// `Some` on the supervisor's transparent retry: prefill this state's
    /// prompt + already-generated tokens instead of tokenizing the prompt,
    /// then continue sampling — already-sent pieces are never re-sent.
    pub resume: Option<ResumeState>,
    /// Where the host reports how the attempt ended. Best-effort: a gone
    /// receiver (daemon shutting down) is ignored.
    pub outcome: oneshot::Sender<GenOutcome>,
}

/// How one supervised generation attempt ended.
#[derive(Debug)]
pub enum GenOutcome {
    /// The job's stream was terminated on `job.tx` (`Done` or `Error`) —
    /// successful completions, validation failures, solo decode errors,
    /// and cancelled (disconnected-client) sequences.
    Finished,
    /// A decode failure against a DISTRIBUTED model: `job.tx` received no
    /// terminal event; the supervisor owns the retry-or-fail decision
    /// (docs/resilience.md step 2 "the daemon does NOT surface it yet").
    Interrupted(Box<InterruptedGen>),
}

/// Everything the supervisor needs to retry an interrupted generation into
/// the same client stream: the job (with its live `tx`), the exact token
/// prefix (prompt + generated so far), how many pieces the client already
/// received, and the stop-scan state so cross-retry stop matching works.
#[derive(Debug)]
pub struct InterruptedGen {
    pub job: GenerateJob,
    /// The original prompt's tokens (fresh attempt: tokenized here; retry:
    /// carried through unchanged).
    pub prompt_tokens: Vec<Token>,
    /// Every token generated so far, across attempts — the retry prefix is
    /// `prompt_tokens + generated_tokens`.
    pub generated_tokens: Vec<Token>,
    /// Pieces already streamed to the client (never re-sent; carried so a
    /// later attempt keeps the cumulative count). Confirmed tokens BEYOND
    /// this count had their pieces parked on a full client channel when
    /// the attempt died — the resumed attempt re-renders and delivers
    /// `generated_tokens[pieces_sent..]` before anything new, so the
    /// client's text never gains a gap.
    pub pieces_sent: usize,
    /// Stop-string scan state, carried so a stop match straddling the
    /// failure point is still caught.
    pub scan: StopScan,
    /// The engine error that interrupted the attempt (user-facing string).
    pub error: String,
}

impl InterruptedGen {
    /// Split into the job to re-issue and the retry prefix for the next
    /// attempt.
    pub fn into_retry(self) -> (GenerateJob, ResumeState) {
        (
            self.job,
            ResumeState {
                prompt_tokens: self.prompt_tokens,
                generated_tokens: self.generated_tokens,
                pieces_sent: self.pieces_sent,
                scan: self.scan,
            },
        )
    }
}

/// Retry prefix for a supervised re-issue (fields mirror
/// [`InterruptedGen`]).
#[derive(Debug)]
pub struct ResumeState {
    pub prompt_tokens: Vec<Token>,
    pub generated_tokens: Vec<Token>,
    pub pieces_sent: usize,
    pub scan: StopScan,
}

/// Messages into the engine-host thread.
#[derive(Debug)]
pub enum HostMsg {
    /// Load `reference` (registry id, `hf:…`, or local path), replacing any
    /// currently loaded model. Download progress flows through `progress`;
    /// the terminal outcome through `resp` (the error string is
    /// user-facing and includes a remedy).
    Load {
        reference: String,
        cache_root: PathBuf,
        ctx_len: u32,
        /// Speculative draft model to load alongside (docs/perf.md §5);
        /// `None` = no speculative decoding for this load.
        draft: Option<DraftRequest>,
        progress: mpsc::UnboundedSender<LoadProgress>,
        resp: oneshot::Sender<Result<LoadedModel, String>>,
    },
    /// Head side of a distributed plan (docs/distributed.md): register the
    /// bridged loopback RPC endpoints and load the model split across their
    /// devices plus (optionally) this node's own. The daemon must have the
    /// accept-once bridges pumping BEFORE sending this — registration
    /// connects out synchronously. The loaded model serves generations
    /// exactly like a solo one.
    LoadDistributed {
        /// Local GGUF part paths in load order — one entry for single-file
        /// models (the head holds every part, ADR 0004 / docs/logistics.md
        /// "Split-GGUF").
        paths: Vec<PathBuf>,
        /// The reference the user loaded by (accepted as a model name in
        /// generation requests, like the solo path).
        reference: String,
        /// Canonical display name for the loaded model.
        name: String,
        ctx_len: u32,
        /// Bridged loopback endpoints ("127.0.0.1:<port>"), in plan stage
        /// order (workers first).
        endpoints: Vec<String>,
        /// One fraction per device: remote devices in `endpoints` order,
        /// then the local device when `use_local_device`.
        tensor_split: Vec<f32>,
        use_local_device: bool,
        /// Speculative draft model, loaded SOLO on this head node even when
        /// the target is sharded (docs/perf.md §5 "draft placement v1").
        draft: Option<DraftRequest>,
        resp: oneshot::Sender<Result<LoadedModel, String>>,
    },
    /// Run one supervised generation attempt (M5): client-visible events
    /// flow through `job.tx`, the attempt outcome through `outcome` —
    /// except a distributed decode failure, which reaches ONLY `outcome`
    /// (the supervisor decides what the client sees).
    Generate(SupervisedGenerate),
    /// One embeddings request (M1 `/v1/embeddings` / `/api/embed`): served
    /// inline by the host thread with a short-lived dedicated embeddings
    /// session against the loaded model — see [`handle_embed`] for the
    /// mechanism and why in-flight generations are never disturbed. The
    /// result (or typed error) goes back on the job's own oneshot.
    Embed(EmbedJob),
    /// Ask what is loaded. Answered over a std channel so non-async callers
    /// can wait with a timeout; the host replies with `try_send`, so a
    /// caller that gave up never blocks or wedges the host. Retained for
    /// the internal API shape — [`EngineHost::loaded_model`] answers from
    /// the shared cache instead and never waits on this thread.
    Models {
        resp: std_mpsc::SyncSender<Option<LoadedModel>>,
    },
    /// Worker side of a distributed plan: this node is about to serve a
    /// pipeline shard for `epoch`. Any locally loaded model is unloaded
    /// (M3 contract: adopting a plan preempts local models — the memory is
    /// needed for the shard the head is about to push). The RPC serve
    /// threads themselves are owned by the daemon's cluster task, not this
    /// host.
    ServeShard { epoch: Epoch },
    /// Drop any loaded model + session (epoch teardown, model swap). The
    /// reply is sent AFTER the drop completes, so callers can sequence
    /// teardown (free the model while RPC bridges are still alive, then
    /// close the bridges — GGML aborts on a torn stream, ADR 0004).
    Unload { resp: oneshot::Sender<()> },
    /// Drop model + session and exit the thread.
    Shutdown,
}

/// Handle to the engine-host thread. Clones share the same thread.
#[derive(Clone)]
pub struct EngineHost {
    tx: std_mpsc::Sender<HostMsg>,
    /// Generation jobs accepted by the gateway and not yet fully handled by
    /// the supervisor (queued + in flight). Feeds [`EngineHost::is_idle`]
    /// (gates the M5 lazy re-plan and the no-job-in-flight death teardown,
    /// docs/resilience.md) and the M7 admission bound
    /// ([`EngineHost::try_start_job`]).
    jobs: Arc<AtomicUsize>,
    /// The loaded-model summary, updated by the host thread at load/unload
    /// boundaries. Status/list queries read THIS — never the host channel —
    /// so they answer instantly however busy the step loop is
    /// (docs/perf.md §6 "status honesty": fixes "model: null while busy").
    loaded: Arc<StdMutex<Option<LoadedModel>>>,
    /// Runtime overrides for the load-time `[perf]` levers (docs/perf.md
    /// §10): read by the host thread at every model (re)load, written by
    /// `POST /api/internal/perf`.
    toggles: Arc<PerfToggles>,
}

impl EngineHost {
    /// Start the engine-host thread. Join the returned handle after sending
    /// [`HostMsg::Shutdown`] and before calling `onebrain_engine::shutdown`.
    /// `decode_delay` is the test-only `[debug] decode_delay_ms` knob
    /// (docs/resilience.md): when set the host sleeps that long after every
    /// step that emitted at least one token piece; `None` (all real
    /// deployments) adds no delay anywhere. `perf` carries the `[perf]`
    /// levers the host consumes (docs/perf.md §3/§6).
    pub fn spawn(
        decode_delay: Option<Duration>,
        perf: HostPerf,
    ) -> (EngineHost, std::thread::JoinHandle<()>) {
        let (tx, rx) = std_mpsc::channel();
        let loaded = Arc::new(StdMutex::new(None));
        let loaded_for_host = loaded.clone();
        // The runtime toggles start at the config's values; the internal
        // API can override them later (docs/perf.md §10).
        let toggles = Arc::new(PerfToggles {
            prefill_overlap: AtomicBool::new(perf.prefill_overlap),
            kv_reuse: AtomicBool::new(perf.kv_reuse),
        });
        let toggles_for_host = toggles.clone();
        let handle = std::thread::Builder::new()
            .name("engine-host".into())
            .spawn(move || host_loop(rx, decode_delay, perf, toggles_for_host, loaded_for_host))
            .expect("spawning the engine host thread failed; the system is out of resources");
        (
            EngineHost {
                tx,
                jobs: Arc::new(AtomicUsize::new(0)),
                loaded,
                toggles,
            },
            handle,
        )
    }

    /// Override the runtime-togglable `[perf]` levers (docs/perf.md §10 —
    /// how `onebrain bench --cluster` constructs the M3 baseline without a
    /// daemon restart). `None` leaves a lever unchanged, so two `None`s
    /// just read the current values. Changes take effect at the NEXT model
    /// (re)load; live sessions keep the mode they were created with,
    /// exactly like the config knobs these shadow. Returns the effective
    /// `(prefill_overlap, kv_reuse)` pair.
    pub fn set_perf_toggles(
        &self,
        prefill_overlap: Option<bool>,
        kv_reuse: Option<bool>,
    ) -> (bool, bool) {
        if let Some(v) = prefill_overlap {
            self.toggles.prefill_overlap.store(v, Ordering::SeqCst);
        }
        if let Some(v) = kv_reuse {
            self.toggles.kv_reuse.store(v, Ordering::SeqCst);
        }
        self.perf_toggles()
    }

    /// The currently effective runtime toggles as
    /// `(prefill_overlap, kv_reuse)`.
    pub fn perf_toggles(&self) -> (bool, bool) {
        (
            self.toggles.prefill_overlap.load(Ordering::SeqCst),
            self.toggles.kv_reuse.load(Ordering::SeqCst),
        )
    }

    /// Send a message; `Err` means the host thread is gone (shutdown).
    pub fn send(&self, msg: HostMsg) -> Result<(), ApiError> {
        self.tx.send(msg).map_err(|_| ApiError::ShuttingDown)
    }

    /// Send a message, handing it BACK when the host thread is gone so the
    /// caller can still terminate the job's client stream itself.
    pub fn send_or_return(&self, msg: HostMsg) -> Result<(), Box<HostMsg>> {
        self.tx.send(msg).map_err(|e| Box::new(e.0))
    }

    /// A generation job entered the daemon (gateway accepted it). Paired
    /// with [`EngineHost::job_finished`] by the supervisor.
    pub fn job_started(&self) {
        self.jobs.fetch_add(1, Ordering::SeqCst);
    }

    /// Admission-bounded variant of [`EngineHost::job_started`]
    /// (docs/perf.md §6): count the job only when fewer than `limit` jobs
    /// are already in the daemon (queued + in flight). Returns `false` —
    /// without counting — when the daemon is at capacity; the caller
    /// surfaces the typed 429-equivalent.
    pub fn try_start_job(&self, limit: usize) -> bool {
        let prev = self.jobs.fetch_add(1, Ordering::SeqCst);
        if prev >= limit {
            self.jobs.fetch_sub(1, Ordering::SeqCst);
            return false;
        }
        true
    }

    /// The supervisor fully finished a job (including any retry).
    pub fn job_finished(&self) {
        self.jobs.fetch_sub(1, Ordering::SeqCst);
    }

    /// `true` when no generation job is queued or in flight (M5 idle probe:
    /// gates the lazy rejoin re-plan and the idle death teardown).
    pub fn is_idle(&self) -> bool {
        self.jobs.load(Ordering::SeqCst) == 0
    }

    /// The loaded-model summary, answered from the shared cache the host
    /// thread maintains — instant however busy the host is (docs/perf.md §6
    /// "status honesty"). The `timeout` parameter is retained for the
    /// internal API shape; nothing waits anymore.
    pub fn loaded_model(&self, _timeout: Duration) -> Option<LoadedModel> {
        self.loaded.lock().ok()?.clone()
    }
}

/// A model reference resolved to local files, downloaded if needed. Split
/// models carry every part in load order; single files carry one path.
struct ResolvedLocal {
    name: String,
    paths: Vec<PathBuf>,
    size_bytes: u64,
}

/// One pending solo load request (carried across the host loop's phases).
struct LoadReq {
    reference: String,
    cache_root: PathBuf,
    ctx_len: u32,
    draft: Option<DraftRequest>,
    progress: mpsc::UnboundedSender<LoadProgress>,
    resp: oneshot::Sender<Result<LoadedModel, String>>,
}

/// One pending distributed load request.
struct DistLoadReq {
    paths: Vec<PathBuf>,
    reference: String,
    name: String,
    ctx_len: u32,
    endpoints: Vec<String>,
    tensor_split: Vec<f32>,
    use_local_device: bool,
    draft: Option<DraftRequest>,
    resp: oneshot::Sender<Result<LoadedModel, String>>,
}

/// A load request of either flavor, stashed while the current model drops.
enum Pending {
    Solo(LoadReq),
    Dist(DistLoadReq),
}

fn host_loop(
    rx: std_mpsc::Receiver<HostMsg>,
    decode_delay: Option<Duration>,
    mut perf: HostPerf,
    toggles: Arc<PerfToggles>,
    loaded: Arc<StdMutex<Option<LoadedModel>>>,
) {
    // Small runtime owned by this thread, used only to drive downloads.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("building the download runtime failed; the system is out of resources");

    let set_loaded = |model: Option<LoadedModel>| {
        if let Ok(mut slot) = loaded.lock() {
            *slot = model;
        }
    };

    let mut pending: Option<Pending> = None;
    // Worker state: the epoch this node is serving a shard for (informational
    // — the serve threads live in the daemon's cluster task).
    let mut serving_shard: Option<u64> = None;
    'outer: loop {
        // Phase 1: nothing loaded — wait for a load request.
        set_loaded(None);
        let req = match pending.take() {
            Some(req) => req,
            None => loop {
                match rx.recv() {
                    Err(_) | Ok(HostMsg::Shutdown) => return,
                    Ok(HostMsg::Models { resp }) => {
                        let _ = resp.try_send(None);
                    }
                    Ok(HostMsg::Generate(sup)) => {
                        let message = match serving_shard {
                            Some(epoch) => format!(
                                "{} (this node is serving a pipeline shard for epoch {epoch}; \
                                 send generations to the cluster head)",
                                ApiError::NoModel
                            ),
                            None => ApiError::NoModel.to_string(),
                        };
                        let _ = sup.job.tx.blocking_send(TokenEvent::Error(message));
                        let _ = sup.outcome.send(GenOutcome::Finished);
                    }
                    Ok(HostMsg::Embed(job)) => {
                        // Same posture as Generate above: no model, typed
                        // refusal — with the shard hint when this node is
                        // a worker (embeddings, like generations, are
                        // served by the head).
                        let error = match serving_shard {
                            Some(epoch) => ApiError::Internal(format!(
                                "{} (this node is serving a pipeline shard for epoch {epoch}; \
                                 send embeddings to the cluster head)",
                                ApiError::NoModel
                            )),
                            None => ApiError::NoModel,
                        };
                        let _ = job.resp.send(Err(error));
                    }
                    Ok(HostMsg::ServeShard { epoch }) => {
                        tracing::info!(epoch = epoch.0, "engine host entering shard-serving state");
                        serving_shard = Some(epoch.0);
                    }
                    Ok(HostMsg::Unload { resp }) => {
                        let _ = resp.send(()); // nothing loaded; trivially done
                    }
                    Ok(HostMsg::Load {
                        reference,
                        cache_root,
                        ctx_len,
                        draft,
                        progress,
                        resp,
                    }) => {
                        break Pending::Solo(LoadReq {
                            reference,
                            cache_root,
                            ctx_len,
                            draft,
                            progress,
                            resp,
                        })
                    }
                    Ok(HostMsg::LoadDistributed {
                        paths,
                        reference,
                        name,
                        ctx_len,
                        endpoints,
                        tensor_split,
                        use_local_device,
                        draft,
                        resp,
                    }) => {
                        break Pending::Dist(DistLoadReq {
                            paths,
                            reference,
                            name,
                            ctx_len,
                            endpoints,
                            tensor_split,
                            use_local_device,
                            draft,
                            resp,
                        })
                    }
                }
            },
        };

        // Refresh the runtime-togglable levers (docs/perf.md §10) exactly
        // at the load boundary: the session/context about to be created
        // runs under the values the internal API last set, and a live
        // session never changes mode mid-flight.
        perf.prefill_overlap = toggles.prefill_overlap.load(Ordering::SeqCst);
        perf.kv_reuse = toggles.kv_reuse.load(Ordering::SeqCst);

        // Phase 2: obtain a loaded model (solo: resolve + download + load;
        // distributed: register bridged RPC servers + split load). The
        // `distributed` flag decides M5 decode-failure handling: only a
        // distributed model's decode failure is supervisor-retryable.
        let (model, info, reference, resp, distributed, draft_req) = match req {
            Pending::Solo(req) => {
                let LoadReq {
                    reference,
                    cache_root,
                    ctx_len,
                    draft,
                    progress,
                    resp,
                } = req;
                let resolved = match ensure_local(&rt, &reference, &cache_root, &progress) {
                    Ok(r) => r,
                    Err(message) => {
                        drop(progress);
                        let _ = resp.send(Err(message));
                        continue 'outer;
                    }
                };
                let _ = progress.send(LoadProgress::Loading);
                // Close the progress stream: everything after this point is
                // the terminal line, delivered via `resp`.
                drop(progress);

                // `load_splits` handles single files identically to `load`
                // (the loader ignores the splits list when the GGUF carries
                // no split metadata), so one call covers both shapes
                // (docs/logistics.md "Split-GGUF").
                let path_refs: Vec<&Path> = resolved.paths.iter().map(PathBuf::as_path).collect();
                let model = match Model::load_splits(&path_refs, &ModelParams::default()) {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = resp.send(Err(e.to_string()));
                        continue 'outer;
                    }
                };
                let info = LoadedModel {
                    name: resolved.name,
                    size_bytes: resolved.size_bytes,
                    n_layer: model.n_layer(),
                    n_ctx: ctx_len,
                };
                (model, info, reference, resp, false, draft)
            }
            Pending::Dist(req) => {
                let DistLoadReq {
                    paths,
                    reference,
                    name,
                    ctx_len,
                    endpoints,
                    tensor_split,
                    use_local_device,
                    draft,
                    resp,
                } = req;
                // [perf] prefill_overlap (docs/perf.md §3): the switch is
                // process-wide and only affects contexts created AFTERWARDS,
                // so it must be set before this load registers its servers.
                // `false` constructs the exact sequential M3 baseline.
                onebrain_engine::rpc::set_pipeline_overlap(perf.prefill_overlap);
                let mut servers = Vec::with_capacity(endpoints.len());
                let mut register_error = None;
                for endpoint in &endpoints {
                    match RemoteServer::register(endpoint) {
                        Ok(server) => servers.push(server),
                        Err(e) => {
                            register_error = Some(e.to_string());
                            break;
                        }
                    }
                }
                if let Some(message) = register_error {
                    let _ = resp.send(Err(message));
                    continue 'outer;
                }
                let server_refs: Vec<&RemoteServer> = servers.iter().collect();
                let path_refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
                let model = match Model::load_distributed_splits(
                    &path_refs,
                    &server_refs,
                    &tensor_split,
                    use_local_device,
                    &ModelParams::default(),
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = resp.send(Err(e.to_string()));
                        continue 'outer;
                    }
                };
                let size_bytes = paths
                    .iter()
                    .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
                    .sum();
                let info = LoadedModel {
                    name,
                    size_bytes,
                    n_layer: model.n_layer(),
                    n_ctx: ctx_len,
                };
                (model, info, reference, resp, true, draft)
            }
        };
        let mut session = match Session::new(
            &model,
            &SessionParams {
                n_ctx: info.n_ctx,
                n_batch: perf.n_batch.max(1),
                // M7 micro-batched decode (docs/perf.md §6): one session
                // serves up to max_concurrent sequences out of ONE unified
                // KV pool of n_ctx tokens — the admission headroom math in
                // the serve loop assumes exactly this layout.
                n_seq_max: perf.max_concurrent.max(1),
                kv_unified: true,
                // §3: the physical micro-batch becomes a config knob; the
                // engine caps it at n_batch.
                n_ubatch: perf.n_ubatch,
                ..SessionParams::default()
            },
        ) {
            Ok(s) => s,
            Err(e) => {
                let _ = resp.send(Err(e.to_string()));
                continue 'outer; // drops `model`
            }
        };
        // Speculative draft slot (docs/perf.md §5): loaded AFTER the target
        // (the single-model invariant is about targets; the draft is the
        // documented exception) and vocabulary-checked against it — a
        // mismatched pair would verify garbage. Declared after `session` so
        // scope-end drops free the draft BEFORE the target (contract
        // unload order), and failure fails the whole load: the user asked
        // for speculative decoding, silently dropping it would be dishonest.
        let draft_loaded: Option<(Model, String)> = match &draft_req {
            None => None,
            Some(req) => match load_draft(&rt, req, &model) {
                Ok(pair) => Some(pair),
                Err(message) => {
                    let _ = resp.send(Err(message));
                    continue 'outer; // drops session + target model
                }
            },
        };
        let mut draft_session: Option<Session<'_>> = None;
        if let Some((draft_model, draft_name)) = &draft_loaded {
            // The draft context mirrors the target's shape minus
            // concurrency: it serves exactly one stream (speculative runs
            // only while a single request is active, docs/perf.md §6).
            match Session::new(
                draft_model,
                &SessionParams {
                    n_ctx: info.n_ctx,
                    n_batch: perf.n_batch.max(1),
                    n_ubatch: perf.n_ubatch,
                    ..SessionParams::default()
                },
            ) {
                Ok(s) => draft_session = Some(s),
                Err(e) => {
                    let _ = resp.send(Err(format!(
                        "creating the draft model session for '{draft_name}' failed: {e}"
                    )));
                    continue 'outer;
                }
            }
        }
        tracing::info!(
            model = %info.name,
            n_layer = info.n_layer,
            n_ctx = info.n_ctx,
            n_seq_max = perf.max_concurrent.max(1),
            draft = draft_loaded.as_ref().map(|(_, name)| name.as_str()),
            "model loaded"
        );
        set_loaded(Some(info.clone()));
        let _ = resp.send(Ok(info.clone()));

        // Phase 3: serve generations in the multi-sequence step loop until
        // a model-replacing message (returned as the barrier's control
        // message) or shutdown.
        let mut draft_ctx = match (&mut draft_session, &draft_loaded) {
            (Some(session), Some((_, name))) => Some(DraftCtx {
                session,
                name,
                synced: None,
            }),
            _ => None,
        };
        let ctrl = serve_model(
            &rx,
            &mut session,
            &info,
            &reference,
            distributed,
            decode_delay,
            &perf,
            draft_ctx.as_mut(),
        );
        // The model is about to drop or be replaced either way: the status
        // cache must never report a model that is gone.
        set_loaded(None);
        match ctrl {
            None => return, // Shutdown (or channel gone): drops session+model
            Some(HostMsg::Unload { resp }) => {
                tracing::info!(model = %info.name, "unloading model");
                // Drop BEFORE replying: the daemon sequences epoch
                // teardown on this reply (free the model while its RPC
                // bridges still stand, then close them — ADR 0004). The
                // draft goes first (docs/perf.md §5 unload order).
                drop(draft_session);
                drop(draft_loaded);
                drop(session);
                drop(model);
                let _ = resp.send(());
                continue 'outer;
            }
            Some(HostMsg::ServeShard { epoch }) => {
                // M3 contract: adopting a plan while a local model is
                // loaded unloads it — the plan needs this node's memory.
                tracing::info!(
                    model = %info.name,
                    epoch = epoch.0,
                    "unloading local model to serve a plan shard"
                );
                serving_shard = Some(epoch.0);
                drop(draft_session);
                drop(draft_loaded);
                drop(session);
                drop(model);
                continue 'outer;
            }
            Some(HostMsg::Load {
                reference,
                cache_root,
                ctx_len,
                draft,
                progress,
                resp,
            }) => {
                tracing::info!(model = %info.name, next = %reference, "unloading for a new model");
                pending = Some(Pending::Solo(LoadReq {
                    reference,
                    cache_root,
                    ctx_len,
                    draft,
                    progress,
                    resp,
                }));
                // Loading a second model unloads the first (contract):
                // `draft_session`/`draft_loaded` drop first (draft before
                // target), then `session` and `model`, as this scope ends.
                continue 'outer;
            }
            Some(HostMsg::LoadDistributed {
                paths,
                reference,
                name,
                ctx_len,
                endpoints,
                tensor_split,
                use_local_device,
                draft,
                resp,
            }) => {
                tracing::info!(model = %info.name, next = %reference, "unloading for a distributed load");
                pending = Some(Pending::Dist(DistLoadReq {
                    paths,
                    reference,
                    name,
                    ctx_len,
                    endpoints,
                    tensor_split,
                    use_local_device,
                    draft,
                    resp,
                }));
                continue 'outer;
            }
            Some(
                HostMsg::Generate(_)
                | HostMsg::Embed(_)
                | HostMsg::Models { .. }
                | HostMsg::Shutdown,
            ) => {
                unreachable!("serve_model handles these inline")
            }
        }
    }
}

/// Resolve a reference to local files, driving any download on `rt` and
/// forwarding progress. Error strings are user-facing (with remedies).
/// Split refs (`-%05d-of-%05d.gguf`) fetch every part into its own
/// download dir (docs/logistics.md "Split-GGUF"); in the daemon the load
/// flow made everything local already, so the downloads here hit the
/// cached-manifest fast path.
fn ensure_local(
    rt: &tokio::runtime::Runtime,
    reference: &str,
    cache_root: &Path,
    progress: &mpsc::UnboundedSender<LoadProgress>,
) -> Result<ResolvedLocal, String> {
    let model_ref: ModelRef = reference.parse().map_err(|e| format!("{e}"))?;
    match model_ref.resolve().map_err(|e| format!("{e}"))? {
        Resolved::Local(path) => {
            let meta = std::fs::metadata(&path).map_err(|e| {
                format!(
                    "cannot read local model {}: {e}; check the path exists and is readable",
                    path.display()
                )
            })?;
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".to_string());
            Ok(ResolvedLocal {
                name: format!("local:{stem}"),
                paths: vec![path],
                size_bytes: meta.len(),
            })
        }
        Resolved::Remote(spec) => {
            let mut throttle = ProgressThrottle::default();
            let mut report = |completed, total| {
                if throttle.should_emit(completed, total) {
                    let _ = progress.send(LoadProgress::Downloading { completed, total });
                }
            };
            if let Some(split) = onebrain_models::split::parse_split_name(&spec.file_name) {
                let mut done = 0u64;
                for part_name in split.part_file_names() {
                    let dir = cache::split_part_dir(cache_root, &spec.cache_key, &part_name)
                        .map_err(|e| e.to_string())?;
                    let part_spec = onebrain_models::registry::DownloadSpec {
                        cache_key: spec.cache_key.clone(),
                        url: onebrain_models::split::sibling_url(&spec.url, &part_name),
                        file_name: part_name,
                    };
                    // Cross-part totals are unknown until every part has
                    // answered; report the cumulative count with total 0.
                    let path = rt
                        .block_on(download::download(&part_spec, &dir, |c, _| {
                            report(done + c, 0)
                        }))
                        .map_err(|e| e.to_string())?;
                    done += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                }
            } else {
                let dest_dir = cache_root.join(&spec.cache_key);
                rt.block_on(download::download(&spec, &dest_dir, report))
                    .map_err(|e| e.to_string())?;
            }
            let paths =
                cache::split_part_paths(cache_root, &spec.cache_key).map_err(|e| e.to_string())?;
            let size_bytes = paths
                .iter()
                .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
                .sum();
            // LRU stamp: eviction order must reflect actual use
            // (docs/logistics.md "LRU GC + pinning": touched on load).
            if let Err(err) = cache::touch(cache_root, &spec.cache_key) {
                tracing::debug!(id = %spec.cache_key, error = %err, "could not touch cache entry");
            }
            Ok(ResolvedLocal {
                name: spec.cache_key,
                paths,
                size_bytes,
            })
        }
    }
}

/// Resolve + load the speculative draft model and check its vocabulary
/// against the target's (docs/perf.md §5: "same-vocab check at load, typed
/// error on mismatch"). Returns the loaded draft and its display name; the
/// error string is user-facing with a remedy.
fn load_draft(
    rt: &tokio::runtime::Runtime,
    req: &DraftRequest,
    target: &Model,
) -> Result<(Model, String), String> {
    // Draft download progress is not streamed anywhere (the internal load
    // endpoint already made the draft local before dispatching, so this
    // resolve is a cache hit in the daemon flow); a closed channel makes
    // ensure_local's progress sends silent no-ops.
    let (progress, _) = mpsc::unbounded_channel();
    let resolved = ensure_local(rt, &req.reference, &req.cache_root, &progress)?;
    let path_refs: Vec<&Path> = resolved.paths.iter().map(PathBuf::as_path).collect();
    let draft = Model::load_splits(&path_refs, &ModelParams::default())
        .map_err(|e| format!("loading the draft model '{}' failed: {e}", resolved.name))?;
    let target_probe = vocab_fingerprint(target)
        .map_err(|e| format!("probing the target model's vocabulary failed: {e}"))?;
    let draft_probe = vocab_fingerprint(&draft).map_err(|e| {
        format!(
            "probing the draft model '{}''s vocabulary failed: {e}",
            resolved.name
        )
    })?;
    if let Some(mismatch) = vocab_mismatch(&target_probe, &draft_probe) {
        return Err(format!(
            "the draft model '{}' cannot speculate for the loaded target: {mismatch}. \
             Speculative decoding needs a same-vocabulary pair — pick a smaller model \
             from the target's own family (in the built-in registry, 'qwen3-0.6b' \
             pairs with 'qwen3-1.7b', 'qwen3-4b', and 'qwen3-32b')",
            resolved.name
        ));
    }
    tracing::info!(
        draft = %resolved.name,
        n_layer = draft.n_layer(),
        "draft model loaded for speculative decoding"
    );
    Ok((draft, resolved.name))
}

/// Text the vocabulary fingerprint runs through both tokenizers: mixed
/// case, digits, punctuation, and non-ASCII so tokenizer differences show.
const VOCAB_PROBE: &str = "The quick brown fox jumps over the lazy dog. 0123456789 \
     Once upon a time, señor Åke said: \"hello, world\"!";

/// What [`vocab_fingerprint`] samples of a model's vocabulary. The engine
/// exposes no direct vocab-table access, so compatibility is judged by
/// observable behavior: the declared tokenizer family, how the probe text
/// tokenizes (BOS/special handling included), and how those token ids
/// render back to text. Distinct vocabularies are practically certain to
/// diverge on at least one of these; a pair passing all three tokenizes and
/// renders identically on real text, which is exactly what the speculative
/// verify loop needs (docs/perf.md §5).
#[derive(Debug, Clone, PartialEq)]
struct VocabFingerprint {
    /// GGUF `tokenizer.ggml.model` (e.g. "llama", "gpt2"), when present.
    tokenizer: Option<String>,
    /// The probe text's token ids (with special tokens added).
    tokens: Vec<Token>,
    /// Each probe token id rendered back to its piece.
    pieces: Vec<String>,
}

fn vocab_fingerprint(model: &Model) -> Result<VocabFingerprint, onebrain_engine::EngineError> {
    let tokens = model.tokenize(VOCAB_PROBE, true)?;
    let pieces = tokens
        .iter()
        .map(|&t| model.token_to_piece(t))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(VocabFingerprint {
        tokenizer: model.meta("tokenizer.ggml.model"),
        tokens,
        pieces,
    })
}

/// `Some(detail)` when the two fingerprints disagree — the detail names
/// exactly what differed (the §5 contract's "typed error naming the
/// mismatch").
fn vocab_mismatch(target: &VocabFingerprint, draft: &VocabFingerprint) -> Option<String> {
    match (&target.tokenizer, &draft.tokenizer) {
        (Some(t), Some(d)) if t != d => {
            return Some(format!(
                "the models use different tokenizers ({t:?} vs {d:?})"
            ));
        }
        _ => {}
    }
    if target.tokens != draft.tokens {
        return Some(format!(
            "the vocabulary probe text tokenizes differently \
             ({} vs {} tokens, or differing ids)",
            target.tokens.len(),
            draft.tokens.len()
        ));
    }
    for (i, (t, d)) in target.pieces.iter().zip(draft.pieces.iter()).enumerate() {
        if t != d {
            return Some(format!(
                "token id {} renders differently ({t:?} vs {d:?})",
                target.tokens[i]
            ));
        }
    }
    None
}

/// Rate-limits download progress events: the first report, then every 1% of
/// the total (at least 1 MiB), plus the final byte, get through.
#[derive(Debug, Default)]
pub(crate) struct ProgressThrottle {
    last: Option<u64>,
}

impl ProgressThrottle {
    pub(crate) fn should_emit(&mut self, completed: u64, total: u64) -> bool {
        const MIN_STEP: u64 = 1024 * 1024;
        let step = (total / 100).max(MIN_STEP);
        let emit = match self.last {
            None => true,
            Some(last) => {
                completed.saturating_sub(last) >= step || (total > 0 && completed >= total)
            }
        };
        if emit {
            self.last = Some(completed);
        }
        emit
    }
}

// ---------------------------------------------------------------------------
// The multi-sequence serve loop (docs/perf.md §6)
// ---------------------------------------------------------------------------

/// llama.cpp's "random seed" sentinel: passed verbatim into a request's
/// sampler chain so a request without an explicit seed stays randomly
/// seeded.
const SEED_RANDOM: u32 = 0xFFFF_FFFF;

/// How long the serve loop waits for new messages when work exists but no
/// sequence can step (every active client's channel is full). Bounds both
/// the retry latency for held pieces and disconnect detection.
const BLOCKED_POLL: Duration = Duration::from_millis(2);

/// Minimum shared token prefix before a retained slot is reused
/// (docs/perf.md §4: "if ≥ a floor (64 tokens)"). Below it, trimming and
/// bookkeeping cost more than the skipped prefill saves.
const REUSE_FLOOR: usize = 64;

/// Draft tokens proposed per speculative verify round (docs/perf.md §5's
/// K=8), capped per round by the remaining decode budget and the batch
/// capacity.
const SPEC_K: usize = 8;

/// A validated generation waiting for a sequence slot + KV headroom.
struct PreparedGen {
    job: GenerateJob,
    outcome: Option<oneshot::Sender<GenOutcome>>,
    prompt_tokens: Vec<Token>,
    prior_generated: Vec<Token>,
    prior_pieces: usize,
    scan: StopScan,
    /// Decode budget left (`max_tokens - prior_generated`).
    remaining: usize,
    /// Unified-KV budget: `prompt + max_tokens` tokens (docs/perf.md §6
    /// admission math; identical for fresh and resumed attempts).
    budget: usize,
    /// `true` for an M5 retry re-issue: the resume contract is a FULL
    /// re-prefill, so retained-prefix matching is skipped (docs/perf.md §4
    /// "the M5 retry path resets the reuse state").
    resumed: bool,
    /// Resume only: pieces for `prior_generated[prior_pieces..]`, rendered
    /// up front — CONFIRMED tokens whose pieces the interrupted attempt
    /// still had parked on a full client channel. They seed `held` at
    /// admission so the sweep delivers them (in order, before anything
    /// new) — dropping them would leave a permanent gap in the client's
    /// text (docs/resilience.md step 4: the retry keeps streaming into the
    /// SAME response; only already-SENT pieces are never re-sent).
    unsent: Vec<String>,
    /// When the host received the job — the TTFT origin.
    arrived: Instant,
}

/// A sequence slot whose KV + token history survived its generation
/// (docs/perf.md §4). The slot's id stays in the free list (it IS free);
/// the entry records what its KV holds so the next request's longest
/// common prefix can be matched against it. Oldest-first order — eviction
/// for admission headroom drops the front.
struct RetainedSlot {
    seq: SeqId,
    /// Every token whose KV state the sequence holds, in position order
    /// (prompt + generated of the finished run).
    history: Vec<Token>,
}

/// The speculative-decoding half of a serving phase (docs/perf.md §5): the
/// draft model's session plus the bookkeeping aligning its single-sequence
/// KV with the confirmed token stream of the generation it drafts for.
struct DraftCtx<'a, 'm> {
    session: &'a mut Session<'m>,
    name: &'a str,
    /// `(uid of the generation the draft follows, how many of that
    /// generation's stream tokens sit in the draft KV)`. `None` = out of
    /// sync (fresh phase, or reset after an error); the next engagement
    /// re-prefills the draft from scratch.
    synced: Option<(u64, usize)>,
}

/// One built speculative verify round, carried from the batch-build phase
/// to post-decode processing (the decode itself and its failure handling
/// are shared with the normal step path).
struct SpecStep {
    /// Batch logits indexes: entry 0 is the pending token, then one per
    /// draft token, in position order.
    indexes: Vec<usize>,
    /// The draft-proposed tokens (may be empty when the draft immediately
    /// proposed end-of-generation — the round degenerates to a plain step).
    drafts: Vec<Token>,
    /// Tokens in the DRAFT session's KV after drafting (confirmed stream +
    /// the pending token + all but the last draft token) — the post-verify
    /// resync trims it against the newly confirmed stream.
    draft_kv: usize,
}

/// One sequence being generated in the step loop.
struct ActiveGen {
    seq: SeqId,
    /// Unique per admission within one serving phase — distinguishes
    /// successive occupants of the same sequence id (draft-KV ownership,
    /// docs/perf.md §5).
    uid: u64,
    job: GenerateJob,
    outcome: Option<oneshot::Sender<GenOutcome>>,
    prompt_tokens: Vec<Token>,
    prior_generated: Vec<Token>,
    scan: StopScan,
    /// Prefill source: prompt + prior generated (retry prefix).
    prefix: Vec<Token>,
    /// Prefix tokens already decoded (== KV length during prefill).
    /// Starts at `reused` on a §4 prefix-reuse hit.
    prefill_done: usize,
    /// Prefix tokens satisfied from a retained slot's KV instead of being
    /// decoded (docs/perf.md §4); the perf log line's prefill count is
    /// `prefix.len() - reused`.
    reused: usize,
    /// Tokens in this sequence's KV (positions are always `0..kv_len`).
    kv_len: usize,
    /// Confirmed tokens generated THIS attempt.
    attempt_tokens: Vec<Token>,
    /// Sampled but not yet confirmed by a successful decode containing it
    /// (confirm-before-send, docs/resilience.md): the token and its
    /// rendered piece.
    pending: Option<(Token, String)>,
    /// CONFIRMED pieces the client's channel had no room for, in emission
    /// order (a speculative round can confirm several at once); the
    /// sequence pauses (its decode skips steps) until they are delivered,
    /// so one slow client never stalls the other sequences.
    held: VecDeque<String>,
    /// A terminal (finish OR error) decided while `held` was non-empty:
    /// applied once the held pieces drain, so no terminal event ever
    /// overtakes a confirmed piece. The wait is bounded by client
    /// liveness — a client that disconnects is reaped by the disconnect
    /// sweep with the backlog undelivered (Aborted semantics: nobody is
    /// listening).
    terminal_after_drain: Option<DeferredTerminal>,
    remaining: usize,
    budget: usize,
    pieces_sent: usize,
    sampler: SamplerParams,
    /// This generation's own sampler chain (per-sequence sampling): built
    /// once at admission from the job's params, so interleaved sampled
    /// requests each keep their own RNG/chain state and behave exactly as
    /// if they ran alone. The M5 retry seed-restart semantics hold because
    /// a resumed attempt is a NEW admission with the original seed.
    chain: Sampler,
    /// Speculative counters (docs/perf.md §5): draft-proposed tokens and
    /// the subset the target accepted.
    drafted: u32,
    accepted: u32,
    arrived: Instant,
    admitted: Instant,
    prefill_finished: Option<Instant>,
    /// TTFT, stamped when the first piece is DELIVERED to the channel.
    ttft_ms: Option<u64>,
    /// Finished/cancelled this iteration; reaped by the retain sweep.
    done: bool,
}

/// A terminal decided while confirmed pieces were still parked on the
/// client's channel ([`ActiveGen::held`]): held pieces are CONFIRMED
/// output, so every terminal — the M8 gate covered finishes; errors follow
/// the same rule — queues behind them instead of overtaking (or dropping)
/// them. The held-retry sweep applies the terminal once the backlog
/// drains; a client that disconnects first is reaped with the backlog
/// undelivered (Aborted semantics — the documented boundary).
enum DeferredTerminal {
    Finish(FinishKind),
    Error(String),
}

/// A terminal event whose client channel was full at finish time; retried
/// each loop, dropped when the client goes away or the phase ends.
struct PendingTerminal {
    tx: mpsc::Sender<TokenEvent>,
    event: Option<TokenEvent>,
}

/// What the serve loop's message pump decided.
// One short-lived value exists at a time; boxing HostMsg here would buy
// nothing but an allocation on the barrier path.
#[allow(clippy::large_enum_variant)]
enum Pumped {
    Handled,
    /// A model-replacing message: becomes the barrier / phase end.
    Ctrl(HostMsg),
    /// Shutdown (or channel gone): exit the thread.
    Exit,
}

fn duration_ms(d: Duration) -> u64 {
    d.as_millis() as u64
}

/// Serve generations on `session` until a model-replacing message arrives
/// (returned for the caller to act on) or shutdown (`None`). See the
/// module docs for the barrier/FIFO rules.
///
/// Scheduling rule (docs/perf.md §6): speculative decoding (§5) composes
/// with this loop ONLY while a single request is active — drafting K
/// tokens steals batch slots the other sequences would use, so with 2+
/// active sequences the speculative path stands down to plain per-sequence
/// stepping (and resumes when the field thins back to one).
#[allow(clippy::too_many_arguments)]
fn serve_model(
    rx: &std_mpsc::Receiver<HostMsg>,
    session: &mut Session<'_>,
    info: &LoadedModel,
    loaded_reference: &str,
    distributed: bool,
    decode_delay: Option<Duration>,
    perf: &HostPerf,
    mut draft: Option<&mut DraftCtx<'_, '_>>,
) -> Option<HostMsg> {
    let n_seq_max = session.n_seq_max().max(1) as usize;
    let n_batch = perf.n_batch.max(1) as usize;
    let n_ctx = info.n_ctx as usize;
    // One reusable batch for every step. Allocation failure (OOM-class,
    // never seen in practice) leaves the host serving control traffic;
    // generations fail typed at prepare time.
    let (mut batch, batch_error) = match Batch::new(n_batch, n_seq_max) {
        Ok(b) => (Some(b), None),
        Err(e) => {
            tracing::error!(error = %e, "token batch allocation failed; generations will error");
            (None, Some(e.to_string()))
        }
    };

    let mut active: Vec<ActiveGen> = Vec::new();
    let mut queue: VecDeque<PreparedGen> = VecDeque::new();
    let mut outbox: Vec<PendingTerminal> = Vec::new();
    let mut free: Vec<SeqId> = (0..n_seq_max as SeqId).rev().collect();
    // Sequence slots whose KV survived a completed generation for §4
    // prefix reuse (empty forever when the knob is off). Invariant: every
    // retained seq id is also in `free`.
    let mut retained: Vec<RetainedSlot> = Vec::new();
    // Monotonic per-admission id (draft-KV ownership across slot reuse).
    let mut next_uid: u64 = 0;
    // Set when the draft session errored: speculative decoding stands down
    // for the rest of the phase (generations continue on the plain path).
    let mut spec_disabled = false;
    let mut ctrl: Option<HostMsg> = None;

    'serve: loop {
        // 1. Retry parked terminal events (finish lines whose client
        // channel was momentarily full).
        outbox.retain_mut(|p| {
            let Some(event) = p.event.take() else {
                return false;
            };
            match p.tx.try_send(event) {
                Ok(()) => false,
                Err(TrySendError::Full(event)) => {
                    p.event = Some(event);
                    true
                }
                Err(TrySendError::Closed(_)) => false,
            }
        });

        // 2. Retry held token pieces (in order); a drained backlog unblocks
        // the sequence's stepping and applies any finish that was waiting
        // behind it (a terminal event must never overtake a held piece).
        for g in active.iter_mut() {
            while let Some(piece) = g.held.pop_front() {
                match g.job.tx.try_send(TokenEvent::Token(piece)) {
                    Ok(()) => {
                        g.pieces_sent += 1;
                        if g.ttft_ms.is_none() {
                            g.ttft_ms = Some(duration_ms(g.arrived.elapsed()));
                        }
                    }
                    Err(TrySendError::Full(TokenEvent::Token(piece))) => {
                        g.held.push_front(piece);
                        break;
                    }
                    Err(_) => break, // closed: the disconnect sweep reaps it
                }
            }
            if g.held.is_empty() && !g.done {
                match g.terminal_after_drain.take() {
                    Some(DeferredTerminal::Finish(finish)) => {
                        // §4 retention requires the slot's KV to mirror
                        // prefix + attempt exactly; a spent-budget resume
                        // finishes here without ever prefilling, so its
                        // empty KV must be released, never retained.
                        let kv_complete = g.prefill_done >= g.prefix.len();
                        finish_seq(
                            session,
                            &mut free,
                            &mut outbox,
                            g,
                            finish,
                            (perf.kv_reuse && kv_complete).then_some(&mut retained),
                        );
                    }
                    Some(DeferredTerminal::Error(message)) => {
                        // Error-after-drain: the backlog is delivered, so
                        // fail_seq now terminates immediately.
                        fail_seq(session, &mut free, &mut outbox, g, message);
                    }
                    None => {}
                }
            }
        }
        active.retain(|g| !g.done);

        // 3. Message pump. Once a model-replacing message is taken the
        // channel is left alone (FIFO barrier: everything behind it is
        // handled by the next phase), so `ctrl` gates the whole pump.
        if ctrl.is_none() {
            if active.is_empty() && queue.is_empty() && outbox.is_empty() {
                // Fully idle: park on the channel.
                match rx.recv() {
                    Err(_) => return None,
                    Ok(msg) => match pump_msg(
                        msg,
                        session,
                        info,
                        loaded_reference,
                        distributed,
                        &batch_error,
                        &mut queue,
                    ) {
                        Pumped::Handled => {}
                        Pumped::Ctrl(msg) => ctrl = Some(msg),
                        Pumped::Exit => return None,
                    },
                }
            }
            while ctrl.is_none() {
                match rx.try_recv() {
                    Ok(msg) => match pump_msg(
                        msg,
                        session,
                        info,
                        loaded_reference,
                        distributed,
                        &batch_error,
                        &mut queue,
                    ) {
                        Pumped::Handled => {}
                        Pumped::Ctrl(msg) => ctrl = Some(msg),
                        Pumped::Exit => return None,
                    },
                    Err(std_mpsc::TryRecvError::Empty) => break,
                    Err(std_mpsc::TryRecvError::Disconnected) => return None,
                }
            }
        }

        // 4. Disconnect sweep (docs/perf.md §6 cancellation): a gone client
        // frees its sequence at the next step boundary — mid-prefill
        // included — never only after the generation would have ended.
        for g in active.iter_mut() {
            if !g.done && g.job.tx.is_closed() {
                tracing::debug!(seq = g.seq, "client disconnected; freeing the sequence");
                cancel_seq(session, &mut free, g);
            }
        }
        active.retain(|g| !g.done);
        queue.retain_mut(|p| {
            if p.job.tx.is_closed() {
                if let Some(outcome) = p.outcome.take() {
                    let _ = outcome.send(GenOutcome::Finished);
                }
                return false;
            }
            true
        });

        // 5. Admission (docs/perf.md §6): FCFS, one sequence slot and
        // enough unified-KV headroom (prompt + max_tokens) required. The
        // queue keeps FIFO order strictly — a large request at the head
        // waits for headroom rather than being overtaken (no starvation).
        // Admission continues while a barrier is pending: already-accepted
        // jobs were dequeued before the control message, so FIFO says they
        // complete first.
        while active.len() < n_seq_max && !queue.is_empty() {
            let used: usize = active.iter().map(|g| g.budget).sum();
            let front = queue.front().expect("checked non-empty");
            let front_budget = front.budget;

            // §4 prefix-reuse matching, fresh requests only (the M5 retry
            // contract keeps resumes full-prefill). Policy (documented,
            // deliberately simple): the retained slot with the longest
            // common token prefix wins, floor 64; the LCP is capped one
            // token below the new prefix so the tail always re-decodes —
            // its logits seed the first sample.
            let mut reuse: Option<(usize, usize)> = None; // (retained idx, lcp)
            if perf.kv_reuse && !front.resumed {
                for (idx, slot) in retained.iter().enumerate() {
                    let lcp = common_prefix_len(&slot.history, &front.prompt_tokens)
                        .min(front.prompt_tokens.len().saturating_sub(1));
                    if lcp >= REUSE_FLOOR && reuse.is_none_or(|(_, best)| lcp > best) {
                        reuse = Some((idx, lcp));
                    }
                }
            }

            // Headroom (§6 admission math extended by §4): active budgets,
            // the new job's budget, and the KV the OTHER retained slots
            // still hold must all fit the unified pool. Retained KV is
            // cache — evict oldest-first until the job fits, so reuse can
            // never block admission.
            loop {
                let reuse_seq = reuse.map(|(idx, _)| retained[idx].seq);
                let cached: usize = retained
                    .iter()
                    .filter(|r| Some(r.seq) != reuse_seq)
                    .map(|r| r.history.len())
                    .sum();
                if used + front_budget + cached <= n_ctx {
                    break;
                }
                let Some(victim_pos) = retained.iter().position(|r| Some(r.seq) != reuse_seq)
                else {
                    break;
                };
                let victim = retained.remove(victim_pos);
                if let Some((idx, lcp)) = reuse {
                    if victim_pos < idx {
                        reuse = Some((idx - 1, lcp));
                    }
                }
                let _ = session.seq_rm(victim.seq, -1, -1);
                tracing::debug!(
                    seq = victim.seq,
                    tokens = victim.history.len(),
                    "evicted a retained KV prefix for admission headroom"
                );
            }
            {
                let reuse_seq = reuse.map(|(idx, _)| retained[idx].seq);
                let cached: usize = retained
                    .iter()
                    .filter(|r| Some(r.seq) != reuse_seq)
                    .map(|r| r.history.len())
                    .sum();
                if used + front_budget + cached > n_ctx {
                    break; // FIFO: the head waits; nothing overtakes it
                }
            }

            let p = queue.pop_front().expect("checked non-empty");
            // Per-sequence sampler chain: built once per admission from the
            // job's own params, before a slot is taken — an allocation
            // failure (OOM-class) terminates the job typed without
            // occupying anything.
            let sampler = SamplerParams {
                temperature: p.job.params.temperature,
                top_p: p.job.params.top_p,
                top_k: p.job.params.top_k,
                seed: p.job.params.seed.unwrap_or(SEED_RANDOM),
            };
            let chain = match Sampler::new(&sampler) {
                Ok(chain) => chain,
                Err(e) => {
                    // The channel may carry earlier traffic (resume), so
                    // park the terminal like any other full-channel case.
                    deliver_terminal(&mut outbox, &p.job.tx, TokenEvent::Error(e.to_string()));
                    if let Some(outcome) = p.outcome {
                        let _ = outcome.send(GenOutcome::Finished);
                    }
                    continue;
                }
            };
            // Slot choice: the reuse match wins its own slot; otherwise
            // prefer a free slot with no retained cache (preserving other
            // caches); when every free slot is a cache, sacrifice the
            // oldest.
            let (seq, reused) = match reuse {
                Some((idx, lcp)) => {
                    let slot = retained.remove(idx);
                    free.retain(|s| *s != slot.seq);
                    // Trim the divergent suffix; the prefix KV stays
                    // (docs/perf.md §4). A memory that cannot drop a
                    // partial range (recurrent/SWA) falls back to a full
                    // reset — correctness first, reuse is an optimization.
                    match session.seq_rm(slot.seq, lcp as i32, -1) {
                        Ok(()) => {
                            tracing::debug!(
                                seq = slot.seq,
                                reused = lcp,
                                prompt = p.prompt_tokens.len(),
                                "prefix reuse hit; decoding only the suffix"
                            );
                            (slot.seq, lcp)
                        }
                        Err(e) => {
                            tracing::warn!(
                                seq = slot.seq,
                                error = %e,
                                "partial KV trim unsupported; falling back to a full prefill"
                            );
                            let _ = session.seq_rm(slot.seq, -1, -1);
                            (slot.seq, 0)
                        }
                    }
                }
                None => {
                    let pos = free
                        .iter()
                        .rposition(|s| !retained.iter().any(|r| r.seq == *s))
                        .unwrap_or_else(|| {
                            // Every free slot carries a cache: evict the
                            // oldest and take its slot.
                            let oldest = retained.remove(0);
                            free.iter()
                                .position(|s| *s == oldest.seq)
                                .expect("retained seq ids are free")
                        });
                    let seq = free.swap_remove(pos);
                    retained.retain(|r| r.seq != seq);
                    // Isolation: a non-reused slot starts with no KV state.
                    let _ = session.seq_rm(seq, -1, -1);
                    (seq, 0)
                }
            };
            let uid = next_uid;
            next_uid += 1;
            active.push(admit(p, sampler, chain, seq, reused, uid));
            if let Some(d) = draft.as_deref() {
                let g = active.last().expect("just pushed");
                if g.sampler.temperature > 0.0 {
                    // §5: non-greedy sampling with a draft is out of scope
                    // (rejection sampling deferred) — honest UX: say so
                    // once and run the plain target path.
                    tracing::info!(
                        draft = %d.name,
                        "request samples with temperature > 0; speculative decoding \
                         stands down for it (greedy-only in M7)"
                    );
                }
            }
        }

        // 6. Barrier completion: every accepted generation has finished.
        // Parked terminal events for stalled clients are dropped (their
        // outcomes were already reported; a client that stopped reading
        // loses its final line rather than blocking a model swap).
        if ctrl.is_some() && active.is_empty() && queue.is_empty() {
            return ctrl;
        }

        // 7. Build one step. Speculative round (docs/perf.md §5) when
        // eligible — exactly one greedy generation active, prefilled, with
        // budget beyond its pending token — otherwise the plain step: one
        // pending token per decodable sequence plus one FCFS prefill chunk
        // (docs/perf.md §6: a sequence's decode never waits on another's
        // prefill by more than sharing the step).
        let Some(batch) = batch.as_mut() else {
            // No batch (allocation failed at phase start): nothing can
            // ever step; jobs were rejected at prepare. Idle-park happens
            // in the pump above.
            continue 'serve;
        };
        batch.clear();
        // (active index, batch logits index) per stepped decode token.
        let mut decodes: Vec<(usize, usize)> = Vec::new();
        // (active index, chunk length, logits index of the prefix tail).
        let mut prefill: Option<(usize, usize, Option<usize>)> = None;
        // A built speculative verify round (post-processed after decode).
        let mut spec: Option<SpecStep> = None;
        // A batch-push failure is unreachable by construction (capacity ≥
        // n_seq_max, ids < n_seq_max); if it ever fires, only the affected
        // sequence fails — typed — and the step is rebuilt.
        let mut push_failure: Option<(usize, String)> = None;
        if !spec_disabled && active.len() == 1 {
            let g = &mut active[0];
            if g.held.is_empty()
                && g.terminal_after_drain.is_none()
                && g.prefill_done >= g.prefix.len()
                && g.pending.is_some()
                && g.remaining > 1
                && g.sampler.temperature <= 0.0
            {
                if let Some(d) = draft.as_deref_mut() {
                    match build_spec_step(d, batch, g) {
                        Ok(step) => spec = Some(step),
                        Err(e) => {
                            // A draft-side failure never fails the
                            // generation — the draft is an accelerator.
                            // Stand down for the phase and rebuild plain.
                            tracing::warn!(
                                draft = %d.name,
                                error = %e,
                                "draft session failed; speculative decoding disabled \
                                 for this model phase"
                            );
                            spec_disabled = true;
                            d.synced = None;
                            batch.clear();
                            continue 'serve;
                        }
                    }
                }
            }
        }
        if spec.is_none() {
            for (i, g) in active.iter().enumerate() {
                if !g.held.is_empty()
                    || g.terminal_after_drain.is_some()
                    || g.prefill_done < g.prefix.len()
                {
                    continue;
                }
                if let Some((tok, _)) = &g.pending {
                    match batch.push(*tok, g.kv_len as i32, g.seq, true) {
                        Ok(index) => decodes.push((i, index)),
                        Err(e) => {
                            push_failure = Some((i, e.to_string()));
                            break;
                        }
                    }
                }
            }
        }
        if spec.is_none() && push_failure.is_none() {
            for (i, g) in active.iter().enumerate() {
                if !g.held.is_empty()
                    || g.terminal_after_drain.is_some()
                    || g.prefill_done >= g.prefix.len()
                {
                    continue;
                }
                let space = n_batch.saturating_sub(batch.len());
                if space == 0 {
                    break;
                }
                let end = (g.prefill_done + space).min(g.prefix.len());
                let mut tail_index = None;
                for (j, &tok) in g.prefix[g.prefill_done..end].iter().enumerate() {
                    let is_tail = g.prefill_done + j + 1 == g.prefix.len();
                    match batch.push(tok, (g.kv_len + j) as i32, g.seq, is_tail) {
                        Ok(index) => {
                            if is_tail {
                                tail_index = Some(index);
                            }
                        }
                        Err(e) => {
                            push_failure = Some((i, e.to_string()));
                            break;
                        }
                    }
                }
                if push_failure.is_none() {
                    prefill = Some((i, end - g.prefill_done, tail_index));
                }
                break; // exactly one prefill chunk per step (FCFS)
            }
        }
        if let Some((i, message)) = push_failure {
            let g = &mut active[i];
            fail_seq(session, &mut free, &mut outbox, g, message);
            active.retain(|g| !g.done);
            continue 'serve;
        }

        if decodes.is_empty() && prefill.is_none() && spec.is_none() {
            // Work exists but nothing can step (clients' channels full, or
            // only queued jobs blocked on headroom held by blocked
            // sequences): wait briefly for channel drain / new messages.
            if ctrl.is_none() {
                match rx.recv_timeout(BLOCKED_POLL) {
                    Ok(msg) => match pump_msg(
                        msg,
                        session,
                        info,
                        loaded_reference,
                        distributed,
                        &batch_error,
                        &mut queue,
                    ) {
                        Pumped::Handled => {}
                        Pumped::Ctrl(msg) => ctrl = Some(msg),
                        Pumped::Exit => return None,
                    },
                    Err(std_mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std_mpsc::RecvTimeoutError::Disconnected) => return None,
                }
            } else {
                std::thread::sleep(BLOCKED_POLL);
            }
            continue 'serve;
        }

        // 8. ONE decode for the whole step (a speculative round's verify
        // decode included — its failure handling is exactly this path).
        if let Err(e) = session.decode_batch(batch) {
            // §4 + §5 invalidation: the KV state is torn (distributed) or
            // suspect (solo) — retained prefixes and the draft sync are
            // discarded either way. On a solo session the retained slots'
            // KV is explicitly freed; a distributed session is about to be
            // torn down by the supervisor.
            if !distributed {
                for slot in retained.iter() {
                    let _ = session.seq_rm(slot.seq, -1, -1);
                }
            }
            retained.clear();
            if let Some(d) = draft.as_deref_mut() {
                d.synced = None;
            }
            if distributed {
                // M5 failure lifecycle step 2 (docs/resilience.md): nothing
                // terminal on any job.tx — every active sequence shared the
                // failed batch and the torn remote state, so ALL of them
                // report Interrupted; the supervisor tears down once and
                // re-prefills each affected sequence.
                tracing::warn!(
                    error = %e,
                    sequences = active.len(),
                    "distributed decode failed; interrupting every active sequence"
                );
                for mut g in active.drain(..) {
                    let mut generated_tokens = std::mem::take(&mut g.prior_generated);
                    generated_tokens.extend(g.attempt_tokens.iter().copied());
                    if let Some(outcome) = g.outcome.take() {
                        let _ = outcome.send(GenOutcome::Interrupted(Box::new(InterruptedGen {
                            job: g.job,
                            prompt_tokens: g.prompt_tokens,
                            generated_tokens,
                            pieces_sent: g.pieces_sent,
                            scan: g.scan,
                            error: e.to_string(),
                        })));
                    }
                }
            } else {
                // Solo decode failure: terminal errors, exactly the pre-M5
                // posture — nothing to retry onto. A sequence with
                // confirmed pieces still parked on a full client channel
                // stays active as a draining zombie (fail_seq defers its
                // Error behind the backlog); it never decodes again and is
                // reaped by the held sweep or the disconnect sweep.
                tracing::warn!(error = %e, "solo decode failed; erroring the active sequences");
                for g in active.iter_mut() {
                    fail_seq(session, &mut free, &mut outbox, g, e.to_string());
                }
                active.retain(|g| !g.done);
            }
            // Every sequence id not owned by a draining zombie is free
            // again (a fresh list beats replaying partial bookkeeping; the
            // distributed arm drained `active` entirely). Queued jobs stay
            // queued: on a torn distributed model their prefill fails fast
            // and the supervisor's retry owns them; on a solo model they
            // fail with the same decode error individually (pre-M7
            // equivalent).
            free = (0..n_seq_max as SeqId)
                .rev()
                .filter(|s| !active.iter().any(|g| g.seq == *s))
                .collect();
            continue 'serve;
        }

        // 9. Post-step per decoding sequence: confirm the pending token
        // (its decode succeeded — confirm-before-send holds), emit its
        // piece, then sample the next token from this step's logits.
        let mut emitted = false;
        for (i, index) in decodes {
            let g = &mut active[i];
            g.kv_len += 1;
            let (tok, piece) = g
                .pending
                .take()
                .expect("a stepped sequence has a pending token");
            g.attempt_tokens.push(tok);
            g.remaining -= 1;
            if !g.scan.admit(&piece) {
                // The piece completes a stop-string match: hold it back and
                // finish with Stop (contract: the completing piece is never
                // sent; the token still counts as generated).
                finish_seq(
                    session,
                    &mut free,
                    &mut outbox,
                    g,
                    FinishKind::Stop,
                    perf.kv_reuse.then_some(&mut retained),
                );
                continue;
            }
            match g.job.tx.try_send(TokenEvent::Token(piece)) {
                Ok(()) => {
                    g.pieces_sent += 1;
                    emitted = true;
                    if g.ttft_ms.is_none() {
                        g.ttft_ms = Some(duration_ms(g.arrived.elapsed()));
                    }
                }
                Err(TrySendError::Full(TokenEvent::Token(piece))) => {
                    // Confirmed but undeliverable right now: park it and
                    // pause this sequence; the others keep stepping.
                    g.held.push_back(piece);
                }
                Err(_) => {
                    cancel_seq(session, &mut free, g);
                    continue;
                }
            }
            if g.remaining == 0 {
                if g.held.is_empty() {
                    finish_seq(
                        session,
                        &mut free,
                        &mut outbox,
                        g,
                        FinishKind::Length,
                        perf.kv_reuse.then_some(&mut retained),
                    );
                } else {
                    // The just-confirmed final piece is parked on the full
                    // channel: the terminal event must never overtake a
                    // held piece (same rule as the speculative path), and
                    // finishing now would drop it — the held-retry sweep
                    // applies this finish once the backlog drains.
                    g.terminal_after_drain = Some(DeferredTerminal::Finish(FinishKind::Length));
                }
                continue;
            }
            let next = session.sample_ith_with(&mut g.chain, index as i32);
            if session.model().is_eog(next) {
                if g.held.is_empty() {
                    finish_seq(
                        session,
                        &mut free,
                        &mut outbox,
                        g,
                        FinishKind::Stop,
                        perf.kv_reuse.then_some(&mut retained),
                    );
                } else {
                    // Same deferral as Length above: EOG was sampled while
                    // this step's piece is still parked.
                    g.terminal_after_drain = Some(DeferredTerminal::Finish(FinishKind::Stop));
                }
                continue;
            }
            match session.model().token_to_piece(next) {
                Ok(piece) => g.pending = Some((next, piece)),
                Err(e) => fail_seq(session, &mut free, &mut outbox, g, e.to_string()),
            }
        }

        // 10. Post-step for the prefill chunk; a completed prefill samples
        // its first token from the prefix tail's logits.
        if let Some((i, chunk, tail_index)) = prefill {
            let g = &mut active[i];
            if !g.done {
                g.kv_len += chunk;
                g.prefill_done += chunk;
                if g.prefill_done == g.prefix.len() {
                    g.prefill_finished = Some(Instant::now());
                    let index = tail_index.expect("the prefix tail carries logits");
                    let first = session.sample_ith_with(&mut g.chain, index as i32);
                    if session.model().is_eog(first) {
                        // EOG straight after the prompt: Stop with zero
                        // generated tokens (matches the solo path).
                        finish_seq(
                            session,
                            &mut free,
                            &mut outbox,
                            g,
                            FinishKind::Stop,
                            perf.kv_reuse.then_some(&mut retained),
                        );
                    } else {
                        match session.model().token_to_piece(first) {
                            Ok(piece) => g.pending = Some((first, piece)),
                            Err(e) => fail_seq(session, &mut free, &mut outbox, g, e.to_string()),
                        }
                    }
                }
            }
        }

        // 11. Post-step for a speculative verify round (docs/perf.md §5):
        // the pending token and the longest draft prefix the target's own
        // greedy choices reproduce are all CONFIRMED by the one decode
        // above (confirm-before-send, one batch earlier); rejected draft
        // positions are rolled back with a real seq_rm.
        if let Some(step) = spec {
            let g = &mut active[0];
            if let Some(e) = process_spec_step(
                session,
                &mut free,
                &mut outbox,
                g,
                &step,
                draft.as_deref_mut(),
                perf.kv_reuse,
                &mut retained,
                &mut emitted,
            ) {
                // Rollback failed (partial seq_rm unsupported): the
                // sequence's KV is unusable — fail it typed and stand the
                // speculative path down for the phase.
                tracing::warn!(error = %e, "speculative rollback failed; disabling speculation");
                spec_disabled = true;
                if !g.done {
                    fail_seq(session, &mut free, &mut outbox, g, e);
                }
            }
        }
        active.retain(|g| !g.done);

        if emitted {
            if let Some(delay) = decode_delay {
                // Test-only `[debug] decode_delay_ms`: a deterministic kill
                // window for the chaos sim (per emitting step ≈ per piece in
                // the single-sequence case the sim drives).
                std::thread::sleep(delay);
            }
        }
    }
}

/// Handle one host message inside the serve loop. Generation requests are
/// validated (and possibly terminated) here; model-replacing messages
/// become the barrier.
fn pump_msg(
    msg: HostMsg,
    session: &Session<'_>,
    info: &LoadedModel,
    loaded_reference: &str,
    distributed: bool,
    batch_error: &Option<String>,
    queue: &mut VecDeque<PreparedGen>,
) -> Pumped {
    match msg {
        HostMsg::Models { resp } => {
            let _ = resp.try_send(Some(info.clone()));
            Pumped::Handled
        }
        HostMsg::Generate(sup) => {
            if let Some(prepared) =
                prepare_generation(session, info, loaded_reference, batch_error, sup)
            {
                queue.push_back(prepared);
            }
            Pumped::Handled
        }
        HostMsg::Embed(job) => {
            handle_embed(session.model(), info, loaded_reference, distributed, job);
            Pumped::Handled
        }
        HostMsg::Shutdown => Pumped::Exit,
        other @ (HostMsg::Load { .. }
        | HostMsg::LoadDistributed { .. }
        | HostMsg::Unload { .. }
        | HostMsg::ServeShard { .. }) => Pumped::Ctrl(other),
    }
}

/// Serve one embeddings request against the loaded model (M1
/// `/v1/embeddings` / `/api/embed`).
///
/// # Mechanism (documented honestly)
///
/// A SHORT-LIVED dedicated embeddings session is created against the
/// already-loaded model, used serially for every text in the request, and
/// dropped before returning. The generation session is a separate llama
/// context whose KV, sampler chain, and retained §4 prefixes are never
/// touched. Because this runs inside the host thread's message pump, it is
/// naturally serialized BETWEEN decode steps: an in-flight generation
/// pauses for the embed's duration (the documented cost of the host loop's
/// single-thread serialization) but is never disturbed, reordered, or
/// corrupted. Creating the session per request trades a context
/// setup/teardown per call for zero idle memory and zero interaction with
/// the generation context's admission math — the right trade until
/// embeddings traffic proves otherwise.
///
/// # Distributed models
///
/// Refused with the typed [`ApiError::EmbeddingsDistributed`] (remedy
/// names a solo load). A second context against an RPC-split model would
/// issue its own remote command stream interleaved with the generation
/// context's pipelined one, which the overlap patches (patches/0002)
/// assume is single-context; nothing proves the combination sound, so
/// honesty beats silent wrongness (docs/resilience.md posture).
fn handle_embed(
    model: &Model,
    info: &LoadedModel,
    loaded_reference: &str,
    distributed: bool,
    job: EmbedJob,
) {
    let EmbedJob {
        model: requested,
        texts,
        resp,
    } = job;
    if distributed {
        let _ = resp.send(Err(ApiError::EmbeddingsDistributed(requested)));
        return;
    }
    // Same name rule as prepare_generation: the canonical loaded name and
    // the reference the load was requested with are both accepted.
    if requested != info.name && requested != loaded_reference {
        let _ = resp.send(Err(ApiError::ModelNotLoaded(requested)));
        return;
    }
    let started = Instant::now();
    // Tokenize everything first: input validation must fail typed before
    // the embeddings context's memory is allocated.
    let mut token_lists = Vec::with_capacity(texts.len());
    let mut prompt_tokens: u32 = 0;
    for (i, text) in texts.iter().enumerate() {
        let tokens = match model.tokenize(text, true) {
            Ok(tokens) if tokens.is_empty() => {
                let _ = resp.send(Err(ApiError::BadRequest(format!(
                    "`input[{i}]` produced no tokens; provide non-empty text"
                ))));
                return;
            }
            Ok(tokens) => tokens,
            Err(e) => {
                let _ = resp.send(Err(ApiError::Internal(e.to_string())));
                return;
            }
        };
        if tokens.len() > info.n_ctx as usize {
            let _ = resp.send(Err(ApiError::BadRequest(format!(
                "`input[{i}]` is {} tokens but the loaded context length is {}; shorten \
                 the input or raise ctx_len in config.toml",
                tokens.len(),
                info.n_ctx
            ))));
            return;
        }
        prompt_tokens += tokens.len() as u32;
        token_lists.push(tokens);
    }
    // n_batch = n_ubatch = n_ctx: an engine-pooled model reduces over ONE
    // decode call (Session::embed refuses inputs past n_batch), and
    // non-causal embedding models additionally require
    // n_ubatch >= n_tokens. The mean-pool fallback chunks itself, so the
    // wide batch costs generative models nothing.
    let mut session = match Session::new(
        model,
        &SessionParams {
            n_ctx: info.n_ctx,
            n_batch: info.n_ctx,
            n_ubatch: info.n_ctx,
            embeddings: true,
            // The model's own declared pooling: purpose-built embedding
            // models use their trained head; generative models resolve to
            // none and Session::embed mean-pools in Rust (documented
            // there).
            pooling: PoolingType::Unspecified,
            ..SessionParams::default()
        },
    ) {
        Ok(s) => s,
        Err(e) => {
            let _ = resp.send(Err(ApiError::Internal(e.to_string())));
            return;
        }
    };
    let mut embeddings = Vec::with_capacity(token_lists.len());
    for tokens in &token_lists {
        match session.embed(tokens) {
            Ok(mut vector) => {
                l2_normalize(&mut vector);
                embeddings.push(vector);
            }
            Err(e) => {
                let _ = resp.send(Err(ApiError::Internal(e.to_string())));
                return;
            }
        }
    }
    tracing::info!(
        inputs = texts.len(),
        tokens = prompt_tokens,
        elapsed_ms = duration_ms(started.elapsed()),
        "embeddings request served"
    );
    let _ = resp.send(Ok(EmbedResult {
        embeddings,
        prompt_tokens,
    }));
}

/// Scale a vector to unit L2 norm in place (OpenAI parity: their
/// embeddings — and llama.cpp's own OpenAI-compatible server — return
/// normalized vectors, so cosine similarity is a plain dot product for
/// unmodified clients). A zero or non-finite norm leaves the vector
/// untouched rather than manufacturing NaNs.
fn l2_normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if norm.is_finite() && norm > 0.0 {
        for x in v.iter_mut() {
            *x = (f64::from(*x) / norm) as f32;
        }
    }
}

/// Validate one supervised generation and turn it into a queue entry.
/// `None` means the job was already terminated here (validation error,
/// spent resume budget, or a client that disconnected before starting).
fn prepare_generation(
    session: &Session<'_>,
    info: &LoadedModel,
    loaded_reference: &str,
    batch_error: &Option<String>,
    sup: SupervisedGenerate,
) -> Option<PreparedGen> {
    let SupervisedGenerate {
        job,
        resume,
        outcome,
    } = sup;
    // Terminate the stream with an error; the attempt is Finished. The
    // channel is fresh (nothing streamed yet), so try_send cannot be full.
    let finish_error =
        |job: &GenerateJob, outcome: oneshot::Sender<GenOutcome>, message: String| {
            let _ = job.tx.try_send(TokenEvent::Error(message));
            let _ = outcome.send(GenOutcome::Finished);
        };

    if let Some(message) = batch_error {
        finish_error(&job, outcome, message.clone());
        return None;
    }

    // Cancellation before any work (docs/perf.md §6): a client that is
    // already gone never occupies queue space or a prefill.
    if job.tx.is_closed() {
        let _ = outcome.send(GenOutcome::Finished);
        return None;
    }

    // Accept the canonical loaded name and the reference the load was
    // requested with (`hf:…` refs cache under a sanitized key; clients may
    // use either spelling).
    if job.model != info.name && job.model != loaded_reference {
        let message = ApiError::ModelNotLoaded(job.model.clone()).to_string();
        finish_error(&job, outcome, message);
        return None;
    }

    let max_tokens = job.params.max_tokens as usize;
    let resumed = resume.is_some();
    let (prompt_tokens, prior_generated, prior_pieces, scan, unsent) = match resume {
        Some(state) => {
            // Confirm-before-send across attempts: every carried token is
            // CONFIRMED, but the interrupted attempt may have died with
            // trailing pieces still parked on a full client channel
            // (`ActiveGen::held` — pieces_sent excludes them). Re-render
            // them now so the resumed attempt delivers them before
            // anything else; their pieces already passed the stop scan, so
            // they are delivered without re-admitting. A render failure is
            // terminal-typed like every other prepare failure.
            let mut unsent = Vec::with_capacity(
                state
                    .generated_tokens
                    .len()
                    .saturating_sub(state.pieces_sent),
            );
            for &tok in state.generated_tokens.iter().skip(state.pieces_sent) {
                match session.model().token_to_piece(tok) {
                    Ok(piece) => unsent.push(piece),
                    Err(e) => {
                        finish_error(&job, outcome, e.to_string());
                        return None;
                    }
                }
            }
            (
                state.prompt_tokens,
                state.generated_tokens,
                state.pieces_sent,
                state.scan,
                unsent,
            )
        }
        None => {
            let prompt_text = match render_prompt(session.model(), info, &job.prompt) {
                Ok(text) => text,
                Err(message) => {
                    finish_error(&job, outcome, message);
                    return None;
                }
            };
            let prompt_tokens = match session.model().tokenize(&prompt_text, true) {
                Ok(toks) => toks,
                Err(e) => {
                    finish_error(&job, outcome, e.to_string());
                    return None;
                }
            };
            if prompt_tokens.len() + max_tokens > info.n_ctx as usize {
                let message = ApiError::BadRequest(format!(
                    "the prompt is {} tokens and max_tokens is {}, which together exceed \
                     the context length of {}; shorten the prompt, lower max_tokens, or \
                     raise ctx_len in config.toml",
                    prompt_tokens.len(),
                    max_tokens,
                    info.n_ctx
                ))
                .to_string();
                finish_error(&job, outcome, message);
                return None;
            }
            (
                prompt_tokens,
                Vec::new(),
                0,
                StopScan::new(job.params.stop.clone()),
                Vec::new(),
            )
        }
    };
    // Budget left for this attempt. The fresh path checked prompt +
    // max_tokens against n_ctx, and prefix + remaining equals exactly that
    // sum, so no re-check is needed on resume.
    let remaining = max_tokens.saturating_sub(prior_generated.len());
    if remaining == 0 && unsent.is_empty() {
        // Interrupted on the very last token's decode with every piece
        // already delivered: finish as Length without touching the engine.
        // With undelivered pieces (`unsent`), the job is queued instead:
        // admission seeds them into `held` with a deferred Length finish,
        // so the sweep delivers them before the terminal event.
        let _ = job.tx.try_send(TokenEvent::Done(DoneStats {
            prompt_tokens: prompt_tokens.len() as u32,
            completion_tokens: prior_generated.len() as u32,
            finish: FinishKind::Length,
            ..DoneStats::default()
        }));
        let _ = outcome.send(GenOutcome::Finished);
        return None;
    }
    let budget = prompt_tokens.len() + max_tokens;
    Some(PreparedGen {
        job,
        outcome: Some(outcome),
        prompt_tokens,
        prior_generated,
        prior_pieces,
        scan,
        remaining,
        budget,
        resumed,
        unsent,
        arrived: Instant::now(),
    })
}

/// Turn a queue entry into an active sequence on `seq`. `reused` is the
/// prefix length already present in the slot's KV from a §4 reuse hit (0 =
/// cold slot); the prefill loop starts decoding there. `sampler`/`chain`
/// are this generation's own sampler params and chain, built by the
/// admission loop from the job's params.
fn admit(
    p: PreparedGen,
    sampler: SamplerParams,
    chain: Sampler,
    seq: SeqId,
    reused: usize,
    uid: u64,
) -> ActiveGen {
    let prefix: Vec<Token> = p
        .prompt_tokens
        .iter()
        .chain(p.prior_generated.iter())
        .copied()
        .collect();
    debug_assert!(
        reused < prefix.len().max(1),
        "reuse must leave at least the prefix tail to decode"
    );
    ActiveGen {
        seq,
        uid,
        job: p.job,
        outcome: p.outcome,
        prompt_tokens: p.prompt_tokens,
        prior_generated: p.prior_generated,
        scan: p.scan,
        prefix,
        prefill_done: reused,
        reused,
        kv_len: reused,
        attempt_tokens: Vec::new(),
        pending: None,
        // A resume's undelivered pieces go out through the held sweep (in
        // order, before any new piece); non-empty `held` also pauses the
        // sequence's stepping until the backlog drains.
        held: VecDeque::from(p.unsent),
        // A resume whose budget is already spent has nothing to decode:
        // deliver the backlog, then finish as Length (the sweep applies
        // this once `held` drains; the batch builder never steps it).
        terminal_after_drain: (p.remaining == 0)
            .then_some(DeferredTerminal::Finish(FinishKind::Length)),
        remaining: p.remaining,
        budget: p.budget,
        pieces_sent: p.prior_pieces,
        sampler,
        chain,
        drafted: 0,
        accepted: 0,
        arrived: p.arrived,
        admitted: Instant::now(),
        prefill_finished: None,
        ttft_ms: None,
        done: false,
    }
}

/// Length of the longest common prefix of two token streams (§4 reuse
/// matching).
fn common_prefix_len(a: &[Token], b: &[Token]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Build one speculative verify round (docs/perf.md §5) for the single
/// active generation `g`: sync the draft session's KV to `g`'s confirmed
/// token stream, draft up to [`SPEC_K`] greedy tokens on it, and stage the
/// pending token plus every draft — all with logits — in `batch` for ONE
/// target verify decode. Only the draft session and the batch are touched;
/// a draft-side engine error bubbles up and the caller stands speculation
/// down without harming the generation.
fn build_spec_step(
    draft: &mut DraftCtx<'_, '_>,
    batch: &mut Batch,
    g: &ActiveGen,
) -> Result<SpecStep, onebrain_engine::EngineError> {
    let (tok, _) = g.pending.as_ref().expect("eligibility checked pending");
    let confirmed = g.kv_len;
    debug_assert_eq!(
        g.prefix.len() + g.attempt_tokens.len(),
        confirmed,
        "a prefilled sequence's KV is exactly prefix + confirmed attempt tokens"
    );
    // Reuse the draft KV only when it already follows this generation;
    // otherwise (fresh engagement, slot handover) re-prefill from scratch —
    // the draft is small, correctness first.
    let synced = match draft.synced {
        Some((uid, n)) if uid == g.uid && n <= confirmed => n,
        _ => {
            draft.session.reset();
            0
        }
    };
    // Marked out-of-sync until the round completes: a decode error below
    // leaves the draft KV mid-write.
    draft.synced = None;
    // Catch-up (stream tokens the draft has not decoded yet) + the pending
    // token, in one chunked decode.
    let stream_tok = |i: usize| -> Token {
        if i < g.prefix.len() {
            g.prefix[i]
        } else {
            g.attempt_tokens[i - g.prefix.len()]
        }
    };
    let mut catchup: Vec<Token> = (synced..confirmed).map(stream_tok).collect();
    catchup.push(*tok);
    draft.session.decode(&catchup)?;
    let mut draft_kv = confirmed + 1;
    // Draft greedily. K is capped by the decode budget beyond the pending
    // token (a draft past the budget could never be emitted) and by the
    // batch capacity (tests shrink n_batch). The draft's own EOG ends the
    // proposal early — EOG is sampled, never decoded, on the plain path
    // too. The last draft token is not decoded into the draft KV (nothing
    // would be sampled after it).
    let k_max = SPEC_K
        .min(g.remaining.saturating_sub(1))
        .min(batch.capacity().saturating_sub(1));
    let mut drafts = Vec::with_capacity(k_max);
    while drafts.len() < k_max {
        let t = draft.session.sample_greedy();
        if draft.session.model().is_eog(t) {
            break;
        }
        drafts.push(t);
        if drafts.len() == k_max {
            break;
        }
        draft.session.decode(&[t])?;
        draft_kv += 1;
    }
    // Provisional sync mark: the draft KV now holds the confirmed stream +
    // the pending token + all but the last draft — a prefix of whatever
    // the verify confirms. `process_spec_step` reconciles it against the
    // outcome; the verify-failure path clears it instead.
    draft.synced = Some((g.uid, draft_kv));
    batch.clear();
    let mut indexes = Vec::with_capacity(1 + drafts.len());
    indexes.push(batch.push(*tok, confirmed as i32, g.seq, true)?);
    for (i, d) in drafts.iter().enumerate() {
        indexes.push(batch.push(*d, (confirmed + 1 + i) as i32, g.seq, true)?);
    }
    Ok(SpecStep {
        indexes,
        drafts,
        draft_kv,
    })
}

/// Post-verify half of a speculative round (docs/perf.md §5): accept the
/// longest draft prefix the target's own greedy choices reproduce, emit
/// the confirmed pieces, roll rejected positions back with a real
/// `seq_rm`, and resync the draft KV to the accepted stream. Returns
/// `Some(error)` only when a rollback `seq_rm` itself fails — the KV is
/// then unusable and the caller fails the sequence + disables speculation.
#[allow(clippy::too_many_arguments)]
fn process_spec_step(
    session: &mut Session<'_>,
    free: &mut Vec<SeqId>,
    outbox: &mut Vec<PendingTerminal>,
    g: &mut ActiveGen,
    step: &SpecStep,
    draft: Option<&mut DraftCtx<'_, '_>>,
    kv_reuse: bool,
    retained: &mut Vec<RetainedSlot>,
    emitted: &mut bool,
) -> Option<String> {
    let old_confirmed = g.kv_len;
    let (tok, piece) = g
        .pending
        .take()
        .expect("a speculative round has a pending token");
    // Acceptance walk over the verify decode's per-position logits: the
    // pending token is confirmed outright (its decode succeeded); each
    // draft token is confirmed when the target's greedy choice after the
    // previous confirmed token IS that draft token — its own decode
    // already happened in the same batch, so emitting it preserves
    // confirm-before-send exactly. Sampling goes through g's own chain
    // (speculation is greedy-only, so the chain is its greedy chain).
    let mut confirmed: Vec<(Token, String)> = vec![(tok, piece)];
    let mut next_pending: Option<Token> = None;
    let mut render_error: Option<String> = None;
    for (i, d) in step.drafts.iter().enumerate() {
        let target_next = session.sample_ith_with(&mut g.chain, step.indexes[i] as i32);
        if target_next != *d {
            next_pending = Some(target_next);
            break;
        }
        match session.model().token_to_piece(target_next) {
            Ok(p) => confirmed.push((target_next, p)),
            Err(e) => {
                render_error = Some(e.to_string());
                break;
            }
        }
    }
    let accepted = confirmed.len() - 1;
    g.drafted += step.drafts.len() as u32;
    g.accepted += accepted as u32;
    if let Some(message) = render_error {
        // Unreachable in practice (the model produced the token); mirror
        // the plain path's posture: the sequence fails typed. fail_seq
        // clears the whole sequence, so no partial trim is needed first.
        fail_seq(session, free, outbox, g, message);
        if let Some(d) = draft {
            d.session.reset();
            d.synced = None;
        }
        return None;
    }
    if next_pending.is_none() {
        // Every draft accepted: the next pending comes from the last
        // draft's logits (full-acceptance continuation).
        next_pending =
            Some(session.sample_ith_with(&mut g.chain, step.indexes[step.drafts.len()] as i32));
    }
    // Roll back rejected draft positions (position rule: a real seq_rm,
    // never a rewound counter). The verify decoded 1 + drafts.len()
    // tokens; we keep 1 + accepted.
    let keep = old_confirmed + confirmed.len();
    if confirmed.len() < 1 + step.drafts.len() {
        if let Err(e) = session.seq_rm(g.seq, keep as i32, -1) {
            if let Some(d) = draft {
                d.session.reset();
                d.synced = None;
            }
            return Some(format!(
                "rolling back rejected speculative tokens failed ({e}); \
                 the generation cannot continue safely"
            ));
        }
    }
    g.kv_len = keep;

    // Bookkeeping + emission in stream order. The stop-scan can end the
    // run mid-list: later confirmed tokens are then discarded — they never
    // count as generated — and their KV entries are trimmed off.
    let confirmed_total = confirmed.len();
    let mut finish: Option<FinishKind> = None;
    let mut kept = 0usize;
    for (t, p) in confirmed {
        g.attempt_tokens.push(t);
        g.remaining -= 1;
        kept += 1;
        if !g.scan.admit(&p) {
            // Stop match: the completing piece is held back entirely; the
            // token still counts as generated (contract).
            finish = Some(FinishKind::Stop);
            break;
        }
        deliver_piece(g, p, emitted);
        if g.remaining == 0 {
            debug_assert_eq!(
                kept, confirmed_total,
                "K is budget-capped, so the budget runs out only on the last token"
            );
            finish = Some(FinishKind::Length);
            break;
        }
    }
    if kept < confirmed_total {
        let cut = old_confirmed + kept;
        if let Err(e) = session.seq_rm(g.seq, cut as i32, -1) {
            if let Some(d) = draft {
                d.session.reset();
                d.synced = None;
            }
            return Some(format!(
                "trimming speculative tokens past a stop match failed ({e}); \
                 the generation cannot continue safely"
            ));
        }
        g.kv_len = cut;
    }

    // Draft resync: the draft KV (old stream + pending + drafts minus its
    // last) is a prefix of the newly confirmed stream up to `matched`;
    // trim anything beyond it so the next round appends cleanly.
    if let Some(d) = draft {
        let matched = step.draft_kv.min(g.kv_len);
        if matched < step.draft_kv {
            if let Err(e) = d.session.seq_rm(0, matched as i32, -1) {
                // Draft-side only: a full reset re-prefills next round.
                tracing::debug!(error = %e, "draft KV trim failed; resetting the draft session");
                d.session.reset();
                d.synced = None;
            } else {
                d.synced = Some((g.uid, matched));
            }
        } else {
            d.synced = Some((g.uid, matched));
        }
    }

    if finish.is_none() {
        let next = next_pending.expect("a mismatch or full-acceptance sample exists");
        if session.model().is_eog(next) {
            finish = Some(FinishKind::Stop);
        } else {
            match session.model().token_to_piece(next) {
                Ok(p) => g.pending = Some((next, p)),
                Err(e) => {
                    fail_seq(session, free, outbox, g, e.to_string());
                    return None;
                }
            }
        }
    }
    if let Some(kind) = finish {
        if g.held.is_empty() {
            finish_seq(session, free, outbox, g, kind, kv_reuse.then_some(retained));
        } else {
            // Held pieces must reach the client before the terminal event;
            // the held-retry sweep applies this finish once they drain.
            g.terminal_after_drain = Some(DeferredTerminal::Finish(kind));
        }
    }
    None
}

/// Deliver a confirmed piece in order: behind any held backlog, otherwise
/// straight to the channel, parking it on a full channel. A closed channel
/// drops the piece — the disconnect sweep reaps the sequence.
fn deliver_piece(g: &mut ActiveGen, piece: String, emitted: &mut bool) {
    if !g.held.is_empty() {
        g.held.push_back(piece);
        return;
    }
    match g.job.tx.try_send(TokenEvent::Token(piece)) {
        Ok(()) => {
            g.pieces_sent += 1;
            *emitted = true;
            if g.ttft_ms.is_none() {
                g.ttft_ms = Some(duration_ms(g.arrived.elapsed()));
            }
        }
        Err(TrySendError::Full(TokenEvent::Token(piece))) => g.held.push_back(piece),
        Err(_) => {}
    }
}

/// Free `g`'s KV state and sequence id (shared tail of every finish path).
fn release_seq(session: &mut Session<'_>, free: &mut Vec<SeqId>, g: &mut ActiveGen) {
    // Removing a whole sequence never fails (engine contract); a failure
    // here would leak KV headroom, so it is at least logged.
    if let Err(e) = session.seq_rm(g.seq, -1, -1) {
        tracing::warn!(seq = g.seq, error = %e, "failed to clear a finished sequence's KV");
    }
    free.push(g.seq);
    g.done = true;
}

/// Finish `g` normally: cumulative stats, the perf log line
/// (docs/perf.md §1/§4/§5), the terminal `Done`, the Finished outcome —
/// and, when `retain` is given (the §4 knob), the slot's KV + token
/// history survive for prefix reuse instead of being cleared.
fn finish_seq(
    session: &mut Session<'_>,
    free: &mut Vec<SeqId>,
    outbox: &mut Vec<PendingTerminal>,
    g: &mut ActiveGen,
    finish: FinishKind,
    retain: Option<&mut Vec<RetainedSlot>>,
) {
    let now = Instant::now();
    let prefill_ms = g
        .prefill_finished
        .map(|t| duration_ms(t.duration_since(g.admitted)))
        .unwrap_or_else(|| duration_ms(now.duration_since(g.admitted)));
    let decode_ms = g
        .prefill_finished
        .map(|t| duration_ms(now.duration_since(t)))
        .unwrap_or(0);
    let ttft_ms = g.ttft_ms.unwrap_or(0);
    // The stable per-generation instrumentation line (docs/perf.md §1) —
    // sim-greppable; counts are THIS attempt's engine work: the prefill
    // count is the tokens actually DECODED this attempt (on a §4 reuse hit
    // that is exactly the decoded suffix; a spent-budget resume finishes
    // without ever prefilling and honestly reports 0), and the §5 draft
    // counters ride at the end. Stats below span all attempts.
    tracing::info!(
        "perf: prefill {}tok {}ms decode {}tok {}ms ttft {}ms drafted {} accepted {}",
        g.prefill_done.saturating_sub(g.reused),
        prefill_ms,
        g.attempt_tokens.len(),
        decode_ms,
        ttft_ms,
        g.drafted,
        g.accepted
    );
    // Client-visible stats span ALL attempts: the original prompt length
    // and the cumulative completion count (M5 retry contract).
    let stats = DoneStats {
        prompt_tokens: g.prompt_tokens.len() as u32,
        completion_tokens: (g.prior_generated.len() + g.attempt_tokens.len()) as u32,
        finish,
        prefill_ms,
        decode_ms,
        ttft_ms,
        drafted: g.drafted,
        accepted: g.accepted,
    };
    deliver_terminal(outbox, &g.job.tx, TokenEvent::Done(stats));
    if let Some(outcome) = g.outcome.take() {
        let _ = outcome.send(GenOutcome::Finished);
    }
    match retain {
        Some(retained) => {
            // §4: the slot keeps its KV; record exactly what it holds. The
            // seq id returns to the free list (retained ⊆ free invariant).
            let mut history = std::mem::take(&mut g.prefix);
            history.extend(g.attempt_tokens.iter().copied());
            debug_assert_eq!(
                history.len(),
                g.kv_len,
                "retained history must mirror the slot's KV exactly"
            );
            retained.push(RetainedSlot {
                seq: g.seq,
                history,
            });
            free.push(g.seq);
            g.done = true;
        }
        None => release_seq(session, free, g),
    }
}

/// Terminate `g` with an error (solo decode failure, piece-render failure).
///
/// Error-after-drain: with confirmed pieces still parked on the client's
/// momentarily-full channel (`held`), the Error queues behind them exactly
/// like a finish would — the sequence stays active as a draining zombie
/// (never stepped again; the batch builder skips any set
/// `terminal_after_drain`) and the held-retry sweep applies the terminal
/// once the backlog empties. The wait's boundary is client liveness: a
/// client that DISCONNECTS is reaped by the disconnect sweep with the
/// backlog undelivered — Aborted semantics, nobody is listening — and a
/// terminal already decided (a finish waiting behind the same backlog)
/// wins over a later error: that generation was complete before the
/// failure.
fn fail_seq(
    session: &mut Session<'_>,
    free: &mut Vec<SeqId>,
    outbox: &mut Vec<PendingTerminal>,
    g: &mut ActiveGen,
    message: String,
) {
    if !g.held.is_empty() {
        if g.terminal_after_drain.is_none() {
            g.terminal_after_drain = Some(DeferredTerminal::Error(message));
        }
        return;
    }
    deliver_terminal(outbox, &g.job.tx, TokenEvent::Error(message));
    if let Some(outcome) = g.outcome.take() {
        let _ = outcome.send(GenOutcome::Finished);
    }
    release_seq(session, free, g);
}

/// Reap a disconnected client's sequence: no terminal event (nobody is
/// listening), the Finished outcome, KV freed.
fn cancel_seq(session: &mut Session<'_>, free: &mut Vec<SeqId>, g: &mut ActiveGen) {
    if let Some(outcome) = g.outcome.take() {
        let _ = outcome.send(GenOutcome::Finished);
    }
    release_seq(session, free, g);
}

/// Send a terminal event now, or park it for retry when the client's
/// channel is momentarily full — the host thread never blocks on a client.
fn deliver_terminal(
    outbox: &mut Vec<PendingTerminal>,
    tx: &mpsc::Sender<TokenEvent>,
    event: TokenEvent,
) {
    match tx.try_send(event) {
        Ok(()) | Err(TrySendError::Closed(_)) => {}
        Err(TrySendError::Full(event)) => outbox.push(PendingTerminal {
            tx: tx.clone(),
            event: Some(event),
        }),
    }
}

/// Turn the job's prompt into the text the engine tokenizes. Chat prompts
/// render through the model's template; models without one use the generic
/// fallback from the internal-api contract (and say so in the logs).
fn render_prompt(
    model: &Model,
    info: &LoadedModel,
    prompt: &PromptInput,
) -> Result<String, String> {
    match prompt {
        PromptInput::Raw(text) => Ok(text.clone()),
        PromptInput::Chat(messages) => {
            let turns: Vec<(String, String)> = messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect();
            match model.apply_chat_template(&turns, true) {
                Ok(Some(rendered)) => Ok(rendered),
                Ok(None) => {
                    tracing::warn!(
                        model = %info.name,
                        "model ships no chat template; using the generic fallback format"
                    );
                    let mut rendered = String::new();
                    for (role, content) in &turns {
                        rendered.push_str(&format!("<|{role}|>\n{content}\n"));
                    }
                    rendered.push_str("<|assistant|>\n");
                    Ok(rendered)
                }
                Err(e) => Err(e.to_string()),
            }
        }
    }
}

/// Stop-string holdback (internal-api contract): pieces are admitted until
/// one completes a stop-string match across the accumulated output; the
/// completing piece is held back entirely.
#[derive(Debug)]
pub struct StopScan {
    stops: Vec<String>,
    /// Accumulated tail, trimmed to the longest useful suffix.
    tail: String,
    /// Bytes of tail a future cross-piece match could still need.
    keep: usize,
}

impl StopScan {
    pub fn new(stops: Vec<String>) -> StopScan {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let keep = stops
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
            .saturating_sub(1);
        StopScan {
            stops,
            tail: String::new(),
            keep,
        }
    }

    /// `true`: `piece` is safe to emit. `false`: it completes a stop-string
    /// match and must be held back; generation should stop with `Stop`.
    pub fn admit(&mut self, piece: &str) -> bool {
        if self.stops.is_empty() {
            return true;
        }
        self.tail.push_str(piece);
        if self.stops.iter().any(|s| self.tail.contains(s)) {
            return false;
        }
        // Drop bytes no future match can straddle (keeps `tail` O(stop len)).
        if self.tail.len() > self.keep {
            let mut cut = self.tail.len() - self.keep;
            while !self.tail.is_char_boundary(cut) {
                cut += 1;
            }
            self.tail.drain(..cut);
        }
        true
    }
}

/// The daemon's [`EngineBackend`]: routes gateway requests into the
/// daemon's supervisor task (which drives the engine-host thread — M5) and
/// the model cache. Since M7 it also enforces admission control
/// (docs/perf.md §6): at most `max_concurrent + queue_depth` generation
/// jobs may be in the daemon at once; beyond that, requests are rejected
/// with the typed 429-equivalent instead of queueing unboundedly.
pub struct DaemonBackend {
    host: EngineHost,
    cache_root: PathBuf,
    /// Generation jobs go here; the supervisor owns their whole lifecycle
    /// (attempt, transparent retry, terminal event — docs/resilience.md).
    supervisor: SupervisorTx,
    /// Mesh handle for LAN-first pulls (docs/logistics.md: before any WAN
    /// byte, every Connected peer is asked what it holds).
    mesh: onebrain_mesh::MeshHandle,
    /// `config.cache_max_bytes` for the post-download GC trigger (0 = off).
    cache_max_bytes: u64,
    /// `[perf] max_concurrent_requests` (admission bound + error wording).
    max_concurrent: u32,
    /// `[perf] queue_depth` (admission bound + error wording).
    queue_depth: u32,
    /// M8 metrics request log (docs/product.md §1): every admitted
    /// generation's terminal `Done` is recorded here via the relay
    /// [`crate::metrics::RequestLog::observe`] wraps around the job —
    /// `generate` below is the single choke point both dialects funnel
    /// through, so one wrap covers the whole public API.
    requests: std::sync::Arc<crate::metrics::RequestLog>,
}

impl DaemonBackend {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host: EngineHost,
        cache_root: PathBuf,
        supervisor: SupervisorTx,
        mesh: onebrain_mesh::MeshHandle,
        cache_max_bytes: u64,
        max_concurrent: u32,
        queue_depth: u32,
        requests: std::sync::Arc<crate::metrics::RequestLog>,
    ) -> DaemonBackend {
        DaemonBackend {
            host,
            cache_root,
            supervisor,
            mesh,
            cache_max_bytes,
            max_concurrent: max_concurrent.max(1),
            queue_depth,
            requests,
        }
    }
}

impl EngineBackend for DaemonBackend {
    fn models(&self) -> Vec<ModelSummary> {
        let loaded = self.host.loaded_model(Duration::from_millis(250));
        let mut out: Vec<ModelSummary> = match cache::list(&self.cache_root) {
            Ok(cached) => cached
                .into_iter()
                .map(|m| {
                    let mut details = BTreeMap::new();
                    details.insert("path".to_string(), m.path.display().to_string());
                    if let Some(b3) = m.blake3 {
                        details.insert("blake3".to_string(), b3);
                    }
                    // M6 `onebrain ls` columns (docs/logistics.md): pin
                    // state, last-use stamp, and the split part count.
                    details.insert("pinned".to_string(), m.pinned.to_string());
                    details.insert("last_used_unix".to_string(), m.last_used_unix.to_string());
                    details.insert("parts".to_string(), m.parts.to_string());
                    ModelSummary {
                        loaded: loaded.as_ref().is_some_and(|l| l.name == m.id),
                        name: m.id,
                        size_bytes: m.size_bytes,
                        details,
                    }
                })
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "listing the model cache failed");
                Vec::new()
            }
        };
        // A loaded `local:` model is not in the cache; list it too.
        if let Some(l) = loaded {
            if !out.iter().any(|m| m.name == l.name) {
                out.push(ModelSummary {
                    name: l.name,
                    size_bytes: l.size_bytes,
                    loaded: true,
                    details: BTreeMap::new(),
                });
            }
        }
        out
    }

    fn generate(&self, job: GenerateJob) -> Result<(), ApiError> {
        // Admission control (docs/perf.md §6): count the job BEFORE it is
        // visible to the supervisor — the M5 idle probe never reads idle
        // while a job is en route — but only when the daemon has room for
        // it (`max_concurrent` running + `queue_depth` waiting). Beyond
        // that: the typed 429-equivalent, never an unbounded queue.
        let limit = self.max_concurrent as usize + self.queue_depth as usize;
        if !self.host.try_start_job(limit) {
            return Err(ApiError::Overloaded {
                max_concurrent: self.max_concurrent,
                queue_depth: self.queue_depth,
            });
        }
        // M8 metrics: relay the admitted job's event stream so the terminal
        // DoneStats lands in the request ring buffer (privacy enforced by
        // construction — the log can only record counts and timings).
        let job = self.requests.observe(job);
        if self.supervisor.send(SupervisorMsg::Generate(job)).is_err() {
            self.host.job_finished();
            return Err(ApiError::ShuttingDown);
        }
        Ok(())
    }

    fn embed(&self, job: EmbedJob) -> Result<(), ApiError> {
        // Straight to the host, bypassing the supervisor: embeddings have
        // no stream to supervise and no distributed retry story (the host
        // refuses distributed targets typed), and they take no §6
        // admission slot — the request serializes on the host thread's
        // message pump regardless, and its short-lived session never
        // shares the generation session's unified-KV pool the admission
        // math budgets.
        self.host.send(HostMsg::Embed(job))
    }

    fn pull(&self, model: String, tx: mpsc::Sender<PullEvent>) -> Result<(), ApiError> {
        let model_ref: ModelRef = model
            .parse()
            .map_err(|e| ApiError::BadRequest(format!("{e}")))?;
        let resolved = model_ref
            .resolve()
            .map_err(|e| ApiError::BadRequest(format!("{e}")))?;
        let cache_root = self.cache_root.clone();
        let mesh = self.mesh.clone();
        let host = self.host.clone();
        let cache_max_bytes = self.cache_max_bytes;
        tokio::spawn(async move {
            let terminal = match resolved {
                Resolved::Local(path) => {
                    if path.exists() {
                        PullEvent::Done
                    } else {
                        PullEvent::Error {
                            message: format!(
                                "local model {} does not exist; check the path",
                                path.display()
                            ),
                        }
                    }
                }
                Resolved::Remote(spec) => {
                    let mut throttle = ProgressThrottle::default();
                    let progress_tx = tx.clone();
                    // LAN-first (docs/logistics.md): peers are asked before
                    // any WAN byte; split sets fetch every part.
                    let result = crate::logistics::ensure_remote_local(
                        &mesh,
                        &cache_root,
                        &spec,
                        move |completed, total| {
                            if throttle.should_emit(completed, total) {
                                // try_send: never block the downloader on a
                                // slow client; skipped lines are harmless.
                                let _ = progress_tx
                                    .try_send(PullEvent::Downloading { completed, total });
                            }
                        },
                    )
                    .await;
                    match result {
                        Ok(_) => {
                            // GC trigger (docs/logistics.md): after every
                            // completed download — never the loaded model,
                            // never the entry just fetched.
                            let root = cache_root.clone();
                            let host = host.clone();
                            let fresh = spec.cache_key.clone();
                            let _ = tokio::task::spawn_blocking(move || {
                                let mut protected = std::collections::HashSet::new();
                                protected.insert(fresh);
                                if let Some(loaded) = host.loaded_model(Duration::from_millis(250))
                                {
                                    protected.insert(loaded.name);
                                }
                                crate::logistics::run_cache_gc(&root, cache_max_bytes, &protected);
                            })
                            .await;
                            PullEvent::Done
                        }
                        Err(message) => PullEvent::Error { message },
                    }
                }
            };
            let _ = tx.send(terminal).await;
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Load the OB_SMOKE_MODEL solo into a fresh host (by its local path)
    /// and hand back everything a supervised-generation test needs. `None`
    /// when the env var is unset (the test should skip).
    fn spawn_with_smoke_model(
        decode_delay: Option<Duration>,
        perf: HostPerf,
    ) -> Option<(EngineHost, std::thread::JoinHandle<()>, String)> {
        spawn_smoke(decode_delay, perf, /* with_draft */ false)
    }

    /// Like [`spawn_with_smoke_model`], optionally loading the smoke model
    /// AGAIN as its own speculative draft (docs/perf.md §5's
    /// draft==target case: greedy determinism makes acceptance total).
    fn spawn_smoke(
        decode_delay: Option<Duration>,
        perf: HostPerf,
        with_draft: bool,
    ) -> Option<(EngineHost, std::thread::JoinHandle<()>, String)> {
        let Ok(smoke) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping supervised generation test");
            return None;
        };
        let (host, handle) = EngineHost::spawn(decode_delay, perf);
        let (ptx, _prx) = mpsc::unbounded_channel();
        let (rtx, rrx) = oneshot::channel();
        host.send(HostMsg::Load {
            reference: smoke.clone(),
            cache_root: std::env::temp_dir(),
            ctx_len: 512,
            draft: with_draft.then(|| DraftRequest {
                reference: smoke.clone(),
                cache_root: std::env::temp_dir(),
            }),
            progress: ptx,
            resp: rtx,
        })
        .unwrap();
        rrx.blocking_recv()
            .expect("host answers")
            .expect("smoke model loads");
        Some((host, handle, smoke))
    }

    fn job_with_prompt(
        model: String,
        prompt: &str,
        max_tokens: u32,
        tx: mpsc::Sender<TokenEvent>,
    ) -> GenerateJob {
        GenerateJob {
            model,
            prompt: PromptInput::Raw(prompt.into()),
            params: onebrain_api::backend::GenParams {
                max_tokens,
                temperature: 0.0,
                ..Default::default()
            },
            dialect: onebrain_api::backend::ApiDialect::Openai,
            tx,
        }
    }

    fn greedy_job(model: String, max_tokens: u32, tx: mpsc::Sender<TokenEvent>) -> GenerateJob {
        job_with_prompt(model, "Once upon a time", max_tokens, tx)
    }

    /// Drive one greedy job to completion, returning (text, DoneStats).
    fn run_to_done(
        host: &EngineHost,
        model: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> (String, DoneStats) {
        let (tx, mut rx) = mpsc::channel(64);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: job_with_prompt(model.to_string(), prompt, max_tokens, tx),
            resume: None,
            outcome: otx,
        }))
        .unwrap();
        let mut text = String::new();
        let done = loop {
            match rx.blocking_recv().expect("stream must terminate") {
                TokenEvent::Token(piece) => text.push_str(&piece),
                TokenEvent::Done(stats) => break stats,
                TokenEvent::Error(e) => panic!("unexpected error: {e}"),
            }
        };
        assert!(matches!(
            orx.blocking_recv().expect("outcome must arrive"),
            GenOutcome::Finished
        ));
        (text, done)
    }

    #[test]
    fn supervised_generation_streams_and_reports_finished() {
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default()) else {
            return;
        };
        let (tx, mut rx) = mpsc::channel(64);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke, 4, tx),
            resume: None,
            outcome: otx,
        }))
        .unwrap();
        let mut tokens = 0;
        let done = loop {
            match rx.blocking_recv().expect("stream must terminate") {
                TokenEvent::Token(_) => tokens += 1,
                TokenEvent::Done(stats) => break stats,
                TokenEvent::Error(e) => panic!("unexpected error: {e}"),
            }
        };
        assert!(tokens > 0, "at least one piece must stream");
        assert_eq!(done.completion_tokens as usize, tokens);
        assert!(matches!(
            orx.blocking_recv().expect("outcome must arrive"),
            GenOutcome::Finished
        ));
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    /// The §6 micro-batch proof at the host level: with one client's stream
    /// deliberately stalled (capacity-1 channel that is not read), a second
    /// concurrent generation must still stream to completion — and both
    /// texts must be byte-identical to their alone-runs (the substrate's
    /// batched-vs-alone equality carried through the daemon's step loop).
    #[test]
    fn concurrent_generations_interleave_and_match_alone_runs() {
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default()) else {
            return;
        };
        const MAX: u32 = 8;
        let prompt_a = "Once upon a time";
        let prompt_b = "The little dog";
        let (alone_a, _) = run_to_done(&host, &smoke, prompt_a, MAX);
        let (alone_b, _) = run_to_done(&host, &smoke, prompt_b, MAX);
        assert!(!alone_a.is_empty() && !alone_b.is_empty());

        // A's channel holds ONE piece and is not read until B finishes: if
        // generations were single-flighted, B could never complete here.
        let (tx_a, mut rx_a) = mpsc::channel(1);
        let (otx_a, orx_a) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: job_with_prompt(smoke.clone(), prompt_a, MAX, tx_a),
            resume: None,
            outcome: otx_a,
        }))
        .unwrap();
        let (tx_b, mut rx_b) = mpsc::channel(64);
        let (otx_b, orx_b) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: job_with_prompt(smoke.clone(), prompt_b, MAX, tx_b),
            resume: None,
            outcome: otx_b,
        }))
        .unwrap();

        // Drain B fully while A is stalled.
        let mut text_b = String::new();
        loop {
            match rx_b
                .blocking_recv()
                .expect("B must terminate while A is stalled")
            {
                TokenEvent::Token(piece) => text_b.push_str(&piece),
                TokenEvent::Done(_) => break,
                TokenEvent::Error(e) => panic!("B errored: {e}"),
            }
        }
        assert!(matches!(
            orx_b.blocking_recv().expect("B outcome"),
            GenOutcome::Finished
        ));
        // Now drain A; it resumes as its channel gains room.
        let mut text_a = String::new();
        loop {
            match rx_a.blocking_recv().expect("A must terminate") {
                TokenEvent::Token(piece) => text_a.push_str(&piece),
                TokenEvent::Done(_) => break,
                TokenEvent::Error(e) => panic!("A errored: {e}"),
            }
        }
        assert!(matches!(
            orx_a.blocking_recv().expect("A outcome"),
            GenOutcome::Finished
        ));

        assert_eq!(
            text_a, alone_a,
            "concurrent A must be byte-identical to its alone-run"
        );
        assert_eq!(
            text_b, alone_b,
            "concurrent B must be byte-identical to its alone-run"
        );
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    /// docs/perf.md §6 cancellation: a disconnected client's sequence is
    /// freed at a step boundary, and the slot serves the next job — with
    /// max_concurrent = 1 a stuck slot would wedge the follow-up forever.
    #[test]
    fn disconnected_client_frees_the_slot() {
        let perf = HostPerf {
            max_concurrent: 1,
            // Tiny batches force multi-chunk prefills even on short
            // prompts, covering the mid-prefill disconnect path.
            n_batch: 4,
            n_ubatch: 4,
            ..HostPerf::default()
        };
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, perf) else {
            return;
        };
        // Client 1 disconnects before its prefill even starts.
        let (tx1, rx1) = mpsc::channel(1);
        drop(rx1);
        let (otx1, orx1) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke.clone(), 64, tx1),
            resume: None,
            outcome: otx1,
        }))
        .unwrap();
        assert!(matches!(
            orx1.blocking_recv()
                .expect("cancelled job reports an outcome"),
            GenOutcome::Finished
        ));
        // Client 2 disconnects mid-generation (after the first piece).
        let (tx2, mut rx2) = mpsc::channel(1);
        let (otx2, orx2) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke.clone(), 64, tx2),
            resume: None,
            outcome: otx2,
        }))
        .unwrap();
        match rx2.blocking_recv().expect("first piece streams") {
            TokenEvent::Token(_) => {}
            other => panic!("expected a token, got {other:?}"),
        }
        drop(rx2);
        assert!(matches!(
            orx2.blocking_recv()
                .expect("disconnected job reports an outcome"),
            GenOutcome::Finished
        ));
        // The single slot must be free again: a healthy job completes.
        let (text, _) = run_to_done(&host, &smoke, "Once upon a time", 4);
        assert!(!text.is_empty(), "the freed slot must serve the next job");
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    /// docs/perf.md §6 status honesty: the loaded-model summary answers
    /// from cached state — instantly — while a generation is running.
    #[test]
    fn loaded_model_answers_instantly_during_generation() {
        let Some((host, handle, smoke)) =
            spawn_with_smoke_model(Some(Duration::from_millis(30)), HostPerf::default())
        else {
            return;
        };
        let (tx, mut rx) = mpsc::channel(64);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke.clone(), 8, tx),
            resume: None,
            outcome: otx,
        }))
        .unwrap();
        // Wait for the first piece so the generation is provably running.
        match rx.blocking_recv().expect("first piece streams") {
            TokenEvent::Token(_) => {}
            other => panic!("expected a token, got {other:?}"),
        }
        let started = Instant::now();
        let model = host.loaded_model(Duration::from_millis(1));
        assert!(
            model.is_some(),
            "status must see the loaded model while a generation runs"
        );
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "the answer must come from the cache, not the busy host thread"
        );
        // Drain to completion.
        loop {
            match rx.blocking_recv().expect("stream must terminate") {
                TokenEvent::Done(_) => break,
                TokenEvent::Token(_) => {}
                TokenEvent::Error(e) => panic!("unexpected error: {e}"),
            }
        }
        assert!(matches!(
            orx.blocking_recv().expect("outcome"),
            GenOutcome::Finished
        ));
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn supervised_resume_with_spent_budget_finishes_as_length() {
        // The retry edge case: interrupted on the very last token's decode —
        // every piece already reached the client, so the resumed attempt
        // must terminate as Length with the CUMULATIVE stats and without
        // re-sending anything.
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default()) else {
            return;
        };
        let (tx, mut rx) = mpsc::channel(8);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke, 2, tx),
            resume: Some(ResumeState {
                prompt_tokens: vec![1, 2, 3],
                generated_tokens: vec![4, 5],
                pieces_sent: 2,
                scan: StopScan::new(vec![]),
            }),
            outcome: otx,
        }))
        .unwrap();
        match rx.blocking_recv().expect("stream must terminate") {
            TokenEvent::Done(stats) => {
                assert_eq!(stats.prompt_tokens, 3, "original prompt length");
                assert_eq!(stats.completion_tokens, 2, "cumulative completion");
                assert_eq!(stats.finish, FinishKind::Length);
            }
            other => panic!("expected an immediate Done, got {other:?}"),
        }
        assert!(matches!(
            orx.blocking_recv().expect("outcome must arrive"),
            GenOutcome::Finished
        ));
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn full_channel_at_final_token_never_drops_the_held_piece() {
        // Confirm-before-send's dual: a CONFIRMED piece parked on a full
        // channel must still reach the client, and the terminal event must
        // never overtake it. Capacity 1 with a deliberately idle reader
        // parks the second (final) piece; the Length finish must defer to
        // the held-retry sweep instead of dropping it.
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default()) else {
            return;
        };
        let (tx, mut rx) = mpsc::channel(1);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke, 2, tx),
            resume: None,
            outcome: otx,
        }))
        .unwrap();
        // Let the host confirm both tokens against the full channel before
        // draining (the smoke model needs only milliseconds).
        std::thread::sleep(Duration::from_millis(500));
        let mut pieces = 0u32;
        let done = loop {
            match rx.blocking_recv().expect("stream must terminate") {
                TokenEvent::Token(_) => pieces += 1,
                TokenEvent::Done(stats) => break stats,
                TokenEvent::Error(e) => panic!("unexpected error: {e}"),
            }
        };
        assert_eq!(done.finish, FinishKind::Length);
        assert_eq!(
            pieces, done.completion_tokens,
            "every confirmed piece must be delivered before Done"
        );
        assert!(matches!(
            orx.blocking_recv().expect("outcome must arrive"),
            GenOutcome::Finished
        ));
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn full_channel_at_error_never_drops_the_held_piece() {
        // The error-path dual of the test above (the M8 gate fixed
        // Length/EOG finishes; this pins fail_seq): a solo decode failure
        // while a CONFIRMED piece sits parked on a full client channel
        // must deliver that piece first and the Error terminal after it —
        // never drop it, never let the Error overtake it.
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default()) else {
            return;
        };
        // A: healthy greedy generation into a capacity-1 channel that is
        // not read yet. After the first piece fills the channel, the next
        // confirmed piece parks in `held` and A pauses with exactly
        // pieces_sent = 1 and one held piece.
        let (tx_a, mut rx_a) = mpsc::channel(1);
        let (otx_a, orx_a) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke.clone(), 8, tx_a),
            resume: None,
            outcome: otx_a,
        }))
        .unwrap();
        // Let A reach the parked state (the smoke model needs only
        // milliseconds; once parked it stays parked until we read).
        std::thread::sleep(Duration::from_millis(500));
        // B: a resume whose prompt carries an out-of-vocabulary token id.
        // Its prefill decode fails (llama validates batch token ids before
        // touching any KV state), which is a SOLO decode failure erroring
        // every active sequence — A included, while A's piece is parked.
        let (tx_b, mut rx_b) = mpsc::channel(8);
        let (otx_b, orx_b) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke.clone(), 4, tx_b),
            resume: Some(ResumeState {
                prompt_tokens: vec![1, 2, 999_999],
                generated_tokens: vec![],
                pieces_sent: 0,
                scan: StopScan::new(vec![]),
            }),
            outcome: otx_b,
        }))
        .unwrap();
        // B had nothing parked: its Error is immediate.
        match rx_b.blocking_recv().expect("B must terminate") {
            TokenEvent::Error(message) => {
                assert!(message.contains("decode failed"), "got: {message}");
            }
            other => panic!("expected Error for B, got {other:?}"),
        }
        assert!(matches!(
            orx_b.blocking_recv().expect("B outcome"),
            GenOutcome::Finished
        ));
        // A: every confirmed piece — the parked one included — must arrive
        // BEFORE the Error terminal.
        let mut pieces = 0u32;
        loop {
            match rx_a.blocking_recv().expect("A must terminate") {
                TokenEvent::Token(_) => pieces += 1,
                TokenEvent::Error(message) => {
                    assert!(message.contains("decode failed"), "got: {message}");
                    break;
                }
                TokenEvent::Done(stats) => panic!("A must error, got Done: {stats:?}"),
            }
        }
        assert!(
            pieces >= 2,
            "the piece parked on the full channel must be delivered before \
             the Error terminal (got {pieces} pieces)"
        );
        assert!(matches!(
            orx_a.blocking_recv().expect("A outcome"),
            GenOutcome::Finished
        ));
        // The failed-over slots serve again: a healthy job completes.
        let (text, _) = run_to_done(&host, &smoke, "Once upon a time", 4);
        assert!(!text.is_empty(), "slots must serve again after the failure");
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    /// A fixed-seed sampled job (temperature > 0).
    fn sampled_job(
        model: String,
        prompt: &str,
        max_tokens: u32,
        seed: u32,
        tx: mpsc::Sender<TokenEvent>,
    ) -> GenerateJob {
        GenerateJob {
            model,
            prompt: PromptInput::Raw(prompt.into()),
            params: onebrain_api::backend::GenParams {
                max_tokens,
                temperature: 0.8,
                top_p: 0.95,
                top_k: 40,
                seed: Some(seed),
                ..Default::default()
            },
            dialect: onebrain_api::backend::ApiDialect::Openai,
            tx,
        }
    }

    /// Drive one sampled job to completion, returning its text.
    fn run_sampled_to_done(
        host: &EngineHost,
        model: &str,
        prompt: &str,
        max_tokens: u32,
        seed: u32,
    ) -> String {
        let (tx, mut rx) = mpsc::channel(64);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: sampled_job(model.to_string(), prompt, max_tokens, seed, tx),
            resume: None,
            outcome: otx,
        }))
        .unwrap();
        let mut text = String::new();
        loop {
            match rx.blocking_recv().expect("stream must terminate") {
                TokenEvent::Token(piece) => text.push_str(&piece),
                TokenEvent::Done(_) => break,
                TokenEvent::Error(e) => panic!("unexpected error: {e}"),
            }
        }
        assert!(matches!(
            orx.blocking_recv().expect("outcome must arrive"),
            GenOutcome::Finished
        ));
        text
    }

    /// Byte-compat pin (per-sequence sampler chains): a sampled request
    /// running ALONE must produce exactly what the session-chain path
    /// produces — the oracle is the engine's own `set_sampler` + `generate`
    /// loop with identical params, i.e. the same chain construction and
    /// seed semantics the host used before standalone chains.
    #[test]
    fn sampled_alone_run_matches_the_session_chain_construction() {
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default()) else {
            return;
        };
        const MAX: u32 = 8;
        const SEED: u32 = 42;
        let prompt = "Once upon a time";
        // Oracle: an independent load of the same tiny file, driven through
        // the session's own chain (pre-change construction).
        let model = Model::load(Path::new(&smoke), &ModelParams::default()).unwrap();
        let prompt_tokens = model.tokenize(prompt, true).unwrap();
        let mut session = Session::new(
            &model,
            &SessionParams {
                n_ctx: 512,
                ..SessionParams::default()
            },
        )
        .unwrap();
        session.set_sampler(&SamplerParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            seed: SEED,
        });
        let mut expected = String::new();
        session
            .generate(&prompt_tokens, MAX as usize, |_, piece| {
                expected.push_str(piece);
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
        assert!(!expected.is_empty(), "the oracle run must emit text");

        let text = run_sampled_to_done(&host, &smoke, prompt, MAX, SEED);
        assert_eq!(
            text, expected,
            "an alone sampled run must match the session-chain construction \
             byte-for-byte"
        );
        // Determinism across host runs with the same seed (each admission
        // builds a fresh chain seeded identically).
        let again = run_sampled_to_done(&host, &smoke, prompt, MAX, SEED);
        assert_eq!(again, text, "fixed-seed sampling must be deterministic");
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    /// Per-sequence sampler chains: two INTERLEAVED fixed-seed sampled
    /// (temperature > 0) generations must each be byte-identical to their
    /// alone-runs. Pre-change, the shared session chain was reinstalled
    /// with a derived seed on every owner switch — a documented divergence
    /// this feature removes.
    #[test]
    fn interleaved_sampled_requests_match_their_alone_runs() {
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default()) else {
            return;
        };
        const MAX: u32 = 8;
        let prompt_a = "Once upon a time";
        let prompt_b = "The little dog";
        let alone_a = run_sampled_to_done(&host, &smoke, prompt_a, MAX, 7);
        let alone_b = run_sampled_to_done(&host, &smoke, prompt_b, MAX, 1234);
        assert!(!alone_a.is_empty() && !alone_b.is_empty());

        // A's channel holds ONE piece and is not read until B finishes, so
        // the two sequences provably interleave (B completes while A is
        // mid-generation) and sampling alternates between their chains.
        let (tx_a, mut rx_a) = mpsc::channel(1);
        let (otx_a, orx_a) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: sampled_job(smoke.clone(), prompt_a, MAX, 7, tx_a),
            resume: None,
            outcome: otx_a,
        }))
        .unwrap();
        let (tx_b, mut rx_b) = mpsc::channel(64);
        let (otx_b, orx_b) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: sampled_job(smoke.clone(), prompt_b, MAX, 1234, tx_b),
            resume: None,
            outcome: otx_b,
        }))
        .unwrap();
        let mut text_b = String::new();
        loop {
            match rx_b.blocking_recv().expect("B terminates while A stalls") {
                TokenEvent::Token(piece) => text_b.push_str(&piece),
                TokenEvent::Done(_) => break,
                TokenEvent::Error(e) => panic!("B errored: {e}"),
            }
        }
        let mut text_a = String::new();
        loop {
            match rx_a.blocking_recv().expect("A must terminate") {
                TokenEvent::Token(piece) => text_a.push_str(&piece),
                TokenEvent::Done(_) => break,
                TokenEvent::Error(e) => panic!("A errored: {e}"),
            }
        }
        assert!(matches!(
            orx_a.blocking_recv().unwrap(),
            GenOutcome::Finished
        ));
        assert!(matches!(
            orx_b.blocking_recv().unwrap(),
            GenOutcome::Finished
        ));
        assert_eq!(
            text_a, alone_a,
            "interleaved sampled A must be byte-identical to its alone-run"
        );
        assert_eq!(
            text_b, alone_b,
            "interleaved sampled B must be byte-identical to its alone-run"
        );
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn resume_redelivers_confirmed_but_unsent_pieces() {
        // M5 retry contract: the interrupted attempt confirmed two tokens
        // whose pieces never reached the client (pieces_sent = 0) — the
        // resumed attempt must deliver those pieces FIRST, then keep
        // generating; the client's text must never gain a gap.
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default()) else {
            return;
        };
        let (tx, mut rx) = mpsc::channel(8);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke, 3, tx),
            resume: Some(ResumeState {
                prompt_tokens: vec![1, 2, 3],
                generated_tokens: vec![4, 5],
                pieces_sent: 0,
                scan: StopScan::new(vec![]),
            }),
            outcome: otx,
        }))
        .unwrap();
        let mut pieces = 0u32;
        let done = loop {
            match rx.blocking_recv().expect("stream must terminate") {
                TokenEvent::Token(_) => pieces += 1,
                TokenEvent::Done(stats) => break stats,
                TokenEvent::Error(e) => panic!("unexpected error: {e}"),
            }
        };
        assert_eq!(done.prompt_tokens, 3, "original prompt length");
        assert!(
            pieces >= 2,
            "both carried-but-unsent pieces must be re-delivered (got {pieces})"
        );
        // The gap-free invariant: one piece per confirmed token, whether
        // re-delivered or newly generated (an immediate EOG after the
        // carried prefix legally yields exactly the two carried pieces).
        assert_eq!(pieces, done.completion_tokens);
        assert!(matches!(
            orx.blocking_recv().expect("outcome must arrive"),
            GenOutcome::Finished
        ));
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn spent_budget_resume_delivers_parked_pieces_before_done() {
        // The spent-budget resume edge with a delivery gap: the second
        // confirmed token's piece was parked when the attempt died, so the
        // resumed attempt must deliver exactly that piece and then finish
        // as Length — without touching the engine (no decode budget left).
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default()) else {
            return;
        };
        let (tx, mut rx) = mpsc::channel(8);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke, 2, tx),
            resume: Some(ResumeState {
                prompt_tokens: vec![1, 2, 3],
                generated_tokens: vec![4, 5],
                pieces_sent: 1,
                scan: StopScan::new(vec![]),
            }),
            outcome: otx,
        }))
        .unwrap();
        let mut pieces = 0u32;
        let done = loop {
            match rx.blocking_recv().expect("stream must terminate") {
                TokenEvent::Token(_) => pieces += 1,
                TokenEvent::Done(stats) => break stats,
                TokenEvent::Error(e) => panic!("unexpected error: {e}"),
            }
        };
        assert_eq!(pieces, 1, "exactly the one parked piece is re-delivered");
        assert_eq!(done.prompt_tokens, 3, "original prompt length");
        assert_eq!(done.completion_tokens, 2, "cumulative completion");
        assert_eq!(done.finish, FinishKind::Length);
        assert!(matches!(
            orx.blocking_recv().expect("outcome must arrive"),
            GenOutcome::Finished
        ));
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn decode_delay_slows_streaming_by_the_configured_amount() {
        // [debug] decode_delay_ms: 3 pieces at a 50 ms per-piece sleep must
        // take at least ~2 sleeps of wall time (conservative lower bound —
        // the knob exists to give the chaos sim a kill window).
        let Some((host, handle, smoke)) =
            spawn_with_smoke_model(Some(Duration::from_millis(50)), HostPerf::default())
        else {
            return;
        };
        let (tx, mut rx) = mpsc::channel(64);
        let (otx, orx) = oneshot::channel();
        let started = std::time::Instant::now();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: greedy_job(smoke, 3, tx),
            resume: None,
            outcome: otx,
        }))
        .unwrap();
        let mut tokens = 0;
        loop {
            match rx.blocking_recv().expect("stream must terminate") {
                TokenEvent::Token(_) => tokens += 1,
                TokenEvent::Done(_) => break,
                TokenEvent::Error(e) => panic!("unexpected error: {e}"),
            }
        }
        assert!(matches!(
            orx.blocking_recv().expect("outcome must arrive"),
            GenOutcome::Finished
        ));
        if tokens >= 3 {
            assert!(
                started.elapsed() >= Duration::from_millis(100),
                "3 pieces at 50 ms each must take at least ~100 ms, took {:?}",
                started.elapsed()
            );
        }
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn stop_scan_holds_back_the_completing_piece() {
        // Contract example: pieces "he" · "llo w" · "orld" with stop "lo w"
        // must emit only "he" and finish with Stop.
        let mut scan = StopScan::new(vec!["lo w".into()]);
        let mut emitted = String::new();
        let mut stop_matched = false;
        for piece in ["he", "llo w", "orld"] {
            if scan.admit(piece) {
                emitted.push_str(piece);
            } else {
                stop_matched = true;
                break;
            }
        }
        assert_eq!(emitted, "he");
        assert!(stop_matched, "the stop string must be detected");
    }

    #[test]
    fn stop_scan_without_stops_admits_everything() {
        let mut scan = StopScan::new(vec![]);
        for piece in ["a", "b", "c"] {
            assert!(scan.admit(piece));
        }
    }

    #[test]
    fn stop_scan_matches_across_many_small_pieces() {
        let mut scan = StopScan::new(vec!["STOP".into()]);
        assert!(scan.admit("S"));
        assert!(scan.admit("T"));
        assert!(scan.admit("O"));
        assert!(!scan.admit("P"), "the final piece completes the match");
    }

    #[test]
    fn stop_scan_ignores_empty_stop_strings() {
        let mut scan = StopScan::new(vec![String::new()]);
        assert!(scan.admit("anything"));
    }

    #[test]
    fn progress_throttle_emits_first_and_final() {
        let mut t = ProgressThrottle::default();
        assert!(t.should_emit(0, 100));
        assert!(!t.should_emit(50, 100), "under the 1 MiB minimum step");
        assert!(t.should_emit(100, 100), "the final byte always reports");
    }

    #[test]
    fn generate_before_load_reports_no_model() {
        let (host, handle) = EngineHost::spawn(None, HostPerf::default());
        let (tx, mut rx) = mpsc::channel(8);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: GenerateJob {
                model: "anything".into(),
                prompt: PromptInput::Raw("hi".into()),
                params: Default::default(),
                dialect: onebrain_api::backend::ApiDialect::Openai,
                tx,
            },
            resume: None,
            outcome: otx,
        }))
        .unwrap();
        let event = rx.blocking_recv().expect("host must terminate the stream");
        match event {
            TokenEvent::Error(message) => {
                assert!(message.contains("no model is loaded"), "got: {message}");
                assert!(
                    message.contains("onebrain run"),
                    "remedy missing: {message}"
                );
            }
            other => panic!("expected Error, got {other:?}"),
        }
        // The attempt outcome is Finished: the stream was terminated here,
        // nothing for the supervisor to retry.
        match orx.blocking_recv().expect("host must report an outcome") {
            GenOutcome::Finished => {}
            other => panic!("expected Finished, got {other:?}"),
        }
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn models_before_load_is_none() {
        let (host, handle) = EngineHost::spawn(None, HostPerf::default());
        assert_eq!(host.loaded_model(Duration::from_secs(5)), None);
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn idle_probe_tracks_the_job_counter() {
        let (host, handle) = EngineHost::spawn(None, HostPerf::default());
        assert!(host.is_idle(), "a fresh host has no jobs");
        host.job_started();
        assert!(!host.is_idle(), "a queued job must clear the idle probe");
        host.job_started();
        host.job_finished();
        assert!(!host.is_idle(), "one of two jobs finishing is not idle");
        host.job_finished();
        assert!(host.is_idle(), "all jobs finished; idle again");
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    /// The admission counter (docs/perf.md §6): jobs count only while room
    /// exists; a rejected job leaves the counter untouched.
    #[test]
    fn try_start_job_enforces_the_limit() {
        let (host, handle) = EngineHost::spawn(None, HostPerf::default());
        assert!(host.try_start_job(2));
        assert!(host.try_start_job(2));
        assert!(!host.try_start_job(2), "third job exceeds the limit");
        assert!(!host.is_idle());
        host.job_finished();
        assert!(host.try_start_job(2), "a finished job frees admission room");
        host.job_finished();
        host.job_finished();
        assert!(host.is_idle(), "rejected jobs must not leak counts");
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    /// Admission control end to end at the backend boundary
    /// (docs/perf.md §6): with `max_concurrent_requests = 1` and
    /// `queue_depth = 1`, the third simultaneous generation is rejected
    /// with the typed 429-equivalent — remedy included — and never enters
    /// the daemon.
    #[tokio::test(flavor = "multi_thread")]
    async fn admission_rejects_beyond_queue_depth_with_429() {
        let dir = tempfile::tempdir().unwrap();
        let key = onebrain_mesh::identity::load_or_create(dir.path()).unwrap();
        let mesh = onebrain_mesh::MeshService::spawn(
            key,
            dir.path().join("peers.toml"),
            "test-node".to_string(),
            onebrain_mesh::MeshConfig {
                enable_mdns: false,
                enable_relays: false,
                engine_build: "test-build".to_string(),
                bind_addrs: vec![(std::net::Ipv4Addr::LOCALHOST, 0).into()],
                ..onebrain_mesh::MeshConfig::default()
            },
        )
        .await
        .unwrap();
        let (host, handle) = EngineHost::spawn(None, HostPerf::default());
        // The receiver stays alive so sends succeed; nothing consumes the
        // jobs — they occupy admission slots, exactly the scenario.
        let (sup_tx, _sup_rx) = crate::supervisor::channel();
        let backend = DaemonBackend::new(
            host.clone(),
            dir.path().to_path_buf(),
            sup_tx,
            mesh,
            0,
            /* max_concurrent */ 1,
            /* queue_depth */ 1,
            crate::metrics::RequestLog::new(),
        );
        let make_job = || {
            let (tx, rx) = mpsc::channel(4);
            (
                GenerateJob {
                    model: "m".into(),
                    prompt: PromptInput::Raw("hi".into()),
                    params: Default::default(),
                    dialect: onebrain_api::backend::ApiDialect::Openai,
                    tx,
                },
                rx,
            )
        };
        let (job1, _rx1) = make_job();
        let (job2, _rx2) = make_job();
        let (job3, _rx3) = make_job();
        backend.generate(job1).expect("first job admitted (runs)");
        backend
            .generate(job2)
            .expect("second job admitted (queues)");
        let err = backend
            .generate(job3)
            .expect_err("third job exceeds max_concurrent + queue_depth");
        assert!(
            matches!(
                err,
                ApiError::Overloaded {
                    max_concurrent: 1,
                    queue_depth: 1
                }
            ),
            "expected Overloaded, got {err:?}"
        );
        let message = err.to_string();
        assert!(message.contains("at capacity"), "got: {message}");
        assert!(
            message.contains("max_concurrent_requests"),
            "remedy missing: {message}"
        );
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn common_prefix_len_measures_shared_tokens() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9]), 2);
        assert_eq!(common_prefix_len(&[], &[1]), 0);
        assert_eq!(common_prefix_len(&[7], &[1]), 0);
    }

    /// The §5 same-vocab check: the pure comparison names exactly what
    /// differed. (A real mismatched GGUF pair is exercised by the sim; the
    /// same-model pass runs in the speculative tests below.)
    #[test]
    fn vocab_mismatch_names_the_difference() {
        let base = VocabFingerprint {
            tokenizer: Some("llama".into()),
            tokens: vec![1, 5, 9],
            pieces: vec!["a".into(), "b".into(), "c".into()],
        };
        assert_eq!(vocab_mismatch(&base, &base.clone()), None);
        // Absent tokenizer metadata on one side is not by itself a
        // mismatch — the behavioral probes below decide.
        let unnamed = VocabFingerprint {
            tokenizer: None,
            ..base.clone()
        };
        assert_eq!(vocab_mismatch(&base, &unnamed), None);
        let family = VocabFingerprint {
            tokenizer: Some("gpt2".into()),
            ..base.clone()
        };
        let msg = vocab_mismatch(&base, &family).expect("tokenizer family differs");
        assert!(msg.contains("different tokenizers"), "got: {msg}");
        assert!(msg.contains("llama") && msg.contains("gpt2"), "got: {msg}");
        let ids = VocabFingerprint {
            tokens: vec![1, 5],
            ..base.clone()
        };
        let msg = vocab_mismatch(&base, &ids).expect("token ids differ");
        assert!(msg.contains("tokenizes differently"), "got: {msg}");
        let pieces = VocabFingerprint {
            pieces: vec!["a".into(), "b".into(), "X".into()],
            ..base.clone()
        };
        let msg = vocab_mismatch(&base, &pieces).expect("piece text differs");
        assert!(msg.contains("renders differently"), "got: {msg}");
        assert!(msg.contains('9'), "must name the token id: {msg}");
    }

    /// docs/perf.md §5 DoD: with the smoke model drafting for itself,
    /// greedy output is byte-identical to the plain path, the counters
    /// flow into DoneStats, and (draft == target ⇒ greedy determinism)
    /// every drafted token is accepted.
    #[test]
    fn speculative_greedy_matches_plain_run_and_counts() {
        let Some((plain, plain_handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default())
        else {
            return;
        };
        const MAX: u32 = 16;
        let (text_plain, stats_plain) = run_to_done(&plain, &smoke, "Once upon a time", MAX);
        plain.send(HostMsg::Shutdown).unwrap();
        plain_handle.join().unwrap();
        assert_eq!(stats_plain.drafted, 0, "no draft loaded, no drafting");

        let Some((spec, spec_handle, smoke)) = spawn_smoke(None, HostPerf::default(), true) else {
            return;
        };
        let (text_spec, stats_spec) = run_to_done(&spec, &smoke, "Once upon a time", MAX);
        assert_eq!(
            text_spec, text_plain,
            "speculative greedy output must be byte-identical to the plain path"
        );
        assert_eq!(stats_spec.completion_tokens, stats_plain.completion_tokens);
        assert!(
            stats_spec.drafted > 0,
            "the draft must have proposed tokens"
        );
        assert!(
            stats_spec.accepted > 0,
            "the target must have accepted some"
        );
        assert_eq!(
            stats_spec.accepted, stats_spec.drafted,
            "draft == target ⇒ greedy determinism accepts every proposal"
        );
        assert!(stats_spec.accepted <= stats_spec.completion_tokens);
        // A second run on the same host composes speculation with §4
        // prefix reuse (both default-on): still byte-identical.
        let (text_again, stats_again) = run_to_done(&spec, &smoke, "Once upon a time", MAX);
        assert_eq!(
            text_again, text_plain,
            "reuse + speculation must stay exact"
        );
        assert!(stats_again.drafted > 0);
        spec.send(HostMsg::Shutdown).unwrap();
        spec_handle.join().unwrap();
    }

    /// docs/perf.md §5: temperature > 0 with a draft loaded runs the plain
    /// target path (with a logged notice) — no drafting.
    #[test]
    fn speculative_sampled_request_runs_target_path() {
        let Some((host, handle, smoke)) = spawn_smoke(None, HostPerf::default(), true) else {
            return;
        };
        let (tx, mut rx) = mpsc::channel(64);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: GenerateJob {
                model: smoke.clone(),
                prompt: PromptInput::Raw("Once upon a time".into()),
                params: onebrain_api::backend::GenParams {
                    max_tokens: 8,
                    temperature: 0.8,
                    seed: Some(42),
                    ..Default::default()
                },
                dialect: onebrain_api::backend::ApiDialect::Openai,
                tx,
            },
            resume: None,
            outcome: otx,
        }))
        .unwrap();
        let done = loop {
            match rx.blocking_recv().expect("stream must terminate") {
                TokenEvent::Token(_) => {}
                TokenEvent::Done(stats) => break stats,
                TokenEvent::Error(e) => panic!("unexpected error: {e}"),
            }
        };
        assert_eq!(done.drafted, 0, "sampled requests must not speculate");
        assert_eq!(done.accepted, 0);
        assert!(matches!(
            orx.blocking_recv().expect("outcome"),
            GenOutcome::Finished
        ));
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    /// docs/perf.md §6 scheduling rule: with two active generations the
    /// speculative path stands down — and byte-identity with the alone
    /// runs still holds (the alone runs themselves speculate).
    #[test]
    fn speculative_stands_down_with_two_active_and_stays_exact() {
        let Some((host, handle, smoke)) = spawn_smoke(None, HostPerf::default(), true) else {
            return;
        };
        const MAX: u32 = 8;
        let prompt_a = "Once upon a time";
        let prompt_b = "The little dog";
        let (alone_a, _) = run_to_done(&host, &smoke, prompt_a, MAX);
        let (alone_b, _) = run_to_done(&host, &smoke, prompt_b, MAX);

        // A's channel holds ONE piece and is not read until B finishes.
        let (tx_a, mut rx_a) = mpsc::channel(1);
        let (otx_a, orx_a) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: job_with_prompt(smoke.clone(), prompt_a, MAX, tx_a),
            resume: None,
            outcome: otx_a,
        }))
        .unwrap();
        let (tx_b, mut rx_b) = mpsc::channel(64);
        let (otx_b, orx_b) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: job_with_prompt(smoke.clone(), prompt_b, MAX, tx_b),
            resume: None,
            outcome: otx_b,
        }))
        .unwrap();
        let mut text_b = String::new();
        loop {
            match rx_b.blocking_recv().expect("B terminates while A stalls") {
                TokenEvent::Token(piece) => text_b.push_str(&piece),
                TokenEvent::Done(_) => break,
                TokenEvent::Error(e) => panic!("B errored: {e}"),
            }
        }
        let mut text_a = String::new();
        loop {
            match rx_a.blocking_recv().expect("A must terminate") {
                TokenEvent::Token(piece) => text_a.push_str(&piece),
                TokenEvent::Done(_) => break,
                TokenEvent::Error(e) => panic!("A errored: {e}"),
            }
        }
        assert!(matches!(
            orx_a.blocking_recv().unwrap(),
            GenOutcome::Finished
        ));
        assert!(matches!(
            orx_b.blocking_recv().unwrap(),
            GenOutcome::Finished
        ));
        assert_eq!(text_a, alone_a, "concurrent A must match its alone-run");
        assert_eq!(text_b, alone_b, "concurrent B must match its alone-run");
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    /// Ask the host to embed `texts` against `model`; returns the typed
    /// result the way the gateway receives it.
    fn run_embed(host: &EngineHost, model: &str, texts: &[&str]) -> Result<EmbedResult, ApiError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        host.send(HostMsg::Embed(EmbedJob {
            model: model.to_string(),
            texts: texts.iter().map(|t| t.to_string()).collect(),
            resp: resp_tx,
        }))
        .unwrap();
        resp_rx.blocking_recv().expect("host must answer embeds")
    }

    /// The embeddings surface through the host: dims match the model's
    /// n_embd, vectors are finite unit-norm and deterministic, name
    /// validation is typed, and a generation still runs cleanly afterwards
    /// (the short-lived embed session leaves the serving session alone).
    #[test]
    fn embed_via_host_returns_unit_norm_vectors_and_typed_errors() {
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None, HostPerf::default()) else {
            return;
        };
        // Independent load of the same tiny file is the n_embd oracle.
        let n_embd = Model::load(Path::new(&smoke), &ModelParams::default())
            .expect("smoke model loads")
            .n_embd() as usize;

        let result = run_embed(&host, &smoke, &["Once upon a time", "The little dog"])
            .expect("embed succeeds");
        assert_eq!(result.embeddings.len(), 2);
        assert!(result.prompt_tokens > 0);
        for vector in &result.embeddings {
            assert_eq!(vector.len(), n_embd, "vector must be n_embd wide");
            assert!(vector.iter().all(|v| v.is_finite()));
            let norm: f64 = vector
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "vectors must be unit L2 norm (got {norm})"
            );
        }
        assert_ne!(
            result.embeddings[0], result.embeddings[1],
            "different texts must embed differently"
        );
        let again = run_embed(&host, &smoke, &["Once upon a time"]).expect("embed succeeds");
        assert_eq!(
            again.embeddings[0], result.embeddings[0],
            "embeddings must be deterministic across requests"
        );

        match run_embed(&host, "not-the-loaded-model", &["hi"]) {
            Err(ApiError::ModelNotLoaded(name)) => assert_eq!(name, "not-the-loaded-model"),
            other => panic!("expected ModelNotLoaded, got {other:?}"),
        }

        // The serving session is untouched: a generation still completes.
        let (text, _) = run_to_done(&host, &smoke, "Once upon a time", 4);
        assert!(!text.is_empty(), "generation must still work after embeds");
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    /// With nothing loaded, an embed fails typed with NoModel (mirrors the
    /// Generate posture in phase 1).
    #[test]
    fn embed_without_model_is_no_model_typed() {
        let (host, handle) = EngineHost::spawn(None, HostPerf::default());
        match run_embed(&host, "anything", &["hi"]) {
            Err(ApiError::NoModel) => {}
            other => panic!("expected NoModel, got {other:?}"),
        }
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn load_with_unknown_reference_fails_with_remedy() {
        let dir = tempfile::tempdir().unwrap();
        let (host, handle) = EngineHost::spawn(None, HostPerf::default());
        let (ptx, _prx) = mpsc::unbounded_channel();
        let (rtx, rrx) = oneshot::channel();
        host.send(HostMsg::Load {
            reference: "definitely-not-a-model".into(),
            cache_root: dir.path().to_path_buf(),
            ctx_len: 4096,
            draft: None,
            progress: ptx,
            resp: rtx,
        })
        .unwrap();
        let outcome = rrx.blocking_recv().expect("host must answer");
        let message = outcome.expect_err("unknown reference must fail");
        assert!(message.contains("definitely-not-a-model"), "got: {message}");
        assert!(
            message.contains("hf:<org>/<repo>/<file>.gguf"),
            "remedy missing: {message}"
        );
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }
}
