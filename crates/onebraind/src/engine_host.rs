//! The engine-owning OS thread and the [`EngineBackend`] handle over it.
//!
//! One `std::thread` owns the loaded [`Model`] and its single [`Session`]
//! (M1 concurrency model: jobs queue on the channel and run serially,
//! internal-api contract "Engine host"). The HTTP side talks to it through
//! [`EngineHost`] (cheap clonable sender) and [`DaemonBackend`], the
//! [`EngineBackend`] implementation the gateway routes into.
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
//! re-plan, and retry transparently into the same client stream. Solo-model
//! decode failures keep the pre-M5 behavior: a terminal
//! [`TokenEvent::Error`] on `job.tx`.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Duration;

use onebrain_api::backend::{
    DoneStats, EngineBackend, FinishKind, GenerateJob, ModelSummary, PromptInput, PullEvent,
    TokenEvent,
};
use onebrain_api::ApiError;
use onebrain_engine::rpc::RemoteServer;
use onebrain_engine::{
    EngineError, FinishReason, Model, ModelParams, SamplerParams, Session, SessionParams, Token,
};
use onebrain_models::registry::{ModelRef, Resolved};
use onebrain_models::{cache, download};
use onebrain_proto::plan::Epoch;
use serde::Serialize;
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
    /// successful completions, validation failures, solo decode errors.
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
    /// later attempt keeps the cumulative count).
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
        resp: oneshot::Sender<Result<LoadedModel, String>>,
    },
    /// Run one supervised generation attempt (M5): client-visible events
    /// flow through `job.tx`, the attempt outcome through `outcome` —
    /// except a distributed decode failure, which reaches ONLY `outcome`
    /// (the supervisor decides what the client sees).
    Generate(SupervisedGenerate),
    /// Ask what is loaded. Answered over a std channel so non-async callers
    /// (the sync [`EngineBackend::models`]) can wait with a timeout; the
    /// host replies with `try_send`, so a caller that gave up never blocks
    /// or wedges the host.
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
    /// the supervisor (queued + in flight). Feeds [`EngineHost::is_idle`],
    /// which gates the M5 lazy re-plan and the no-job-in-flight death
    /// teardown (docs/resilience.md).
    jobs: Arc<AtomicUsize>,
}

impl EngineHost {
    /// Start the engine-host thread. Join the returned handle after sending
    /// [`HostMsg::Shutdown`] and before calling `onebrain_engine::shutdown`.
    /// `decode_delay` is the test-only `[debug] decode_delay_ms` knob
    /// (docs/resilience.md): when set the host sleeps that long after
    /// emitting each token piece; `None` (all real deployments) adds no
    /// delay anywhere.
    pub fn spawn(decode_delay: Option<Duration>) -> (EngineHost, std::thread::JoinHandle<()>) {
        let (tx, rx) = std_mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("engine-host".into())
            .spawn(move || host_loop(rx, decode_delay))
            .expect("spawning the engine host thread failed; the system is out of resources");
        (
            EngineHost {
                tx,
                jobs: Arc::new(AtomicUsize::new(0)),
            },
            handle,
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

    /// The supervisor fully finished a job (including any retry).
    pub fn job_finished(&self) {
        self.jobs.fetch_sub(1, Ordering::SeqCst);
    }

    /// `true` when no generation job is queued or in flight (M5 idle probe:
    /// gates the lazy rejoin re-plan and the idle death teardown).
    pub fn is_idle(&self) -> bool {
        self.jobs.load(Ordering::SeqCst) == 0
    }

    /// Blocking round-trip for the loaded-model summary. `None` after
    /// `timeout` means either nothing is loaded or the host is busy with a
    /// long generation — callers degrade to "nothing loaded" rather than
    /// stalling a status endpoint.
    pub fn loaded_model(&self, timeout: Duration) -> Option<LoadedModel> {
        let (resp, rx) = std_mpsc::sync_channel(1);
        self.send(HostMsg::Models { resp }).ok()?;
        rx.recv_timeout(timeout).ok().flatten()
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
    resp: oneshot::Sender<Result<LoadedModel, String>>,
}

/// A load request of either flavor, stashed while the current model drops.
enum Pending {
    Solo(LoadReq),
    Dist(DistLoadReq),
}

fn host_loop(rx: std_mpsc::Receiver<HostMsg>, decode_delay: Option<Duration>) {
    // Small runtime owned by this thread, used only to drive downloads.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("building the download runtime failed; the system is out of resources");

    let mut pending: Option<Pending> = None;
    // Worker state: the epoch this node is serving a shard for (informational
    // — the serve threads live in the daemon's cluster task).
    let mut serving_shard: Option<u64> = None;
    'outer: loop {
        // Phase 1: nothing loaded — wait for a load request.
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
                        progress,
                        resp,
                    }) => {
                        break Pending::Solo(LoadReq {
                            reference,
                            cache_root,
                            ctx_len,
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
                            resp,
                        })
                    }
                }
            },
        };

        // Phase 2: obtain a loaded model (solo: resolve + download + load;
        // distributed: register bridged RPC servers + split load). The
        // `distributed` flag decides M5 decode-failure handling: only a
        // distributed model's decode failure is supervisor-retryable.
        let (model, info, reference, resp, distributed) = match req {
            Pending::Solo(req) => {
                let LoadReq {
                    reference,
                    cache_root,
                    ctx_len,
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
                (model, info, reference, resp, false)
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
                    resp,
                } = req;
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
                (model, info, reference, resp, true)
            }
        };
        let mut session = match Session::new(
            &model,
            &SessionParams {
                n_ctx: info.n_ctx,
                n_batch: 512,
                n_threads: 0,
            },
        ) {
            Ok(s) => s,
            Err(e) => {
                let _ = resp.send(Err(e.to_string()));
                continue 'outer; // drops `model`
            }
        };
        tracing::info!(model = %info.name, n_layer = info.n_layer, n_ctx = info.n_ctx, "model loaded");
        let _ = resp.send(Ok(info.clone()));

        // Phase 3: serve jobs until another load replaces this model, an
        // unload tears it down, or a plan turns this node into a worker.
        loop {
            match rx.recv() {
                Err(_) | Ok(HostMsg::Shutdown) => return,
                Ok(HostMsg::Models { resp }) => {
                    let _ = resp.try_send(Some(info.clone()));
                }
                Ok(HostMsg::Generate(sup)) => run_generation(
                    &mut session,
                    &info,
                    &reference,
                    distributed,
                    decode_delay,
                    sup,
                ),
                Ok(HostMsg::Unload { resp }) => {
                    tracing::info!(model = %info.name, "unloading model");
                    // Drop BEFORE replying: the daemon sequences epoch
                    // teardown on this reply (free the model while its RPC
                    // bridges still stand, then close them — ADR 0004).
                    drop(session);
                    drop(model);
                    let _ = resp.send(());
                    continue 'outer;
                }
                Ok(HostMsg::ServeShard { epoch }) => {
                    // M3 contract: adopting a plan while a local model is
                    // loaded unloads it — the plan needs this node's memory.
                    tracing::info!(
                        model = %info.name,
                        epoch = epoch.0,
                        "unloading local model to serve a plan shard"
                    );
                    serving_shard = Some(epoch.0);
                    drop(session);
                    drop(model);
                    continue 'outer;
                }
                Ok(HostMsg::Load {
                    reference,
                    cache_root,
                    ctx_len,
                    progress,
                    resp,
                }) => {
                    tracing::info!(model = %info.name, next = %reference, "unloading for a new model");
                    pending = Some(Pending::Solo(LoadReq {
                        reference,
                        cache_root,
                        ctx_len,
                        progress,
                        resp,
                    }));
                    // Loading a second model unloads the first (contract):
                    // `session` and `model` drop as this scope ends.
                    continue 'outer;
                }
                Ok(HostMsg::LoadDistributed {
                    paths,
                    reference,
                    name,
                    ctx_len,
                    endpoints,
                    tensor_split,
                    use_local_device,
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
                        resp,
                    }));
                    continue 'outer;
                }
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

/// One queued generation attempt, executed to completion on the host
/// thread. Fresh attempts tokenize the prompt; retries (`sup.resume`)
/// prefill the carried prompt + generated tokens and continue sampling,
/// streaming only NEW pieces (docs/resilience.md step 4). A decode failure
/// against a distributed model reports [`GenOutcome::Interrupted`] instead
/// of a terminal client event; everything else terminates `job.tx` exactly
/// as before M5.
fn run_generation(
    session: &mut Session<'_>,
    info: &LoadedModel,
    loaded_reference: &str,
    distributed: bool,
    decode_delay: Option<Duration>,
    sup: SupervisedGenerate,
) {
    let SupervisedGenerate {
        job,
        resume,
        outcome,
    } = sup;
    // Terminate the stream with an error; the attempt is Finished.
    let finish_error =
        |job: &GenerateJob, outcome: oneshot::Sender<GenOutcome>, message: String| {
            let _ = job.tx.blocking_send(TokenEvent::Error(message));
            let _ = outcome.send(GenOutcome::Finished);
        };

    // Accept the canonical loaded name and the reference the load was
    // requested with (`hf:…` refs cache under a sanitized key; clients may
    // use either spelling).
    if job.model != info.name && job.model != loaded_reference {
        let message = ApiError::ModelNotLoaded(job.model.clone()).to_string();
        finish_error(&job, outcome, message);
        return;
    }

    let max_tokens = job.params.max_tokens as usize;
    let (prompt_tokens, prior_generated, prior_pieces, mut scan) = match resume {
        Some(state) => (
            state.prompt_tokens,
            state.generated_tokens,
            state.pieces_sent,
            state.scan,
        ),
        None => {
            let prompt_text = match render_prompt(session.model(), info, &job.prompt) {
                Ok(text) => text,
                Err(message) => {
                    finish_error(&job, outcome, message);
                    return;
                }
            };
            let prompt_tokens = match session.model().tokenize(&prompt_text, true) {
                Ok(toks) => toks,
                Err(e) => {
                    finish_error(&job, outcome, e.to_string());
                    return;
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
                return;
            }
            (
                prompt_tokens,
                Vec::new(),
                0,
                StopScan::new(job.params.stop.clone()),
            )
        }
    };
    // Budget left for this attempt. The fresh path checked prompt +
    // max_tokens against n_ctx, and prefix + remaining equals exactly that
    // sum, so no re-check is needed on resume.
    let remaining = max_tokens.saturating_sub(prior_generated.len());
    if remaining == 0 {
        // Interrupted on the very last token's decode: every piece already
        // reached the client — finish as Length without touching the engine.
        let _ = job.tx.blocking_send(TokenEvent::Done(DoneStats {
            prompt_tokens: prompt_tokens.len() as u32,
            completion_tokens: prior_generated.len() as u32,
            finish: FinishKind::Length,
        }));
        let _ = outcome.send(GenOutcome::Finished);
        return;
    }
    let prefix: Vec<Token> = prompt_tokens
        .iter()
        .chain(prior_generated.iter())
        .copied()
        .collect();

    session.reset();
    // On a retry the sampler (and its seed chain) restarts — exact for
    // greedy/temp<=0; documented-acceptable for sampled runs
    // (docs/resilience.md step 4).
    session.set_sampler(&SamplerParams {
        temperature: job.params.temperature,
        top_p: job.params.top_p,
        top_k: job.params.top_k,
        seed: job.params.seed.unwrap_or(0xFFFF_FFFF),
    });

    let mut stop_matched = false;
    let mut attempt_tokens: Vec<Token> = Vec::new();
    let mut pieces_sent = prior_pieces;
    let result = session.generate(&prefix, remaining, |tok, piece| {
        // Record BEFORE the send/decode: a token whose piece reached the
        // client must be part of any retry prefix.
        attempt_tokens.push(tok);
        if !scan.admit(piece) {
            // The piece completes a stop-string match: hold it back and end
            // the generation (contract: finish Stop without sending it).
            stop_matched = true;
            return ControlFlow::Break(());
        }
        match job.tx.blocking_send(TokenEvent::Token(piece.to_string())) {
            Ok(()) => {
                pieces_sent += 1;
                if let Some(delay) = decode_delay {
                    // Test-only `[debug] decode_delay_ms`: a deterministic
                    // kill window for the chaos sim.
                    std::thread::sleep(delay);
                }
                ControlFlow::Continue(())
            }
            Err(_) => ControlFlow::Break(()), // client went away
        }
    });
    match result {
        Err(e @ EngineError::Decode { .. }) if distributed => {
            // M5 failure lifecycle step 2: nothing terminal on job.tx — the
            // supervisor owns retry-or-fail. The torn model stays loaded
            // until the supervisor unloads it (patched frees tolerate dead
            // bridges, patches/0002).
            tracing::warn!(
                error = %e,
                generated = prior_generated.len() + attempt_tokens.len(),
                pieces_sent,
                "distributed decode failed; reporting the interruption to the supervisor"
            );
            let mut generated_tokens = prior_generated;
            generated_tokens.extend(attempt_tokens);
            let _ = outcome.send(GenOutcome::Interrupted(Box::new(InterruptedGen {
                job,
                prompt_tokens,
                generated_tokens,
                pieces_sent,
                scan,
                error: e.to_string(),
            })));
        }
        Err(e) => finish_error(&job, outcome, e.to_string()),
        Ok(stats) => {
            let finish = match stats.finished {
                FinishReason::Stop => FinishKind::Stop,
                FinishReason::Length => FinishKind::Length,
                FinishReason::Aborted if stop_matched => FinishKind::Stop,
                FinishReason::Aborted => FinishKind::Abort,
            };
            // Client-visible stats span ALL attempts: the original prompt
            // length and the cumulative completion count.
            let _ = job.tx.blocking_send(TokenEvent::Done(DoneStats {
                prompt_tokens: prompt_tokens.len() as u32,
                completion_tokens: (prior_generated.len() + stats.generated_tokens) as u32,
                finish,
            }));
            let _ = outcome.send(GenOutcome::Finished);
        }
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
/// the model cache.
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
}

impl DaemonBackend {
    pub fn new(
        host: EngineHost,
        cache_root: PathBuf,
        supervisor: SupervisorTx,
        mesh: onebrain_mesh::MeshHandle,
        cache_max_bytes: u64,
    ) -> DaemonBackend {
        DaemonBackend {
            host,
            cache_root,
            supervisor,
            mesh,
            cache_max_bytes,
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
        // Count the job BEFORE it is visible to the supervisor so the M5
        // idle probe never reads idle while a job is en route; the
        // supervisor decrements when the job's lifecycle fully ends.
        self.host.job_started();
        if self.supervisor.send(SupervisorMsg::Generate(job)).is_err() {
            self.host.job_finished();
            return Err(ApiError::ShuttingDown);
        }
        Ok(())
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
    ) -> Option<(EngineHost, std::thread::JoinHandle<()>, String)> {
        let Ok(smoke) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping supervised generation test");
            return None;
        };
        let (host, handle) = EngineHost::spawn(decode_delay);
        let (ptx, _prx) = mpsc::unbounded_channel();
        let (rtx, rrx) = oneshot::channel();
        host.send(HostMsg::Load {
            reference: smoke.clone(),
            cache_root: std::env::temp_dir(),
            ctx_len: 512,
            progress: ptx,
            resp: rtx,
        })
        .unwrap();
        rrx.blocking_recv()
            .expect("host answers")
            .expect("smoke model loads");
        Some((host, handle, smoke))
    }

    fn greedy_job(model: String, max_tokens: u32, tx: mpsc::Sender<TokenEvent>) -> GenerateJob {
        GenerateJob {
            model,
            prompt: PromptInput::Raw("Once upon a time".into()),
            params: onebrain_api::backend::GenParams {
                max_tokens,
                temperature: 0.0,
                ..Default::default()
            },
            tx,
        }
    }

    #[test]
    fn supervised_generation_streams_and_reports_finished() {
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None) else {
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

    #[test]
    fn supervised_resume_with_spent_budget_finishes_as_length() {
        // The retry edge case: interrupted on the very last token's decode —
        // every piece already reached the client, so the resumed attempt
        // must terminate as Length with the CUMULATIVE stats and without
        // re-sending anything.
        let Some((host, handle, smoke)) = spawn_with_smoke_model(None) else {
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
    fn decode_delay_slows_streaming_by_the_configured_amount() {
        // [debug] decode_delay_ms: 3 pieces at a 50 ms per-piece sleep must
        // take at least ~2 sleeps of wall time (conservative lower bound —
        // the knob exists to give the chaos sim a kill window).
        let Some((host, handle, smoke)) = spawn_with_smoke_model(Some(Duration::from_millis(50)))
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
        let (host, handle) = EngineHost::spawn(None);
        let (tx, mut rx) = mpsc::channel(8);
        let (otx, orx) = oneshot::channel();
        host.send(HostMsg::Generate(SupervisedGenerate {
            job: GenerateJob {
                model: "anything".into(),
                prompt: PromptInput::Raw("hi".into()),
                params: Default::default(),
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
        let (host, handle) = EngineHost::spawn(None);
        assert_eq!(host.loaded_model(Duration::from_secs(5)), None);
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn idle_probe_tracks_the_job_counter() {
        let (host, handle) = EngineHost::spawn(None);
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

    #[test]
    fn load_with_unknown_reference_fails_with_remedy() {
        let dir = tempfile::tempdir().unwrap();
        let (host, handle) = EngineHost::spawn(None);
        let (ptx, _prx) = mpsc::unbounded_channel();
        let (rtx, rrx) = oneshot::channel();
        host.send(HostMsg::Load {
            reference: "definitely-not-a-model".into(),
            cache_root: dir.path().to_path_buf(),
            ctx_len: 4096,
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
