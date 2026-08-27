//! The engine-owning OS thread and the [`EngineBackend`] handle over it.
//!
//! One `std::thread` owns the loaded [`Model`] and its single [`Session`]
//! (M1 concurrency model: jobs queue on the channel and run serially,
//! internal-api contract "Engine host"). The HTTP side talks to it through
//! [`EngineHost`] (cheap clonable sender) and [`DaemonBackend`], the
//! [`EngineBackend`] implementation the gateway routes into.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::time::Duration;

use onebrain_api::backend::{
    DoneStats, EngineBackend, FinishKind, GenerateJob, ModelSummary, PromptInput, PullEvent,
    TokenEvent,
};
use onebrain_api::ApiError;
use onebrain_engine::rpc::RemoteServer;
use onebrain_engine::{FinishReason, Model, ModelParams, SamplerParams, Session, SessionParams};
use onebrain_models::registry::{ModelRef, Resolved};
use onebrain_models::{cache, download};
use onebrain_proto::plan::Epoch;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};

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

/// Messages into the engine-host thread.
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
        /// Local GGUF path (the head holds the full file — ADR 0004).
        path: PathBuf,
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
    /// Run one generation; all outcomes flow through `job.tx`.
    Generate(GenerateJob),
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
}

impl EngineHost {
    /// Start the engine-host thread. Join the returned handle after sending
    /// [`HostMsg::Shutdown`] and before calling `onebrain_engine::shutdown`.
    pub fn spawn() -> (EngineHost, std::thread::JoinHandle<()>) {
        let (tx, rx) = std_mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("engine-host".into())
            .spawn(move || host_loop(rx))
            .expect("spawning the engine host thread failed; the system is out of resources");
        (EngineHost { tx }, handle)
    }

    /// Send a message; `Err` means the host thread is gone (shutdown).
    pub fn send(&self, msg: HostMsg) -> Result<(), ApiError> {
        self.tx.send(msg).map_err(|_| ApiError::ShuttingDown)
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

/// A model reference resolved to a local file, downloaded if needed.
struct ResolvedLocal {
    name: String,
    path: PathBuf,
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
    path: PathBuf,
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

fn host_loop(rx: std_mpsc::Receiver<HostMsg>) {
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
                    Ok(HostMsg::Generate(job)) => {
                        let message = match serving_shard {
                            Some(epoch) => format!(
                                "{} (this node is serving a pipeline shard for epoch {epoch}; \
                                 send generations to the cluster head)",
                                ApiError::NoModel
                            ),
                            None => ApiError::NoModel.to_string(),
                        };
                        let _ = job.tx.blocking_send(TokenEvent::Error(message));
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
                        path,
                        reference,
                        name,
                        ctx_len,
                        endpoints,
                        tensor_split,
                        use_local_device,
                        resp,
                    }) => {
                        break Pending::Dist(DistLoadReq {
                            path,
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
        // distributed: register bridged RPC servers + split load).
        let (model, info, reference, resp) = match req {
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

                let model = match Model::load(&resolved.path, &ModelParams::default()) {
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
                (model, info, reference, resp)
            }
            Pending::Dist(req) => {
                let DistLoadReq {
                    path,
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
                let model = match Model::load_distributed(
                    &path,
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
                let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let info = LoadedModel {
                    name,
                    size_bytes,
                    n_layer: model.n_layer(),
                    n_ctx: ctx_len,
                };
                (model, info, reference, resp)
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
                Ok(HostMsg::Generate(job)) => run_generation(&mut session, &info, &reference, job),
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
                    path,
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
                        path,
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

/// Resolve a reference to a local file, driving any download on `rt` and
/// forwarding progress. Error strings are user-facing (with remedies).
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
                path,
                size_bytes: meta.len(),
            })
        }
        Resolved::Remote(spec) => {
            let dest_dir = cache_root.join(&spec.cache_key);
            let mut throttle = ProgressThrottle::default();
            let path = rt
                .block_on(download::download(&spec, &dest_dir, |completed, total| {
                    if throttle.should_emit(completed, total) {
                        let _ = progress.send(LoadProgress::Downloading { completed, total });
                    }
                }))
                .map_err(|e| e.to_string())?;
            let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            Ok(ResolvedLocal {
                name: spec.cache_key,
                path,
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

/// One queued generation, executed to completion on the host thread.
fn run_generation(
    session: &mut Session<'_>,
    info: &LoadedModel,
    loaded_reference: &str,
    job: GenerateJob,
) {
    // Accept the canonical loaded name and the reference the load was
    // requested with (`hf:…` refs cache under a sanitized key; clients may
    // use either spelling).
    if job.model != info.name && job.model != loaded_reference {
        let _ = job.tx.blocking_send(TokenEvent::Error(
            ApiError::ModelNotLoaded(job.model.clone()).to_string(),
        ));
        return;
    }

    let prompt_text = match render_prompt(session.model(), info, &job.prompt) {
        Ok(text) => text,
        Err(message) => {
            let _ = job.tx.blocking_send(TokenEvent::Error(message));
            return;
        }
    };
    let prompt_tokens = match session.model().tokenize(&prompt_text, true) {
        Ok(toks) => toks,
        Err(e) => {
            let _ = job.tx.blocking_send(TokenEvent::Error(e.to_string()));
            return;
        }
    };
    let max_tokens = job.params.max_tokens as usize;
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
        let _ = job.tx.blocking_send(TokenEvent::Error(message));
        return;
    }

    session.reset();
    session.set_sampler(&SamplerParams {
        temperature: job.params.temperature,
        top_p: job.params.top_p,
        top_k: job.params.top_k,
        seed: job.params.seed.unwrap_or(0xFFFF_FFFF),
    });

    let mut scan = StopScan::new(job.params.stop.clone());
    let mut stop_matched = false;
    let result = session.generate(&prompt_tokens, max_tokens, |_tok, piece| {
        if !scan.admit(piece) {
            // The piece completes a stop-string match: hold it back and end
            // the generation (contract: finish Stop without sending it).
            stop_matched = true;
            return ControlFlow::Break(());
        }
        match job.tx.blocking_send(TokenEvent::Token(piece.to_string())) {
            Ok(()) => ControlFlow::Continue(()),
            Err(_) => ControlFlow::Break(()), // client went away
        }
    });
    match result {
        Err(e) => {
            let _ = job.tx.blocking_send(TokenEvent::Error(e.to_string()));
        }
        Ok(stats) => {
            let finish = match stats.finished {
                FinishReason::Stop => FinishKind::Stop,
                FinishReason::Length => FinishKind::Length,
                FinishReason::Aborted if stop_matched => FinishKind::Stop,
                FinishReason::Aborted => FinishKind::Abort,
            };
            let _ = job.tx.blocking_send(TokenEvent::Done(DoneStats {
                prompt_tokens: stats.prompt_tokens as u32,
                completion_tokens: stats.generated_tokens as u32,
                finish,
            }));
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
/// engine-host thread and the model cache.
pub struct DaemonBackend {
    host: EngineHost,
    cache_root: PathBuf,
}

impl DaemonBackend {
    pub fn new(host: EngineHost, cache_root: PathBuf) -> DaemonBackend {
        DaemonBackend { host, cache_root }
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
        self.host.send(HostMsg::Generate(job))
    }

    fn pull(&self, model: String, tx: mpsc::Sender<PullEvent>) -> Result<(), ApiError> {
        let model_ref: ModelRef = model
            .parse()
            .map_err(|e| ApiError::BadRequest(format!("{e}")))?;
        let resolved = model_ref
            .resolve()
            .map_err(|e| ApiError::BadRequest(format!("{e}")))?;
        let cache_root = self.cache_root.clone();
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
                    let dest_dir = cache_root.join(&spec.cache_key);
                    let mut throttle = ProgressThrottle::default();
                    let progress_tx = tx.clone();
                    let result = download::download(&spec, &dest_dir, move |completed, total| {
                        if throttle.should_emit(completed, total) {
                            // try_send: never block the downloader on a slow
                            // client; skipped progress lines are harmless.
                            let _ =
                                progress_tx.try_send(PullEvent::Downloading { completed, total });
                        }
                    })
                    .await;
                    match result {
                        Ok(_) => PullEvent::Done,
                        Err(e) => PullEvent::Error {
                            message: e.to_string(),
                        },
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
        let (host, handle) = EngineHost::spawn();
        let (tx, mut rx) = mpsc::channel(8);
        host.send(HostMsg::Generate(GenerateJob {
            model: "anything".into(),
            prompt: PromptInput::Raw("hi".into()),
            params: Default::default(),
            tx,
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
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn models_before_load_is_none() {
        let (host, handle) = EngineHost::spawn();
        assert_eq!(host.loaded_model(Duration::from_secs(5)), None);
        host.send(HostMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn load_with_unknown_reference_fails_with_remedy() {
        let dir = tempfile::tempdir().unwrap();
        let (host, handle) = EngineHost::spawn();
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
