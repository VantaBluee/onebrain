//! The internal control API (`/api/internal/*`, docs/internal-api.md).
//!
//! These endpoints are ALWAYS token-authenticated: the public gateway's
//! localhost exemption deliberately does not apply here, so a random local
//! process cannot stop the daemon or swap models without reading the token
//! file (same-user filesystem access).

use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use onebrain_api::auth::AuthConfig;
use onebrain_api::ApiError;
use onebrain_mesh::{MeshError, MeshHandle, PairEvent, PairTarget, PeerState, PeerStatus};
use onebrain_models::registry::{ModelRef, Resolved};
use onebrain_proto::message::{Envelope, Message};
use onebrain_proto::plan::{Assignment, NodeId, Plan, Strategy};
use onebrain_scheduler::{ComputeProfile, LinkRtt, NodeCaps, PlanRequest, StoredProfile};
use tokio::sync::{mpsc, oneshot, Notify};
use tokio_stream::wrappers::ReceiverStream;

use crate::cluster::{ActivePlanView, ClusterState};
use crate::engine_host::{EngineHost, HostMsg, LoadProgress, ProgressThrottle};

/// Registry id of the microbench test model (docs/scheduler-v1.md
/// "Profiles"): pulled through the normal registry path on first bench —
/// never bundled in the binary.
pub const BENCH_MODEL_ID: &str = "tinystories-260k";

/// The daemon's in-memory view of the persisted device profile
/// (`<config_dir>/profile.toml`): loaded at startup, refreshed by
/// `POST /api/internal/bench`, read by the `NodeStatus` provider and the
/// planner.
pub type SharedProfile = Arc<StdMutex<Option<StoredProfile>>>;

/// Shared state for the internal router.
pub struct InternalState {
    pub host: EngineHost,
    /// Token check only; `localhost_exempt` is forced off for these routes.
    pub auth: AuthConfig,
    pub cache_root: PathBuf,
    pub ctx_len: u32,
    pub port: u16,
    pub started: Instant,
    pub product_version: &'static str,
    /// Notified by `POST /api/internal/shutdown`; the runtime's graceful-
    /// shutdown future awaits it.
    pub shutdown: Arc<Notify>,
    /// Handle to the mesh service (pairing, peers, link state).
    pub mesh: MeshHandle,
    /// Cluster-session state: epoch counter, active plan, plan acks.
    pub cluster: Arc<ClusterState>,
    /// Test-only `[debug] usable_memory_override_bytes` (docs/distributed.md).
    pub usable_memory_override: Option<u64>,
    /// Test-only `[debug] decode_tps_override` (docs/scheduler-v1.md): when
    /// set it replaces the measured decode throughput in `NodeStatus` and in
    /// local placement scoring.
    pub decode_tps_override: Option<f64>,
    /// The persisted device profile, shared with the `NodeStatus` provider.
    pub profile: SharedProfile,
    /// Where `POST /api/internal/bench` persists the profile
    /// (`<config_dir>/profile.toml`).
    pub profile_path: PathBuf,
}

/// Build the internal router with its always-on token middleware.
pub fn internal_router(state: Arc<InternalState>) -> Router {
    Router::new()
        .route("/api/internal/status", get(status))
        .route("/api/internal/load", post(load))
        .route("/api/internal/bench", post(bench))
        .route("/api/internal/shutdown", post(shutdown))
        .route("/api/internal/pair/start", post(pair_start))
        .route("/api/internal/pair/join", post(pair_join))
        .route("/api/internal/peers", get(peers))
        .route("/api/internal/unpair", post(unpair))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_internal_token,
        ))
        .with_state(state)
}

/// Bearer-token middleware for the internal endpoints. Unlike the public
/// gateway's `require_bearer` there is no loopback exemption — the contract
/// pins these routes to the token unconditionally.
async fn require_internal_token(
    State(state): State<Arc<InternalState>>,
    request: Request,
    next: Next,
) -> Response {
    let presented = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match presented {
        Some(token) if state.auth.token_matches(token) => next.run(request).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": {
                    "message": "internal endpoints always require the bearer token \
                                (no localhost exemption); `onebrain status` prints it",
                    "type": "authentication_error"
                }
            })),
        )
            .into_response(),
    }
}

/// `GET /api/internal/status`.
async fn status(State(state): State<Arc<InternalState>>) -> Json<serde_json::Value> {
    let host = state.host.clone();
    // The host answers instantly unless it is mid-generation; bounce the
    // wait off the blocking pool so a busy host cannot stall the runtime.
    let model = tokio::task::spawn_blocking(move || host.loaded_model(Duration::from_millis(250)))
        .await
        .unwrap_or(None);
    // peers_summary (docs/mesh.md): paired = store size, connected = live
    // sessions with fresh heartbeats. A stopped mesh service reports zeros
    // rather than failing the whole status call.
    let peer_list = state.mesh.peers().await.unwrap_or_else(|err| {
        tracing::warn!(error = %err, "status could not read peers; reporting zeros");
        Vec::new()
    });
    let connected = peer_list
        .iter()
        .filter(|p| p.state == PeerState::Connected)
        .count();
    Json(serde_json::json!({
        "version": state.product_version,
        "engine_build": onebrain_engine::engine_build_hash().0,
        "port": state.port,
        "uptime_secs": state.started.elapsed().as_secs(),
        "model": model,
        "peers_summary": { "paired": peer_list.len(), "connected": connected },
        // docs/distributed.md: the active plan (epoch, strategy,
        // assignments) or null when nothing distributed is active.
        "plan": state.cluster.active(),
    }))
}

#[derive(Debug, serde::Deserialize)]
struct LoadBody {
    model: String,
    /// `--nodes N`: force solo (1) or distribution across exactly N nodes.
    #[serde(default)]
    nodes: Option<u32>,
    /// `--explain`: include the scheduler's prose in the `plan` line.
    #[serde(default)]
    explain: bool,
}

/// One NDJSON line sender.
type LineSender = mpsc::Sender<Result<String, std::convert::Infallible>>;

async fn emit(tx: &LineSender, line: serde_json::Value) -> bool {
    tx.send(Ok(format!("{line}\n"))).await.is_ok()
}

async fn emit_error(tx: &LineSender, message: String) {
    let _ = emit(
        tx,
        serde_json::json!({ "status": "error", "message": message }),
    )
    .await;
}

fn ndjson_response(line_rx: mpsc::Receiver<Result<String, std::convert::Infallible>>) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(axum::body::Body::from_stream(ReceiverStream::new(line_rx)))
        .expect("static response construction cannot fail")
}

/// `POST /api/internal/load` — NDJSON stream (contract + docs/distributed.md
/// "Daemon & API"): `downloading…` lines while the head fetches the full
/// GGUF, then `planning`, `plan` (with `explanation` when `explain` is set),
/// then `loading` and the terminal `ready`/`error`. A Solo plan runs the
/// unchanged single-node path; a PipelineParallel plan drives the epoch
/// lifecycle (proposals → acks → rpc streams → distributed engine load).
async fn load(State(state): State<Arc<InternalState>>, Json(body): Json<LoadBody>) -> Response {
    let (line_tx, line_rx) = mpsc::channel::<Result<String, std::convert::Infallible>>(32);
    tokio::spawn(drive_load(state, body, line_tx));
    ndjson_response(line_rx)
}

fn bench_error(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": { "message": message, "type": "bench_error" }
        })),
    )
        .into_response()
}

/// `POST /api/internal/bench` (docs/scheduler-v1.md "onebrain bench"):
/// run the full local profile — compute microbench on the registry test
/// model (pulled through the normal registry path if absent) plus the disk
/// sequential-read probe — persist it to `profile.toml`, refresh every
/// connected peer's link probe, and answer with the profile and the link
/// table. Peers learn the fresh profile immediately via a best-effort
/// `NodeStatus` push (and again on every future session).
async fn bench(State(state): State<Arc<InternalState>>) -> Response {
    // 1. Ensure the test model is cached — exactly a normal registry pull.
    let spec = match BENCH_MODEL_ID
        .parse::<ModelRef>()
        .map_err(|e| e.to_string())
        .and_then(|r| r.resolve().map_err(|e| e.to_string()))
    {
        Ok(Resolved::Remote(spec)) => spec,
        Ok(Resolved::Local(_)) => {
            return bench_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "registry id {BENCH_MODEL_ID:?} resolved to a local path; this is a \
                     OneBrain packaging bug — please report it"
                ),
            );
        }
        Err(message) => {
            return bench_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "the embedded registry cannot resolve the test model \
                     {BENCH_MODEL_ID:?}: {message}"
                ),
            );
        }
    };
    let dest_dir = state.cache_root.join(&spec.cache_key);
    let model_path = match onebrain_models::download::download(&spec, &dest_dir, |_, _| {}).await {
        Ok(path) => path,
        Err(e) => return bench_error(StatusCode::BAD_GATEWAY, e.to_string()),
    };

    // 2. The compute microbench and the disk probe (blocking, ~seconds).
    let probe_path = model_path.clone();
    let measured = tokio::task::spawn_blocking(move || {
        let compute = onebrain_scheduler::measure_compute(&probe_path)?;
        let disk_mbps = onebrain_scheduler::measure_disk(&probe_path)?;
        Ok::<_, onebrain_scheduler::ProfileError>((compute, disk_mbps))
    })
    .await;
    let (compute, disk_mbps) = match measured {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => return bench_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(_) => {
            return bench_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "the microbench task failed unexpectedly; retry `onebrain bench`".to_string(),
            );
        }
    };
    let stored = StoredProfile {
        measured_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        prefill_tps: compute.prefill_tps,
        decode_tps: compute.decode_tps,
        disk_mbps,
    };

    // 3. Persist, then publish to the state the NodeStatus provider and the
    // planner read. The measured values are stored/returned even under the
    // decode_tps_override — the override applies at reporting/scoring time.
    if let Err(e) = onebrain_scheduler::save_profile(&state.profile_path, &stored) {
        return bench_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    *state.profile.lock().expect("profile state poisoned") = Some(stored);
    tracing::info!(
        prefill_tps = stored.prefill_tps,
        decode_tps = stored.decode_tps,
        disk_mbps = stored.disk_mbps,
        "device profile refreshed by bench"
    );

    // 4. Refresh every connected peer's link probe, then read the table
    // back. A failed probe keeps the peer's last-known figures.
    let peers = state.mesh.peers().await.unwrap_or_default();
    for peer in peers.iter().filter(|p| p.state == PeerState::Connected) {
        if let Err(err) = state.mesh.probe(&peer.name).await {
            tracing::warn!(
                peer = %peer.name,
                error = %err,
                "bench link probe failed; reporting last-known figures"
            );
        }
    }
    let refreshed = state.mesh.peers().await.unwrap_or(peers);
    let connected: Vec<&PeerStatus> = refreshed
        .iter()
        .filter(|p| p.state == PeerState::Connected)
        .collect();
    let links: Vec<serde_json::Value> = connected
        .iter()
        .map(|p| {
            serde_json::json!({
                "peer": p.name,
                "id": p.id,
                "rtt_ms": p.rtt_ms,
                "bandwidth_mbps": p.bandwidth_mbps,
                "loss": p.loss,
            })
        })
        .collect();

    // 5. Best-effort push of the refreshed NodeStatus so peers do not wait
    // for the next session to learn the new profile; the measured usable
    // memory also joins the report's profile object (the bench report's
    // NODE table shows memory alongside the throughputs).
    let override_bytes = state.usable_memory_override;
    let mut profile_json = serde_json::to_value(stored).unwrap_or_else(|_| serde_json::json!({}));
    if let Ok((usable_memory_bytes, devices)) =
        tokio::task::spawn_blocking(move || crate::cluster::local_node_status(override_bytes)).await
    {
        profile_json["usable_memory_bytes"] = serde_json::json!(usable_memory_bytes);
        let envelope = Envelope::new(Message::NodeStatus {
            usable_memory_bytes,
            devices,
            prefill_tps: Some(stored.prefill_tps),
            // The test-only override wins wherever decode is reported.
            decode_tps: Some(state.decode_tps_override.unwrap_or(stored.decode_tps)),
            disk_mbps: Some(stored.disk_mbps),
        });
        for peer in &connected {
            if let Err(err) = state.mesh.send_control(&peer.id, envelope.clone()).await {
                tracing::debug!(peer = %peer.name, "NodeStatus push after bench failed: {err}");
            }
        }
    }

    Json(serde_json::json!({ "profile": profile_json, "links": links })).into_response()
}

/// A model reference resolved to a local file (downloaded if needed).
struct LocalModel {
    path: PathBuf,
    name: String,
    size_bytes: u64,
}

/// Resolve + download the reference, emitting `downloading` lines. `None`
/// means an error line was already emitted.
async fn ensure_local(
    state: &Arc<InternalState>,
    reference: &str,
    tx: &LineSender,
) -> Option<LocalModel> {
    let model_ref: ModelRef = match reference.parse() {
        Ok(r) => r,
        Err(e) => {
            emit_error(tx, format!("{e}")).await;
            return None;
        }
    };
    let resolved = match model_ref.resolve() {
        Ok(r) => r,
        Err(e) => {
            emit_error(tx, format!("{e}")).await;
            return None;
        }
    };
    match resolved {
        Resolved::Local(path) => {
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(e) => {
                    emit_error(
                        tx,
                        format!(
                            "cannot read local model {}: {e}; check the path exists and is \
                             readable",
                            path.display()
                        ),
                    )
                    .await;
                    return None;
                }
            };
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".to_string());
            Some(LocalModel {
                name: format!("local:{stem}"),
                size_bytes: meta.len(),
                path,
            })
        }
        Resolved::Remote(spec) => {
            let dest_dir = state.cache_root.join(&spec.cache_key);
            let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<(u64, u64)>();
            let mut throttle = ProgressThrottle::default();
            let download =
                onebrain_models::download::download(&spec, &dest_dir, move |completed, total| {
                    if throttle.should_emit(completed, total) {
                        let _ = progress_tx.send((completed, total));
                    }
                });
            tokio::pin!(download);
            loop {
                tokio::select! {
                    result = &mut download => match result {
                        Ok(path) => {
                            let size_bytes =
                                std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                            return Some(LocalModel {
                                name: spec.cache_key.clone(),
                                path,
                                size_bytes,
                            });
                        }
                        Err(e) => {
                            emit_error(tx, e.to_string()).await;
                            return None;
                        }
                    },
                    Some((completed, total)) = progress_rx.recv() => {
                        // A gone client does not cancel the download: model
                        // presence is daemon-level state (M1 contract).
                        let _ = emit(tx, serde_json::json!({
                            "status": "downloading", "completed": completed, "total": total,
                        })).await;
                    }
                }
            }
        }
    }
}

async fn drive_load(state: Arc<InternalState>, body: LoadBody, tx: LineSender) {
    // Phase A: the head holds the full GGUF (ADR 0004 head-push weight
    // flow) — make it local before planning needs its header.
    let Some(local) = ensure_local(&state, &body.model, &tx).await else {
        return;
    };

    // Phase B: plan (M4 scheduler v1: real model dims, device profiles, and
    // measured link RTTs — docs/scheduler-v1.md).
    let _ = emit(&tx, serde_json::json!({ "status": "planning" })).await;
    let override_bytes = state.usable_memory_override;
    let head_status =
        tokio::task::spawn_blocking(move || crate::cluster::local_node_status(override_bytes))
            .await;
    let Ok((head_usable, _devices)) = head_status else {
        emit_error(
            &tx,
            "measuring local memory failed; retry the load".to_string(),
        )
        .await;
        return;
    };
    // Workers eligible for this plan: connected peers that reported a
    // budget via NodeStatus, in the mesh's (name-sorted, deterministic)
    // order. A peer can be Connected before its NodeStatus lands (heartbeat
    // and control streams race — observed on slow CI runners when a load
    // follows pairing within milliseconds), so wait briefly for budgets of
    // connected peers instead of silently planning without them.
    let budget_deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    let peers: Vec<PeerStatus> = loop {
        let all = match state.mesh.peers().await {
            Ok(p) => p,
            Err(e) => {
                emit_error(&tx, e.to_string()).await;
                return;
            }
        };
        let connected: Vec<PeerStatus> = all
            .into_iter()
            .filter(|p| p.state == PeerState::Connected)
            .collect();
        let missing = connected.iter().any(|p| p.usable_memory_bytes.is_none());
        if !missing || std::time::Instant::now() >= budget_deadline {
            if missing {
                tracing::warn!(
                    "planning without budgets from some connected peers \
                     (NodeStatus never arrived); they are excluded from this plan"
                );
            }
            break connected;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    };
    // Model dims (per-layer weight bytes + KV growth rate) from the real
    // GGUF header — replaces the M3 layer-count-only read.
    let header_path = local.path.clone();
    let header =
        match tokio::task::spawn_blocking(move || crate::cluster::read_gguf_header(&header_path))
            .await
        {
            Ok(Ok(header)) => header,
            Ok(Err(message)) => {
                emit_error(&tx, message).await;
                return;
            }
            Err(_) => {
                emit_error(
                    &tx,
                    "reading the model header failed; retry the load".to_string(),
                )
                .await;
                return;
            }
        };
    let dims = match onebrain_scheduler::model_dims(&header, local.size_bytes) {
        Ok(dims) => dims,
        Err(e) => {
            emit_error(&tx, e.to_string()).await;
            return;
        }
    };
    let own_id = state.mesh.endpoint_id().to_string();
    // Head compute profile: the persisted local profile, with the test-only
    // `[debug] decode_tps_override` winning (docs/scheduler-v1.md DoD
    // hooks). With an override but no stored profile, prefill is reported
    // as 0.0 — v1 placement scoring only consumes decode_tps.
    let stored = *state.profile.lock().expect("profile state poisoned");
    let head_compute = match (state.decode_tps_override, stored) {
        (Some(decode), stored) => Some(ComputeProfile {
            prefill_tps: stored.map(|p| p.prefill_tps).unwrap_or(0.0),
            decode_tps: decode,
        }),
        (None, Some(p)) => Some(ComputeProfile {
            prefill_tps: p.prefill_tps,
            decode_tps: p.decode_tps,
        }),
        (None, None) => None,
    };
    let mut workers = Vec::with_capacity(peers.len());
    let mut links = Vec::new();
    for p in &peers {
        let Some(bytes) = p.usable_memory_bytes else {
            continue;
        };
        workers.push(NodeCaps {
            node: NodeId(p.id.clone()),
            usable_memory_bytes: bytes,
            // From the peer's NodeStatus; None (or a nonsense zero) means
            // the peer has not benched — memory-only weighting for it.
            compute: p
                .decode_tps
                .filter(|d| *d > 0.0)
                .map(|decode| ComputeProfile {
                    prefill_tps: p.prefill_tps.unwrap_or(0.0),
                    decode_tps: decode,
                }),
        });
        // Head <-> worker RTT from the heartbeat EWMA. Worker <-> worker
        // links are not measured in M4 and are omitted — the scheduler
        // defaults missing pairs to DEFAULT_LINK_RTT_MS.
        if let Some(rtt_ms) = p.rtt_ms {
            links.push(LinkRtt {
                a: NodeId(own_id.clone()),
                b: NodeId(p.id.clone()),
                rtt_ms,
            });
        }
    }
    let request = PlanRequest {
        head: NodeCaps {
            node: NodeId(own_id.clone()),
            usable_memory_bytes: head_usable,
            compute: head_compute,
        },
        workers,
        dims,
        ctx_len: state.ctx_len,
        forced_nodes: body.nodes,
        links,
    };
    let placed = match onebrain_scheduler::plan_v1(&request) {
        Ok(p) => p,
        Err(e) => {
            emit_error(&tx, e.to_string()).await;
            return;
        }
    };
    let mut plan = placed.plan;
    plan.model = local.name.clone();
    let solo = plan.strategy == Strategy::Solo;
    if !solo {
        plan.epoch = state.cluster.next_epoch();
    }
    let predicted = (!solo).then(|| predicted_tpt_ms(&plan.assignments, &request));
    let mut plan_json = serde_json::to_value(&plan).unwrap_or_else(|_| serde_json::json!({}));
    if let Some(tpt) = predicted {
        plan_json["predicted_tpt_ms"] = serde_json::json!(tpt);
    }
    if body.explain {
        plan_json["explanation"] = serde_json::json!(placed.explanation);
    }
    if !emit(
        &tx,
        serde_json::json!({ "status": "plan", "plan": plan_json }),
    )
    .await
    {
        // Client gone before any side effect: abandon quietly.
        return;
    }

    if solo {
        solo_load(&state, &body.model, plan, placed.explanation, &tx).await;
    } else {
        distributed_load(
            &state,
            body.model,
            local,
            plan,
            placed.tensor_split,
            placed.explanation,
            predicted,
            own_id,
            &tx,
        )
        .await;
    }
}

/// Mirror of the scheduler's plan-comparison metric for the FINAL plan
/// (v1 module docs): `max_stage(layers / decode_tps) × 1000 + Σ boundary
/// RTT ms`, where an unprofiled node is costed at the slowest profiled
/// node's rate (conservative) or contributes zero when nothing is profiled,
/// and unmeasured links default to
/// [`onebrain_scheduler::DEFAULT_LINK_RTT_MS`]. A RELATIVE figure (the
/// decode rates come from the tiny-model microbench), surfaced as
/// `predicted_tpt_ms` on the plan view — never as a latency promise (§1.6).
fn predicted_tpt_ms(assignments: &[Assignment], request: &PlanRequest) -> f64 {
    let nodes = || std::iter::once(&request.head).chain(request.workers.iter());
    let min_decode = nodes()
        .filter_map(|c| c.compute.map(|p| p.decode_tps))
        .filter(|d| *d > 0.0)
        .fold(f64::NAN, f64::min);
    let decode_of = |id: &NodeId| -> Option<f64> {
        nodes()
            .find(|c| &c.node == id)
            .and_then(|c| c.compute.map(|p| p.decode_tps))
            .filter(|d| *d > 0.0)
            .or(if min_decode.is_finite() {
                Some(min_decode)
            } else {
                None
            })
    };
    let compute_ms = assignments
        .iter()
        .map(|a| match decode_of(&a.node) {
            Some(tps) => a.layers.len() as f64 / tps * 1000.0,
            None => 0.0,
        })
        .fold(0.0f64, f64::max);
    let boundary_ms: f64 = assignments
        .windows(2)
        .map(|pair| {
            request
                .links
                .iter()
                .find(|l| {
                    (l.a == pair[0].node && l.b == pair[1].node)
                        || (l.a == pair[1].node && l.b == pair[0].node)
                })
                .map(|l| l.rtt_ms)
                .unwrap_or(onebrain_scheduler::DEFAULT_LINK_RTT_MS)
        })
        .sum();
    compute_ms + boundary_ms
}

/// The unchanged single-node path: the engine host resolves the (already
/// cached) reference, loads, and answers; progress is forwarded as NDJSON.
async fn solo_load(
    state: &Arc<InternalState>,
    reference: &str,
    plan: Plan,
    explanation: String,
    tx: &LineSender,
) {
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<LoadProgress>();
    let (resp_tx, resp_rx) = oneshot::channel();
    if state
        .host
        .send(HostMsg::Load {
            reference: reference.to_string(),
            cache_root: state.cache_root.clone(),
            ctx_len: state.ctx_len,
            progress: progress_tx,
            resp: resp_tx,
        })
        .is_err()
    {
        emit_error(tx, ApiError::ShuttingDown.to_string()).await;
        return;
    }
    while let Some(event) = progress_rx.recv().await {
        let line = match event {
            LoadProgress::Downloading { completed, total } => serde_json::json!({
                "status": "downloading", "completed": completed, "total": total,
            }),
            LoadProgress::Loading => serde_json::json!({ "status": "loading" }),
        };
        // A gone client does not cancel the load: model presence is
        // daemon-level state, not tied to this request (M1 contract).
        let _ = emit(tx, line).await;
    }
    match resp_rx.await {
        Ok(Ok(model)) => {
            state.cluster.set_active(Some(ActivePlanView {
                role: "head",
                plan,
                explanation: Some(explanation),
                predicted_tpt_ms: None,
            }));
            // Bridges of any replaced distributed plan are stale now — the
            // host dropped that model during the swap (ADR 0004 ordering:
            // model freed first, bridges closed after).
            state.cluster.teardown_head_bridges();
            let _ = emit(tx, serde_json::json!({ "status": "ready", "model": model })).await;
        }
        Ok(Err(message)) => emit_error(tx, message).await,
        Err(_) => {
            emit_error(
                tx,
                "the engine host exited unexpectedly; restart with `onebrain up`".to_string(),
            )
            .await
        }
    }
}

/// The distributed path (docs/distributed.md "Epoch lifecycle"): propose the
/// plan, collect acks (15 s, typed error naming the node), open one `rpc`
/// stream + accept-once loopback bridge per worker, then load through the
/// engine host with the plan's tensor split.
#[allow(clippy::too_many_arguments)]
async fn distributed_load(
    state: &Arc<InternalState>,
    reference: String,
    local: LocalModel,
    plan: Plan,
    tensor_split: Vec<f32>,
    explanation: String,
    predicted_tpt_ms: Option<f64>,
    own_id: String,
    tx: &LineSender,
) {
    let epoch = plan.epoch;
    let peers = state.mesh.peers().await.unwrap_or_default();
    let label_of = |id: &str| {
        peers
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| id.chars().take(8).collect())
    };
    // Worker assignments in stage order; the device order handed to the
    // engine must match the plan's tensor_split order (workers…, head).
    let workers: Vec<(String, String)> = plan
        .assignments
        .iter()
        .filter(|a| a.node.0 != own_id)
        .map(|a| (a.node.0.clone(), label_of(&a.node.0)))
        .collect();
    let use_local_device = plan.assignments.iter().any(|a| a.node.0 == own_id);

    for (id, label) in &workers {
        let envelope = Envelope::new(Message::PlanProposal(plan.clone()));
        if let Err(err) = state.mesh.send_control(id, envelope).await {
            emit_error(
                tx,
                format!("could not send the plan to node '{label}': {err}"),
            )
            .await;
            return;
        }
    }
    if let Err(message) = state
        .cluster
        .await_acks(epoch.0, &workers, Duration::from_secs(15))
        .await
    {
        emit_error(tx, message).await;
        return;
    }

    let mut endpoints = Vec::with_capacity(workers.len());
    let mut bridges = Vec::with_capacity(workers.len());
    for (id, label) in &workers {
        // The bridge opens one fresh mesh rpc stream per accepted loopback
        // connection (the RPC client dials the endpoint repeatedly), so it
        // owns the mesh handle rather than a single pre-opened stream.
        match crate::cluster::head_bridge(state.mesh.clone(), id.clone(), epoch).await {
            Ok((endpoint, task)) => {
                endpoints.push(endpoint);
                bridges.push(task);
            }
            Err(message) => {
                abort_all(bridges);
                emit_error(
                    tx,
                    format!("could not set up the rpc bridge for node '{label}': {message}"),
                )
                .await;
                return;
            }
        }
    }

    let _ = emit(tx, serde_json::json!({ "status": "loading" })).await;
    let (resp_tx, resp_rx) = oneshot::channel();
    if state
        .host
        .send(HostMsg::LoadDistributed {
            path: local.path.clone(),
            reference,
            name: local.name.clone(),
            ctx_len: state.ctx_len,
            endpoints,
            tensor_split,
            use_local_device,
            resp: resp_tx,
        })
        .is_err()
    {
        abort_all(bridges);
        emit_error(tx, ApiError::ShuttingDown.to_string()).await;
        return;
    }
    match resp_rx.await {
        Ok(Ok(model)) => {
            // Success: the new epoch's bridges replace (and close) any
            // previous epoch's — its model was already dropped by the host
            // during the swap (ADR 0004 ordering).
            state.cluster.replace_head_bridges(bridges);
            state.cluster.set_active(Some(ActivePlanView {
                role: "head",
                plan,
                explanation: Some(explanation),
                predicted_tpt_ms,
            }));
            let _ = emit(tx, serde_json::json!({ "status": "ready", "model": model })).await;
        }
        Ok(Err(message)) => {
            abort_all(bridges);
            emit_error(tx, message).await;
        }
        Err(_) => {
            abort_all(bridges);
            emit_error(
                tx,
                "the engine host exited unexpectedly; restart with `onebrain up`".to_string(),
            )
            .await;
        }
    }
}

fn abort_all(tasks: Vec<tokio::task::JoinHandle<()>>) {
    for task in tasks {
        task.abort();
    }
}

/// `POST /api/internal/shutdown` — acknowledge, then let the runtime's
/// graceful shutdown drain this very response.
async fn shutdown(State(state): State<Arc<InternalState>>) -> Json<serde_json::Value> {
    tracing::info!("shutdown requested via /api/internal/shutdown");
    state.shutdown.notify_one();
    Json(serde_json::json!({ "status": "stopping" }))
}

/// Map a [`MeshError`] to the internal API's error envelope. The message is
/// the error's own Display (every variant carries its remedy).
fn mesh_error_response(err: MeshError) -> Response {
    let status = match &err {
        MeshError::BadPairTarget { .. }
        | MeshError::CodeRequired
        | MeshError::PairRejected { .. }
        | MeshError::NoCandidates
        | MeshError::NotConnected { .. } => StatusCode::BAD_REQUEST,
        MeshError::UnknownPeerName { .. } => StatusCode::NOT_FOUND,
        MeshError::WindowAlreadyOpen => StatusCode::CONFLICT,
        MeshError::Connect { .. } | MeshError::Timeout { .. } => StatusCode::BAD_GATEWAY,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "mesh_error"
    };
    (
        status,
        Json(serde_json::json!({
            "error": { "message": err.to_string(), "type": kind }
        })),
    )
        .into_response()
}

/// `POST /api/internal/pair/start` — open a pairing window and stream its
/// events as NDJSON: a `window` line (code + ticket) first, then
/// `attempt` / terminal `paired` / `expired` / `failed` (docs/mesh.md).
async fn pair_start(State(state): State<Arc<InternalState>>) -> Response {
    let window = match state.mesh.pair_start().await {
        Ok(window) => window,
        Err(err) => return mesh_error_response(err),
    };

    let (line_tx, line_rx) = mpsc::channel::<Result<String, std::convert::Infallible>>(32);
    tokio::spawn(async move {
        let first = serde_json::json!({
            "status": "window", "code": window.code, "ticket": window.ticket,
        });
        if line_tx.send(Ok(format!("{first}\n"))).await.is_err() {
            return;
        }
        let mut events = window.events;
        // The mesh closes the channel after a terminal event; drain to end.
        while let Some(event) = events.recv().await {
            let line = match event {
                PairEvent::Attempt => serde_json::json!({ "status": "attempt" }),
                PairEvent::Paired(peer) => {
                    serde_json::json!({ "status": "paired", "peer": peer })
                }
                PairEvent::Expired => serde_json::json!({ "status": "expired" }),
                PairEvent::Failed(message) => {
                    serde_json::json!({ "status": "failed", "message": message })
                }
            };
            if line_tx.send(Ok(format!("{line}\n"))).await.is_err() {
                // Client went away; the window stays open server-side until
                // it expires — closing it here would break "read the code
                // out loud, then close the terminal" flows.
                return;
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(axum::body::Body::from_stream(ReceiverStream::new(line_rx)))
        .expect("static response construction cannot fail")
}

#[derive(Debug, serde::Deserialize)]
struct PairJoinBody {
    target: String,
    #[serde(default)]
    code: Option<String>,
}

/// `POST /api/internal/pair/join` — join a window hosted elsewhere. The
/// target is a ticket unless it looks like a bare 6-digit code (mDNS
/// candidate discovery); the mesh validates either form.
async fn pair_join(
    State(state): State<Arc<InternalState>>,
    Json(body): Json<PairJoinBody>,
) -> Response {
    let trimmed = body.target.trim();
    let target = if trimmed.len() == 6 && trimmed.bytes().all(|b| b.is_ascii_digit()) {
        PairTarget::Code(trimmed.to_string())
    } else {
        PairTarget::Ticket(trimmed.to_string())
    };
    match state.mesh.pair_join(target, body.code).await {
        Ok(peer) => Json(serde_json::json!({ "peer": peer })).into_response(),
        Err(err) => mesh_error_response(err),
    }
}

/// `GET /api/internal/peers` — the peer store merged with live link state.
async fn peers(State(state): State<Arc<InternalState>>) -> Response {
    match state.mesh.peers().await {
        Ok(list) => Json(serde_json::json!({ "peers": list })).into_response(),
        Err(err) => mesh_error_response(err),
    }
}

#[derive(Debug, serde::Deserialize)]
struct UnpairBody {
    name: String,
}

/// `POST /api/internal/unpair` — remove a peer by name; unknown names get a
/// 404 listing the known ones (the mesh error carries them).
async fn unpair(State(state): State<Arc<InternalState>>, Json(body): Json<UnpairBody>) -> Response {
    match state.mesh.unpair(&body.name).await {
        Ok(()) => Json(serde_json::json!({ "status": "ok" })).into_response(),
        Err(err) => mesh_error_response(err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_host::EngineHost;

    /// Spawn a hermetic mesh service for router tests: loopback-only, no
    /// mDNS, no relays (same shape as the onebrain-mesh integration tests).
    async fn spawn_test_mesh(dir: &std::path::Path) -> MeshHandle {
        let key = onebrain_mesh::identity::load_or_create(dir).unwrap();
        onebrain_mesh::MeshService::spawn(
            key,
            dir.join("peers.toml"),
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
        .unwrap()
    }

    /// Serve an internal router on an ephemeral port; returns its base URL,
    /// the shutdown notify, and guards that keep the host thread alive.
    async fn serve_internal(
        token: &str,
    ) -> (
        String,
        Arc<Notify>,
        EngineHost,
        std::thread::JoinHandle<()>,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let (host, host_thread) = EngineHost::spawn();
        let shutdown = Arc::new(Notify::new());
        let mesh = spawn_test_mesh(dir.path()).await;
        let state = Arc::new(InternalState {
            host: host.clone(),
            auth: AuthConfig {
                token: token.to_string(),
                localhost_exempt: false,
            },
            cache_root: dir.path().join("models"),
            ctx_len: 4096,
            port: 0,
            started: Instant::now(),
            product_version: "test",
            shutdown: shutdown.clone(),
            mesh,
            cluster: ClusterState::new(),
            usable_memory_override: None,
            decode_tps_override: None,
            profile: Arc::new(StdMutex::new(None)),
            // Nested like a real <config_dir>: bench must create parents.
            profile_path: dir.path().join("config").join("profile.toml"),
        });
        let app = internal_router(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), shutdown, host, host_thread, dir)
    }

    fn teardown(host: EngineHost, host_thread: std::thread::JoinHandle<()>) {
        host.send(HostMsg::Shutdown).unwrap();
        host_thread.join().unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn internal_endpoints_reject_localhost_without_token() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let client = reqwest::Client::new();
        // Loopback exemption must NOT apply to internal endpoints.
        let resp = client
            .get(format!("{base}/api/internal/status"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let resp = client
            .get(format!("{base}/api/internal/status"))
            .bearer_auth("wrong-token")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        teardown(host, host_thread);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn status_reports_shape_with_no_model() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let resp = reqwest::Client::new()
            .get(format!("{base}/api/internal/status"))
            .bearer_auth("sekrit")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["version"], "test");
        assert!(body["engine_build"].as_str().unwrap().contains("llama.cpp"));
        assert!(body["uptime_secs"].is_u64());
        assert!(body["model"].is_null());
        // docs/mesh.md: status gains peers_summary; nothing is paired here.
        assert_eq!(body["peers_summary"]["paired"], 0);
        assert_eq!(body["peers_summary"]["connected"], 0);
        // docs/distributed.md: status reports the active plan (null here).
        assert!(body["plan"].is_null());
        teardown(host, host_thread);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn peers_reports_an_empty_list_when_nothing_is_paired() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let resp = reqwest::Client::new()
            .get(format!("{base}/api/internal/peers"))
            .bearer_auth("sekrit")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["peers"], serde_json::json!([]));
        teardown(host, host_thread);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pair_join_with_a_bogus_ticket_is_a_clean_400() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let resp = reqwest::Client::new()
            .post(format!("{base}/api/internal/pair/join"))
            .bearer_auth("sekrit")
            .json(&serde_json::json!({
                "target": "definitely-not-a-ticket", "code": "123456",
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let body: serde_json::Value = resp.json().await.unwrap();
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("definitely-not-a-ticket"), "{message}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        teardown(host, host_thread);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unpair_of_unknown_name_is_a_404_listing_known_names() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let resp = reqwest::Client::new()
            .post(format!("{base}/api/internal/unpair"))
            .bearer_auth("sekrit")
            .json(&serde_json::json!({ "name": "nonexistent-peer" }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("nonexistent-peer"));
        teardown(host, host_thread);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn load_of_unknown_model_streams_a_terminal_error_line() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let resp = reqwest::Client::new()
            .post(format!("{base}/api/internal/load"))
            .bearer_auth("sekrit")
            .json(&serde_json::json!({ "model": "definitely-not-a-model" }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.headers()[header::CONTENT_TYPE.as_str()],
            "application/x-ndjson"
        );
        let body = resp.text().await.unwrap();
        let last_line = body.lines().last().expect("stream must not be empty");
        let parsed: serde_json::Value = serde_json::from_str(last_line).unwrap();
        assert_eq!(parsed["status"], "error");
        assert!(parsed["message"]
            .as_str()
            .unwrap()
            .contains("definitely-not-a-model"));
        teardown(host, host_thread);
    }

    /// Bench endpoint shape over the hermetic mesh. Needs the engine, so it
    /// runs only when OB_SMOKE_MODEL points at a tiny GGUF (same gating as
    /// the engine smoke tests). The registry test model is seeded into the
    /// cache with a manifest matching the registry URL, so `download`'s
    /// fast path serves it without touching the network.
    #[tokio::test(flavor = "multi_thread")]
    async fn bench_measures_a_profile_persists_it_and_reports_links() {
        let Ok(smoke) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping bench endpoint test");
            return;
        };
        let (base, _notify, host, host_thread, dir) = serve_internal("sekrit").await;
        let spec = match BENCH_MODEL_ID
            .parse::<ModelRef>()
            .unwrap()
            .resolve()
            .unwrap()
        {
            Resolved::Remote(spec) => spec,
            other => panic!("registry id must resolve remote, got {other:?}"),
        };
        let dest = dir.path().join("models").join(&spec.cache_key);
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::copy(&smoke, dest.join(&spec.file_name)).unwrap();
        let size = std::fs::metadata(dest.join(&spec.file_name)).unwrap().len();
        let manifest = onebrain_models::download::Manifest {
            url: spec.url.clone(),
            size_bytes: size,
            // The cached-file fast path checks url + size, not the hash.
            blake3: "seeded-by-test".to_string(),
        };
        std::fs::write(
            dest.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let resp = reqwest::Client::new()
            .post(format!("{base}/api/internal/bench"))
            .bearer_auth("sekrit")
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let profile = &body["profile"];
        assert!(
            profile["prefill_tps"].as_f64().unwrap() > 0.0,
            "prefill must be positive: {body}"
        );
        assert!(
            profile["decode_tps"].as_f64().unwrap() > 0.0,
            "decode must be positive: {body}"
        );
        assert!(
            profile["disk_mbps"].as_f64().unwrap() > 0.0,
            "disk must be positive: {body}"
        );
        assert!(
            profile["measured_unix"].as_u64().unwrap() > 0,
            "measured_unix stamp missing: {body}"
        );
        // The report's NODE table shows memory next to the throughputs.
        assert!(
            profile["usable_memory_bytes"].is_u64(),
            "usable_memory_bytes missing: {body}"
        );
        // No peers on the hermetic mesh: the link table is present but empty.
        assert_eq!(body["links"], serde_json::json!([]));

        // The profile persisted where the daemon reloads it at startup.
        let stored =
            onebrain_scheduler::load_profile(&dir.path().join("config").join("profile.toml"))
                .expect("bench persists profile.toml");
        assert_eq!(
            stored.measured_unix,
            profile["measured_unix"].as_u64().unwrap()
        );
        assert!(stored.decode_tps > 0.0);
        teardown(host, host_thread);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn shutdown_acknowledges_and_notifies() {
        let (base, notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let notified = {
            let notify = notify.clone();
            tokio::spawn(async move { notify.notified().await })
        };
        let resp = reqwest::Client::new()
            .post(format!("{base}/api/internal/shutdown"))
            .bearer_auth("sekrit")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "stopping");
        tokio::time::timeout(Duration::from_secs(5), notified)
            .await
            .expect("shutdown must fire the notify")
            .unwrap();
        teardown(host, host_thread);
    }
}
