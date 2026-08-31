//! The internal control API (`/api/internal/*`, docs/internal-api.md).
//!
//! These endpoints are ALWAYS token-authenticated: the public gateway's
//! localhost exemption deliberately does not apply here, so a random local
//! process cannot stop the daemon or swap models without reading the token
//! file (same-user filesystem access).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
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
use onebrain_mesh::{
    MeshError, MeshHandle, PairEvent, PairTarget, PeerBenchReport, PeerState, PeerStatus,
};
use onebrain_models::registry::{ModelRef, Resolved};
use onebrain_proto::message::{Envelope, Message};
use onebrain_proto::plan::{Assignment, NodeId, Plan, Strategy};
use onebrain_scheduler::{
    ComputeProfile, LinkRtt, NodeCaps, PlanRequest, ScheduleError, StoredProfile,
};
use tokio::sync::{mpsc, oneshot, Notify};
use tokio_stream::wrappers::ReceiverStream;

use crate::cluster::{ActivePlanView, ClusterState, LoadedSource};
use crate::engine_host::{
    DraftRequest, EngineHost, HostMsg, LoadProgress, LoadedModel, ProgressThrottle,
};

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
    /// This node's human-readable name (the one peers see), surfaced by
    /// the metrics endpoint's `node` section (M8, docs/product.md §1).
    pub node_name: String,
    /// M8 metrics request ring (docs/product.md §1): written by the daemon
    /// backend's generation relay, read by `GET /api/internal/metrics`.
    pub requests: Arc<crate::metrics::RequestLog>,
    pub cache_root: PathBuf,
    pub ctx_len: u32,
    /// `[perf] n_ubatch` (docs/perf.md §3/§7): the effective microbatch,
    /// handed to the planner so the v2 transfer term and pipeline reserve
    /// size the per-ubatch boundary copy from the real knob.
    pub n_ubatch: u32,
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
    /// Battery probe for the drain policy (docs/resilience.md); same
    /// instance the `NodeStatus` provider uses so head and workers apply
    /// one policy.
    pub battery_probe: Arc<dyn crate::power::BatteryProbe + Send + Sync>,
    /// `config.battery_drain_threshold`.
    pub battery_threshold: u8,
    /// Where `POST /api/internal/bench` persists the profile
    /// (`<config_dir>/profile.toml`).
    pub profile_path: PathBuf,
    /// `config.cache_max_bytes`: the post-download GC trigger's cap
    /// (docs/logistics.md "LRU GC + pinning"; 0 = GC disabled).
    pub cache_max_bytes: u64,
    /// `[perf] draft_model` (docs/perf.md §5): the default speculative
    /// draft reference when a load asks for `speculative` without naming
    /// one.
    pub draft_model: Option<String>,
    /// Serialization point for epoch surgery (M7, docs/perf.md §6): the
    /// supervisor's interruption lifecycle, no-job teardown, and rejoin
    /// re-plan take this one at a time, and it remembers each failed
    /// epoch's outcome so concurrently interrupted jobs resolve
    /// consistently (see crate::supervisor's module docs).
    pub retry: tokio::sync::Mutex<crate::supervisor::RetryLedger>,
}

/// Build the internal router with its always-on token middleware.
pub fn internal_router(state: Arc<InternalState>) -> Router {
    Router::new()
        .route("/api/internal/status", get(status))
        .route("/api/internal/metrics", get(metrics))
        .route("/api/internal/load", post(load))
        .route("/api/internal/bench", post(bench))
        .route("/api/internal/bench/peers", post(bench_peers))
        .route("/api/internal/perf", post(perf_toggles))
        .route("/api/internal/shutdown", post(shutdown))
        .route("/api/internal/pair/start", post(pair_start))
        .route("/api/internal/pair/join", post(pair_join))
        .route("/api/internal/peers", get(peers))
        .route("/api/internal/unpair", post(unpair))
        .route("/api/internal/models/pin", post(pin_model))
        .route("/api/internal/models/unpin", post(unpin_model))
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
        // The reference the loaded model was requested with (registry id,
        // `hf:…`, or local path) — what a client re-issues to reload the
        // same model (`onebrain bench --cluster` does exactly that for its
        // end-to-end runs, docs/perf.md §10). Null when nothing is loaded.
        "model_reference": state.cluster.loaded_source().map(|s| s.reference),
        "peers_summary": { "paired": peer_list.len(), "connected": connected },
        // docs/distributed.md: the active plan (epoch, strategy,
        // assignments) or null when nothing distributed is active.
        "plan": state.cluster.active(),
    }))
}

/// `GET /api/internal/metrics` (M8, docs/product.md §1): ONE JSON document
/// feeding the dashboard, additive-stable — fields are only ever added:
///
/// - `node`: name, platform, version, engine build, measured memory,
///   devices, the persisted bench profile (null until benched), battery
///   verdict, and whether the sleep inhibitor is held.
/// - `peers[]`: name, id-prefix, state, measured link figures, last
///   NodeStatus memory/profile, and version + engine build from STORED
///   Hello data — skew is computable client- and server-side.
/// - `plan`: the [`ActivePlanView`] or null.
/// - `requests[]`: the in-memory ring of the last 50 finished generations
///   (head only, never any prompt text — see `crate::metrics`).
/// - `advisor[]`: `{severity, text}` findings from the pure rules in
///   `crate::advisor`, each backed by a measurement.
async fn metrics(State(state): State<Arc<InternalState>>) -> Json<serde_json::Value> {
    // Local device probes are blocking reads; ride the blocking pool like
    // every other caller of `local_node_status`.
    let override_bytes = state.usable_memory_override;
    let (usable_memory_bytes, devices) =
        tokio::task::spawn_blocking(move || crate::cluster::local_node_status(override_bytes))
            .await
            .unwrap_or((0, Vec::new()));
    let total_bytes = devices
        .iter()
        .find(|d| d.kind == "cpu")
        .map(|d| d.total_bytes)
        .unwrap_or(0);
    let battery =
        crate::power::battery_status(state.battery_probe.as_ref(), state.battery_threshold);
    // The persisted bench profile; the test-only decode override wins
    // wherever decode is reported (same rule as NodeStatus and status).
    let stored = *state.profile.lock().expect("profile state poisoned");
    let profile_json = stored.map(|p| {
        serde_json::json!({
            "prefill_tps": p.prefill_tps,
            "decode_tps": state.decode_tps_override.unwrap_or(p.decode_tps),
            "disk_mbps": p.disk_mbps,
            "measured_unix": p.measured_unix,
        })
    });
    let peers = state.mesh.peers().await.unwrap_or_else(|err| {
        tracing::warn!(error = %err, "metrics could not read peers; reporting none");
        Vec::new()
    });
    let plan = state.cluster.active();
    let loaded = state.cluster.loaded_source();
    // Sleep-inhibited is REPORTED through the same pure predicate the
    // runtime's inhibitor watcher applies every 2 s, over the same inputs —
    // the report can never disagree with the policy.
    let serving_shard = state.cluster.worker_shard().is_some();
    let model_loaded = loaded.is_some() || serving_shard;
    let in_flight = !state.host.is_idle();
    let sleep_inhibited =
        crate::power::should_hold_sleep(model_loaded, in_flight || serving_shard, plan.is_some());

    let engine_build = onebrain_engine::engine_build_hash().0;
    let own_id = state.mesh.endpoint_id().to_string();
    let advisor = crate::advisor::advise(&crate::advisor::AdvisorInput {
        node_name: &state.node_name,
        own_id: &own_id,
        product_version: state.product_version,
        engine_build: &engine_build,
        usable_memory_bytes,
        draining: battery.draining,
        peers: &peers,
        plan: plan.as_ref(),
        loaded_size_bytes: loaded.as_ref().map(|s| s.size_bytes),
    });

    let peers_json: Vec<serde_json::Value> = peers
        .iter()
        .map(|p| {
            // A peer that has benched reports at least one profile figure;
            // one that never did gets an honest null, not zeros.
            let profile = (p.prefill_tps.is_some()
                || p.decode_tps.is_some()
                || p.disk_mbps.is_some())
            .then(|| {
                serde_json::json!({
                    "prefill_tps": p.prefill_tps,
                    "decode_tps": p.decode_tps,
                    "disk_mbps": p.disk_mbps,
                })
            });
            let mut peer = serde_json::json!({
                "name": p.name,
                // Prefix only: enough to correlate with plan assignments and
                // logs without dumping full endpoint keys into the page.
                "id_prefix": p.id.chars().take(8).collect::<String>(),
                "state": p.state,
                "rtt_ms": p.rtt_ms,
                "bandwidth_mbps": p.bandwidth_mbps,
                "loss": p.loss,
                "last_seen_unix": p.last_seen_unix,
                "usable_memory_bytes": p.usable_memory_bytes,
                "profile": profile,
                "draining": p.draining,
            });
            // Hello-derived fields are OMITTED (not null) until a hello has
            // been exchanged: absent = never introduced, and tolerant
            // consumers default a missing string where a null would not
            // parse.
            if let Some(version) = &p.product_version {
                peer["version"] = serde_json::json!(version);
            }
            if let Some(build) = &p.engine_build {
                peer["engine_build"] = serde_json::json!(build);
            }
            peer
        })
        .collect();

    Json(serde_json::json!({
        "node": {
            "name": state.node_name,
            "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            "version": state.product_version,
            "engine_build": engine_build,
            // The plan-share id peers see this node as (assignments use it).
            "id_prefix": own_id.chars().take(8).collect::<String>(),
            "memory": {
                "usable_bytes": usable_memory_bytes,
                "total_bytes": total_bytes,
            },
            "devices": devices,
            "profile": profile_json,
            "battery": {
                "level_percent": battery.level,
                "draining": battery.draining,
            },
            "sleep_inhibited": sleep_inhibited,
        },
        "peers": peers_json,
        "plan": plan,
        "requests": state.requests.snapshot(),
        "advisor": advisor,
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
    /// `--speculative` (docs/perf.md §5): load a draft model alongside the
    /// target. The draft is `draft` when given, else the config's
    /// `[perf] draft_model`; neither ⇒ a typed error naming the remedy.
    #[serde(default)]
    speculative: bool,
    /// `--draft <ref>`: explicit draft-model reference (implies
    /// speculative).
    #[serde(default)]
    draft: Option<String>,
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
    // LAN-first like every other download path (docs/logistics.md).
    let model_path = match crate::logistics::ensure_remote_local(
        &state.mesh,
        &state.cache_root,
        &spec,
        |_, _| {},
    )
    .await
    {
        Ok(fetched) => fetched.paths[0].clone(),
        Err(message) => return bench_error(StatusCode::BAD_GATEWAY, message),
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
            draining: crate::power::battery_status(
                state.battery_probe.as_ref(),
                state.battery_threshold,
            )
            .draining,
        });
        for peer in &connected {
            if let Err(err) = state.mesh.send_control(&peer.id, envelope.clone()).await {
                tracing::debug!(peer = %peer.name, "NodeStatus push after bench failed: {err}");
            }
        }
    }

    Json(serde_json::json!({ "profile": profile_json, "links": links })).into_response()
}

/// `POST /api/internal/bench/peers` (docs/perf.md §10): ask every Connected
/// peer to run its microbench on demand via the mesh's
/// `BenchRequest`/`BenchReport` exchange. Queries run CONCURRENTLY — each
/// peer's bench takes seconds and the mesh bounds each reply at 60 s, so
/// serializing them would make a 3-node bench a 3-minute wait. Per-peer
/// failures (timeout, disconnect) become per-peer `error` entries, never a
/// failed call: `bench --cluster` reports what it could measure.
async fn bench_peers(State(state): State<Arc<InternalState>>) -> Response {
    let peers = match state.mesh.peers().await {
        Ok(peers) => peers,
        Err(err) => return mesh_error_response(err),
    };
    let mut queries = tokio::task::JoinSet::new();
    for p in peers
        .into_iter()
        .filter(|p| p.state == PeerState::Connected)
    {
        let mesh = state.mesh.clone();
        queries.spawn(async move {
            // Resolve by endpoint id: names can be re-used across unpair/
            // re-pair cycles within one bench, ids cannot.
            let result = mesh.bench_query(&p.id).await;
            (p.name, p.id, result)
        });
    }
    let mut rows: Vec<(String, serde_json::Value)> = Vec::new();
    while let Some(joined) = queries.join_next().await {
        let Ok((name, id, result)) = joined else {
            continue; // a panicked query task reports nothing for its peer
        };
        let sort_key = name.clone();
        let row = match result {
            Ok(report) if !report.is_unavailable() => serde_json::json!({
                "peer": name, "id": id, "available": true,
                "prefill_tps": report.prefill_tps,
                "decode_tps": report.decode_tps,
                "disk_mbps": report.disk_mbps,
                "measured_unix": report.measured_unix,
            }),
            // The wire's cannot-bench-now marker (measured_unix == 0): the
            // throughput fields are meaningless and deliberately omitted.
            Ok(_) => serde_json::json!({ "peer": name, "id": id, "available": false }),
            Err(err) => serde_json::json!({
                "peer": name, "id": id, "available": false, "error": err.to_string(),
            }),
        };
        rows.push((sort_key, row));
    }
    // Concurrent completion order is nondeterministic; name-sort for the
    // reproducible-report rule (docs/perf.md §10).
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    let rows: Vec<serde_json::Value> = rows.into_iter().map(|(_, row)| row).collect();
    Json(serde_json::json!({ "peers": rows })).into_response()
}

#[derive(Debug, serde::Deserialize)]
struct PerfBody {
    /// `Some` overrides the lever; `None` leaves it as it is.
    #[serde(default)]
    prefill_overlap: Option<bool>,
    #[serde(default)]
    kv_reuse: Option<bool>,
}

/// `POST /api/internal/perf` (docs/perf.md §10): read or override the
/// runtime-togglable `[perf]` levers. An empty body `{}` changes nothing
/// and answers with the current values. Overrides take effect at the NEXT
/// model (re)load — live sessions keep the mode they were created with —
/// which is how `onebrain bench --cluster` constructs the M3 baseline
/// (`prefill_overlap=false` + `kv_reuse=false`) without a daemon restart:
/// flip, reload, measure, flip back, reload. Overrides do not persist;
/// a daemon restart returns to config.toml's values.
async fn perf_toggles(
    State(state): State<Arc<InternalState>>,
    Json(body): Json<PerfBody>,
) -> Json<serde_json::Value> {
    let (prefill_overlap, kv_reuse) = state
        .host
        .set_perf_toggles(body.prefill_overlap, body.kv_reuse);
    tracing::info!(
        prefill_overlap,
        kv_reuse,
        "runtime perf toggles read/set via /api/internal/perf"
    );
    Json(serde_json::json!({
        "prefill_overlap": prefill_overlap,
        "kv_reuse": kv_reuse,
        // Documented for humans poking the endpoint; machines key off the
        // two booleans.
        "applies_at": "next model load",
    }))
}

/// The daemon's [`onebrain_mesh::BenchSource`]: answers peers'
/// `BenchRequest` control messages (docs/perf.md §10) by running the SAME
/// measurement `POST /api/internal/bench` runs — the compute microbench on
/// the registry test model plus the disk probe — and persisting the result
/// to `profile.toml` and the [`SharedProfile`], so a bench a peer asked for
/// leaves this node's own profile fresh too. The mesh calls
/// [`onebrain_mesh::BenchSource::bench`] on its blocking pool, so the
/// seconds-long measurement never stalls mesh traffic.
///
/// Declines (`None` → the wire's cannot-bench-now marker) instead of
/// measuring when:
/// - a generation is queued or in flight (the figures would be noise, and
///   the microbench would steal compute from a real request);
/// - this node is serving a pipeline shard (head-driven decode traffic
///   arrives outside the local job counter's view, and the shard owns this
///   node's memory);
/// - the test model is not fully cached (a passive peer never spends WAN
///   bandwidth answering a query — run `onebrain bench` on that node once).
pub struct DaemonBenchSource {
    pub host: EngineHost,
    pub cluster: Arc<ClusterState>,
    pub cache_root: PathBuf,
    pub profile: SharedProfile,
    pub profile_path: PathBuf,
}

impl onebrain_mesh::BenchSource for DaemonBenchSource {
    fn bench(&self) -> Option<PeerBenchReport> {
        if !self.host.is_idle() {
            tracing::info!("declining a peer's bench request: a generation is queued or in flight");
            return None;
        }
        if let Some((epoch, _head)) = self.cluster.worker_shard() {
            tracing::info!(
                epoch = epoch.0,
                "declining a peer's bench request: serving a pipeline shard"
            );
            return None;
        }
        let Some(path) = cached_bench_model(&self.cache_root) else {
            tracing::info!(
                "declining a peer's bench request: the {BENCH_MODEL_ID:?} test model is not \
                 cached here (run `onebrain bench` on this node once to fetch it)"
            );
            return None;
        };
        let compute = match onebrain_scheduler::measure_compute(&path) {
            Ok(compute) => compute,
            Err(e) => {
                tracing::warn!(error = %e, "peer-requested compute microbench failed; declining");
                return None;
            }
        };
        let disk_mbps = match onebrain_scheduler::measure_disk(&path) {
            Ok(disk) => disk,
            Err(e) => {
                tracing::warn!(error = %e, "peer-requested disk probe failed; declining");
                return None;
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
        // Persist + publish exactly like `POST /api/internal/bench`; a
        // failed persist keeps the fresh in-memory profile (the NodeStatus
        // provider and planner read that) and still answers the peer.
        if let Err(e) = onebrain_scheduler::save_profile(&self.profile_path, &stored) {
            tracing::warn!(
                error = %e,
                "peer-requested bench could not persist profile.toml; keeping it in memory"
            );
        }
        *self.profile.lock().expect("profile state poisoned") = Some(stored);
        tracing::info!(
            prefill_tps = stored.prefill_tps,
            decode_tps = stored.decode_tps,
            disk_mbps = stored.disk_mbps,
            "device profile refreshed by a peer's bench request"
        );
        Some(PeerBenchReport {
            prefill_tps: stored.prefill_tps,
            decode_tps: stored.decode_tps,
            disk_mbps: stored.disk_mbps,
            measured_unix: stored.measured_unix,
        })
    }
}

/// The bench test model's local path IF it is fully cached: a manifest
/// recording the registry URL plus a complete file — the same fast-path
/// check the download machinery applies, without ever touching the
/// network. `None` = not cached (or the registry cannot resolve, which
/// `POST /api/internal/bench` reports as a packaging bug on the local
/// path).
fn cached_bench_model(cache_root: &Path) -> Option<PathBuf> {
    let spec = match BENCH_MODEL_ID.parse::<ModelRef>().ok()?.resolve().ok()? {
        Resolved::Remote(spec) => spec,
        Resolved::Local(_) => return None,
    };
    let dir = cache_root.join(&spec.cache_key);
    let manifest = onebrain_models::download::read_manifest(&dir).ok()?;
    if manifest.url != spec.url {
        return None;
    }
    let path = dir.join(&spec.file_name);
    let complete = std::fs::metadata(&path)
        .map(|m| m.len() == manifest.size_bytes)
        .unwrap_or(false);
    complete.then_some(path)
}

/// A model reference resolved to local files (downloaded if needed). Split
/// models carry every part in load order; single files carry one path.
pub(crate) struct LocalModel {
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) name: String,
    pub(crate) size_bytes: u64,
}

impl LocalModel {
    /// The path planning reads the GGUF header from (part 1 carries the
    /// model-wide metadata for split sets).
    pub(crate) fn header_path(&self) -> &Path {
        &self.paths[0]
    }
}

impl From<&LoadedSource> for LocalModel {
    /// The supervisor's reload path rebuilds the local-model view from the
    /// recorded source of the model that was loaded (M5).
    fn from(source: &LoadedSource) -> LocalModel {
        LocalModel {
            paths: source.paths.clone(),
            name: source.name.clone(),
            size_bytes: source.size_bytes,
        }
    }
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
                paths: vec![path],
            })
        }
        Resolved::Remote(spec) => {
            let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<(u64, u64)>();
            let mut throttle = ProgressThrottle::default();
            // LAN-first + split-aware (docs/logistics.md): peers are asked
            // before any WAN byte; split refs fetch every part.
            let download = crate::logistics::ensure_remote_local(
                &state.mesh,
                &state.cache_root,
                &spec,
                move |completed, total| {
                    if throttle.should_emit(completed, total) {
                        let _ = progress_tx.send((completed, total));
                    }
                },
            );
            tokio::pin!(download);
            let fetched = loop {
                tokio::select! {
                    result = &mut download => match result {
                        Ok(fetched) => break fetched,
                        Err(message) => {
                            emit_error(tx, message).await;
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
            };
            // GC trigger (docs/logistics.md): after every completed
            // download — never the loaded model, never the fresh entry.
            let root = state.cache_root.clone();
            let max = state.cache_max_bytes;
            let mut protected: HashSet<String> = HashSet::new();
            protected.insert(spec.cache_key.clone());
            if let Some(source) = state.cluster.loaded_source() {
                protected.insert(source.name);
            }
            let _ = tokio::task::spawn_blocking(move || {
                crate::logistics::run_cache_gc(&root, max, &protected)
            })
            .await;
            Some(LocalModel {
                name: spec.cache_key.clone(),
                paths: fetched.paths,
                size_bytes: fetched.size_bytes,
            })
        }
    }
}

/// How a planning attempt failed (factored from `drive_load` for the M5
/// supervisor, docs/resilience.md step 3: the retry path needs the
/// infeasible case STRUCTURED — the typed lost-node error carries both MB
/// numbers — while every other failure stays a user-facing string).
#[derive(Debug)]
pub(crate) enum PlanLoadError {
    /// The model does not fit the eligible node set pooled.
    DoesNotFit {
        required_mb: u64,
        available_mb: u64,
        /// The scheduler's own user-facing message (used verbatim by the
        /// normal load path).
        message: String,
    },
    /// Anything else (mesh failures, header parse, other scheduler errors).
    Other(String),
}

impl PlanLoadError {
    pub(crate) fn into_message(self) -> String {
        match self {
            PlanLoadError::DoesNotFit { message, .. } => message,
            PlanLoadError::Other(message) => message,
        }
    }
}

/// A stamped placement, ready to activate (factored from `drive_load`).
pub(crate) struct PlannedLoad {
    /// The plan with `model` set and — for a distributed plan — a freshly
    /// stamped epoch.
    pub(crate) plan: Plan,
    pub(crate) tensor_split: Vec<f32>,
    pub(crate) explanation: String,
    /// Present exactly on distributed plans (the relative comparison
    /// metric, never a latency promise — §1.6).
    pub(crate) predicted_tpt_ms: Option<f64>,
    /// The v2 scheduler's SECONDARY comparison key mirrored for the same
    /// plan (docs/perf.md §7): Σ per pipeline boundary of
    /// `RTT + (4·n_embd·n_ubatch)/measured_bandwidth` ms. Present exactly
    /// on distributed plans, like `predicted_tpt_ms`; links without a
    /// measured bandwidth contribute RTT only.
    pub(crate) predicted_prefill_ms: Option<f64>,
    /// This node's mesh endpoint id (the head).
    pub(crate) own_id: String,
}

/// Plan a load of the already-local model at `path` (M7 scheduler v2-lite:
/// real model dims, device profiles, measured link RTTs + bandwidth, MoE
/// active-expert scaling and the pipeline-copy reserve on top of the M4 v1
/// rules — docs/scheduler-v1.md + docs/perf.md §7).
/// Shared by the normal load flow, the M5 supervisor's retry re-plan, and
/// the lazy rejoin re-plan. Eligible workers are the Connected peers that
/// reported a budget, minus `exclude` (nodes lost to the failed epoch);
/// with `exclude_draining` the retry path drops draining peers entirely
/// (docs/resilience.md step 3: "dead/suspect/draining excluded") instead of
/// leaving them to the scheduler's include-only-if-infeasible rule.
pub(crate) async fn plan_load(
    state: &Arc<InternalState>,
    model_name: &str,
    path: &Path,
    size_bytes: u64,
    forced_nodes: Option<u32>,
    exclude: &HashSet<String>,
    exclude_draining: bool,
) -> Result<PlannedLoad, PlanLoadError> {
    let override_bytes = state.usable_memory_override;
    let head_status =
        tokio::task::spawn_blocking(move || crate::cluster::local_node_status(override_bytes))
            .await;
    let Ok((head_usable, _devices)) = head_status else {
        return Err(PlanLoadError::Other(
            "measuring local memory failed; retry the load".to_string(),
        ));
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
            Err(e) => return Err(PlanLoadError::Other(e.to_string())),
        };
        let connected: Vec<PeerStatus> = all
            .into_iter()
            .filter(|p| p.state == PeerState::Connected)
            .filter(|p| !exclude.contains(&p.id))
            .filter(|p| !(exclude_draining && p.draining))
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
    let header_path = path.to_path_buf();
    let header =
        match tokio::task::spawn_blocking(move || crate::cluster::read_gguf_header(&header_path))
            .await
        {
            Ok(Ok(header)) => header,
            Ok(Err(message)) => return Err(PlanLoadError::Other(message)),
            Err(_) => {
                return Err(PlanLoadError::Other(
                    "reading the model header failed; retry the load".to_string(),
                ))
            }
        };
    let dims = match onebrain_scheduler::model_dims(&header, size_bytes) {
        Ok(dims) => dims,
        Err(e) => return Err(PlanLoadError::Other(e.to_string())),
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
            draining: p.draining,
        });
        // Head <-> worker RTT from the heartbeat EWMA. Worker <-> worker
        // links are not measured in M4 and are omitted — the scheduler
        // defaults missing pairs to DEFAULT_LINK_RTT_MS.
        if let Some(rtt_ms) = p.rtt_ms {
            links.push(LinkRtt {
                a: NodeId(own_id.clone()),
                b: NodeId(p.id.clone()),
                rtt_ms,
                // Measured link bandwidth from the mesh probe rides along
                // for the v2 transfer term (docs/perf.md §7); None (or a
                // nonsense non-positive figure) keeps the link RTT-only —
                // exactly the pre-M7 costing.
                bandwidth_mbps: p.bandwidth_mbps.filter(|b| *b > 0.0),
            });
        }
    }
    let request = PlanRequest {
        head: NodeCaps {
            node: NodeId(own_id.clone()),
            usable_memory_bytes: head_usable,
            compute: head_compute,
            // One policy for head and workers: a draining head still
            // participates when nothing fits without it (scheduler rule).
            draining: crate::power::battery_status(
                state.battery_probe.as_ref(),
                state.battery_threshold,
            )
            .draining,
        },
        workers,
        dims,
        ctx_len: state.ctx_len,
        forced_nodes,
        links,
        n_ubatch: state.n_ubatch,
    };
    // plan_v2 (M7, docs/perf.md §7): same admission rules as plan_v1, plus
    // the candidate-family search, the bandwidth-priced prefill transfer
    // term, MoE active-expert compute scaling, and the pipeline-parallel
    // copy reserve. With no profiles, no bandwidth, and dense dims it
    // reproduces plan_v1's decisions exactly (scheduler contract).
    let placed = match onebrain_scheduler::plan_v2(&request) {
        Ok(p) => p,
        Err(e @ ScheduleError::DoesNotFit { .. }) => {
            let ScheduleError::DoesNotFit {
                required_mb,
                available_mb,
            } = e
            else {
                unreachable!("matched DoesNotFit above");
            };
            return Err(PlanLoadError::DoesNotFit {
                required_mb,
                available_mb,
                message: e.to_string(),
            });
        }
        Err(e) => return Err(PlanLoadError::Other(e.to_string())),
    };
    let mut plan = placed.plan;
    plan.model = model_name.to_string();
    let solo = plan.strategy == Strategy::Solo;
    if !solo {
        plan.epoch = state.cluster.next_epoch();
    }
    let predicted_tpt_ms = (!solo).then(|| predicted_tpt_ms(&plan.assignments, &request));
    let predicted_prefill_ms = (!solo).then(|| predicted_prefill_ms(&plan.assignments, &request));
    Ok(PlannedLoad {
        plan,
        tensor_split: placed.tensor_split,
        explanation: placed.explanation,
        predicted_tpt_ms,
        predicted_prefill_ms,
        own_id,
    })
}

async fn drive_load(state: Arc<InternalState>, body: LoadBody, tx: LineSender) {
    // Phase A: the head holds the full GGUF (ADR 0004 head-push weight
    // flow) — make it local before planning needs its header.
    let Some(local) = ensure_local(&state, &body.model, &tx).await else {
        return;
    };

    // Speculative draft (docs/perf.md §5): an explicit `draft` implies
    // speculative; `speculative` alone falls back to `[perf] draft_model`.
    // The draft is made local HERE (progress streams to the client) so the
    // engine host's own resolve is a cache hit; it always loads solo on
    // this head node, whatever the target's plan says.
    let draft_req: Option<DraftRequest> = if body.speculative || body.draft.is_some() {
        let reference = match body.draft.clone().or_else(|| state.draft_model.clone()) {
            Some(reference) => reference,
            None => {
                emit_error(
                    &tx,
                    "speculative decoding needs a draft model: pass --draft <ref> or set \
                     [perf] draft_model in config.toml. The draft must share the target's \
                     vocabulary — in the built-in registry, 'qwen3-0.6b' pairs with \
                     'qwen3-1.7b', 'qwen3-4b', and 'qwen3-32b'"
                        .to_string(),
                )
                .await;
                return;
            }
        };
        if ensure_local(&state, &reference, &tx).await.is_none() {
            return; // the error line was already emitted
        }
        Some(DraftRequest {
            reference,
            cache_root: state.cache_root.clone(),
        })
    } else {
        None
    };

    // Phase B: plan (M7 scheduler v2-lite — factored into `plan_load`,
    // shared with the M5 supervisor's re-plan paths).
    let _ = emit(&tx, serde_json::json!({ "status": "planning" })).await;
    let planned = match plan_load(
        &state,
        &local.name,
        local.header_path(),
        local.size_bytes,
        body.nodes,
        &HashSet::new(),
        false,
    )
    .await
    {
        Ok(planned) => planned,
        Err(e) => {
            emit_error(&tx, e.into_message()).await;
            return;
        }
    };
    let solo = planned.plan.strategy == Strategy::Solo;
    let mut plan_json =
        serde_json::to_value(&planned.plan).unwrap_or_else(|_| serde_json::json!({}));
    if let Some(tpt) = planned.predicted_tpt_ms {
        plan_json["predicted_tpt_ms"] = serde_json::json!(tpt);
    }
    // Additive (M7): the v2 secondary key — existing fields keep their
    // meaning, this one only appears alongside predicted_tpt_ms.
    if let Some(prefill) = planned.predicted_prefill_ms {
        plan_json["predicted_prefill_ms"] = serde_json::json!(prefill);
    }
    if body.explain {
        plan_json["explanation"] = serde_json::json!(planned.explanation);
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
        solo_load(&state, &body.model, &local, planned, draft_req, &tx).await;
    } else {
        let _ = emit(&tx, serde_json::json!({ "status": "loading" })).await;
        match activate_distributed_plan(&state, &body.model, &local, planned, draft_req).await {
            Ok(model) => {
                let _ = emit(
                    &tx,
                    serde_json::json!({ "status": "ready", "model": model }),
                )
                .await;
            }
            Err(message) => emit_error(&tx, message).await,
        }
    }
}

/// Mirror of the scheduler's PRIMARY plan-comparison metric for the FINAL
/// plan (v2 module docs, docs/perf.md §7-§8):
/// `max_stage(active layer units / decode_tps) × 1000 + Σ boundary RTT ms`,
/// where a stage's active units are [`onebrain_scheduler::ModelDims::
/// active_compute_units`] over its assigned range (a MoE layer costs only
/// the weight fraction a token actually touches; dense models reduce to
/// the plain layer count, i.e. the v1 figure), an unprofiled node is
/// costed at the slowest profiled node's rate (conservative) or
/// contributes zero when nothing is profiled, and unmeasured links default
/// to [`onebrain_scheduler::DEFAULT_LINK_RTT_MS`]. A RELATIVE figure (the
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
            // MoE active-unit scaling (v2): the assigned range's active
            // compute units, not its raw layer count. Dense: identical.
            Some(tps) => {
                request
                    .dims
                    .active_compute_units(a.layers.start, a.layers.end)
                    / tps
                    * 1000.0
            }
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

/// Mirror of the scheduler's SECONDARY plan-comparison key for the FINAL
/// plan (v2 module docs, docs/perf.md §7): Σ per pipeline boundary of
/// `RTT + (4·n_embd·n_ubatch)/measured_bandwidth` ms — the per-ubatch
/// prefill boundary cost. Links without a measured bandwidth (or with an
/// unknown embedding width) contribute RTT only, and unprobed pairs
/// default to [`onebrain_scheduler::DEFAULT_LINK_RTT_MS`] — exactly the
/// scheduler's own degradation. Surfaced additively as
/// `predicted_prefill_ms` next to `predicted_tpt_ms`; same relative-only
/// caveat (§1.6).
fn predicted_prefill_ms(assignments: &[Assignment], request: &PlanRequest) -> f64 {
    assignments
        .windows(2)
        .map(|pair| {
            let link = request.links.iter().find(|l| {
                (l.a == pair[0].node && l.b == pair[1].node)
                    || (l.a == pair[1].node && l.b == pair[0].node)
            });
            let rtt = link
                .map(|l| l.rtt_ms)
                .unwrap_or(onebrain_scheduler::DEFAULT_LINK_RTT_MS);
            let transfer = link
                .and_then(|l| l.bandwidth_mbps)
                .map(|bw| {
                    onebrain_scheduler::boundary_transfer_ms(
                        request.dims.n_embd,
                        request.n_ubatch,
                        bw,
                    )
                })
                .unwrap_or(0.0);
            rtt + transfer
        })
        .sum()
}

/// Install the head-side state of a freshly ready SOLO load: the active
/// plan view, the loaded-model source (M5 — the supervisor's re-plan paths
/// read it), and the teardown of any replaced distributed plan's bridges
/// (they are stale — the host dropped that model during the swap, ADR 0004
/// ordering: model freed first, bridges closed after).
fn install_solo_active(
    state: &Arc<InternalState>,
    reference: &str,
    local: &LocalModel,
    plan: Plan,
    explanation: String,
    draft_reference: Option<String>,
) {
    state.cluster.set_active(Some(ActivePlanView {
        role: "head",
        plan,
        explanation: Some(explanation),
        predicted_tpt_ms: None,
        predicted_prefill_ms: None,
    }));
    state.cluster.teardown_head_bridges();
    state.cluster.note_loaded(LoadedSource {
        reference: reference.to_string(),
        name: local.name.clone(),
        paths: local.paths.clone(),
        size_bytes: local.size_bytes,
        draft_reference,
    });
}

/// The unchanged single-node path: the engine host resolves the (already
/// cached) reference, loads, and answers; progress is forwarded as NDJSON.
async fn solo_load(
    state: &Arc<InternalState>,
    reference: &str,
    local: &LocalModel,
    planned: PlannedLoad,
    draft: Option<DraftRequest>,
    tx: &LineSender,
) {
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<LoadProgress>();
    let (resp_tx, resp_rx) = oneshot::channel();
    let draft_reference = draft.as_ref().map(|d| d.reference.clone());
    if state
        .host
        .send(HostMsg::Load {
            reference: reference.to_string(),
            cache_root: state.cache_root.clone(),
            ctx_len: state.ctx_len,
            draft,
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
            install_solo_active(
                state,
                reference,
                local,
                planned.plan,
                planned.explanation,
                draft_reference,
            );
            let _ = emit(tx, serde_json::json!({ "status": "ready", "model": model })).await;
        }
        Ok(Err(message)) => {
            // The host dropped any previous model before this load failed:
            // nothing is loaded now (M5: the supervisor must not reload a
            // stale source).
            state.cluster.clear_loaded();
            emit_error(tx, message).await;
        }
        Err(_) => {
            state.cluster.clear_loaded();
            emit_error(
                tx,
                "the engine host exited unexpectedly; restart with `onebrain up`".to_string(),
            )
            .await
        }
    }
}

/// SOLO reload without an NDJSON client — the M5 supervisor's retry and
/// rejoin paths (docs/resilience.md steps 4–5). Progress is drained
/// silently; the terminal outcome installs the same state as `solo_load`.
pub(crate) async fn activate_solo_plan(
    state: &Arc<InternalState>,
    reference: &str,
    local: &LocalModel,
    planned: PlannedLoad,
    draft: Option<DraftRequest>,
) -> Result<LoadedModel, String> {
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<LoadProgress>();
    let (resp_tx, resp_rx) = oneshot::channel();
    let draft_reference = draft.as_ref().map(|d| d.reference.clone());
    if state
        .host
        .send(HostMsg::Load {
            reference: reference.to_string(),
            cache_root: state.cache_root.clone(),
            ctx_len: state.ctx_len,
            draft,
            progress: progress_tx,
            resp: resp_tx,
        })
        .is_err()
    {
        return Err(ApiError::ShuttingDown.to_string());
    }
    while progress_rx.recv().await.is_some() {}
    match resp_rx.await {
        Ok(Ok(model)) => {
            install_solo_active(
                state,
                reference,
                local,
                planned.plan,
                planned.explanation,
                draft_reference,
            );
            Ok(model)
        }
        Ok(Err(message)) => {
            state.cluster.clear_loaded();
            Err(message)
        }
        Err(_) => {
            state.cluster.clear_loaded();
            Err("the engine host exited unexpectedly; restart with `onebrain up`".to_string())
        }
    }
}

/// The distributed activation (docs/distributed.md "Epoch lifecycle"):
/// propose the plan, collect acks (15 s, typed error naming the node), open
/// one `rpc` stream + loopback bridge per worker, then load through the
/// engine host with the plan's tensor split. Shared by the NDJSON load flow
/// and the M5 supervisor's retry/rejoin reloads; on success it installs the
/// new epoch's bridges, active-plan view, and loaded-model source.
pub(crate) async fn activate_distributed_plan(
    state: &Arc<InternalState>,
    reference: &str,
    local: &LocalModel,
    planned: PlannedLoad,
    draft: Option<DraftRequest>,
) -> Result<LoadedModel, String> {
    let PlannedLoad {
        plan,
        tensor_split,
        explanation,
        predicted_tpt_ms,
        predicted_prefill_ms,
        own_id,
    } = planned;
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
            return Err(format!("could not send the plan to node '{label}': {err}"));
        }
    }
    state
        .cluster
        .await_acks(epoch.0, &workers, Duration::from_secs(15))
        .await?;

    let mut endpoints = Vec::with_capacity(workers.len());
    let mut bridges = Vec::with_capacity(workers.len());
    for (id, label) in &workers {
        // The bridge opens one fresh mesh rpc stream per accepted loopback
        // connection (the RPC client dials the endpoint repeatedly), so it
        // owns the mesh handle rather than a single pre-opened stream.
        match crate::cluster::head_bridge(
            state.mesh.clone(),
            state.cluster.clone(),
            id.clone(),
            epoch,
        )
        .await
        {
            Ok((endpoint, task)) => {
                endpoints.push(endpoint);
                bridges.push(task);
            }
            Err(message) => {
                abort_all(bridges);
                return Err(format!(
                    "could not set up the rpc bridge for node '{label}': {message}"
                ));
            }
        }
    }

    let (resp_tx, resp_rx) = oneshot::channel();
    let draft_reference = draft.as_ref().map(|d| d.reference.clone());
    if state
        .host
        .send(HostMsg::LoadDistributed {
            paths: local.paths.clone(),
            reference: reference.to_string(),
            name: local.name.clone(),
            ctx_len: state.ctx_len,
            endpoints,
            tensor_split,
            use_local_device,
            draft,
            resp: resp_tx,
        })
        .is_err()
    {
        abort_all(bridges);
        return Err(ApiError::ShuttingDown.to_string());
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
                predicted_prefill_ms,
            }));
            state.cluster.note_loaded(LoadedSource {
                reference: reference.to_string(),
                name: local.name.clone(),
                paths: local.paths.clone(),
                size_bytes: local.size_bytes,
                draft_reference,
            });
            Ok(model)
        }
        Ok(Err(message)) => {
            // Any previously loaded model was dropped during the swap.
            abort_all(bridges);
            state.cluster.clear_loaded();
            Err(message)
        }
        Err(_) => {
            abort_all(bridges);
            state.cluster.clear_loaded();
            Err("the engine host exited unexpectedly; restart with `onebrain up`".to_string())
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

#[derive(Debug, serde::Deserialize)]
struct PinBody {
    model: String,
}

/// Map a [`onebrain_models::cache::CacheError`] to the internal API's error
/// envelope (the message is the error's own Display, remedy included).
fn cache_error_response(err: onebrain_models::cache::CacheError) -> Response {
    use onebrain_models::cache::CacheError;
    let status = match &err {
        CacheError::NotCached { .. } => StatusCode::NOT_FOUND,
        CacheError::InvalidId { .. } | CacheError::Pinned { .. } => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "cache_error"
    };
    (
        status,
        Json(serde_json::json!({
            "error": { "message": err.to_string(), "type": kind }
        })),
    )
        .into_response()
}

/// Shared body of the pin/unpin endpoints (docs/logistics.md "LRU GC +
/// pinning"): flip the entry's pin flag in its manifest. Blocking file I/O
/// rides the blocking pool.
async fn set_model_pinned(state: Arc<InternalState>, model: String, pinned: bool) -> Response {
    let root = state.cache_root.clone();
    let id = model.clone();
    let result =
        tokio::task::spawn_blocking(move || onebrain_models::cache::set_pinned(&root, &id, pinned))
            .await;
    match result {
        Ok(Ok(())) => Json(serde_json::json!({
            "status": "ok", "model": model, "pinned": pinned,
        }))
        .into_response(),
        Ok(Err(err)) => cache_error_response(err),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": "the pin task failed unexpectedly; retry the command",
                    "type": "cache_error"
                }
            })),
        )
            .into_response(),
    }
}

/// `POST /api/internal/models/pin` `{"model": id}` — pinned entries are
/// never GC eviction candidates.
async fn pin_model(State(state): State<Arc<InternalState>>, Json(body): Json<PinBody>) -> Response {
    set_model_pinned(state, body.model, true).await
}

/// `POST /api/internal/models/unpin` `{"model": id}`.
async fn unpin_model(
    State(state): State<Arc<InternalState>>,
    Json(body): Json<PinBody>,
) -> Response {
    set_model_pinned(state, body.model, false).await
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
        let (base, notify, host, thread, dir, _state) = serve_internal_with_state(token).await;
        (base, notify, host, thread, dir)
    }

    /// [`serve_internal`] that also hands back the shared state, for tests
    /// that seed it directly (the metrics request-log tests).
    async fn serve_internal_with_state(
        token: &str,
    ) -> (
        String,
        Arc<Notify>,
        EngineHost,
        std::thread::JoinHandle<()>,
        tempfile::TempDir,
        Arc<InternalState>,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let (host, host_thread) = EngineHost::spawn(None, crate::engine_host::HostPerf::default());
        let shutdown = Arc::new(Notify::new());
        let mesh = spawn_test_mesh(dir.path()).await;
        let state = Arc::new(InternalState {
            host: host.clone(),
            auth: AuthConfig {
                token: token.to_string(),
                localhost_exempt: false,
            },
            node_name: "test-node".to_string(),
            requests: crate::metrics::RequestLog::new(),
            cache_root: dir.path().join("models"),
            ctx_len: 4096,
            n_ubatch: 512,
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
            // Deterministic in tests: a desktop-shaped probe never drains.
            battery_probe: Arc::new(crate::power::mock::MockBattery {
                level: None,
                ac: Some(true),
            }),
            battery_threshold: 25,
            cache_max_bytes: 0,
            draft_model: None,
            retry: tokio::sync::Mutex::new(crate::supervisor::RetryLedger::default()),
        });
        let app = internal_router(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (
            format!("http://{addr}"),
            shutdown,
            host,
            host_thread,
            dir,
            state,
        )
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

    /// docs/product.md §1: the metrics endpoint is an internal route —
    /// token always required, no loopback exemption.
    #[tokio::test(flavor = "multi_thread")]
    async fn metrics_requires_the_token() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let resp = reqwest::Client::new()
            .get(format!("{base}/api/internal/metrics"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        teardown(host, host_thread);
    }

    /// The §1 document shape on a bare daemon: every section present, the
    /// unmeasured parts honestly null/empty rather than zero-faked.
    #[tokio::test(flavor = "multi_thread")]
    async fn metrics_reports_the_contract_shape_on_a_bare_daemon() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let resp = reqwest::Client::new()
            .get(format!("{base}/api/internal/metrics"))
            .bearer_auth("sekrit")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let node = &body["node"];
        assert_eq!(node["name"], "test-node");
        assert_eq!(node["version"], "test");
        assert!(node["engine_build"].as_str().unwrap().contains("llama.cpp"));
        assert!(node["platform"].as_str().unwrap().contains('-'));
        assert!(node["memory"]["usable_bytes"].is_u64());
        assert!(
            node["memory"]["total_bytes"].as_u64().unwrap() > 0,
            "total memory must be measured: {body}"
        );
        assert!(node["devices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["kind"] == "cpu"));
        assert!(node["profile"].is_null(), "never benched ⇒ null profile");
        assert_eq!(node["battery"]["draining"], false, "mock probe is on AC");
        assert_eq!(node["sleep_inhibited"], false, "idle, nothing loaded");
        assert_eq!(body["peers"], serde_json::json!([]));
        assert!(body["plan"].is_null());
        assert_eq!(body["requests"], serde_json::json!([]));
        assert_eq!(body["advisor"], serde_json::json!([]));
        teardown(host, host_thread);
    }

    /// A finished generation lands in `requests[]` with its DoneStats —
    /// and the document NEVER carries the prompt text (docs/product.md §1
    /// privacy line, asserted end to end through the observe relay).
    #[tokio::test(flavor = "multi_thread")]
    async fn metrics_reports_finished_requests_without_prompt_text() {
        const SENTINEL: &str = "EXTREMELY-PRIVATE-PROMPT";
        let (base, _notify, host, host_thread, _dir, state) =
            serve_internal_with_state("sekrit").await;
        // Drive the same relay `DaemonBackend::generate` wraps jobs with.
        let (client_tx, mut client_rx) = mpsc::channel(8);
        let wrapped = state.requests.observe(onebrain_api::backend::GenerateJob {
            model: "tinystories-260k".into(),
            prompt: onebrain_api::backend::PromptInput::Raw(SENTINEL.into()),
            params: onebrain_api::backend::GenParams::default(),
            dialect: onebrain_api::backend::ApiDialect::Ollama,
            tx: client_tx,
        });
        wrapped
            .tx
            .send(onebrain_api::backend::TokenEvent::Done(
                onebrain_api::backend::DoneStats {
                    prompt_tokens: 5,
                    completion_tokens: 11,
                    finish: onebrain_api::backend::FinishKind::Length,
                    prefill_ms: 21,
                    decode_ms: 34,
                    ttft_ms: 8,
                    drafted: 0,
                    accepted: 0,
                },
            ))
            .await
            .unwrap();
        // Wait for the relay to forward (recording happens before this).
        assert!(client_rx.recv().await.is_some());

        let resp = reqwest::Client::new()
            .get(format!("{base}/api/internal/metrics"))
            .bearer_auth("sekrit")
            .send()
            .await
            .unwrap();
        let text = resp.text().await.unwrap();
        assert!(
            !text.contains(SENTINEL),
            "prompt text must never appear in metrics: {text}"
        );
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        let requests = body["requests"].as_array().unwrap();
        assert_eq!(requests.len(), 1);
        let entry = &requests[0];
        assert_eq!(entry["model"], "tinystories-260k");
        assert_eq!(entry["dialect"], "ollama");
        assert_eq!(entry["prompt_tokens"], 5);
        assert_eq!(entry["completion_tokens"], 11);
        assert_eq!(entry["prefill_ms"], 21);
        assert_eq!(entry["decode_ms"], 34);
        assert_eq!(entry["ttft_ms"], 8);
        assert_eq!(entry["finish"], "length");
        assert!(entry["timestamp_unix"].as_u64().unwrap() > 0);
        assert!(entry["id"].as_str().unwrap().starts_with("req-"));
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

    /// docs/perf.md §5: `speculative` without a draft (request or config)
    /// is a typed error naming both remedies and a registry pairing —
    /// emitted before any planning or engine work.
    #[tokio::test(flavor = "multi_thread")]
    async fn speculative_load_without_a_draft_errors_with_remedy() {
        let Ok(smoke) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping speculative load test");
            return;
        };
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let resp = reqwest::Client::new()
            .post(format!("{base}/api/internal/load"))
            .bearer_auth("sekrit")
            .json(&serde_json::json!({ "model": smoke, "speculative": true }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.text().await.unwrap();
        let last_line = body.lines().last().expect("stream must not be empty");
        let parsed: serde_json::Value = serde_json::from_str(last_line).unwrap();
        assert_eq!(parsed["status"], "error", "body: {body}");
        let message = parsed["message"].as_str().unwrap();
        assert!(message.contains("--draft"), "remedy missing: {message}");
        assert!(message.contains("draft_model"), "remedy missing: {message}");
        assert!(message.contains("qwen3-0.6b"), "pairing missing: {message}");
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

    /// docs/perf.md §10: `POST /api/internal/perf` reads with an empty body
    /// and overrides per-lever; overrides land in the engine host's runtime
    /// toggles (applied at the next model load).
    #[tokio::test(flavor = "multi_thread")]
    async fn perf_endpoint_reads_and_flips_the_runtime_toggles() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let client = reqwest::Client::new();
        let read = |body: serde_json::Value| {
            let client = client.clone();
            let url = format!("{base}/api/internal/perf");
            async move {
                let resp = client
                    .post(&url)
                    .bearer_auth("sekrit")
                    .json(&body)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(resp.status(), 200);
                resp.json::<serde_json::Value>().await.unwrap()
            }
        };
        // Empty body: pure read of the config-time defaults.
        let current = read(serde_json::json!({})).await;
        assert_eq!(current["prefill_overlap"], true);
        assert_eq!(current["kv_reuse"], true);
        // Flip one lever; the other keeps its value.
        let flipped = read(serde_json::json!({ "prefill_overlap": false })).await;
        assert_eq!(flipped["prefill_overlap"], false);
        assert_eq!(flipped["kv_reuse"], true);
        assert_eq!(host.perf_toggles(), (false, true));
        // Flip both back and forth; the host handle sees every change.
        let both = read(serde_json::json!({ "prefill_overlap": true, "kv_reuse": false })).await;
        assert_eq!(both["prefill_overlap"], true);
        assert_eq!(both["kv_reuse"], false);
        assert_eq!(host.perf_toggles(), (true, false));
        teardown(host, host_thread);
    }

    /// docs/perf.md §10: `POST /api/internal/bench/peers` with nothing
    /// connected answers an empty peers list, not an error.
    #[tokio::test(flavor = "multi_thread")]
    async fn bench_peers_with_no_peers_is_an_empty_list() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let resp = reqwest::Client::new()
            .post(format!("{base}/api/internal/bench/peers"))
            .bearer_auth("sekrit")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["peers"], serde_json::json!([]));
        teardown(host, host_thread);
    }

    /// Seed one completed cache entry (model file + M1-shaped manifest).
    fn seed_cache_entry(cache_root: &std::path::Path, id: &str, bytes: &[u8]) {
        let dir = cache_root.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.gguf"), bytes).unwrap();
        let manifest = onebrain_models::download::Manifest {
            url: format!("https://example.invalid/{id}.gguf"),
            size_bytes: bytes.len() as u64,
            blake3: "test".to_string(),
        };
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pin_and_unpin_flip_the_entry_state() {
        let (base, _notify, host, host_thread, dir) = serve_internal("sekrit").await;
        let cache_root = dir.path().join("models");
        seed_cache_entry(&cache_root, "pin-me", b"weights");

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/api/internal/models/pin"))
            .bearer_auth("sekrit")
            .json(&serde_json::json!({ "model": "pin-me" }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["pinned"], true);
        let listed = onebrain_models::cache::list(&cache_root).unwrap();
        assert!(listed[0].pinned, "pin must persist to the entry manifest");

        let resp = client
            .post(format!("{base}/api/internal/models/unpin"))
            .bearer_auth("sekrit")
            .json(&serde_json::json!({ "model": "pin-me" }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["pinned"], false);
        let listed = onebrain_models::cache::list(&cache_root).unwrap();
        assert!(!listed[0].pinned);
        teardown(host, host_thread);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pin_of_unknown_model_is_a_404_with_the_error_envelope() {
        let (base, _notify, host, host_thread, _dir) = serve_internal("sekrit").await;
        let resp = reqwest::Client::new()
            .post(format!("{base}/api/internal/models/pin"))
            .bearer_auth("sekrit")
            .json(&serde_json::json!({ "model": "ghost-model" }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("ghost-model"));
        // Traversal ids are a clean 400, never a filesystem touch.
        let resp = reqwest::Client::new()
            .post(format!("{base}/api/internal/models/pin"))
            .bearer_auth("sekrit")
            .json(&serde_json::json!({ "model": "../evil" }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        teardown(host, host_thread);
    }

    /// The `ls`/models listing payload gains pinned, last_used_unix and the
    /// part count (docs/logistics.md; the CLI renders these columns).
    #[tokio::test(flavor = "multi_thread")]
    async fn models_listing_reports_pin_lru_and_parts() {
        let dir = tempfile::tempdir().unwrap();
        let cache_root = dir.path().join("models");
        seed_cache_entry(&cache_root, "listed-model", b"0123456789");
        onebrain_models::cache::set_pinned(&cache_root, "listed-model", true).unwrap();
        onebrain_models::cache::touch(&cache_root, "listed-model").unwrap();

        let (host, host_thread) = EngineHost::spawn(None, crate::engine_host::HostPerf::default());
        let mesh = spawn_test_mesh(dir.path()).await;
        let (sup_tx, _sup_rx) = crate::supervisor::channel();
        let backend = crate::engine_host::DaemonBackend::new(
            host.clone(),
            cache_root.clone(),
            sup_tx,
            mesh.clone(),
            0,
            4,
            8,
            crate::metrics::RequestLog::new(),
        );
        let models = tokio::task::spawn_blocking(move || {
            onebrain_api::backend::EngineBackend::models(&backend)
        })
        .await
        .unwrap();
        let m = models
            .iter()
            .find(|m| m.name == "listed-model")
            .expect("seeded model listed");
        assert_eq!(m.details.get("pinned").map(String::as_str), Some("true"));
        assert_eq!(m.details.get("parts").map(String::as_str), Some("1"));
        let last_used: u64 = m.details.get("last_used_unix").unwrap().parse().unwrap();
        assert!(last_used > 0, "touch must be visible in the listing");
        let _ = mesh.shutdown().await;
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
