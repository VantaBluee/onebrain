//! The internal control API (`/api/internal/*`, docs/internal-api.md).
//!
//! These endpoints are ALWAYS token-authenticated: the public gateway's
//! localhost exemption deliberately does not apply here, so a random local
//! process cannot stop the daemon or swap models without reading the token
//! file (same-user filesystem access).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use onebrain_api::auth::AuthConfig;
use onebrain_mesh::{MeshError, MeshHandle, PairEvent, PairTarget, PeerState};
use tokio::sync::{mpsc, oneshot, Notify};
use tokio_stream::wrappers::ReceiverStream;

use crate::engine_host::{EngineHost, HostMsg, LoadProgress};

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
}

/// Build the internal router with its always-on token middleware.
pub fn internal_router(state: Arc<InternalState>) -> Router {
    Router::new()
        .route("/api/internal/status", get(status))
        .route("/api/internal/load", post(load))
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
    }))
}

#[derive(Debug, serde::Deserialize)]
struct LoadBody {
    model: String,
}

/// `POST /api/internal/load` — NDJSON stream of download/load progress with
/// a terminal `ready` or `error` line (contract).
async fn load(State(state): State<Arc<InternalState>>, Json(body): Json<LoadBody>) -> Response {
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<LoadProgress>();
    let (resp_tx, resp_rx) = oneshot::channel();
    if state
        .host
        .send(HostMsg::Load {
            reference: body.model,
            cache_root: state.cache_root.clone(),
            ctx_len: state.ctx_len,
            progress: progress_tx,
            resp: resp_tx,
        })
        .is_err()
    {
        return onebrain_api::ApiError::ShuttingDown.into_response();
    }

    let (line_tx, line_rx) = mpsc::channel::<Result<String, std::convert::Infallible>>(32);
    tokio::spawn(async move {
        while let Some(event) = progress_rx.recv().await {
            let line = match event {
                LoadProgress::Downloading { completed, total } => serde_json::json!({
                    "status": "downloading", "completed": completed, "total": total,
                }),
                LoadProgress::Loading => serde_json::json!({ "status": "loading" }),
            };
            if line_tx.send(Ok(format!("{line}\n"))).await.is_err() {
                // Client went away. Dropping progress_rx/resp_rx does not
                // cancel the load: swapping the daemon's model is a daemon-
                // level state change, not tied to this request's lifetime.
                return;
            }
        }
        let terminal = match resp_rx.await {
            Ok(Ok(model)) => serde_json::json!({ "status": "ready", "model": model }),
            Ok(Err(message)) => serde_json::json!({ "status": "error", "message": message }),
            Err(_) => serde_json::json!({
                "status": "error",
                "message": "the engine host exited unexpectedly; restart with `onebrain up`",
            }),
        };
        let _ = line_tx.send(Ok(format!("{terminal}\n"))).await;
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(axum::body::Body::from_stream(ReceiverStream::new(line_rx)))
        .expect("static response construction cannot fail")
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
