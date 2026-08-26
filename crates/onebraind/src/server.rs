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
}

/// Build the internal router with its always-on token middleware.
pub fn internal_router(state: Arc<InternalState>) -> Router {
    Router::new()
        .route("/api/internal/status", get(status))
        .route("/api/internal/load", post(load))
        .route("/api/internal/shutdown", post(shutdown))
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
    Json(serde_json::json!({
        "version": state.product_version,
        "engine_build": onebrain_engine::engine_build_hash().0,
        "port": state.port,
        "uptime_secs": state.started.elapsed().as_secs(),
        "model": model,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_host::EngineHost;

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
