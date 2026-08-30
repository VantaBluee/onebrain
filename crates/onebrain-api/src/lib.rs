//! HTTP API gateway: OpenAI (`/v1/*`) and Ollama (`/api/*`) dialects over
//! one internal backend abstraction, plus bearer auth (localhost exempt by
//! default; there is no way to disable auth for non-loopback clients).
//!
//! The gateway is engine-agnostic: everything inference-shaped goes through
//! [`backend::EngineBackend`], implemented by the daemon (real engine) and
//! by [`backend::testing::FakeBackend`] in the conformance tests.

pub mod auth;
pub mod backend;
pub mod ollama;
pub mod openai;
pub mod types;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;

use crate::auth::AuthConfig;
use crate::backend::EngineBackend;

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("model {0:?} is not loaded; run `onebrain run {0}` first")]
    ModelNotLoaded(String),
    #[error("no model is loaded; run `onebrain run <model>` first")]
    NoModel,
    #[error("the engine is busy shutting down")]
    ShuttingDown,
    /// Admission control (docs/perf.md §6): the concurrent set AND the wait
    /// queue are full. Maps to HTTP 429 with a remedy — never an unbounded
    /// queue.
    #[error(
        "this node is at capacity: {max_concurrent} generations are running and \
         {queue_depth} more are queued; retry shortly, or raise [perf] \
         max_concurrent_requests / queue_depth in config.toml"
    )]
    Overloaded {
        max_concurrent: u32,
        queue_depth: u32,
    },
    #[error("request validation failed: {0}")]
    BadRequest(String),
    #[error("this endpoint is not implemented yet: {0}")]
    NotImplemented(&'static str),
    #[error("internal error: {0}")]
    Internal(String),
}

/// Shared state for all routes.
#[derive(Clone)]
pub struct ApiState {
    pub backend: Arc<dyn EngineBackend>,
    pub auth: Arc<AuthConfig>,
    /// Product version string surfaced by version endpoints.
    pub product_version: &'static str,
}

/// Build the public router (both dialects) with auth applied. The daemon
/// mounts its internal control endpoints separately.
pub fn router(state: ApiState) -> Router {
    Router::new()
        .merge(openai::routes())
        .merge(ollama::routes())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::require_bearer,
        ))
        .with_state(state)
}

/// Serve `app` on `addr` until `shutdown` resolves. Binds loopback or
/// non-loopback alike — callers choose; auth policy is enforced per-request.
pub async fn serve(
    app: Router,
    addr: SocketAddr,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await
}
