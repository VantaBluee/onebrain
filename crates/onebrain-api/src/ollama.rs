//! Ollama-compatible dialect (`/api/*`). Placeholder — implemented by the
//! M1 dialect task.

use axum::Router;

use crate::ApiState;

pub fn routes() -> Router<ApiState> {
    Router::new()
}
