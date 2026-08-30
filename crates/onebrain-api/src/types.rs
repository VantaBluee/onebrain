//! Types shared by both API dialects, and the error → HTTP mapping.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::ApiError;

/// A chat message in either dialect's wire shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ApiError {
    pub fn status(&self) -> StatusCode {
        match self {
            ApiError::ModelNotLoaded(_) | ApiError::NoModel => StatusCode::NOT_FOUND,
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::ShuttingDown => StatusCode::SERVICE_UNAVAILABLE,
            // Admission control (docs/perf.md §6): the typed 429-equivalent.
            ApiError::Overloaded { .. } => StatusCode::TOO_MANY_REQUESTS,
            ApiError::NotImplemented(_) => StatusCode::NOT_IMPLEMENTED,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    /// OpenAI-style error envelope; Ollama clients read `error` too.
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": match self {
                    ApiError::BadRequest(_) => "invalid_request_error",
                    ApiError::ModelNotLoaded(_) | ApiError::NoModel => "not_found_error",
                    // OpenAI's own wording for 429-class rejections.
                    ApiError::Overloaded { .. } => "rate_limit_error",
                    _ => "api_error",
                }
            },
            // Ollama dialect compatibility: a top-level error string.
            "error_message": self.to_string(),
        });
        (self.status(), axum::Json(body)).into_response()
    }
}
