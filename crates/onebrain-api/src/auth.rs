//! Bearer auth: required everywhere, with a configurable exemption for
//! loopback clients (on by default). There is no exemption for non-loopback
//! clients and none can be configured (product spec §1.3).

use std::net::SocketAddr;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::ApiState;

/// Auth policy + the expected token.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// The bearer token (hex string generated at daemon init).
    pub token: String,
    /// When true (default), loopback connections skip the token check.
    pub localhost_exempt: bool,
}

impl AuthConfig {
    /// Constant-time comparison; length differences fail without early exit
    /// on content.
    pub fn token_matches(&self, presented: &str) -> bool {
        let a = self.token.as_bytes();
        let b = presented.as_bytes();
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
    }
}

/// Axum middleware enforcing the policy above.
pub async fn require_bearer(
    State(state): State<ApiState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    if state.auth.localhost_exempt && peer.ip().is_loopback() {
        return next.run(request).await;
    }
    let presented = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match presented {
        Some(tok) if state.auth.token_matches(tok) => next.run(request).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({
                "error": {
                    "message": "missing or invalid bearer token; `onebrain status` prints it",
                    "type": "authentication_error"
                }
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_compare_is_exact() {
        let auth = AuthConfig {
            token: "abc123".into(),
            localhost_exempt: true,
        };
        assert!(auth.token_matches("abc123"));
        assert!(!auth.token_matches("abc124"));
        assert!(!auth.token_matches("abc12"));
        assert!(!auth.token_matches(""));
    }
}
