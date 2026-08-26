//! HTTP API gateway.
//!
//! M1 implements the axum server with two dialects — OpenAI (`/v1/*`) and
//! Ollama (`/api/*`) — SSE streaming, and bearer auth (localhost exempt by
//! default, §7). M0 ships the request/response vocabulary so the daemon and
//! CLI can compile against stable types.

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("the HTTP API is not implemented yet (arrives in milestone M1)")]
    NotImplemented,
}

/// OpenAI-dialect chat message (also the internal lingua franca; the Ollama
/// dialect maps onto it).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Minimal chat-completion request shape shared by both dialects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_shape_deserializes() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"model":"qwen3-4b","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        )
        .unwrap();
        assert!(req.stream);
        assert_eq!(req.messages[0].role, "user");
    }
}
