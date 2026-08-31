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
            // Honest "not on this deployment shape" — the endpoint exists,
            // the current (distributed) load cannot serve it.
            ApiError::EmbeddingsDistributed(_) => StatusCode::NOT_IMPLEMENTED,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// Parse an embeddings `input` field (shared by `/v1/embeddings` and
/// `/api/embed`): a single string or an array of strings, validated
/// non-empty. OpenAI also accepts pre-tokenized integer arrays; OneBrain
/// tokenizes with the loaded model's own tokenizer, so token arrays are
/// refused with a clear remedy rather than mis-decoded.
pub(crate) fn embed_input_texts(input: &serde_json::Value) -> Result<Vec<String>, ApiError> {
    use serde_json::Value;
    let texts = match input {
        Value::String(s) => vec![s.clone()],
        Value::Array(items) => {
            if items.is_empty() {
                return Err(ApiError::BadRequest(
                    "`input` must contain at least one string".into(),
                ));
            }
            let mut texts = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::String(s) => texts.push(s.clone()),
                    Value::Number(_) | Value::Array(_) => {
                        return Err(ApiError::BadRequest(
                            "token-array `input` is not supported; send the original text \
                             (OneBrain tokenizes with the loaded model's own tokenizer)"
                                .into(),
                        ));
                    }
                    other => {
                        return Err(ApiError::BadRequest(format!(
                            "`input` array entries must be strings, got {other}"
                        )));
                    }
                }
            }
            texts
        }
        Value::Null => {
            return Err(ApiError::BadRequest(
                "`input` is required: a string or an array of strings".into(),
            ));
        }
        other => {
            return Err(ApiError::BadRequest(format!(
                "`input` must be a string or an array of strings, got {other}"
            )));
        }
    };
    if texts.iter().any(String::is_empty) {
        return Err(ApiError::BadRequest(
            "`input` strings must be non-empty".into(),
        ));
    }
    Ok(texts)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn embed_input_accepts_string_and_string_arrays() {
        assert_eq!(embed_input_texts(&json!("hi")).unwrap(), vec!["hi"]);
        assert_eq!(
            embed_input_texts(&json!(["a", "b"])).unwrap(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn embed_input_rejects_token_arrays_with_a_remedy() {
        for bad in [json!([1, 2, 3]), json!([[1, 2], [3]])] {
            match embed_input_texts(&bad) {
                Err(ApiError::BadRequest(msg)) => {
                    assert!(msg.contains("token-array"), "{msg}");
                    assert!(msg.contains("original text"), "remedy missing: {msg}");
                }
                other => panic!("expected BadRequest, got {other:?}"),
            }
        }
    }

    #[test]
    fn embed_input_rejects_empty_and_wrong_shapes() {
        for bad in [
            json!([]),
            json!(""),
            json!(["ok", ""]),
            json!(7),
            json!(null),
        ] {
            assert!(
                matches!(embed_input_texts(&bad), Err(ApiError::BadRequest(_))),
                "expected BadRequest for {bad}"
            );
        }
    }

    #[test]
    fn embeddings_distributed_maps_to_501_and_names_solo() {
        let err = ApiError::EmbeddingsDistributed("qwen3-4b".into());
        assert_eq!(err.status(), StatusCode::NOT_IMPLEMENTED);
        let msg = err.to_string();
        assert!(msg.contains("solo"), "remedy must name solo: {msg}");
        assert!(msg.contains("qwen3-4b"), "must name the model: {msg}");
    }
}
