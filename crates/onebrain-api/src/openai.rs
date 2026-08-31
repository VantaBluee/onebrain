//! OpenAI-compatible dialect (`/v1/*`).
//!
//! Implements `POST /v1/chat/completions`, `POST /v1/completions`,
//! `GET /v1/models`, and a 501 placeholder for `POST /v1/embeddings`,
//! per the M1 contract in `docs/internal-api.md`: SSE (`data: {json}\n\n`
//! terminated by `data: [DONE]\n\n`) when `"stream": true`, plain JSON
//! otherwise, `usage` filled from [`DoneStats`].

use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::backend::{
    ApiDialect, DoneStats, FinishKind, GenParams, GenerateJob, PromptInput, TokenEvent,
};
use crate::types::ChatMessage;
use crate::{ApiError, ApiState};

/// Channel capacity between the backend and the HTTP layer for one job.
const EVENT_CHANNEL_CAPACITY: usize = 64;

pub fn routes() -> Router<ApiState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
}

// ---------------------------------------------------------------------------
// Request wire shapes
// ---------------------------------------------------------------------------

/// OpenAI `stop`: a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSpec {
    One(String),
    Many(Vec<String>),
}

impl From<StopSpec> for Vec<String> {
    fn from(spec: StopSpec) -> Vec<String> {
        match spec {
            StopSpec::One(s) => vec![s],
            StopSpec::Many(v) => v,
        }
    }
}

/// OpenAI `prompt`: a single string or an array of strings (we accept one).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PromptSpec {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<StopSpec>,
    seed: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: PromptSpec,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<StopSpec>,
    seed: Option<u32>,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn chat_completions(
    State(state): State<ApiState>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let req: ChatCompletionRequest = parse_body(body)?;
    if req.messages.is_empty() {
        return Err(ApiError::BadRequest(
            "`messages` must contain at least one message; add a user message and retry".into(),
        ));
    }
    let params = build_params(
        req.max_completion_tokens.or(req.max_tokens),
        req.temperature,
        req.top_p,
        req.seed,
        req.stop,
    );
    let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    state.backend.generate(GenerateJob {
        model: req.model.clone(),
        prompt: PromptInput::Chat(req.messages),
        params,
        dialect: ApiDialect::Openai,
        tx,
    })?;

    let id = next_id("chatcmpl");
    let created = unix_now();
    if req.stream {
        return Ok(sse_response(rx, id, created, req.model, Dialect::Chat));
    }
    let (content, stats) = collect(rx).await?;
    Ok(Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "logprobs": null,
            "finish_reason": finish_reason(stats.finish),
        }],
        "usage": usage(&stats),
    }))
    .into_response())
}

async fn completions(
    State(state): State<ApiState>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let req: CompletionRequest = parse_body(body)?;
    let prompt = match req.prompt {
        PromptSpec::One(s) => s,
        PromptSpec::Many(mut v) if v.len() == 1 => v.remove(0),
        PromptSpec::Many(v) => {
            return Err(ApiError::BadRequest(format!(
                "`prompt` arrays with {} entries are not supported; send exactly one prompt per request",
                v.len()
            )));
        }
    };
    let params = build_params(
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.seed,
        req.stop,
    );
    let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    state.backend.generate(GenerateJob {
        model: req.model.clone(),
        prompt: PromptInput::Raw(prompt),
        params,
        dialect: ApiDialect::Openai,
        tx,
    })?;

    let id = next_id("cmpl");
    let created = unix_now();
    if req.stream {
        return Ok(sse_response(rx, id, created, req.model, Dialect::Text));
    }
    let (text, stats) = collect(rx).await?;
    Ok(Json(json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "text": text,
            "logprobs": null,
            "finish_reason": finish_reason(stats.finish),
        }],
        "usage": usage(&stats),
    }))
    .into_response())
}

async fn models(State(state): State<ApiState>) -> Json<Value> {
    let data: Vec<Value> = state
        .backend
        .models()
        .into_iter()
        .map(|m| {
            json!({
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": "onebrain",
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

async fn embeddings() -> ApiError {
    ApiError::NotImplemented("embeddings arrive later in M1")
}

// ---------------------------------------------------------------------------
// Shared machinery
// ---------------------------------------------------------------------------

fn parse_body<T: serde::de::DeserializeOwned>(body: Value) -> Result<T, ApiError> {
    serde_json::from_value(body).map_err(|e| {
        ApiError::BadRequest(format!(
            "could not parse the request body ({e}); check required fields and types against the OpenAI API reference"
        ))
    })
}

/// Map request fields onto [`GenParams`], defaulting absent fields from
/// [`GenParams::default`]. Client-supplied values (including OpenAI's own
/// default `temperature` of 1.0, which many SDKs send explicitly) pass
/// through untouched.
fn build_params(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u32>,
    stop: Option<StopSpec>,
) -> GenParams {
    let defaults = GenParams::default();
    GenParams {
        max_tokens: max_tokens.unwrap_or(defaults.max_tokens),
        temperature: temperature.unwrap_or(defaults.temperature),
        top_p: top_p.unwrap_or(defaults.top_p),
        top_k: defaults.top_k,
        seed,
        stop: stop.map(Vec::from).unwrap_or_default(),
    }
}

fn finish_reason(kind: FinishKind) -> &'static str {
    match kind {
        // OpenAI has no "abort" wording; a cancelled generation still ended
        // at a point the model chose nothing past, so "stop" is the closest
        // truthful mapping for unmodified clients.
        FinishKind::Stop | FinishKind::Abort => "stop",
        FinishKind::Length => "length",
    }
}

fn usage(stats: &DoneStats) -> Value {
    json!({
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "total_tokens": stats.prompt_tokens + stats.completion_tokens,
    })
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn next_id(prefix: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!("{prefix}-{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

/// Drain a generation to completion for a non-streaming response.
async fn collect(mut rx: mpsc::Receiver<TokenEvent>) -> Result<(String, DoneStats), ApiError> {
    let mut text = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            TokenEvent::Token(piece) => text.push_str(&piece),
            TokenEvent::Done(stats) => return Ok((text, stats)),
            TokenEvent::Error(message) => return Err(ApiError::Internal(message)),
        }
    }
    Err(ApiError::Internal(
        "the generation stream ended without a result; retry the request".into(),
    ))
}

#[derive(Clone, Copy)]
enum Dialect {
    Chat,
    Text,
}

enum ChunkBody {
    Role,
    Content(String),
    Final,
}

fn chunk(
    dialect: Dialect,
    id: &str,
    created: u64,
    model: &str,
    body: ChunkBody,
    finish: Option<&str>,
) -> Value {
    match dialect {
        Dialect::Chat => {
            let delta = match body {
                ChunkBody::Role => json!({ "role": "assistant" }),
                ChunkBody::Content(piece) => json!({ "content": piece }),
                ChunkBody::Final => json!({}),
            };
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
            })
        }
        Dialect::Text => {
            let text = match body {
                ChunkBody::Content(piece) => piece,
                ChunkBody::Role | ChunkBody::Final => String::new(),
            };
            json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{ "index": 0, "text": text, "finish_reason": finish }],
            })
        }
    }
}

fn error_event_json(message: &str) -> Value {
    json!({ "error": { "message": message, "type": "api_error" } })
}

async fn emit(tx: &mpsc::Sender<Event>, payload: &Value) -> bool {
    tx.send(Event::default().data(payload.to_string()))
        .await
        .is_ok()
}

/// Turn a [`TokenEvent`] stream into an OpenAI SSE response: an initial role
/// delta (chat only), one chunk per token piece, a terminal chunk carrying
/// `finish_reason`, then `data: [DONE]`. A mid-stream backend error becomes
/// an error-JSON event followed by `[DONE]`.
fn sse_response(
    mut rx: mpsc::Receiver<TokenEvent>,
    id: String,
    created: u64,
    model: String,
    dialect: Dialect,
) -> Response {
    let (etx, erx) = mpsc::channel::<Event>(EVENT_CHANNEL_CAPACITY);
    tokio::spawn(async move {
        if matches!(dialect, Dialect::Chat) {
            let first = chunk(dialect, &id, created, &model, ChunkBody::Role, None);
            if !emit(&etx, &first).await {
                return; // client went away
            }
        }
        let mut terminated = false;
        while let Some(event) = rx.recv().await {
            match event {
                TokenEvent::Token(piece) => {
                    let c = chunk(
                        dialect,
                        &id,
                        created,
                        &model,
                        ChunkBody::Content(piece),
                        None,
                    );
                    if !emit(&etx, &c).await {
                        return;
                    }
                }
                TokenEvent::Done(stats) => {
                    let c = chunk(
                        dialect,
                        &id,
                        created,
                        &model,
                        ChunkBody::Final,
                        Some(finish_reason(stats.finish)),
                    );
                    let _ = emit(&etx, &c).await;
                    terminated = true;
                    break;
                }
                TokenEvent::Error(message) => {
                    let _ = emit(&etx, &error_event_json(&message)).await;
                    terminated = true;
                    break;
                }
            }
        }
        if !terminated {
            let payload =
                error_event_json("the generation stream ended without a result; retry the request");
            let _ = emit(&etx, &payload).await;
        }
        let _ = etx.send(Event::default().data("[DONE]")).await;
    });
    Sse::new(ReceiverStream::new(erx).map(Ok::<_, std::convert::Infallible>)).into_response()
}
