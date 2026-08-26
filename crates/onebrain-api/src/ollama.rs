//! Ollama-compatible dialect (`/api/*`).
//!
//! Wire notes (per `docs/internal-api.md`): Ollama streams **NDJSON**
//! (`application/x-ndjson`, one JSON object per line) — not SSE — and
//! streaming is the **default**; `"stream": false` must be honored.
//! `/api/pull` streams the same download-progress shape as internal load.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::backend::{
    DoneStats, FinishKind, GenParams, GenerateJob, ModelSummary, PromptInput, PullEvent, TokenEvent,
};
use crate::types::ChatMessage;
use crate::{ApiError, ApiState};

pub fn routes() -> Router<ApiState> {
    Router::new()
        .route("/api/generate", post(generate))
        .route("/api/chat", post(chat))
        .route("/api/tags", get(tags))
        .route("/api/show", post(show))
        .route("/api/ps", get(ps))
        .route("/api/pull", post(pull))
        .route("/api/version", get(version))
}

// ---------------------------------------------------------------------------
// Request bodies
// ---------------------------------------------------------------------------

/// Ollama `options` map — the subset OneBrain honors. Unknown keys are
/// ignored so unmodified clients keep working.
#[derive(Debug, Default, Deserialize)]
struct Options {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<i64>,
    num_predict: Option<i64>,
    stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    model: String,
    #[serde(default)]
    prompt: String,
    stream: Option<bool>,
    options: Option<Options>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: String,
    #[serde(default)]
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
    options: Option<Options>,
}

#[derive(Debug, Deserialize)]
struct ShowRequest {
    model: String,
}

#[derive(Debug, Deserialize)]
struct PullRequest {
    model: String,
    stream: Option<bool>,
}

fn gen_params(options: Option<Options>) -> GenParams {
    let mut params = GenParams::default();
    if let Some(o) = options {
        if let Some(v) = o.temperature {
            params.temperature = v;
        }
        if let Some(v) = o.top_p {
            params.top_p = v;
        }
        if let Some(v) = o.top_k {
            params.top_k = v;
        }
        if let Some(v) = o.seed {
            if v >= 0 {
                params.seed = Some(v as u32);
            }
        }
        // Ollama semantics: num_predict <= 0 means "no explicit limit";
        // OneBrain keeps its default budget in that case.
        if let Some(v) = o.num_predict {
            if v > 0 {
                params.max_tokens = v.min(u32::MAX as i64) as u32;
            }
        }
        if let Some(v) = o.stop {
            params.stop = v;
        }
    }
    params
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn generate(
    State(state): State<ApiState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, ApiError> {
    let (tx, rx) = mpsc::channel(64);
    state.backend.generate(GenerateJob {
        model: req.model.clone(),
        prompt: PromptInput::Raw(req.prompt),
        params: gen_params(req.options),
        tx,
    })?;
    if req.stream.unwrap_or(true) {
        Ok(stream_ndjson(rx, req.model, Kind::Generate))
    } else {
        collect(rx, req.model, Kind::Generate).await
    }
}

async fn chat(
    State(state): State<ApiState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let (tx, rx) = mpsc::channel(64);
    state.backend.generate(GenerateJob {
        model: req.model.clone(),
        prompt: PromptInput::Chat(req.messages),
        params: gen_params(req.options),
        tx,
    })?;
    if req.stream.unwrap_or(true) {
        Ok(stream_ndjson(rx, req.model, Kind::Chat))
    } else {
        collect(rx, req.model, Kind::Chat).await
    }
}

async fn tags(State(state): State<ApiState>) -> Json<Value> {
    let models: Vec<Value> = state
        .backend
        .models()
        .into_iter()
        .map(model_entry)
        .collect();
    Json(json!({ "models": models }))
}

async fn ps(State(state): State<ApiState>) -> Json<Value> {
    let models: Vec<Value> = state
        .backend
        .models()
        .into_iter()
        .filter(|m| m.loaded)
        .map(model_entry)
        .collect();
    Json(json!({ "models": models }))
}

async fn show(
    State(state): State<ApiState>,
    Json(req): Json<ShowRequest>,
) -> Result<Json<Value>, ApiError> {
    let summary = state
        .backend
        .models()
        .into_iter()
        .find(|m| m.name == req.model)
        .ok_or(ApiError::ModelNotLoaded(req.model))?;
    Ok(Json(json!({
        "details": summary.details,
        "model_info": summary.details,
    })))
}

async fn pull(
    State(state): State<ApiState>,
    Json(req): Json<PullRequest>,
) -> Result<Response, ApiError> {
    let (tx, mut rx) = mpsc::channel(64);
    state.backend.pull(req.model, tx)?;
    if req.stream.unwrap_or(true) {
        let stream = futures::stream::unfold((rx, false), |(mut rx, finished)| async move {
            if finished {
                return None;
            }
            let event = rx.recv().await?;
            let terminal = matches!(event, PullEvent::Done | PullEvent::Error { .. });
            Some((
                Ok::<_, Infallible>(to_line(pull_json(&event))),
                (rx, terminal),
            ))
        });
        Ok(ndjson_response(stream))
    } else {
        while let Some(event) = rx.recv().await {
            match event {
                PullEvent::Downloading { .. } => continue,
                PullEvent::Done => return Ok(Json(json!({ "status": "success" })).into_response()),
                PullEvent::Error { message } => return Err(ApiError::Internal(message)),
            }
        }
        Err(ApiError::Internal(
            "model download ended without a result; retry `onebrain pull` or check connectivity"
                .to_string(),
        ))
    }
}

async fn version(State(state): State<ApiState>) -> Json<Value> {
    Json(json!({ "version": state.product_version }))
}

// ---------------------------------------------------------------------------
// Wire shapes
// ---------------------------------------------------------------------------

/// Which endpoint's line shape to emit.
enum Kind {
    Generate,
    Chat,
}

fn model_entry(m: ModelSummary) -> Value {
    json!({
        "name": m.name,
        "model": m.name,
        "size": m.size_bytes,
        "details": m.details,
    })
}

/// One streamed progress line for a token piece.
fn piece_json(kind: &Kind, model: &str, piece: &str) -> Value {
    match kind {
        Kind::Generate => json!({
            "model": model,
            "created_at": created_at(),
            "response": piece,
            "done": false,
        }),
        Kind::Chat => json!({
            "model": model,
            "created_at": created_at(),
            "message": { "role": "assistant", "content": piece },
            "done": false,
        }),
    }
}

/// The terminal object: `content` is the full text for non-streaming
/// responses and `""` for the final streamed line.
fn done_json(kind: &Kind, model: &str, content: &str, stats: &DoneStats) -> Value {
    let mut value = match kind {
        Kind::Generate => json!({
            "model": model,
            "created_at": created_at(),
            "response": content,
        }),
        Kind::Chat => json!({
            "model": model,
            "created_at": created_at(),
            "message": { "role": "assistant", "content": content },
        }),
    };
    let obj = value
        .as_object_mut()
        .expect("done_json builds a JSON object");
    obj.insert("done".to_string(), json!(true));
    obj.insert("done_reason".to_string(), json!(done_reason(stats.finish)));
    obj.insert("prompt_eval_count".to_string(), json!(stats.prompt_tokens));
    obj.insert("eval_count".to_string(), json!(stats.completion_tokens));
    value
}

fn done_reason(finish: FinishKind) -> &'static str {
    match finish {
        FinishKind::Stop => "stop",
        FinishKind::Length => "length",
        // Not part of the contract's normal vocabulary; only reachable when
        // a generation is cancelled server-side.
        FinishKind::Abort => "abort",
    }
}

fn pull_json(event: &PullEvent) -> Value {
    match event {
        PullEvent::Downloading { completed, total } => json!({
            "status": "pulling",
            "completed": completed,
            "total": total,
        }),
        PullEvent::Done => json!({ "status": "success" }),
        PullEvent::Error { message } => json!({ "status": "error", "error": message }),
    }
}

// ---------------------------------------------------------------------------
// NDJSON plumbing
// ---------------------------------------------------------------------------

fn to_line(value: Value) -> String {
    let mut line = value.to_string();
    line.push('\n');
    line
}

fn ndjson_response<S>(stream: S) -> Response
where
    S: futures::Stream<Item = Result<String, Infallible>> + Send + 'static,
{
    (
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        Body::from_stream(stream),
    )
        .into_response()
}

/// Turn a generation event stream into an NDJSON body: one line per token
/// piece, a terminal line for `Done`, and `{"error": …}` if the backend
/// fails after streaming began.
fn stream_ndjson(rx: mpsc::Receiver<TokenEvent>, model: String, kind: Kind) -> Response {
    let stream = futures::stream::unfold(
        (rx, model, kind, false),
        |(mut rx, model, kind, finished)| async move {
            if finished {
                return None;
            }
            let event = rx.recv().await?;
            let (line, finished) = match event {
                TokenEvent::Token(piece) => (piece_json(&kind, &model, &piece), false),
                TokenEvent::Done(stats) => (done_json(&kind, &model, "", &stats), true),
                TokenEvent::Error(message) => (json!({ "error": message }), true),
            };
            Some((
                Ok::<_, Infallible>(to_line(line)),
                (rx, model, kind, finished),
            ))
        },
    );
    ndjson_response(stream)
}

/// Non-streaming path: accumulate the whole response, answer with one object.
async fn collect(
    mut rx: mpsc::Receiver<TokenEvent>,
    model: String,
    kind: Kind,
) -> Result<Response, ApiError> {
    let mut full = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            TokenEvent::Token(piece) => full.push_str(&piece),
            TokenEvent::Done(stats) => {
                return Ok(Json(done_json(&kind, &model, &full, &stats)).into_response());
            }
            TokenEvent::Error(message) => return Err(ApiError::Internal(message)),
        }
    }
    Err(ApiError::Internal(
        "generation ended without a result; retry the request or check `onebrain status`"
            .to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Timestamps (RFC 3339, UTC) without a date-time dependency
// ---------------------------------------------------------------------------

fn created_at() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    rfc3339_utc(now.as_secs(), now.subsec_nanos())
}

fn rfc3339_utc(unix_secs: u64, nanos: u32) -> String {
    let days = (unix_secs / 86_400) as i64;
    let rem = unix_secs % 86_400;
    let (hour, min, sec) = (rem / 3_600, (rem % 3_600) / 60, rem % 60);
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}.{nanos:09}Z")
}

/// Days-since-epoch → (year, month, day), Howard Hinnant's `civil_from_days`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_epoch() {
        assert_eq!(rfc3339_utc(0, 0), "1970-01-01T00:00:00.000000000Z");
    }

    #[test]
    fn rfc3339_known_instant() {
        // 1_700_000_000 = 2023-11-14T22:13:20Z.
        assert_eq!(
            rfc3339_utc(1_700_000_000, 123_456_789),
            "2023-11-14T22:13:20.123456789Z"
        );
    }

    #[test]
    fn civil_dates_roll_over_correctly() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(31), (1970, 2, 1));
        // 2000-02-29 (leap year), day 11_016 from the epoch.
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
    }

    #[test]
    fn num_predict_and_stop_map_into_params() {
        let params = gen_params(Some(Options {
            temperature: Some(0.1),
            top_p: Some(0.5),
            top_k: Some(7),
            seed: Some(42),
            num_predict: Some(3),
            stop: Some(vec!["END".to_string()]),
        }));
        assert_eq!(params.max_tokens, 3);
        assert_eq!(params.temperature, 0.1);
        assert_eq!(params.top_p, 0.5);
        assert_eq!(params.top_k, 7);
        assert_eq!(params.seed, Some(42));
        assert_eq!(params.stop, vec!["END".to_string()]);
    }

    #[test]
    fn num_predict_non_positive_keeps_default_budget() {
        let defaults = GenParams::default();
        let params = gen_params(Some(Options {
            num_predict: Some(-1),
            ..Options::default()
        }));
        assert_eq!(params.max_tokens, defaults.max_tokens);
    }
}
