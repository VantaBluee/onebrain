//! OpenAI-dialect conformance tests: real router + real HTTP server on an
//! ephemeral loopback port, with `backend::testing::FakeBackend` behind it.

use std::net::SocketAddr;
use std::sync::Arc;

use onebrain_api::auth::AuthConfig;
use onebrain_api::backend::testing::FakeBackend;
use onebrain_api::ApiState;
use serde_json::{json, Value};

const SCRIPT: &str = "the quick brown fox jumps";
const TOKEN: &str = "t0k";

/// Start the real server on 127.0.0.1:0 and return its base URL.
///
/// `onebrain_api::serve` consumes the addr without reporting the bound port,
/// so this replicates its two lines: bind a listener, read `local_addr`,
/// then `axum::serve` with connect-info (the auth middleware requires it).
async fn start_server(localhost_exempt: bool) -> String {
    let state = ApiState {
        backend: Arc::new(FakeBackend::new(SCRIPT, &["fake-model"])),
        auth: Arc::new(AuthConfig {
            token: TOKEN.into(),
            localhost_exempt,
        }),
        product_version: "0.0.0-test",
    };
    let app = onebrain_api::router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    format!("http://{addr}")
}

/// Split an SSE body into its `data:` payloads, in order.
fn sse_data(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            frame
                .strip_prefix("data: ")
                .unwrap_or_else(|| panic!("SSE frame without data prefix: {frame:?}"))
                .to_string()
        })
        .collect()
}

/// Collect (concatenated content deltas, content chunk count, finish_reason)
/// from parsed chat-stream data payloads (excluding the trailing `[DONE]`).
fn digest_chat_stream(datas: &[String]) -> (String, usize, Option<String>) {
    let mut content = String::new();
    let mut content_chunks = 0usize;
    let mut finish = None;
    for data in &datas[..datas.len() - 1] {
        let v: Value = serde_json::from_str(data).unwrap();
        if let Some(piece) = v["choices"][0]["delta"]["content"].as_str() {
            content.push_str(piece);
            content_chunks += 1;
        }
        if let Some(f) = v["choices"][0]["finish_reason"].as_str() {
            finish = Some(f.to_string());
        }
    }
    (content, content_chunks, finish)
}

#[tokio::test]
async fn chat_completion_non_stream() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], SCRIPT);
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    // FakeBackend reports prompt_tokens=3; the script has 5 words.
    assert_eq!(body["usage"]["prompt_tokens"], 3);
    assert_eq!(body["usage"]["completion_tokens"], 5);
    assert_eq!(body["usage"]["total_tokens"], 8);
}

#[tokio::test]
async fn chat_completion_streaming() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        content_type.starts_with("text/event-stream"),
        "expected SSE content-type, got {content_type}"
    );

    let body = resp.text().await.unwrap();
    let datas = sse_data(&body);
    assert_eq!(
        datas.last().unwrap(),
        "[DONE]",
        "stream must end with [DONE]"
    );

    let first: Value = serde_json::from_str(&datas[0]).unwrap();
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    assert!(first["choices"][0]["finish_reason"].is_null());

    let (content, _, finish) = digest_chat_stream(&datas);
    assert_eq!(content, SCRIPT);
    assert_eq!(finish.as_deref(), Some("stop"));
}

#[tokio::test]
async fn chat_streaming_max_tokens_finishes_length() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 2,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    let datas = sse_data(&body);
    assert_eq!(datas.last().unwrap(), "[DONE]");
    let (content, content_chunks, finish) = digest_chat_stream(&datas);
    assert_eq!(content_chunks, 2, "exactly two content chunks expected");
    assert_eq!(content, "the quick");
    assert_eq!(finish.as_deref(), Some("length"));
}

#[tokio::test]
async fn chat_stop_string_finishes_stop_and_omits_match() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["brown"],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        !content.contains("brown"),
        "stop string leaked into output: {content:?}"
    );
    assert_eq!(content, "the quick");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn models_lists_fake_model() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "fake-model");
    assert_eq!(body["data"][0]["object"], "model");
    assert_eq!(body["data"][0]["created"], 0);
    assert_eq!(body["data"][0]["owned_by"], "onebrain");
}

#[tokio::test]
async fn unknown_model_is_404_with_error_envelope() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "missing-model",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "not_found_error");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("missing-model"));
}

#[tokio::test]
async fn missing_model_field_is_400() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn text_completion_non_stream() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&json!({
            "model": "fake-model",
            "prompt": "once upon a time",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "text_completion");
    assert!(body["id"].as_str().unwrap().starts_with("cmpl-"));
    assert_eq!(body["choices"][0]["text"], SCRIPT);
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["total_tokens"], 8);
}

#[tokio::test]
async fn text_completion_accepts_prompt_array_of_one() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&json!({
            "model": "fake-model",
            "prompt": ["once upon a time"],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["text"], SCRIPT);

    // More than one prompt is a 400, not a silent truncation.
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&json!({
            "model": "fake-model",
            "prompt": ["a", "b"],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn text_completion_streaming() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&json!({
            "model": "fake-model",
            "prompt": "hi",
            "stream": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    let datas = sse_data(&body);
    assert_eq!(datas.last().unwrap(), "[DONE]");
    let mut text = String::new();
    let mut finish = None;
    for data in &datas[..datas.len() - 1] {
        let v: Value = serde_json::from_str(data).unwrap();
        assert_eq!(v["object"], "text_completion");
        if let Some(piece) = v["choices"][0]["text"].as_str() {
            text.push_str(piece);
        }
        if let Some(f) = v["choices"][0]["finish_reason"].as_str() {
            finish = Some(f.to_string());
        }
    }
    assert_eq!(text, SCRIPT);
    assert_eq!(finish.as_deref(), Some("stop"));
}

#[tokio::test]
async fn embeddings_returns_501_envelope() {
    let base = start_server(true).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/embeddings"))
        .json(&json!({ "model": "fake-model", "input": "hi" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 501);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("embeddings"));
}

#[tokio::test]
async fn auth_enforced_when_localhost_not_exempt() {
    let base = start_server(false).await;
    let client = reqwest::Client::new();

    let no_token = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(no_token.status(), 401);

    let with_token = client
        .get(format!("{base}/v1/models"))
        .header("Authorization", format!("Bearer {TOKEN}"))
        .send()
        .await
        .unwrap();
    assert_eq!(with_token.status(), 200);
    let body: Value = with_token.json().await.unwrap();
    assert_eq!(body["data"][0]["id"], "fake-model");
}
