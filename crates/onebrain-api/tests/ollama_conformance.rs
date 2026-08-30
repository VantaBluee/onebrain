//! Conformance tests for the Ollama dialect (`/api/*`): real TCP server,
//! real HTTP client, NDJSON streaming verified line by line.

use std::net::SocketAddr;
use std::sync::Arc;

use onebrain_api::auth::AuthConfig;
use onebrain_api::backend::testing::{FakeBackend, FAKE_DECODE_MS, FAKE_PREFILL_MS};
use onebrain_api::{router, ApiState};
use serde_json::{json, Value};

const SCRIPT: &str = "the quick brown fox jumps";
/// Milliseconds → nanoseconds: the scale factor the dialect must apply to
/// the engine's DoneStats wall-clocks (real Ollama reports nanoseconds).
const NS_PER_MS: u64 = 1_000_000;

/// Assert the terminal object carries real Ollama's duration field set
/// (M7, docs/perf.md §1): exact nanosecond scaling of the fake backend's
/// millisecond stats, total = prefill + decode, and a present-but-zero
/// `load_duration` (OneBrain does not attribute load time to a request).
fn assert_duration_fields(last: &Value) {
    assert_eq!(
        last["prompt_eval_duration"],
        FAKE_PREFILL_MS * NS_PER_MS,
        "prompt_eval_duration must be prefill_ms in ns: {last}"
    );
    assert_eq!(
        last["eval_duration"],
        FAKE_DECODE_MS * NS_PER_MS,
        "eval_duration must be decode_ms in ns: {last}"
    );
    assert_eq!(
        last["total_duration"],
        (FAKE_PREFILL_MS + FAKE_DECODE_MS) * NS_PER_MS,
        "total_duration must be prefill+decode in ns: {last}"
    );
    assert_eq!(last["load_duration"], 0, "load_duration present, 0: {last}");
}

/// Bind a fresh server on an ephemeral loopback port; return its base URL.
async fn spawn_server() -> String {
    let state = ApiState {
        backend: Arc::new(FakeBackend::new(SCRIPT, &["fake-model"])),
        auth: Arc::new(AuthConfig {
            token: "t0k".to_string(),
            localhost_exempt: true,
        }),
        product_version: "0.0.0-test",
    };
    let app = router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .expect("serve");
    });
    format!("http://{addr}")
}

fn parse_ndjson(body: &str) -> Vec<Value> {
    body.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad NDJSON line {l:?}: {e}")))
        .collect()
}

#[tokio::test]
async fn generate_streams_ndjson_by_default() {
    let base = spawn_server().await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/generate"))
        .json(&json!({ "model": "fake-model", "prompt": "hello" }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status().as_u16(), 200);
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("application/x-ndjson"),
        "expected NDJSON content type, got {content_type:?}"
    );

    let body = resp.text().await.expect("body");
    let lines = parse_ndjson(&body);
    assert!(lines.len() >= 2, "expected pieces + final line: {lines:?}");
    let (last, pieces) = lines.split_last().expect("at least one line");

    let mut text = String::new();
    for line in pieces {
        assert_eq!(line["done"], false, "non-final line must have done:false");
        assert_eq!(line["model"], "fake-model");
        assert!(line["created_at"].is_string(), "created_at missing: {line}");
        text.push_str(line["response"].as_str().expect("response piece"));
    }
    assert_eq!(text, SCRIPT);

    assert_eq!(last["done"], true);
    assert_eq!(last["done_reason"], "stop");
    assert_eq!(last["response"], "");
    assert_eq!(last["prompt_eval_count"], 3);
    assert_eq!(last["eval_count"], 5);
    assert_duration_fields(last);
}

#[tokio::test]
async fn generate_honors_stream_false() {
    let base = spawn_server().await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/generate"))
        .json(&json!({ "model": "fake-model", "prompt": "hello", "stream": false }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status().as_u16(), 200);
    let body: Value = resp.json().await.expect("single JSON object");
    assert_eq!(body["model"], "fake-model");
    assert_eq!(body["response"], SCRIPT);
    assert_eq!(body["done"], true);
    assert_eq!(body["done_reason"], "stop");
    assert_eq!(body["prompt_eval_count"], 3);
    assert_eq!(body["eval_count"], 5);
    assert_duration_fields(&body);
}

#[tokio::test]
async fn chat_streams_message_pieces() {
    let base = spawn_server().await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/chat"))
        .json(&json!({
            "model": "fake-model",
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status().as_u16(), 200);

    let body = resp.text().await.expect("body");
    let lines = parse_ndjson(&body);
    let (last, pieces) = lines.split_last().expect("at least one line");

    let mut text = String::new();
    for line in pieces {
        assert_eq!(line["done"], false);
        assert_eq!(line["message"]["role"], "assistant");
        text.push_str(line["message"]["content"].as_str().expect("content piece"));
    }
    assert_eq!(text, SCRIPT);

    assert_eq!(last["done"], true);
    assert_eq!(last["done_reason"], "stop");
    assert_eq!(last["message"]["content"], "");
    assert_eq!(last["eval_count"], 5);
    assert_duration_fields(last);
}

#[tokio::test]
async fn chat_honors_stream_false() {
    let base = spawn_server().await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/chat"))
        .json(&json!({
            "model": "fake-model",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": false
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status().as_u16(), 200);
    let body: Value = resp.json().await.expect("single JSON object");
    assert_eq!(body["message"]["role"], "assistant");
    assert_eq!(body["message"]["content"], SCRIPT);
    assert_eq!(body["done"], true);
    assert_eq!(body["eval_count"], 5);
    assert_duration_fields(&body);
}

#[tokio::test]
async fn tags_lists_models_in_ollama_shape() {
    let base = spawn_server().await;
    let body: Value = reqwest::get(format!("{base}/api/tags"))
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let models = body["models"].as_array().expect("models array");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], "fake-model");
    assert_eq!(models[0]["model"], "fake-model");
    assert_eq!(models[0]["size"], 1024);
    assert_eq!(models[0]["details"]["general.architecture"], "fake");
}

#[tokio::test]
async fn ps_lists_only_loaded_models() {
    let base = spawn_server().await;
    let body: Value = reqwest::get(format!("{base}/api/ps"))
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let models = body["models"].as_array().expect("models array");
    // FakeBackend reports its model as loaded.
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], "fake-model");
}

#[tokio::test]
async fn show_returns_details_and_404_for_unknown() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/api/show"))
        .json(&json!({ "model": "fake-model" }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status().as_u16(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["details"]["general.architecture"], "fake");
    assert_eq!(body["model_info"]["general.architecture"], "fake");

    let resp = client
        .post(format!("{base}/api/show"))
        .json(&json!({ "model": "no-such-model" }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status().as_u16(), 404);
}

#[tokio::test]
async fn generate_unknown_model_is_404() {
    let base = spawn_server().await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/generate"))
        .json(&json!({ "model": "no-such-model", "prompt": "hello" }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status().as_u16(), 404);
}

#[tokio::test]
async fn pull_streams_pulling_then_success() {
    let base = spawn_server().await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/pull"))
        .json(&json!({ "model": "fake-model" }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status().as_u16(), 200);

    let body = resp.text().await.expect("body");
    let lines = parse_ndjson(&body);
    assert!(lines.len() >= 2, "expected progress + success: {lines:?}");
    let (last, progress) = lines.split_last().expect("at least one line");
    for line in progress {
        assert_eq!(line["status"], "pulling");
        assert!(line["completed"].is_u64(), "completed missing: {line}");
        assert_eq!(line["total"], 1024);
    }
    assert_eq!(progress[0]["completed"], 512);
    assert_eq!(last["status"], "success");
}

#[tokio::test]
async fn num_predict_limits_generation_with_length_reason() {
    let base = spawn_server().await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/generate"))
        .json(&json!({
            "model": "fake-model",
            "prompt": "hello",
            "options": { "num_predict": 2 }
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status().as_u16(), 200);

    let body = resp.text().await.expect("body");
    let lines = parse_ndjson(&body);
    let (last, pieces) = lines.split_last().expect("at least one line");

    let text: String = pieces
        .iter()
        .map(|l| l["response"].as_str().expect("piece"))
        .collect();
    assert_eq!(text, "the quick");
    assert_eq!(last["done"], true);
    assert_eq!(last["done_reason"], "length");
    assert_eq!(last["eval_count"], 2);
    assert_duration_fields(last);
}

#[tokio::test]
async fn version_reports_product_version() {
    let base = spawn_server().await;
    let body: Value = reqwest::get(format!("{base}/api/version"))
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(body["version"], "0.0.0-test");
}
