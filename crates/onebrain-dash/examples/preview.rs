//! Dev preview: the real dashboard router plus a stubbed
//! `/api/internal/metrics`, so the SPA can be developed and eyeballed
//! without a running daemon (and without a token — the stub is the whole
//! point). Never shipped; examples build with dev-dependencies only.
//!
//! ```text
//! cargo run -p onebrain-dash --example preview
//! ```
//! then open the printed URL.

use axum::response::IntoResponse;
use axum::routing::get;

/// A plausible 2-peer cluster mid-generation, §1-shaped: exercises every
/// view including a Suspect lossy link, a draining battery peer, and a
/// pipeline plan. Kept as literal JSON so the preview needs no serde.
const FIXTURE: &str = r#"{
  "node": {
    "name": "alpha", "platform": "windows-x86_64", "version": "0.1.0",
    "engine_build_id": "b4589-cuda",
    "memory": { "usable": 12884901888, "total": 17179869184 },
    "devices": ["cpu", "vulkan0"],
    "profile": { "prefill_tps": 231.4, "decode_tps": 12.4 },
    "battery": { "percent": 88, "draining": false },
    "sleep_inhibited": true
  },
  "peers": [
    { "name": "beta", "id_prefix": "9f3a1c", "state": "Connected",
      "rtt_ms": 3.2, "bandwidth_mbps": 941, "loss": 0,
      "memory": { "usable": 21474836480, "total": 34359738368 },
      "profile": { "prefill_tps": 300, "decode_tps": 18 },
      "version": "0.1.0", "engine_build": "b4589-metal" },
    { "name": "gamma", "state": "Suspect", "rtt_ms": 48,
      "bandwidth_mbps": 72, "loss": 0.021,
      "memory": { "usable": 6442450944, "total": 8589934592 },
      "draining": true }
  ],
  "plan": {
    "epoch": 7, "model": "llama-3.1-8b-q4", "strategy": "pipeline",
    "predicted_tpt_ms": 81, "predicted_prefill_ms": 950,
    "assignments": [
      { "node": "alpha", "stage": 0, "layers": [0, 16] },
      { "node": "beta", "stage": 1, "layers": [16, 32] }
    ]
  },
  "requests": [
    { "id": "r1", "dialect": "openai", "model": "llama-3.1-8b-q4",
      "prompt_tokens": 128, "completion_tokens": 412, "prefill_ms": 900,
      "decode_ms": 33000, "ttft_ms": 1000, "drafted": 38, "accepted": 31,
      "finish_reason": "stop", "timestamp": 1756608000 },
    { "id": "r2", "dialect": "ollama", "model": "llama-3.1-8b-q4",
      "prompt_tokens": 64, "completion_tokens": 90, "prefill_ms": 400,
      "decode_ms": 7300, "ttft_ms": 450,
      "finish_reason": "stop", "timestamp": 1756608100 }
  ],
  "advisor": [
    { "severity": "warn",
      "text": "link alpha~gamma measures ~72 Mbps - a wired connection would lift the pipeline's boundary transfer" },
    { "severity": "info", "text": "gamma is on battery and draining" }
  ]
}"#;

#[tokio::main]
async fn main() {
    let app = onebrain_dash::router().route(
        "/api/internal/metrics",
        get(|| async {
            (
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                FIXTURE,
            )
                .into_response()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8642")
        .await
        .expect("bind preview port 8642");
    println!("dashboard preview: http://127.0.0.1:8642/");
    axum::serve(listener, app).await.expect("serve preview");
}
