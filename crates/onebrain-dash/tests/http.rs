//! HTTP-level tests for the dashboard router: real server on an ephemeral
//! loopback port, hit over real HTTP (same pattern as onebrain-api's
//! conformance tests). These prove what the sim DoD later re-proves in a
//! live cluster: `GET /` serves the shell with the app-root marker, and
//! `/dash/*` serves assets with correct content types.

/// Bind the real router on 127.0.0.1:0 and return its base URL.
async fn start_server() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, onebrain_dash::router())
            .await
            .expect("serve dashboard router");
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn root_serves_shell_with_app_root_marker() {
    let base = start_server().await;
    let resp = reqwest::get(format!("{base}/")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    assert!(ct.starts_with("text/html"), "content-type was {ct}");
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("id=\"ob-dash-root\""),
        "shell must contain the app-root marker the sim DoD greps for"
    );
    // ADR 0005: the served page references no external origins.
    assert!(!body.contains("http://") && !body.contains("https://"));
}

#[tokio::test]
async fn assets_serve_with_correct_content_types() {
    let base = start_server().await;
    for (path, want_ct) in [
        ("/dash/app.js", "text/javascript"),
        ("/dash/render.js", "text/javascript"),
        ("/dash/app.css", "text/css"),
    ] {
        let resp = reqwest::get(format!("{base}{path}")).await.unwrap();
        assert_eq!(resp.status(), 200, "{path}");
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        assert!(ct.starts_with(want_ct), "{path}: content-type was {ct}");
        assert!(!resp.bytes().await.unwrap().is_empty(), "{path} is empty");
    }
}

#[tokio::test]
async fn unknown_asset_is_404() {
    let base = start_server().await;
    let resp = reqwest::get(format!("{base}/dash/nope.js")).await.unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn traversal_lookalike_is_404() {
    // rust-embed keys are literal names — an encoded `../` can only fail
    // the lookup; prove it stays a 404 rather than reaching any file.
    let base = start_server().await;
    let resp = reqwest::get(format!("{base}/dash/..%2FCargo.toml"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}
