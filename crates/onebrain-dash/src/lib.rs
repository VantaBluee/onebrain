//! Dashboard v1 (docs/product.md §2): a hand-written static SPA — no
//! framework, no build step, no CDN (docs/decisions/0005) — embedded in the
//! daemon binary via `rust-embed` and served from the daemon's own listener.
//!
//! This crate owns ONLY the static assets and the router that serves them:
//! `GET /` returns the HTML shell, `GET /dash/*` returns the JS/CSS the
//! shell references. There is deliberately no auth logic here: the shell is
//! the one Bearer-exempt page (it contains no data), and every number on
//! screen comes from the page itself calling `GET /api/internal/metrics`
//! with the token the user pastes once (kept in `localStorage`). The daemon
//! mounts this router and keeps enforcing auth on the metrics route exactly
//! as it does for every internal endpoint.

use axum::extract::Path;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;

/// Static dashboard assets, embedded at compile time so the release binary
/// is fully self-contained: nothing to install next to the executable and
/// nothing fetched from a CDN at runtime (ADR 0005).
#[derive(rust_embed::RustEmbed)]
#[folder = "assets"]
struct DashAssets;

/// The dashboard router: `GET /` serves the HTML shell, `GET /dash/*`
/// serves the assets it references. The daemon mounts this alongside its
/// API routers; auth policy for the metrics endpoint stays with the daemon.
pub fn router() -> Router {
    Router::new()
        .route("/", get(index))
        .route("/dash/{*path}", get(asset))
}

async fn index() -> Response {
    serve_embedded("index.html")
}

async fn asset(Path(path): Path<String>) -> Response {
    serve_embedded(&path)
}

/// Look `name` up in the embedded set and serve it with its content type.
///
/// `rust-embed` keys are literal relative paths, so a traversal attempt
/// like `..%2FCargo.toml` simply fails the lookup — there is no filesystem
/// underneath to escape into.
fn serve_embedded(name: &str) -> Response {
    match DashAssets::get(name) {
        Some(file) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, content_type_for(name)),
                // The shell must never go stale across a daemon upgrade
                // (additive metrics schema, but the renderer still moves):
                // no-cache forces revalidation while keeping conditional
                // requests possible later.
                (header::CACHE_CONTROL, "no-cache"),
                (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            ],
            file.data.into_owned(),
        )
            .into_response(),
        None => {
            tracing::debug!(asset = name, "dashboard asset not found");
            (StatusCode::NOT_FOUND, "no such dashboard asset").into_response()
        }
    }
}

/// Content type by extension. Only types we actually ship (or plausibly
/// will: svg/png/ico for future favicons) — anything else is served opaque.
fn content_type_for(name: &str) -> &'static str {
    match name.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset_str(name: &str) -> String {
        let file = DashAssets::get(name).unwrap_or_else(|| {
            panic!("asset {name} must be embedded (crates/onebrain-dash/assets)")
        });
        String::from_utf8(file.data.into_owned()).expect("asset is UTF-8")
    }

    #[test]
    fn all_expected_assets_are_embedded() {
        for name in ["index.html", "app.css", "app.js", "render.js"] {
            assert!(
                DashAssets::get(name).is_some(),
                "expected embedded asset {name}"
            );
        }
    }

    #[test]
    fn index_contains_app_root_marker() {
        // The sim DoD greps `GET /` for this marker; it is the contract
        // that "the dashboard shell was actually served".
        assert!(asset_str("index.html").contains("id=\"ob-dash-root\""));
    }

    #[test]
    fn index_references_only_embedded_local_assets() {
        let index = asset_str("index.html");
        // ADR 0005: no CDN, no external fonts — the shell must not point at
        // any other origin.
        assert!(
            !index.contains("http://") && !index.contains("https://"),
            "index.html must not reference external origins (ADR 0005)"
        );
        // Every /dash/<name> the shell mentions must exist in the embed, or
        // the page would 404 on itself at runtime.
        for chunk in index.split("/dash/").skip(1) {
            let name: String = chunk
                .chars()
                .take_while(|c| !matches!(c, '"' | '\'' | ')' | '<' | ' '))
                .collect();
            assert!(
                DashAssets::get(&name).is_some(),
                "index.html references /dash/{name}, which is not embedded"
            );
        }
    }

    #[test]
    fn js_stays_under_the_adr_budget() {
        // ADR 0005's revisit trigger: ~1.5k lines of JS. Crossing it means
        // re-opening the ADR, not deleting this test.
        let total: usize = ["app.js", "render.js"]
            .iter()
            .map(|n| asset_str(n).lines().count())
            .sum();
        assert!(
            total < 1500,
            "dashboard JS is {total} lines; ADR 0005 says revisit the \
             no-framework decision at ~1500"
        );
    }

    #[test]
    fn content_types_cover_shipped_extensions() {
        assert_eq!(content_type_for("index.html"), "text/html; charset=utf-8");
        assert_eq!(content_type_for("app.css"), "text/css; charset=utf-8");
        assert_eq!(content_type_for("app.js"), "text/javascript; charset=utf-8");
        assert_eq!(content_type_for("noext"), "application/octet-stream");
    }
}
