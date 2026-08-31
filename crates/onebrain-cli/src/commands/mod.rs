pub mod bench;
pub mod doctor;
pub mod ls;
pub mod pair;
pub mod pin;
pub mod pull;
pub mod rm;
pub mod run;
pub mod self_update;
pub mod status;
pub mod stop;
pub mod unpair;
pub mod up;
pub mod version;

use std::fmt;

use crate::client::ClientError;

#[derive(Debug)]
pub struct CliError(pub String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

impl From<ClientError> for CliError {
    fn from(e: ClientError) -> CliError {
        CliError(e.to_string())
    }
}

impl From<onebraind::DaemonError> for CliError {
    fn from(e: onebraind::DaemonError) -> CliError {
        CliError(e.to_string())
    }
}

impl From<crate::update::UpdateError> for CliError {
    fn from(e: crate::update::UpdateError) -> CliError {
        CliError(e.to_string())
    }
}

/// Format a byte count for humans: decimal units, one decimal below 10
/// (e.g. `397 MB`, `5.4 GB`, `999 B`).
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    if bytes < 1000 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if value < 10.0 {
        format!("{value:.1} {}", UNITS[unit])
    } else {
        format!("{value:.0} {}", UNITS[unit])
    }
}

/// How old a timestamp is, for humans: `just now` under 10 s, then coarse
/// single-unit steps (`45s ago`, `12m ago`, `3h ago`, `2d ago`). A moment
/// in the future (clock skew) reads as `just now`. Shared by `bench`
/// (profile age) and `ls` (model last-used age).
pub fn age_text(then_unix: u64, now_unix: u64) -> String {
    let secs = now_unix.saturating_sub(then_unix);
    match secs {
        0..=9 => "just now".to_string(),
        10..=59 => format!("{secs}s ago"),
        60..=3599 => format!("{}m ago", secs / 60),
        3600..=86399 => format!("{}h ago", secs / 3600),
        _ => format!("{}d ago", secs / 86400),
    }
}

/// Render a placement plan (the `plan` object from a `{"status":"plan"}`
/// NDJSON line or from `/api/internal/status`) for humans: a strategy
/// headline, then one indented line per assignment with its layer range.
/// Tolerant of missing fields (forward compatibility) — an unknown shape
/// degrades to `?` cells, never a panic.
pub fn plan_lines(plan: &serde_json::Value) -> Vec<String> {
    let strategy = plan
        .get("strategy")
        .and_then(|s| s.as_str())
        .unwrap_or("?")
        .to_string();
    let assignments = plan
        .get("assignments")
        .and_then(|a| a.as_array())
        .cloned()
        .unwrap_or_default();

    let mut headline = strategy;
    if assignments.len() >= 2 {
        headline.push_str(&format!(" across {} nodes", assignments.len()));
    }
    if let Some(epoch) = plan.get("epoch").and_then(|e| e.as_u64()) {
        headline.push_str(&format!(" (epoch {epoch})"));
    }

    let mut lines = vec![headline];
    for a in &assignments {
        lines.push(assignment_line(a));
    }
    lines
}

/// One assignment → `stage N  <node8>  layers A-B (K layers)`. The layer
/// range is displayed inclusively (`end` is exclusive on the wire).
fn assignment_line(a: &serde_json::Value) -> String {
    let stage = a
        .get("stage")
        .and_then(|s| s.as_u64())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "?".to_string());
    let node = match a.get("node").and_then(|n| n.as_str()) {
        Some(id) if !id.is_empty() => id.chars().take(8).collect(),
        _ => "?".to_string(),
    };
    let layers = match (
        a.pointer("/layers/start").and_then(|v| v.as_u64()),
        a.pointer("/layers/end").and_then(|v| v.as_u64()),
    ) {
        (Some(start), Some(end)) if end > start => {
            format!(
                "layers {start}-{} ({} layer{})",
                end - 1,
                end - start,
                if end - start == 1 { "" } else { "s" }
            )
        }
        _ => "layers ?".to_string(),
    };
    format!("stage {stage}  {node}  {layers}")
}

/// Node display name: the configured `node_name` when set, else the OS
/// hostname (best-effort via environment).
pub fn node_name(paths: &onebraind::paths::AppPaths) -> String {
    if let Ok(cfg) = onebraind::config::Config::load(&paths.config_file()) {
        if let Some(name) = cfg.node_name {
            return name;
        }
    }
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "this device".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_formats_across_units() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1_000), "1.0 KB");
        assert_eq!(human_bytes(1_500), "1.5 KB");
        assert_eq!(human_bytes(999_000), "999 KB");
        assert_eq!(human_bytes(397_000_000), "397 MB");
        assert_eq!(human_bytes(5_368_709_120), "5.4 GB");
        assert_eq!(human_bytes(2_000_000_000_000), "2.0 TB");
        // TB is the cap: never panics, just grows the number.
        assert_eq!(human_bytes(u64::MAX), "18446744 TB");
    }

    #[test]
    fn age_text_steps_through_units() {
        assert_eq!(age_text(1000, 1000), "just now");
        assert_eq!(age_text(1000, 1009), "just now");
        assert_eq!(age_text(1000, 1010), "10s ago");
        assert_eq!(age_text(1000, 1059), "59s ago");
        assert_eq!(age_text(1000, 1060), "1m ago");
        assert_eq!(age_text(1000, 1000 + 3599), "59m ago");
        assert_eq!(age_text(1000, 1000 + 3600), "1h ago");
        assert_eq!(age_text(1000, 1000 + 86399), "23h ago");
        assert_eq!(age_text(1000, 1000 + 86400), "1d ago");
        assert_eq!(age_text(1000, 1000 + 7 * 86400), "7d ago");
        // Clock skew (a timestamp "in the future") degrades to just-now.
        assert_eq!(age_text(2000, 1000), "just now");
    }

    #[test]
    fn plan_lines_render_pipeline_fixture() {
        // Shape per docs/distributed.md: proto Plan (epoch newtype = bare
        // number, NodeId = bare string) + the daemon's explanation field.
        let plan = serde_json::json!({
            "epoch": 3,
            "model": "blake3:deadbeef",
            "strategy": "PipelineParallel",
            "assignments": [
                {"node": "ab12cd34ef56", "layers": {"start": 0, "end": 3}, "stage": 0},
                {"node": "ffee00112233", "layers": {"start": 3, "end": 5}, "stage": 1}
            ],
            "ctx_len": 4096,
            "explanation": "why text"
        });
        let lines = plan_lines(&plan);
        assert_eq!(
            lines,
            vec![
                "PipelineParallel across 2 nodes (epoch 3)".to_string(),
                "stage 0  ab12cd34  layers 0-2 (3 layers)".to_string(),
                "stage 1  ffee0011  layers 3-4 (2 layers)".to_string(),
            ]
        );
    }

    #[test]
    fn plan_lines_render_solo_without_node_count() {
        let plan = serde_json::json!({
            "epoch": 1,
            "strategy": "Solo",
            "assignments": [
                {"node": "ab12cd34", "layers": {"start": 0, "end": 5}, "stage": 0}
            ]
        });
        let lines = plan_lines(&plan);
        assert_eq!(lines[0], "Solo (epoch 1)");
        assert_eq!(lines[1], "stage 0  ab12cd34  layers 0-4 (5 layers)");
    }

    #[test]
    fn plan_lines_tolerate_unknown_shapes() {
        let lines = plan_lines(&serde_json::json!({}));
        assert_eq!(lines, vec!["?".to_string()]);

        let lines = plan_lines(&serde_json::json!({
            "strategy": "PipelineParallel",
            "assignments": [{"stage": 0}]
        }));
        assert_eq!(lines[1], "stage 0  ?  layers ?");

        // A single-layer range reads singular.
        let lines = plan_lines(&serde_json::json!({
            "strategy": "Solo",
            "assignments": [
                {"node": "aa", "layers": {"start": 4, "end": 5}, "stage": 0}
            ]
        }));
        assert_eq!(lines[1], "stage 0  aa  layers 4-4 (1 layer)");
    }
}
