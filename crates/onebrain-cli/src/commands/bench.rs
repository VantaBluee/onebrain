//! `onebrain bench`: make sure the daemon is up, ask it to re-profile this
//! node and probe every connected peer's link (`POST /api/internal/bench`,
//! docs/scheduler-v1.md "`onebrain bench`"), then print the one-page
//! report: a NODE table (memory, prefill/decode tok/s, disk MB/s, profile
//! age) and a LINKS table (peer, RTT, bandwidth, loss). `--json` prints
//! the daemon's response raw.
//!
//! `--cluster` (M7, docs/perf.md §10) widens the report to the whole
//! cluster: every connected peer's fresh on-demand microbench (mesh
//! `BenchRequest`/`BenchReport`), the link table, and a timed end-to-end
//! generation of a standard prompt compared against the constructed M3
//! baseline (`prefill_overlap=false` + `kv_reuse=false`, flipped at
//! runtime via `POST /api/internal/perf` and applied by reloading) and a
//! forced-solo run. Output is a reproducible markdown report (`--json`
//! for the raw aggregate); every figure is a measurement labeled with the
//! model, plan, and config it was taken under — never a promise (§1.6).

use std::time::{SystemTime, UNIX_EPOCH};

use onebraind::paths::AppPaths;

use super::{age_text, human_bytes, node_name, plan_lines, up, CliError};
use crate::client::{DaemonClient, LoadOptions};

/// Registry reference the end-to-end section falls back to when nothing is
/// loaded — the same tiny model the microbench uses, so it is always
/// pullable and cheap.
const BENCH_MODEL_REF: &str = "tinystories-260k";

/// New-token budget for each timed end-to-end run.
const E2E_MAX_NEW_TOKENS: u32 = 64;

/// The standard bench prompt (docs/perf.md §10): FIXED so reports stay
/// comparable across runs and machines. ~1.2 KB — a few hundred prompt
/// tokens on typical vocabularies: enough prefill for the overlap
/// comparison to mean something, small enough for any context window.
fn standard_prompt() -> String {
    "OneBrain measures what it ships: every lever lands with the instrument that proves it. "
        .repeat(14)
}

pub fn run(json: bool, cluster: bool) -> Result<(), CliError> {
    if cluster {
        return run_cluster(json);
    }
    let paths = AppPaths::resolve()?;
    let outcome = up::ensure_up(&paths)?;
    let client = outcome.client;

    if !json {
        println!("benchmarking (microbench ~10 s; the first run may pull the tiny test model)...");
    }
    let report = client.bench()?;

    if json {
        // Raw pass-through: scripts see exactly what the daemon measured.
        println!("{report}");
        return Ok(());
    }

    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    print!("{}", render_report(&report, &node_name(&paths), now_unix));
    Ok(())
}

/// The whole human-readable report, each line `\n`-terminated. Pure over
/// the daemon's bench response so fixtures can pin it; tolerant of missing
/// fields (forward compatibility) — unknowns render as `-`, never a panic.
fn render_report(report: &serde_json::Value, node: &str, now_unix: u64) -> String {
    let mut out = format!("node   {node}\n");
    let profile = report.get("profile").cloned().unwrap_or_default();
    out.push_str(&table(
        &[
            "MEMORY",
            "PREFILL tok/s",
            "DECODE tok/s",
            "DISK MB/s",
            "PROFILE AGE",
        ],
        &[profile_row(&profile, now_unix)],
        // MEMORY left-aligned (mixed-unit strings); the three throughput
        // columns right-aligned; the age column left again.
        &[false, true, true, true, false],
    ));
    out.push('\n');
    match report.get("links").and_then(|l| l.as_array()) {
        Some(links) if !links.is_empty() => {
            out.push_str("links\n");
            let rows: Vec<[String; 4]> = links.iter().map(link_row).collect();
            out.push_str(&table(
                &["PEER", "RTT ms", "BW Mbps", "LOSS %"],
                &rows,
                &[false, true, true, true],
            ));
        }
        Some(_) => out.push_str("links  none (pair another device with `onebrain pair`)\n"),
        // A daemon that sent no links array at all: stay silent rather
        // than implying "no links".
        None => {}
    }
    out
}

/// The NODE table's single row, from the bench response's `profile`.
fn profile_row(profile: &serde_json::Value, now_unix: u64) -> [String; 5] {
    let memory = profile
        .get("usable_memory_bytes")
        .and_then(|v| v.as_u64())
        .map(human_bytes)
        .unwrap_or_else(|| "-".to_string());
    let age = profile
        .get("measured_unix")
        .and_then(|v| v.as_u64())
        .map(|measured| age_text(measured, now_unix))
        .unwrap_or_else(|| "-".to_string());
    [
        memory,
        num(profile, "prefill_tps"),
        num(profile, "decode_tps"),
        num(profile, "disk_mbps"),
        age,
    ]
}

/// One link record → its four LINKS cells. `loss` arrives as a fraction
/// (missed-heartbeat share, docs/mesh.md) and renders as a percentage.
fn link_row(link: &serde_json::Value) -> [String; 4] {
    let peer = link
        .get("peer")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| "-".to_string());
    let loss = link
        .get("loss")
        .and_then(|v| v.as_f64())
        .map(|f| format!("{:.1}", f * 100.0))
        .unwrap_or_else(|| "-".to_string());
    [peer, num(link, "rtt_ms"), num(link, "bandwidth_mbps"), loss]
}

/// A numeric field to one decimal, or `-` when absent/non-numeric.
fn num(obj: &serde_json::Value, key: &str) -> String {
    obj.get(key)
        .and_then(|v| v.as_f64())
        .map(|n| format!("{n:.1}"))
        .unwrap_or_else(|| "-".to_string())
}

/// Render an aligned table (two-space indent, two spaces between columns,
/// no trailing whitespace): `right[i]` right-aligns column `i`. The same
/// conventions as `onebrain status`'s peers table.
fn table<const N: usize>(headers: &[&str; N], rows: &[[String; N]], right: &[bool; N]) -> String {
    let mut widths = headers.map(str::len);
    for row in rows {
        for (w, cell) in widths.iter_mut().zip(row.iter()) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let mut out = String::new();
    let mut render = |cells: [&str; N]| {
        out.push_str("  ");
        for (i, (cell, width)) in cells.iter().zip(widths.iter()).enumerate() {
            let width = *width;
            if i > 0 {
                out.push_str("  ");
            }
            if right[i] {
                out.push_str(&format!("{cell:>width$}"));
            } else {
                out.push_str(&format!("{cell:<width$}"));
            }
        }
        while out.ends_with(' ') {
            out.pop();
        }
        out.push('\n');
    };
    render(*headers);
    for row in rows {
        let cells: [&str; N] = std::array::from_fn(|i| row[i].as_str());
        render(cells);
    }
    out
}

/// `onebrain bench --cluster` (docs/perf.md §10). Stdout carries ONLY the
/// report — markdown, or the raw aggregate with `--json` — so
/// `onebrain bench --cluster > bench.md` is a valid report file; progress
/// chatter goes to stderr.
fn run_cluster(json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;
    let outcome = up::ensure_up(&paths)?;
    let client = outcome.client;

    eprintln!(
        "cluster bench: local microbench (~10 s; the first run may pull the tiny test model)..."
    );
    let local = client.bench()?;
    eprintln!("cluster bench: asking every connected peer to bench (concurrent, up to 60 s)...");
    let peers = client.bench_peers()?;
    let e2e = run_e2e(&client)?;
    let report = serde_json::json!({
        "local": local,
        "peers": peers.get("peers").cloned().unwrap_or_else(|| serde_json::json!([])),
        "e2e": e2e,
    });
    if json {
        println!("{report}");
        return Ok(());
    }
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    print!(
        "{}",
        render_cluster_report(&report, &node_name(&paths), now_unix)
    );
    Ok(())
}

/// The end-to-end section: three timed runs of the standard prompt —
/// "as configured" (the runtime toggles as they stand), the constructed
/// "M3 baseline" (both toggles off), and "solo local" (forced `--nodes 1`)
/// — each on a FRESH load, so the toggles take effect and every run
/// prefills cold. Individual run failures become row-level errors, never a
/// failed bench: the report shows what could be measured.
fn run_e2e(client: &DaemonClient) -> Result<serde_json::Value, CliError> {
    let status = client.status()?;
    let mut notes: Vec<String> = Vec::new();
    // Measure the model the user actually has loaded (its reference rides
    // on /api/internal/status); with nothing loaded, the tiny bench model
    // stands in — noted, because it stays loaded afterwards.
    let (model_ref, initially_loaded) = match status.get("model_reference").and_then(|r| r.as_str())
    {
        Some(reference) => (reference.to_string(), true),
        None => {
            notes.push(format!(
                "no model was loaded; measured the bench model '{BENCH_MODEL_REF}' \
                 (it stays loaded solo afterwards)"
            ));
            (BENCH_MODEL_REF.to_string(), false)
        }
    };
    let prompt = standard_prompt();

    // The toggles as they currently stand = the "as configured" row (an
    // empty POST reads them without changing anything).
    let current = client.set_perf_toggles(None, None)?;
    let orig_po = current
        .get("prefill_overlap")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let orig_kv = current
        .get("kv_reuse")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let specs: [(&str, bool, bool, Option<u32>); 3] = [
        ("as configured", orig_po, orig_kv, None),
        ("M3 baseline", false, false, None),
        ("solo local", orig_po, orig_kv, Some(1)),
    ];
    let mut runs = Vec::with_capacity(specs.len());
    for (label, prefill_overlap, kv_reuse, nodes) in specs {
        eprintln!("cluster bench: end-to-end run '{label}' (reload + timed generation)...");
        runs.push(e2e_run(
            client,
            &model_ref,
            &prompt,
            label,
            prefill_overlap,
            kv_reuse,
            nodes,
        ));
    }

    // Leave the daemon as we found it: the config-time toggles back on,
    // and the pre-bench model re-planned the way a plain load would place
    // it (the solo run left it forced onto this node).
    if let Err(e) = client.set_perf_toggles(Some(orig_po), Some(orig_kv)) {
        notes.push(format!("restoring the perf toggles failed: {e}"));
    }
    if initially_loaded {
        eprintln!("cluster bench: restoring the pre-bench placement...");
        if let Err(e) = load_for_bench(client, &model_ref, None) {
            notes.push(format!("restoring the pre-bench placement failed: {e}"));
        }
    }
    Ok(serde_json::json!({
        "model": model_ref,
        "prompt_chars": prompt.len(),
        "max_new_tokens": E2E_MAX_NEW_TOKENS,
        "runs": runs,
        "notes": notes,
    }))
}

/// One timed run: flip the toggles, reload (so they take effect and the
/// prefill starts cold), then one greedy `/api/generate` of the standard
/// prompt. The row records the run's ACTUAL plan and the Ollama duration
/// fields verbatim (nanoseconds, docs/perf.md §1) — honesty over averaging.
fn e2e_run(
    client: &DaemonClient,
    model_ref: &str,
    prompt: &str,
    label: &str,
    prefill_overlap: bool,
    kv_reuse: bool,
    nodes: Option<u32>,
) -> serde_json::Value {
    let mut row = serde_json::json!({
        "run": label,
        "prefill_overlap": prefill_overlap,
        "kv_reuse": kv_reuse,
    });
    if let Err(e) = client.set_perf_toggles(Some(prefill_overlap), Some(kv_reuse)) {
        row["error"] = serde_json::json!(format!("setting the perf toggles failed: {e}"));
        return row;
    }
    match load_for_bench(client, model_ref, nodes) {
        Ok(plan) => row["plan"] = plan,
        Err(e) => {
            row["error"] = serde_json::json!(format!("load failed: {e}"));
            return row;
        }
    }
    match client.generate_timed(model_ref, prompt, E2E_MAX_NEW_TOKENS) {
        Ok(done) => {
            for field in [
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
                "total_duration",
            ] {
                if let Some(v) = done.get(field) {
                    row[field] = v.clone();
                }
            }
        }
        Err(e) => row["error"] = serde_json::json!(format!("generation failed: {e}")),
    }
    row
}

/// Load (or reload) `model` through the internal API, silently draining
/// progress; returns the plan object from the `plan` progress line (Null
/// when the daemon never sent one).
fn load_for_bench(
    client: &DaemonClient,
    model: &str,
    nodes: Option<u32>,
) -> Result<serde_json::Value, CliError> {
    let mut plan = serde_json::Value::Null;
    let opts = LoadOptions {
        nodes,
        ..LoadOptions::default()
    };
    let terminal = client.load(model, &opts, |event| {
        if event.get("status").and_then(|s| s.as_str()) == Some("plan") {
            plan = event
                .get("plan")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
        }
    })?;
    match terminal.get("status").and_then(|s| s.as_str()) {
        Some("ready") => Ok(plan),
        Some("error") => Err(CliError(
            terminal
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error")
                .to_string(),
        )),
        _ => Err(CliError(
            "the daemon sent an unrecognized terminal status; update both onebrain and retry"
                .to_string(),
        )),
    }
}

/// The full `--cluster` markdown report. Pure over the aggregated report
/// object so fixtures can pin it; tolerant of missing fields (forward
/// compatibility) — unknowns render as `-`, never a panic.
fn render_cluster_report(report: &serde_json::Value, node: &str, now_unix: u64) -> String {
    let mut out = String::from("# OneBrain cluster bench\n\n");
    out.push_str(&format!("node {node}\n\n"));

    // Nodes: the local profile row + one row per peer bench reply.
    out.push_str("## Nodes\n\n");
    let profile = report
        .pointer("/local/profile")
        .cloned()
        .unwrap_or_default();
    let mut rows: Vec<[String; 6]> = vec![[
        format!("{node} (local)"),
        profile
            .get("usable_memory_bytes")
            .and_then(|v| v.as_u64())
            .map(human_bytes)
            .unwrap_or_else(|| "-".to_string()),
        num(&profile, "prefill_tps"),
        num(&profile, "decode_tps"),
        num(&profile, "disk_mbps"),
        profile
            .get("measured_unix")
            .and_then(|v| v.as_u64())
            .map(|measured| age_text(measured, now_unix))
            .unwrap_or_else(|| "-".to_string()),
    ]];
    let mut notes: Vec<String> = Vec::new();
    for peer in report
        .get("peers")
        .and_then(|p| p.as_array())
        .into_iter()
        .flatten()
    {
        let name = peer
            .get("peer")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let available = peer
            .get("available")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let measured = if available {
            peer.get("measured_unix")
                .and_then(|v| v.as_u64())
                .map(|measured| age_text(measured, now_unix))
                .unwrap_or_else(|| "-".to_string())
        } else if let Some(error) = peer.get("error").and_then(|v| v.as_str()) {
            notes.push(format!("peer '{name}' did not answer: {error}"));
            "no answer".to_string()
        } else {
            // The wire's cannot-bench-now marker: busy, shard-serving, or
            // the test model is not cached there.
            "cannot bench now".to_string()
        };
        rows.push([
            name,
            // Peer bench replies carry throughputs only; memory shows in
            // `onebrain status`'s peer table.
            "-".to_string(),
            num(peer, "prefill_tps"),
            num(peer, "decode_tps"),
            num(peer, "disk_mbps"),
            measured,
        ]);
    }
    out.push_str(&md_table(
        &[
            "node",
            "memory",
            "prefill tok/s",
            "decode tok/s",
            "disk MB/s",
            "measured",
        ],
        &rows,
    ));
    if !notes.is_empty() {
        out.push('\n');
        for note in &notes {
            out.push_str(&format!("- note: {note}\n"));
        }
    }

    // Links: the head's measured link table.
    out.push_str("\n## Links\n\n");
    match report.pointer("/local/links").and_then(|l| l.as_array()) {
        Some(links) if !links.is_empty() => {
            let rows: Vec<[String; 4]> = links.iter().map(link_row).collect();
            out.push_str(&md_table(
                &["peer", "RTT ms", "bandwidth Mbps", "loss %"],
                &rows,
            ));
        }
        _ => out.push_str("no links measured (pair another device with `onebrain pair`)\n"),
    }

    // End-to-end: the timed generation comparison, labeled with the model,
    // plan, and config each figure was taken under (§1.6).
    out.push_str("\n## End-to-end generation\n\n");
    let e2e = report.get("e2e").cloned().unwrap_or_default();
    let model = e2e.get("model").and_then(|v| v.as_str()).unwrap_or("?");
    let chars = e2e
        .get("prompt_chars")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let max_new = e2e
        .get("max_new_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    out.push_str(&format!(
        "model `{model}` · prompt {chars} chars · max {max_new} new tokens · \
         greedy (temperature 0)\n\n"
    ));
    let empty = Vec::new();
    let runs = e2e.get("runs").and_then(|r| r.as_array()).unwrap_or(&empty);
    let rows: Vec<[String; 11]> = runs.iter().map(e2e_row).collect();
    out.push_str(&md_table(
        &[
            "run",
            "prefill_overlap",
            "kv_reuse",
            "plan",
            "prefill tok",
            "prefill ms",
            "prefill tok/s",
            "decode tok",
            "decode ms",
            "decode tok/s",
            "total ms",
        ],
        &rows,
    ));
    let mut trailer: Vec<String> = Vec::new();
    for run in runs {
        if let Some(error) = run.get("error").and_then(|v| v.as_str()) {
            let label = run.get("run").and_then(|v| v.as_str()).unwrap_or("?");
            trailer.push(format!("- run '{label}' failed: {error}"));
        }
    }
    for note in e2e
        .get("notes")
        .and_then(|n| n.as_array())
        .into_iter()
        .flatten()
    {
        if let Some(note) = note.as_str() {
            trailer.push(format!("- note: {note}"));
        }
    }
    if !trailer.is_empty() {
        out.push('\n');
        for line in &trailer {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str(
        "\nMeasured values on this hardware, model, plan, and config — \
         comparisons, not promises.\n",
    );
    out
}

/// One end-to-end run → its 11 report cells. Durations arrive in
/// NANOSECONDS (the Ollama wire fields, docs/perf.md §1) and render as
/// milliseconds; tok/s derive from count/duration. A failed run keeps its
/// label and config with `-` figures — its error renders under the table.
fn e2e_row(run: &serde_json::Value) -> [String; 11] {
    let dash = || "-".to_string();
    let count = |key: &str| {
        run.get(key)
            .and_then(|v| v.as_u64())
            .map(|n| n.to_string())
            .unwrap_or_else(dash)
    };
    let ns_ms = |key: &str| {
        run.get(key)
            .and_then(|v| v.as_u64())
            .map(|ns| format!("{:.1}", ns as f64 / 1e6))
            .unwrap_or_else(dash)
    };
    let tps = |count_key: &str, dur_key: &str| match (
        run.get(count_key).and_then(|v| v.as_u64()),
        run.get(dur_key).and_then(|v| v.as_u64()),
    ) {
        (Some(n), Some(ns)) if ns > 0 => format!("{:.1}", n as f64 * 1e9 / ns as f64),
        _ => dash(),
    };
    let flag = |key: &str| {
        run.get(key)
            .and_then(|v| v.as_bool())
            .map(|b| b.to_string())
            .unwrap_or_else(dash)
    };
    let plan = match run.get("plan") {
        Some(p) if !p.is_null() => plan_lines(p).into_iter().next().unwrap_or_else(dash),
        _ => dash(),
    };
    [
        run.get("run")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string(),
        flag("prefill_overlap"),
        flag("kv_reuse"),
        plan,
        count("prompt_eval_count"),
        ns_ms("prompt_eval_duration"),
        tps("prompt_eval_count", "prompt_eval_duration"),
        count("eval_count"),
        ns_ms("eval_duration"),
        tps("eval_count", "eval_duration"),
        ns_ms("total_duration"),
    ]
}

/// A GitHub-flavored markdown table. Cells are unpadded — renderers align
/// for display, and unpadded cells keep the raw report diffable when
/// figures change width.
fn md_table<const N: usize>(headers: &[&str; N], rows: &[[String; N]]) -> String {
    let mut out = format!("| {} |\n", headers.join(" | "));
    out.push('|');
    for _ in 0..N {
        out.push_str("---|");
    }
    out.push('\n');
    for row in rows {
        out.push_str(&format!("| {} |\n", row.join(" | ")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The response shape from docs/scheduler-v1.md / the M4 internal-api
    /// contract for `POST /api/internal/bench`.
    fn fixture() -> serde_json::Value {
        json!({
            "profile": {
                "prefill_tps": 812.53,
                "decode_tps": 41.25,
                "disk_mbps": 1732.04,
                "usable_memory_bytes": 15_200_000_000u64,
                "measured_unix": 1_756_200_000u64
            },
            "links": [
                { "peer": "gaming-pc", "rtt_ms": 0.42, "bandwidth_mbps": 941.7, "loss": 0.01 },
                { "peer": "old-imac", "rtt_ms": null, "bandwidth_mbps": null, "loss": null }
            ]
        })
    }

    #[test]
    fn report_renders_node_and_links_tables() {
        // 125 s after the measurement -> "2m ago".
        let got = render_report(&fixture(), "this-pc", 1_756_200_125);
        let want = "\
node   this-pc\n\
\x20 MEMORY  PREFILL tok/s  DECODE tok/s  DISK MB/s  PROFILE AGE\n\
\x20 15 GB           812.5          41.2     1732.0  2m ago\n\
\n\
links\n\
\x20 PEER       RTT ms  BW Mbps  LOSS %\n\
\x20 gaming-pc     0.4    941.7     1.0\n\
\x20 old-imac        -        -       -\n";
        assert_eq!(got, want);
    }

    #[test]
    fn report_with_no_links_names_the_remedy() {
        let mut report = fixture();
        report["links"] = json!([]);
        let got = render_report(&report, "solo-pc", 1_756_200_000);
        assert!(got.contains("links  none (pair another device with `onebrain pair`)"));
        // The node table still renders, with a fresh profile age.
        assert!(got.contains("just now"));
    }

    #[test]
    fn report_tolerates_a_missing_profile_and_links() {
        // A daemon that answered with an empty object must still produce a
        // table of dashes, never a panic.
        let got = render_report(&json!({}), "n", 0);
        let row = got.lines().nth(2).unwrap();
        let cells: Vec<&str> = row.split_whitespace().collect();
        assert_eq!(cells, ["-", "-", "-", "-", "-"]);
        // No links array at all: no links section (vs an explicit empty
        // array, which names the remedy).
        assert!(!got.contains("links"));
    }

    #[test]
    fn loss_fraction_renders_as_percent() {
        let row = link_row(&json!({ "peer": "pc", "loss": 0.125 }));
        assert_eq!(row[3], "12.5");
        let row = link_row(&json!({ "peer": "pc", "loss": 0.0 }));
        assert_eq!(row[3], "0.0");
    }

    /// The aggregate `bench --cluster` builds: the local bench response,
    /// the peers' bench replies, and the end-to-end section
    /// (docs/perf.md §10).
    fn cluster_fixture() -> serde_json::Value {
        json!({
            "local": fixture(),
            "peers": [
                { "peer": "gaming-pc", "id": "aa", "available": true,
                  "prefill_tps": 900.0, "decode_tps": 55.12, "disk_mbps": 2100.0,
                  "measured_unix": 1_756_200_100u64 },
                { "peer": "old-imac", "id": "bb", "available": false },
                { "peer": "flaky", "id": "cc", "available": false, "error": "mesh timeout" }
            ],
            "e2e": {
                "model": "tinystories-260k", "prompt_chars": 1232, "max_new_tokens": 64,
                "runs": [
                    { "run": "as configured", "prefill_overlap": true, "kv_reuse": true,
                      "plan": { "strategy": "PipelineParallel", "epoch": 7,
                                "assignments": [{}, {}] },
                      "prompt_eval_count": 320, "prompt_eval_duration": 512_000_000u64,
                      "eval_count": 64, "eval_duration": 800_000_000u64,
                      "total_duration": 1_312_000_000u64 },
                    { "run": "M3 baseline", "prefill_overlap": false, "kv_reuse": false,
                      "plan": { "strategy": "PipelineParallel", "epoch": 8,
                                "assignments": [{}, {}] },
                      "prompt_eval_count": 320, "prompt_eval_duration": 640_000_000u64,
                      "eval_count": 64, "eval_duration": 820_000_000u64,
                      "total_duration": 1_460_000_000u64 },
                    { "run": "solo local", "prefill_overlap": true, "kv_reuse": true,
                      "error": "load failed: boom" }
                ],
                "notes": ["no model was loaded; measured the bench model 'tinystories-260k'"]
            }
        })
    }

    /// Pins the whole `--cluster` markdown report: node + links tables,
    /// the e2e comparison rows (ns → ms + tok/s derivation), unavailable
    /// and erroring peers, a failed run's trailer line, and the §1.6
    /// measurement disclaimer.
    #[test]
    fn cluster_report_renders_the_full_markdown() {
        let mut report = cluster_fixture();
        // A single link keeps the fixture focused (the fixture() links
        // include a null-field row already covered by the plain report test).
        report["local"]["links"] = json!([
            { "peer": "gaming-pc", "rtt_ms": 0.42, "bandwidth_mbps": 941.7, "loss": 0.01 }
        ]);
        let got = render_cluster_report(&report, "this-pc", 1_756_200_125);
        let want = "\
# OneBrain cluster bench\n\
\n\
node this-pc\n\
\n\
## Nodes\n\
\n\
| node | memory | prefill tok/s | decode tok/s | disk MB/s | measured |\n\
|---|---|---|---|---|---|\n\
| this-pc (local) | 15 GB | 812.5 | 41.2 | 1732.0 | 2m ago |\n\
| gaming-pc | - | 900.0 | 55.1 | 2100.0 | 25s ago |\n\
| old-imac | - | - | - | - | cannot bench now |\n\
| flaky | - | - | - | - | no answer |\n\
\n\
- note: peer 'flaky' did not answer: mesh timeout\n\
\n\
## Links\n\
\n\
| peer | RTT ms | bandwidth Mbps | loss % |\n\
|---|---|---|---|\n\
| gaming-pc | 0.4 | 941.7 | 1.0 |\n\
\n\
## End-to-end generation\n\
\n\
model `tinystories-260k` · prompt 1232 chars · max 64 new tokens · greedy (temperature 0)\n\
\n\
| run | prefill_overlap | kv_reuse | plan | prefill tok | prefill ms | prefill tok/s | decode tok | decode ms | decode tok/s | total ms |\n\
|---|---|---|---|---|---|---|---|---|---|---|\n\
| as configured | true | true | PipelineParallel across 2 nodes (epoch 7) | 320 | 512.0 | 625.0 | 64 | 800.0 | 80.0 | 1312.0 |\n\
| M3 baseline | false | false | PipelineParallel across 2 nodes (epoch 8) | 320 | 640.0 | 500.0 | 64 | 820.0 | 78.0 | 1460.0 |\n\
| solo local | true | true | - | - | - | - | - | - | - | - |\n\
\n\
- run 'solo local' failed: load failed: boom\n\
- note: no model was loaded; measured the bench model 'tinystories-260k'\n\
\n\
Measured values on this hardware, model, plan, and config — comparisons, not promises.\n";
        assert_eq!(got, want);
    }

    #[test]
    fn cluster_report_tolerates_an_empty_aggregate() {
        // A daemon that answered with nothing still renders every section
        // with dashes/remedies — never a panic.
        let got = render_cluster_report(&json!({}), "n", 0);
        assert!(got.starts_with("# OneBrain cluster bench\n"));
        assert!(got.contains("| n (local) | - | - | - | - | - |"), "{got}");
        assert!(got.contains("no links measured (pair another device with `onebrain pair`)"));
        assert!(got.contains("## End-to-end generation"));
        assert!(got.contains("comparisons, not promises."));
    }

    #[test]
    fn e2e_row_derives_tokens_per_second_and_survives_zeroes() {
        // All-zero durations ("not measured", docs/perf.md §1) must render
        // as dashes, never a division by zero.
        let row = e2e_row(&json!({
            "run": "as configured", "prefill_overlap": true, "kv_reuse": false,
            "prompt_eval_count": 10, "prompt_eval_duration": 0,
            "eval_count": 5, "eval_duration": 2_500_000_000u64
        }));
        assert_eq!(row[4], "10");
        assert_eq!(row[5], "0.0"); // measured-as-zero ms still prints
        assert_eq!(row[6], "-"); // …but no tok/s can be derived from it
        assert_eq!(row[8], "2500.0");
        assert_eq!(row[9], "2.0");
        assert_eq!(row[10], "-"); // absent total_duration
    }

    #[test]
    fn profile_age_uses_measured_unix() {
        let row = profile_row(
            &json!({ "usable_memory_bytes": 999u64, "measured_unix": 500u64 }),
            500 + 90,
        );
        assert_eq!(row[0], "999 B");
        assert_eq!(row[4], "1m ago");
    }
}
