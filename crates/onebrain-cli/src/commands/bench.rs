//! `onebrain bench`: make sure the daemon is up, ask it to re-profile this
//! node and probe every connected peer's link (`POST /api/internal/bench`,
//! docs/scheduler-v1.md "`onebrain bench`"), then print the one-page
//! report: a NODE table (memory, prefill/decode tok/s, disk MB/s, profile
//! age) and a LINKS table (peer, RTT, bandwidth, loss). `--json` prints
//! the daemon's response raw.

use std::time::{SystemTime, UNIX_EPOCH};

use onebraind::paths::AppPaths;

use super::{age_text, human_bytes, node_name, up, CliError};

pub fn run(json: bool) -> Result<(), CliError> {
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
