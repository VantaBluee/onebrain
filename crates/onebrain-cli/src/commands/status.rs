//! `onebrain status`: node, version, engine, loaded model, endpoint, and
//! API token — or exit 1 with a remedy when the daemon isn't running.

use onebraind::paths::AppPaths;

use super::{human_bytes, node_name, CliError};
use crate::client::DaemonClient;

pub fn run(json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;

    let not_running = |detail: String| -> Result<(), CliError> {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "running": false,
                    "error": detail,
                    "remedy": "run `onebrain up`",
                })
            );
            std::process::exit(1);
        }
        Err(CliError(format!(
            "daemon not running; run `onebrain up` to start it ({detail})"
        )))
    };

    let client = match DaemonClient::from_paths(&paths) {
        Ok(c) => c,
        Err(e) => return not_running(e.to_string()),
    };
    let status = match client.status() {
        Ok(s) => s,
        Err(e) => return not_running(e.to_string()),
    };

    let node = node_name(&paths);
    let version = status
        .get("version")
        .and_then(|v| v.as_str())
        .unwrap_or(&client.state().version)
        .to_string();
    let engine = status
        .get("engine_build")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let model = status
        .get("model")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let endpoint = client.base_url().to_string();

    // Peer list (M2). `null` when the endpoint is unavailable — e.g. a
    // pre-M2 daemon still running — so `status` never fails over peers.
    let peers = client
        .peers()
        .ok()
        .and_then(|v| v.get("peers").cloned())
        .unwrap_or(serde_json::Value::Null);

    if json {
        println!(
            "{}",
            serde_json::json!({
                "running": true,
                "node": node,
                "pid": client.state().pid,
                "version": version,
                "engine_build": engine,
                "uptime_secs": status.get("uptime_secs").cloned().unwrap_or(serde_json::Value::Null),
                "started_unix": client.state().started_unix,
                "model": model,
                "endpoint": endpoint,
                "openai_base_url": format!("{endpoint}/v1"),
                "token": client.token(),
                "peers": peers,
            })
        );
        return Ok(());
    }

    println!("node      {node}");
    println!("version   {version}");
    println!("engine    {engine}");
    if model.is_null() {
        println!("model     none (load one with `onebrain run <model>`)");
    } else {
        let name = model.get("name").and_then(|n| n.as_str()).unwrap_or("?");
        let mut details = Vec::new();
        if let Some(size) = model.get("size_bytes").and_then(|v| v.as_u64()) {
            details.push(human_bytes(size));
        }
        if let Some(n_layer) = model.get("n_layer").and_then(|v| v.as_u64()) {
            details.push(format!("{n_layer} layers"));
        }
        if let Some(n_ctx) = model.get("n_ctx").and_then(|v| v.as_u64()) {
            details.push(format!("ctx {n_ctx}"));
        }
        if details.is_empty() {
            println!("model     {name}");
        } else {
            println!("model     {name} ({})", details.join(", "));
        }
    }
    println!("endpoint  {endpoint}");
    println!("token     {}", client.token());
    match peers.as_array() {
        Some(list) if !list.is_empty() => {
            println!();
            println!("peers");
            print!("{}", peers_table(list));
        }
        Some(_) => println!("peers     none (add one with `onebrain pair`)"),
        // Endpoint unavailable (pre-M2 daemon): stay silent rather than
        // implying "no peers".
        None => {}
    }
    Ok(())
}

/// Format the PEERS table: NAME, ID (8 chars), STATE, RTT ms, BW Mbps —
/// `-` for unknowns. Tolerant of missing fields (forward compatibility).
/// Returns the whole table, each line `\n`-terminated.
fn peers_table(peers: &[serde_json::Value]) -> String {
    const HEADERS: [&str; 5] = ["NAME", "ID", "STATE", "RTT ms", "BW Mbps"];
    let rows: Vec<[String; 5]> = peers.iter().map(peer_row).collect();

    let mut widths = HEADERS.map(str::len);
    for row in &rows {
        for (w, cell) in widths.iter_mut().zip(row.iter()) {
            *w = (*w).max(cell.chars().count());
        }
    }

    let mut out = String::new();
    let render = |out: &mut String, cells: [&str; 5]| {
        out.push_str("  ");
        for (i, (cell, width)) in cells.iter().zip(widths.iter()).enumerate() {
            let width = *width;
            if i > 0 {
                out.push_str("  ");
            }
            // NAME/ID/STATE left-aligned; the numeric columns right-aligned.
            if i < 3 {
                out.push_str(&format!("{cell:<width$}"));
            } else {
                out.push_str(&format!("{cell:>width$}"));
            }
        }
        while out.ends_with(' ') {
            out.pop();
        }
        out.push('\n');
    };

    render(&mut out, HEADERS);
    for row in &rows {
        render(
            &mut out,
            [
                row[0].as_str(),
                row[1].as_str(),
                row[2].as_str(),
                row[3].as_str(),
                row[4].as_str(),
            ],
        );
    }
    out
}

/// One peer record → its five table cells.
fn peer_row(peer: &serde_json::Value) -> [String; 5] {
    let text = |key: &str| {
        peer.get(key)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| "-".to_string())
    };
    let name = text("name");
    let id = match peer.get("id").and_then(|v| v.as_str()) {
        Some(id) if !id.is_empty() => id.chars().take(8).collect(),
        _ => "-".to_string(),
    };
    let state = text("state");
    let number = |key: &str| {
        peer.get(key)
            .and_then(|v| v.as_f64())
            .map(|n| format!("{n:.1}"))
            .unwrap_or_else(|| "-".to_string())
    };
    let rtt = number("rtt_ms");
    let bw = number("bandwidth_mbps");
    [name, id, state, rtt, bw]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peers_table_formats_fixture_rows() {
        // Field set from docs/mesh.md `GET /api/internal/peers`.
        let peers: Vec<serde_json::Value> = serde_json::from_str(
            r#"[
              {"name":"gaming-pc","id":"ab12cd34ef56ab78ff","state":"connected",
               "rtt_ms":0.42,"bandwidth_mbps":941.7,"loss":0.0,"last_seen_unix":1756200000},
              {"name":"old-imac","id":"ffee00112233445566","state":"down",
               "rtt_ms":null,"bandwidth_mbps":null,"loss":null,"last_seen_unix":null}
            ]"#,
        )
        .unwrap();
        let table = peers_table(&peers);
        assert_eq!(
            table,
            "  NAME       ID        STATE      RTT ms  BW Mbps\n\
             \x20 gaming-pc  ab12cd34  connected     0.4    941.7\n\
             \x20 old-imac   ffee0011  down            -        -\n"
        );
    }

    #[test]
    fn peers_table_tolerates_missing_fields() {
        let peers = vec![serde_json::json!({ "name": "mystery" })];
        let table = peers_table(&peers);
        let row = table.lines().nth(1).unwrap();
        let cells: Vec<&str> = row.split_whitespace().collect();
        assert_eq!(cells, ["mystery", "-", "-", "-", "-"]);
    }

    #[test]
    fn peer_row_truncates_id_and_dashes_unknowns() {
        let row = peer_row(&serde_json::json!({
            "name": "laptop",
            "id": "0123456789abcdef",
            "state": "reachable",
        }));
        assert_eq!(row, ["laptop", "01234567", "reachable", "-", "-"]);
    }

    #[test]
    fn peer_row_formats_numbers_to_one_decimal() {
        let row = peer_row(&serde_json::json!({
            "name": "pc", "id": "abcd1234", "state": "connected",
            "rtt_ms": 12.349, "bandwidth_mbps": 1000.0,
        }));
        assert_eq!(row[3], "12.3");
        assert_eq!(row[4], "1000.0");
    }
}
