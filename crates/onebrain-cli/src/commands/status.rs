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
    Ok(())
}
