//! `onebrain run <model>`: ensure the daemon is up, stream the load
//! progress, then print how to talk to the model (endpoint, OpenAI base
//! URL, token, example curl).

use std::io::Write;

use onebraind::paths::AppPaths;

use super::{up, CliError};

pub fn run(
    model: &str,
    ctx: Option<u32>,
    explain: bool,
    nodes: Option<u32>,
    json: bool,
) -> Result<(), CliError> {
    if explain || nodes.is_some() {
        eprintln!(
            "note: --explain and --nodes engage with distributed inference (milestone M3); \
             M1 always runs single-node."
        );
    }

    let paths = AppPaths::resolve()?;
    let outcome = up::ensure_up(&paths)?;
    let client = outcome.client;

    let mut progress_line_open = false;
    let result = client.load(model, ctx, |event| {
        if json {
            // NDJSON pass-through: scripts see exactly what the daemon sent.
            println!("{event}");
            return;
        }
        match event.get("status").and_then(|s| s.as_str()) {
            Some("downloading") => {
                let completed = event.get("completed").and_then(|v| v.as_u64()).unwrap_or(0);
                let total = event.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
                // checked_div: a server that never sent a total (0) still
                // gets a byte counter instead of a percentage.
                if let Some(pct) = completed.saturating_mul(100).checked_div(total) {
                    print!(
                        "\rdownloading {pct}% ({}/{} MB)   ",
                        completed / 1_000_000,
                        total / 1_000_000
                    );
                } else {
                    print!("\rdownloading {} MB   ", completed / 1_000_000);
                }
                std::io::stdout().flush().ok();
                progress_line_open = true;
            }
            Some("loading") => {
                if progress_line_open {
                    println!();
                    progress_line_open = false;
                }
                println!("loading model into memory...");
            }
            _ => {}
        }
    });
    if progress_line_open {
        println!(); // close the \r progress line before anything else prints
    }
    let terminal = result?;

    match terminal.get("status").and_then(|s| s.as_str()) {
        Some("ready") => {}
        Some("error") => {
            let message = terminal
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            if json {
                println!("{terminal}");
            }
            return Err(CliError(format!("model load failed: {message}")));
        }
        _ => {
            return Err(CliError(
                "the daemon sent an unrecognized terminal status; \
                 update both onebrain and retry"
                    .to_string(),
            ));
        }
    }

    let model_info = terminal
        .get("model")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let endpoint = client.base_url().to_string();
    let openai_base = format!("{endpoint}/v1");

    if json {
        println!(
            "{}",
            serde_json::json!({
                "status": "ready",
                "model": model_info,
                "endpoint": endpoint,
                "openai_base_url": openai_base,
                "token": client.token(),
            })
        );
        return Ok(());
    }

    let loaded_name = model_info
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or(model);
    let mut details = Vec::new();
    if let Some(size) = model_info.get("size_bytes").and_then(|v| v.as_u64()) {
        details.push(super::human_bytes(size));
    }
    if let Some(n_layer) = model_info.get("n_layer").and_then(|v| v.as_u64()) {
        details.push(format!("{n_layer} layers"));
    }
    if let Some(n_ctx) = model_info.get("n_ctx").and_then(|v| v.as_u64()) {
        details.push(format!("ctx {n_ctx}"));
    }
    if details.is_empty() {
        println!("model ready: {loaded_name}");
    } else {
        println!("model ready: {loaded_name} ({})", details.join(", "));
    }
    println!("endpoint         {endpoint}");
    println!("OpenAI base_url  {openai_base}");
    println!("token            {}", client.token());
    println!();
    println!("try it:");
    println!(
        "  curl {endpoint}/api/generate -d '{{\"model\":\"{loaded_name}\",\"prompt\":\"Why is the sky blue?\"}}'"
    );
    Ok(())
}
