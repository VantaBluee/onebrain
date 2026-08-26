//! `onebrain pull <ref>`: download a model straight into the local cache —
//! no daemon involved. Registry ids and `hf:` refs stream with a plain
//! carriage-return progress line; local paths are a no-op (models are
//! loaded in place), so pull works offline for them.

use std::io::Write;
use std::str::FromStr;
use std::time::{Duration, Instant};

use onebrain_models::download;
use onebrain_models::registry::{ModelRef, Resolved};
use onebraind::paths::AppPaths;

use super::{human_bytes, CliError};

pub fn run(reference: &str, json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;
    let model_ref = ModelRef::from_str(reference).map_err(|e| CliError(e.to_string()))?;
    let resolved = model_ref.resolve().map_err(|e| CliError(e.to_string()))?;

    let spec = match resolved {
        Resolved::Local(path) => {
            if !path.exists() {
                return Err(CliError(format!(
                    "local model file {} does not exist; check the path (local models are \
                     loaded in place and never downloaded)",
                    path.display()
                )));
            }
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "local",
                        "path": path.display().to_string(),
                    })
                );
            } else {
                println!(
                    "{} is a local file; nothing to download. `onebrain run {}` loads it in place.",
                    path.display(),
                    path.display()
                );
            }
            return Ok(());
        }
        Resolved::Remote(spec) => spec,
    };

    let dest_dir = paths.model_cache_dir().join(&spec.cache_key);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| {
            CliError(format!(
                "failed to start the download runtime ({e}); close other programs and retry"
            ))
        })?;

    // Throttled progress: humans get a \r line, --json gets NDJSON events.
    let mut last_print = Instant::now() - Duration::from_secs(1);
    let mut line_open = false;
    let progress = |completed: u64, total: u64| {
        let done = total > 0 && completed >= total;
        if last_print.elapsed() < Duration::from_millis(100) && !done {
            return;
        }
        last_print = Instant::now();
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "status": "downloading",
                    "completed": completed,
                    "total": total,
                })
            );
        } else {
            // checked_div: total may be 0 while the server hasn't said.
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
            line_open = true;
        }
    };

    let result = runtime.block_on(download::download(&spec, &dest_dir, progress));
    if line_open {
        println!();
    }
    let final_path = result.map_err(|e| CliError(e.to_string()))?;

    // The manifest the downloader just wrote (or the pre-existing one on
    // the already-cached fast path) carries size + hash for the report.
    let manifest = download::read_manifest(&dest_dir).ok();
    if json {
        println!(
            "{}",
            serde_json::json!({
                "status": "done",
                "id": spec.cache_key,
                "path": final_path.display().to_string(),
                "size_bytes": manifest.as_ref().map(|m| m.size_bytes),
                "blake3": manifest.as_ref().map(|m| m.blake3.clone()),
            })
        );
    } else {
        match &manifest {
            Some(m) => println!(
                "pulled {} ({}, blake3 {}…)",
                spec.cache_key,
                human_bytes(m.size_bytes),
                &m.blake3[..12.min(m.blake3.len())]
            ),
            None => println!("pulled {}", spec.cache_key),
        }
        println!("  {}", final_path.display());
        println!("run it: `onebrain run {reference}`");
    }
    Ok(())
}
