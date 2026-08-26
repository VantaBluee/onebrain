//! `onebrain ls`: the local model cache as a table — name, human-readable
//! size, hash prefix. (Per-node footprint joins in M2 when there is more
//! than one node to report.)

use onebraind::paths::AppPaths;

use super::{human_bytes, CliError};

pub fn run(json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;
    let models = onebrain_models::cache::list(&paths.model_cache_dir())
        .map_err(|e| CliError(e.to_string()))?;

    if json {
        let rows: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "path": m.path.display().to_string(),
                    "size_bytes": m.size_bytes,
                    "blake3": m.blake3,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&rows).expect("rows serialize")
        );
        return Ok(());
    }

    if models.is_empty() {
        println!("no models cached yet; `onebrain pull <model>` downloads one.");
        return Ok(());
    }

    let name_width = models
        .iter()
        .map(|m| m.id.len())
        .chain(std::iter::once("NAME".len()))
        .max()
        .unwrap_or(4);
    let size_width = 9; // fits "999 MB" / "18446744 TB" stays readable
    println!("{:<name_width$}  {:>size_width$}  HASH", "NAME", "SIZE");
    for m in &models {
        let hash = m
            .blake3
            .as_deref()
            .map(|h| h[..12.min(h.len())].to_string())
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<name_width$}  {:>size_width$}  {hash}",
            m.id,
            human_bytes(m.size_bytes)
        );
    }
    Ok(())
}
