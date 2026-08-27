//! `onebrain ls`: the local model cache as a table — name, human-readable
//! size, split part count, pin marker, last-used age, hash prefix. A split
//! model is ONE row: the cache aggregates its parts into a single entry
//! (summed size, `parts` count), per docs/logistics.md "Split-GGUF". Reads
//! the cache directly (like `pull`), so it works with the daemon down.

use std::time::{SystemTime, UNIX_EPOCH};

use onebrain_models::cache::CachedModel;
use onebraind::paths::AppPaths;

use super::{age_text, human_bytes, CliError};

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
                    "parts": m.parts,
                    "pinned": m.pinned,
                    "last_used_unix": m.last_used_unix,
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

    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    print!("{}", render_table(&models, now_unix));
    Ok(())
}

/// The whole table, each line `\n`-terminated, no trailing whitespace. Pure
/// over the cache listing so tests can pin it.
fn render_table(models: &[CachedModel], now_unix: u64) -> String {
    const HEADERS: [&str; 6] = ["NAME", "SIZE", "PARTS", "PIN", "LAST USED", "HASH"];
    // NAME left; SIZE and PARTS right (numeric); the rest left.
    const RIGHT: [bool; 6] = [false, true, true, false, false, false];

    let rows: Vec<[String; 6]> = models.iter().map(|m| row(m, now_unix)).collect();
    let mut widths = HEADERS.map(str::len);
    for row in &rows {
        for (w, cell) in widths.iter_mut().zip(row.iter()) {
            *w = (*w).max(cell.chars().count());
        }
    }

    let mut out = String::new();
    let mut render = |cells: [&str; 6]| {
        for (i, (cell, width)) in cells.iter().zip(widths.iter()).enumerate() {
            let width = *width;
            if i > 0 {
                out.push_str("  ");
            }
            if RIGHT[i] {
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
    render(HEADERS);
    for row in &rows {
        let cells: [&str; 6] = std::array::from_fn(|i| row[i].as_str());
        render(cells);
    }
    out
}

/// One cached model → its six cells. `last_used_unix == 0` means the entry
/// has never been loaded (a pre-M6 manifest, or pulled but not yet run).
fn row(m: &CachedModel, now_unix: u64) -> [String; 6] {
    let hash = m
        .blake3
        .as_deref()
        .map(|h| h[..12.min(h.len())].to_string())
        .unwrap_or_else(|| "-".to_string());
    let last_used = if m.last_used_unix == 0 {
        "never".to_string()
    } else {
        age_text(m.last_used_unix, now_unix)
    };
    [
        m.id.clone(),
        human_bytes(m.size_bytes),
        m.parts.to_string(),
        if m.pinned { "*" } else { "" }.to_string(),
        last_used,
        hash,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model(id: &str) -> CachedModel {
        CachedModel {
            id: id.to_string(),
            path: PathBuf::from(format!("C:\\cache\\{id}\\model.gguf")),
            size_bytes: 5_368_709_120,
            blake3: Some("abcdef0123456789".to_string()),
            parts: 1,
            pinned: false,
            last_used_unix: 0,
        }
    }

    #[test]
    fn table_renders_pin_age_and_split_parts() {
        let mut split = model("glm-4.5-air");
        split.parts = 2;
        split.pinned = true;
        split.size_bytes = 78_920_000_000;
        split.blake3 = None; // split entries carry per-part manifests
        let mut used = model("qwen3-4b");
        used.size_bytes = 2_500_000_000;
        used.last_used_unix = 1_756_200_000;

        // 2 hours after the qwen load.
        let got = render_table(&[split, used], 1_756_200_000 + 7200);
        let want = "\
NAME           SIZE  PARTS  PIN  LAST USED  HASH\n\
glm-4.5-air   79 GB      2  *    never      -\n\
qwen3-4b     2.5 GB      1       2h ago     abcdef012345\n";
        assert_eq!(got, want);
    }

    #[test]
    fn never_used_unpinned_single_file_is_the_quiet_row() {
        // A fresh pull: no pin marker, "never", full hash prefix.
        let got = render_table(&[model("m")], 42);
        let lines: Vec<&str> = got.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].contains("never"));
        assert!(!lines[1].contains('*'));
        assert!(lines[1].contains("abcdef012345"));
    }
}
