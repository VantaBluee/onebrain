//! `cargo xtask smoke`: fetch (once) a tiny GGUF and run the engine's
//! CPU smoke test against it. Used locally and by CI on all three OSes.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};

/// Tiny public-domain-training-data models used by llama.cpp's own CI.
/// Tried in order; first successful download wins.
const CANDIDATES: &[(&str, &str)] = &[
    (
        "stories260K.gguf",
        "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf",
    ),
    (
        "stories15M-q4_0.gguf",
        "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf",
    ),
];

pub fn run() -> Result<()> {
    let root = crate::workspace_root();
    let cache = root.join("target-smoke");
    std::fs::create_dir_all(&cache)?;

    let model = ensure_model(&cache)?;
    println!("smoke model: {}", model.display());

    let status = Command::new("cargo")
        .current_dir(&root)
        .env("OB_SMOKE_MODEL", &model)
        .args([
            "test",
            "-p",
            "onebrain-engine",
            "smoke_generate_greedy",
            "--",
            "--nocapture",
        ])
        .status()
        .context("failed to invoke cargo test")?;
    if !status.success() {
        bail!("engine smoke test failed");
    }
    println!("smoke test passed");
    Ok(())
}

fn ensure_model(cache: &std::path::Path) -> Result<PathBuf> {
    for (name, _) in CANDIDATES {
        let path = cache.join(name);
        if path.exists() {
            return Ok(path);
        }
    }
    for (name, url) in CANDIDATES {
        let path = cache.join(name);
        let tmp = cache.join(format!("{name}.part"));
        // curl ships on Windows 10+, macOS, and the Linux CI images alike.
        let status = Command::new("curl")
            .args(["-L", "--fail", "--retry", "3", "-o"])
            .arg(&tmp)
            .arg(*url)
            .status()
            .context("failed to invoke curl")?;
        if status.success() {
            std::fs::rename(&tmp, &path)?;
            return Ok(path);
        }
        let _ = std::fs::remove_file(&tmp);
        eprintln!("download failed for {url}, trying next candidate");
    }
    bail!("could not download any smoke-test model; check network access to huggingface.co")
}
