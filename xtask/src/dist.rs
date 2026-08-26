//! `cargo xtask dist`: build the release `onebrain` binary and stage it,
//! with checksums, under `dist/onebrain-v<version>-<target>/`. CI zips the
//! staged folder per-OS; installers (msi/pkg/brew) arrive in M8.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

pub fn run() -> Result<()> {
    let root = crate::workspace_root();

    // Ask cargo for the artifact path instead of guessing the target dir
    // (local builds redirect it outside the OneDrive tree).
    let output = Command::new("cargo")
        .current_dir(&root)
        .args([
            "build",
            "--release",
            "-p",
            "onebrain-cli",
            "--message-format=json-render-diagnostics",
        ])
        .output()
        .context("failed to invoke cargo build")?;
    if !output.status.success() {
        bail!(
            "release build failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let mut binary: Option<PathBuf> = None;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        // The package is onebrain-cli but its [[bin]] target is "onebrain";
        // compiler-artifact messages carry the target name.
        let is_bin = msg["target"]["kind"]
            .as_array()
            .is_some_and(|k| k.iter().any(|v| v == "bin"));
        if msg["reason"] == "compiler-artifact"
            && is_bin
            && msg["target"]["name"] == "onebrain"
            && !msg["executable"].is_null()
        {
            binary = Some(PathBuf::from(msg["executable"].as_str().unwrap()));
        }
    }
    let binary = binary.context("cargo did not report an executable for onebrain-cli")?;

    let version = env!("CARGO_PKG_VERSION");
    let target = target_triple()?;
    let stage_name = format!("onebrain-v{version}-{target}");
    let stage = root.join("dist").join(&stage_name);
    std::fs::create_dir_all(&stage)?;

    let exe_name = binary.file_name().unwrap();
    let staged_exe = stage.join(exe_name);
    std::fs::copy(&binary, &staged_exe)
        .with_context(|| format!("copying {} into {}", binary.display(), stage.display()))?;
    for doc in ["README.md", "LICENSE-MIT", "LICENSE-APACHE"] {
        let src = root.join(doc);
        if src.exists() {
            std::fs::copy(&src, stage.join(doc))?;
        }
    }

    let digest = Sha256::digest(std::fs::read(&staged_exe)?);
    std::fs::write(
        stage.join("SHA256SUMS"),
        format!("{:x}  {}\n", digest, exe_name.to_string_lossy()),
    )?;

    println!("staged: {}", stage.display());
    Ok(())
}

fn target_triple() -> Result<String> {
    let out = Command::new("rustc").args(["-vV"]).output()?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.lines()
        .find_map(|l| l.strip_prefix("host: "))
        .map(str::to_string)
        .context("could not determine host target triple")
}
