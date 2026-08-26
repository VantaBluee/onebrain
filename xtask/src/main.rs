//! Repo automation, invoked as `cargo xtask <command>`.

mod dist;
mod smoke;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask", about = "OneBrain repo automation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build the release binary and stage per-OS distribution artifacts.
    Dist,
    /// Download (once) a tiny GGUF and run the engine smoke test on CPU.
    Smoke,
    /// Spawn a simulated multi-node cluster on this host (arrives in M3).
    Sim,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Dist => dist::run(),
        Command::Smoke => smoke::run(),
        Command::Sim => anyhow::bail!(
            "the cluster simulator arrives with milestone M3 (distributed inference); \
             see STATUS.md for progress"
        ),
    }
}

/// Workspace root (parent of xtask/).
pub(crate) fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level under the workspace root")
        .to_path_buf()
}
