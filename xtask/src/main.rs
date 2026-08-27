//! Repo automation, invoked as `cargo xtask <command>`.

mod dist;
mod e2e;
mod pair_sim;
mod sim;
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
    /// Run the M1 end-to-end rehearsal: build, sandboxed daemon, both API
    /// dialects streaming, kill -9 recovery, graceful stop.
    E2e,
    /// Run the M2 mesh rehearsal: two sandboxed daemons pair via the
    /// internal API (ticket + code), report RTT/bandwidth, and degrade on
    /// unpair.
    PairSim {
        /// Linux + root only: run the daemons in network namespaces joined
        /// by a veth pair shaped with tc netem to 1gbit / 0.5ms, and assert
        /// the measured bandwidth/RTT land in the contract's sanity bands.
        /// Prints SKIP and exits 0 elsewhere.
        #[arg(long)]
        netem: bool,
    },
    /// Run the M3 distributed-inference rehearsal: two memory-capped
    /// daemons auto-distribute a tiny model (PipelineParallel), greedy
    /// tokens match an uncapped solo run byte-for-byte, a socket scan
    /// proves loopback-only listeners, and `--nodes 2` forces distribution.
    Sim {
        /// Linux + root only: run the daemons inside netem-shaped network
        /// namespaces (1 Gbit / 0.5 ms per direction, the pair-sim
        /// machinery). Prints SKIP and exits 0 elsewhere.
        #[arg(long)]
        netem: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Dist => dist::run(),
        Command::Smoke => smoke::run(),
        Command::E2e => e2e::run(),
        Command::PairSim { netem } => pair_sim::run(netem),
        Command::Sim { netem } => sim::run(netem),
    }
}

/// Workspace root (parent of xtask/).
pub(crate) fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level under the workspace root")
        .to_path_buf()
}
