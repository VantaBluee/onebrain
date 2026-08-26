//! The `onebrain` CLI. Command surface is fixed from M0 (product spec §7);
//! commands light up as their milestones land, and unimplemented ones say
//! exactly which milestone brings them instead of pretending.

mod commands;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "onebrain",
    about = "Run local AI models across your own computers as one machine",
    version,
    disable_version_flag = true
)]
struct Cli {
    /// Print version details (product, engine, vendored llama.cpp).
    #[arg(short = 'V', long, global = true)]
    version: bool,
    /// Emit machine-readable JSON instead of human output.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start the daemon on this device (head role).
    Up,
    /// Pair this device with another (shows a code; or pass one).
    Pair { code: Option<String> },
    /// Ensure weights, plan placement, load, and serve a model.
    Run {
        model: String,
        /// Print why the plan looks the way it does.
        #[arg(long)]
        explain: bool,
        /// Force a specific node count instead of auto-solo logic.
        #[arg(long)]
        nodes: Option<u32>,
        /// Requested context length.
        #[arg(long)]
        ctx: Option<u32>,
    },
    /// Show topology, active plan, endpoint, and API token.
    Status,
    /// Download a model (registry name or hf:org/repo/file.gguf).
    Pull { reference: String },
    /// List cached models and per-node footprint.
    Ls,
    /// Remove a cached model.
    Rm { reference: String },
    /// Re-profile this node and its links; print the report.
    Bench,
    /// Diagnose GPU/driver/firewall/version problems with remedies.
    Doctor,
    /// Stop the daemon, draining politely.
    Stop,
    /// Revoke pairing with a peer.
    Unpair { name: String },
}

fn main() {
    let cli = Cli::parse();

    if cli.version || cli.command.is_none() {
        commands::version::run(cli.json);
        return;
    }

    let outcome = match cli.command.expect("checked above") {
        Command::Doctor => commands::doctor::run(cli.json),
        Command::Up => commands::not_yet("up", "M1"),
        Command::Run { .. } => commands::not_yet("run", "M1"),
        Command::Status => commands::not_yet("status", "M1"),
        Command::Pull { .. } => commands::not_yet("pull", "M1"),
        Command::Ls => commands::not_yet("ls", "M1"),
        Command::Rm { .. } => commands::not_yet("rm", "M1"),
        Command::Stop => commands::not_yet("stop", "M1"),
        Command::Pair { .. } => commands::not_yet("pair", "M2"),
        Command::Unpair { .. } => commands::not_yet("unpair", "M2"),
        Command::Bench => commands::not_yet("bench", "M4"),
    };

    if let Err(err) = outcome {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
