//! The `onebrain` CLI. Command surface is fixed from M0 (product spec §7);
//! commands light up as their milestones land, and unimplemented ones say
//! exactly which milestone brings them instead of pretending.

mod client;
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
    Pair {
        /// Ticket or 6-digit code from the host device. Omit to host a
        /// pairing window on this device instead.
        target: Option<String>,
        /// 6-digit code to go with a ticket (otherwise prompted).
        #[arg(long)]
        code: Option<String>,
    },
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
    /// Internal: run the daemon in the foreground (`onebrain up` spawns
    /// this detached; never invoke it by hand).
    #[command(name = "__daemon", hide = true)]
    InternalDaemon,
}

fn main() {
    let cli = Cli::parse();

    if cli.version || cli.command.is_none() {
        commands::version::run(cli.json);
        return;
    }

    let outcome = match cli.command.expect("checked above") {
        Command::Doctor => commands::doctor::run(cli.json),
        Command::Up => commands::up::run(cli.json),
        Command::Run {
            model,
            explain,
            nodes,
            ctx,
        } => commands::run::run(&model, ctx, explain, nodes, cli.json),
        Command::Status => commands::status::run(cli.json),
        Command::Pull { reference } => commands::pull::run(&reference, cli.json),
        Command::Ls => commands::ls::run(cli.json),
        Command::Rm { reference } => commands::rm::run(&reference, cli.json),
        Command::Stop => commands::stop::run(cli.json),
        Command::InternalDaemon => match onebraind::runtime::run_blocking() {
            Ok(()) => Ok(()),
            Err(e) => Err(commands::CliError(e.to_string())),
        },
        Command::Pair { target, code } => {
            commands::pair::run(target.as_deref(), code.as_deref(), cli.json)
        }
        Command::Unpair { name } => commands::unpair::run(&name, cli.json),
        Command::Bench => commands::not_yet("bench", "M4"),
    };

    if let Err(err) = outcome {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
