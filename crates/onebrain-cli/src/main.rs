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
        /// Speculative decoding: also load a draft model (from --draft or
        /// the config's [perf] draft_model) that proposes tokens the target
        /// verifies.
        #[arg(long)]
        speculative: bool,
        /// Draft model reference for speculative decoding (implies
        /// --speculative). Must share the target's vocabulary.
        #[arg(long)]
        draft: Option<String>,
    },
    /// Show topology, active plan, endpoint, and API token.
    Status,
    /// Download a model (registry name or hf:org/repo/file.gguf).
    Pull { reference: String },
    /// List cached models and per-node footprint.
    Ls,
    /// Remove a cached model.
    Rm { reference: String },
    /// Protect a cached model from automatic cache eviction.
    Pin { model: String },
    /// Make a pinned model evictable again.
    Unpin { model: String },
    /// Re-profile this node and its links; print the report.
    Bench {
        /// Also ask every connected peer for a fresh microbench and time an
        /// end-to-end generation against the constructed M3 baseline and a
        /// solo run (markdown report; --json for machines).
        #[arg(long)]
        cluster: bool,
    },
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
            speculative,
            draft,
        } => commands::run::run(
            &model,
            ctx,
            explain,
            nodes,
            speculative,
            draft.as_deref(),
            cli.json,
        ),
        Command::Status => commands::status::run(cli.json),
        Command::Pull { reference } => commands::pull::run(&reference, cli.json),
        Command::Ls => commands::ls::run(cli.json),
        Command::Rm { reference } => commands::rm::run(&reference, cli.json),
        Command::Pin { model } => commands::pin::run(&model, true, cli.json),
        Command::Unpin { model } => commands::pin::run(&model, false, cli.json),
        Command::Stop => commands::stop::run(cli.json),
        Command::InternalDaemon => match onebraind::runtime::run_blocking() {
            Ok(()) => Ok(()),
            Err(e) => Err(commands::CliError(e.to_string())),
        },
        Command::Pair { target, code } => {
            commands::pair::run(target.as_deref(), code.as_deref(), cli.json)
        }
        Command::Unpair { name } => commands::unpair::run(&name, cli.json),
        Command::Bench { cluster } => commands::bench::run(cli.json, cluster),
    };

    if let Err(err) = outcome {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// clap's own consistency check: conflicting flags, bad defaults and
    /// the like fail here instead of at the first user's `--help`.
    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn pin_and_unpin_parse_their_model_argument() {
        let cli = Cli::try_parse_from(["onebrain", "pin", "qwen3-4b"]).unwrap();
        match cli.command {
            Some(Command::Pin { model }) => assert_eq!(model, "qwen3-4b"),
            _ => panic!("expected Pin"),
        }
        let cli = Cli::try_parse_from(["onebrain", "unpin", "glm-4.5-air"]).unwrap();
        match cli.command {
            Some(Command::Unpin { model }) => assert_eq!(model, "glm-4.5-air"),
            _ => panic!("expected Unpin"),
        }
        // The model argument is mandatory — a bare `pin` is a usage error.
        assert!(Cli::try_parse_from(["onebrain", "pin"]).is_err());
    }

    #[test]
    fn run_parses_speculative_and_draft() {
        let cli = Cli::try_parse_from(["onebrain", "run", "qwen3-4b", "--speculative"]).unwrap();
        match cli.command {
            Some(Command::Run {
                model,
                speculative,
                draft,
                ..
            }) => {
                assert_eq!(model, "qwen3-4b");
                assert!(speculative);
                assert!(draft.is_none());
            }
            _ => panic!("expected Run"),
        }
        // --draft alone implies speculative daemon-side; the CLI just
        // forwards both fields.
        let cli =
            Cli::try_parse_from(["onebrain", "run", "qwen3-4b", "--draft", "qwen3-0.6b"]).unwrap();
        match cli.command {
            Some(Command::Run {
                speculative, draft, ..
            }) => {
                assert!(!speculative);
                assert_eq!(draft.as_deref(), Some("qwen3-0.6b"));
            }
            _ => panic!("expected Run"),
        }
        // A bare run keeps both off.
        let cli = Cli::try_parse_from(["onebrain", "run", "m"]).unwrap();
        match cli.command {
            Some(Command::Run {
                speculative, draft, ..
            }) => {
                assert!(!speculative);
                assert!(draft.is_none());
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn bench_parses_the_cluster_flag() {
        let cli = Cli::try_parse_from(["onebrain", "bench"]).unwrap();
        match cli.command {
            Some(Command::Bench { cluster }) => assert!(!cluster),
            _ => panic!("expected Bench"),
        }
        let cli = Cli::try_parse_from(["onebrain", "bench", "--cluster", "--json"]).unwrap();
        assert!(cli.json);
        match cli.command {
            Some(Command::Bench { cluster }) => assert!(cluster),
            _ => panic!("expected Bench"),
        }
    }

    #[test]
    fn global_json_flag_reaches_subcommands() {
        let cli = Cli::try_parse_from(["onebrain", "pin", "m", "--json"]).unwrap();
        assert!(cli.json);
        let cli = Cli::try_parse_from(["onebrain", "ls"]).unwrap();
        assert!(!cli.json);
    }
}
