//! `onebrain self-update` — the CLI face of [`crate::update`]: assembles
//! the real-world config (GitHub, this executable, cosign-if-present),
//! refuses to swap under a running daemon, and renders the outcome for
//! humans or machines. Everything decision-shaped lives (and is tested)
//! in `crate::update`; this file is deliberately thin glue.

use onebraind::paths::AppPaths;

use super::CliError;
use crate::client::{ClientError, DaemonClient};
use crate::update::{self, CosignStatus, UpdateConfig, UpdateOutcome};

/// The repository self-update tracks. A rebrand or fork edits this and
/// the workspace `repository` key together.
const REPO: &str = "VantaBluee/onebrain";

pub fn run(check: bool, allow_downgrade: bool, json: bool) -> Result<(), CliError> {
    let cfg = config(allow_downgrade)?;

    if check {
        let outcome = update::check(&cfg)?;
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "current": outcome.current,
                    "latest": outcome.latest,
                    "update_available": outcome.update_available,
                    "downgrade": outcome.downgrade,
                    "asset": outcome.asset,
                })
            );
        } else {
            println!("current: v{}", outcome.current);
            println!("latest:  v{}", outcome.latest);
            if outcome.update_available {
                match &outcome.asset {
                    Some(asset) => {
                        println!("update available — `onebrain self-update` installs {asset}")
                    }
                    None => println!(
                        "update available, but the release has no asset for this platform \
                         ({}); install it manually from the release page",
                        cfg.triple
                    ),
                }
            } else if outcome.downgrade {
                println!(
                    "the latest release is older than this build — nothing to do \
                     (`--allow-downgrade` would install it)"
                );
            } else {
                println!("up to date");
            }
        }
        return Ok(());
    }

    refuse_if_daemon_running()?;
    let outcome = update::perform(&cfg, |line| {
        if !json {
            println!("{line}");
        }
    })?;

    match outcome {
        UpdateOutcome::UpToDate { current } => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({ "status": "up-to-date", "version": current })
                );
            } else {
                println!("already up to date (v{current})");
            }
        }
        UpdateOutcome::Installed(installed) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "installed",
                        "from": installed.from,
                        "to": installed.to,
                        "asset": installed.asset,
                        "exe": installed.exe.display().to_string(),
                        "cosign": match &installed.cosign {
                            CosignStatus::Verified => "verified".to_string(),
                            CosignStatus::Skipped(reason) => format!("skipped: {reason}"),
                        },
                        "old_kept": installed.old_kept.as_ref().map(|p| p.display().to_string()),
                    })
                );
            } else {
                println!(
                    "installed v{} (was v{}) at {}",
                    installed.to,
                    installed.from,
                    installed.exe.display()
                );
                if let CosignStatus::Skipped(reason) = &installed.cosign {
                    println!("cosign: skipped ({reason})");
                }
                if let Some(old) = &installed.old_kept {
                    println!(
                        "note: the previous binary is parked at {} (Windows keeps a running \
                         exe locked); the next self-update removes it, or delete it once this \
                         process exits",
                        old.display()
                    );
                }
            }
        }
    }
    Ok(())
}

/// Build the production [`UpdateConfig`]. `ONEBRAIN_UPDATE_API` overrides
/// the API origin — the hook the sim/e2e harness uses to point self-update
/// at a fixture server, mirroring how the tests inside `crate::update` do.
fn config(allow_downgrade: bool) -> Result<UpdateConfig, CliError> {
    let current_exe = std::env::current_exe().map_err(|e| {
        CliError(format!(
            "could not determine this executable's path ({e}); update manually from the \
             releases page"
        ))
    })?;
    let api_base = std::env::var("ONEBRAIN_UPDATE_API")
        .unwrap_or_else(|_| "https://api.github.com".to_string());
    Ok(UpdateConfig {
        api_base,
        repo: REPO.to_string(),
        triple: update::host_triple(),
        current_version: env!("CARGO_PKG_VERSION").to_string(),
        current_exe,
        allow_downgrade,
        cosign: update::find_in_path("cosign"),
    })
}

/// A live daemon serves requests from the very file about to be replaced,
/// so the contract (docs/product.md §3) is refuse-with-remedy, never
/// stop-it-for-you. Anything short of a confirmed live daemon proceeds:
/// stale run state, unreadable token — the swap cannot hurt a process
/// that is not there.
fn refuse_if_daemon_running() -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;
    match DaemonClient::from_paths(&paths) {
        Ok(client) => match client.status() {
            Ok(_) => Err(CliError(format!(
                "the daemon is running (pid {}); run `onebrain stop` first, then \
                 `onebrain self-update`",
                client.state().pid
            ))),
            Err(_) => Ok(()),
        },
        Err(ClientError::NotRunning { .. }) => Ok(()),
        Err(_) => Ok(()),
    }
}
