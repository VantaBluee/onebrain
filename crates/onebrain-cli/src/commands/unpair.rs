//! `onebrain unpair <name>`: revoke a pairing by peer name via
//! `POST /api/internal/unpair`. The peer store is re-read on every mesh
//! accept, so revocation takes effect without a daemon restart.

use onebraind::paths::AppPaths;

use super::CliError;
use crate::client::{ClientError, DaemonClient};

pub fn run(name: &str, json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;
    let client = DaemonClient::from_paths(&paths).map_err(not_running)?;
    if let Err(e) = client.status() {
        return Err(not_running(e));
    }

    match client.unpair(name) {
        Ok(_) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({ "status": "unpaired", "name": name })
                );
            } else {
                println!("unpaired {name}; that device can no longer connect to this one.");
            }
            Ok(())
        }
        // Strip the "HTTP nnn" wrapper: the daemon's message already reads
        // well on its own (unknown names list the known ones).
        Err(ClientError::Api { message, .. }) if !message.trim().is_empty() => {
            Err(CliError(message))
        }
        Err(e) => Err(e.into()),
    }
}

/// Unpair needs the daemon (the peer store is its state); explain how to
/// get one rather than mutating files behind its back.
fn not_running(e: ClientError) -> CliError {
    CliError(format!(
        "daemon not running; run `onebrain up`, then retry ({e})"
    ))
}
