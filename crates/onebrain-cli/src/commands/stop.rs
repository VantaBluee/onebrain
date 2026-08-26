//! `onebrain stop`: ask the daemon to shut down gracefully, then wait (5 s)
//! for its health endpoint to disappear. Idempotent: stopping a stopped
//! daemon reports that and succeeds.

use std::time::{Duration, Instant};

use onebraind::paths::AppPaths;

use super::CliError;
use crate::client::{ClientError, DaemonClient};

pub fn run(json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;

    let report_not_running = || {
        if json {
            println!("{}", serde_json::json!({ "status": "not_running" }));
        } else {
            println!("daemon is not running; nothing to stop.");
        }
    };

    let client = match DaemonClient::from_paths(&paths) {
        Ok(c) => c,
        Err(ClientError::NotRunning { .. }) => {
            report_not_running();
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };

    // A stale daemon.json (e.g. after kill -9) is normal; the health check
    // decides whether there is anything to stop.
    if client.status().is_err() {
        report_not_running();
        return Ok(());
    }

    match client.shutdown() {
        Ok(()) => {}
        // The daemon may exit before the response fully lands; the health
        // poll below is the real confirmation.
        Err(ClientError::Unreachable { .. }) => {}
        Err(e) => return Err(e.into()),
    }

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if client.status().is_err() {
            if json {
                println!("{}", serde_json::json!({ "status": "stopped" }));
            } else {
                println!("daemon stopped.");
            }
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(CliError(format!(
                "the daemon acknowledged shutdown but was still answering after 5s; \
                 check `onebrain status`, and as a last resort end process {}",
                client.state().pid
            )));
        }
        std::thread::sleep(Duration::from_millis(150));
    }
}
