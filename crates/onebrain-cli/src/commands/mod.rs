pub mod doctor;
pub mod ls;
pub mod pair;
pub mod pull;
pub mod rm;
pub mod run;
pub mod status;
pub mod stop;
pub mod unpair;
pub mod up;
pub mod version;

use std::fmt;

use crate::client::ClientError;

#[derive(Debug)]
pub struct CliError(pub String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

impl From<ClientError> for CliError {
    fn from(e: ClientError) -> CliError {
        CliError(e.to_string())
    }
}

impl From<onebraind::DaemonError> for CliError {
    fn from(e: onebraind::DaemonError) -> CliError {
        CliError(e.to_string())
    }
}

/// Uniform message for commands whose milestone hasn't landed yet.
pub fn not_yet(command: &str, milestone: &str) -> Result<(), CliError> {
    Err(CliError(format!(
        "`onebrain {command}` is not implemented yet; it arrives in milestone {milestone}. \
         Track progress in STATUS.md."
    )))
}

/// Format a byte count for humans: decimal units, one decimal below 10
/// (e.g. `397 MB`, `5.4 GB`, `999 B`).
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    if bytes < 1000 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if value < 10.0 {
        format!("{value:.1} {}", UNITS[unit])
    } else {
        format!("{value:.0} {}", UNITS[unit])
    }
}

/// Node display name: the configured `node_name` when set, else the OS
/// hostname (best-effort via environment).
pub fn node_name(paths: &onebraind::paths::AppPaths) -> String {
    if let Ok(cfg) = onebraind::config::Config::load(&paths.config_file()) {
        if let Some(name) = cfg.node_name {
            return name;
        }
    }
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "this device".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_formats_across_units() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1_000), "1.0 KB");
        assert_eq!(human_bytes(1_500), "1.5 KB");
        assert_eq!(human_bytes(999_000), "999 KB");
        assert_eq!(human_bytes(397_000_000), "397 MB");
        assert_eq!(human_bytes(5_368_709_120), "5.4 GB");
        assert_eq!(human_bytes(2_000_000_000_000), "2.0 TB");
        // TB is the cap: never panics, just grows the number.
        assert_eq!(human_bytes(u64::MAX), "18446744 TB");
    }
}
