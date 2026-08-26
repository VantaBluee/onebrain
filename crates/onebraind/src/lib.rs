//! The OneBrain daemon: per-session role logic (head or worker), config and
//! state management, and process supervision.
//!
//! M0 scope: platform paths, the config file, and the vocabulary. The long-
//! running daemon (API server, mesh endpoint, engine host) starts in M1/M2.

pub mod config;
pub mod paths;

/// A node's role in the current cluster session. Roles are per-session, not
/// per-install: the same binary serves both (§3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The device the user is driving: scheduler, API gateway, dashboard.
    Head,
    /// Executor for a head's plans.
    Worker,
}

#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
    #[error("could not determine platform config directory for {product}", product = onebrain_proto::PRODUCT_NAME)]
    NoConfigDir,
    #[error("failed to read config at {path}: {source}")]
    ConfigRead {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to write config at {path}: {source}")]
    ConfigWrite {
        path: String,
        source: std::io::Error,
    },
    #[error("config at {path} is not valid TOML: {source}")]
    ConfigParse {
        path: String,
        source: Box<toml::de::Error>,
    },
}
