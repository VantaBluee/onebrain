//! The OneBrain daemon: per-session role logic (head or worker), config and
//! state management, and process supervision.
//!
//! M1 scope: the long-running single-node daemon — API token, single-
//! instance lock, the engine-host thread, the internal control API, and the
//! runtime that wires them to the public HTTP gateway. M2 adds the mesh
//! service (device identity, pairing, peers) and its internal endpoints.

pub mod cluster;
pub mod config;
pub mod engine_host;
pub mod lock;
pub mod logistics;
pub mod paths;
pub mod power;
pub mod runtime;
pub mod server;
pub mod supervisor;
pub mod token;

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
    #[error(
        "failed to read the API token at {path}: {source}. Fix the file's \
         permissions, or delete it to generate a fresh token on next start."
    )]
    TokenRead {
        path: String,
        source: std::io::Error,
    },
    #[error(
        "failed to write the API token at {path}: {source}. Check the \
         directory exists and is writable."
    )]
    TokenWrite {
        path: String,
        source: std::io::Error,
    },
    #[error(
        "the API token file at {path} is malformed (expected 64 lowercase \
         hex characters). Delete it and restart the daemon to generate a \
         fresh token."
    )]
    TokenInvalid { path: String },
    #[error(
        "the OS random number generator failed ({0}); retry, and if it \
         persists run `onebrain doctor`"
    )]
    Entropy(String),
    #[error("another onebrain daemon is already running (onebrain status)")]
    AlreadyRunning,
    #[error(
        "failed to open or lock {path}: {source}. Check the directory is \
         writable and no other program holds the file."
    )]
    LockIo {
        path: String,
        source: std::io::Error,
    },
    #[error(
        "failed to write daemon run state at {path}: {source}. Check the \
         directory is writable."
    )]
    RunStateWrite {
        path: String,
        source: std::io::Error,
    },
    #[error(
        "failed to read daemon run state at {path}: {source}. The daemon \
         may not be running; start it with `onebrain up`."
    )]
    RunStateRead {
        path: String,
        source: std::io::Error,
    },
    #[error(
        "daemon run state at {path} is not valid JSON: {source}. Delete the \
         file; the daemon rewrites it at startup."
    )]
    RunStateParse {
        path: String,
        source: serde_json::Error,
    },
    #[error(
        "failed to create data directory {path}: {source}. Check the \
         directory is writable, or point ONEBRAIN_HOME somewhere writable."
    )]
    DataDir {
        path: String,
        source: std::io::Error,
    },
    #[error(
        "failed to bind the API address {addr}: {source}. Free the port or \
         change `api_bind` in config.toml."
    )]
    Bind {
        addr: String,
        source: std::io::Error,
    },
    #[error(
        "failed to build the async runtime: {source}. This machine may be \
         out of resources; retry after closing other programs."
    )]
    Runtime { source: std::io::Error },
    #[error(
        "the HTTP server failed while running: {source}. Restart with \
         `onebrain up`; `onebrain doctor` shows the log locations."
    )]
    Serve { source: std::io::Error },
    // The mesh error's own message carries the remedy (docs/mesh.md).
    #[error("the mesh service failed to start: {source}")]
    Mesh {
        #[source]
        source: onebrain_mesh::MeshError,
    },
}
