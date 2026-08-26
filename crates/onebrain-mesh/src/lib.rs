//! Mesh transport: device identities, pairing, discovery, authenticated
//! streams, link probing, and heartbeats.
//!
//! Implemented on iroh 1.x: Ed25519 device identities (dial by public key),
//! QUIC with mutual authentication, mDNS LAN discovery
//! (`iroh-mdns-address-lookup`), and an RTT/bandwidth prober. Nothing in this
//! crate ever opens an unauthenticated non-loopback listener (§1.3 of the
//! product spec): the only sockets are iroh's, and the mesh ALPN
//! (`onebrain/mesh/1`) closes connections from endpoints that are not in the
//! peer store with error code 1 (`unpaired`) — the §10 guarantee.
//!
//! Application close codes used on mesh connections:
//! - `0` — normal close (duplicate connection, shutdown, pairing done).
//! - `1` — `unpaired`: the remote endpoint is not in the peer store.
//! - `2` — `no-pairing-window`: pair ALPN connection outside an open window.
//! - `3` — `incompatible`: the `Hello` handshake judged the peers
//!   incompatible (protocol or engine build mismatch).
//!
//! Entry point: [`MeshService::spawn`] returns a [`MeshHandle`] whose async
//! methods (`pair_start`, `pair_join`, `peers`, `unpair`, `probe`,
//! `shutdown`) drive the service task.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use onebrain_proto::plan::NodeId;

pub mod identity;
mod pairing;
mod service;
pub mod store;

pub use service::{MeshHandle, MeshService, PairEvent, PairWindow, PeerInfo, ALPN_MESH, ALPN_PAIR};

/// Errors from the mesh service. Every user-facing variant carries a one-line
/// remedy in its message (§12 of the product spec).
#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    /// The device key file exists but could not be read.
    #[error(
        "failed to read device key {path}: {source}; check permissions on the config directory"
    )]
    IdentityRead {
        /// Path of the device key file.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The device key file does not contain 64 hex characters.
    #[error(
        "device key {path} is malformed (expected 64 hex chars); restore it from a backup, or \
         delete the file to generate a NEW identity — deleting unpairs this device everywhere"
    )]
    IdentityMalformed {
        /// Path of the device key file.
        path: PathBuf,
    },
    /// The device key file could not be created or written.
    #[error(
        "failed to write device key {path}: {source}; check permissions on the config directory"
    )]
    IdentityWrite {
        /// Path of the device key file.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The operating system random number generator failed.
    #[error(
        "the OS random number generator failed: {0}; this machine cannot generate keys safely"
    )]
    Rng(String),
    /// The peer store file could not be read.
    #[error(
        "failed to read peer store {path}: {source}; check permissions on the config directory"
    )]
    StoreRead {
        /// Path of `peers.toml`.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The peer store file is not valid TOML.
    #[error(
        "peer store {path} is not valid TOML: {source}; fix the file, or delete it to unpair \
         all peers"
    )]
    StoreParse {
        /// Path of `peers.toml`.
        path: PathBuf,
        /// Underlying TOML error (boxed: it is large).
        #[source]
        source: Box<toml::de::Error>,
    },
    /// The peer store file could not be written.
    #[error("failed to write peer store {path}: {source}; check permissions and free disk space")]
    StoreWrite {
        /// Path of `peers.toml`.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// `unpair`/`probe` named a peer that is not in the store.
    #[error(
        "no paired peer named {name:?}; known peers: {}",
        if known.is_empty() { "(none)".to_string() } else { known.join(", ") }
    )]
    UnknownPeerName {
        /// The name that was requested.
        name: String,
        /// Names currently in the store.
        known: Vec<String>,
    },
    /// The iroh endpoint could not be bound.
    #[error("failed to bind the mesh endpoint: {0}; check that UDP is not blocked by a firewall")]
    Bind(String),
    /// A dial to a remote endpoint failed.
    #[error(
        "could not reach {target}: {detail}; check that the other device is online and on the \
         same network, or pair with a ticket instead of a code"
    )]
    Connect {
        /// Short form of the target endpoint id.
        target: String,
        /// Underlying connect error.
        detail: String,
    },
    /// The pairing exchange was rejected (wrong code, or a protocol error).
    #[error("pairing failed: {reason}; check the 6-digit code and try again")]
    PairRejected {
        /// Why the exchange failed.
        reason: String,
    },
    /// A pairing window is already open on this device.
    #[error("a pairing window is already open; wait for it to close (120 s) or pair now")]
    WindowAlreadyOpen,
    /// `pair_join` got a ticket but no code.
    #[error("a ticket needs a code: pass the 6-digit code shown next to it on the host")]
    CodeRequired,
    /// The pairing target is neither a valid ticket nor a 6-digit code.
    #[error(
        "invalid pairing input {input:?}: expected an endpoint ticket or a 6-digit code; \
         re-copy the ticket from `onebrain pair` on the host"
    )]
    BadPairTarget {
        /// The rejected input (possibly truncated).
        input: String,
    },
    /// Code-only pairing found no LAN candidates to try.
    #[error(
        "no pairing candidates found on the LAN; make sure both devices are on the same \
         network, or use the ticket instead of the code"
    )]
    NoCandidates,
    /// `probe` was called for a peer without a live mesh connection.
    #[error(
        "peer {name:?} has no live mesh connection to probe; wait for it to connect and retry"
    )]
    NotConnected {
        /// Name of the peer.
        name: String,
    },
    /// A QUIC stream failed mid-exchange.
    #[error("the mesh stream failed: {detail}; the connection will be retried automatically")]
    Stream {
        /// Underlying stream error.
        detail: String,
    },
    /// Encoding or decoding a wire message failed.
    #[error("protocol error: {0}; both devices may need the same OneBrain version")]
    Proto(#[from] onebrain_proto::ProtoError),
    /// An exchange did not complete in time.
    #[error("{what} timed out after {secs}s; check the network link between the devices")]
    Timeout {
        /// What timed out.
        what: &'static str,
        /// The timeout that elapsed.
        secs: u64,
    },
    /// The mesh service task is gone.
    #[error("the mesh service has stopped; restart the daemon (`onebrain up`)")]
    ServiceStopped,
}

/// Configuration for [`MeshService::spawn`].
#[derive(Debug, Clone)]
pub struct MeshConfig {
    /// Advertise and discover peers on the LAN via mDNS (default `true`).
    pub enable_mdns: bool,
    /// Use n0's relay + pkarr infrastructure for dial-by-key across networks
    /// (default `true`). Disabled, only direct addresses (tickets, mDNS)
    /// can establish connections — used by hermetic tests.
    pub enable_relays: bool,
    /// Engine build hash carried in the mesh `Hello` (llama.cpp commit +
    /// backend flags + proto version). Peers with different values are marked
    /// incompatible. The daemon passes the real hash; defaults to `"dev"`.
    pub engine_build: String,
    /// How long a pairing window stays open. Default 120 s per the contract;
    /// shrinkable so tests do not sleep two minutes.
    pub pair_window: Duration,
    /// Explicit UDP bind addresses. Empty (default) uses iroh's defaults
    /// (all interfaces). Tests bind `127.0.0.1:0` for hermetic loopback runs.
    pub bind_addrs: Vec<std::net::SocketAddr>,
}

impl Default for MeshConfig {
    fn default() -> Self {
        MeshConfig {
            enable_mdns: true,
            enable_relays: true,
            engine_build: "dev".to_string(),
            pair_window: Duration::from_secs(120),
            bind_addrs: Vec::new(),
        }
    }
}

/// How to reach the device we want to pair with.
#[derive(Debug, Clone)]
pub enum PairTarget {
    /// An endpoint ticket printed by `onebrain pair` on the host. Works
    /// across networks (carries relay + direct addresses).
    Ticket(String),
    /// A bare 6-digit code: candidates are discovered via mDNS on the LAN
    /// and each is tried — the PAKE makes dialing a wrong or hostile
    /// candidate safe.
    Code(String),
}

/// Live state of a paired peer, as reported by `GET /api/internal/peers`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PeerState {
    /// Paired but never seen since the daemon started.
    Unknown,
    /// Visible on the LAN (mDNS) but no live mesh session yet.
    Reachable,
    /// Live mesh session with fresh heartbeats.
    Connected,
    /// Three consecutive heartbeats missed (spec §5).
    Suspect,
    /// Ten seconds of heartbeat silence, or the session dropped (spec §5).
    Down,
    /// The `Hello` handshake failed: protocol or engine build mismatch.
    Incompatible,
}

/// One row of `peers()`: persisted store entry merged with live link state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerStatus {
    /// Human-readable name chosen at pairing time.
    pub name: String,
    /// Endpoint id string (lowercase hex of the public key).
    pub id: String,
    /// Live state.
    pub state: PeerState,
    /// EWMA (α = 0.3) of heartbeat round-trips, milliseconds.
    pub rtt_ms: Option<f64>,
    /// Last bulk-probe result, megabits per second.
    pub bandwidth_mbps: Option<f64>,
    /// Missed-heartbeat fraction over the last 100 heartbeats.
    pub loss: Option<f32>,
    /// Unix timestamp of the last heartbeat echo.
    pub last_seen_unix: Option<u64>,
}

/// Measured quality of a link between two nodes. Populated by the prober;
/// consumed by the scheduler for boundary placement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinkProfile {
    /// Round-trip time in microseconds.
    pub rtt_micros: u64,
    /// Measured bandwidth in megabits per second.
    pub bandwidth_mbps: f64,
    /// Fraction of probe packets lost (Wi-Fi warning signal, §1.7).
    pub loss: f32,
}

/// A paired peer as persisted in the peer store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Peer {
    /// Stable node identifier (endpoint public key, hex).
    pub id: NodeId,
    /// Human-readable name chosen at pairing time.
    pub name: String,
}
