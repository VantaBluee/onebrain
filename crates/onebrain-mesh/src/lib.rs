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
//! Application close codes used on mesh connections and streams are listed
//! authoritatively in `onebrain_proto::message` (0 normal, 1 unpaired,
//! 2 no-pairing-window, 3 incompatible, 4 bad-epoch).
//!
//! # Typed streams (M3)
//!
//! Every mesh bi-stream starts with a postcard
//! [`onebrain_proto::message::StreamHeader`] frame declaring its
//! [`StreamKind`] and epoch. `Control` streams carry length-prefixed
//! [`Envelope`]s (Hello, heartbeats — wire content unchanged after the
//! header — and plan traffic); `Rpc` streams carry one opaque tunneled GGML
//! RPC session and are delivered to the daemon via
//! [`MeshHandle::incoming_rpc`]; `Probe` is reserved. Control messages other
//! than heartbeats flow to [`MeshHandle::incoming_control`]; sending uses
//! [`MeshHandle::send_control`], which opens a short-lived `Control` stream
//! per message (the persistent heartbeat stream stays heartbeat-only).
//!
//! # Peer events (M5)
//!
//! [`MeshHandle::peer_events`] hands out a single-consumer stream of
//! [`PeerEvent`]s: one per live peer-state transition (`Connected`,
//! `Suspect`, `Down`, `Incompatible`), plus a [`PeerState::Draining`] event
//! whenever a proto `Draining` envelope arrives from that peer over control
//! (the envelope itself is still forwarded to the control consumer — the
//! daemon needs its epoch). The daemon's failure lifecycle keys off this
//! stream (docs/resilience.md).
//!
//! # Reconnection across restarts (M2 DoD)
//!
//! The peer store (`peers.toml`) persists each peer's last-known
//! addressing — `direct_addrs` (socket addresses) and an optional
//! `relay_url` — alongside its name. It is written:
//!
//! - at pairing time: the joiner stores the host's addresses from the
//!   ticket (or mDNS candidate) it dialed; the host stores the joiner's
//!   observed source address from the pairing connection's QUIC paths;
//! - on every established mesh session, on BOTH the dial and accept side,
//!   from the remote addresses of the connection's live QUIC paths
//!   (`Connection::paths()`), refreshed before the peer is reported
//!   `Connected`.
//!
//! A reconnect loop in the service task runs every ~3 s (2–4 s, jittered):
//! each stored peer with no live session is redialed with an
//! `EndpointAddr` assembled from the persisted addressing plus any live
//! mDNS/pairing hints. Dial failures are debug-level and back off per peer
//! (3 s doubling to a 30 s cap); the backoff resets when a connection is
//! established in either direction or a new address is learned. The
//! duplicate-session tiebreak is unchanged: on simultaneous connect, the
//! connection dialed by the lower endpoint id survives.
//!
//! Hermetic mode (mDNS and relays disabled) therefore reconnects purely
//! from the stored direct addresses: this works as long as peers rebind
//! the same UDP port across restarts (pin it via [`MeshConfig::bind_addrs`])
//! or keep a stable NAT binding. Stores written before these fields existed
//! load fine — the addressing simply starts empty and fills in on the next
//! pairing or session.
//!
//! Entry point: [`MeshService::spawn`] returns a [`MeshHandle`] whose async
//! methods (`pair_start`, `pair_join`, `peers`, `unpair`, `probe`,
//! `open_stream`, `send_control`, `shutdown`) drive the service task.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use onebrain_proto::message::DeviceBrief;
use onebrain_proto::plan::NodeId;

pub mod identity;
mod pairing;
mod service;
pub mod store;

pub use iroh::endpoint::{RecvStream, SendStream};
pub use service::{
    ControlMessage, IncomingRpcStream, MeshHandle, MeshService, PairEvent, PairWindow, PeerEvent,
    PeerInfo, ALPN_MESH, ALPN_PAIR,
};

/// This node's `NodeStatus` payload as supplied by the daemon's provider:
/// schedulable memory (measured free minus OS reserve — never total RAM),
/// the device inventory, and — since M4 — the optional compute/disk
/// microbench figures from the persisted device profile. The profile fields
/// are `None` until the node has run `onebrain bench` (the scheduler then
/// falls back to memory-only weighting, docs/scheduler-v1.md).
///
/// `Default` is the empty report (no memory, no devices, unprofiled, not
/// draining) so construction sites can spell only the fields they know via
/// `..Default::default()` as the report grows across milestones.
#[derive(Debug, Clone, Default)]
pub struct NodeStatusReport {
    pub usable_memory_bytes: u64,
    pub devices: Vec<DeviceBrief>,
    /// Measured prefill throughput (tokens/sec), if profiled.
    pub prefill_tps: Option<f64>,
    /// Measured decode throughput (tokens/sec), if profiled.
    pub decode_tps: Option<f64>,
    /// Measured sequential disk read rate (MB/s), if profiled.
    pub disk_mbps: Option<f64>,
    /// `true` while this node's battery policy asks new plans to avoid it
    /// (M5, docs/resilience.md). `false` for desktops / on AC.
    pub draining: bool,
}

/// Provider for this node's `NodeStatus` message. Called once per
/// established mesh session, right after a compatible `Hello`, so peers
/// learn this node's schedulable memory and (when benched) its profile.
pub type NodeStatusFn = Arc<dyn Fn() -> NodeStatusReport + Send + Sync>;

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
    /// `incoming_rpc`/`incoming_control`/`peer_events` called twice.
    #[error(
        "the incoming {what} receiver was already taken; only one daemon task may consume \
         mesh {what} traffic — restart the daemon if this recurs"
    )]
    ConsumerTaken {
        /// Which receiver ("rpc", "control", or "peer-events").
        what: &'static str,
    },
}

/// Configuration for [`MeshService::spawn`].
#[derive(Clone)]
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
    /// (all interfaces). Tests bind `127.0.0.1:0` for hermetic loopback
    /// runs. Pinning a fixed port here keeps the addresses persisted in
    /// peers' stores valid across daemon restarts, which is what lets
    /// hermetic (no-mDNS, no-relay) deployments reconnect automatically.
    pub bind_addrs: Vec<std::net::SocketAddr>,
    /// When set, a `NodeStatus` message built from this provider is sent on
    /// every established mesh session (after a compatible `Hello`), so peers
    /// can budget this node into placement plans. `None` (default) sends
    /// nothing — used by tests and non-daemon embedders.
    pub node_status: Option<NodeStatusFn>,
}

impl std::fmt::Debug for MeshConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshConfig")
            .field("enable_mdns", &self.enable_mdns)
            .field("enable_relays", &self.enable_relays)
            .field("engine_build", &self.engine_build)
            .field("pair_window", &self.pair_window)
            .field("bind_addrs", &self.bind_addrs)
            .field("node_status", &self.node_status.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

impl Default for MeshConfig {
    fn default() -> Self {
        MeshConfig {
            enable_mdns: true,
            enable_relays: true,
            engine_build: "dev".to_string(),
            pair_window: Duration::from_secs(120),
            bind_addrs: Vec::new(),
            node_status: None,
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
    /// The peer announced a polite drain: a proto `Draining` envelope
    /// arrived on control (battery policy or `onebrain stop` — M5,
    /// docs/resilience.md). Surfaced through [`MeshHandle::peer_events`]
    /// only; `peers()` keeps reporting the session-derived state (usually
    /// still [`PeerState::Connected`] until the peer actually goes away).
    /// Planners treat a draining peer like a dead one when building NEW
    /// plans; no QUIC close code is involved (the close-code list in
    /// `onebrain_proto::message` is unchanged — draining is an envelope,
    /// not a close).
    Draining,
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
    /// Schedulable memory the peer reported in its last `NodeStatus`
    /// (measured free minus OS reserve — never total RAM). `None` until the
    /// peer sends one; retains the last value across reconnects.
    pub usable_memory_bytes: Option<u64>,
    /// Prefill throughput (tokens/sec) from the peer's last `NodeStatus`
    /// — its compute-microbench result, `None` until the peer has benched.
    /// Like `usable_memory_bytes`, each `NodeStatus` overwrites it whole.
    pub prefill_tps: Option<f64>,
    /// Decode throughput (tokens/sec) from the peer's last `NodeStatus`.
    pub decode_tps: Option<f64>,
    /// Sequential disk read rate (MB/s) from the peer's last `NodeStatus`
    /// (page-cache upper bound; relative ordering only).
    pub disk_mbps: Option<f64>,
    /// `true` when the peer's last `NodeStatus` advertised battery drain
    /// (M5, docs/resilience.md): the scheduler excludes it from new plans
    /// unless a plan is infeasible without it. Overwritten whole by every
    /// `NodeStatus`, like the profile fields; `false` until one arrives.
    #[serde(default)]
    pub draining: bool,
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
