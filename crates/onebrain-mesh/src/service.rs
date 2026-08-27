//! The mesh service task: iroh endpoint ownership, ALPN dispatch, pairing
//! windows, peer sessions (Hello, heartbeats, probes), and mDNS discovery.

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use iroh::endpoint::{presets, Connection, PathId, RecvStream, SendStream};
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayUrl, SecretKey, TransportAddr, Watcher};
use iroh_mdns_address_lookup::{DiscoveryEvent, MdnsAddressLookup};
use iroh_tickets::endpoint::EndpointTicket;
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{timeout, Instant};
use tracing::{debug, error, info, warn};

use onebrain_proto::capabilities::Capabilities;
use onebrain_proto::handshake::{judge, EngineBuildHash, HandshakeVerdict, Hello};
use onebrain_proto::message::{Envelope, Message, StreamHeader, StreamKind};
use onebrain_proto::plan::{Epoch, NodeId};

use crate::pairing::{
    self, generate_code, read_frame, stream_err, truncate_for_display, validate_code, write_frame,
    PairOutcome,
};
use crate::store::PeerStore;
use crate::{MeshConfig, MeshError, NodeStatusFn, PairTarget, PeerState, PeerStatus};

/// ALPN for pairing exchanges. Accepted from ANY endpoint while (and only
/// while) a pairing window is open.
pub const ALPN_PAIR: &[u8] = b"onebrain/pair/1";
/// ALPN for all paired traffic. Accepts require the remote id to be in the
/// peer store; unknown peers are closed with code 1 (`unpaired`).
pub const ALPN_MESH: &[u8] = b"onebrain/mesh/1";

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
const SUSPECT_AFTER_MISSED: u32 = 3;
const DOWN_AFTER: Duration = Duration::from_secs(10);
const RTT_EWMA_ALPHA: f64 = 0.3;
const LOSS_WINDOW: usize = 100;
const PROBE_BYTES: u64 = 4 * 1024 * 1024;
const PROBE_TIMEOUT: Duration = Duration::from_secs(60);
const PAIR_ATTEMPTS: u32 = 3;
const PAIR_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(60);
const HELLO_TIMEOUT: Duration = Duration::from_secs(15);
const ADDR_WAIT: Duration = Duration::from_secs(10);
/// Cadence of the reconnect loop: one pass every 2–4 s (3 s ± 1 s jitter, so
/// two daemons restarted together do not dial in lockstep).
const RECONNECT_TICK_BASE_MS: u64 = 2_000;
const RECONNECT_TICK_JITTER_MS: u64 = 2_000;
/// Per-peer redial backoff: first failure waits 3 s, then doubles to a 30 s
/// cap so logs stay quiet while a peer is away. Reset on success or when a
/// new address is learned (mDNS, pairing).
const RECONNECT_BACKOFF_BASE: Duration = Duration::from_secs(3);
const RECONNECT_BACKOFF_CAP: Duration = Duration::from_secs(30);
/// Bound on a single outbound mesh dial, so a peer that stays dark cannot
/// pin its `dialing` slot forever.
const DIAL_TIMEOUT: Duration = Duration::from_secs(30);

/// A newly paired (or requested) peer: id + final (deduplicated) name.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PeerInfo {
    /// Endpoint id string (lowercase hex).
    pub id: String,
    /// Name persisted in the peer store.
    pub name: String,
}

/// Progress events streamed while a pairing window is open. Terminal events
/// are `Paired`, `Expired`, and `Failed`.
#[derive(Debug, Clone)]
pub enum PairEvent {
    /// A pairing attempt started (a device dialed the pair ALPN).
    Attempt,
    /// Pairing succeeded; the peer is persisted. Terminal.
    Paired(PeerInfo),
    /// The 120-second window elapsed without a success. Terminal.
    Expired,
    /// The window closed early (attempt budget exhausted or a store
    /// failure). Terminal.
    Failed(String),
}

/// A non-heartbeat [`Envelope`] received on a peer's `Control` stream
/// (PlanProposal, PlanAck, NodeStatus, Draining, …). The `peer` is the
/// authenticated endpoint id of the sender, which the accept path guarantees
/// is in the peer store.
#[derive(Debug)]
pub struct ControlMessage {
    /// Authenticated sender (endpoint id, hex).
    pub peer: NodeId,
    /// The received envelope.
    pub envelope: Envelope,
}

/// One peer state transition, delivered to the single
/// [`MeshHandle::peer_events`] consumer (M5, docs/resilience.md): emitted on
/// every live-state change (`Connected`, `Suspect`, `Down`, `Incompatible`)
/// and, with [`PeerState::Draining`], whenever a proto `Draining` envelope
/// arrives from the peer over control (the envelope itself still reaches the
/// control consumer, epoch included).
#[derive(Debug, Clone)]
pub struct PeerEvent {
    /// The peer whose state changed (endpoint id, hex).
    pub peer: NodeId,
    /// The peer's human-readable name from the peer store.
    pub name: String,
    /// The state entered (or [`PeerState::Draining`] for a drain notice).
    pub state: PeerState,
}

/// An accepted `rpc` bi-stream, delivered to the daemon for bridging into a
/// local GGML RPC session. The mesh validates the sender is a paired peer
/// (sessions only exist for store members); the *epoch* is validated by the
/// consumer — the daemon knows the active epoch, the mesh does not — which
/// refuses stale streams with [`IncomingRpcStream::refuse`] (close code 4,
/// `bad-epoch`).
pub struct IncomingRpcStream {
    /// Authenticated sender (endpoint id, hex).
    pub peer: NodeId,
    /// Epoch declared in the stream's header.
    pub epoch: Epoch,
    /// Send half toward the peer.
    pub send: SendStream,
    /// Receive half from the peer (header already consumed).
    pub recv: RecvStream,
}

impl std::fmt::Debug for IncomingRpcStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncomingRpcStream")
            .field("peer", &self.peer)
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl IncomingRpcStream {
    /// Refuse the stream: reset/stop both halves with QUIC application error
    /// `code` (`4` = `bad-epoch` per the close-code list in
    /// `onebrain_proto::message`).
    pub fn refuse(mut self, code: u32) {
        let _ = self.send.reset(code.into());
        let _ = self.recv.stop(code.into());
    }
}

/// An open pairing window, as returned by [`MeshHandle::pair_start`].
#[derive(Debug)]
pub struct PairWindow {
    /// The 6-digit code to read to the joining device.
    pub code: String,
    /// Endpoint ticket (serialized `EndpointAddr`) for cross-network joins.
    pub ticket: String,
    /// Window progress events; closed after a terminal event.
    pub events: mpsc::Receiver<PairEvent>,
}

/// The mesh service. [`MeshService::spawn`] builds the iroh endpoint and
/// starts the service task; interact through the returned [`MeshHandle`].
#[derive(Debug)]
pub struct MeshService;

/// Cheap-to-clone async handle to the mesh service task.
#[derive(Debug, Clone)]
pub struct MeshHandle {
    tx: mpsc::Sender<Internal>,
    id: EndpointId,
}

enum Internal {
    // Commands from the handle.
    PairStart {
        reply: oneshot::Sender<Result<PairWindow, MeshError>>,
    },
    PairJoin {
        target: PairTarget,
        code: Option<String>,
        reply: oneshot::Sender<Result<PeerInfo, MeshError>>,
    },
    Peers {
        reply: oneshot::Sender<Result<Vec<PeerStatus>, MeshError>>,
    },
    Unpair {
        name: String,
        reply: oneshot::Sender<Result<(), MeshError>>,
    },
    Probe {
        name: String,
        reply: oneshot::Sender<Result<f64, MeshError>>,
    },
    TakeRpc {
        reply: oneshot::Sender<Option<mpsc::Receiver<IncomingRpcStream>>>,
    },
    TakeControl {
        reply: oneshot::Sender<Option<mpsc::Receiver<ControlMessage>>>,
    },
    TakeEvents {
        reply: oneshot::Sender<Option<mpsc::Receiver<PeerEvent>>>,
    },
    SendControl {
        peer: String,
        envelope: Envelope,
        reply: oneshot::Sender<Result<(), MeshError>>,
    },
    OpenStream {
        peer: String,
        kind: StreamKind,
        epoch: Epoch,
        reply: oneshot::Sender<Result<(SendStream, RecvStream), MeshError>>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
    // Internal events.
    IncomingPair(Connection),
    IncomingMesh(Connection),
    PairHostDone {
        window_id: u64,
        result: Result<PairOutcome, MeshError>,
        /// Remote addressing observed on the pairing connection (the
        /// joiner's source address as seen by the host), persisted on
        /// success so the host can redial the joiner after a restart.
        observed: Addressing,
    },
    WindowTimeout {
        window_id: u64,
    },
    JoinedPeer {
        peer: EndpointId,
        addr: EndpointAddr,
    },
    DialDone {
        peer: EndpointId,
        conn: Option<Connection>,
    },
    TryDial {
        peer: EndpointId,
    },
    SessionEnded {
        peer: EndpointId,
        stable_id: usize,
    },
    Discovered {
        peer: EndpointId,
        addr: EndpointAddr,
    },
    DiscoveryExpired {
        peer: EndpointId,
    },
    /// One pass of the reconnect loop (sent every 2–4 s by the ticker).
    ReconnectTick,
}

/// Last-known addressing for a peer: direct socket addresses plus an
/// optional relay URL — the persistable projection of an `EndpointAddr`.
type Addressing = (Vec<SocketAddr>, Option<String>);

/// Per-peer redial backoff state. Absent entry = dial immediately.
#[derive(Debug, Clone, Copy)]
struct Backoff {
    /// Earliest instant the next dial attempt may start.
    next_at: Instant,
    /// Delay to schedule after the NEXT failure (doubles up to the cap).
    delay: Duration,
}

/// Live per-peer link state, shared between session tasks and the service.
#[derive(Debug, Default, Clone)]
struct Live {
    state: Option<PeerState>,
    rtt_ms: Option<f64>,
    bandwidth_mbps: Option<f64>,
    loss: Option<f32>,
    last_seen_unix: Option<u64>,
    /// From the peer's last `NodeStatus`; cached here so reading a budget is
    /// never a network round trip.
    usable_memory_bytes: Option<u64>,
    /// Microbench profile from the peer's last `NodeStatus` (M4). Every
    /// `NodeStatus` overwrites all three — a peer that lost its profile
    /// truthfully reports `None` again.
    prefill_tps: Option<f64>,
    decode_tps: Option<f64>,
    disk_mbps: Option<f64>,
    /// Battery-drain flag from the peer's last `NodeStatus` (M5). Like the
    /// profile fields, overwritten whole on every `NodeStatus`.
    draining: bool,
}

type LiveMap = Arc<StdMutex<HashMap<EndpointId, Live>>>;

fn set_live(live: &LiveMap, peer: EndpointId, update: impl FnOnce(&mut Live)) {
    let mut map = live.lock().expect("live map poisoned");
    update(map.entry(peer).or_default());
}

/// Record `state` for `peer` in the live map and, when that CHANGES the
/// peer's state, emit a [`PeerEvent`] toward the `peer_events()` consumer.
/// Events are sent with `try_send`: peer transitions are rare, so the
/// buffer only fills if the consumer stalls for dozens of transitions — a
/// dropped event is logged, never blocks a session task.
fn note_transition(
    live: &LiveMap,
    events: &mpsc::Sender<PeerEvent>,
    peer: EndpointId,
    name: &str,
    state: PeerState,
) {
    let changed = {
        let mut map = live.lock().expect("live map poisoned");
        let entry = map.entry(peer).or_default();
        let changed = entry.state != Some(state);
        entry.state = Some(state);
        changed
    };
    if changed {
        emit_peer_event(events, peer, name, state);
    }
}

/// Send one [`PeerEvent`] (best effort; see [`note_transition`]).
fn emit_peer_event(
    events: &mpsc::Sender<PeerEvent>,
    peer: EndpointId,
    name: &str,
    state: PeerState,
) {
    let event = PeerEvent {
        peer: NodeId(peer.to_string()),
        name: name.to_string(),
        state,
    };
    if events.try_send(event).is_err() {
        debug!(
            peer = %peer.fmt_short(),
            ?state,
            "peer event dropped: consumer buffer full or service stopping"
        );
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

struct Window {
    id: u64,
    code: String,
    attempts_left: u32,
    deadline: Instant,
    events: mpsc::Sender<PairEvent>,
}

struct Session {
    conn: Connection,
    dialer: bool,
    runner: JoinHandle<()>,
}

struct Service {
    ep: Endpoint,
    store: PeerStore,
    node_name: String,
    cfg: MeshConfig,
    tx: mpsc::Sender<Internal>,
    rx: mpsc::Receiver<Internal>,
    live: LiveMap,
    window: Option<Window>,
    window_seq: u64,
    sessions: HashMap<EndpointId, Session>,
    dialing: HashSet<EndpointId>,
    discovered: HashMap<EndpointId, EndpointAddr>,
    known_addrs: HashMap<EndpointId, EndpointAddr>,
    /// Per-peer redial backoff, driven by [`Internal::ReconnectTick`].
    reconnect: HashMap<EndpointId, Backoff>,
    /// Background accept + mDNS loops; aborted on drop.
    background: JoinSet<()>,
    /// Sender side of the incoming-rpc-stream channel (cloned into sessions).
    rpc_tx: mpsc::Sender<IncomingRpcStream>,
    /// Receiver side, held until a daemon task takes it via `incoming_rpc`.
    rpc_rx: Option<mpsc::Receiver<IncomingRpcStream>>,
    /// Sender side of the incoming-control-message channel.
    ctrl_tx: mpsc::Sender<ControlMessage>,
    /// Receiver side, held until taken via `incoming_control`.
    ctrl_rx: Option<mpsc::Receiver<ControlMessage>>,
    /// Sender side of the peer-event channel (cloned into sessions).
    event_tx: mpsc::Sender<PeerEvent>,
    /// Receiver side, held until taken via `peer_events`.
    event_rx: Option<mpsc::Receiver<PeerEvent>>,
}

impl MeshService {
    /// Build the iroh endpoint (default n0 preset — relays + pkarr — unless
    /// disabled) and start the mesh service task.
    pub async fn spawn(
        secret_key: SecretKey,
        peer_store_path: PathBuf,
        node_name: String,
        config: MeshConfig,
    ) -> Result<MeshHandle, MeshError> {
        let mut builder = if config.enable_relays {
            Endpoint::builder(presets::N0)
        } else {
            Endpoint::builder(presets::Minimal)
        }
        .secret_key(secret_key)
        .alpns(vec![ALPN_PAIR.to_vec(), ALPN_MESH.to_vec()]);
        if !config.bind_addrs.is_empty() {
            builder = builder.clear_ip_transports();
            for addr in &config.bind_addrs {
                builder = builder
                    .bind_addr(*addr)
                    .map_err(|e| MeshError::Bind(e.to_string()))?;
            }
        }
        let ep = builder
            .bind()
            .await
            .map_err(|e| MeshError::Bind(e.to_string()))?;
        let id = ep.id();
        info!(endpoint = %id.fmt_short(), "mesh endpoint bound");

        wait_for_local_addrs(&ep).await;

        let (tx, rx) = mpsc::channel(64);
        let store = PeerStore::new(peer_store_path);
        let mut background = JoinSet::new();
        background.spawn(accept_loop(ep.clone(), store.clone(), tx.clone()));
        background.spawn(reconnect_ticker(tx.clone()));

        if config.enable_mdns {
            match MdnsAddressLookup::builder().build(ep.id()) {
                Ok(mdns) => match ep.address_lookup() {
                    Ok(services) => {
                        services.add(mdns.clone());
                        background.spawn(mdns_loop(mdns, tx.clone()));
                    }
                    Err(err) => warn!("mDNS registration unavailable: {err}"),
                },
                Err(err) => warn!("mDNS discovery unavailable: {err}"),
            }
        }

        let (rpc_tx, rpc_rx) = mpsc::channel(16);
        let (ctrl_tx, ctrl_rx) = mpsc::channel(64);
        let (event_tx, event_rx) = mpsc::channel(64);
        let service = Service {
            ep,
            store,
            node_name,
            cfg: config,
            tx: tx.clone(),
            rx,
            live: Arc::default(),
            window: None,
            window_seq: 0,
            sessions: HashMap::new(),
            dialing: HashSet::new(),
            discovered: HashMap::new(),
            known_addrs: HashMap::new(),
            reconnect: HashMap::new(),
            background,
            rpc_tx,
            rpc_rx: Some(rpc_rx),
            ctrl_tx,
            ctrl_rx: Some(ctrl_rx),
            event_tx,
            event_rx: Some(event_rx),
        };
        // Kick one dial attempt per stored peer at startup (using the
        // addressing persisted in the peer store); the reconnect ticker
        // keeps retrying with backoff until sessions are up.
        let startup_peers: Vec<EndpointId> = service
            .store
            .load()
            .unwrap_or_default()
            .keys()
            .filter_map(|id| EndpointId::from_str(id).ok())
            .collect();
        let kick = tx.clone();
        tokio::spawn(async move {
            for peer in startup_peers {
                let _ = kick.send(Internal::TryDial { peer }).await;
            }
        });
        tokio::spawn(service.run());
        Ok(MeshHandle { tx, id })
    }
}

impl MeshHandle {
    /// This device's endpoint id (public key).
    pub fn endpoint_id(&self) -> EndpointId {
        self.id
    }

    /// Open a 120-second pairing window: 6-digit code, at most 3 failed
    /// attempts, single success. Returns the code, the endpoint ticket, and
    /// a stream of [`PairEvent`]s.
    pub async fn pair_start(&self) -> Result<PairWindow, MeshError> {
        self.request(|reply| Internal::PairStart { reply }).await?
    }

    /// Join a pairing hosted on another device. A [`PairTarget::Ticket`]
    /// requires `code`; a [`PairTarget::Code`] discovers candidates via
    /// mDNS. One attempt per invocation.
    pub async fn pair_join(
        &self,
        target: PairTarget,
        code: Option<String>,
    ) -> Result<PeerInfo, MeshError> {
        self.request(|reply| Internal::PairJoin {
            target,
            code,
            reply,
        })
        .await?
    }

    /// All paired peers: store contents merged with live link state.
    pub async fn peers(&self) -> Result<Vec<PeerStatus>, MeshError> {
        self.request(|reply| Internal::Peers { reply }).await?
    }

    /// Remove a peer by name; its live connection (if any) is closed
    /// immediately and future accepts are rejected (the store is re-read on
    /// every accept).
    pub async fn unpair(&self, name: &str) -> Result<(), MeshError> {
        let name = name.to_string();
        self.request(|reply| Internal::Unpair { name, reply })
            .await?
    }

    /// Re-run the 4 MiB bandwidth probe against a connected peer. Returns
    /// megabits per second.
    pub async fn probe(&self, name: &str) -> Result<f64, MeshError> {
        let name = name.to_string();
        self.request(|reply| Internal::Probe { name, reply })
            .await?
    }

    /// Take the receiver of accepted `rpc` streams. Single consumer: a
    /// second call fails with [`MeshError::ConsumerTaken`]. The mesh
    /// delivers every `rpc` stream from paired peers; the consumer checks
    /// the epoch and refuses stale streams with close code 4.
    pub async fn incoming_rpc(&self) -> Result<mpsc::Receiver<IncomingRpcStream>, MeshError> {
        self.request(|reply| Internal::TakeRpc { reply })
            .await?
            .ok_or(MeshError::ConsumerTaken { what: "rpc" })
    }

    /// Take the receiver of non-heartbeat control [`Envelope`]s (plan
    /// traffic, node status). Single consumer, like [`Self::incoming_rpc`].
    pub async fn incoming_control(&self) -> Result<mpsc::Receiver<ControlMessage>, MeshError> {
        self.request(|reply| Internal::TakeControl { reply })
            .await?
            .ok_or(MeshError::ConsumerTaken { what: "control" })
    }

    /// Take the receiver of [`PeerEvent`]s: one event per live peer-state
    /// transition (`Connected`, `Suspect`, `Down`, `Incompatible`), plus a
    /// [`PeerState::Draining`] event whenever the peer announces a polite
    /// drain over control (M5, docs/resilience.md — the daemon's failure
    /// lifecycle keys off this stream). Single consumer, like
    /// [`Self::incoming_control`].
    pub async fn peer_events(&self) -> Result<mpsc::Receiver<PeerEvent>, MeshError> {
        self.request(|reply| Internal::TakeEvents { reply })
            .await?
            .ok_or(MeshError::ConsumerTaken {
                what: "peer-events",
            })
    }

    /// Send one [`Envelope`] to a peer (by name or endpoint id) on a fresh
    /// short-lived `Control` stream. Returns once the frame is written and
    /// the stream finished; QUIC delivers it reliably while the session
    /// lives. Errors if the peer is unknown or has no live session.
    pub async fn send_control(&self, peer: &str, envelope: Envelope) -> Result<(), MeshError> {
        let peer = peer.to_string();
        self.request(|reply| Internal::SendControl {
            peer,
            envelope,
            reply,
        })
        .await?
    }

    /// Open a typed bi-stream to a peer (by name or endpoint id): the
    /// `StreamHeader { kind, epoch }` frame is written before the pair is
    /// returned. The head uses `StreamKind::Rpc` streams to tunnel GGML RPC
    /// sessions to workers.
    pub async fn open_stream(
        &self,
        peer: &str,
        kind: StreamKind,
        epoch: Epoch,
    ) -> Result<(SendStream, RecvStream), MeshError> {
        let peer = peer.to_string();
        self.request(|reply| Internal::OpenStream {
            peer,
            kind,
            epoch,
            reply,
        })
        .await?
    }

    /// Stop the service: close every mesh connection and the endpoint.
    /// Idempotent — succeeds if the service is already gone.
    pub async fn shutdown(&self) -> Result<(), MeshError> {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(Internal::Shutdown { reply }).await.is_err() {
            return Ok(());
        }
        let _ = rx.await;
        Ok(())
    }

    async fn request<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> Internal,
    ) -> Result<T, MeshError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(make(reply))
            .await
            .map_err(|_| MeshError::ServiceStopped)?;
        rx.await.map_err(|_| MeshError::ServiceStopped)
    }
}

impl Service {
    async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                Internal::PairStart { reply } => {
                    let _ = reply.send(self.pair_start());
                }
                Internal::PairJoin {
                    target,
                    code,
                    reply,
                } => self.pair_join(target, code, reply),
                Internal::Peers { reply } => {
                    let _ = reply.send(self.peer_statuses());
                }
                Internal::Unpair { name, reply } => {
                    let _ = reply.send(self.unpair(&name));
                }
                Internal::Probe { name, reply } => self.probe(&name, reply),
                Internal::TakeRpc { reply } => {
                    let _ = reply.send(self.rpc_rx.take());
                }
                Internal::TakeControl { reply } => {
                    let _ = reply.send(self.ctrl_rx.take());
                }
                Internal::TakeEvents { reply } => {
                    let _ = reply.send(self.event_rx.take());
                }
                Internal::SendControl {
                    peer,
                    envelope,
                    reply,
                } => match self.resolve_conn(&peer) {
                    Ok(conn) => {
                        tokio::spawn(async move {
                            let _ = reply.send(send_control_stream(&conn, &envelope).await);
                        });
                    }
                    Err(err) => {
                        let _ = reply.send(Err(err));
                    }
                },
                Internal::OpenStream {
                    peer,
                    kind,
                    epoch,
                    reply,
                } => match self.resolve_conn(&peer) {
                    Ok(conn) => {
                        tokio::spawn(async move {
                            let _ = reply.send(open_typed_stream(&conn, kind, epoch).await);
                        });
                    }
                    Err(err) => {
                        let _ = reply.send(Err(err));
                    }
                },
                Internal::Shutdown { reply } => {
                    self.shutdown().await;
                    let _ = reply.send(());
                    return;
                }
                Internal::IncomingPair(conn) => self.incoming_pair(conn),
                Internal::IncomingMesh(conn) => self.adopt_conn(conn, false),
                Internal::PairHostDone {
                    window_id,
                    result,
                    observed,
                } => self.pair_host_done(window_id, result, observed),
                Internal::WindowTimeout { window_id } => {
                    if let Some(window) = &self.window {
                        if window.id == window_id {
                            info!("pairing window expired");
                            let _ = window.events.try_send(PairEvent::Expired);
                            self.window = None;
                        }
                    }
                }
                Internal::JoinedPeer { peer, addr } => {
                    self.known_addrs.insert(peer, addr);
                    // A fresh address was learned: forget any backoff.
                    self.reconnect.remove(&peer);
                    self.ensure_session(peer);
                }
                Internal::DialDone { peer, conn } => {
                    self.dialing.remove(&peer);
                    match conn {
                        Some(conn) => self.adopt_conn(conn, true),
                        None => self.note_dial_failure(peer),
                    }
                }
                Internal::TryDial { peer } => self.ensure_session(peer),
                Internal::SessionEnded { peer, stable_id } => {
                    let matches_current = self
                        .sessions
                        .get(&peer)
                        .is_some_and(|s| s.conn.stable_id() == stable_id);
                    if matches_current {
                        self.sessions.remove(&peer);
                    }
                    // Compatible peers are redialed by the next reconnect
                    // tick (at most ~4 s away); incompatible peers are
                    // skipped by `ensure_session` via their live state.
                }
                Internal::Discovered { peer, addr } => {
                    if peer != self.ep.id() {
                        debug!(peer = %peer.fmt_short(), "mDNS discovered endpoint");
                        self.discovered.insert(peer, addr.clone());
                        if self.store.contains(&peer.to_string()).unwrap_or(false) {
                            self.known_addrs.insert(peer, addr);
                            // A fresh address was learned: forget any
                            // backoff and dial right away.
                            self.reconnect.remove(&peer);
                            self.ensure_session(peer);
                        }
                    }
                }
                Internal::DiscoveryExpired { peer } => {
                    self.discovered.remove(&peer);
                }
                Internal::ReconnectTick => self.reconnect_tick(),
            }
        }
        // All handles and background senders gone; close down quietly.
        self.shutdown().await;
    }

    fn local_hello(&self) -> Hello {
        Hello {
            proto_version: onebrain_proto::PROTO_VERSION,
            capabilities: Capabilities::current(),
            engine_build: EngineBuildHash(self.cfg.engine_build.clone()),
            node_name: self.node_name.clone(),
            product_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn pair_start(&mut self) -> Result<PairWindow, MeshError> {
        if let Some(window) = &self.window {
            if Instant::now() < window.deadline && window.attempts_left > 0 {
                return Err(MeshError::WindowAlreadyOpen);
            }
        }
        let code = generate_code()?;
        let addr = self.ep.addr();
        if addr.is_empty() {
            warn!("endpoint has no addresses yet; the pairing ticket may not be dialable");
        }
        let ticket = EndpointTicket::new(addr).to_string();
        let (events_tx, events_rx) = mpsc::channel(16);
        self.window_seq += 1;
        let window_id = self.window_seq;
        let deadline = Instant::now() + self.cfg.pair_window;
        self.window = Some(Window {
            id: window_id,
            code: code.clone(),
            attempts_left: PAIR_ATTEMPTS,
            deadline,
            events: events_tx,
        });
        let tx = self.tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep_until(deadline).await;
            let _ = tx.send(Internal::WindowTimeout { window_id }).await;
        });
        info!("pairing window open for {:?}", self.cfg.pair_window);
        Ok(PairWindow {
            code,
            ticket,
            events: events_rx,
        })
    }

    fn incoming_pair(&mut self, conn: Connection) {
        let Some(window) = &self.window else {
            warn!(
                peer = %conn.remote_id().fmt_short(),
                "pairing connection outside an open window; closing"
            );
            conn.close(2u32.into(), b"no-pairing-window");
            return;
        };
        let now = Instant::now();
        if now >= window.deadline || window.attempts_left == 0 {
            conn.close(2u32.into(), b"no-pairing-window");
            return;
        }
        let _ = window.events.try_send(PairEvent::Attempt);
        let code = window.code.clone();
        let window_id = window.id;
        let budget = (window.deadline - now).min(PAIR_EXCHANGE_TIMEOUT);
        let local_id = self.ep.id();
        let node_name = self.node_name.clone();
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let result = match timeout(
                budget,
                pairing::host_attempt(&conn, &code, local_id, &node_name),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Err(MeshError::Timeout {
                    what: "pairing attempt",
                    secs: budget.as_secs(),
                }),
            };
            // The host never dials during pairing, so the joiner's observed
            // source address is the only addressing it can learn here.
            // Capture it before the close tears the paths down.
            let observed = conn_addressing(&conn);
            conn.close(0u32.into(), b"pair-done");
            let _ = tx
                .send(Internal::PairHostDone {
                    window_id,
                    result,
                    observed,
                })
                .await;
        });
    }

    fn pair_host_done(
        &mut self,
        window_id: u64,
        result: Result<PairOutcome, MeshError>,
        observed: Addressing,
    ) {
        let Some(window) = &mut self.window else {
            return;
        };
        if window.id != window_id {
            return;
        }
        match result {
            Ok(outcome) => {
                let id_str = outcome.peer_id.to_string();
                match self.store.add(&id_str, &outcome.node_name) {
                    Ok(name) => {
                        let (direct, relay) = observed;
                        if let Err(err) = self.store.update_addrs(&id_str, direct, relay) {
                            warn!(
                                peer = %outcome.peer_id.fmt_short(),
                                "could not persist peer addressing: {err}"
                            );
                        }
                        info!(
                            peer = %outcome.peer_id.fmt_short(),
                            name,
                            version = outcome.product_version,
                            "paired new device"
                        );
                        let _ = window
                            .events
                            .try_send(PairEvent::Paired(PeerInfo { id: id_str, name }));
                        self.window = None;
                        self.reconnect.remove(&outcome.peer_id);
                        self.ensure_session(outcome.peer_id);
                    }
                    Err(err) => {
                        error!("pairing succeeded but the peer store write failed: {err}");
                        let _ = window.events.try_send(PairEvent::Failed(err.to_string()));
                        self.window = None;
                    }
                }
            }
            Err(err) => {
                window.attempts_left -= 1;
                warn!(
                    attempts_left = window.attempts_left,
                    "pairing attempt failed: {err}"
                );
                if window.attempts_left == 0 {
                    let _ = window.events.try_send(PairEvent::Failed(
                        "the pairing code was guessed wrong too often; run `onebrain pair` \
                         again to open a new window"
                            .to_string(),
                    ));
                    self.window = None;
                }
            }
        }
    }

    fn pair_join(
        &mut self,
        target: PairTarget,
        code: Option<String>,
        reply: oneshot::Sender<Result<PeerInfo, MeshError>>,
    ) {
        let ep = self.ep.clone();
        let store = self.store.clone();
        let node_name = self.node_name.clone();
        let tx = self.tx.clone();
        let candidates: Vec<EndpointAddr> = self
            .discovered
            .values()
            .filter(|addr| addr.id != self.ep.id())
            .cloned()
            .collect();
        tokio::spawn(async move {
            let result = do_pair_join(&ep, &store, &node_name, target, code, candidates).await;
            if let Ok((_, peer, addr)) = &result {
                let _ = tx
                    .send(Internal::JoinedPeer {
                        peer: *peer,
                        addr: addr.clone(),
                    })
                    .await;
            }
            let _ = reply.send(result.map(|(info, _, _)| info));
        });
    }

    fn peer_statuses(&self) -> Result<Vec<PeerStatus>, MeshError> {
        let stored = self.store.load()?;
        let live = self.live.lock().expect("live map poisoned");
        let mut out = Vec::with_capacity(stored.len());
        for (id_str, record) in stored {
            let parsed = EndpointId::from_str(&id_str).ok();
            let entry = parsed.and_then(|id| live.get(&id).cloned());
            let state = entry.as_ref().and_then(|l| l.state).unwrap_or({
                if parsed.is_some_and(|id| self.discovered.contains_key(&id)) {
                    PeerState::Reachable
                } else {
                    PeerState::Unknown
                }
            });
            let entry = entry.unwrap_or_default();
            out.push(PeerStatus {
                name: record.name,
                id: id_str,
                state,
                rtt_ms: entry.rtt_ms,
                bandwidth_mbps: entry.bandwidth_mbps,
                loss: entry.loss,
                last_seen_unix: entry.last_seen_unix,
                usable_memory_bytes: entry.usable_memory_bytes,
                prefill_tps: entry.prefill_tps,
                decode_tps: entry.decode_tps,
                disk_mbps: entry.disk_mbps,
                draining: entry.draining,
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    fn unpair(&mut self, name: &str) -> Result<(), MeshError> {
        let id_str = self.store.remove_by_name(name)?;
        if let Ok(peer) = EndpointId::from_str(&id_str) {
            if let Some(session) = self.sessions.remove(&peer) {
                session.runner.abort();
                session.conn.close(1u32.into(), b"unpaired");
            }
            self.live.lock().expect("live map poisoned").remove(&peer);
            self.known_addrs.remove(&peer);
            self.dialing.remove(&peer);
            self.reconnect.remove(&peer);
        }
        info!(name, "unpaired peer");
        Ok(())
    }

    fn probe(&mut self, name: &str, reply: oneshot::Sender<Result<f64, MeshError>>) {
        let found = match self.store.load() {
            Ok(peers) => peers.into_iter().find(|(_, r)| r.name == name),
            Err(err) => {
                let _ = reply.send(Err(err));
                return;
            }
        };
        let Some((id_str, _)) = found else {
            let known = self
                .store
                .load()
                .map(|p| {
                    let mut names: Vec<String> = p.into_values().map(|r| r.name).collect();
                    names.sort();
                    names
                })
                .unwrap_or_default();
            let _ = reply.send(Err(MeshError::UnknownPeerName {
                name: name.to_string(),
                known,
            }));
            return;
        };
        let Some(session) = EndpointId::from_str(&id_str)
            .ok()
            .and_then(|id| self.sessions.get(&id).map(|s| (id, s.conn.clone())))
        else {
            let _ = reply.send(Err(MeshError::NotConnected {
                name: name.to_string(),
            }));
            return;
        };
        let (peer, conn) = session;
        let live = self.live.clone();
        tokio::spawn(async move {
            let result = run_probe(&conn).await;
            if let Ok(mbps) = result {
                set_live(&live, peer, |l| l.bandwidth_mbps = Some(mbps));
            }
            let _ = reply.send(result);
        });
    }

    /// Resolve a peer reference (store name, or endpoint id hex) to its live
    /// mesh connection. Unknown references list the known names; known peers
    /// without a live session report `NotConnected`.
    fn resolve_conn(&self, peer_ref: &str) -> Result<Connection, MeshError> {
        if let Ok(id) = EndpointId::from_str(peer_ref) {
            if let Some(session) = self.sessions.get(&id) {
                return Ok(session.conn.clone());
            }
            let name = self
                .store
                .load()
                .ok()
                .and_then(|peers| peers.get(peer_ref).map(|r| r.name.clone()))
                .unwrap_or_else(|| peer_ref.to_string());
            return Err(MeshError::NotConnected { name });
        }
        let stored = self.store.load()?;
        match stored.iter().find(|(_, r)| r.name == peer_ref) {
            Some((id_str, _)) => {
                let session = EndpointId::from_str(id_str)
                    .ok()
                    .and_then(|id| self.sessions.get(&id));
                match session {
                    Some(session) => Ok(session.conn.clone()),
                    None => Err(MeshError::NotConnected {
                        name: peer_ref.to_string(),
                    }),
                }
            }
            None => {
                let mut known: Vec<String> = stored.into_values().map(|r| r.name).collect();
                known.sort();
                Err(MeshError::UnknownPeerName {
                    name: peer_ref.to_string(),
                    known,
                })
            }
        }
    }

    async fn shutdown(&mut self) {
        for (_, session) in self.sessions.drain() {
            session.runner.abort();
            session.conn.close(0u32.into(), b"shutdown");
        }
        self.background.abort_all();
        self.ep.close().await;
        info!("mesh service stopped");
    }

    /// Lazily establish a mesh session with a paired peer, dialing an
    /// `EndpointAddr` assembled from the store's last-known addressing
    /// merged with any live in-memory hints (pairing exchange, mDNS) — plus
    /// whatever discovery services the endpoint itself runs.
    fn ensure_session(&mut self, peer: EndpointId) {
        if peer == self.ep.id() || self.sessions.contains_key(&peer) || self.dialing.contains(&peer)
        {
            return;
        }
        // Fresh store read: honors concurrent unpair, and picks up
        // addressing persisted by session tasks since the last dial.
        let record = match self.store.load() {
            Ok(mut stored) => match stored.remove(&peer.to_string()) {
                Some(record) => record,
                None => return,
            },
            Err(err) => {
                debug!(peer = %peer.fmt_short(), "peer store unreadable; not dialing: {err}");
                return;
            }
        };
        let incompatible = self
            .live
            .lock()
            .expect("live map poisoned")
            .get(&peer)
            .and_then(|l| l.state)
            == Some(PeerState::Incompatible);
        if incompatible {
            return;
        }
        let mut addr = EndpointAddr::new(peer)
            .with_addrs(record.direct_addrs.iter().map(|sa| TransportAddr::Ip(*sa)));
        if let Some(url) = record
            .relay_url
            .as_deref()
            .and_then(|raw| raw.parse::<RelayUrl>().ok())
        {
            addr = addr.with_relay_url(url);
        }
        if let Some(hint) = self.known_addrs.get(&peer) {
            addr = addr.with_addrs(hint.addrs.iter().cloned());
        }
        if let Some(hint) = self.discovered.get(&peer) {
            addr = addr.with_addrs(hint.addrs.iter().cloned());
        }
        self.dialing.insert(peer);
        let ep = self.ep.clone();
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let conn = match timeout(DIAL_TIMEOUT, ep.connect(addr, ALPN_MESH)).await {
                Ok(Ok(conn)) => Some(conn),
                Ok(Err(err)) => {
                    debug!(peer = %peer.fmt_short(), "mesh dial failed: {err}");
                    None
                }
                Err(_) => {
                    debug!(peer = %peer.fmt_short(), "mesh dial timed out after {DIAL_TIMEOUT:?}");
                    None
                }
            };
            let _ = tx.send(Internal::DialDone { peer, conn }).await;
        });
    }

    /// A dial came back empty: schedule the next attempt with exponential
    /// backoff (3 s doubling to a 30 s cap) so an absent peer stays quiet.
    fn note_dial_failure(&mut self, peer: EndpointId) {
        let now = Instant::now();
        let entry = self.reconnect.entry(peer).or_insert(Backoff {
            next_at: now,
            delay: RECONNECT_BACKOFF_BASE,
        });
        entry.next_at = now + entry.delay;
        entry.delay = (entry.delay * 2).min(RECONNECT_BACKOFF_CAP);
    }

    /// One pass of the reconnect loop: redial every stored peer that has no
    /// live session and whose backoff window has elapsed. This is what
    /// reconnects paired daemons after restarts — peer addresses come from
    /// the store, not from any process memory.
    fn reconnect_tick(&mut self) {
        let stored = match self.store.load() {
            Ok(stored) => stored,
            Err(err) => {
                debug!("reconnect pass skipped, peer store unreadable: {err}");
                return;
            }
        };
        let now = Instant::now();
        for id_str in stored.keys() {
            let Ok(peer) = EndpointId::from_str(id_str) else {
                continue;
            };
            if self.sessions.contains_key(&peer) || self.dialing.contains(&peer) {
                continue;
            }
            if self.reconnect.get(&peer).is_some_and(|b| now < b.next_at) {
                continue;
            }
            self.ensure_session(peer);
        }
    }

    /// Adopt a mesh connection (incoming or dialed), breaking simultaneous-
    /// connect ties deterministically: the connection dialed by the LOWER
    /// endpoint id survives.
    fn adopt_conn(&mut self, conn: Connection, dialer: bool) {
        let peer = conn.remote_id();
        if peer == self.ep.id() {
            conn.close(0u32.into(), b"self-connection");
            return;
        }
        // A connection arrived (either direction): the peer is reachable,
        // so its redial backoff resets.
        self.reconnect.remove(&peer);
        if let Some(existing) = self.sessions.get(&peer) {
            if existing.runner.is_finished() {
                // Stale entry; its SessionEnded event is still queued.
                let old = self.sessions.remove(&peer).expect("checked above");
                old.conn.close(0u32.into(), b"superseded");
            } else {
                let i_am_lower = self.ep.id().as_bytes() < peer.as_bytes();
                let new_wins = dialer == i_am_lower && existing.dialer != i_am_lower;
                if new_wins {
                    debug!(
                        peer = %peer.fmt_short(),
                        "simultaneous connect: replacing session (lower id wins)"
                    );
                    let old = self.sessions.remove(&peer).expect("checked above");
                    old.runner.abort();
                    old.conn.close(0u32.into(), b"superseded");
                } else {
                    debug!(peer = %peer.fmt_short(), "dropping duplicate mesh connection");
                    conn.close(0u32.into(), b"duplicate");
                    return;
                }
            }
        }
        // Resolve the peer's store name once for peer events; sessions only
        // exist for store members, so a miss (unpair race) falls back to the
        // short id form.
        let peer_name = self
            .store
            .load()
            .ok()
            .and_then(|mut stored| stored.remove(&peer.to_string()))
            .map(|record| record.name)
            .unwrap_or_else(|| peer.fmt_short().to_string());
        let ctx = SessionCtx {
            conn: conn.clone(),
            dialer,
            peer,
            peer_name,
            hello: self.local_hello(),
            live: self.live.clone(),
            tx: self.tx.clone(),
            store: self.store.clone(),
            node_status: self.cfg.node_status.clone(),
            rpc_tx: self.rpc_tx.clone(),
            ctrl_tx: self.ctrl_tx.clone(),
            event_tx: self.event_tx.clone(),
        };
        let runner = tokio::spawn(session_runner(ctx));
        self.sessions.insert(
            peer,
            Session {
                conn,
                dialer,
                runner,
            },
        );
    }
}

/// Joiner-side pairing: resolve the target, dial, run the exchange, persist.
async fn do_pair_join(
    ep: &Endpoint,
    store: &PeerStore,
    node_name: &str,
    target: PairTarget,
    code: Option<String>,
    candidates: Vec<EndpointAddr>,
) -> Result<(PeerInfo, EndpointId, EndpointAddr), MeshError> {
    match target {
        PairTarget::Ticket(raw) => {
            let ticket =
                EndpointTicket::from_str(raw.trim()).map_err(|_| MeshError::BadPairTarget {
                    input: truncate_for_display(&raw),
                })?;
            let code = validate_code(&code.ok_or(MeshError::CodeRequired)?)?;
            let addr = ticket.endpoint_addr().clone();
            let outcome = join_one(ep, addr.clone(), &code, node_name).await?;
            persist_join(store, outcome, addr)
        }
        PairTarget::Code(raw) => {
            let code = validate_code(&raw)?;
            if candidates.is_empty() {
                return Err(MeshError::NoCandidates);
            }
            let mut last_err = MeshError::NoCandidates;
            for addr in candidates {
                match join_one(ep, addr.clone(), &code, node_name).await {
                    Ok(outcome) => return persist_join(store, outcome, addr),
                    Err(err) => {
                        debug!(candidate = %addr.id.fmt_short(), "pairing candidate failed: {err}");
                        last_err = err;
                    }
                }
            }
            Err(last_err)
        }
    }
}

fn persist_join(
    store: &PeerStore,
    outcome: PairOutcome,
    addr: EndpointAddr,
) -> Result<(PeerInfo, EndpointId, EndpointAddr), MeshError> {
    let id_str = outcome.peer_id.to_string();
    let name = store.add(&id_str, &outcome.node_name)?;
    // The joiner knows the host's addressing from the ticket (or the mDNS
    // candidate) it just dialed successfully; persist it for redialing
    // after a restart.
    let direct: Vec<SocketAddr> = addr.ip_addrs().copied().collect();
    let relay = addr.relay_urls().next().map(|url| url.to_string());
    if let Err(err) = store.update_addrs(&id_str, direct, relay) {
        warn!(
            peer = %outcome.peer_id.fmt_short(),
            "could not persist peer addressing: {err}"
        );
    }
    info!(
        peer = %outcome.peer_id.fmt_short(),
        name,
        version = outcome.product_version,
        "paired with host"
    );
    Ok((PeerInfo { id: id_str, name }, outcome.peer_id, addr))
}

async fn join_one(
    ep: &Endpoint,
    addr: EndpointAddr,
    code: &str,
    node_name: &str,
) -> Result<PairOutcome, MeshError> {
    let target = addr.id.fmt_short().to_string();
    let conn = ep
        .connect(addr, ALPN_PAIR)
        .await
        .map_err(|err| MeshError::Connect {
            target,
            detail: err.to_string(),
        })?;
    let result = match timeout(
        PAIR_EXCHANGE_TIMEOUT,
        pairing::joiner_attempt(&conn, code, ep.id(), node_name),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => Err(MeshError::Timeout {
            what: "pairing exchange",
            secs: PAIR_EXCHANGE_TIMEOUT.as_secs(),
        }),
    };
    conn.close(0u32.into(), b"pair-done");
    result
}

/// Wait (bounded) for the endpoint to publish at least one address, so
/// tickets minted right after startup are dialable.
async fn wait_for_local_addrs(ep: &Endpoint) {
    let mut watcher = ep.watch_addr();
    let deadline = Instant::now() + ADDR_WAIT;
    loop {
        if !watcher.get().is_empty() {
            return;
        }
        let now = Instant::now();
        if now >= deadline {
            warn!("endpoint published no addresses within {ADDR_WAIT:?}");
            return;
        }
        match timeout(deadline - now, watcher.updated()).await {
            Ok(Ok(_)) => continue,
            Ok(Err(_)) => return,
            Err(_) => {
                warn!("endpoint published no addresses within {ADDR_WAIT:?}");
                return;
            }
        }
    }
}

/// Accept loop: complete handshakes off the service task and dispatch by
/// ALPN. Mesh accepts from ids missing from the peer store (re-read on
/// EVERY accept) are closed with code 1 (`unpaired`) — the §10 guarantee.
async fn accept_loop(ep: Endpoint, store: PeerStore, tx: mpsc::Sender<Internal>) {
    let mut handshakes = JoinSet::new();
    while let Some(incoming) = ep.accept().await {
        let store = store.clone();
        let tx = tx.clone();
        handshakes.spawn(async move {
            let accepting = match incoming.accept() {
                Ok(accepting) => accepting,
                Err(err) => {
                    debug!("incoming connection failed before handshake: {err}");
                    return;
                }
            };
            let conn = match accepting.await {
                Ok(conn) => conn,
                Err(err) => {
                    debug!("incoming connection failed during handshake: {err}");
                    return;
                }
            };
            let alpn = conn.alpn().to_vec();
            if alpn == ALPN_PAIR {
                let _ = tx.send(Internal::IncomingPair(conn)).await;
            } else if alpn == ALPN_MESH {
                let remote = conn.remote_id();
                match store.contains(&remote.to_string()) {
                    Ok(true) => {
                        let _ = tx.send(Internal::IncomingMesh(conn)).await;
                    }
                    Ok(false) => {
                        warn!(
                            peer = %remote.fmt_short(),
                            "rejecting mesh connection from unpaired endpoint"
                        );
                        conn.close(1u32.into(), b"unpaired");
                    }
                    Err(err) => {
                        warn!(
                            peer = %remote.fmt_short(),
                            "peer store unreadable ({err}); rejecting mesh connection"
                        );
                        conn.close(1u32.into(), b"unpaired");
                    }
                }
            } else {
                conn.close(0u32.into(), b"unknown-alpn");
            }
        });
        while handshakes.try_join_next().is_some() {}
    }
}

/// Drive the reconnect loop: one [`Internal::ReconnectTick`] every 2–4 s
/// (3 s ± 1 s uniform jitter). The service task decides per peer whether a
/// dial actually happens (live session, in-flight dial, backoff).
async fn reconnect_ticker(tx: mpsc::Sender<Internal>) {
    loop {
        let jitter_ms = getrandom::u32()
            .map(|v| u64::from(v) % RECONNECT_TICK_JITTER_MS)
            .unwrap_or(RECONNECT_TICK_JITTER_MS / 2);
        tokio::time::sleep(Duration::from_millis(RECONNECT_TICK_BASE_MS + jitter_ms)).await;
        if tx.send(Internal::ReconnectTick).await.is_err() {
            return;
        }
    }
}

/// Forward mDNS discovery events into the service task, maintaining the
/// live candidate set for code-only pairing.
async fn mdns_loop(mdns: MdnsAddressLookup, tx: mpsc::Sender<Internal>) {
    let mut events = mdns.subscribe().await;
    while let Some(event) = events.next().await {
        let forward = match event {
            DiscoveryEvent::Discovered { endpoint_info, .. } => Internal::Discovered {
                peer: endpoint_info.endpoint_id,
                addr: endpoint_info.into_endpoint_addr(),
            },
            DiscoveryEvent::Expired { endpoint_id } => {
                Internal::DiscoveryExpired { peer: endpoint_id }
            }
            _ => continue,
        };
        if tx.send(forward).await.is_err() {
            return;
        }
    }
}

struct SessionCtx {
    conn: Connection,
    dialer: bool,
    peer: EndpointId,
    /// The peer's store name, resolved at session adoption (used in
    /// [`PeerEvent`]s).
    peer_name: String,
    hello: Hello,
    live: LiveMap,
    tx: mpsc::Sender<Internal>,
    store: PeerStore,
    node_status: Option<NodeStatusFn>,
    rpc_tx: mpsc::Sender<IncomingRpcStream>,
    ctrl_tx: mpsc::Sender<ControlMessage>,
    event_tx: mpsc::Sender<PeerEvent>,
}

/// The remote transport addresses of a connection's live QUIC paths, split
/// into direct socket addresses and (at most one) relay URL. Available on
/// BOTH the dial and accept side: the accept side observes the peer's
/// source address, which is redialable as long as the peer pins its bind
/// port (or sits behind a stable NAT binding).
fn conn_addressing(conn: &Connection) -> Addressing {
    let mut direct = Vec::new();
    let mut relay = None;
    for path in conn.paths().iter() {
        match path.remote_addr() {
            TransportAddr::Ip(addr) => direct.push(*addr),
            TransportAddr::Relay(url) => relay = Some(url.to_string()),
            // `TransportAddr` is non-exhaustive (custom transports); those
            // are not persistable.
            _ => {}
        }
    }
    direct.sort();
    direct.dedup();
    (direct, relay)
}

enum SessionEnd {
    Incompatible,
    Established,
    NeverEstablished,
}

async fn session_runner(ctx: SessionCtx) {
    let stable_id = ctx.conn.stable_id();
    let peer = ctx.peer;
    let end = run_session(&ctx).await;
    match end {
        SessionEnd::Incompatible => note_transition(
            &ctx.live,
            &ctx.event_tx,
            peer,
            &ctx.peer_name,
            PeerState::Incompatible,
        ),
        SessionEnd::Established => note_transition(
            &ctx.live,
            &ctx.event_tx,
            peer,
            &ctx.peer_name,
            PeerState::Down,
        ),
        SessionEnd::NeverEstablished => {}
    }
    let _ = ctx
        .tx
        .send(Internal::SessionEnded { peer, stable_id })
        .await;
}

async fn run_session(ctx: &SessionCtx) -> SessionEnd {
    let theirs = match timeout(
        HELLO_TIMEOUT,
        exchange_hello(&ctx.conn, ctx.dialer, &ctx.hello),
    )
    .await
    {
        Ok(Ok(hello)) => hello,
        Ok(Err(err)) => {
            debug!(peer = %ctx.peer.fmt_short(), "hello exchange failed: {err}");
            return SessionEnd::NeverEstablished;
        }
        Err(_) => {
            debug!(peer = %ctx.peer.fmt_short(), "hello exchange timed out");
            ctx.conn.close(0u32.into(), b"hello-timeout");
            return SessionEnd::NeverEstablished;
        }
    };
    match judge(&ctx.hello, &theirs) {
        HandshakeVerdict::Compatible => {}
        HandshakeVerdict::ProtoMismatch { ours, theirs: t } => {
            error!(
                peer = %ctx.peer.fmt_short(),
                "protocol mismatch (ours v{ours}, theirs v{t}); update the node running the \
                 older version with `onebrain doctor --self-update`"
            );
            ctx.conn.close(3u32.into(), b"incompatible");
            return SessionEnd::Incompatible;
        }
        HandshakeVerdict::EngineMismatch { ours, theirs: t } => {
            error!(
                peer = %ctx.peer.fmt_short(),
                "engine build mismatch (ours {ours}, theirs {t}); run `onebrain doctor \
                 --self-update` on {} so both nodes run the same build",
                theirs.node_name
            );
            ctx.conn.close(3u32.into(), b"incompatible");
            return SessionEnd::Incompatible;
        }
    }
    if let Some(rtt) = ctx.conn.rtt(PathId::ZERO) {
        debug!(
            peer = %ctx.peer.fmt_short(),
            quic_rtt_ms = rtt.as_secs_f64() * 1000.0,
            "mesh session established"
        );
    } else {
        debug!(peer = %ctx.peer.fmt_short(), "mesh session established");
    }
    // Persist the peer's observed addressing (remote side of the live QUIC
    // paths) so this daemon can redial it after either side restarts.
    // Deliberately BEFORE the live state flips to Connected: anyone who
    // observes `Connected` knows the store is already fresh.
    let (direct, relay) = conn_addressing(&ctx.conn);
    if direct.is_empty() && relay.is_none() {
        debug!(
            peer = %ctx.peer.fmt_short(),
            "session exposes no persistable remote addresses"
        );
    } else {
        match ctx.store.update_addrs(&ctx.peer.to_string(), direct, relay) {
            Ok(true) => debug!(peer = %ctx.peer.fmt_short(), "stored peer addressing updated"),
            Ok(false) => {}
            Err(err) => warn!(
                peer = %ctx.peer.fmt_short(),
                "could not persist peer addressing: {err}"
            ),
        }
    }
    set_live(&ctx.live, ctx.peer, |l| {
        l.last_seen_unix = Some(unix_now());
    });
    note_transition(
        &ctx.live,
        &ctx.event_tx,
        ctx.peer,
        &ctx.peer_name,
        PeerState::Connected,
    );

    let mut workers = JoinSet::new();
    workers.spawn(stream_acceptor(
        ctx.conn.clone(),
        StreamPeerCtx {
            peer: ctx.peer,
            name: ctx.peer_name.clone(),
            live: ctx.live.clone(),
            rpc_tx: ctx.rpc_tx.clone(),
            ctrl_tx: ctx.ctrl_tx.clone(),
            event_tx: ctx.event_tx.clone(),
        },
    ));
    workers.spawn(uni_drainer(ctx.conn.clone()));
    {
        // One bandwidth probe on connect (repeatable via `probe()`).
        let conn = ctx.conn.clone();
        let live = ctx.live.clone();
        let peer = ctx.peer;
        workers.spawn(async move {
            match run_probe(&conn).await {
                Ok(mbps) => set_live(&live, peer, |l| l.bandwidth_mbps = Some(mbps)),
                Err(err) => debug!(peer = %peer.fmt_short(), "bandwidth probe failed: {err}"),
            }
        });
    }
    if let Some(provider) = ctx.node_status.clone() {
        // Report this node's schedulable memory right after the handshake so
        // the peer can budget it into placement plans (docs/distributed.md).
        let conn = ctx.conn.clone();
        let peer = ctx.peer;
        workers.spawn(async move {
            let status = provider();
            let usable_memory_bytes = status.usable_memory_bytes;
            let envelope = Envelope::new(Message::NodeStatus {
                usable_memory_bytes: status.usable_memory_bytes,
                devices: status.devices,
                prefill_tps: status.prefill_tps,
                decode_tps: status.decode_tps,
                disk_mbps: status.disk_mbps,
                draining: status.draining,
            });
            match send_control_stream(&conn, &envelope).await {
                Ok(()) => debug!(
                    peer = %peer.fmt_short(),
                    usable_memory_bytes, "sent NodeStatus"
                ),
                Err(err) => debug!(peer = %peer.fmt_short(), "NodeStatus send failed: {err}"),
            }
        });
    }
    tokio::select! {
        _ = run_heartbeat(&ctx.conn, &ctx.live, ctx.peer, &ctx.peer_name, &ctx.event_tx) => {
            ctx.conn.close(0u32.into(), b"heartbeat-lost");
        }
        err = ctx.conn.closed() => {
            debug!(peer = %ctx.peer.fmt_short(), "mesh connection closed: {err}");
        }
    }
    workers.abort_all();
    SessionEnd::Established
}

/// The header every locally opened control stream starts with. Control
/// streams are not epoch-scoped; receivers ignore the epoch (contract in
/// `onebrain_proto::message`).
fn control_header() -> StreamHeader {
    StreamHeader {
        kind: StreamKind::Control,
        epoch: Epoch(0),
    }
}

/// Write one envelope on a fresh short-lived `Control` stream: header,
/// envelope, FIN. QUIC delivers the buffered frames reliably for as long as
/// the connection lives.
async fn send_control_stream(conn: &Connection, envelope: &Envelope) -> Result<(), MeshError> {
    let (mut tx, _rx) = conn.open_bi().await.map_err(stream_err)?;
    write_frame(&mut tx, &control_header()).await?;
    write_frame(&mut tx, envelope).await?;
    tx.finish().map_err(stream_err)?;
    Ok(())
}

/// Open a typed bi-stream and write its `StreamHeader` before handing the
/// halves to the caller.
async fn open_typed_stream(
    conn: &Connection,
    kind: StreamKind,
    epoch: Epoch,
) -> Result<(SendStream, RecvStream), MeshError> {
    let (mut tx, rx) = conn.open_bi().await.map_err(stream_err)?;
    write_frame(&mut tx, &StreamHeader { kind, epoch }).await?;
    Ok((tx, rx))
}

/// Exchange `Hello`s on the connection's first bi stream (opened by the
/// dialer). The stream starts with a `Control` header like every mesh
/// bi-stream. Returns the peer's `Hello`.
async fn exchange_hello(conn: &Connection, dialer: bool, ours: &Hello) -> Result<Hello, MeshError> {
    let envelope = Envelope::new(Message::Hello(ours.clone()));
    if dialer {
        let (mut tx, mut rx) = conn.open_bi().await.map_err(stream_err)?;
        write_frame(&mut tx, &control_header()).await?;
        write_frame(&mut tx, &envelope).await?;
        let theirs = expect_hello(read_frame::<Envelope>(&mut rx).await?)?;
        let _ = tx.finish();
        Ok(theirs)
    } else {
        let (mut tx, mut rx) = conn.accept_bi().await.map_err(stream_err)?;
        expect_control_header(read_frame::<StreamHeader>(&mut rx).await?)?;
        let theirs = expect_hello(read_frame::<Envelope>(&mut rx).await?)?;
        write_frame(&mut tx, &envelope).await?;
        let _ = tx.finish();
        Ok(theirs)
    }
}

fn expect_control_header(header: StreamHeader) -> Result<(), MeshError> {
    if header.kind == StreamKind::Control {
        Ok(())
    } else {
        Err(MeshError::Stream {
            detail: format!(
                "expected a Control stream header on the hello stream, got {:?}",
                header.kind
            ),
        })
    }
}

fn expect_hello(envelope: Envelope) -> Result<Hello, MeshError> {
    match envelope.message {
        Message::Hello(hello) => Ok(hello),
        other => Err(MeshError::Stream {
            detail: format!("expected Hello as the first mesh message, got {other:?}"),
        }),
    }
}

/// Send `Envelope(Heartbeat)` every 2 s on a dedicated bi stream and measure
/// echo round-trips. Returns when the peer is `down` (10 s silence) or the
/// stream fails. 3 missed heartbeats mark the peer `suspect` (spec §5).
/// Suspect/Connected flips go through [`note_transition`], so the
/// `peer_events()` consumer sees each one exactly once.
async fn run_heartbeat(
    conn: &Connection,
    live: &LiveMap,
    peer: EndpointId,
    peer_name: &str,
    events: &mpsc::Sender<PeerEvent>,
) {
    let (mut tx, mut rx) = match conn.open_bi().await {
        Ok(pair) => pair,
        Err(err) => {
            debug!(peer = %peer.fmt_short(), "heartbeat stream failed to open: {err}");
            return;
        }
    };
    // Type the stream (M3): heartbeat wire content is unchanged after the
    // header — the peer's control handler echoes heartbeats exactly as
    // before.
    if write_frame(&mut tx, &control_header()).await.is_err() {
        return;
    }
    let mut interval = tokio::time::interval(HEARTBEAT_INTERVAL);
    let mut pending: VecDeque<Instant> = VecDeque::new();
    let mut history: VecDeque<bool> = VecDeque::with_capacity(LOSS_WINDOW);
    let mut missed = 0u32;
    let mut last_echo = Instant::now();
    loop {
        tokio::select! {
            _ = interval.tick() => {
                if !pending.is_empty() {
                    missed += 1;
                    push_history(&mut history, false);
                    let loss = loss_fraction(&history);
                    set_live(live, peer, |l| l.loss = Some(loss));
                }
                if last_echo.elapsed() >= DOWN_AFTER {
                    warn!(peer = %peer.fmt_short(), "no heartbeat echo for {DOWN_AFTER:?}; peer is down");
                    return;
                }
                if missed >= SUSPECT_AFTER_MISSED {
                    note_transition(live, events, peer, peer_name, PeerState::Suspect);
                }
                let beat = Envelope::new(Message::Heartbeat { epoch: Epoch(0) });
                if write_frame(&mut tx, &beat).await.is_err() {
                    return;
                }
                pending.push_back(Instant::now());
            }
            echoed = read_frame::<Envelope>(&mut rx) => {
                if echoed.is_err() {
                    return;
                }
                let now = Instant::now();
                last_echo = now;
                missed = 0;
                if let Some(sent) = pending.pop_front() {
                    let sample_ms = (now - sent).as_secs_f64() * 1000.0;
                    push_history(&mut history, true);
                    let loss = loss_fraction(&history);
                    set_live(live, peer, |l| {
                        l.rtt_ms = Some(match l.rtt_ms {
                            Some(prev) => RTT_EWMA_ALPHA * sample_ms + (1.0 - RTT_EWMA_ALPHA) * prev,
                            None => sample_ms,
                        });
                        l.loss = Some(loss);
                        l.last_seen_unix = Some(unix_now());
                    });
                    // Only an actual Suspect -> Connected recovery emits an
                    // event; steady echoes are transition-free.
                    note_transition(live, events, peer, peer_name, PeerState::Connected);
                }
            }
        }
    }
}

fn push_history(history: &mut VecDeque<bool>, answered: bool) {
    if history.len() == LOSS_WINDOW {
        history.pop_front();
    }
    history.push_back(answered);
}

fn loss_fraction(history: &VecDeque<bool>) -> f32 {
    if history.is_empty() {
        return 0.0;
    }
    let missed = history.iter().filter(|answered| !**answered).count();
    missed as f32 / history.len() as f32
}

/// Everything a per-stream handler needs to know about the peer whose
/// connection the stream arrived on: identity, store name, and the shared
/// channels/state its traffic feeds.
#[derive(Clone)]
struct StreamPeerCtx {
    peer: EndpointId,
    /// The peer's store name (used in [`PeerEvent`]s).
    name: String,
    live: LiveMap,
    rpc_tx: mpsc::Sender<IncomingRpcStream>,
    ctrl_tx: mpsc::Sender<ControlMessage>,
    event_tx: mpsc::Sender<PeerEvent>,
}

/// Accept the peer's bi-streams and dispatch them by their `StreamHeader`
/// kind: `Control` streams run the envelope loop (heartbeat echo + control
/// message delivery), `Rpc` streams are handed to the daemon's consumer, and
/// `Probe` is reserved (drained).
async fn stream_acceptor(conn: Connection, ctx: StreamPeerCtx) {
    let mut handlers = JoinSet::new();
    loop {
        match conn.accept_bi().await {
            Ok((tx, rx)) => {
                handlers.spawn(handle_stream(ctx.clone(), tx, rx));
            }
            Err(_) => break,
        }
        while handlers.try_join_next().is_some() {}
    }
    handlers.abort_all();
}

async fn handle_stream(ctx: StreamPeerCtx, tx: SendStream, mut rx: RecvStream) {
    let peer = ctx.peer;
    let header = match read_frame::<StreamHeader>(&mut rx).await {
        Ok(header) => header,
        Err(err) => {
            debug!(peer = %peer.fmt_short(), "mesh stream without a valid header: {err}");
            return;
        }
    };
    match header.kind {
        StreamKind::Control => control_stream(ctx, tx, rx).await,
        StreamKind::Rpc => {
            let incoming = IncomingRpcStream {
                peer: NodeId(peer.to_string()),
                epoch: header.epoch,
                send: tx,
                recv: rx,
            };
            if let Err(returned) = ctx.rpc_tx.send(incoming).await {
                // No daemon consumer (or it is gone): refuse rather than
                // leaving the head's bridge hanging. Code 4 — the receiving
                // side cannot have an active epoch without a consumer.
                warn!(
                    peer = %peer.fmt_short(),
                    epoch = header.epoch.0,
                    "no rpc consumer registered; refusing rpc stream"
                );
                returned.0.refuse(4);
            }
        }
        StreamKind::Probe => {
            // Reserved for the M4 link prober; drain so the peer's writes
            // complete.
            let mut rx = rx;
            let mut buf = vec![0u8; 64 * 1024];
            while let Ok(Some(_)) = rx.read(&mut buf).await {}
        }
    }
}

/// The `Control` envelope loop: heartbeats are echoed back unchanged (the
/// pre-M3 wire behavior), `NodeStatus` is cached into the live map (budget,
/// profile, and — M5 — the `draining` flag), a `Draining` envelope emits a
/// [`PeerState::Draining`] peer event, and every non-heartbeat envelope is
/// forwarded to the daemon's control consumer (`Draining` included: the
/// daemon needs its epoch, docs/resilience.md).
async fn control_stream(ctx: StreamPeerCtx, mut tx: SendStream, mut rx: RecvStream) {
    let peer = ctx.peer;
    loop {
        match read_frame::<Envelope>(&mut rx).await {
            Ok(envelope) => {
                if matches!(envelope.message, Message::Heartbeat { .. }) {
                    if write_frame(&mut tx, &envelope).await.is_err() {
                        return;
                    }
                    continue;
                }
                if let Message::NodeStatus {
                    usable_memory_bytes,
                    prefill_tps,
                    decode_tps,
                    disk_mbps,
                    draining,
                    ..
                } = &envelope.message
                {
                    let usable = *usable_memory_bytes;
                    let (prefill, decode, disk) = (*prefill_tps, *decode_tps, *disk_mbps);
                    let draining = *draining;
                    set_live(&ctx.live, peer, |l| {
                        l.usable_memory_bytes = Some(usable);
                        // The profile fields are overwritten whole: a peer
                        // that has not (or no longer) benched reports None.
                        l.prefill_tps = prefill;
                        l.decode_tps = decode;
                        l.disk_mbps = disk;
                        // Same rule for the drain flag: every NodeStatus is
                        // the whole truth (a recharged node reports false).
                        l.draining = draining;
                    });
                }
                if let Message::Draining { epoch, reason } = &envelope.message {
                    // Polite drain notice: surface it to the peer-events
                    // consumer. The live state is NOT changed — the session
                    // stays Connected until the peer actually goes away —
                    // and the envelope still flows to the control consumer
                    // below (the daemon fences on its epoch).
                    debug!(
                        peer = %peer.fmt_short(),
                        epoch = epoch.0,
                        reason,
                        "peer announced a polite drain"
                    );
                    emit_peer_event(&ctx.event_tx, peer, &ctx.name, PeerState::Draining);
                }
                let message = ControlMessage {
                    peer: NodeId(peer.to_string()),
                    envelope,
                };
                if ctx.ctrl_tx.send(message).await.is_err() {
                    debug!(
                        peer = %peer.fmt_short(),
                        "control message dropped: no consumer registered"
                    );
                }
            }
            Err(_) => return, // EOF (short-lived control stream) or error.
        }
    }
}

/// Drain the peer's uni streams (bandwidth probes) so its `stopped()` timer
/// completes.
async fn uni_drainer(conn: Connection) {
    let mut drains = JoinSet::new();
    loop {
        match conn.accept_uni().await {
            Ok(mut rx) => {
                drains.spawn(async move {
                    let mut buf = vec![0u8; 64 * 1024];
                    loop {
                        match rx.read(&mut buf).await {
                            Ok(Some(_)) => {}
                            Ok(None) | Err(_) => return,
                        }
                    }
                });
            }
            Err(_) => break,
        }
        while drains.try_join_next().is_some() {}
    }
    drains.abort_all();
}

/// Time a 4 MiB zero-filled uni stream to the peer; returns megabits/sec.
async fn run_probe(conn: &Connection) -> Result<f64, MeshError> {
    let mut tx = conn.open_uni().await.map_err(stream_err)?;
    let start = Instant::now();
    let chunk = vec![0u8; 64 * 1024];
    let mut sent: u64 = 0;
    while sent < PROBE_BYTES {
        tx.write_all(&chunk).await.map_err(stream_err)?;
        sent += chunk.len() as u64;
    }
    let _ = tx.finish();
    match timeout(PROBE_TIMEOUT, tx.stopped()).await {
        Ok(Ok(_)) => {}
        Ok(Err(err)) => return Err(stream_err(err)),
        Err(_) => {
            return Err(MeshError::Timeout {
                what: "bandwidth probe",
                secs: PROBE_TIMEOUT.as_secs(),
            })
        }
    }
    let secs = start.elapsed().as_secs_f64().max(1e-9);
    Ok((PROBE_BYTES as f64 * 8.0) / secs / 1e6)
}
