//! The mesh service task: iroh endpoint ownership, ALPN dispatch, pairing
//! windows, peer sessions (Hello, heartbeats, probes), and mDNS discovery.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use iroh::endpoint::{presets, Connection, PathId, RecvStream, SendStream};
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey, Watcher};
use iroh_mdns_address_lookup::{DiscoveryEvent, MdnsAddressLookup};
use iroh_tickets::endpoint::EndpointTicket;
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{timeout, Instant};
use tracing::{debug, error, info, warn};

use onebrain_proto::capabilities::Capabilities;
use onebrain_proto::handshake::{judge, EngineBuildHash, HandshakeVerdict, Hello};
use onebrain_proto::message::{Envelope, Message};
use onebrain_proto::plan::Epoch;

use crate::pairing::{
    self, generate_code, read_frame, stream_err, truncate_for_display, validate_code, write_frame,
    PairOutcome,
};
use crate::store::PeerStore;
use crate::{MeshConfig, MeshError, PairTarget, PeerState, PeerStatus};

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
const REDIAL_DELAY: Duration = Duration::from_secs(5);
const ADDR_WAIT: Duration = Duration::from_secs(10);

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
    Shutdown {
        reply: oneshot::Sender<()>,
    },
    // Internal events.
    IncomingPair(Connection),
    IncomingMesh(Connection),
    PairHostDone {
        window_id: u64,
        result: Result<PairOutcome, MeshError>,
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
        incompatible: bool,
    },
    Discovered {
        peer: EndpointId,
        addr: EndpointAddr,
    },
    DiscoveryExpired {
        peer: EndpointId,
    },
}

/// Live per-peer link state, shared between session tasks and the service.
#[derive(Debug, Default, Clone)]
struct Live {
    state: Option<PeerState>,
    rtt_ms: Option<f64>,
    bandwidth_mbps: Option<f64>,
    loss: Option<f32>,
    last_seen_unix: Option<u64>,
}

type LiveMap = Arc<StdMutex<HashMap<EndpointId, Live>>>;

fn set_live(live: &LiveMap, peer: EndpointId, update: impl FnOnce(&mut Live)) {
    let mut map = live.lock().expect("live map poisoned");
    update(map.entry(peer).or_default());
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
    /// Background accept + mDNS loops; aborted on drop.
    background: JoinSet<()>,
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
            background,
        };
        // Lazily connect to already-paired peers as they become reachable;
        // kick one dial attempt per stored peer at startup.
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
                Internal::Shutdown { reply } => {
                    self.shutdown().await;
                    let _ = reply.send(());
                    return;
                }
                Internal::IncomingPair(conn) => self.incoming_pair(conn),
                Internal::IncomingMesh(conn) => self.adopt_conn(conn, false),
                Internal::PairHostDone { window_id, result } => {
                    self.pair_host_done(window_id, result)
                }
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
                    self.ensure_session(peer);
                }
                Internal::DialDone { peer, conn } => {
                    self.dialing.remove(&peer);
                    match conn {
                        Some(conn) => self.adopt_conn(conn, true),
                        None => self.schedule_redial(peer),
                    }
                }
                Internal::TryDial { peer } => self.ensure_session(peer),
                Internal::SessionEnded {
                    peer,
                    stable_id,
                    incompatible,
                } => {
                    let matches_current = self
                        .sessions
                        .get(&peer)
                        .is_some_and(|s| s.conn.stable_id() == stable_id);
                    if matches_current {
                        self.sessions.remove(&peer);
                    }
                    if !incompatible {
                        self.schedule_redial(peer);
                    }
                }
                Internal::Discovered { peer, addr } => {
                    if peer != self.ep.id() {
                        debug!(peer = %peer.fmt_short(), "mDNS discovered endpoint");
                        self.discovered.insert(peer, addr.clone());
                        if self.store.contains(&peer.to_string()).unwrap_or(false) {
                            self.known_addrs.insert(peer, addr);
                            self.ensure_session(peer);
                        }
                    }
                }
                Internal::DiscoveryExpired { peer } => {
                    self.discovered.remove(&peer);
                }
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
            conn.close(0u32.into(), b"pair-done");
            let _ = tx.send(Internal::PairHostDone { window_id, result }).await;
        });
    }

    fn pair_host_done(&mut self, window_id: u64, result: Result<PairOutcome, MeshError>) {
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

    async fn shutdown(&mut self) {
        for (_, session) in self.sessions.drain() {
            session.runner.abort();
            session.conn.close(0u32.into(), b"shutdown");
        }
        self.background.abort_all();
        self.ep.close().await;
        info!("mesh service stopped");
    }

    /// Lazily establish a mesh session with a paired peer.
    fn ensure_session(&mut self, peer: EndpointId) {
        if peer == self.ep.id() || self.sessions.contains_key(&peer) || self.dialing.contains(&peer)
        {
            return;
        }
        if !self.store.contains(&peer.to_string()).unwrap_or(false) {
            return;
        }
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
        let addr = self
            .known_addrs
            .get(&peer)
            .or_else(|| self.discovered.get(&peer))
            .cloned()
            .unwrap_or_else(|| EndpointAddr::new(peer));
        self.dialing.insert(peer);
        let ep = self.ep.clone();
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let conn = match ep.connect(addr, ALPN_MESH).await {
                Ok(conn) => Some(conn),
                Err(err) => {
                    debug!(peer = %peer.fmt_short(), "mesh dial failed: {err}");
                    None
                }
            };
            let _ = tx.send(Internal::DialDone { peer, conn }).await;
        });
    }

    fn schedule_redial(&self, peer: EndpointId) {
        let tx = self.tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(REDIAL_DELAY).await;
            let _ = tx.send(Internal::TryDial { peer }).await;
        });
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
        let ctx = SessionCtx {
            conn: conn.clone(),
            dialer,
            peer,
            hello: self.local_hello(),
            live: self.live.clone(),
            tx: self.tx.clone(),
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
    hello: Hello,
    live: LiveMap,
    tx: mpsc::Sender<Internal>,
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
    let incompatible = matches!(end, SessionEnd::Incompatible);
    set_live(&ctx.live, peer, |l| match end {
        SessionEnd::Incompatible => l.state = Some(PeerState::Incompatible),
        SessionEnd::Established => l.state = Some(PeerState::Down),
        SessionEnd::NeverEstablished => {}
    });
    let _ = ctx
        .tx
        .send(Internal::SessionEnded {
            peer,
            stable_id,
            incompatible,
        })
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
    set_live(&ctx.live, ctx.peer, |l| {
        l.state = Some(PeerState::Connected);
        l.last_seen_unix = Some(unix_now());
    });

    let mut workers = JoinSet::new();
    workers.spawn(echo_acceptor(ctx.conn.clone()));
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
    tokio::select! {
        _ = run_heartbeat(&ctx.conn, &ctx.live, ctx.peer) => {
            ctx.conn.close(0u32.into(), b"heartbeat-lost");
        }
        err = ctx.conn.closed() => {
            debug!(peer = %ctx.peer.fmt_short(), "mesh connection closed: {err}");
        }
    }
    workers.abort_all();
    SessionEnd::Established
}

/// Exchange `Hello`s on the connection's first bi stream (opened by the
/// dialer). Returns the peer's `Hello`.
async fn exchange_hello(conn: &Connection, dialer: bool, ours: &Hello) -> Result<Hello, MeshError> {
    let envelope = Envelope::new(Message::Hello(ours.clone()));
    if dialer {
        let (mut tx, mut rx) = conn.open_bi().await.map_err(stream_err)?;
        write_frame(&mut tx, &envelope).await?;
        let theirs = expect_hello(read_frame::<Envelope>(&mut rx).await?)?;
        let _ = tx.finish();
        Ok(theirs)
    } else {
        let (mut tx, mut rx) = conn.accept_bi().await.map_err(stream_err)?;
        let theirs = expect_hello(read_frame::<Envelope>(&mut rx).await?)?;
        write_frame(&mut tx, &envelope).await?;
        let _ = tx.finish();
        Ok(theirs)
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
async fn run_heartbeat(conn: &Connection, live: &LiveMap, peer: EndpointId) {
    let (mut tx, mut rx) = match conn.open_bi().await {
        Ok(pair) => pair,
        Err(err) => {
            debug!(peer = %peer.fmt_short(), "heartbeat stream failed to open: {err}");
            return;
        }
    };
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
                    set_live(live, peer, |l| l.state = Some(PeerState::Suspect));
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
                        l.state = Some(PeerState::Connected);
                        l.last_seen_unix = Some(unix_now());
                    });
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

/// Accept the peer's heartbeat streams and echo every frame back.
async fn echo_acceptor(conn: Connection) {
    let mut echoes = JoinSet::new();
    loop {
        match conn.accept_bi().await {
            Ok((tx, rx)) => {
                echoes.spawn(echo_stream(tx, rx));
            }
            Err(_) => break,
        }
        while echoes.try_join_next().is_some() {}
    }
    echoes.abort_all();
}

async fn echo_stream(mut tx: SendStream, mut rx: RecvStream) {
    loop {
        match read_frame::<Envelope>(&mut rx).await {
            Ok(envelope) => match envelope.message {
                Message::Heartbeat { .. } => {
                    if write_frame(&mut tx, &envelope).await.is_err() {
                        return;
                    }
                }
                other => {
                    debug!("unexpected message on echo stream: {other:?}");
                    return;
                }
            },
            Err(_) => return,
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
