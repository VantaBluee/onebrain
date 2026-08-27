//! Cluster-session state and the mesh-driven cluster task (M3,
//! docs/distributed.md "Epoch lifecycle" / "Daemon & API").
//!
//! One cluster task per daemon consumes the mesh's incoming control messages
//! and `rpc` streams:
//!
//! - **Worker path**: a `PlanProposal` from a paired peer is adopted when its
//!   epoch is newer than the active one (stale epochs are fenced), the
//!   engine host is told to [`crate::engine_host::HostMsg::ServeShard`]
//!   (unloading any local model — M3 contract: plans preempt local models),
//!   and a `PlanAck` is returned. Accepted `rpc` streams for the active
//!   epoch (and only from that epoch's head) are bridged into an in-process
//!   GGML RPC serve session via a platform socket pair; anything else is
//!   refused with close code 4 (`bad-epoch`).
//! - **Head path**: `PlanAck`s are recorded for
//!   [`ClusterState::await_acks`]; the load flow in `server.rs` drives the
//!   proposals, the accept-loop loopback bridges ([`head_bridge`] — one
//!   fresh mesh stream per accepted RPC-client connection), and the
//!   distributed engine load.
//!
//! Teardown ordering (ADR 0004): the engine frees a distributed model while
//! its rpc bridges still stand (the free sends remote FREE_BUFFERs; GGML
//! aborts on a torn stream), and the bridges close afterwards. On the worker
//! the serve threads end when their streams close, and are joined here.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use onebrain_engine::rpc::{BridgeStream, RpcSession};
use onebrain_engine::DeviceKind;
use onebrain_mesh::{ControlMessage, IncomingRpcStream, MeshHandle, RecvStream, SendStream};
use onebrain_models::gguf::{GgufError, GgufHeader};
use onebrain_proto::message::{DeviceBrief, Envelope, Message, StreamKind};
use onebrain_proto::plan::{Epoch, NodeId, Plan};
use serde::Serialize;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;

use crate::engine_host::{EngineHost, HostMsg};

/// Fixed OS reserve subtracted from measured free memory when computing a
/// node's schedulable budget (docs/distributed.md "Placement": usable memory
/// is measured free minus a fixed OS reserve — never total RAM).
pub const OS_RESERVE_BYTES: u64 = 1 << 30;

/// How long a worker serve thread gets to finish after its bridge closes.
const SERVE_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

/// This node's `NodeStatus` payload: `(usable_memory_bytes, devices)`.
/// Usable memory is the CPU device's measured free bytes minus
/// [`OS_RESERVE_BYTES`] — or `override_bytes` when the test-only
/// `[debug] usable_memory_override_bytes` config knob is set (the same value
/// is then reported to peers and budgeted locally; real allocation is never
/// touched).
pub fn local_node_status(override_bytes: Option<u64>) -> (u64, Vec<DeviceBrief>) {
    let devices = onebrain_engine::devices();
    let briefs = devices
        .iter()
        .map(|d| DeviceBrief {
            kind: match d.kind {
                DeviceKind::Cpu => "cpu",
                DeviceKind::Gpu => "gpu",
                DeviceKind::IntegratedGpu => "igpu",
                DeviceKind::Accelerator => "accel",
                DeviceKind::Other => "other",
            }
            .to_string(),
            free_bytes: d.free_bytes,
            total_bytes: d.total_bytes,
        })
        .collect();
    let usable = override_bytes.unwrap_or_else(|| {
        devices
            .iter()
            .find(|d| matches!(d.kind, DeviceKind::Cpu))
            .map(|d| d.free_bytes)
            .unwrap_or(0)
            .saturating_sub(OS_RESERVE_BYTES)
    });
    (usable, briefs)
}

/// Index (into the engine's device enumeration) of the device a worker
/// serves over RPC — the CPU device in M3 (docs/distributed.md).
fn serve_device_index() -> i32 {
    onebrain_engine::devices()
        .iter()
        .position(|d| matches!(d.kind, DeviceKind::Cpu))
        .unwrap_or(0) as i32
}

/// Read the transformer layer count (`{arch}.block_count`) from a GGUF
/// file's header. Blocking file I/O — call from a blocking context.
pub fn gguf_layer_count(path: &Path) -> Result<u32, String> {
    use std::io::Read;
    let display = path.display();
    let file_len = std::fs::metadata(path)
        .map_err(|e| format!("cannot read model file {display}: {e}; check the path exists"))?
        .len();
    let mut want: u64 = (1u64 << 20).min(file_len);
    loop {
        let mut bytes = Vec::with_capacity(want as usize);
        std::fs::File::open(path)
            .and_then(|f| f.take(want).read_to_end(&mut bytes))
            .map_err(|e| format!("cannot read model file {display}: {e}; check permissions"))?;
        match GgufHeader::parse(&bytes) {
            Ok(header) => {
                let arch = header
                    .metadata
                    .get("general.architecture")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        format!(
                            "model {display} declares no general.architecture; the file may be \
                             corrupt — re-download it with `onebrain pull`"
                        )
                    })?
                    .to_string();
                let key = format!("{arch}.block_count");
                let count = header
                    .metadata
                    .get(&key)
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        format!(
                            "model {display} declares no {key}; a layer split cannot be \
                             computed — re-download it with `onebrain pull`"
                        )
                    })?;
                return u32::try_from(count).map_err(|_| {
                    format!("model {display} declares an absurd layer count ({count})")
                });
            }
            Err(GgufError::NeedMoreData { need_hint }) if want < file_len => {
                want = need_hint.max(want.saturating_mul(2)).min(file_len);
            }
            Err(e) => {
                return Err(format!(
                    "cannot parse the GGUF header of {display}: {e}; re-download the model \
                     with `onebrain pull`"
                ))
            }
        }
    }
}

/// The active plan as reported by `GET /api/internal/status`.
#[derive(Debug, Clone, Serialize)]
pub struct ActivePlanView {
    /// This node's role in the plan: `"head"` or `"worker"`.
    pub role: &'static str,
    /// The plan itself (epoch, strategy, assignments, ctx_len, model).
    pub plan: Plan,
    /// The scheduler's prose explanation (head side only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
}

/// A worker's answer to a plan proposal, recorded for the awaiting head.
#[derive(Debug, Clone)]
pub struct AckResult {
    pub ready: bool,
    pub detail: Option<String>,
}

/// Shared cluster-session state: the epoch counter, the active plan (for
/// status), received plan acks, and the head's bridge tasks.
pub struct ClusterState {
    epoch_counter: AtomicU64,
    active: StdMutex<Option<ActivePlanView>>,
    acks: StdMutex<HashMap<(String, u64), AckResult>>,
    ack_notify: Notify,
    head_bridges: StdMutex<Vec<JoinHandle<()>>>,
}

impl ClusterState {
    pub fn new() -> Arc<ClusterState> {
        Arc::new(ClusterState {
            epoch_counter: AtomicU64::new(0),
            active: StdMutex::new(None),
            acks: StdMutex::new(HashMap::new()),
            ack_notify: Notify::new(),
            head_bridges: StdMutex::new(Vec::new()),
        })
    }

    /// Stamp the next plan epoch (monotonic per daemon lifetime, first is 1).
    pub fn next_epoch(&self) -> Epoch {
        Epoch(self.epoch_counter.fetch_add(1, Ordering::SeqCst) + 1)
    }

    /// The active plan, if any.
    pub fn active(&self) -> Option<ActivePlanView> {
        self.active.lock().expect("cluster state poisoned").clone()
    }

    /// Replace (or clear) the active plan.
    pub fn set_active(&self, view: Option<ActivePlanView>) {
        *self.active.lock().expect("cluster state poisoned") = view;
    }

    /// Record a worker's `PlanAck` and wake any waiter.
    pub fn record_ack(&self, peer_id: &str, epoch: u64, ready: bool, detail: Option<String>) {
        self.acks
            .lock()
            .expect("cluster state poisoned")
            .insert((peer_id.to_string(), epoch), AckResult { ready, detail });
        self.ack_notify.notify_waiters();
    }

    /// Await a ready `PlanAck` from every listed worker for `epoch`. Workers
    /// are `(peer_id, display_label)` pairs; errors name the offending node
    /// (contract: any nack or timeout aborts activation).
    pub async fn await_acks(
        &self,
        epoch: u64,
        workers: &[(String, String)],
        wait: Duration,
    ) -> Result<(), String> {
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            // Register for wakeups BEFORE checking, so a notify between the
            // check and the await is never lost.
            let notified = self.ack_notify.notified();
            {
                let acks = self.acks.lock().expect("cluster state poisoned");
                let mut missing: Option<&str> = None;
                for (id, label) in workers {
                    match acks.get(&(id.clone(), epoch)) {
                        Some(ack) if ack.ready => {}
                        Some(ack) => {
                            let detail = ack
                                .detail
                                .clone()
                                .unwrap_or_else(|| "no reason given".to_string());
                            return Err(format!(
                                "node '{label}' rejected plan epoch {epoch}: {detail}; run \
                                 `onebrain status` on that node and retry"
                            ));
                        }
                        None => missing = missing.or(Some(label)),
                    }
                }
                match missing {
                    None => return Ok(()),
                    Some(label) if tokio::time::Instant::now() >= deadline => {
                        return Err(format!(
                            "node '{label}' did not acknowledge plan epoch {epoch} within \
                             {}s; check it is online (`onebrain status`) and retry",
                            wait.as_secs()
                        ));
                    }
                    Some(_) => {}
                }
            }
            let _ = tokio::time::timeout_at(deadline, notified).await;
        }
    }

    /// Install the head's bridge tasks for a freshly activated epoch,
    /// aborting any leftovers from the previous one. Call only AFTER the
    /// previous model has been unloaded/replaced (ADR 0004 ordering).
    pub fn replace_head_bridges(&self, tasks: Vec<JoinHandle<()>>) {
        let mut slot = self.head_bridges.lock().expect("cluster state poisoned");
        for old in slot.drain(..) {
            old.abort();
        }
        *slot = tasks;
    }

    /// Abort all head bridge tasks (model unload/teardown, after the model
    /// dropped).
    pub fn teardown_head_bridges(&self) {
        self.replace_head_bridges(Vec::new());
    }
}

/// What the worker decided about a received `PlanProposal`.
#[derive(Debug, PartialEq, Eq)]
enum ProposalDecision {
    /// Sender is not paired (should be impossible through the authenticated
    /// mesh, but never act on it): no state change, no reply.
    Ignore,
    /// Stale epoch: fenced, replied with a nack.
    Reject(String),
    /// Newer epoch from a paired peer: adopt.
    Adopt,
}

/// The pure fencing rule (docs/distributed.md "Epoch lifecycle"): only a
/// paired sender may propose, and only an epoch strictly newer than the
/// active one is adopted.
fn decide_proposal(
    active: Option<Epoch>,
    sender_paired: bool,
    proposed: Epoch,
) -> ProposalDecision {
    if !sender_paired {
        return ProposalDecision::Ignore;
    }
    match active {
        Some(current) if proposed <= current => ProposalDecision::Reject(format!(
            "epoch {} is not newer than the active epoch {} (stale plans are fenced)",
            proposed.0, current.0
        )),
        _ => ProposalDecision::Adopt,
    }
}

/// One bridged serve session on the worker: the engine's RPC serve thread
/// plus the pump task relaying its socket to the mesh stream.
struct WorkerServe {
    session: RpcSession,
    pump: JoinHandle<()>,
}

/// The worker's adopted plan and its live serve sessions.
struct AdoptedPlan {
    epoch: Epoch,
    head: NodeId,
    serves: Vec<WorkerServe>,
}

/// Spawn the cluster task. It exits when the mesh service stops (both
/// channels close), tearing down any worker serve sessions first.
pub fn spawn_cluster_task(
    mesh: MeshHandle,
    host: EngineHost,
    state: Arc<ClusterState>,
    ctrl_rx: mpsc::Receiver<ControlMessage>,
    rpc_rx: mpsc::Receiver<IncomingRpcStream>,
) -> JoinHandle<()> {
    tokio::spawn(cluster_task(mesh, host, state, ctrl_rx, rpc_rx))
}

async fn cluster_task(
    mesh: MeshHandle,
    host: EngineHost,
    state: Arc<ClusterState>,
    mut ctrl_rx: mpsc::Receiver<ControlMessage>,
    mut rpc_rx: mpsc::Receiver<IncomingRpcStream>,
) {
    let mut adopted: Option<AdoptedPlan> = None;
    let mut ctrl_open = true;
    let mut rpc_open = true;
    while ctrl_open || rpc_open {
        tokio::select! {
            msg = ctrl_rx.recv(), if ctrl_open => match msg {
                Some(msg) => handle_control(&mesh, &host, &state, &mut adopted, msg).await,
                None => ctrl_open = false,
            },
            stream = rpc_rx.recv(), if rpc_open => match stream {
                Some(stream) => handle_rpc_stream(&mut adopted, stream),
                None => rpc_open = false,
            },
        }
    }
    if let Some(plan) = adopted.take() {
        teardown_adopted(plan).await;
    }
    tracing::debug!("cluster task stopped");
}

async fn handle_control(
    mesh: &MeshHandle,
    host: &EngineHost,
    state: &Arc<ClusterState>,
    adopted: &mut Option<AdoptedPlan>,
    msg: ControlMessage,
) {
    let sender = msg.peer;
    match msg.envelope.message {
        Message::PlanProposal(plan) => {
            handle_proposal(mesh, host, state, adopted, sender, plan).await;
        }
        Message::PlanAck {
            epoch,
            ready,
            detail,
        } => {
            tracing::debug!(peer = %sender.0, epoch = epoch.0, ready, "plan ack received");
            state.record_ack(&sender.0, epoch.0, ready, detail);
        }
        Message::NodeStatus { .. } => {
            // Cached by the mesh into the peer view; nothing to do here.
        }
        other => {
            tracing::debug!(peer = %sender.0, "ignoring unexpected control message: {other:?}");
        }
    }
}

async fn handle_proposal(
    mesh: &MeshHandle,
    host: &EngineHost,
    state: &Arc<ClusterState>,
    adopted: &mut Option<AdoptedPlan>,
    sender: NodeId,
    plan: Plan,
) {
    // The mesh only delivers control traffic from live sessions of paired
    // peers, but verify against the store anyway: pairing is the trust
    // boundary (§10) and unpairing may race delivery.
    let sender_paired = match mesh.peers().await {
        Ok(peers) => peers.iter().any(|p| p.id == sender.0),
        Err(err) => {
            tracing::warn!(error = %err, "cannot verify proposal sender; ignoring proposal");
            false
        }
    };
    let epoch = plan.epoch;
    match decide_proposal(adopted.as_ref().map(|a| a.epoch), sender_paired, epoch) {
        ProposalDecision::Ignore => {
            tracing::warn!(
                peer = %sender.0,
                epoch = epoch.0,
                "ignoring plan proposal from an unpaired sender"
            );
        }
        ProposalDecision::Reject(reason) => {
            tracing::info!(peer = %sender.0, epoch = epoch.0, %reason, "rejecting plan proposal");
            let ack = Envelope::new(Message::PlanAck {
                epoch,
                ready: false,
                detail: Some(reason),
            });
            if let Err(err) = mesh.send_control(&sender.0, ack).await {
                tracing::warn!(peer = %sender.0, "could not send plan nack: {err}");
            }
        }
        ProposalDecision::Adopt => {
            tracing::info!(
                peer = %sender.0,
                epoch = epoch.0,
                model = %plan.model,
                "adopting plan; serving a shard for its head"
            );
            if let Some(old) = adopted.take() {
                teardown_adopted(old).await;
            }
            // M3 contract: adopting a plan unloads any locally loaded model.
            let _ = host.send(HostMsg::ServeShard { epoch });
            state.set_active(Some(ActivePlanView {
                role: "worker",
                plan: plan.clone(),
                explanation: None,
            }));
            *adopted = Some(AdoptedPlan {
                epoch,
                head: sender.clone(),
                serves: Vec::new(),
            });
            let ack = Envelope::new(Message::PlanAck {
                epoch,
                ready: true,
                detail: None,
            });
            if let Err(err) = mesh.send_control(&sender.0, ack).await {
                tracing::warn!(peer = %sender.0, "could not send plan ack: {err}");
            }
        }
    }
}

/// Bridge one accepted `rpc` stream into an in-process GGML RPC serve
/// session — or refuse it with close code 4 (`bad-epoch`) when it is not for
/// the active epoch from that epoch's head (the M3 fencing rule).
fn handle_rpc_stream(adopted: &mut Option<AdoptedPlan>, stream: IncomingRpcStream) {
    let Some(active) = adopted.as_mut() else {
        tracing::warn!(
            peer = %stream.peer.0,
            epoch = stream.epoch.0,
            "rpc stream with no active epoch; refusing (bad-epoch)"
        );
        stream.refuse(4);
        return;
    };
    if stream.epoch != active.epoch || stream.peer != active.head {
        tracing::warn!(
            peer = %stream.peer.0,
            epoch = stream.epoch.0,
            active_epoch = active.epoch.0,
            "rpc stream fenced: wrong epoch or not the epoch's head (bad-epoch)"
        );
        stream.refuse(4);
        return;
    }
    let mut session = match RpcSession::start(0, serve_device_index()) {
        Ok(session) => session,
        Err(err) => {
            tracing::error!(error = %err, "cannot start an rpc serve session");
            stream.refuse(4);
            return;
        }
    };
    let bridge = session
        .take_bridge()
        .expect("a fresh RpcSession always holds its bridge end");
    let (read_half, write_half) = match wrap_bridge(bridge) {
        Ok(halves) => halves,
        Err(err) => {
            tracing::error!(error = %err, "cannot make the bridge socket async");
            stream.refuse(4);
            return;
        }
    };
    tracing::info!(
        epoch = stream.epoch.0,
        "rpc serve session bridged to mesh stream"
    );
    let pump = spawn_pump(stream.recv, stream.send, read_half, write_half);
    active.serves.push(WorkerServe { session, pump });
}

/// Stop every serve session of a retired plan: abort the pumps (closing the
/// bridge sockets, which ends the serve threads) and join the threads.
async fn teardown_adopted(plan: AdoptedPlan) {
    for serve in plan.serves {
        serve.pump.abort();
        let session = serve.session;
        let joined = tokio::task::spawn_blocking(move || session.shutdown(SERVE_JOIN_TIMEOUT));
        match joined.await {
            Ok(true) => {}
            Ok(false) => tracing::warn!(
                "an rpc serve thread did not end within {SERVE_JOIN_TIMEOUT:?}; leaking it"
            ),
            Err(err) => tracing::warn!("serve teardown task failed: {err}"),
        }
    }
    tracing::info!(epoch = plan.epoch.0, "worker plan torn down");
}

#[cfg(unix)]
type BridgeReadHalf = tokio::net::unix::OwnedReadHalf;
#[cfg(unix)]
type BridgeWriteHalf = tokio::net::unix::OwnedWriteHalf;
#[cfg(windows)]
type BridgeReadHalf = tokio::net::tcp::OwnedReadHalf;
#[cfg(windows)]
type BridgeWriteHalf = tokio::net::tcp::OwnedWriteHalf;

/// Wrap the std bridge socket into async halves.
#[cfg(unix)]
fn wrap_bridge(bridge: BridgeStream) -> std::io::Result<(BridgeReadHalf, BridgeWriteHalf)> {
    bridge.set_nonblocking(true)?;
    Ok(tokio::net::UnixStream::from_std(bridge)?.into_split())
}

/// Wrap the std bridge socket into async halves.
#[cfg(windows)]
fn wrap_bridge(bridge: BridgeStream) -> std::io::Result<(BridgeReadHalf, BridgeWriteHalf)> {
    bridge.set_nonblocking(true)?;
    Ok(tokio::net::TcpStream::from_std(bridge)?.into_split())
}

/// Relay bytes 1:1 between a mesh `rpc` stream and a local socket, both
/// directions concurrently. EOF (or error) on one side half-closes the
/// other, so the relay winds down as soon as either peer closes.
async fn pump_streams<R, W>(recv: RecvStream, send: SendStream, sock_read: R, sock_write: W)
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let to_socket = async move {
        let mut recv = recv;
        let mut sock_write = sock_write;
        if let Err(err) = tokio::io::copy(&mut recv, &mut sock_write).await {
            tracing::debug!(error = %err, "rpc bridge mesh->socket ended with error");
        }
        let _ = sock_write.shutdown().await;
    };
    let to_mesh = async move {
        let mut sock_read = sock_read;
        let mut send = send;
        if let Err(err) = tokio::io::copy(&mut sock_read, &mut send).await {
            tracing::debug!(error = %err, "rpc bridge socket->mesh ended with error");
        }
        let _ = send.finish();
    };
    tokio::join!(to_socket, to_mesh);
}

/// [`pump_streams`] on its own task (worker-side bridges own their lifetime
/// through the serve thread, so a detached task is correct there).
fn spawn_pump<R, W>(
    recv: RecvStream,
    send: SendStream,
    sock_read: R,
    sock_write: W,
) -> JoinHandle<()>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    tokio::spawn(pump_streams(recv, send, sock_read, sock_write))
}

/// Head side of one rpc tunnel: bind a loopback listener on an ephemeral
/// port and keep accepting for the epoch's lifetime, opening one fresh mesh
/// `rpc` stream per accepted connection.
///
/// Accept-once is NOT enough here (learned empirically, ADR 0004 amendment):
/// the vendored RPC client dials the endpoint string repeatedly — a
/// device-count probe at registration that closes immediately, per-device
/// property/memory queries during load, then the long-lived buffer/compute
/// connection — six sequential connections observed for one load. The
/// listener is loopback-only and lives exactly as long as the bridge task
/// (aborted at teardown, which also aborts every in-flight pump via the
/// JoinSet). The socket-scan rule permits loopback bridge listeners while a
/// distributed session is active and asserts they are gone after teardown.
pub async fn head_bridge(
    mesh: MeshHandle,
    peer: String,
    epoch: Epoch,
) -> Result<(String, JoinHandle<()>), String> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|e| {
            format!(
                "cannot bind a loopback rpc bridge: {e}; check that local firewall or \
                 endpoint-security software allows loopback listeners for this process"
            )
        })?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("cannot read the rpc bridge port: {e}"))?
        .port();
    let task = tokio::spawn(async move {
        // Dropping the JoinSet (task end or abort) aborts every pump, which
        // closes the bridged sockets and thereby the workers' serve sessions.
        let mut pumps = tokio::task::JoinSet::new();
        loop {
            let (sock, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(err) => {
                    tracing::warn!(error = %err, "rpc bridge accept failed; bridge closing");
                    break;
                }
            };
            let _ = sock.set_nodelay(true);
            let (send, recv) = match mesh.open_stream(&peer, StreamKind::Rpc, epoch).await {
                Ok(pair) => pair,
                Err(err) => {
                    // Dropping the accepted socket gives the RPC client a
                    // clean EOF instead of a hang.
                    tracing::warn!(
                        error = %err, peer = %peer,
                        "opening an rpc stream for a bridge connection failed"
                    );
                    continue;
                }
            };
            let (read_half, write_half) = sock.into_split();
            pumps.spawn(pump_streams(recv, send, read_half, write_half));
        }
    });
    Ok((format!("127.0.0.1:{port}"), task))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpaired_proposal_is_ignored() {
        // Even a strictly newer epoch is ignored when the sender is not
        // paired — no ack, no adoption (the §10 trust boundary).
        assert_eq!(
            decide_proposal(None, false, Epoch(5)),
            ProposalDecision::Ignore
        );
        assert_eq!(
            decide_proposal(Some(Epoch(1)), false, Epoch(9)),
            ProposalDecision::Ignore
        );
    }

    #[test]
    fn stale_epochs_are_fenced() {
        match decide_proposal(Some(Epoch(4)), true, Epoch(4)) {
            ProposalDecision::Reject(reason) => {
                assert!(reason.contains("epoch 4"), "{reason}");
                assert!(reason.contains("fenced"), "{reason}");
            }
            other => panic!("expected Reject, got {other:?}"),
        }
        assert!(matches!(
            decide_proposal(Some(Epoch(4)), true, Epoch(3)),
            ProposalDecision::Reject(_)
        ));
    }

    #[test]
    fn newer_epoch_from_paired_sender_is_adopted() {
        assert_eq!(
            decide_proposal(None, true, Epoch(1)),
            ProposalDecision::Adopt
        );
        assert_eq!(
            decide_proposal(Some(Epoch(4)), true, Epoch(5)),
            ProposalDecision::Adopt
        );
    }

    #[test]
    fn epoch_counter_is_monotonic_from_one() {
        let state = ClusterState::new();
        assert_eq!(state.next_epoch(), Epoch(1));
        assert_eq!(state.next_epoch(), Epoch(2));
    }

    #[tokio::test]
    async fn await_acks_reports_ready_nack_and_timeout() {
        let state = ClusterState::new();
        let workers = vec![("id-a".to_string(), "alpha".to_string())];

        // Timeout names the silent node.
        let err = state
            .await_acks(7, &workers, Duration::from_millis(50))
            .await
            .expect_err("no ack must time out");
        assert!(err.contains("alpha"), "{err}");
        assert!(err.contains("epoch 7"), "{err}");

        // A nack names the node and carries the detail.
        state.record_ack("id-a", 8, false, Some("busy".into()));
        let err = state
            .await_acks(8, &workers, Duration::from_secs(5))
            .await
            .expect_err("a nack must fail activation");
        assert!(err.contains("alpha") && err.contains("busy"), "{err}");

        // A ready ack (even recorded before the wait) succeeds.
        state.record_ack("id-a", 9, true, None);
        state
            .await_acks(9, &workers, Duration::from_secs(5))
            .await
            .expect("ready ack must succeed");
    }

    #[test]
    fn local_node_status_override_wins_and_devices_are_reported() {
        let (usable, devices) = local_node_status(Some(123_456));
        assert_eq!(usable, 123_456);
        assert!(
            devices.iter().any(|d| d.kind == "cpu"),
            "the CPU device must always be reported"
        );
        // Without the override: measured free minus the OS reserve, never
        // total RAM.
        let (measured, devices) = local_node_status(None);
        let cpu = devices.iter().find(|d| d.kind == "cpu").unwrap();
        assert!(measured <= cpu.free_bytes.saturating_sub(0));
        assert!(measured < cpu.total_bytes, "must never budget total RAM");
    }

    #[test]
    fn gguf_layer_count_rejects_non_gguf_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.gguf");
        std::fs::write(&path, b"definitely not a gguf header").unwrap();
        let err = gguf_layer_count(&path).expect_err("garbage must not parse");
        assert!(err.contains("onebrain pull"), "remedy missing: {err}");
    }

    #[test]
    fn gguf_layer_count_reads_the_smoke_model() {
        // Real-model check, only when the smoke model is available (CI and
        // local runs set OB_SMOKE_MODEL).
        let Ok(path) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping gguf layer count smoke test");
            return;
        };
        let layers = gguf_layer_count(Path::new(&path)).expect("smoke model header parses");
        assert!(layers > 0, "layer count must be positive");
    }
}
