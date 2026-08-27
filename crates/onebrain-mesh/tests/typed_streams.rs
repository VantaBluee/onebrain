//! M3 typed-stream integration tests: two loopback `MeshService`s exercise
//! the `StreamHeader` framing — control envelopes (plan traffic, NodeStatus)
//! and raw `rpc` byte streams — while the pre-M3 behaviors (pairing,
//! heartbeats driving `Connected`) keep working with the header in place.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use onebrain_mesh::{
    identity, MeshConfig, MeshError, MeshHandle, PairTarget, PeerState, PeerStatus,
};
use onebrain_proto::message::{DeviceBrief, Envelope, Message, StreamKind};
use onebrain_proto::plan::{Assignment, Epoch, LayerRange, NodeId, Plan, Strategy};
use tempfile::TempDir;
use tokio::time::timeout;

const BETA_USABLE: u64 = 7 * 1024 * 1024 * 1024;
const BETA_PREFILL_TPS: f64 = 812.5;
const BETA_DECODE_TPS: f64 = 41.25;
const BETA_DISK_MBPS: f64 = 1732.0;

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

async fn spawn_node(dir: &TempDir, name: &str, usable: Option<u64>) -> MeshHandle {
    let key = identity::load_or_create(dir.path()).expect("device key");
    let node_status = usable.map(|bytes| {
        Arc::new(move || onebrain_mesh::NodeStatusReport {
            usable_memory_bytes: bytes,
            devices: vec![DeviceBrief {
                kind: "cpu".to_string(),
                free_bytes: bytes,
                total_bytes: bytes * 2,
            }],
            prefill_tps: Some(BETA_PREFILL_TPS),
            decode_tps: Some(BETA_DECODE_TPS),
            disk_mbps: Some(BETA_DISK_MBPS),
        }) as onebrain_mesh::NodeStatusFn
    });
    onebrain_mesh::MeshService::spawn(
        key,
        dir.path().join("peers.toml"),
        name.to_string(),
        MeshConfig {
            enable_mdns: false,
            enable_relays: false,
            engine_build: "test-build".to_string(),
            pair_window: Duration::from_secs(120),
            bind_addrs: vec![loopback()],
            node_status,
        },
    )
    .await
    .expect("mesh service spawns")
}

async fn wait_for_peer(handle: &MeshHandle, what: &str, pred: impl Fn(&PeerStatus) -> bool) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut last: Vec<PeerStatus> = Vec::new();
    while tokio::time::Instant::now() < deadline {
        last = handle.peers().await.expect("peers() answers");
        if last.iter().any(&pred) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("timed out waiting for {what}; last peers snapshot: {last:?}");
}

fn test_plan(epoch: u64, worker: &str) -> Plan {
    Plan {
        epoch: Epoch(epoch),
        model: "blake3:test".into(),
        strategy: Strategy::PipelineParallel,
        assignments: vec![Assignment {
            node: NodeId(worker.into()),
            layers: LayerRange { start: 0, end: 4 },
            stage: 0,
        }],
        ctx_len: 2048,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn typed_streams_control_and_rpc_roundtrip() {
    let alpha_dir = TempDir::new().unwrap();
    let beta_dir = TempDir::new().unwrap();
    let alpha = spawn_node(&alpha_dir, "alpha", None).await;
    let beta = spawn_node(&beta_dir, "beta", Some(BETA_USABLE)).await;

    // Take the consumers before any traffic can arrive.
    let mut alpha_ctrl = alpha.incoming_control().await.expect("alpha control rx");
    let mut beta_ctrl = beta.incoming_control().await.expect("beta control rx");
    let mut beta_rpc = beta.incoming_rpc().await.expect("beta rpc rx");
    // Single-consumer contract: a second take is refused.
    assert!(matches!(
        beta.incoming_control().await,
        Err(MeshError::ConsumerTaken { what: "control" })
    ));
    assert!(matches!(
        beta.incoming_rpc().await,
        Err(MeshError::ConsumerTaken { what: "rpc" })
    ));

    // Pair via ticket, then heartbeats (now header-framed Control streams)
    // must still drive both sides to Connected.
    let window = alpha.pair_start().await.expect("window opens");
    beta.pair_join(PairTarget::Ticket(window.ticket.clone()), Some(window.code))
        .await
        .expect("pairing succeeds");
    wait_for_peer(&alpha, "alpha->beta connected", |p| {
        p.state == PeerState::Connected && p.rtt_ms.is_some()
    })
    .await;
    wait_for_peer(&beta, "beta->alpha connected", |p| {
        p.state == PeerState::Connected && p.rtt_ms.is_some()
    })
    .await;

    // Beta's NodeStatus (sent right after Hello) lands in alpha's peer view,
    // including the M4 microbench profile fields.
    wait_for_peer(&alpha, "beta's NodeStatus budget + profile", |p| {
        p.usable_memory_bytes == Some(BETA_USABLE)
            && p.prefill_tps == Some(BETA_PREFILL_TPS)
            && p.decode_tps == Some(BETA_DECODE_TPS)
            && p.disk_mbps == Some(BETA_DISK_MBPS)
    })
    .await;

    let alpha_id = alpha.endpoint_id().to_string();
    let beta_id = beta.endpoint_id().to_string();

    // Control by NAME: alpha proposes a plan; beta receives it attributed to
    // alpha's authenticated endpoint id. (Skip NodeStatus envelopes: alpha
    // may or may not have sent one depending on its provider.)
    let plan = test_plan(3, &beta_id);
    alpha
        .send_control("beta", Envelope::new(Message::PlanProposal(plan.clone())))
        .await
        .expect("send_control by name");
    let received = timeout(Duration::from_secs(15), async {
        loop {
            let msg = beta_ctrl.recv().await.expect("beta control channel open");
            if let Message::PlanProposal(p) = msg.envelope.message {
                break (msg.peer, p);
            }
        }
    })
    .await
    .expect("beta receives the proposal");
    assert_eq!(received.0, NodeId(alpha_id.clone()));
    assert_eq!(received.1, plan);

    // Control by ENDPOINT ID: beta acks; alpha receives it.
    beta.send_control(
        &alpha_id,
        Envelope::new(Message::PlanAck {
            epoch: Epoch(3),
            ready: true,
            detail: None,
        }),
    )
    .await
    .expect("send_control by id");
    let ack = timeout(Duration::from_secs(15), async {
        loop {
            let msg = alpha_ctrl.recv().await.expect("alpha control channel open");
            if let Message::PlanAck { epoch, ready, .. } = msg.envelope.message {
                break (msg.peer, epoch, ready);
            }
        }
    })
    .await
    .expect("alpha receives the ack");
    assert_eq!(ack.0, NodeId(beta_id.clone()));
    assert_eq!(ack.1, Epoch(3));
    assert!(ack.2);

    // Rpc stream: opened with an epoch, delivered to beta's consumer, and
    // byte-transparent in both directions after the header.
    let (mut a_send, mut a_recv) = alpha
        .open_stream("beta", StreamKind::Rpc, Epoch(3))
        .await
        .expect("open rpc stream");
    let mut incoming = timeout(Duration::from_secs(15), beta_rpc.recv())
        .await
        .expect("rpc stream delivered")
        .expect("rpc channel open");
    assert_eq!(incoming.peer, NodeId(alpha_id.clone()));
    assert_eq!(incoming.epoch, Epoch(3));

    a_send.write_all(b"ping-rpc").await.expect("alpha writes");
    let mut buf = [0u8; 8];
    incoming
        .recv
        .read_exact(&mut buf)
        .await
        .expect("beta reads");
    assert_eq!(&buf, b"ping-rpc");
    incoming
        .send
        .write_all(b"pong-rpc")
        .await
        .expect("beta writes");
    a_recv.read_exact(&mut buf).await.expect("alpha reads");
    assert_eq!(&buf, b"pong-rpc");

    // Sending to an unknown peer names the known ones.
    let err = alpha
        .send_control(
            "nonexistent",
            Envelope::new(Message::Heartbeat { epoch: Epoch(0) }),
        )
        .await
        .expect_err("unknown peer errors");
    assert!(matches!(err, MeshError::UnknownPeerName { .. }), "{err}");

    alpha.shutdown().await.unwrap();
    beta.shutdown().await.unwrap();
}
