//! M7 cluster-bench integration tests (docs/perf.md §10): the
//! `BenchRequest` → `BenchReport` control exchange over two live loopback
//! `MeshService`s — a stub [`BenchSource`] answering with real figures, the
//! sourceless default answering with the cannot-bench-now marker, and a
//! wired source that declines collapsing to the same marker.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use onebrain_mesh::{
    identity, BenchSource, MeshConfig, MeshHandle, MeshService, PairTarget, PeerBenchReport,
    PeerState, PeerStatus,
};
use tempfile::TempDir;

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

async fn spawn_node(dir: &TempDir, name: &str, config: MeshConfig) -> MeshHandle {
    let key = identity::load_or_create(dir.path()).expect("device key");
    MeshService::spawn(key, dir.path().join("peers.toml"), name.to_string(), config)
        .await
        .expect("mesh service spawns")
}

fn test_config() -> MeshConfig {
    MeshConfig {
        enable_mdns: false,
        enable_relays: false,
        engine_build: "test-build".to_string(),
        bind_addrs: vec![loopback()],
        ..MeshConfig::default()
    }
}

async fn pair_and_connect(a: &MeshHandle, b: &MeshHandle) {
    let window = a.pair_start().await.expect("window opens");
    b.pair_join(PairTarget::Ticket(window.ticket.clone()), Some(window.code))
        .await
        .expect("pairing succeeds");
    for (handle, what) in [(a, "a sees b connected"), (b, "b sees a connected")] {
        wait_for_peer(handle, what, |p| p.state == PeerState::Connected).await;
    }
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

/// The figures the stub "measures" — mirrors the M4 profile shape.
const STUB_REPORT: PeerBenchReport = PeerBenchReport {
    prefill_tps: 812.5,
    decode_tps: 41.25,
    disk_mbps: 1732.0,
    measured_unix: 1_756_252_800,
};

/// Stub bench: a canned report, or a decline (`None`) — standing in for
/// "the daemon is busy generating right now".
struct StubBench {
    report: Option<PeerBenchReport>,
}

impl BenchSource for StubBench {
    fn bench(&self) -> Option<PeerBenchReport> {
        self.report
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn bench_query_roundtrips_over_a_live_pair() {
    let a_dir = TempDir::new().unwrap();
    let b_dir = TempDir::new().unwrap();
    let mut a_config = test_config();
    a_config.bench_source = Some(Arc::new(StubBench {
        report: Some(STUB_REPORT),
    }));
    let a = spawn_node(&a_dir, "node-a", a_config).await;
    let b = spawn_node(&b_dir, "node-b", test_config()).await;
    pair_and_connect(&a, &b).await;

    // Wired source: the stub's figures arrive intact (by store name).
    let report = b.bench_query("node-a").await.expect("bench query answers");
    assert_eq!(report, STUB_REPORT);
    assert!(
        !report.is_unavailable(),
        "a real measurement must not read as the marker"
    );

    // Sourceless default (B has no bench source): the cannot-bench-now
    // marker, queried by endpoint id to exercise id-based resolution too.
    let b_id = b.endpoint_id().to_string();
    let none = a
        .bench_query(&b_id)
        .await
        .expect("sourceless node still answers");
    assert_eq!(none, PeerBenchReport::UNAVAILABLE);
    assert!(
        none.is_unavailable(),
        "missing source must reply with the measured_unix=0 marker"
    );

    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn declining_bench_source_replies_with_the_marker() {
    let a_dir = TempDir::new().unwrap();
    let b_dir = TempDir::new().unwrap();
    // A HAS a source, but it declines (e.g. a generation in flight): the
    // wire reply must be indistinguishable from "no source" — the marker.
    let mut a_config = test_config();
    a_config.bench_source = Some(Arc::new(StubBench { report: None }));
    let a = spawn_node(&a_dir, "node-a", a_config).await;
    let b = spawn_node(&b_dir, "node-b", test_config()).await;
    pair_and_connect(&a, &b).await;

    let report = b
        .bench_query("node-a")
        .await
        .expect("declining source still answers");
    assert!(report.is_unavailable(), "a decline must reply the marker");
    assert_eq!(report, PeerBenchReport::UNAVAILABLE);

    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
}
