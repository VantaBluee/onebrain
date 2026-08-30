//! Integration tests for the daemon's [`DaemonBenchSource`]
//! (docs/perf.md §10): a peer's `bench_query` runs THIS node's real
//! microbench through the mesh's `BenchRequest`/`BenchReport` exchange and
//! leaves the local profile fresh, while a busy / shard-serving /
//! unbenched daemon declines with the wire's cannot-bench-now marker.
//!
//! The live-pair test needs `OB_SMOKE_MODEL` (the stories260K smoke model)
//! and quietly skips without it, matching the engine-crate convention; the
//! decline paths never touch the engine and always run.

use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use onebrain_mesh::{
    identity, BenchSource, MeshConfig, MeshHandle, MeshService, PairTarget, PeerState, PeerStatus,
};
use onebrain_models::registry::{ModelRef, Resolved};
use onebrain_proto::plan::{Epoch, NodeId};
use onebraind::cluster::ClusterState;
use onebraind::engine_host::{EngineHost, HostMsg, HostPerf};
use onebraind::server::{DaemonBenchSource, SharedProfile, BENCH_MODEL_ID};
use tempfile::TempDir;

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
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

async fn spawn_node(dir: &TempDir, name: &str, config: MeshConfig) -> MeshHandle {
    let key = identity::load_or_create(dir.path()).expect("device key");
    MeshService::spawn(key, dir.path().join("peers.toml"), name.to_string(), config)
        .await
        .expect("mesh service spawns")
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

/// Seed the registry bench model into `cache_root` from the smoke GGUF,
/// with a manifest matching the registry URL — the exact cached-file
/// fast-path shape `DaemonBenchSource` (and `download`) checks.
fn seed_bench_model(cache_root: &Path, smoke: &str) {
    let spec = match BENCH_MODEL_ID
        .parse::<ModelRef>()
        .unwrap()
        .resolve()
        .unwrap()
    {
        Resolved::Remote(spec) => spec,
        other => panic!("registry id must resolve remote, got {other:?}"),
    };
    let dest = cache_root.join(&spec.cache_key);
    std::fs::create_dir_all(&dest).unwrap();
    std::fs::copy(smoke, dest.join(&spec.file_name)).unwrap();
    let size = std::fs::metadata(dest.join(&spec.file_name)).unwrap().len();
    let manifest = onebrain_models::download::Manifest {
        url: spec.url.clone(),
        size_bytes: size,
        // The cached-file fast path checks url + size, not the hash.
        blake3: "seeded-by-test".to_string(),
    };
    std::fs::write(
        dest.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

/// The decline paths need no engine and no mesh: `bench()` must answer
/// `None` (the mesh then puts the cannot-bench-now marker on the wire)
/// whenever measuring would lie or steal — before any model is touched.
#[test]
fn bench_source_declines_when_busy_shard_serving_or_unbenched() {
    let dir = tempfile::tempdir().unwrap();
    let (host, host_thread) = EngineHost::spawn(None, HostPerf::default());
    let cluster = ClusterState::new();
    let profile: SharedProfile = Arc::new(Mutex::new(None));
    let source = DaemonBenchSource {
        host: host.clone(),
        cluster: cluster.clone(),
        cache_root: dir.path().join("models"),
        profile: profile.clone(),
        profile_path: dir.path().join("profile.toml"),
    };

    // Idle, no shard — but the test model is not cached: decline (a
    // passive peer never downloads to answer a query).
    assert!(
        source.bench().is_none(),
        "an uncached test model must decline"
    );
    // A generation in the daemon (queued or in flight): decline.
    host.job_started();
    assert!(source.bench().is_none(), "a busy daemon must decline");
    host.job_finished();
    // Serving a pipeline shard: decline (head-driven decode traffic is
    // invisible to the local job counter).
    cluster.set_worker_shard(Some((Epoch(4), NodeId("head-id".into()))));
    assert!(
        source.bench().is_none(),
        "a shard-serving worker must decline"
    );
    cluster.set_worker_shard(None);

    // No decline path may touch the profile.
    assert!(profile.lock().unwrap().is_none());
    host.send(HostMsg::Shutdown).unwrap();
    host_thread.join().unwrap();
}

/// The full path over a live loopback pair: B's `bench_query` makes A run
/// its real compute/disk microbench and A's own profile (memory +
/// profile.toml) comes back fresh — exactly what `POST /api/internal/bench`
/// would have left, but triggered by a peer (docs/perf.md §10).
#[tokio::test(flavor = "multi_thread")]
async fn bench_query_runs_the_daemon_microbench_over_a_live_pair() {
    let Ok(smoke) = std::env::var("OB_SMOKE_MODEL") else {
        eprintln!("OB_SMOKE_MODEL not set; skipping the live bench-source test");
        return;
    };
    let a_dir = TempDir::new().unwrap();
    let b_dir = TempDir::new().unwrap();
    let cache_root = a_dir.path().join("models");
    seed_bench_model(&cache_root, &smoke);

    let (host, host_thread) = EngineHost::spawn(None, HostPerf::default());
    let profile: SharedProfile = Arc::new(Mutex::new(None));
    // Nested like a real <config_dir>: the save must create parents.
    let profile_path = a_dir.path().join("config").join("profile.toml");
    let mut a_config = test_config();
    a_config.bench_source = Some(Arc::new(DaemonBenchSource {
        host: host.clone(),
        cluster: ClusterState::new(),
        cache_root: cache_root.clone(),
        profile: profile.clone(),
        profile_path: profile_path.clone(),
    }));
    let a = spawn_node(&a_dir, "node-a", a_config).await;
    let b = spawn_node(&b_dir, "node-b", test_config()).await;
    pair_and_connect(&a, &b).await;

    let report = b.bench_query("node-a").await.expect("bench query answers");
    assert!(
        !report.is_unavailable(),
        "an idle daemon with the model cached must measure, got {report:?}"
    );
    assert!(report.prefill_tps > 0.0, "{report:?}");
    assert!(report.decode_tps > 0.0, "{report:?}");
    assert!(report.disk_mbps > 0.0, "{report:?}");

    // The measurement refreshed A's own shared profile and persisted it —
    // like a local `POST /api/internal/bench` would have.
    let stored = (*profile.lock().unwrap()).expect("SharedProfile refreshed");
    assert_eq!(stored.measured_unix, report.measured_unix);
    assert_eq!(stored.decode_tps, report.decode_tps);
    let persisted =
        onebrain_scheduler::load_profile(&profile_path).expect("profile.toml persisted");
    assert_eq!(persisted.measured_unix, report.measured_unix);

    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
    host.send(HostMsg::Shutdown).unwrap();
    host_thread.join().unwrap();
}
