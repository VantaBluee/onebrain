//! M6 LAN-first logistics over a real paired mesh (docs/logistics.md):
//! two loopback `MeshService`s wired exactly like the daemon wires them
//! (persistent blob store + `LocalRangeInventory`), proving
//!
//! - a pull completes with ZERO WAN bytes when a paired peer holds the
//!   model (the WAN URL is unreachable, so success itself is the proof),
//!   with a byte-exact file and manifest;
//! - a worker's layer-range fetch pulls ONLY the header + its layers'
//!   ranges from a peer's range store, and a re-run moves nothing.
//!
//! No WAN is touched anywhere: every URL points at `wan.invalid`.

use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use onebrain_mesh::{
    identity, MeshConfig, MeshHandle, MeshService, PairTarget, PeerState, PeerStatus,
};
use onebrain_models::gguf::GgufHeader;
use onebrain_models::registry::DownloadSpec;
use onebrain_models::{download, ranges};
use onebraind::logistics::{ensure_remote_local, fetch_layer_ranges, LocalRangeInventory};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Mesh scaffolding (the blob_sharing.rs pattern, daemon-shaped)
// ---------------------------------------------------------------------------

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

/// Spawn one node the way `runtime.rs` does: hermetic transport, an
/// on-disk blob store, and the local range inventory over its cache root.
async fn spawn_node(dir: &TempDir, name: &str) -> (MeshHandle, std::path::PathBuf) {
    let cache_root = dir.path().join("models");
    std::fs::create_dir_all(&cache_root).unwrap();
    let key = identity::load_or_create(dir.path()).expect("device key");
    let config = MeshConfig {
        enable_mdns: false,
        enable_relays: false,
        engine_build: "test-build".to_string(),
        bind_addrs: vec![loopback()],
        blobs_dir: Some(dir.path().join("blobs")),
        range_source: Some(Arc::new(LocalRangeInventory::new(cache_root.clone()))),
        ..MeshConfig::default()
    };
    let handle = MeshService::spawn(key, dir.path().join("peers.toml"), name.to_string(), config)
        .await
        .expect("mesh service spawns");
    (handle, cache_root)
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

// ---------------------------------------------------------------------------
// A tiny synthetic GGUF (the models-crate range_tests builder, condensed)
// ---------------------------------------------------------------------------

fn string_into(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn random_blob(len: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(len);
    out
}

const TENSORS: [(&str, u64); 5] = [
    ("token_embd.weight", 0),
    ("blk.0.ffn_gate.weight", 1024),
    ("blk.1.ffn_gate.weight", 3072),
    ("blk.2.ffn_gate.weight", 3584),
    ("output.weight", 7680),
];
const DATA_LEN: usize = 7936;

/// A complete synthetic GGUF: parseable v3 header + deterministic data.
fn synthetic_model() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend(0x4655_4747_u32.to_le_bytes()); // "GGUF"
    buf.extend(3u32.to_le_bytes());
    buf.extend((TENSORS.len() as u64).to_le_bytes());
    buf.extend(2u64.to_le_bytes()); // kv count
                                    // general.architecture = "llama" (string type 8)
    string_into(&mut buf, "general.architecture");
    buf.extend(8u32.to_le_bytes());
    string_into(&mut buf, "llama");
    // llama.block_count = 3 (u32 type 4)
    string_into(&mut buf, "llama.block_count");
    buf.extend(4u32.to_le_bytes());
    buf.extend(3u32.to_le_bytes());
    for (name, offset) in TENSORS {
        string_into(&mut buf, name);
        buf.extend(2u32.to_le_bytes()); // n_dims
        for d in [16u64, 16u64] {
            buf.extend(d.to_le_bytes());
        }
        buf.extend(0u32.to_le_bytes()); // ggml type
        buf.extend(offset.to_le_bytes());
    }
    let header = GgufHeader::parse(&buf).expect("synthetic header must parse");
    buf.resize(header.data_offset as usize, 0);
    buf.extend(random_blob(DATA_LEN));
    buf
}

/// Seed one completed single-file cache entry (file + integrity manifest).
fn seed_full_entry(cache_root: &Path, id: &str, url: &str, bytes: &[u8]) -> std::path::PathBuf {
    let dir = cache_root.join(id);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.gguf");
    std::fs::write(&path, bytes).unwrap();
    let manifest = download::Manifest {
        url: url.to_string(),
        size_bytes: bytes.len() as u64,
        blake3: blake3::hash(bytes).to_hex().to_string(),
    };
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    path
}

// ---------------------------------------------------------------------------
// Zero-WAN pull (spec §6 DoD shape)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn pull_completes_with_zero_wan_bytes_when_a_peer_holds_the_model() {
    let a_dir = TempDir::new().unwrap();
    let b_dir = TempDir::new().unwrap();
    let (a, a_root) = spawn_node(&a_dir, "node-a").await;
    let (b, b_root) = spawn_node(&b_dir, "node-b").await;
    pair_and_connect(&a, &b).await;

    // Node A holds the complete model (as if pulled earlier) and has shared
    // it into its blob store — what `finish_completed_file` does post-pull.
    let blob = synthetic_model();
    let url = "https://wan.invalid/models/model.gguf";
    let a_path = seed_full_entry(&a_root, "test-model", url, &blob);
    a.share_blob(&a_path).await.expect("A shares its model");

    // Node B pulls the same spec. The WAN URL is unreachable, so the pull
    // can only succeed by fetching every byte from A — success IS the
    // zero-WAN proof.
    let spec = DownloadSpec {
        cache_key: "test-model".into(),
        url: url.into(),
        file_name: "model.gguf".into(),
    };
    let fetched = ensure_remote_local(&b, &b_root, &spec, |_, _| {})
        .await
        .expect("LAN-first pull succeeds without any WAN");
    assert_eq!(fetched.paths.len(), 1);
    assert_eq!(fetched.size_bytes, blob.len() as u64);
    assert_eq!(
        std::fs::read(&fetched.paths[0]).unwrap(),
        blob,
        "the pulled file must be byte-exact"
    );

    // B's manifest is byte-equal in every integrity field to A's (the sim's
    // "manifest byte-exact" assertion).
    let a_manifest = download::read_manifest(&a_root.join("test-model")).unwrap();
    let b_manifest = download::read_manifest(&b_root.join("test-model")).unwrap();
    assert_eq!(a_manifest, b_manifest);

    // And B can immediately serve the model onward: its inventory now
    // advertises the file, so a third node would fetch from B (spec §6).
    let inventory = a
        .range_query("node-b", "test-model")
        .await
        .expect("B answers range queries");
    assert_eq!(inventory.total_size, blob.len() as u64);
    assert!(
        inventory
            .ranges
            .contains(&(0, blob.len() as u64, *blake3::hash(&blob).as_bytes())),
        "B must advertise the full file: {:?}",
        inventory.ranges
    );

    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
}

// ---------------------------------------------------------------------------
// Worker layer-range fetch from a peer's range store
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn worker_fetches_only_its_layers_from_a_peer_and_replans_reuse_disk() {
    let a_dir = TempDir::new().unwrap();
    let b_dir = TempDir::new().unwrap();
    let (a, a_root) = spawn_node(&a_dir, "node-a").await;
    let (b, b_root) = spawn_node(&b_dir, "node-b").await;
    pair_and_connect(&a, &b).await;

    // Node A holds the model as RANGE FILES (a worker that range-fetched
    // earlier): every tensor-aligned range on disk, hashed in ranges.json,
    // each file shared as a blob.
    let blob = synthetic_model();
    let header = GgufHeader::parse(&blob).unwrap();
    let every_layer: BTreeSet<u64> = (0..3).collect();
    let full_plan = ranges::plan_ranges(&header, blob.len() as u64, &every_layer).unwrap();
    let a_entry = a_root.join("test-model");
    let a_ranges = a_entry.join(ranges::RANGES_DIR);
    std::fs::create_dir_all(&a_ranges).unwrap();
    let mut entries = Vec::new();
    for r in &full_plan.ranges {
        let bytes = &blob[r.start as usize..r.end as usize];
        let path = a_ranges.join(format!("{}-{}", r.start, r.end));
        std::fs::write(&path, bytes).unwrap();
        entries.push(ranges::RangeEntry {
            start: r.start,
            end: r.end,
            blake3: blake3::hash(bytes).to_hex().to_string(),
        });
        a.share_blob(&path).await.expect("A shares the range file");
    }
    let manifest = ranges::RangeManifest {
        url: String::new(), // A never knew a WAN URL either
        total_size: blob.len() as u64,
        header_len: full_plan.header_len,
        ranges: entries,
    };
    std::fs::write(
        ranges::ranges_manifest_path(&a_entry),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    // Node B adopts a plan assigning it layer 1 of "test-model" — an id
    // that resolves to NO download URL, so peers are the only source.
    let layers: BTreeSet<u64> = [1].into_iter().collect();
    let fetched = fetch_layer_ranges(&b, &b_root, "test-model", &layers)
        .await
        .expect("layer fetch succeeds")
        .expect("the model is range-fetchable");
    assert_eq!(fetched.total_size, blob.len() as u64);
    assert_eq!(fetched.outcome.wan_bytes, 0, "no WAN exists to spend");
    assert!(fetched.outcome.p2p_bytes > 0, "the ranges came from A");

    // B holds the header + shared tensors + layer 1, byte-exact…
    let b_entry = b_root.join("test-model");
    let owned_plan = ranges::plan_ranges(&header, blob.len() as u64, &layers).unwrap();
    for r in &owned_plan.ranges {
        let bytes = ranges::read_range(&b_entry, r.start, r.end).expect("owned range readable");
        assert_eq!(bytes, &blob[r.start as usize..r.end as usize]);
    }
    // …and NOT the unowned layers (fetch ONLY the assigned ranges).
    let unowned = full_plan
        .ranges
        .iter()
        .find(|r| !owned_plan.ranges.contains(r))
        .expect("some range is unowned");
    assert!(
        ranges::read_range(&b_entry, unowned.start, unowned.end).is_err(),
        "unowned layer ranges must not be fetched"
    );

    // A re-plan with the same layers reuses every byte on disk: nothing
    // moves on either path (docs/logistics.md: nothing re-downloads).
    let again = fetch_layer_ranges(&b, &b_root, "test-model", &layers)
        .await
        .expect("re-plan fetch succeeds")
        .expect("still range-fetchable");
    assert_eq!(again.outcome.p2p_bytes, 0, "re-plans reuse the range store");
    assert_eq!(again.outcome.wan_bytes, 0);

    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
}
