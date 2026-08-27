//! M6 P2P blob-sharing integration tests: two loopback `MeshService`s where
//! one provides a range file over the blobs ALPN and its paired peer fetches
//! it byte-exact, while an UNPAIRED endpoint is rejected on that same ALPN
//! (the §10 rule). Plus the `RangeQuery` → `RangeInventory` control exchange
//! against a stub inventory source, and the hash identity the whole design
//! rests on: iroh-blobs addresses a blob by the plain BLAKE3 of its content,
//! i.e. exactly the hash our range manifests store.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use iroh::endpoint::{presets, ConnectionError, VarInt};
use iroh::Endpoint;
use iroh_tickets::endpoint::EndpointTicket;
use onebrain_mesh::{
    identity, MeshConfig, MeshHandle, MeshService, PairTarget, PeerRangeInventory, PeerState,
    PeerStatus, RangeInventorySource, ALPN_BLOBS,
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

/// Deterministic non-trivial content standing in for one tensor-aligned
/// range file (256 KiB — large enough for several bao chunk groups).
fn range_file_bytes() -> Vec<u8> {
    (0..256 * 1024u32)
        .map(|i| (i.wrapping_mul(31).wrapping_add(i >> 8)) as u8)
        .collect()
}

/// The identity the M6 contract rests on (docs/logistics.md): the iroh-blobs
/// address of a raw blob IS the plain BLAKE3 of its content, so range
/// manifest hashes double as blob addresses with no re-hashing. (Large blobs
/// gain a bao outboard for verified streaming — transfer encoding only; the
/// root hash stays blake3(content).)
#[test]
fn iroh_blobs_hash_is_plain_blake3() {
    for data in [b"".as_slice(), b"onebrain".as_slice(), &range_file_bytes()] {
        assert_eq!(
            iroh_blobs::Hash::new(data).as_bytes(),
            blake3::hash(data).as_bytes(),
            "iroh-blobs and the blake3 crate must agree on {} bytes",
            data.len()
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn paired_peer_fetches_blob_byte_exact_and_unpaired_is_rejected() {
    let a_dir = TempDir::new().unwrap();
    let b_dir = TempDir::new().unwrap();
    // A runs the on-disk store (what the daemon uses: shared files are
    // referenced in place); B runs the in-memory default — both code paths.
    let mut a_config = test_config();
    a_config.blobs_dir = Some(a_dir.path().join("blobs"));
    let a = spawn_node(&a_dir, "node-a", a_config).await;
    let b = spawn_node(&b_dir, "node-b", test_config()).await;
    pair_and_connect(&a, &b).await;

    // A shares one range file; the returned blob hash must equal what the
    // range manifest would store (blake3 crate over the same bytes) — the
    // hash-identity proof over the REAL provider path.
    let data = range_file_bytes();
    let range_path = a_dir.path().join("0-262144");
    std::fs::write(&range_path, &data).unwrap();
    let hash = a
        .share_blob(&range_path)
        .await
        .expect("share_blob succeeds");
    assert_eq!(
        &hash,
        blake3::hash(&data).as_bytes(),
        "blob address must be the plain BLAKE3 of the range file"
    );

    // B (paired) fetches it by (peer, hash) into a target file, byte-exact.
    let target = b_dir.path().join("fetched-0-262144");
    let fetched = b
        .fetch_blob("node-a", hash, &target)
        .await
        .expect("paired fetch succeeds");
    assert!(
        fetched >= data.len() as u64,
        "a full transfer reads at least the payload ({fetched} < {})",
        data.len()
    );
    assert_eq!(
        std::fs::read(&target).unwrap(),
        data,
        "fetched range must be byte-exact"
    );

    // A second fetch is served from B's local blob store: zero network
    // bytes, file still written — the property that lets ranges spread
    // through a cluster without re-transfers (spec §6).
    let target2 = b_dir.path().join("fetched-again");
    let refetched = b
        .fetch_blob("node-a", hash, &target2)
        .await
        .expect("re-fetch succeeds");
    assert_eq!(refetched, 0, "complete local blob must not re-transfer");
    assert_eq!(std::fs::read(&target2).unwrap(), data);

    // An UNPAIRED endpoint on the blobs ALPN is closed with code 1
    // (`unpaired`) before any blob byte moves — same §10 rule as the mesh
    // ALPN. The pairing ticket is only used as a dialable address here.
    let window = a.pair_start().await.expect("window opens");
    let ticket: EndpointTicket = window.ticket.parse().expect("ticket parses");
    let addr = ticket.endpoint_addr().clone();
    let stranger = Endpoint::builder(presets::Minimal)
        .clear_ip_transports()
        .bind_addr(loopback())
        .expect("valid bind addr")
        .bind()
        .await
        .expect("stranger endpoint binds");
    match stranger.connect(addr, ALPN_BLOBS).await {
        Ok(conn) => match conn.closed().await {
            ConnectionError::ApplicationClosed(close) => {
                assert_eq!(close.error_code, VarInt::from(1u32), "close: {close}");
                assert_eq!(&close.reason[..], b"unpaired");
            }
            other => panic!("expected an application close, got: {other:?}"),
        },
        Err(err) => {
            // The close can race the connect completing.
            let text = format!("{err:?}");
            assert!(
                text.contains("unpaired") || text.contains("ApplicationClosed"),
                "unexpected connect error: {text}"
            );
        }
    }

    stranger.close().await;
    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
}

/// Stub inventory: knows one model, nothing else.
struct StubSource {
    total: u64,
    ranges: Vec<(u64, u64, [u8; 32])>,
}

impl RangeInventorySource for StubSource {
    fn inventory(&self, model: &str) -> Option<(u64, Vec<(u64, u64, [u8; 32])>)> {
        (model == "blake3:stories").then(|| (self.total, self.ranges.clone()))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn range_query_roundtrips_over_a_live_pair() {
    let a_dir = TempDir::new().unwrap();
    let b_dir = TempDir::new().unwrap();
    let h1: [u8; 32] = core::array::from_fn(|i| i as u8);
    let h2: [u8; 32] = core::array::from_fn(|i| 128 + i as u8);
    let mut a_config = test_config();
    a_config.range_source = Some(Arc::new(StubSource {
        total: 1_000_000,
        ranges: vec![(0, 4096, h1), (4096, 1_000_000, h2)],
    }));
    let a = spawn_node(&a_dir, "node-a", a_config).await;
    let b = spawn_node(&b_dir, "node-b", test_config()).await;
    pair_and_connect(&a, &b).await;

    // Known model: the stub's inventory arrives intact (by store name).
    let inventory = b
        .range_query("node-a", "blake3:stories")
        .await
        .expect("range query answers");
    assert_eq!(
        inventory,
        PeerRangeInventory {
            total_size: 1_000_000,
            ranges: vec![(0, 4096, h1), (4096, 1_000_000, h2)],
        }
    );

    // Unknown model: the "peer has none" empty reply (by endpoint id, which
    // exercises the id-based peer resolution too).
    let a_id = a.endpoint_id().to_string();
    let empty = b
        .range_query(&a_id, "blake3:unknown")
        .await
        .expect("unknown model still answers");
    assert_eq!(empty.total_size, 0);
    assert!(empty.ranges.is_empty(), "unknown model must reply empty");

    // No source configured (B): the default no-op replies empty as well.
    let none = a
        .range_query("node-b", "blake3:stories")
        .await
        .expect("sourceless node still answers");
    assert_eq!(none.total_size, 0);
    assert!(none.ranges.is_empty(), "missing source must reply empty");

    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
}
