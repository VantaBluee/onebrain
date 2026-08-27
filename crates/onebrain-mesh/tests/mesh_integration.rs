//! Integration tests: two `MeshService`s in one process over real iroh
//! endpoints bound to loopback. Hermetic: mDNS and relays are disabled, so
//! all connectivity flows through the direct addresses carried in tickets.

use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

use iroh::endpoint::{presets, ConnectionError, VarInt};
use iroh::Endpoint;
use iroh_tickets::endpoint::EndpointTicket;
use onebrain_mesh::{
    identity, MeshConfig, MeshError, MeshHandle, MeshService, PairEvent, PairTarget, PeerState,
    PeerStatus, ALPN_MESH,
};
use tempfile::TempDir;
use tokio::time::timeout;

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

/// Pick a currently-free loopback UDP port by binding then dropping a std
/// socket. Used so a "restarted" service can rebind the SAME port, keeping
/// the addresses persisted in its peer's store valid.
fn reserve_udp_port() -> u16 {
    let socket = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind a probe socket");
    let port = socket
        .local_addr()
        .expect("probe socket has an addr")
        .port();
    drop(socket);
    port
}

fn test_config(pair_window: Duration) -> MeshConfig {
    MeshConfig {
        enable_mdns: false,
        enable_relays: false,
        engine_build: "test-build".to_string(),
        pair_window,
        bind_addrs: vec![loopback()],
        node_status: None,
    }
}

async fn spawn_node(dir: &TempDir, name: &str, pair_window: Duration) -> MeshHandle {
    let key = identity::load_or_create(dir.path()).expect("device key");
    MeshService::spawn(
        key,
        dir.path().join("peers.toml"),
        name.to_string(),
        test_config(pair_window),
    )
    .await
    .expect("mesh service spawns")
}

/// Like [`spawn_node`], but pinned to a fixed loopback port so the node
/// keeps the same dialable address across "restarts" (re-spawns from the
/// same directory: same device key, same peer store).
async fn spawn_node_at(dir: &TempDir, name: &str, port: u16) -> MeshHandle {
    let key = identity::load_or_create(dir.path()).expect("device key");
    let mut config = test_config(Duration::from_secs(120));
    config.bind_addrs = vec![SocketAddr::from((Ipv4Addr::LOCALHOST, port))];
    MeshService::spawn(key, dir.path().join("peers.toml"), name.to_string(), config)
        .await
        .expect("mesh service spawns")
}

/// Poll `peers()` until the single peer satisfies `pred`, panicking with the
/// last snapshot on timeout.
async fn wait_for_peer(
    handle: &MeshHandle,
    what: &str,
    limit: Duration,
    pred: impl Fn(&PeerStatus) -> bool,
) {
    let deadline = tokio::time::Instant::now() + limit;
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

#[tokio::test(flavor = "multi_thread")]
async fn pair_via_ticket_heartbeats_and_bandwidth() {
    let host_dir = TempDir::new().unwrap();
    let joiner_dir = TempDir::new().unwrap();
    let host = spawn_node(&host_dir, "host-node", Duration::from_secs(120)).await;
    let joiner = spawn_node(&joiner_dir, "joiner-node", Duration::from_secs(120)).await;

    let mut window = host.pair_start().await.expect("window opens");
    assert_eq!(window.code.len(), 6);
    assert!(window.code.bytes().all(|b| b.is_ascii_digit()));
    assert!(!window.ticket.is_empty());

    // Happy path: join via the ticket with the right code.
    let info = joiner
        .pair_join(
            PairTarget::Ticket(window.ticket.clone()),
            Some(window.code.clone()),
        )
        .await
        .expect("pairing succeeds");
    assert_eq!(info.name, "host-node");

    // The host streams Attempt then Paired.
    let paired = timeout(Duration::from_secs(15), async {
        loop {
            match window.events.recv().await {
                Some(PairEvent::Paired(peer)) => break peer,
                Some(PairEvent::Attempt) => continue,
                other => panic!("unexpected pairing event: {other:?}"),
            }
        }
    })
    .await
    .expect("host observes the pairing");
    assert_eq!(paired.name, "joiner-node");

    // Both stores persisted the peer with the exchanged names.
    let host_peers = host.peers().await.unwrap();
    assert_eq!(host_peers.len(), 1, "host store: {host_peers:?}");
    assert_eq!(host_peers[0].name, "joiner-node");
    let joiner_peers = joiner.peers().await.unwrap();
    assert_eq!(joiner_peers.len(), 1, "joiner store: {joiner_peers:?}");
    assert_eq!(joiner_peers[0].name, "host-node");
    assert_eq!(joiner_peers[0].id, host.endpoint_id().to_string());

    // Heartbeats drive both sides to `connected` with a measured RTT.
    wait_for_peer(
        &joiner,
        "joiner->host connected",
        Duration::from_secs(30),
        |p| p.state == PeerState::Connected && p.rtt_ms.is_some(),
    )
    .await;
    wait_for_peer(
        &host,
        "host->joiner connected",
        Duration::from_secs(30),
        |p| p.state == PeerState::Connected && p.rtt_ms.is_some(),
    )
    .await;

    // Bandwidth: the on-connect probe populates it, and `probe()` re-runs it.
    let mbps = joiner.probe("host-node").await.expect("probe runs");
    assert!(mbps > 0.0, "probe reported {mbps} Mbps");
    wait_for_peer(&host, "host-side bandwidth", Duration::from_secs(30), |p| {
        p.bandwidth_mbps.unwrap_or(0.0) > 0.0
    })
    .await;

    host.shutdown().await.unwrap();
    joiner.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn wrong_code_burns_an_attempt_and_leaves_stores_empty() {
    let host_dir = TempDir::new().unwrap();
    let joiner_dir = TempDir::new().unwrap();
    let host = spawn_node(&host_dir, "host-node", Duration::from_secs(120)).await;
    let joiner = spawn_node(&joiner_dir, "joiner-node", Duration::from_secs(120)).await;

    let mut window = host.pair_start().await.expect("window opens");

    // Flip the first digit so the code is definitely wrong.
    let mut wrong: Vec<u8> = window.code.clone().into_bytes();
    wrong[0] = b'0' + ((wrong[0] - b'0' + 1) % 10);
    let wrong = String::from_utf8(wrong).unwrap();

    let err = joiner
        .pair_join(PairTarget::Ticket(window.ticket.clone()), Some(wrong))
        .await
        .expect_err("wrong code must fail");
    assert!(
        matches!(err, MeshError::PairRejected { .. }),
        "unexpected error: {err}"
    );

    // The failed attempt was counted (host emitted Attempt, window stays open).
    let event = timeout(Duration::from_secs(10), window.events.recv())
        .await
        .expect("host emits an event");
    assert!(matches!(event, Some(PairEvent::Attempt)), "{event:?}");

    // No state change on either side.
    assert!(host.peers().await.unwrap().is_empty());
    assert!(joiner.peers().await.unwrap().is_empty());

    // The window survives a burned attempt: the correct code still pairs.
    let info = joiner
        .pair_join(
            PairTarget::Ticket(window.ticket.clone()),
            Some(window.code.clone()),
        )
        .await
        .expect("correct code pairs after a burned attempt");
    assert_eq!(info.name, "host-node");
    assert_eq!(host.peers().await.unwrap().len(), 1);

    host.shutdown().await.unwrap();
    joiner.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn pairing_window_expires() {
    let host_dir = TempDir::new().unwrap();
    let joiner_dir = TempDir::new().unwrap();
    // Test-only knob: shrink the 120 s window so the test does not sleep.
    let host = spawn_node(&host_dir, "host-node", Duration::from_millis(500)).await;
    let joiner = spawn_node(&joiner_dir, "joiner-node", Duration::from_secs(120)).await;

    let mut window = host.pair_start().await.expect("window opens");
    let event = timeout(Duration::from_secs(10), window.events.recv())
        .await
        .expect("expiry event arrives");
    assert!(matches!(event, Some(PairEvent::Expired)), "{event:?}");

    // Joining after expiry fails and stores stay empty.
    let result = joiner
        .pair_join(
            PairTarget::Ticket(window.ticket.clone()),
            Some(window.code.clone()),
        )
        .await;
    assert!(result.is_err(), "join after expiry must fail: {result:?}");
    assert!(host.peers().await.unwrap().is_empty());
    assert!(joiner.peers().await.unwrap().is_empty());

    host.shutdown().await.unwrap();
    joiner.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn unpaired_mesh_connect_is_rejected() {
    let host_dir = TempDir::new().unwrap();
    let host = spawn_node(&host_dir, "host-node", Duration::from_secs(120)).await;

    // Grab the host's dialable address from a pairing ticket.
    let window = host.pair_start().await.expect("window opens");
    let ticket: EndpointTicket = window.ticket.parse().expect("ticket parses");
    let addr = ticket.endpoint_addr().clone();

    // A stranger endpoint (never paired) dials the mesh ALPN directly.
    let stranger = Endpoint::builder(presets::Minimal)
        .alpns(vec![ALPN_MESH.to_vec()])
        .clear_ip_transports()
        .bind_addr(loopback())
        .expect("valid bind addr")
        .bind()
        .await
        .expect("stranger endpoint binds");

    match stranger.connect(addr, ALPN_MESH).await {
        Ok(conn) => {
            // QUIC handshake completed; the host must close with code 1.
            let reason = conn.closed().await;
            match reason {
                ConnectionError::ApplicationClosed(close) => {
                    assert_eq!(close.error_code, VarInt::from(1u32), "close: {close}");
                    assert_eq!(&close.reason[..], b"unpaired");
                }
                other => panic!("expected an application close, got: {other:?}"),
            }
        }
        Err(err) => {
            // The close can also race the connect completing.
            let text = format!("{err:?}");
            assert!(
                text.contains("unpaired") || text.contains("ApplicationClosed"),
                "unexpected connect error: {text}"
            );
        }
    }

    // The host's store is untouched.
    assert!(host.peers().await.unwrap().is_empty());

    stranger.close().await;
    host.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn restart_reconnects_without_repairing() {
    let a_dir = TempDir::new().unwrap();
    let b_dir = TempDir::new().unwrap();
    // Pinned ports: with mDNS and relays off, the ONLY way a restarted
    // service can be found is through the addresses its peer persisted, so
    // those addresses must stay valid across the restart.
    let port_a = reserve_udp_port();
    let port_b = reserve_udp_port();
    let a = spawn_node_at(&a_dir, "node-a", port_a).await;
    let b = spawn_node_at(&b_dir, "node-b", port_b).await;

    let window = a.pair_start().await.expect("window opens");
    b.pair_join(
        PairTarget::Ticket(window.ticket.clone()),
        Some(window.code.clone()),
    )
    .await
    .expect("pairing succeeds");
    wait_for_peer(
        &a,
        "b connected after pairing",
        Duration::from_secs(20),
        |p| p.state == PeerState::Connected,
    )
    .await;
    wait_for_peer(
        &b,
        "a connected after pairing",
        Duration::from_secs(20),
        |p| p.state == PeerState::Connected,
    )
    .await;

    // Full restart of BOTH sides: shut down, drop, respawn from the same
    // dirs (same device keys, same peer stores) on the same ports. No pair
    // call afterwards - the reconnect loop must re-form the link from the
    // persisted addressing alone (the M2 DoD "survive daemon restarts").
    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
    drop(a);
    drop(b);
    // Give the OS a beat to release the UDP ports (Windows lags here).
    tokio::time::sleep(Duration::from_millis(500)).await;

    let a = spawn_node_at(&a_dir, "node-a", port_a).await;
    let b = spawn_node_at(&b_dir, "node-b", port_b).await;

    wait_for_peer(
        &a,
        "b reconnected after restart",
        Duration::from_secs(30),
        |p| p.state == PeerState::Connected,
    )
    .await;
    wait_for_peer(
        &b,
        "a reconnected after restart",
        Duration::from_secs(30),
        |p| p.state == PeerState::Connected,
    )
    .await;

    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
}
