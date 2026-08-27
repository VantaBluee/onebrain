//! In-process de-risk of the M3 distributed path (docs/distributed.md):
//! a GGML RPC session served over a process-local socket pair, bridged by
//! a byte pump to a loopback endpoint the client dials — the miniature of
//! what the daemon does over authenticated mesh streams. The critical
//! property proven here is §9 correctness: greedy tokens through the
//! distributed path are byte-identical to a solo load of the same model.
//!
//! Note: the GGML RPC server printf's directly ("Serving RPC ...",
//! "Devices:"); that noise is expected in this test's output.

use std::io;
use std::net::TcpListener;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use onebrain_engine::rpc::{pump, Pump, RemoteServer, RpcServeSession, SocketPair};
use onebrain_engine::{
    devices, DeviceKind, EngineError, Model, ModelParams, Session, SessionParams,
};

fn cpu_device_index() -> i32 {
    devices()
        .iter()
        .position(|d| matches!(d.kind, DeviceKind::Cpu))
        .expect("a CPU device always exists") as i32
}

/// The head-side half of the bridge: a loopback listener whose every
/// accepted connection gets its own serve session (socket pair + dedicated
/// serve thread) and byte pump. The GGML RPC client opens more than one
/// sequential connection per endpoint (a device-count probe at
/// registration, then the load/compute connection), so the bridge accepts
/// in a loop — one serve session per accepted connection, exactly like the
/// daemon maps one mesh `rpc` stream per connection.
struct LoopbackBridge {
    port: u16,
    stop: Arc<AtomicBool>,
    acceptor: thread::JoinHandle<Vec<(RpcServeSession, Pump)>>,
}

impl LoopbackBridge {
    fn spawn(dev_index: i32) -> LoopbackBridge {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback listener");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let port = listener.local_addr().expect("listener addr").port();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        let acceptor = thread::spawn(move || {
            let mut sessions = Vec::new();
            while !stop_flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((conn, _)) => {
                        // Accepted sockets can inherit non-blocking mode.
                        conn.set_nonblocking(false).expect("blocking conn");
                        conn.set_nodelay(true).ok();
                        let (raw, bridge) = SocketPair::new().expect("socket pair").into_parts();
                        let serve =
                            RpcServeSession::spawn(raw, 2, dev_index).expect("spawn serve session");
                        let p = pump(conn, bridge).expect("spawn pump");
                        sessions.push((serve, p));
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
            sessions
        });
        LoopbackBridge {
            port,
            stop,
            acceptor,
        }
    }

    fn finish(self) -> Vec<(RpcServeSession, Pump)> {
        self.stop.store(true, Ordering::Relaxed);
        self.acceptor.join().expect("acceptor thread")
    }
}

fn greedy_tokens(model: &Model, max_new: usize) -> Vec<i32> {
    let prompt = model.tokenize("Once upon a time", true).expect("tokenize");
    assert!(!prompt.is_empty());
    let mut session = Session::new(
        model,
        &SessionParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 0,
        },
    )
    .expect("session");
    session
        .generate_greedy(&prompt, max_new, |_, _| {})
        .expect("greedy generation")
}

/// The de-risk test: register a bridged loopback RPC server, load the tiny
/// model split 50/50 across the RPC device and the local CPU device,
/// generate greedily, and demand byte-identical tokens versus a solo load.
/// Then prove teardown: dropping the client model ends every serve session
/// within 5 seconds.
#[test]
fn rpc_loopback_roundtrip() {
    let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
        eprintln!("OB_SMOKE_MODEL not set; skipping rpc loopback test");
        return;
    };
    let model_path = Path::new(&model_path);
    let dev_index = cpu_device_index();

    // Solo ground truth (drop it before the distributed run so the two
    // paths never share engine state beyond the process-wide backend).
    let solo_tokens = {
        let solo = Model::load(model_path, &ModelParams::default()).expect("solo load");
        greedy_tokens(&solo, 8)
    };
    assert!(!solo_tokens.is_empty(), "solo baseline generated no tokens");

    let bridge = LoopbackBridge::spawn(dev_index);
    let endpoint = format!("127.0.0.1:{}", bridge.port);

    let server = RemoteServer::register(&endpoint).expect("register bridged rpc server");
    assert!(
        server.device_count() >= 1,
        "bridged server must expose the CPU device"
    );

    let model = Model::load_distributed(
        model_path,
        &[&server],
        &[0.5, 0.5],
        /*use_local_device=*/ true,
        &ModelParams::default(),
    )
    .expect("distributed load through the loopback bridge");
    assert!(model.n_layer() > 0);

    let dist_tokens = greedy_tokens(&model, 8);
    assert_eq!(
        dist_tokens, solo_tokens,
        "distributed greedy tokens must be byte-identical to the solo run (§9 correctness)"
    );

    // Teardown: dropping the model closes the client's RPC connections;
    // every serve session must end and its thread join within 5s.
    drop(model);
    for (i, (serve, p)) in bridge.finish().into_iter().enumerate() {
        assert!(
            serve.join_timeout(Duration::from_secs(5)).is_ok(),
            "serve session {i} did not end within 5s of the model being dropped"
        );
        p.join();
    }
}

/// Registering against a port nothing listens on must fail with the typed
/// connect error, not a hang or an abort.
#[test]
fn register_dead_endpoint_fails() {
    onebrain_engine::init();
    // Grab an ephemeral port and immediately free it: connecting to it now
    // gets a fast loopback refusal.
    let port = {
        let l = TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        l.local_addr().expect("addr").port()
    };
    let endpoint = format!("127.0.0.1:{port}");
    match RemoteServer::register(&endpoint) {
        Err(EngineError::RpcConnect { endpoint: e }) => assert_eq!(e, endpoint),
        other => panic!("expected RpcConnect error, got {other:?}"),
    }
}
