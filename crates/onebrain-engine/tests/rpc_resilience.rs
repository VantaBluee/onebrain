//! M5 resilience enabler (docs/resilience.md, patches/0002): client-side RPC
//! transport failures must surface as error returns, never process aborts.
//! A model is loaded distributed over a bridged loopback server (the
//! rpc_loopback pattern), the bridge is torn down mid-generation, and the
//! generation must fail with `EngineError::Decode` — after which dropping the
//! model (whose frees go over the now-dead bridge) must also not abort.
//!
//! Note: the GGML RPC server printf's directly and the patched client logs a
//! transport-failure line; that noise is expected in this test's output.

use std::io;
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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

/// The rpc_loopback bridge, extended with kill handles: every accepted
/// connection's stream is cloned before it goes into the byte pump, so the
/// test can hard-close all of them mid-generation — the in-process miniature
/// of a worker dying under an active distributed epoch.
struct KillableBridge {
    port: u16,
    stop: Arc<AtomicBool>,
    conns: Arc<Mutex<Vec<TcpStream>>>,
    acceptor: thread::JoinHandle<Vec<(RpcServeSession, Pump)>>,
}

impl KillableBridge {
    fn spawn(dev_index: i32) -> KillableBridge {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback listener");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let port = listener.local_addr().expect("listener addr").port();
        let stop = Arc::new(AtomicBool::new(false));
        let conns: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
        let stop_flag = Arc::clone(&stop);
        let conns_out = Arc::clone(&conns);
        let acceptor = thread::spawn(move || {
            let mut sessions = Vec::new();
            while !stop_flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((conn, _)) => {
                        conn.set_nonblocking(false).expect("blocking conn");
                        conn.set_nodelay(true).ok();
                        conns_out
                            .lock()
                            .expect("conns lock")
                            .push(conn.try_clone().expect("clone kill handle"));
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
        KillableBridge {
            port,
            stop,
            conns,
            acceptor,
        }
    }

    /// Hard-close every accepted connection: the pump directions error out,
    /// the serve sessions see EOF, and the RPC client's next round trips fail.
    fn kill_handles(&self) -> Arc<Mutex<Vec<TcpStream>>> {
        Arc::clone(&self.conns)
    }

    fn finish(self) -> Vec<(RpcServeSession, Pump)> {
        self.stop.store(true, Ordering::Relaxed);
        self.acceptor.join().expect("acceptor thread")
    }
}

fn kill_all(conns: &Mutex<Vec<TcpStream>>) {
    for conn in conns.lock().expect("conns lock").iter() {
        let _ = conn.shutdown(Shutdown::Both);
    }
}

/// Tear the bridge down after two streamed tokens: the generation must end
/// in `EngineError::Decode` (not a process abort), and freeing the session
/// and model over the dead bridge must not abort either.
#[test]
fn bridge_death_mid_generation_is_an_error_not_an_abort() {
    let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
        eprintln!("OB_SMOKE_MODEL not set; skipping rpc resilience test");
        return;
    };
    let model_path = Path::new(&model_path);
    let dev_index = cpu_device_index();

    let bridge = KillableBridge::spawn(dev_index);
    let endpoint = format!("127.0.0.1:{}", bridge.port);

    let server = RemoteServer::register(&endpoint).expect("register bridged rpc server");
    assert!(server.device_count() >= 1);

    let model = Model::load_distributed(
        model_path,
        &[&server],
        &[0.5, 0.5],
        /*use_local_device=*/ true,
        &ModelParams::default(),
    )
    .expect("distributed load through the loopback bridge");

    let prompt = model.tokenize("Once upon a time", true).expect("tokenize");
    let result = {
        let mut session = Session::new(
            &model,
            &SessionParams {
                n_ctx: 256,
                n_batch: 64,
                n_threads: 0,
            },
        )
        .expect("session");

        let kill = bridge.kill_handles();
        let mut streamed = 0usize;
        // Ask for far more tokens than the healthy path would need so the
        // only way out (besides EOG on a tiny model, which the loopback test
        // proves does not happen this early) is the injected failure.
        session.generate_greedy(&prompt, 64, |_tok, _piece| {
            streamed += 1;
            if streamed == 2 {
                // >= 2 tokens streamed: kill the bridge mid-generation.
                kill_all(&kill);
            }
        })
    };

    match result {
        Err(EngineError::Decode { .. }) => {}
        other => {
            panic!("expected Err(EngineError::Decode {{..}}) after bridge teardown, got {other:?}")
        }
    }

    // Freeing the model sends buffer frees over the dead bridge; the patched
    // client must tolerate them silently (no abort, no hang).
    drop(model);

    // The serve sessions saw their sockets die; all must have ended.
    for (i, (serve, p)) in bridge.finish().into_iter().enumerate() {
        assert!(
            serve.join_timeout(Duration::from_secs(10)).is_ok(),
            "serve session {i} did not end within 10s of the bridge being killed"
        );
        p.join();
    }
}
