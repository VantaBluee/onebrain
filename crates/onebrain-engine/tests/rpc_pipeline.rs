//! M7 overlapped chunked prefill (docs/perf.md §3, patches/0003): the RPC
//! client backend now advertises async + events, so llama.cpp's own
//! pipeline-parallel gate engages for distributed loads and multi-ubatch
//! prompts submit graphs without waiting for per-ubatch acks. These tests
//! pin the three properties the patch must not trade away:
//!
//! 1. §9 correctness: a multi-ubatch (> n_batch tokens) distributed greedy
//!    prefill produces byte-identical tokens to a solo load — and the
//!    vendor's `pipeline parallelism enabled` line proves the pipelined
//!    path (not the old sequential one) produced them.
//! 2. Teardown with pending acks: closing a session right after a decode
//!    (submitted-but-unacked graph work on the wire) neither hangs nor
//!    aborts.
//! 3. The 0002 error regime survives pipelining: a socket torn while the
//!    ledger holds pending work surfaces as `EngineError::Decode` on the
//!    next decode, and freeing the model over the dead bridge stays clean.
//!
//! Note: the GGML RPC server printf's directly ("Serving RPC ...",
//! "Devices:"); that noise is expected in this test's output.

use std::io;
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use onebrain_engine::rpc::{
    pipeline_parallel_engagements, pump, Pump, RemoteServer, RpcServeSession, SocketPair,
};
use onebrain_engine::{
    devices, DeviceKind, EngineError, Model, ModelParams, Session, SessionParams,
};

fn cpu_device_index() -> i32 {
    devices()
        .iter()
        .position(|d| matches!(d.kind, DeviceKind::Cpu))
        .expect("a CPU device always exists") as i32
}

/// Session shape for every test here: a prompt longer than `n_batch` decodes
/// as multiple chunks, and each chunk splits into several `n_ubatch`
/// micro-batches — multiple pipelined graph submissions per `llama_decode`.
const N_BATCH: u32 = 128;
const N_UBATCH: u32 = 32;

fn pipeline_session_params() -> SessionParams {
    SessionParams {
        n_ctx: 1024,
        n_batch: N_BATCH,
        n_ubatch: N_UBATCH,
        ..SessionParams::default()
    }
}

/// A prompt guaranteed to span several batches (asserted in the tests).
fn long_prompt(model: &Model) -> Vec<i32> {
    let text = "Once upon a time there was a little girl named Lily. ".repeat(40);
    let prompt = model.tokenize(&text, true).expect("tokenize long prompt");
    assert!(
        prompt.len() > 2 * N_BATCH as usize,
        "prompt must exceed 2*n_batch tokens for a multi-ubatch prefill (got {})",
        prompt.len()
    );
    prompt
}

/// The rpc_loopback bridge (one serve session + byte pump per accepted
/// connection), with kill handles kept so the torn-socket test can tear
/// every connection down mid-flight.
struct Bridge {
    port: u16,
    stop: Arc<AtomicBool>,
    conns: Arc<Mutex<Vec<TcpStream>>>,
    acceptor: thread::JoinHandle<Vec<(RpcServeSession, Pump)>>,
}

impl Bridge {
    fn spawn(dev_index: i32) -> Bridge {
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
                            RpcServeSession::spawn(raw, 2, dev_index, /*cache_dir=*/ None)
                                .expect("spawn serve session");
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
        Bridge {
            port,
            stop,
            conns,
            acceptor,
        }
    }

    /// Hard-close every accepted connection (torn socket mid-pipeline).
    fn kill_all(&self) {
        for conn in self.conns.lock().expect("conns lock").iter() {
            let _ = conn.shutdown(Shutdown::Both);
        }
    }

    fn finish(self) -> Vec<(RpcServeSession, Pump)> {
        self.stop.store(true, Ordering::Relaxed);
        self.acceptor.join().expect("acceptor thread")
    }

    /// Join every serve session, asserting none hangs past `timeout`.
    fn finish_join(self, timeout: Duration, what: &str) {
        for (i, (serve, p)) in self.finish().into_iter().enumerate() {
            assert!(
                serve.join_timeout(timeout).is_ok(),
                "{what}: serve session {i} did not end within {timeout:?}"
            );
            p.join();
        }
    }
}

fn load_distributed(model_path: &Path, server: &RemoteServer) -> Model {
    Model::load_distributed(
        model_path,
        &[server],
        &[0.5, 0.5],
        /*use_local_device=*/ true,
        &ModelParams::default(),
    )
    .expect("distributed load through the loopback bridge")
}

/// §3 headline correctness proof: a prompt spanning many ubatches, decoded
/// through the pipelined distributed path, must yield byte-identical greedy
/// tokens to a solo load — and llama.cpp itself must report that pipeline
/// parallelism engaged (the caps flip passed its gate), so the equality
/// was proven against the overlapped path rather than a silent fallback.
#[test]
fn pipelined_multi_ubatch_prefill_matches_solo() {
    let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
        eprintln!("OB_SMOKE_MODEL not set; skipping rpc pipeline test");
        return;
    };
    let model_path = Path::new(&model_path);
    let dev_index = cpu_device_index();

    // Solo ground truth with the same session shape (dropped before the
    // distributed run so the paths share nothing beyond the backend).
    let (solo_tokens, prompt_len) = {
        let solo = Model::load(model_path, &ModelParams::default()).expect("solo load");
        let prompt = long_prompt(&solo);
        let mut session = Session::new(&solo, &pipeline_session_params()).expect("solo session");
        let tokens = session
            .generate_greedy(&prompt, 8, |_, _| {})
            .expect("solo greedy generation");
        (tokens, prompt.len())
    };
    assert!(!solo_tokens.is_empty(), "solo baseline generated no tokens");

    let bridge = Bridge::spawn(dev_index);
    let endpoint = format!("127.0.0.1:{}", bridge.port);
    let server = RemoteServer::register(&endpoint).expect("register bridged rpc server");

    let model = load_distributed(model_path, &server);
    let prompt = long_prompt(&model);
    assert_eq!(
        prompt.len(),
        prompt_len,
        "tokenization must not vary by path"
    );

    // Creating the distributed context is what logs the engagement line;
    // other tests in this binary may also engage in parallel, so assert on
    // an increase, not an exact count.
    let engaged_before = pipeline_parallel_engagements();
    let dist_tokens = {
        let mut session =
            Session::new(&model, &pipeline_session_params()).expect("distributed session");
        session
            .generate_greedy(&prompt, 8, |_, _| {})
            .expect("distributed greedy generation")
    };
    let engaged_after = pipeline_parallel_engagements();

    assert!(
        engaged_after > engaged_before,
        "llama.cpp must log 'pipeline parallelism enabled' for this load \
         (caps async/events + the sched gate; docs/perf.md §3)"
    );
    assert_eq!(
        dist_tokens, solo_tokens,
        "pipelined distributed greedy tokens must be byte-identical to the solo run (§9)"
    );

    drop(model);
    bridge.finish_join(Duration::from_secs(5), "pipeline roundtrip");
}

/// Teardown with pending acks: decode a multi-ubatch prompt and close the
/// session/model immediately, without ever sampling — the pending ledger
/// still holds submitted-but-unacked work when the frees start. The frees
/// are response-bearing, so they drain naturally against the live server;
/// the assert is simply "everything ends, promptly, without a hang".
#[test]
fn teardown_with_pending_acks_does_not_hang() {
    let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
        eprintln!("OB_SMOKE_MODEL not set; skipping rpc pipeline teardown test");
        return;
    };
    let model_path = Path::new(&model_path);
    let dev_index = cpu_device_index();

    let bridge = Bridge::spawn(dev_index);
    let endpoint = format!("127.0.0.1:{}", bridge.port);
    let server = RemoteServer::register(&endpoint).expect("register bridged rpc server");

    let model = load_distributed(model_path, &server);
    let prompt = long_prompt(&model);
    {
        let mut session =
            Session::new(&model, &pipeline_session_params()).expect("distributed session");
        session.decode(&prompt).expect("multi-ubatch decode");
        // Session drops here with pipelined submissions still unfetched.
    }
    drop(model);

    bridge.finish_join(
        Duration::from_secs(10),
        "teardown with pending acks (no hang at session close)",
    );
}

/// 0002 composition: tear the bridge while the pending ledger holds work
/// from a pipelined decode. The next decode must fail with the typed
/// `EngineError::Decode` (a pending ack that fails surfaces as an error on
/// the call that drains it — never an abort), and freeing session + model
/// over the dead bridge must stay silent and prompt.
#[test]
fn torn_socket_mid_pipeline_is_an_error_with_clean_frees() {
    let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
        eprintln!("OB_SMOKE_MODEL not set; skipping rpc pipeline tear test");
        return;
    };
    let model_path = Path::new(&model_path);
    let dev_index = cpu_device_index();

    let bridge = Bridge::spawn(dev_index);
    let endpoint = format!("127.0.0.1:{}", bridge.port);
    let server = RemoteServer::register(&endpoint).expect("register bridged rpc server");

    let model = load_distributed(model_path, &server);
    let prompt = long_prompt(&model);
    let result = {
        let mut session =
            Session::new(&model, &pipeline_session_params()).expect("distributed session");
        // A healthy pipelined decode first, so the tear lands on a session
        // whose sockets carried pipelined traffic (and may still hold
        // unacked submissions — decode never fetched logits).
        session.decode(&prompt).expect("healthy pipelined decode");
        bridge.kill_all();
        // Deterministic: with every bridge connection dead, the next decode
        // must fail fast — dead sockets short-circuit before blocking I/O.
        // (One n_batch chunk keeps the total well inside n_ctx, so the only
        // possible failure is the torn transport.)
        session.decode(&prompt[..N_BATCH as usize])
        // Session drop runs its frees against the dead bridge here.
    };
    match result {
        Err(EngineError::Decode { .. }) => {}
        other => panic!(
            "expected Err(EngineError::Decode {{..}}) after tearing the bridge \
             mid-pipeline, got {other:?}"
        ),
    }

    // Freeing the model sends buffer frees over the dead bridge; the 0002
    // regime (kept by 0003) tolerates them silently — no abort, no hang.
    drop(model);

    bridge.finish_join(
        Duration::from_secs(10),
        "torn socket mid-pipeline (serve sessions must observe the tear)",
    );
}
