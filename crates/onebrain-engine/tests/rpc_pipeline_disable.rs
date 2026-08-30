//! The `[perf] prefill_overlap = false` engine hook (docs/perf.md §3,
//! patches/0003): [`set_pipeline_overlap`]`(false)` must stop the RPC
//! devices from advertising async + events, so llama.cpp's own
//! pipeline-parallel gate fails and the distributed load runs the exact M3
//! sequential path — the constructed baseline the benches and the sim
//! compare against. Proven here by the absence of the vendor's
//! `pipeline parallelism enabled` line (counter flat) while greedy output
//! stays byte-identical to a solo run.
//!
//! Lives in its own test binary on purpose: the switch is process-wide, and
//! sharing a process with rpc_pipeline.rs would race its enabled-path
//! engagement assertion.
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

use onebrain_engine::rpc::{
    pipeline_parallel_engagements, pump, set_pipeline_overlap, Pump, RemoteServer, RpcServeSession,
    SocketPair,
};
use onebrain_engine::{devices, DeviceKind, Model, ModelParams, Session, SessionParams};

/// Same session shape as rpc_pipeline.rs: multiple ubatches per decode, so
/// the sequential path is exercised on exactly the load shape the pipelined
/// path is benched against.
const N_BATCH: u32 = 128;
const N_UBATCH: u32 = 32;

#[test]
fn disabled_overlap_runs_sequential_path_with_identical_output() {
    let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
        eprintln!("OB_SMOKE_MODEL not set; skipping rpc pipeline disable test");
        return;
    };
    let model_path = Path::new(&model_path);
    let dev_index = devices()
        .iter()
        .position(|d| matches!(d.kind, DeviceKind::Cpu))
        .expect("a CPU device always exists") as i32;
    let session_params = SessionParams {
        n_ctx: 1024,
        n_batch: N_BATCH,
        n_ubatch: N_UBATCH,
        ..SessionParams::default()
    };

    // The knob must land before the distributed context is created — llama
    // caches the gate result per context.
    set_pipeline_overlap(false);

    // Solo ground truth with the same session shape.
    let (solo_tokens, prompt) = {
        let solo = Model::load(model_path, &ModelParams::default()).expect("solo load");
        let text = "Once upon a time there was a little girl named Lily. ".repeat(40);
        let prompt = solo.tokenize(&text, true).expect("tokenize long prompt");
        assert!(
            prompt.len() > 2 * N_BATCH as usize,
            "prompt must exceed 2*n_batch tokens for a multi-ubatch prefill (got {})",
            prompt.len()
        );
        let mut session = Session::new(&solo, &session_params).expect("solo session");
        let tokens = session
            .generate_greedy(&prompt, 8, |_, _| {})
            .expect("solo greedy generation");
        (tokens, prompt)
    };
    assert!(!solo_tokens.is_empty(), "solo baseline generated no tokens");

    // Minimal loopback bridge (one serve session + byte pump per accepted
    // connection), as in rpc_pipeline.rs but without the kill handles.
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback listener");
    listener
        .set_nonblocking(true)
        .expect("nonblocking listener");
    let endpoint = format!("127.0.0.1:{}", listener.local_addr().expect("addr").port());
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);
    let acceptor = thread::spawn(move || {
        let mut sessions: Vec<(RpcServeSession, Pump)> = Vec::new();
        while !stop_flag.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((conn, _)) => {
                    conn.set_nonblocking(false).expect("blocking conn");
                    conn.set_nodelay(true).ok();
                    let (raw, bridge) = SocketPair::new().expect("socket pair").into_parts();
                    let serve = RpcServeSession::spawn(raw, 2, dev_index, None)
                        .expect("spawn serve session");
                    sessions.push((serve, pump(conn, bridge).expect("spawn pump")));
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
        sessions
    });

    let server = RemoteServer::register(&endpoint).expect("register bridged rpc server");
    let engaged_before = pipeline_parallel_engagements();
    let dist_tokens = {
        let model = Model::load_distributed(
            model_path,
            &[&server],
            &[0.5, 0.5],
            /*use_local_device=*/ true,
            &ModelParams::default(),
        )
        .expect("distributed load through the loopback bridge");
        let mut session = Session::new(&model, &session_params).expect("distributed session");
        session
            .generate_greedy(&prompt, 8, |_, _| {})
            .expect("distributed greedy generation")
        // Session + model drop here, before the bridge is torn down.
    };
    let engaged_after = pipeline_parallel_engagements();

    assert_eq!(
        engaged_after, engaged_before,
        "with prefill overlap disabled, llama.cpp must NOT log 'pipeline \
         parallelism enabled' — the caps advertisement gates the sched shape \
         (docs/perf.md §3 constructed M3 baseline)"
    );
    assert_eq!(
        dist_tokens, solo_tokens,
        "sequential distributed greedy tokens must be byte-identical to the solo run"
    );

    // Restore the process default for hygiene (own binary, but explicit).
    set_pipeline_overlap(true);

    stop.store(true, Ordering::Relaxed);
    for (i, (serve, p)) in acceptor
        .join()
        .expect("acceptor thread")
        .into_iter()
        .enumerate()
    {
        assert!(
            serve.join_timeout(Duration::from_secs(5)).is_ok(),
            "disable-path serve session {i} did not end within 5s"
        );
        p.join();
    }
}
