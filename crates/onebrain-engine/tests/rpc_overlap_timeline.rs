//! M7 overlap investigation instrument (docs/perf.md §3): a LOCAL,
//! wire-throttled reproduction of the CI netem leg's overlapped-prefill
//! measurement, built entirely from engine-owned pieces:
//!
//!  - a transfer-heavy synthetic GGUF (the sim's perf-overlap shape:
//!    n_embd 8192, 2 layers, thin 16-wide attention and a 32-wide FFN —
//!    a fat hidden state to ship, almost nothing to multiply it by);
//!  - a bandwidth-paced loopback bridge (default 125 MB/s ≈ the netem
//!    leg's 1 Gbit) in place of netem, which Windows lacks, with BOUNDED
//!    path queues (`OB_OVERLAP_QUEUE_KB` per queue, default 4096; two
//!    queues per direction: sndbuf+qdisc before the wire, rcvbuf after)
//!    whose producers BLOCK when full — TCP backpressure, the behavior a
//!    real bounded netem qdisc forces end to end. An optional
//!    `OB_OVERLAP_DROP_PENALTY_MS` adds a crude per-overflow retransmit
//!    stall to demonstrate the drop-tail failure mode;
//!  - a distributed load over one bridged RPC server + the local CPU
//!    device (the exact 2-node CI topology: worker = first stage remote,
//!    head = final stage local), n_batch 512 / n_ubatch 64.
//!
//! The test times a multi-chunk prefill with `prefill_overlap` off (the
//! constructed M3 baseline) and on, prints both, and asserts only sanity
//! (identical greedy tokens, overlap not slower than 1.35x sequential) so
//! it stays green on noisy shared runners. Set `OB_OVERLAP_MAX_RATIO`
//! (e.g. `0.75`) to enforce a ratio locally, `OB_OVERLAP_BW_MBPS` to move
//! the emulated bandwidth off the 125 MB/s default, and
//! `OB_RPC_INFLIGHT_CAP` to move (or 0 to disable) the client's in-flight
//! byte cap when studying pipelining depth.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use onebrain_engine::rpc::{set_pipeline_overlap, RemoteServer, RpcServeSession, SocketPair};
use onebrain_engine::{devices, DeviceKind, Model, ModelParams, Session, SessionParams, Token};

// ---- synthetic transfer-heavy model (mirrors xtask sim's perf model) ----

const PERF_N_LAYERS: u32 = 2;
const PERF_N_EMBD: u32 = 8192;
const PERF_N_HEAD: u32 = 1;
const PERF_N_HEAD_KV: u32 = 1;
const PERF_HEAD_DIM: u32 = 16;
const PERF_N_FF: u32 = 32;
const PERF_VOCAB: u32 = 259;

const N_BATCH: u32 = 512;
const N_UBATCH: u32 = 64;
/// ~53 ubatches of 64 across ~7 n_batch chunks, like the CI leg.
const PROMPT_TOKENS: usize = 3424;
const DECODE_TOKENS: usize = 16;

/// Minimal GGUF v3 writer (subset of the sim's builder): metadata + F32
/// tensor infos + an all-zero 32-aligned data section.
struct GgufBuilder {
    kv_count: u64,
    kvs: Vec<u8>,
    tensor_count: u64,
    infos: Vec<u8>,
    data_len: u64,
}

impl GgufBuilder {
    fn new() -> GgufBuilder {
        GgufBuilder {
            kv_count: 0,
            kvs: Vec::new(),
            tensor_count: 0,
            infos: Vec::new(),
            data_len: 0,
        }
    }

    fn string_into(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    fn kv_str(&mut self, key: &str, val: &str) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(8u32.to_le_bytes()); // string
        Self::string_into(&mut self.kvs, val);
        self.kv_count += 1;
    }

    fn kv_u32(&mut self, key: &str, val: u32) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(4u32.to_le_bytes()); // u32
        self.kvs.extend(val.to_le_bytes());
        self.kv_count += 1;
    }

    fn kv_f32(&mut self, key: &str, val: f32) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(6u32.to_le_bytes()); // f32
        self.kvs.extend(val.to_le_bytes());
        self.kv_count += 1;
    }

    fn kv_str_array(&mut self, key: &str, vals: &[String]) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(9u32.to_le_bytes()); // array
        self.kvs.extend(8u32.to_le_bytes()); // of string
        self.kvs.extend((vals.len() as u64).to_le_bytes());
        for v in vals {
            Self::string_into(&mut self.kvs, v);
        }
        self.kv_count += 1;
    }

    fn kv_f32_array_zeroed(&mut self, key: &str, n: u64) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(9u32.to_le_bytes()); // array
        self.kvs.extend(6u32.to_le_bytes()); // of f32
        self.kvs.extend(n.to_le_bytes());
        self.kvs.extend(std::iter::repeat_n(0u8, (n * 4) as usize));
        self.kv_count += 1;
    }

    fn kv_i32_array(&mut self, key: &str, vals: &[i32]) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(9u32.to_le_bytes()); // array
        self.kvs.extend(5u32.to_le_bytes()); // of i32
        self.kvs.extend((vals.len() as u64).to_le_bytes());
        for v in vals {
            self.kvs.extend(v.to_le_bytes());
        }
        self.kv_count += 1;
    }

    fn tensor_f32(&mut self, name: &str, dims: &[u64]) {
        let bytes = dims.iter().product::<u64>() * 4;
        assert_eq!(bytes % 32, 0, "tensor {name} breaks 32-byte alignment");
        Self::string_into(&mut self.infos, name);
        self.infos.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            self.infos.extend(d.to_le_bytes());
        }
        self.infos.extend(0u32.to_le_bytes()); // GGML_TYPE_F32
        self.infos.extend(self.data_len.to_le_bytes());
        self.tensor_count += 1;
        self.data_len += bytes;
    }

    fn build(self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(0x4655_4747u32.to_le_bytes()); // "GGUF"
        out.extend(3u32.to_le_bytes());
        out.extend(self.tensor_count.to_le_bytes());
        out.extend(self.kv_count.to_le_bytes());
        out.extend(&self.kvs);
        out.extend(&self.infos);
        let data_offset = (out.len() as u64).div_ceil(32) * 32;
        out.resize((data_offset + self.data_len) as usize, 0);
        out
    }
}

fn push_spm_byte_vocab(g: &mut GgufBuilder) {
    g.kv_str("tokenizer.ggml.model", "llama");
    let mut tokens: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    tokens.extend((0u32..256).map(|b| format!("<0x{b:02X}>")));
    let mut types: Vec<i32> = vec![2, 3, 3];
    types.extend(std::iter::repeat_n(6, 256));
    g.kv_str_array("tokenizer.ggml.tokens", &tokens);
    g.kv_f32_array_zeroed("tokenizer.ggml.scores", u64::from(PERF_VOCAB));
    g.kv_i32_array("tokenizer.ggml.token_type", &types);
    g.kv_u32("tokenizer.ggml.bos_token_id", 1);
    g.kv_u32("tokenizer.ggml.eos_token_id", 2);
    g.kv_u32("tokenizer.ggml.unknown_token_id", 0);
}

fn build_perf_model() -> Vec<u8> {
    let mut g = GgufBuilder::new();
    g.kv_str("general.architecture", "llama");
    g.kv_str("general.name", "onebrain-engine-overlap-timeline");
    g.kv_u32("llama.block_count", PERF_N_LAYERS);
    g.kv_u32("llama.context_length", 4096);
    g.kv_u32("llama.embedding_length", PERF_N_EMBD);
    g.kv_u32("llama.feed_forward_length", PERF_N_FF);
    g.kv_u32("llama.attention.head_count", PERF_N_HEAD);
    g.kv_u32("llama.attention.head_count_kv", PERF_N_HEAD_KV);
    g.kv_u32("llama.attention.key_length", PERF_HEAD_DIM);
    g.kv_u32("llama.attention.value_length", PERF_HEAD_DIM);
    g.kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    g.kv_u32("llama.rope.dimension_count", PERF_HEAD_DIM);
    push_spm_byte_vocab(&mut g);

    let e = u64::from(PERF_N_EMBD);
    let v = u64::from(PERF_VOCAB);
    let ff = u64::from(PERF_N_FF);
    let hd = u64::from(PERF_HEAD_DIM);
    g.tensor_f32("token_embd.weight", &[e, v]);
    for i in 0..PERF_N_LAYERS {
        g.tensor_f32(&format!("blk.{i}.attn_norm.weight"), &[e]);
        g.tensor_f32(&format!("blk.{i}.attn_q.weight"), &[e, hd]);
        g.tensor_f32(&format!("blk.{i}.attn_k.weight"), &[e, hd]);
        g.tensor_f32(&format!("blk.{i}.attn_v.weight"), &[e, hd]);
        g.tensor_f32(&format!("blk.{i}.attn_output.weight"), &[hd, e]);
        g.tensor_f32(&format!("blk.{i}.ffn_norm.weight"), &[e]);
        g.tensor_f32(&format!("blk.{i}.ffn_gate.weight"), &[e, ff]);
        g.tensor_f32(&format!("blk.{i}.ffn_up.weight"), &[e, ff]);
        g.tensor_f32(&format!("blk.{i}.ffn_down.weight"), &[ff, e]);
    }
    g.tensor_f32("output_norm.weight", &[e]);
    g.tensor_f32("output.weight", &[e, v]);
    g.build()
}

// ---- bandwidth-paced loopback bridge (the local stand-in for netem) ----

/// A byte-bounded blocking FIFO of chunks — the emulation's stand-in for a
/// BOUNDED network buffer. A producer that finds it full BLOCKS until the
/// consumer drains it; end to end that is exactly what TCP flow control
/// does to the sending application. Real netem is a bounded qdisc (default
/// limit ~1000 packets ≈ 1.5 MB) in front of a bounded kernel rcvbuf —
/// modeling the path as an unbounded queue is what let the
/// unbounded-pipelining regression through to CI.
struct ByteQueue {
    state: std::sync::Mutex<ByteQueueState>,
    cond: std::sync::Condvar,
    cap: usize,
    /// Crude drop-tail model (off by default; OB_OVERLAP_DROP_PENALTY_MS):
    /// a producer that finds the queue full first serves this delay per
    /// overflowing chunk — standing in for the retransmit/recovery stall a
    /// real tail-drop causes — then blocks for space. A byte relay cannot
    /// literally drop bytes: real TCP retransmits them and the RPC stream
    /// must stay intact, so the penalty models the stall, not the loss.
    overflow_penalty: Duration,
}

struct ByteQueueState {
    chunks: std::collections::VecDeque<Vec<u8>>,
    bytes: usize,
    closed: bool,
}

impl ByteQueue {
    fn new(cap: usize, overflow_penalty: Duration) -> ByteQueue {
        ByteQueue {
            state: std::sync::Mutex::new(ByteQueueState {
                chunks: std::collections::VecDeque::new(),
                bytes: 0,
                closed: false,
            }),
            cond: std::sync::Condvar::new(),
            cap,
            overflow_penalty,
        }
    }

    /// Blocking push; a chunk is never split, so a non-empty queue that
    /// would exceed the cap blocks (one chunk may overshoot an otherwise
    /// empty queue, like a packet entering an empty qdisc).
    fn push(&self, chunk: Vec<u8>) {
        let mut st = self.state.lock().unwrap();
        if !st.chunks.is_empty() && st.bytes + chunk.len() > self.cap {
            if !self.overflow_penalty.is_zero() {
                drop(st);
                thread::sleep(self.overflow_penalty);
                st = self.state.lock().unwrap();
            }
            while !st.chunks.is_empty() && st.bytes + chunk.len() > self.cap {
                st = self.cond.wait(st).unwrap();
            }
        }
        st.bytes += chunk.len();
        st.chunks.push_back(chunk);
        self.cond.notify_all();
    }

    /// Blocking pop; `None` once closed and drained.
    fn pop(&self) -> Option<Vec<u8>> {
        let mut st = self.state.lock().unwrap();
        loop {
            if let Some(chunk) = st.chunks.pop_front() {
                st.bytes -= chunk.len();
                self.cond.notify_all();
                return Some(chunk);
            }
            if st.closed {
                return None;
            }
            st = self.cond.wait(st).unwrap();
        }
    }

    fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.cond.notify_all();
    }
}

/// Per-queue byte bound (`OB_OVERLAP_QUEUE_KB`, default 4096 KiB). The
/// relay has two bounded queues per direction (sndbuf+qdisc before the
/// wire, rcvbuf after it), so total path buffering is twice this.
fn queue_bytes() -> usize {
    std::env::var("OB_OVERLAP_QUEUE_KB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4096)
        * 1024
}

fn overflow_penalty() -> Duration {
    Duration::from_millis(
        std::env::var("OB_OVERLAP_DROP_PENALTY_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0),
    )
}

/// The two stream types a relay leg can carry: the accepted loopback
/// `TcpStream` on every OS, and the bridge end of a [`SocketPair`], which
/// is `TcpStream` on Windows but `UnixStream` elsewhere — the CI matrix
/// caught the concrete-`TcpStream` version of this signature.
trait RelayStream: Read + Write + Send + 'static {
    fn shutdown_write(&self);
}

impl RelayStream for TcpStream {
    fn shutdown_write(&self) {
        let _ = self.shutdown(std::net::Shutdown::Write);
    }
}

#[cfg(unix)]
impl RelayStream for std::os::unix::net::UnixStream {
    fn shutdown_write(&self) {
        let _ = self.shutdown(std::net::Shutdown::Write);
    }
}

/// Pace one direction at `bytes_per_sec` through BOUNDED buffers:
///
/// ```text
///   reader -> [pre queue]   -> pacer  -> [post queue] -> writer
///             (sndbuf+qdisc)   (wire)     (rcvbuf)
/// ```
///
/// Both queues are bounded (`OB_OVERLAP_QUEUE_KB` each) and BLOCK their
/// producer when full — TCP backpressure end to end: the reader blocking
/// propagates to the client's `send`, and the pacer blocking on a full
/// post queue stalls the wire clock exactly as a closed receive window
/// does. Deadline pacing is banking-free (an idle wire earns no credit —
/// it cannot send yesterday's unused bytes faster today) and uses
/// yield-spinning: Windows `thread::sleep` is far too coarse (~15 ms) for
/// sub-millisecond chunk budgets.
fn paced_relay(mut from: impl RelayStream, mut to: impl RelayStream, bytes_per_sec: f64) {
    let cap = queue_bytes();
    let pre = Arc::new(ByteQueue::new(cap, overflow_penalty()));
    let post = Arc::new(ByteQueue::new(cap, Duration::ZERO));

    let pre_r = Arc::clone(&pre);
    let reader = thread::spawn(move || {
        let mut buf = vec![0u8; 256 * 1024];
        loop {
            match from.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => pre_r.push(buf[..n].to_vec()),
            }
        }
        pre_r.close();
    });

    let post_w = Arc::clone(&post);
    let writer = thread::spawn(move || {
        while let Some(chunk) = post_w.pop() {
            if to.write_all(&chunk).is_err() {
                break;
            }
            let _ = to.flush();
        }
        to.shutdown_write();
    });

    let mut deadline = Instant::now();
    while let Some(chunk) = pre.pop() {
        let now = Instant::now();
        if deadline < now {
            deadline = now;
        }
        deadline += Duration::from_secs_f64(chunk.len() as f64 / bytes_per_sec);
        // Windows needs yield-spinning (thread::sleep is ~15 ms coarse);
        // everywhere else sleep to the deadline — spinning pacer threads
        // oversubscribe the 3-core macOS CI runners and starve the very
        // compute this test times (macos-14 measured ratio 1.45 from
        // pacer-thread contention alone).
        #[cfg(windows)]
        while Instant::now() < deadline {
            thread::yield_now();
        }
        #[cfg(not(windows))]
        {
            let now = Instant::now();
            if now < deadline {
                thread::sleep(deadline - now);
            }
        }
        post.push(chunk);
    }
    post.close();
    let _ = writer.join();
    let _ = reader.join();
}

struct ThrottledBridge {
    port: u16,
    stop: Arc<AtomicBool>,
    acceptor: thread::JoinHandle<Vec<RpcServeSession>>,
}

impl ThrottledBridge {
    fn spawn(dev_index: i32, bytes_per_sec: f64) -> ThrottledBridge {
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
                        conn.set_nonblocking(false).expect("blocking conn");
                        conn.set_nodelay(true).ok();
                        let (raw, bridge) = SocketPair::new().expect("socket pair").into_parts();
                        let serve = RpcServeSession::spawn(raw, 2, dev_index, None)
                            .expect("spawn serve session");
                        // Two paced relay threads per connection (each
                        // direction is its own 1 Gbit leg, like netem).
                        let conn_r = conn.try_clone().expect("clone conn");
                        let bridge_w = bridge.try_clone().expect("clone bridge");
                        let bridge_r = bridge;
                        let conn_w = conn;
                        thread::spawn(move || paced_relay(conn_r, bridge_w, bytes_per_sec));
                        thread::spawn(move || paced_relay(bridge_r, conn_w, bytes_per_sec));
                        sessions.push(serve);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
            sessions
        });
        ThrottledBridge {
            port,
            stop,
            acceptor,
        }
    }

    fn finish(self, timeout: Duration, what: &str) {
        self.stop.store(true, Ordering::Relaxed);
        let sessions = self.acceptor.join().expect("acceptor thread");
        for (i, serve) in sessions.into_iter().enumerate() {
            assert!(
                serve.join_timeout(timeout).is_ok(),
                "{what}: serve session {i} did not end within {timeout:?}"
            );
        }
    }
}

fn cpu_device_index() -> i32 {
    devices()
        .iter()
        .position(|d| matches!(d.kind, DeviceKind::Cpu))
        .expect("a CPU device always exists") as i32
}

fn prompt_tokens() -> Vec<Token> {
    // BOS then an arbitrary in-vocab byte token; the all-zero model makes
    // the content meaningless (logits tie at 0, greedy picks token 0).
    let mut p = vec![1i32];
    p.extend(std::iter::repeat_n(5i32, PROMPT_TOKENS - 1));
    p
}

struct RunStats {
    load_ms: u128,
    prefill_ms: u128,
    decode_ms: u128,
    tokens: Vec<Token>,
}

fn run_once(model_path: &Path, overlap: bool, bytes_per_sec: f64) -> RunStats {
    set_pipeline_overlap(overlap);
    let bridge = ThrottledBridge::spawn(cpu_device_index(), bytes_per_sec);
    let endpoint = format!("127.0.0.1:{}", bridge.port);
    let server = RemoteServer::register(&endpoint).expect("register bridged rpc server");

    let t0 = Instant::now();
    let model = Model::load_distributed(
        model_path,
        &[&server],
        &[0.5, 0.5],
        /*use_local_device=*/ true,
        &ModelParams::default(),
    )
    .expect("distributed load through the throttled bridge");
    let load_ms = t0.elapsed().as_millis();

    let params = SessionParams {
        n_ctx: 4096,
        n_batch: N_BATCH,
        n_ubatch: N_UBATCH,
        ..SessionParams::default()
    };
    let mut session = Session::new(&model, &params).expect("distributed session");

    let prompt = prompt_tokens();
    let t1 = Instant::now();
    session.decode(&prompt).expect("multi-ubatch prefill");
    let prefill_ms = t1.elapsed().as_millis();

    let mut tokens = Vec::new();
    let t2 = Instant::now();
    for _ in 0..DECODE_TOKENS {
        let tok = session.sample_greedy();
        tokens.push(tok);
        session.decode(&[tok]).expect("decode step");
    }
    let decode_ms = t2.elapsed().as_millis();

    drop(session);
    drop(model);
    bridge.finish(Duration::from_secs(10), "overlap timeline bridge");

    // Restore the process-wide default for whatever test runs next.
    set_pipeline_overlap(true);

    RunStats {
        load_ms,
        prefill_ms,
        decode_ms,
        tokens,
    }
}

fn write_model() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir for synthetic model");
    let path = dir.path().join("overlap-timeline.gguf");
    std::fs::write(&path, build_perf_model()).expect("write synthetic model");
    (dir, path)
}

/// Sequential (constructed M3 baseline) vs overlapped prefill over the
/// paced bridge. Prints the measured wall times; the CI-noise-safe asserts
/// are correctness (identical greedy tokens) plus a loose "overlap must
/// not be materially slower" bound. `OB_OVERLAP_MAX_RATIO` tightens the
/// ratio assert for local/netem-grade environments.
#[test]
fn overlap_ratio_timeline() {
    let bytes_per_sec = std::env::var("OB_OVERLAP_BW_MBPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(125.0)
        * 1_000_000.0;
    let (_dir, model_path) = write_model();

    let seq = run_once(&model_path, false, bytes_per_sec);
    let ovl = run_once(&model_path, true, bytes_per_sec);

    let n_ubatches = PROMPT_TOKENS.div_ceil(N_UBATCH as usize);
    let ratio = ovl.prefill_ms as f64 / seq.prefill_ms.max(1) as f64;
    println!(
        "overlap-timeline @ {:.0} MB/s emulated wire, {} KiB path queues:\n\
         sequential: load {} ms, prefill {} ms ({:.1} ms/ubatch), decode {} ms ({} tok)\n\
         overlapped: load {} ms, prefill {} ms ({:.1} ms/ubatch), decode {} ms ({} tok)\n\
         prefill ratio overlap/sequential = {ratio:.3}",
        bytes_per_sec / 1e6,
        queue_bytes() / 1024,
        seq.load_ms,
        seq.prefill_ms,
        seq.prefill_ms as f64 / n_ubatches as f64,
        seq.decode_ms,
        DECODE_TOKENS,
        ovl.load_ms,
        ovl.prefill_ms,
        ovl.prefill_ms as f64 / n_ubatches as f64,
        ovl.decode_ms,
        DECODE_TOKENS,
    );

    assert_eq!(
        seq.tokens, ovl.tokens,
        "overlapped greedy tokens must be byte-identical to the sequential run"
    );
    assert!(
        ratio < 1.35,
        "overlapped prefill must not be materially slower than sequential \
         (ratio {ratio:.3}; sequential {} ms, overlapped {} ms)",
        seq.prefill_ms,
        ovl.prefill_ms
    );
    if let Ok(max) = std::env::var("OB_OVERLAP_MAX_RATIO") {
        let max: f64 = max.parse().expect("OB_OVERLAP_MAX_RATIO must be a float");
        assert!(
            ratio <= max,
            "overlap ratio {ratio:.3} exceeds OB_OVERLAP_MAX_RATIO {max}"
        );
    }
}
