//! latbench — MEASUREMENT-ONLY latency microbench for the engine hot path.
//!
//! Profiles per-token decode latency, TTFT composition (including the
//! confirm-before-send decode), thread scaling on hybrid P/E-core boxes,
//! sampler-chain cost, host-step-loop machinery overhead, KV/flash/ubatch
//! axes, and distributed loopback per-token RPC traffic (command counts via
//! a frame-parsing relay). It changes NO product behavior; every number it
//! prints is a measurement of the existing code paths.
//!
//! Usage (release builds only — debug timings are meaningless):
//!   latbench synth <out.gguf>                 write the ~540MB mid synthetic
//!   latbench solo <model> [prompt] [gen]      per-token cost breakdown
//!   latbench ttft <model> [prompt]            TTFT component breakdown
//!   latbench threads <model> <n|d:b|det,...>  prefill/decode tps per count
//!                                             (n ties decode+batch; d:b sets
//!                                             them separately; det = the cpu
//!                                             module's detected default)
//!   latbench axes <model>                     flash_attn / kv type / n_ubatch
//!   latbench sampler <model>                  chain-vs-greedy sample cost
//!   latbench steploop <model> [gen]           generate() vs host-style loop
//!   latbench dist <model> [prompt] [gen]      loopback 2-device decode + RPC counts

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use onebrain_engine::rpc::{RemoteServer, RpcServeSession, SocketPair};
use onebrain_engine::{
    devices, Batch, DeviceKind, FlashAttnType, KvCacheType, Model, ModelParams, Sampler,
    SamplerParams, Session, SessionParams, Token,
};

// ---- counting allocator (Rust-side hot-path allocation audit) ----

struct CountingAlloc;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn alloc_snapshot() -> (u64, u64) {
    (
        ALLOCS.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
    )
}

// ---- timing helpers ----

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn pct(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() - 1) as f64 * p).round() as usize;
    v[idx]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

// ---- synthetic mid-size GGUF (adapted from tests/rpc_overlap_timeline.rs) ----

const MID_N_LAYERS: u32 = 6;
const MID_N_EMBD: u32 = 896;
const MID_N_HEAD: u32 = 14;
const MID_HEAD_DIM: u32 = 64;
const MID_N_FF: u32 = 3584;
const MID_VOCAB: u32 = 32000;

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
        self.kvs.extend(8u32.to_le_bytes());
        Self::string_into(&mut self.kvs, val);
        self.kv_count += 1;
    }

    fn kv_u32(&mut self, key: &str, val: u32) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(4u32.to_le_bytes());
        self.kvs.extend(val.to_le_bytes());
        self.kv_count += 1;
    }

    fn kv_f32(&mut self, key: &str, val: f32) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(6u32.to_le_bytes());
        self.kvs.extend(val.to_le_bytes());
        self.kv_count += 1;
    }

    fn kv_str_array(&mut self, key: &str, vals: &[String]) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(9u32.to_le_bytes());
        self.kvs.extend(8u32.to_le_bytes());
        self.kvs.extend((vals.len() as u64).to_le_bytes());
        for v in vals {
            Self::string_into(&mut self.kvs, v);
        }
        self.kv_count += 1;
    }

    fn kv_f32_array_zeroed(&mut self, key: &str, n: u64) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(9u32.to_le_bytes());
        self.kvs.extend(6u32.to_le_bytes());
        self.kvs.extend(n.to_le_bytes());
        self.kvs.extend(std::iter::repeat_n(0u8, (n * 4) as usize));
        self.kv_count += 1;
    }

    fn kv_i32_array(&mut self, key: &str, vals: &[i32]) {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(9u32.to_le_bytes());
        self.kvs.extend(5u32.to_le_bytes());
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

/// A llama-shaped ~540MB F32 model with a REAL-SIZED vocab (32000) so
/// logits-row work (greedy scan, top-k, token_to_piece) costs what it
/// costs on production models, unlike the 259/512-token test vocabs.
fn build_mid_model() -> Vec<u8> {
    let mut g = GgufBuilder::new();
    g.kv_str("general.architecture", "llama");
    g.kv_str("general.name", "onebrain-latbench-mid");
    g.kv_u32("llama.block_count", MID_N_LAYERS);
    g.kv_u32("llama.context_length", 4096);
    g.kv_u32("llama.embedding_length", MID_N_EMBD);
    g.kv_u32("llama.feed_forward_length", MID_N_FF);
    g.kv_u32("llama.attention.head_count", MID_N_HEAD);
    g.kv_u32("llama.attention.head_count_kv", MID_N_HEAD);
    g.kv_u32("llama.attention.key_length", MID_HEAD_DIM);
    g.kv_u32("llama.attention.value_length", MID_HEAD_DIM);
    g.kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    g.kv_u32("llama.rope.dimension_count", MID_HEAD_DIM);

    g.kv_str("tokenizer.ggml.model", "llama");
    let mut tokens: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    tokens.extend((0u32..256).map(|b| format!("<0x{b:02X}>")));
    let mut types: Vec<i32> = vec![2, 3, 3];
    types.extend(std::iter::repeat_n(6, 256));
    while tokens.len() < MID_VOCAB as usize {
        tokens.push(format!("tok{:05}", tokens.len()));
        types.push(1); // NORMAL
    }
    g.kv_str_array("tokenizer.ggml.tokens", &tokens);
    g.kv_f32_array_zeroed("tokenizer.ggml.scores", u64::from(MID_VOCAB));
    g.kv_i32_array("tokenizer.ggml.token_type", &types);
    g.kv_u32("tokenizer.ggml.bos_token_id", 1);
    g.kv_u32("tokenizer.ggml.eos_token_id", 2);
    g.kv_u32("tokenizer.ggml.unknown_token_id", 0);

    let e = u64::from(MID_N_EMBD);
    let v = u64::from(MID_VOCAB);
    let ff = u64::from(MID_N_FF);
    let qkv = u64::from(MID_N_HEAD * MID_HEAD_DIM);
    g.tensor_f32("token_embd.weight", &[e, v]);
    for i in 0..MID_N_LAYERS {
        g.tensor_f32(&format!("blk.{i}.attn_norm.weight"), &[e]);
        g.tensor_f32(&format!("blk.{i}.attn_q.weight"), &[e, qkv]);
        g.tensor_f32(&format!("blk.{i}.attn_k.weight"), &[e, qkv]);
        g.tensor_f32(&format!("blk.{i}.attn_v.weight"), &[e, qkv]);
        g.tensor_f32(&format!("blk.{i}.attn_output.weight"), &[qkv, e]);
        g.tensor_f32(&format!("blk.{i}.ffn_norm.weight"), &[e]);
        g.tensor_f32(&format!("blk.{i}.ffn_gate.weight"), &[e, ff]);
        g.tensor_f32(&format!("blk.{i}.ffn_up.weight"), &[e, ff]);
        g.tensor_f32(&format!("blk.{i}.ffn_down.weight"), &[ff, e]);
    }
    g.tensor_f32("output_norm.weight", &[e]);
    g.tensor_f32("output.weight", &[e, v]);
    g.build()
}

/// Prompt of valid ids for any model: BOS then cycling in-vocab tokens.
fn make_prompt(model: &Model, n: usize) -> Vec<Token> {
    let seed = model
        .tokenize("Once upon a time there was a small model", true)
        .unwrap_or_default();
    if seed.is_empty() {
        let mut p = vec![1i32];
        p.extend(std::iter::repeat_n(5i32, n.saturating_sub(1)));
        return p;
    }
    seed.iter().copied().cycle().take(n).collect()
}

fn load(path: &Path) -> Model {
    Model::load(path, &ModelParams::default()).expect("model load")
}

// ---- solo per-token breakdown ----

fn cmd_solo(path: &Path, prompt_n: usize, gen_n: usize) {
    let model = load(path);
    let mut session = Session::new(&model, &SessionParams::default()).expect("session");
    let prompt = make_prompt(&model, prompt_n);

    // Warmup: page faults, threadpool spinup, graph build.
    session.decode(&prompt).expect("warmup prefill");
    for _ in 0..8 {
        let t = session.sample_greedy();
        session.decode(&[t]).expect("warmup decode");
    }
    session.reset();

    // Consumer thread so channel sends behave like a live client.
    let (tx, rx) = std::sync::mpsc::sync_channel::<String>(256);
    let drain = thread::spawn(move || while rx.recv().is_ok() {});

    let t0 = Instant::now();
    session.decode(&prompt).expect("prefill");
    let prefill = t0.elapsed();

    let mut d_decode = Vec::new();
    let mut d_sample = Vec::new();
    let mut d_t2p = Vec::new();
    let mut d_send = Vec::new();
    let mut d_total = Vec::new();
    let loop_t0 = Instant::now();
    for _ in 0..gen_n {
        let it0 = Instant::now();
        let t1 = Instant::now();
        let tok = session.sample();
        let sample_d = t1.elapsed();
        let _eog = model.is_eog(tok);
        let t2 = Instant::now();
        let piece = model.token_to_piece(tok).expect("piece");
        let t2p_d = t2.elapsed();
        let t3 = Instant::now();
        session.decode(&[tok]).expect("decode");
        let dec_d = t3.elapsed();
        let t4 = Instant::now();
        let _ = tx.try_send(piece);
        let send_d = t4.elapsed();
        d_sample.push(ms(sample_d));
        d_t2p.push(ms(t2p_d));
        d_decode.push(ms(dec_d));
        d_send.push(ms(send_d));
        d_total.push(ms(it0.elapsed()));
    }
    let loop_total = loop_t0.elapsed();
    drop(tx);
    let _ = drain.join();

    let per_tok = ms(loop_total) / gen_n as f64;
    let m_dec = median(d_decode.clone());
    let m_smp = median(d_sample.clone());
    let m_t2p = median(d_t2p.clone());
    let m_snd = median(d_send.clone());
    let m_tot = median(d_total.clone());
    println!(
        "solo breakdown model={} prompt={} gen={}",
        path.display(),
        prompt_n,
        gen_n
    );
    println!(
        "  prefill: {:.2} ms total, {:.3} ms/tok",
        ms(prefill),
        ms(prefill) / prompt_n as f64
    );
    println!("  per-token medians (p90):");
    println!(
        "    decode        {:8.3} ms ({:.3})",
        m_dec,
        pct(d_decode, 0.9)
    );
    println!(
        "    sample        {:8.4} ms ({:.4})",
        m_smp,
        pct(d_sample, 0.9)
    );
    println!(
        "    token_to_piece{:8.4} ms ({:.4})",
        m_t2p,
        pct(d_t2p, 0.9)
    );
    println!(
        "    channel send  {:8.4} ms ({:.4})",
        m_snd,
        pct(d_send, 0.9)
    );
    println!(
        "    iter total    {:8.3} ms ({:.3})",
        m_tot,
        pct(d_total, 0.9)
    );
    println!("  loop wall {:.3} ms/tok over {} tokens", per_tok, gen_n);
    println!(
        "  llama compute (decode) fraction of iter: {:.1}%  machinery: {:.3} ms/tok",
        100.0 * m_dec / m_tot,
        m_tot - m_dec
    );
}

// ---- TTFT breakdown ----

fn cmd_ttft(path: &Path, prompt_n: usize) {
    let model = load(path);
    let mut session = Session::new(&model, &SessionParams::default()).expect("session");
    let prompt = make_prompt(&model, prompt_n);

    // Warmup.
    session.decode(&prompt).expect("warmup");
    let t = session.sample_greedy();
    session.decode(&[t]).expect("warmup2");

    const REPS: usize = 5;
    let mut c_prefill = Vec::new();
    let mut c_sample = Vec::new();
    let mut c_t2p = Vec::new();
    let mut c_confirm = Vec::new();
    for _ in 0..REPS {
        session.reset();
        let t0 = Instant::now();
        session.decode(&prompt).expect("prefill");
        c_prefill.push(ms(t0.elapsed()));
        let t1 = Instant::now();
        let tok = session.sample();
        c_sample.push(ms(t1.elapsed()));
        let t2 = Instant::now();
        let _piece = model.token_to_piece(tok).expect("piece");
        c_t2p.push(ms(t2.elapsed()));
        // Confirm-before-send: the emitted piece waits for this decode.
        let t3 = Instant::now();
        session.decode(&[tok]).expect("confirm decode");
        c_confirm.push(ms(t3.elapsed()));
    }
    let p = median(c_prefill);
    let s = median(c_sample);
    let z = median(c_t2p);
    let c = median(c_confirm);
    println!(
        "ttft breakdown model={} prompt={} (median of {REPS})",
        path.display(),
        prompt_n
    );
    println!("  prefill          {p:9.3} ms");
    println!("  first sample     {s:9.4} ms");
    println!("  token_to_piece   {z:9.4} ms");
    println!("  confirming decode{c:9.3} ms   <- delays first emit (confirm-before-send)");
    println!("  TTFT as shipped  {:9.3} ms", p + s + z + c);
    println!(
        "  TTFT w/o confirm {:9.3} ms  ({:.1}% lower)",
        p + s + z,
        100.0 * c / (p + s + z + c)
    );
}

// ---- generate() end-to-end TTFT (whatever emission policy is compiled in) ----

fn cmd_genttft(path: &Path, prompt_n: usize) {
    let model = load(path);
    let mut session = Session::new(&model, &SessionParams::default()).expect("session");
    let prompt = make_prompt(&model, prompt_n);
    session
        .generate(&prompt, 4, |_, _| std::ops::ControlFlow::Continue(()))
        .expect("warmup");
    const REPS: usize = 5;
    let mut ttfts = Vec::new();
    for _ in 0..REPS {
        session.reset();
        let stats = session
            .generate(&prompt, 4, |_, _| std::ops::ControlFlow::Continue(()))
            .expect("generate");
        ttfts.push(stats.ttft_ms);
    }
    println!(
        "generate() ttft model={} prompt={}: median {:.3} ms (of {REPS})",
        path.display(),
        prompt_n,
        median(ttfts)
    );
}

// ---- thread scaling ----

fn cmd_threads(path: &Path, counts: &[(i32, i32)]) {
    let model = load(path);
    let prompt = make_prompt(&model, 256);
    const DECODE_STEPS: usize = 32;
    const REPS: usize = 3;
    println!(
        "thread scaling model={} prompt=256 decode={} reps={} (medians)",
        path.display(),
        DECODE_STEPS,
        REPS
    );
    println!(
        "{:>10} {:>14} {:>14} {:>12} {:>12}",
        "threads", "prefill tok/s", "decode tok/s", "prefill ms", "ms/decode-tok"
    );
    for &(n, nb) in counts {
        let mut session = Session::new(
            &model,
            &SessionParams {
                n_threads: n,
                n_threads_batch: nb,
                ..SessionParams::default()
            },
        )
        .expect("session");
        // Warmup.
        session.decode(&prompt).expect("warmup");
        for _ in 0..4 {
            let t = session.sample_greedy();
            session.decode(&[t]).expect("warmup decode");
        }
        let mut pre = Vec::new();
        let mut dec = Vec::new();
        for _ in 0..REPS {
            session.reset();
            let t0 = Instant::now();
            session.decode(&prompt).expect("prefill");
            pre.push(ms(t0.elapsed()));
            let t1 = Instant::now();
            for _ in 0..DECODE_STEPS {
                let t = session.sample_greedy();
                session.decode(&[t]).expect("decode");
            }
            dec.push(ms(t1.elapsed()));
        }
        let pre_ms = median(pre);
        let dec_ms = median(dec);
        println!(
            "{:>10} {:>14.1} {:>14.1} {:>12.1} {:>12.3}",
            match (n <= 0, nb <= 0) {
                (true, true) => "auto(4)".to_string(),
                (false, true) => n.to_string(),
                (_, false) => format!("{n}:{nb}"),
            },
            256.0 / (pre_ms / 1e3),
            DECODE_STEPS as f64 / (dec_ms / 1e3),
            pre_ms,
            dec_ms / DECODE_STEPS as f64
        );
    }
}

// ---- flash_attn / kv-type / n_ubatch axes ----

fn run_axis(model: &Model, params: &SessionParams, label: &str, prompt: &[Token]) {
    const DECODE_STEPS: usize = 32;
    const REPS: usize = 3;
    let mut session = match Session::new(model, params) {
        Ok(s) => s,
        Err(e) => {
            println!("{label:<28} UNSUPPORTED: {e}");
            return;
        }
    };
    session.decode(prompt).expect("warmup");
    for _ in 0..4 {
        let t = session.sample_greedy();
        session.decode(&[t]).expect("warmup decode");
    }
    let mut pre = Vec::new();
    let mut dec = Vec::new();
    for _ in 0..REPS {
        session.reset();
        let t0 = Instant::now();
        session.decode(prompt).expect("prefill");
        pre.push(ms(t0.elapsed()));
        let t1 = Instant::now();
        for _ in 0..DECODE_STEPS {
            let t = session.sample_greedy();
            session.decode(&[t]).expect("decode");
        }
        dec.push(ms(t1.elapsed()));
    }
    println!(
        "{label:<28} prefill {:>9.1} ms ({:>8.1} tok/s)   decode {:>7.3} ms/tok",
        median(pre.clone()),
        prompt.len() as f64 / (median(pre) / 1e3),
        median(dec) / DECODE_STEPS as f64
    );
}

fn cmd_axes(path: &Path) {
    let model = load(path);
    let prompt = make_prompt(&model, 1024);
    println!("axes model={} prompt=1024 (medians of 3)", path.display());
    let base = SessionParams::default();

    println!("-- flash_attn --");
    for (fa, name) in [
        (FlashAttnType::Auto, "flash=Auto (default)"),
        (FlashAttnType::Disabled, "flash=Disabled"),
        (FlashAttnType::Enabled, "flash=Enabled"),
    ] {
        run_axis(
            &model,
            &SessionParams {
                flash_attn_type: fa,
                ..base.clone()
            },
            name,
            &prompt,
        );
    }

    println!("-- kv cache type --");
    for (tk, tv, name) in [
        (KvCacheType::F16, KvCacheType::F16, "kv=F16/F16 (default)"),
        (KvCacheType::F32, KvCacheType::F32, "kv=F32/F32"),
        (KvCacheType::Q8_0, KvCacheType::Q8_0, "kv=Q8_0/Q8_0"),
    ] {
        run_axis(
            &model,
            &SessionParams {
                type_k: tk,
                type_v: tv,
                ..base.clone()
            },
            name,
            &prompt,
        );
    }

    println!("-- n_ubatch (prefill) --");
    for ub in [64u32, 128, 256, 512] {
        run_axis(
            &model,
            &SessionParams {
                n_ubatch: ub,
                ..base.clone()
            },
            &format!("n_ubatch={ub}"),
            &prompt,
        );
    }
}

// ---- sampler chain cost ----

fn cmd_sampler(path: &Path) {
    let model = load(path);
    let mut session = Session::new(&model, &SessionParams::default()).expect("session");
    let prompt = make_prompt(&model, 64);
    session.decode(&prompt).expect("prefill");

    const N: usize = 2000;
    // Greedy via session chain (argmax over vocab).
    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..N {
        sink += i64::from(session.sample_greedy());
    }
    let greedy = t0.elapsed();

    // sample_ith(-1) via session chain (same greedy chain, index path).
    let t1 = Instant::now();
    for _ in 0..N {
        sink += i64::from(session.sample_ith(-1));
    }
    let ith = t1.elapsed();

    // Standalone greedy chain via sample_ith_with (the host's actual call).
    let mut greedy_chain = Sampler::new(&SamplerParams {
        temperature: 0.0,
        ..SamplerParams::default()
    })
    .expect("greedy chain");
    let t2 = Instant::now();
    for _ in 0..N {
        sink += i64::from(session.sample_ith_with(&mut greedy_chain, -1));
    }
    let ith_with = t2.elapsed();

    // Full dist chain (top-k 40, top-p 0.95, temp 0.8) — the sampled path.
    let mut dist_chain = Sampler::new(&SamplerParams::default()).expect("dist chain");
    let t3 = Instant::now();
    for _ in 0..N {
        sink += i64::from(session.sample_ith_with(&mut dist_chain, -1));
    }
    let dist = t3.elapsed();

    // token_to_piece cost + allocation count.
    let (a0, b0) = alloc_snapshot();
    let t4 = Instant::now();
    for i in 0..N {
        let tok = (i % 200) as Token + 3;
        sink += model
            .token_to_piece(tok)
            .map(|p| p.len() as i64)
            .unwrap_or(0);
    }
    let t2p = t4.elapsed();
    let (a1, b1) = alloc_snapshot();

    println!(
        "sampler cost model={} ({} calls each; per-call)",
        path.display(),
        N
    );
    println!(
        "  sample_greedy (session chain)  {:9.2} us",
        ms(greedy) * 1e3 / N as f64
    );
    println!(
        "  sample_ith(-1) session chain   {:9.2} us",
        ms(ith) * 1e3 / N as f64
    );
    println!(
        "  sample_ith_with greedy chain   {:9.2} us",
        ms(ith_with) * 1e3 / N as f64
    );
    println!(
        "  sample_ith_with dist chain     {:9.2} us",
        ms(dist) * 1e3 / N as f64
    );
    println!(
        "  token_to_piece                 {:9.2} us  ({:.1} allocs, {:.0} B alloc'd per call)",
        ms(t2p) * 1e3 / N as f64,
        (a1 - a0) as f64 / N as f64,
        (b1 - b0) as f64 / N as f64
    );
    let _ = sink;
}

// ---- generate() vs host-style step loop (single request) ----

fn cmd_steploop(path: &Path, gen_n: usize) {
    let model = load(path);
    let prompt = make_prompt(&model, 64);

    // A: pre-M7 direct loop — Session::generate on the default session
    //    (n_seq_max=1, kv_unified=false).
    let mut plain = Session::new(&model, &SessionParams::default()).expect("plain session");
    plain
        .generate(&prompt, 8, |_, _| std::ops::ControlFlow::Continue(()))
        .expect("warmup");
    plain.reset();
    let (pa0, _pb0) = alloc_snapshot();
    let stats = plain
        .generate(&prompt, gen_n, |_, _| std::ops::ControlFlow::Continue(()))
        .expect("generate");
    // decode_ms excludes the prefill, matching loop C's measured window.
    let a_ms = stats.decode_ms;
    let (pa1, _pb1) = alloc_snapshot();
    let a_tokens = stats.generated_tokens.max(1);
    drop(plain);

    // B: generate() on the DAEMON-shaped session (kv_unified, n_seq_max=4)
    //    — isolates the KV-layout cost from the machinery cost.
    let daemon_params = SessionParams {
        n_seq_max: 4,
        kv_unified: true,
        ..SessionParams::default()
    };
    let mut uni = Session::new(&model, &daemon_params).expect("unified session");
    uni.generate(&prompt, 8, |_, _| std::ops::ControlFlow::Continue(()))
        .expect("warmup");
    uni.reset();
    let stats_b = uni
        .generate(&prompt, gen_n, |_, _| std::ops::ControlFlow::Continue(()))
        .expect("generate unified");
    let b_ms = stats_b.decode_ms;
    let b_tokens = stats_b.generated_tokens.max(1);
    uni.reset();

    // C: emulated host step loop on the same daemon-shaped session — one
    //    active sequence stepped through Batch::push/decode_batch/
    //    sample_ith_with/token_to_piece/try_send plus the serve loop's
    //    per-iteration bookkeeping (empty sweeps, channel poll, disconnect
    //    check), faithfully reproducing engine_host.rs steps 1-9 for the
    //    single-request case.
    let mut batch = Batch::new(512, 4).expect("batch");
    let mut chain = Sampler::new(&SamplerParams {
        temperature: 0.0,
        ..SamplerParams::default()
    })
    .expect("chain");
    let (tx, rx) = std::sync::mpsc::sync_channel::<String>(256);
    let drain = thread::spawn(move || while rx.recv().is_ok() {});
    let (ctl_tx, ctl_rx) = std::sync::mpsc::channel::<u8>(); // stands in for HostMsg rx
    let mut held: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let mut outbox: Vec<Option<String>> = Vec::new();

    // Prefill via the step loop's own path: chunked pushes with the tail
    // carrying logits, exactly like serve_model's prefill arm.
    let mut kv_len = 0usize;
    let mut prefill_done = 0usize;
    let mut pending: Option<(Token, String)> = None;
    let seq = 0;
    while prefill_done < prompt.len() {
        batch.clear();
        let end = (prefill_done + 512).min(prompt.len());
        let mut tail_index = None;
        for (j, &tok) in prompt[prefill_done..end].iter().enumerate() {
            let is_tail = prefill_done + j + 1 == prompt.len();
            let idx = batch
                .push(tok, (kv_len + j) as i32, seq, is_tail)
                .expect("push");
            if is_tail {
                tail_index = Some(idx);
            }
        }
        uni.decode_batch(&batch).expect("prefill chunk");
        kv_len += end - prefill_done;
        prefill_done = end;
        if prefill_done == prompt.len() {
            let idx = tail_index.expect("tail");
            let first = uni.sample_ith_with(&mut chain, idx as i32);
            let piece = model.token_to_piece(first).expect("piece");
            pending = Some((first, piece));
        }
    }

    let (ca0, _cb0) = alloc_snapshot();
    let t2 = Instant::now();
    let mut generated = 0usize;
    while generated < gen_n {
        // Serve-loop bookkeeping the real host runs every iteration.
        outbox.retain_mut(|p| p.is_none());
        while let Some(p) = held.pop_front() {
            let _ = tx.try_send(p);
        }
        match ctl_rx.try_recv() {
            Ok(_) | Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
        }
        //

        batch.clear();
        let (tok, piece) = pending.take().expect("pending");
        let index = batch.push(tok, kv_len as i32, seq, true).expect("push");
        uni.decode_batch(&batch).expect("decode step");
        kv_len += 1;
        generated += 1;
        let _ = model.is_eog(tok);
        let _ = tx.try_send(piece);
        let next = uni.sample_ith_with(&mut chain, index as i32);
        let _ = model.is_eog(next);
        let piece = model.token_to_piece(next).expect("piece");
        pending = Some((next, piece));
    }
    let c_wall = t2.elapsed();
    let (ca1, _cb1) = alloc_snapshot();
    drop(ctl_tx);
    drop(tx);
    let _ = drain.join();

    println!("steploop model={} gen={}", path.display(), gen_n);
    println!(
        "  A generate() n_seq=1 !unified   {:8.3} ms/tok  ({:.1} allocs/tok)",
        a_ms / a_tokens as f64,
        (pa1 - pa0) as f64 / a_tokens as f64
    );
    println!(
        "  B generate() n_seq=4 unified    {:8.3} ms/tok   <- KV-layout delta vs A",
        b_ms / b_tokens as f64
    );
    println!(
        "  C host-style step loop (same)   {:8.3} ms/tok  ({:.1} allocs/tok)   <- machinery delta vs B",
        ms(c_wall) / generated.max(1) as f64,
        (ca1 - ca0) as f64 / generated.max(1) as f64
    );
}

// ---- distributed loopback with RPC command counting ----

const CMD_NAMES: [&str; 18] = [
    "ALLOC_BUFFER",
    "GET_ALIGNMENT",
    "GET_MAX_SIZE",
    "BUFFER_GET_BASE",
    "FREE_BUFFER",
    "BUFFER_CLEAR",
    "SET_TENSOR",
    "SET_TENSOR_HASH",
    "GET_TENSOR",
    "COPY_TENSOR",
    "GRAPH_COMPUTE",
    "GET_DEVICE_MEMORY",
    "INIT_TENSOR",
    "GET_ALLOC_SIZE",
    "HELLO",
    "DEVICE_COUNT",
    "GRAPH_RECOMPUTE",
    "MEMSET_TENSOR",
];

#[derive(Default)]
struct RpcCounters {
    cmds: [AtomicU64; 32],
    c2s_bytes: AtomicU64,
    s2c_bytes: AtomicU64,
    s2c_msgs: AtomicU64,
}

impl RpcCounters {
    fn snapshot(&self) -> (Vec<u64>, u64, u64, u64) {
        (
            self.cmds
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect(),
            self.c2s_bytes.load(Ordering::Relaxed),
            self.s2c_bytes.load(Ordering::Relaxed),
            self.s2c_msgs.load(Ordering::Relaxed),
        )
    }
}

fn read_exact_or_eof(s: &mut impl Read, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut read = 0;
    while read < buf.len() {
        match s.read(&mut buf[read..]) {
            Ok(0) => return Ok(false),
            Ok(n) => read += n,
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

/// Relay client->server traffic, parsing the RPC framing
/// (| cmd 1B | size 8B | payload |) to count commands.
fn counting_relay_c2s(mut from: TcpStream, mut to: TcpStream, ctr: Arc<RpcCounters>) {
    let mut hdr = [0u8; 9];
    let mut payload = Vec::new();
    while let Ok(true) = read_exact_or_eof(&mut from, &mut hdr) {
        let cmd = hdr[0] as usize;
        let size = u64::from_le_bytes(hdr[1..9].try_into().unwrap()) as usize;
        if cmd < 32 {
            ctr.cmds[cmd].fetch_add(1, Ordering::Relaxed);
        }
        ctr.c2s_bytes.fetch_add(9 + size as u64, Ordering::Relaxed);
        payload.resize(size, 0);
        if to.write_all(&hdr).is_err() {
            break;
        }
        if size > 0 {
            match read_exact_or_eof(&mut from, &mut payload) {
                Ok(true) => {}
                _ => break,
            }
            if to.write_all(&payload).is_err() {
                break;
            }
        }
        let _ = to.flush();
    }
    let _ = to.shutdown(std::net::Shutdown::Write);
}

/// Relay server->client traffic, parsing | size 8B | payload | responses.
fn counting_relay_s2c(mut from: TcpStream, mut to: TcpStream, ctr: Arc<RpcCounters>) {
    let mut hdr = [0u8; 8];
    let mut payload = Vec::new();
    while let Ok(true) = read_exact_or_eof(&mut from, &mut hdr) {
        let size = u64::from_le_bytes(hdr) as usize;
        ctr.s2c_msgs.fetch_add(1, Ordering::Relaxed);
        ctr.s2c_bytes.fetch_add(8 + size as u64, Ordering::Relaxed);
        payload.resize(size, 0);
        if to.write_all(&hdr).is_err() {
            break;
        }
        if size > 0 {
            match read_exact_or_eof(&mut from, &mut payload) {
                Ok(true) => {}
                _ => break,
            }
            if to.write_all(&payload).is_err() {
                break;
            }
        }
        let _ = to.flush();
    }
    let _ = to.shutdown(std::net::Shutdown::Write);
}

struct CountingBridge {
    port: u16,
    stop: Arc<AtomicBool>,
    acceptor: thread::JoinHandle<Vec<RpcServeSession>>,
}

impl CountingBridge {
    fn spawn(dev_index: i32, ctr: Arc<RpcCounters>) -> CountingBridge {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let port = listener.local_addr().expect("addr").port();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        let acceptor = thread::spawn(move || {
            let mut sessions = Vec::new();
            while !stop_flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((conn, _)) => {
                        conn.set_nonblocking(false).expect("blocking");
                        conn.set_nodelay(true).ok();
                        let (raw, bridge) = SocketPair::new().expect("pair").into_parts();
                        let serve = RpcServeSession::spawn(raw, 0, dev_index, None).expect("serve");
                        let conn_r = conn.try_clone().expect("clone");
                        let bridge_w = bridge.try_clone().expect("clone");
                        let c1 = Arc::clone(&ctr);
                        let c2 = Arc::clone(&ctr);
                        thread::spawn(move || counting_relay_c2s(conn_r, bridge_w, c1));
                        thread::spawn(move || counting_relay_s2c(bridge, conn, c2));
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
        CountingBridge {
            port,
            stop,
            acceptor,
        }
    }

    fn finish(self) {
        self.stop.store(true, Ordering::Relaxed);
        let sessions = self.acceptor.join().expect("acceptor");
        for serve in sessions {
            let _ = serve.join_timeout(Duration::from_secs(10));
        }
    }
}

fn print_cmd_delta(label: &str, before: &[u64], after: &[u64], per: f64) {
    print!("  {label}: ");
    let mut first = true;
    for (i, name) in CMD_NAMES.iter().enumerate() {
        let d = after[i].saturating_sub(before[i]);
        if d > 0 {
            if !first {
                print!(", ");
            }
            print!("{name} {d}");
            if per > 0.0 {
                print!(" ({:.2}/tok)", d as f64 / per);
            }
            first = false;
        }
    }
    println!();
}

fn cmd_dist(path: &Path, prompt_n: usize, gen_n: usize) {
    let cpu = devices()
        .iter()
        .position(|d| matches!(d.kind, DeviceKind::Cpu))
        .expect("cpu device") as i32;
    let ctr = Arc::new(RpcCounters::default());
    let bridge = CountingBridge::spawn(cpu, Arc::clone(&ctr));
    let endpoint = format!("127.0.0.1:{}", bridge.port);
    let server = RemoteServer::register(&endpoint).expect("register");

    let t0 = Instant::now();
    let model =
        Model::load_distributed(path, &[&server], &[0.5, 0.5], true, &ModelParams::default())
            .expect("distributed load");
    let load_wall = t0.elapsed();
    let mut session = Session::new(&model, &SessionParams::default()).expect("session");
    let prompt = make_prompt(&model, prompt_n);

    let s_load = ctr.snapshot();
    let t1 = Instant::now();
    session.decode(&prompt).expect("prefill");
    let prefill_wall = t1.elapsed();
    let s_prefill = ctr.snapshot();

    // Warmup decode steps (graph build, first RECOMPUTE) before measuring.
    for _ in 0..4 {
        let t = session.sample_greedy();
        session.decode(&[t]).expect("warmup");
    }
    let s_warm = ctr.snapshot();
    let mut per_tok = Vec::new();
    let t2 = Instant::now();
    for _ in 0..gen_n {
        let it = Instant::now();
        let tok = session.sample_greedy();
        session.decode(&[tok]).expect("decode");
        per_tok.push(ms(it.elapsed()));
    }
    let decode_wall = t2.elapsed();
    let s_decode = ctr.snapshot();

    drop(session);
    drop(model);
    bridge.finish();

    let dist_median = median(per_tok.clone());
    println!(
        "dist loopback model={} prompt={} gen={}",
        path.display(),
        prompt_n,
        gen_n
    );
    println!(
        "  load {:.0} ms  prefill {:.1} ms  decode {:.3} ms/tok (median {:.3}, p90 {:.3})",
        ms(load_wall),
        ms(prefill_wall),
        ms(decode_wall) / gen_n as f64,
        dist_median,
        pct(per_tok, 0.9),
    );
    print_cmd_delta("prefill cmds", &s_load.0, &s_prefill.0, 0.0);
    println!(
        "    prefill wire: {:.2} MB c2s, {:.2} MB s2c, {} responses",
        (s_prefill.1 - s_load.1) as f64 / 1e6,
        (s_prefill.2 - s_load.2) as f64 / 1e6,
        s_prefill.3 - s_load.3
    );
    print_cmd_delta("decode cmds", &s_warm.0, &s_decode.0, gen_n as f64);
    println!(
        "    decode wire: {:.1} KB/tok c2s, {:.1} KB/tok s2c, {:.2} responses(=blocking round trips)/tok",
        (s_decode.1 - s_warm.1) as f64 / 1e3 / gen_n as f64,
        (s_decode.2 - s_warm.2) as f64 / 1e3 / gen_n as f64,
        (s_decode.3 - s_warm.3) as f64 / gen_n as f64
    );
    let gc = s_decode.0[10] - s_warm.0[10];
    let gr = s_decode.0[16] - s_warm.0[16];
    if gc + gr > 0 {
        println!(
            "    graph reuse hit rate: {:.1}% ({} RECOMPUTE vs {} full GRAPH_COMPUTE)",
            100.0 * gr as f64 / (gc + gr) as f64,
            gr,
            gc
        );
    }

    // Same-run solo A/B on the same model file.
    let solo_model = load(path);
    let mut solo = Session::new(&solo_model, &SessionParams::default()).expect("solo session");
    solo.decode(&prompt).expect("solo prefill warm");
    for _ in 0..4 {
        let t = solo.sample_greedy();
        solo.decode(&[t]).expect("warm");
    }
    solo.reset();
    let t3 = Instant::now();
    solo.decode(&prompt).expect("solo prefill");
    let solo_prefill = t3.elapsed();
    let mut solo_tok = Vec::new();
    for _ in 0..gen_n {
        let it = Instant::now();
        let t = solo.sample_greedy();
        solo.decode(&[t]).expect("solo decode");
        solo_tok.push(ms(it.elapsed()));
    }
    let solo_median = median(solo_tok);
    println!(
        "  solo A/B same model: prefill {:.1} ms, decode median {:.3} ms/tok -> distributed overhead {:+.3} ms/tok",
        ms(solo_prefill),
        solo_median,
        dist_median - solo_median
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "latbench <synth|solo|ttft|threads|axes|sampler|steploop|dist> ...";
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    match cmd {
        "synth" => {
            let out = PathBuf::from(args.get(2).expect("synth <out.gguf>"));
            let bytes = build_mid_model();
            std::fs::write(&out, &bytes).expect("write synth model");
            println!(
                "wrote {} ({:.1} MB)",
                out.display(),
                bytes.len() as f64 / 1e6
            );
        }
        "solo" => {
            let path = PathBuf::from(args.get(2).expect("solo <model>"));
            let p = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
            let g = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
            cmd_solo(&path, p, g);
        }
        "ttft" => {
            let path = PathBuf::from(args.get(2).expect("ttft <model>"));
            let p = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
            cmd_ttft(&path, p);
        }
        "genttft" => {
            let path = PathBuf::from(args.get(2).expect("genttft <model>"));
            let p = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
            cmd_genttft(&path, p);
        }
        "threads" => {
            let path = PathBuf::from(args.get(2).expect("threads <model>"));
            // Each entry is `n` (tied decode+batch, pre-widening behavior)
            // or `d:b` (separate decode:batch counts, e.g. `8:24`); `det`
            // takes the cpu module's detected recommendation.
            let counts: Vec<(i32, i32)> = args
                .get(3)
                .map(|s| {
                    s.split(',')
                        .filter_map(|x| {
                            if x == "det" {
                                let r = onebrain_engine::cpu::recommended_threads();
                                return Some((r.n_threads, r.n_threads_batch));
                            }
                            match x.split_once(':') {
                                Some((d, b)) => Some((d.parse().ok()?, b.parse().ok()?)),
                                None => x.parse().ok().map(|n| (n, 0)),
                            }
                        })
                        .collect()
                })
                .unwrap_or_else(|| {
                    vec![
                        (0, 0),
                        (1, 0),
                        (4, 0),
                        (8, 0),
                        (12, 0),
                        (16, 0),
                        (20, 0),
                        (24, 0),
                    ]
                });
            cmd_threads(&path, &counts);
        }
        "axes" => {
            let path = PathBuf::from(args.get(2).expect("axes <model>"));
            cmd_axes(&path);
        }
        "sampler" => {
            let path = PathBuf::from(args.get(2).expect("sampler <model>"));
            cmd_sampler(&path);
        }
        "steploop" => {
            let path = PathBuf::from(args.get(2).expect("steploop <model>"));
            let g = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
            cmd_steploop(&path, g);
        }
        "dist" => {
            let path = PathBuf::from(args.get(2).expect("dist <model>"));
            let p = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
            let g = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
            cmd_dist(&path, p, g);
        }
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    }
}
