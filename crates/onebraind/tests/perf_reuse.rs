//! Prefix/KV-reuse and speculative-counter proofs (docs/perf.md §4/§5),
//! asserted through the daemon's own perf log line —
//! `perf: prefill {n}tok {ms}ms decode {n}tok {ms}ms ttft {ms}ms drafted {n} accepted {n}`
//! — exactly the instrument the sim greps. This lives in its own
//! integration binary so the process-global tracing subscriber belongs to
//! these tests alone; a mutex serializes them so captured lines are
//! attributable.
//!
//! Every test needs `OB_SMOKE_MODEL` (the stories260K smoke model) and
//! quietly skips without it, matching the engine-crate convention.

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use onebrain_api::backend::{DoneStats, GenParams, GenerateJob, PromptInput, TokenEvent};
use onebraind::engine_host::{
    DraftRequest, EngineHost, GenOutcome, HostMsg, HostPerf, SupervisedGenerate,
};
use tokio::sync::{mpsc, oneshot};

// ---------------------------------------------------------------------------
// Log capture: one process-global subscriber writing into a shared buffer.
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<String>>);

impl Capture {
    fn len(&self) -> usize {
        self.0.lock().expect("capture poisoned").len()
    }

    /// Everything captured after byte offset `mark`.
    fn since(&self, mark: usize) -> String {
        self.0.lock().expect("capture poisoned")[mark..].to_string()
    }
}

struct CaptureWriter(Capture);

impl std::io::Write for CaptureWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
             .0
            .lock()
            .expect("capture poisoned")
            .push_str(&String::from_utf8_lossy(buf));
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The capture buffer, installing the global subscriber on first use.
fn capture() -> &'static Capture {
    static CAPTURE: OnceLock<Capture> = OnceLock::new();
    CAPTURE.get_or_init(|| {
        let cap = Capture::default();
        let writer_cap = cap.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .with_writer(move || CaptureWriter(writer_cap.clone()))
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("this test binary owns the global subscriber");
        cap
    })
}

/// Serializes the tests so each one's captured log region is its own.
fn test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    // A prior panic while holding the lock does not invalidate the guarded
    // resource (each test spawns fresh hosts); clear the poison.
    match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// One parsed perf log line.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PerfLine {
    prefill: usize,
    decode: usize,
    drafted: u64,
    accepted: u64,
}

/// Parse every perf line in a captured chunk, in emission order.
fn perf_lines(chunk: &str) -> Vec<PerfLine> {
    chunk
        .lines()
        .filter_map(|line| {
            let rest = line.split("perf: prefill ").nth(1)?;
            let prefill: usize = rest.split("tok").next()?.trim().parse().ok()?;
            let decode_part = rest.split(" decode ").nth(1)?;
            let decode: usize = decode_part.split("tok").next()?.trim().parse().ok()?;
            let drafted_part = rest.split(" drafted ").nth(1)?;
            let mut words = drafted_part.split_whitespace();
            let drafted: u64 = words.next()?.parse().ok()?;
            let accepted_word = words.next()?;
            if accepted_word != "accepted" {
                return None;
            }
            let accepted: u64 = words.next()?.parse().ok()?;
            Some(PerfLine {
                prefill,
                decode,
                drafted,
                accepted,
            })
        })
        .collect()
}

/// The single perf line a chunk must contain.
fn one_perf_line(chunk: &str) -> PerfLine {
    let lines = perf_lines(chunk);
    assert_eq!(
        lines.len(),
        1,
        "expected exactly one perf line, got {lines:?} in:\n{chunk}"
    );
    lines[0]
}

// ---------------------------------------------------------------------------
// Host helpers (public API only — this is an integration test).
// ---------------------------------------------------------------------------

/// Long enough to clear the 64-token reuse floor on the smoke model's
/// tokenizer by a wide margin (asserted by the tests, not assumed).
const LONG_PROMPT: &str = "Once upon a time there was a little dog named Rex \
    who loved to play in the park with a red ball. Once upon a time there \
    was a little dog named Rex who loved to play in the park with a red \
    ball. Once upon a time there was a little dog named Rex who loved to \
    play in the park with a red ball.";

fn smoke_model() -> Option<String> {
    match std::env::var("OB_SMOKE_MODEL") {
        Ok(path) => Some(path),
        Err(_) => {
            eprintln!("OB_SMOKE_MODEL not set; skipping perf-reuse test");
            None
        }
    }
}

fn load(host: &EngineHost, reference: &str, draft: bool) {
    let (ptx, _prx) = mpsc::unbounded_channel();
    let (rtx, rrx) = oneshot::channel();
    host.send(HostMsg::Load {
        reference: reference.to_string(),
        cache_root: std::env::temp_dir(),
        ctx_len: 512,
        draft: draft.then(|| DraftRequest {
            reference: reference.to_string(),
            cache_root: std::env::temp_dir(),
        }),
        progress: ptx,
        resp: rtx,
    })
    .expect("host accepts the load");
    rrx.blocking_recv()
        .expect("host answers")
        .expect("smoke model loads");
}

fn spawn_loaded(
    perf: HostPerf,
    draft: bool,
) -> Option<(EngineHost, std::thread::JoinHandle<()>, String)> {
    let smoke = smoke_model()?;
    let (host, handle) = EngineHost::spawn(None, perf);
    load(&host, &smoke, draft);
    Some((host, handle, smoke))
}

/// Drive one greedy generation to Done, returning its text and stats.
fn run(host: &EngineHost, model: &str, prompt: &str, max_tokens: u32) -> (String, DoneStats) {
    let (tx, mut rx) = mpsc::channel(64);
    let (otx, orx) = oneshot::channel();
    host.send(HostMsg::Generate(SupervisedGenerate {
        job: GenerateJob {
            model: model.to_string(),
            prompt: PromptInput::Raw(prompt.to_string()),
            params: GenParams {
                max_tokens,
                temperature: 0.0,
                ..Default::default()
            },
            tx,
        },
        resume: None,
        outcome: otx,
    }))
    .expect("host accepts the job");
    let mut text = String::new();
    let done = loop {
        match rx.blocking_recv().expect("stream must terminate") {
            TokenEvent::Token(piece) => text.push_str(&piece),
            TokenEvent::Done(stats) => break stats,
            TokenEvent::Error(e) => panic!("unexpected error: {e}"),
        }
    };
    assert!(matches!(
        orx.blocking_recv().expect("outcome must arrive"),
        GenOutcome::Finished
    ));
    (text, done)
}

fn shutdown(host: EngineHost, handle: std::thread::JoinHandle<()>) {
    host.send(HostMsg::Shutdown).expect("host accepts shutdown");
    handle.join().expect("host thread joins");
}

// ---------------------------------------------------------------------------
// §4 proofs
// ---------------------------------------------------------------------------

/// The §4 headline (docs/perf.md): a second identical request prefills
/// exactly the suffix — here one token, the re-decoded prefix tail — and
/// its output is byte-identical to the cold run.
#[test]
fn reuse_hit_prefills_only_the_suffix_and_matches_cold() {
    let _guard = test_lock();
    let cap = capture();
    let Some((host, handle, smoke)) = spawn_loaded(
        HostPerf {
            max_concurrent: 1,
            ..HostPerf::default()
        },
        false,
    ) else {
        return;
    };
    const MAX: u32 = 8;

    let mark = cap.len();
    let (cold_text, _) = run(&host, &smoke, LONG_PROMPT, MAX);
    let cold = one_perf_line(&cap.since(mark));
    assert!(
        cold.prefill >= 65,
        "premise: the long prompt must clear the 64-token floor, got {}",
        cold.prefill
    );

    let mark = cap.len();
    let (warm_text, _) = run(&host, &smoke, LONG_PROMPT, MAX);
    let warm = one_perf_line(&cap.since(mark));
    assert_eq!(
        warm.prefill, 1,
        "an identical prompt reuses everything but the re-decoded tail"
    );
    assert_eq!(warm.decode, cold.decode, "same budget, same decode count");
    assert_eq!(
        warm_text, cold_text,
        "greedy output must be byte-identical to the cold run"
    );
    shutdown(host, handle);
}

/// Below the 64-token floor the slot resets: both runs prefill the full
/// prompt (and stay byte-identical).
#[test]
fn below_floor_prefills_fully() {
    let _guard = test_lock();
    let cap = capture();
    let Some((host, handle, smoke)) = spawn_loaded(
        HostPerf {
            max_concurrent: 1,
            ..HostPerf::default()
        },
        false,
    ) else {
        return;
    };
    let prompt = "Once upon a time";

    let mark = cap.len();
    let (first_text, _) = run(&host, &smoke, prompt, 8);
    let first = one_perf_line(&cap.since(mark));
    assert!(
        first.prefill < 64,
        "premise: this prompt must sit below the floor, got {}",
        first.prefill
    );

    let mark = cap.len();
    let (second_text, _) = run(&host, &smoke, prompt, 8);
    let second = one_perf_line(&cap.since(mark));
    assert_eq!(
        second.prefill, first.prefill,
        "below the floor the full prompt prefills again"
    );
    assert_eq!(second_text, first_text);
    shutdown(host, handle);
}

/// `[perf] kv_reuse = false` restores the reset-per-request behavior:
/// identical prompts always prefill fully.
#[test]
fn kv_reuse_off_prefills_fully() {
    let _guard = test_lock();
    let cap = capture();
    let Some((host, handle, smoke)) = spawn_loaded(
        HostPerf {
            max_concurrent: 1,
            kv_reuse: false,
            ..HostPerf::default()
        },
        false,
    ) else {
        return;
    };

    let mark = cap.len();
    let (first_text, _) = run(&host, &smoke, LONG_PROMPT, 8);
    let first = one_perf_line(&cap.since(mark));

    let mark = cap.len();
    let (second_text, _) = run(&host, &smoke, LONG_PROMPT, 8);
    let second = one_perf_line(&cap.since(mark));
    assert_eq!(
        second.prefill, first.prefill,
        "kv_reuse=false must full-prefill every request"
    );
    assert_eq!(second_text, first_text);
    shutdown(host, handle);
}

/// docs/perf.md §4 interactions: a model swap resets the reuse state — the
/// next identical request full-prefills (and still matches the cold text).
#[test]
fn model_swap_invalidates_retained_prefixes() {
    let _guard = test_lock();
    let cap = capture();
    let Some((host, handle, smoke)) = spawn_loaded(
        HostPerf {
            max_concurrent: 1,
            ..HostPerf::default()
        },
        false,
    ) else {
        return;
    };
    const MAX: u32 = 8;

    let mark = cap.len();
    let (cold_text, _) = run(&host, &smoke, LONG_PROMPT, MAX);
    let cold = one_perf_line(&cap.since(mark));

    // Prove the cache is live first (guards against a silently-broken
    // reuse path making the swap assert vacuous).
    let mark = cap.len();
    let _ = run(&host, &smoke, LONG_PROMPT, MAX);
    assert_eq!(one_perf_line(&cap.since(mark)).prefill, 1);

    // Swap: loading again (same reference) drops the session + KV.
    load(&host, &smoke, false);

    let mark = cap.len();
    let (after_text, _) = run(&host, &smoke, LONG_PROMPT, MAX);
    let after = one_perf_line(&cap.since(mark));
    assert_eq!(
        after.prefill, cold.prefill,
        "a model swap must invalidate every retained prefix"
    );
    assert_eq!(after_text, cold_text);
    shutdown(host, handle);
}

/// A diverging continuation reuses the shared prefix only: the warm run
/// prefills fewer tokens than a cold run of the same prompt and produces
/// byte-identical output to it.
#[test]
fn diverging_suffix_reuses_the_shared_prefix() {
    let _guard = test_lock();
    let cap = capture();
    let Some((host, handle, smoke)) = spawn_loaded(
        HostPerf {
            max_concurrent: 1,
            ..HostPerf::default()
        },
        false,
    ) else {
        return;
    };
    const MAX: u32 = 8;
    let prompt_a = format!("{LONG_PROMPT} The dog was very happy that day.");
    let prompt_b = format!("{LONG_PROMPT} Suddenly a big cat appeared there.");

    let mark = cap.len();
    let _ = run(&host, &smoke, &prompt_a, MAX);
    let cold_a = one_perf_line(&cap.since(mark));
    assert!(cold_a.prefill >= 65, "premise: floor cleared");

    // Warm B: shares LONG_PROMPT with A's retained history, diverges after.
    let mark = cap.len();
    let (warm_b_text, _) = run(&host, &smoke, &prompt_b, MAX);
    let warm_b = one_perf_line(&cap.since(mark));

    // Cold B control: swap the model (resets the cache), run B again.
    load(&host, &smoke, false);
    let mark = cap.len();
    let (cold_b_text, _) = run(&host, &smoke, &prompt_b, MAX);
    let cold_b = one_perf_line(&cap.since(mark));

    assert!(
        warm_b.prefill < cold_b.prefill,
        "the warm run must prefill only the divergent suffix \
         (warm {} vs cold {})",
        warm_b.prefill,
        cold_b.prefill
    );
    assert_eq!(
        warm_b_text, cold_b_text,
        "prefix reuse must not change greedy output"
    );
    shutdown(host, handle);
}

// ---------------------------------------------------------------------------
// §5 counters on the log line
// ---------------------------------------------------------------------------

/// docs/perf.md §5: `drafted`/`accepted` ride the perf log line (the sim
/// asserts acceptance > 0 through exactly this).
#[test]
fn speculative_counters_reach_the_perf_line() {
    let _guard = test_lock();
    let cap = capture();
    let Some((host, handle, smoke)) = spawn_loaded(HostPerf::default(), /* draft */ true) else {
        return;
    };
    let mark = cap.len();
    let (_, stats) = run(&host, &smoke, "Once upon a time", 16);
    let line = one_perf_line(&cap.since(mark));
    assert!(line.drafted > 0, "perf line must carry drafted: {line:?}");
    assert!(line.accepted > 0, "perf line must carry accepted: {line:?}");
    assert_eq!(line.drafted, u64::from(stats.drafted), "line == DoneStats");
    assert_eq!(line.accepted, u64::from(stats.accepted));
    shutdown(host, handle);
}
