//! Batch/sequence substrate tests (docs/perf.md §2): explicit multi-
//! sequence batches, per-index sampling, and seq_rm rollback against the
//! smoke model. Everything is gated on OB_SMOKE_MODEL like the other
//! engine integration tests (CI downloads it; `cargo xtask smoke` wires it
//! locally).
//!
//! The alone-vs-batched byte-equality assertion is the §6 primary assert.
//! If CPU batching ever breaks it (batching changes GEMM shapes, so FP
//! rounding CAN differ), the contract's fallback applies: record the
//! measured divergence in docs/perf.md's appendix and downgrade ONLY that
//! assert to run-to-run determinism — never delete it silently.

use std::path::Path;

use onebrain_engine::{Batch, Model, ModelParams, SeqToken, Session, SessionParams, Token};

fn smoke_model(test: &str) -> Option<Model> {
    let Ok(path) = std::env::var("OB_SMOKE_MODEL") else {
        eprintln!("OB_SMOKE_MODEL not set; skipping {test}");
        return None;
    };
    Some(Model::load(Path::new(&path), &ModelParams::default()).expect("smoke model loads"))
}

/// Ground truth: the prompt decoded alone in a fresh single-sequence
/// session via the pre-M7 chunked-decode path.
fn alone_greedy(model: &Model, prompt: &[Token], max_new: usize) -> Vec<Token> {
    let mut session = Session::new(
        model,
        &SessionParams {
            n_ctx: 256,
            n_batch: 128,
            ..SessionParams::default()
        },
    )
    .expect("solo session");
    session
        .generate_greedy(prompt, max_new, |_, _| {})
        .expect("solo greedy generation")
}

/// Push a whole prompt onto `seq` starting at position 0, flagging logits
/// only on the last token; returns that token's batch index.
fn push_prompt(batch: &mut Batch, prompt: &[Token], seq: i32) -> usize {
    let mut last = 0;
    for (i, &tok) in prompt.iter().enumerate() {
        last = batch
            .push(tok, i as i32, seq, i + 1 == prompt.len())
            .expect("push prompt token");
    }
    last
}

/// Two prompts prefilled and greedily decoded in ONE batch per step must
/// reproduce, per sequence, the alone-run greedy tokens — and sample_ith
/// must map each logits index to its own sequence (the ordering assert:
/// the two continuations differ at the first token, so a swapped index
/// cannot pass).
#[test]
fn two_sequences_in_one_batch_match_alone_runs() {
    let Some(model) = smoke_model("two_sequences_in_one_batch_match_alone_runs") else {
        return;
    };
    const MAX_NEW: usize = 8;
    let prompt_a = model.tokenize("Once upon a time", true).unwrap();
    let prompt_b = model.tokenize("The little dog", true).unwrap();
    let alone_a = alone_greedy(&model, &prompt_a, MAX_NEW);
    let alone_b = alone_greedy(&model, &prompt_b, MAX_NEW);
    assert_eq!(
        alone_a.len(),
        MAX_NEW,
        "prompt A hit EOG too early for this test"
    );
    assert_eq!(
        alone_b.len(),
        MAX_NEW,
        "prompt B hit EOG too early for this test"
    );
    assert_ne!(
        alone_a[0], alone_b[0],
        "test precondition: the prompts must have different continuations \
         so a sample_ith index mixup cannot pass"
    );

    let mut session = Session::new(
        &model,
        &SessionParams {
            n_ctx: 256,
            n_batch: 128,
            n_seq_max: 2,
            kv_unified: true,
            ..SessionParams::default()
        },
    )
    .expect("multi-sequence session");
    assert_eq!(session.n_seq_max(), 2);

    // Joint prefill: both prompts in one batch, logits on each last token.
    let mut batch = Batch::new(128, 2).expect("batch alloc");
    let ia = push_prompt(&mut batch, &prompt_a, 0);
    let ib = push_prompt(&mut batch, &prompt_b, 1);
    session.decode_batch(&batch).expect("joint prefill decode");

    // sample_ith ordering: each flagged index continues ITS sequence.
    let mut tok_a = session.sample_ith(ia as i32);
    let mut tok_b = session.sample_ith(ib as i32);
    assert_eq!(
        tok_a, alone_a[0],
        "sample_ith(last A index) must continue A"
    );
    assert_eq!(
        tok_b, alone_b[0],
        "sample_ith(last B index) must continue B"
    );

    // Greedy decode both sequences, one token per sequence per batch.
    let mut got_a = vec![tok_a];
    let mut got_b = vec![tok_b];
    for _ in 1..MAX_NEW {
        let next = session
            .decode_step(
                &mut batch,
                &[
                    SeqToken {
                        seq_id: 0,
                        token: tok_a,
                    },
                    SeqToken {
                        seq_id: 1,
                        token: tok_b,
                    },
                ],
            )
            .expect("multi-sequence decode step");
        tok_a = next[0];
        tok_b = next[1];
        got_a.push(tok_a);
        got_b.push(tok_b);
    }
    assert_eq!(
        got_a, alone_a,
        "seq 0 batched greedy must match its alone run"
    );
    assert_eq!(
        got_b, alone_b,
        "seq 1 batched greedy must match its alone run"
    );
}

/// Rollback proof: decode junk past the prompt, seq_rm it (plus the last
/// prompt token, whose logits the junk decode overwrote), re-decode that
/// token, and greedily continue — the result must be byte-identical to a
/// straight-line run that never took the detour. This is the §4 KV-reuse /
/// §5 speculative-reject primitive.
#[test]
fn seq_rm_rollback_rejoins_straight_line_run() {
    let Some(model) = smoke_model("seq_rm_rollback_rejoins_straight_line_run") else {
        return;
    };
    const MAX_NEW: usize = 8;
    let prompt = model.tokenize("Once upon a time", true).unwrap();
    let plen = prompt.len() as i32;
    assert!(plen >= 2, "rollback test needs a multi-token prompt");
    let straight = alone_greedy(&model, &prompt, MAX_NEW);
    assert_eq!(
        straight.len(),
        MAX_NEW,
        "prompt hit EOG too early for this test"
    );

    let mut session = Session::new(
        &model,
        &SessionParams {
            n_ctx: 256,
            n_batch: 128,
            ..SessionParams::default()
        },
    )
    .expect("session");

    // Prefill the prompt on sequence 0 through the explicit batch API.
    let mut batch = Batch::new(128, 1).expect("batch alloc");
    push_prompt(&mut batch, &prompt, 0);
    session.decode_batch(&batch).expect("prefill decode");
    assert_eq!(session.seq_pos_max(0), Some(plen - 1));

    // Pollute: decode 4 junk tokens (re-used prompt ids are valid vocab)
    // at the positions a wrong speculation would have occupied.
    batch.clear();
    for (i, &tok) in prompt.iter().take(4).enumerate() {
        batch
            .push(tok, plen + i as i32, 0, false)
            .expect("push junk");
    }
    session.decode_batch(&batch).expect("junk decode");
    assert_eq!(session.seq_pos_max(0), Some(plen + 3));

    // Rollback = REAL seq_rm (positions must stay consecutive, so we also
    // remove the last prompt token and re-decode it to refresh the tail
    // logits — never a rewound counter).
    session.seq_rm(0, plen - 1, -1).expect("rollback seq_rm");
    assert_eq!(session.seq_pos_max(0), Some(plen - 2));
    batch.clear();
    let last = batch
        .push(prompt[prompt.len() - 1], plen - 1, 0, true)
        .expect("re-decode tail token");
    session.decode_batch(&batch).expect("tail re-decode");

    let mut tok = session.sample_ith(last as i32);
    let mut got = vec![tok];
    for _ in 1..MAX_NEW {
        let next = session
            .decode_step(
                &mut batch,
                &[SeqToken {
                    seq_id: 0,
                    token: tok,
                }],
            )
            .expect("post-rollback decode step");
        tok = next[0];
        got.push(tok);
    }
    assert_eq!(
        got, straight,
        "a rolled-back sequence must continue exactly like a run that never diverged"
    );
}

/// seq_cp shares a decoded prefix with a second sequence: continuing the
/// copy greedily must equal continuing the original — KV sharing changes
/// where state lives, never what it says (the §4 reuse primitive).
#[test]
fn seq_cp_shares_prefix_without_redecoding() {
    let Some(model) = smoke_model("seq_cp_shares_prefix_without_redecoding") else {
        return;
    };
    const MAX_NEW: usize = 8;
    let prompt = model.tokenize("Once upon a time", true).unwrap();
    let alone = alone_greedy(&model, &prompt, MAX_NEW);
    assert_eq!(
        alone.len(),
        MAX_NEW,
        "prompt hit EOG too early for this test"
    );

    let mut session = Session::new(
        &model,
        &SessionParams {
            n_ctx: 256,
            n_batch: 128,
            n_seq_max: 2,
            kv_unified: true,
            ..SessionParams::default()
        },
    )
    .expect("session");

    let mut batch = Batch::new(128, 2).expect("batch alloc");
    let last = push_prompt(&mut batch, &prompt, 0);
    session.decode_batch(&batch).expect("prefill decode");
    let first = session.sample_ith(last as i32);
    assert_eq!(first, alone[0]);

    // Copy the whole prefix onto sequence 1 and continue THERE.
    session.seq_cp(0, 1, -1, -1);
    assert_eq!(session.seq_pos_max(1), session.seq_pos_max(0));
    let mut tok = first;
    let mut got = vec![tok];
    for _ in 1..MAX_NEW {
        let next = session
            .decode_step(
                &mut batch,
                &[SeqToken {
                    seq_id: 1,
                    token: tok,
                }],
            )
            .expect("decode step on the copied sequence");
        tok = next[0];
        got.push(tok);
    }
    assert_eq!(
        got, alone,
        "continuing a seq_cp'd prefix must equal continuing the original"
    );

    // seq_keep drops every other sequence; the survivor keeps its state.
    session.seq_keep(1);
    assert_eq!(session.seq_pos_max(0), None, "seq 0 dropped by seq_keep(1)");
    assert!(
        session.seq_pos_max(1).is_some(),
        "seq 1 survives seq_keep(1)"
    );
}
