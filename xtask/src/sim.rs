//! `cargo xtask sim`: the M3 distributed-inference Definition-of-Done
//! rehearsal (docs/distributed.md, "Tests / DoD hooks").
//!
//! Spawns TWO sandboxed daemons on this host (reusing the pair-sim
//! machinery: separate `ONEBRAIN_HOME`s, distinct API ports, mDNS/relays
//! off) and walks the four contract scenarios in one run:
//!
//! 1. **Distribute** — both daemons capped via `[debug]
//!    usable_memory_override_bytes` so the tiny model fits neither alone.
//!    The cap follows the M4 v1 budget rule (docs/scheduler-v1.md: budget =
//!    usable minus the 512 MiB overhead reserve, layer capacity from real
//!    weights + KV at the configured ctx): each node gets the reserve plus
//!    exactly `n_layers - 1` layer-costs, so solo fails by one layer while
//!    two nodes pool `2(n-1) >= n` layers. A `load` WITHOUT `--nodes` must
//!    auto-engage `PipelineParallel` across 2 nodes, streaming `planning` +
//!    `plan` NDJSON lines; both API dialects answer through the distributed
//!    plan, and the OpenAI completion (temperature 0, max_tokens 12, fixed
//!    prompt) is captured.
//! 2. **Socket scan** — while the distributed session is open, every
//!    listening TCP socket of both daemon pids is loopback-only (non-
//!    loopback is forbidden ALWAYS); the head's per-epoch loopback rpc
//!    bridge listener is legitimate during the session (accept-loop, ADR
//!    0004 amendment) and must be gone once the session ends — the solo
//!    re-scan asserts only the api binds remain.
//! 3. **Auto-solo + correctness** — both daemons restart uncapped; the same
//!    load plans `Solo` (status agrees, socket scan still clean = no rpc
//!    sessions), and the same greedy completion is **byte-identical** to
//!    the distributed one (the §9 correctness property).
//! 4. **Forced** — `--nodes 2` on the uncapped pair distributes anyway;
//!    `explain` prose is asserted on every plan line.
//!
//! After the M3 steps, the M4 scheduler-v1 scenarios run on the same
//! paired daemons (docs/scheduler-v1.md "DoD sim hooks"), restarting them
//! with new configs where the caps/ctx must change:
//!
//! 5. **Ctx 2048 (KV shifts with ctx, part 1)** — both daemons restarted
//!    with EQUAL memory caps ([`m4_cap`]: the v1 512 MiB overhead reserve
//!    plus a budget sized from the model's real dims so weights + KV fit
//!    the head at ctx 2048 but not at 16384), `[debug]
//!    decode_tps_override` 100.0 on A vs 50.0 on B, and `ctx_len = 2048`
//!    (the load body has no ctx override — the daemon plans at its config
//!    ctx, so ctx moves via config + restart). The load must plan `Solo`
//!    on the head with all layers.
//! 6. **Ctx 16384 + asymmetric** — same caps and overrides, `ctx_len =
//!    16384`: KV per layer grows 8×, solo no longer fits, and the plan
//!    must be `PipelineParallel` across both nodes with A (decode 100)
//!    taking MORE layers than B (decode 50), each within ±1 layer of the
//!    score prediction computed here from the contract's own formula
//!    (shares ∝ `capacity × (0.5 + 0.5 × decode/max_decode)`; equal caps
//!    cancel the capacity term). With two nodes and a fixed layer total,
//!    "fewer layers per node at 16k" is asserted as: every 16k assignment
//!    holds fewer layers than the 2k Solo plan's single assignment.
//! 7. **Third node** — SKIPPED here (a third daemon would triple the sim's
//!    wall time); the ≥5% rule is covered by the scheduler unit tests
//!    named in the printed `[SKIP]` line.
//!
//! After M4, the M5 **CHAOS** section runs (docs/resilience.md "Sim / DoD
//! hooks"). Under `--netem` it is skipped with a `[SKIP]` line: the
//! pair-sim namespace machinery provides exactly two namespaces and the
//! chaos scenarios need a third daemon; they run in the default loopback
//! mode on every OS.
//!
//! 8.  **chaos setup** — A restarts with the distribute cap plus `[debug]
//!     decode_delay_ms = 150` (the engine host sleeps per emitted token, so
//!     a 40-token generation runs ≥ 6 s and the kill lands mid-stream); B
//!     restarts and a THIRD daemon C starts with the same cap (any two of
//!     the three hold the model, one alone cannot); C pairs with A.
//! 9.  **chaos-1 (kill + transparent retry)** — a capped load distributes
//!     across the head and ONE worker (the plan's assignments say WHICH);
//!     a streaming OpenAI chat (temperature 0, max_tokens 40) is read on a
//!     collector thread, the in-epoch worker is killed -9 after ≥ 3
//!     content chunks, and the SAME stream must complete with
//!     finish_reason "length" and no error events; status must then show a
//!     NEW epoch excluding the dead node, and the full streamed text must
//!     equal a control run made afterwards against the recovered topology
//!     (greedy determinism — the §9 correctness property under failure).
//! 10. **chaos-2 (no fallback, typed error)** — the survivors restart with
//!     the same caps (head + ONE worker required, only one worker left);
//!     that worker is killed mid-stream again, and the stream must end
//!     with the structured error naming the lost node and BOTH MB figures
//!     (docs/resilience.md failure lifecycle step 3); the daemon must stay
//!     healthy and a fresh load must fail FAST with the planning error
//!     (both MB figures again) rather than hanging.
//! 11. **chaos-3 (rejoin ⇒ lazy re-plan)** — the chaos-2 worker revives
//!     (decode_tps_override 50) and a reload re-establishes a distributed
//!     epoch; then the chaos-1 dead worker restarts (same home/ports,
//!     decode_tps_override 100 so the score ranks it first) and the head
//!     must reach a NEW epoch including it within 45 s with no client
//!     activity beyond status polling.
//! 12. **chaos-4 (drain)** — `onebrain stop` on the now-idle other worker:
//!     it must leave `connected` in the head's peers, and the next plan
//!     (trigger: a reload) must exclude it while keeping the live worker.
//!
//! `--netem` (Linux, root only — SKIP + exit 0 anywhere else): the same
//! M3/M4 scenario inside the pair-sim network namespaces, shaped to
//! 1 Gbit / 0.5 ms per direction (chaos skipped, see above).
//!
//! One `[PASS]`/`[FAIL]` checklist line per step; daemon-log tails are
//! dumped on failure; `OB_E2E_SKIP_BUILD=1` skips the inner build.

use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::e2e::{dump_daemon_log, kill_hard, locate_onebrain_binary, step};
use crate::pair_sim::{
    cleanup, is_root, netem_setup, peer_ref, two_free_ports, Node, PeerRef, NS_A, NS_B,
};

/// The first NDJSON line (`status: "window"`) must arrive within this.
const WINDOW_TIMEOUT: Duration = Duration::from_secs(10);
/// Budget for `pair/join` (dials, SPAKE2, confirm, introduce, persist).
const JOIN_TIMEOUT: Duration = Duration::from_secs(60);
/// A's stream must report `paired` within this once the join returned.
const PAIRED_TIMEOUT: Duration = Duration::from_secs(15);
/// Heartbeats must drive both sides to `connected` within this.
const CONNECTED_TIMEOUT: Duration = Duration::from_secs(15);
/// After `onebrain stop`, the status endpoint must be gone within this.
const STOP_TIMEOUT: Duration = Duration::from_secs(10);
/// Whole-load budget: plan + RPC weight push of a ~1 MB model + warmup.
const LOAD_TIMEOUT: Duration = Duration::from_secs(240);
/// Per-generation budget (tiny model, ≤12 tokens, one-time warmup).
const GEN_TIMEOUT: Duration = Duration::from_secs(120);
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// The fixed prompt behind the §9 correctness property.
const PROMPT: &str = "Once upon a time";

pub fn run(netem: bool) -> Result<()> {
    if netem && !cfg!(target_os = "linux") {
        println!(
            "[SKIP] sim --netem needs Linux (network namespaces + tc netem); \
             nothing to do on this OS"
        );
        return Ok(());
    }
    if netem && !is_root() {
        println!("[SKIP] sim --netem needs root for `ip netns`/`tc`; rerun under sudo -E");
        return Ok(());
    }

    let root = crate::workspace_root();
    println!(
        "== cargo xtask sim{}: M3 distributed-inference rehearsal ==",
        if netem { " --netem" } else { "" }
    );

    // Build first (streams cargo's own output), then locate the binary.
    // xtask itself is excluded for the same reason as e2e: rebuilding the
    // xtask binary that is running this command fails on Windows.
    let binary = step("build: cargo build --workspace", || {
        if std::env::var("OB_E2E_SKIP_BUILD").as_deref() == Ok("1") {
            println!("  (skipping inner build: OB_E2E_SKIP_BUILD=1)");
            return locate_onebrain_binary(&root);
        }
        let status = Command::new("cargo")
            .current_dir(&root)
            .args(["build", "--workspace", "--exclude", "xtask"])
            .status()
            .context("failed to invoke cargo build")?;
        if !status.success() {
            bail!("cargo build --workspace failed (see compiler output above); fix and rerun");
        }
        locate_onebrain_binary(&root)
    })?;

    let (model_path, dims, cap) = step(
        "model: tiny GGUF + v1 caps (solo fails, pooled fits)",
        || {
            let cache = root.join("target-smoke");
            std::fs::create_dir_all(&cache)
                .with_context(|| format!("creating {}", cache.display()))?;
            let path = crate::smoke::ensure_model(&cache)?;
            let dims = read_gguf_dims(&path)?;
            let cap = m3_distribute_cap(&dims);
            let budget = m3_distribute_budget(&dims);
            println!(
            "  model {} ({} layers, kv {} B/token/layer, weights {} B)\n  per-node cap {cap} B \
             = the v1 512 MiB reserve + budget {budget} B, which holds {} of {} layers at ctx \
             {M3_DEFAULT_CTX} (solo needs {} B; two nodes pool {} layers)",
            path.display(),
            dims.n_layers,
            dims.kv_rate,
            dims.total_weight_bytes,
            budget / dims.per_layer_cost(M3_DEFAULT_CTX),
            dims.n_layers,
            dims.required_bytes(M3_DEFAULT_CTX),
            2 * (budget / dims.per_layer_cost(M3_DEFAULT_CTX)),
        );
            Ok((path, dims, cap))
        },
    )?;

    if netem {
        step(
            "netem: namespaces + veth pair shaped to 1gbit / 0.5ms per direction",
            netem_setup,
        )?;
    }

    let (port_a, port_b) = two_free_ports()?;
    // Mesh UDP ports, pinned so stored peer addresses survive the restart
    // scenario. Picked while the api ports are already released — the same
    // small pick-then-bind race the api ports accept.
    let (mesh_a, mesh_b) = two_free_ports()?;
    // The third daemon of the M5 chaos section (started there, not here).
    let (port_c, mesh_c) = two_free_ports()?;
    let base = std::env::temp_dir();
    let run_id = std::process::id();
    let a = Node::new(
        "daemon A (head)",
        "sim-a",
        base.join(format!("onebrain-sim-{run_id}-a")),
        port_a,
        binary.clone(),
        netem.then_some(NS_A),
    )?;
    let b = Node::new(
        "daemon B (worker)",
        "sim-b",
        base.join(format!("onebrain-sim-{run_id}-b")),
        port_b,
        binary.clone(),
        netem.then_some(NS_B),
    )?;
    // No namespace for C: the chaos section is skipped under --netem
    // (module docs), so C only ever runs on the shared loopback.
    let c = Node::new(
        "daemon C (worker)",
        "sim-c",
        base.join(format!("onebrain-sim-{run_id}-c")),
        port_c,
        binary,
        None,
    )?;
    // Node::new wrote the plain pair-sim config; overwrite with the same
    // switches PLUS the memory cap and pinned mesh port before the daemons
    // ever start. C's config is written by the chaos section before its
    // first `up`.
    write_config(&a, mesh_a, netem, SimKnobs::capped(cap))?;
    write_config(&b, mesh_b, netem, SimKnobs::capped(cap))?;
    println!(
        "sandbox A: {} (api {port_a}, mesh {mesh_a})",
        a.home.display()
    );
    println!(
        "sandbox B: {} (api {port_b}, mesh {mesh_b})",
        b.home.display()
    );
    println!(
        "sandbox C: {} (api {port_c}, mesh {mesh_c}, chaos section only)",
        c.home.display()
    );

    let outcome = scenario(
        &a,
        &b,
        &c,
        &model_path,
        &dims,
        (mesh_a, mesh_b, mesh_c),
        netem,
    );
    if outcome.is_err() {
        dump_daemon_log(&a.home);
        dump_daemon_log(&b.home);
        dump_daemon_log(&c.home);
    }
    cleanup(&[&a, &b, &c], netem);
    outcome?;
    println!("sim: all steps passed");
    Ok(())
}

/// The whole rehearsal. Steps abort on first failure (later ones depend on
/// earlier ones); `cleanup` runs in `run` regardless.
fn scenario(
    a: &Node,
    b: &Node,
    c: &Node,
    model_path: &Path,
    dims: &SimModelDims,
    mesh: (u16, u16, u16),
    netem: bool,
) -> Result<()> {
    let (mesh_a, mesh_b, _mesh_c) = mesh;
    let model_arg = model_path
        .to_str()
        .context("model path is not valid UTF-8")?;

    step("up: both capped daemons healthy", || {
        for node in [a, b] {
            let out = node.onebrain(&["up"])?;
            node.wait_healthy().map_err(|e| {
                anyhow!(
                    "{e:#}\n`onebrain up` exit code {:?}\nstdout: {}\nstderr: {}",
                    out.status.code(),
                    String::from_utf8_lossy(&out.stdout).trim(),
                    String::from_utf8_lossy(&out.stderr).trim()
                )
            })?;
        }
        Ok(())
    })?;

    let (peer_a, peer_b) = step("pair: A and B paired and connected over the mesh", || {
        let (peer_a, peer_b) = pair(a, b)?;
        wait_connected(a, b, &peer_a, &peer_b, "after pairing")?;
        Ok((peer_a, peer_b))
    })?;

    // Scenario 1: distribution engages WITHOUT --nodes because neither
    // capped node fits the model alone.
    let dist_plan = step(
        "distribute: capped load auto-engages PipelineParallel across 2 nodes",
        || {
            let plan = load_model(a, &json!({ "model": model_arg, "explain": true }))?;
            assert_pipeline(&plan)?;
            assert_explained(&plan)?;
            Ok(plan)
        },
    )?;
    println!(
        "       plan epoch {}: {} assignments",
        dist_plan["epoch"],
        dist_plan["assignments"].as_array().map_or(0, Vec::len)
    );

    step(
        "status: A reports the active plan as PipelineParallel",
        || assert_pipeline(&active_plan(a)?),
    )?;

    // Scenario 3 (contract numbering): scan while the distributed session
    // is open — the rpc streams are alive, yet nothing may LISTEN anywhere
    // beyond the two loopback api binds.
    step(
        "sockets: loopback-only while the session is open (bridge listeners allowed)",
        || socket_scan(&[a, b], true),
    )?;

    let text_dist = step(
        "openai: distributed completion (temperature 0, max_tokens 12)",
        || {
            let name = model_name(a)?;
            chat_text(a, &name)
        },
    )?;
    println!("       distributed text: {text_dist:?}");

    step(
        "ollama: /api/generate streams done:true through the distributed plan",
        || {
            let name = model_name(a)?;
            ollama_streams(a, &name)
        },
    )?;

    step(
        "restart: both daemons back up uncapped, pairing intact",
        || {
            // Head first: its epoch teardown closes the rpc streams before the
            // worker goes away, so nothing tears mid-session.
            for (node, mesh_port) in [(a, mesh_a), (b, mesh_b)] {
                restart_with(node, mesh_port, netem, SimKnobs::default())?;
            }
            wait_connected(a, b, &peer_a, &peer_b, "after the uncapped restart")
        },
    )?;

    // Scenario 2: auto-solo on the uncapped pair — distribution must NOT
    // engage, asserted via the plan line, the status endpoint, and the
    // socket scan (no rpc session artifacts).
    step("auto-solo: uncapped load plans Solo on the head", || {
        let plan = load_model(a, &json!({ "model": model_arg, "explain": true }))?;
        assert_solo(&plan)?;
        assert_explained(&plan)?;
        assert_solo(&active_plan(a)?)
    })?;

    step(
        "sockets: solo run stays loopback-only (no rpc leftovers)",
        || socket_scan(&[a, b], false),
    )?;

    step(
        "greedy: solo text byte-identical to distributed text (§9 property)",
        || {
            let name = model_name(a)?;
            let text_solo = chat_text(a, &name)?;
            if text_solo != text_dist {
                bail!(
                    "distributed and solo greedy completions differ:\n\
                     distributed: {text_dist:?}\n\
                     solo:        {text_solo:?}"
                );
            }
            Ok(())
        },
    )?;

    // Scenario 4: --nodes 2 forces distribution even though solo would fit.
    step(
        "forced: --nodes 2 uncapped distributes, with --explain prose",
        || {
            let plan = load_model(
                a,
                &json!({ "model": model_arg, "nodes": 2, "explain": true }),
            )?;
            assert_pipeline(&plan)?;
            assert_explained(&plan)
        },
    )?;

    // ---- M4 scheduler-v1 scenarios (docs/scheduler-v1.md "DoD sim hooks",
    // module docs steps 5-7). Reuses the paired daemons from the forced
    // step; restarts them because caps, decode overrides, and ctx all live
    // in config (the load body carries no ctx override).
    step(
        "m4 caps: the ctx-shift + asymmetric invariants hold for this model",
        || {
            check_m4_scenario(dims)?;
            println!(
                "  per-node budget {} B (cap {} B = budget + the v1 512 MiB overhead \
                 reserve): ctx {M4_CTX_SMALL} needs {} B (solo fits), ctx {M4_CTX_BIG} \
                 needs {} B (solo fails)",
                m4_budget(dims),
                m4_cap(dims),
                dims.required_bytes(M4_CTX_SMALL),
                dims.required_bytes(M4_CTX_BIG),
            );
            Ok(())
        },
    )?;
    let m4_knobs = |decode_tps: f64, ctx: u32| SimKnobs {
        cap_bytes: Some(m4_cap(dims)),
        decode_tps_override: Some(decode_tps),
        ctx_len: Some(ctx),
        ..SimKnobs::default()
    };

    step(
        "m4 restart: equal caps, decode_tps_override 100 (A) / 50 (B), ctx 2048",
        || {
            // Head first, as in the uncapped restart (clean epoch teardown).
            for (node, mesh_port, decode) in
                [(a, mesh_a, M4_DECODE_FAST), (b, mesh_b, M4_DECODE_SLOW)]
            {
                restart_with(node, mesh_port, netem, m4_knobs(decode, M4_CTX_SMALL))?;
            }
            wait_connected(a, b, &peer_a, &peer_b, "after the m4 ctx-2048 restart")
        },
    )?;

    // KV shifts with ctx, part 1: at ctx 2048 the weights + KV fit the
    // head's budget, so the same caps that will force a split at 16k must
    // plan Solo here (all layers on the head).
    let solo_layers = step(
        "m4 ctx 2048: weights + KV fit the head -> Solo plan",
        || {
            let plan = load_model(a, &json!({ "model": model_arg, "explain": true }))?;
            assert_solo(&plan)?;
            assert_explained(&plan)?;
            assert_plan_ctx(&plan, M4_CTX_SMALL)?;
            let layers = assignment_layers(&plan.assignments()?[0])?;
            if layers != dims.n_layers {
                bail!(
                    "the Solo plan holds {layers} layers but the model has {}: {plan}",
                    dims.n_layers
                );
            }
            Ok(layers)
        },
    )?;

    step("m4 restart: same caps and overrides, ctx 16384", || {
        for (node, mesh_port, decode) in [(a, mesh_a, M4_DECODE_FAST), (b, mesh_b, M4_DECODE_SLOW)]
        {
            restart_with(node, mesh_port, netem, m4_knobs(decode, M4_CTX_BIG))?;
        }
        wait_connected(a, b, &peer_a, &peer_b, "after the m4 ctx-16384 restart")
    })?;

    // KV shifts with ctx, part 2: KV per layer grew 8x, so solo no longer
    // fits and the plan must split across both nodes.
    let big_plan = step(
        "m4 ctx 16384: solo no longer fits -> PipelineParallel across 2",
        || {
            let plan = load_model(a, &json!({ "model": model_arg, "explain": true }))?;
            assert_pipeline(&plan)?;
            assert_explained(&plan)?;
            assert_plan_ctx(&plan, M4_CTX_BIG)?;
            Ok(plan)
        },
    )?;

    // Asymmetric: equal caps, decode 100 vs 50. The score prediction is
    // computed HERE from the contract's own formula (docs/scheduler-v1.md
    // "Placement algorithm" §2: shares ∝ capacity × (0.5 + 0.5 ×
    // decode/max_decode); equal caps cancel the capacity term), and the
    // actual split must land within ±1 layer of it, with A strictly ahead.
    step(
        "m4 asymmetric: A (decode 100) takes MORE layers than B (decode 50), within ±1 of the score prediction",
        || {
            let (a_layers, b_layers) = layers_by_node(&big_plan, &peer_a.id, &peer_b.id)?;
            let total = a_layers + b_layers;
            if total != dims.n_layers {
                bail!(
                    "the 16k plan covers {total} layers but the model has {}: {big_plan}",
                    dims.n_layers
                );
            }
            if a_layers <= b_layers {
                bail!(
                    "A (decode_tps_override {M4_DECODE_FAST}) got {a_layers} layers vs B \
                     (override {M4_DECODE_SLOW}) with {b_layers}; the memory-and-compute \
                     score must tilt the split toward the faster node: {big_plan}"
                );
            }
            let (exp_a, exp_b) = expected_split(dims.n_layers);
            for (label, actual, expected) in
                [("A", a_layers, exp_a), ("B", b_layers, exp_b)]
            {
                if (actual as f64 - expected).abs() > 1.0 {
                    bail!(
                        "{label} got {actual} layers but the score formula predicts \
                         {expected:.2} (tolerance ±1): {big_plan}"
                    );
                }
            }
            println!(
                "       split A/B = {a_layers}/{b_layers} (score prediction \
                 {exp_a:.2}/{exp_b:.2})"
            );
            Ok(())
        },
    )?;

    // With two nodes and a fixed layer total, "fewer layers per node at
    // 16k" can only mean: solo at 2k, split at 16k — so every 16k
    // assignment must hold fewer layers than the 2k Solo plan's one
    // assignment did (module docs step 6).
    step(
        "m4 ctx shift: every 16k assignment holds fewer layers than the 2k plan's",
        || {
            for asg in big_plan.assignments()? {
                let layers = assignment_layers(asg)?;
                if layers >= solo_layers {
                    bail!(
                        "assignment {asg} holds {layers} layers, not fewer than the \
                         ctx-2048 plan's {solo_layers}"
                    );
                }
            }
            Ok(())
        },
    )?;

    // Third-node rule: not rehearsed here — a third daemon would triple the
    // sim's wall time for coverage the scheduler unit tests already pin.
    println!(
        "[SKIP] m4 third node helps only when it helps: covered by the scheduler unit tests \
         `third_node_excluded_when_gain_below_threshold` and \
         `third_node_included_when_two_cannot_hold_the_model` \
         (crates/onebrain-scheduler/src/v1.rs; docs/scheduler-v1.md \"DoD sim hooks\")"
    );

    // ---- M5 chaos section (docs/resilience.md "Sim / DoD hooks") --------
    if netem {
        println!(
            "[SKIP] chaos (M5) under --netem: the pair-sim machinery provides exactly two \
             network namespaces and the chaos scenarios need a third daemon; they run in the \
             default loopback mode on every OS"
        );
        return Ok(());
    }
    chaos_section(
        &ChaosEnv {
            a,
            b,
            c,
            peer_a: &peer_a,
            peer_b: &peer_b,
            mesh,
            model_arg,
        },
        dims,
    )
}

// ---------------------------------------------------------------------------
// M5 chaos section (docs/resilience.md "Sim / DoD hooks")
// ---------------------------------------------------------------------------

/// `[debug] decode_delay_ms` on the head during the chaos phases: the
/// engine host sleeps this long per emitted token, stretching a 40-token
/// generation past 6 s so the kill reliably lands mid-stream.
const CHAOS_DECODE_DELAY_MS: u64 = 150;
/// `max_tokens` of the chaos generations; the surviving stream must end
/// with finish_reason "length", i.e. all of them.
const CHAOS_MAX_TOKENS: u32 = 40;
/// The kill fires once this many NON-EMPTY content chunks have streamed.
const CHAOS_KILL_AFTER_CHUNKS: usize = 3;
/// Between consecutive SSE events of a chaos stream: must cover death
/// detection, epoch teardown, the re-plan, the distributed reload, and the
/// prefix re-prefill of the transparent retry.
const CHAOS_EVENT_TIMEOUT: Duration = Duration::from_secs(90);
/// Whole-request cap on one chaos stream (kill, retry, and tail included).
const CHAOS_STREAM_TIMEOUT: Duration = Duration::from_secs(300);
/// The rejoin lazy re-plan must reach a new epoch within this
/// (docs/resilience.md scenario 3; "~45 s" in the sim contract).
const REJOIN_REPLAN_TIMEOUT: Duration = Duration::from_secs(45);
/// A killed or politely stopped worker must leave `connected` in the
/// head's peers within this (heartbeat death is 10 s, plus polling slack).
const PEER_LOSS_TIMEOUT: Duration = Duration::from_secs(30);
/// A load that cannot plan (no workers left) must fail within this — the
/// "returns the planning error rather than hanging" assertion.
const FAST_FAIL_TIMEOUT: Duration = Duration::from_secs(30);

/// Everything the chaos section needs from the earlier steps.
struct ChaosEnv<'a> {
    a: &'a Node,
    b: &'a Node,
    c: &'a Node,
    /// A as its peers list it (`id` = A's endpoint id, used in assignments).
    peer_a: &'a PeerRef,
    /// B as A lists it.
    peer_b: &'a PeerRef,
    /// Pinned mesh UDP ports of (A, B, C).
    mesh: (u16, u16, u16),
    model_arg: &'a str,
}

/// One chaos worker with everything needed to kill and revive it.
struct ChaosWorker<'a> {
    node: &'a Node,
    mesh_port: u16,
    peer: &'a PeerRef,
}

/// The M5 chaos rehearsal (module docs steps 8–12). Runs after the M4
/// steps, non-netem only; aborts on first failure like the rest.
fn chaos_section(env: &ChaosEnv, dims: &SimModelDims) -> Result<()> {
    let (mesh_a, mesh_b, mesh_c) = env.mesh;
    // The M3 distribute cap already has the chaos-1 shape: each node holds
    // n-1 of n layers, so ANY TWO of the three hold the model pooled while
    // one alone cannot (the unit tests pin this per model shape).
    let cap = m3_distribute_cap(dims);
    let head_knobs = SimKnobs {
        cap_bytes: Some(cap),
        decode_delay_ms: Some(CHAOS_DECODE_DELAY_MS),
        ..SimKnobs::default()
    };

    step(
        "chaos: A+B restarted capped (decode_delay_ms 150 on the head); C up capped",
        || {
            // Head first: clean epoch teardown while its worker still lives.
            restart_with(env.a, mesh_a, false, head_knobs)?;
            restart_with(env.b, mesh_b, false, SimKnobs::capped(cap))?;
            start_with(env.c, mesh_c, false, SimKnobs::capped(cap))
        },
    )?;

    let peer_c = step(
        "chaos: C paired with A; both worker links connected",
        || {
            let (_a_as_c_sees_it, peer_c) = pair(env.a, env.c)?;
            wait_connected(env.a, env.c, env.peer_a, &peer_c, "after pairing C")?;
            wait_connected(
                env.a,
                env.b,
                env.peer_a,
                env.peer_b,
                "after the chaos restart",
            )?;
            Ok(peer_c)
        },
    )?;

    // ---- chaos-1: kill mid-generation, the SAME stream completes --------
    let plan1 = step(
        "chaos-1: capped 3-node load distributes across the head + ONE worker",
        || {
            let plan = load_model(env.a, &json!({ "model": env.model_arg, "explain": true }))?;
            assert_distributed(&plan)?;
            if !assignment_node_ids(&plan)
                .iter()
                .any(|id| id == &env.peer_a.id)
            {
                bail!("the head is not in the plan's assignments: {plan}");
            }
            Ok(plan)
        },
    )?;
    let epoch1 = plan_epoch(&plan1)?;
    // The plan says WHICH worker holds a shard (the scheduler picks one of
    // {B, C}); the other is the survivor the retry must re-plan onto.
    let in_epoch = epoch_worker_ids(&plan1, &[env.peer_b.id.as_str(), peer_c.id.as_str()])?;
    let victim_id = in_epoch[0].clone();
    let (victim, survivor) = if victim_id == env.peer_b.id {
        (
            ChaosWorker {
                node: env.b,
                mesh_port: mesh_b,
                peer: env.peer_b,
            },
            ChaosWorker {
                node: env.c,
                mesh_port: mesh_c,
                peer: &peer_c,
            },
        )
    } else {
        (
            ChaosWorker {
                node: env.c,
                mesh_port: mesh_c,
                peer: &peer_c,
            },
            ChaosWorker {
                node: env.b,
                mesh_port: mesh_b,
                peer: env.peer_b,
            },
        )
    };
    println!(
        "       epoch {epoch1}: in-epoch worker is {} ({} worker assignment(s))",
        victim.node.label,
        in_epoch.len(),
    );

    let model = model_name(env.a)?;
    let watch = step(
        "chaos-1: kill -9 the in-epoch worker mid-stream; the SAME stream completes (finish_reason length)",
        || {
            let pid = daemon_pid(victim.node)?;
            let watch = run_chaos_stream(env.a, &model, || {
                kill_hard(pid)?;
                println!(
                    "       killed {} (pid {pid}) after {CHAOS_KILL_AFTER_CHUNKS} content chunks",
                    victim.node.label
                );
                Ok(())
            })?;
            if watch.kill_at != Some(CHAOS_KILL_AFTER_CHUNKS) {
                bail!(
                    "the kill never fired: only {} content chunks arrived before the stream \
                     ended (errors: {:?})",
                    watch.pieces.len(),
                    watch.errors
                );
            }
            if !watch.errors.is_empty() {
                bail!(
                    "the retry must be transparent, but the stream carried error events: {:?}",
                    watch.errors
                );
            }
            if !watch.done {
                bail!("the stream ended without `data: [DONE]`");
            }
            if !watch.finish.iter().any(|f| f == "length") {
                bail!(
                    "no chunk carried finish_reason \"length\" (saw {:?}) — the stream did \
                     not run to max_tokens",
                    watch.finish
                );
            }
            if watch.pieces_after_kill() == 0 {
                bail!("no content chunks arrived after the kill — the retry never resumed");
            }
            Ok(watch)
        },
    )?;
    println!(
        "       {CHAOS_KILL_AFTER_CHUNKS} chunks before the kill + {} after",
        watch.pieces_after_kill()
    );

    step(
        "chaos-1: a NEW epoch is active excluding the dead node",
        || {
            let plan = wait_for_plan(
                env.a,
                Duration::from_secs(10),
                "a new epoch excluding the dead worker",
                |p| {
                    plan_epoch(p).ok() != Some(epoch1)
                        && !assignment_node_ids(p).iter().any(|id| id == &victim_id)
                },
            )?;
            assert_distributed(&plan)?;
            if !assignment_node_ids(&plan)
                .iter()
                .any(|id| id == &survivor.peer.id)
            {
                bail!(
                    "the recovery plan does not include the surviving worker {}: {plan}",
                    survivor.node.label
                );
            }
            Ok(())
        },
    )?;

    step(
        "chaos-1: streamed text equals a control run on the recovered topology (greedy)",
        || {
            let (text, tokens, finish) = chat_full(env.a, &model, CHAOS_MAX_TOKENS)?;
            if finish.as_deref() != Some("length") {
                bail!("control run finished with {finish:?}, expected \"length\"");
            }
            if tokens != Some(u64::from(CHAOS_MAX_TOKENS)) {
                bail!("control run generated {tokens:?} completion tokens, expected {CHAOS_MAX_TOKENS}");
            }
            let streamed = watch.text();
            if text != streamed {
                bail!(
                    "retried stream and control run differ:\n  stream:  {streamed:?}\n  \
                     control: {text:?}"
                );
            }
            Ok(())
        },
    )?;

    // ---- chaos-2: kill with no fallback -> structured typed error -------
    step(
        "chaos-2: survivors restarted capped; load distributes across the head + the ONLY worker",
        || {
            restart_with(env.a, mesh_a, false, head_knobs)?;
            restart_with(
                survivor.node,
                survivor.mesh_port,
                false,
                SimKnobs::capped(cap),
            )?;
            wait_connected(
                env.a,
                survivor.node,
                env.peer_a,
                survivor.peer,
                "after the chaos-2 restart",
            )?;
            let plan = load_model(env.a, &json!({ "model": env.model_arg, "explain": true }))?;
            assert_distributed(&plan)?;
            if !assignment_node_ids(&plan)
                .iter()
                .any(|id| id == &survivor.peer.id)
            {
                bail!(
                    "the chaos-2 plan does not include {}: {plan}",
                    survivor.node.label
                );
            }
            Ok(())
        },
    )?;

    step(
        "chaos-2: kill with no fallback -> structured error naming the node + both MB figures",
        || {
            let pid = daemon_pid(survivor.node)?;
            let watch = run_chaos_stream(env.a, &model, || {
                kill_hard(pid)?;
                println!(
                    "       killed {} (pid {pid}) after {CHAOS_KILL_AFTER_CHUNKS} content chunks",
                    survivor.node.label
                );
                Ok(())
            })?;
            if watch.kill_at.is_none() {
                bail!(
                    "the kill never fired: only {} content chunks arrived (errors: {:?})",
                    watch.pieces.len(),
                    watch.errors
                );
            }
            if watch.finish.iter().any(|f| f == "length") {
                bail!(
                    "the stream ran to completion, but no fallback exists — it must end with \
                     the structured error"
                );
            }
            if watch.errors.is_empty() {
                bail!(
                    "the stream ended with no error event ({} chunks, done={}, finish {:?})",
                    watch.pieces.len(),
                    watch.done,
                    watch.finish
                );
            }
            structured_loss_check(
                &watch.errors.join("\n"),
                survivor.node.name,
                &survivor.peer.id,
            )
        },
    )?;

    step(
        "chaos-2: daemon stays healthy; a fresh load fails fast with the planning error",
        || {
            env.a.wait_peer(
                &survivor.peer.id,
                PEER_LOSS_TIMEOUT,
                "the killed worker leaving connected",
                |p| p["state"] != "connected",
            )?;
            env.a.try_status()?;
            let started = Instant::now();
            let outcome = load_model(env.a, &json!({ "model": env.model_arg }));
            let elapsed = started.elapsed();
            let Err(e) = outcome else {
                bail!(
                    "the fresh load succeeded, but every worker is dead — it must fail with \
                     the planning error"
                );
            };
            if elapsed > FAST_FAIL_TIMEOUT {
                bail!(
                    "the failing load took {elapsed:?} (over {FAST_FAIL_TIMEOUT:?}); it must \
                     fail fast, not hang"
                );
            }
            let message = format!("{e:#}");
            if count_mb_figures(&message) < 2 {
                bail!("the planning error lacks the two MB figures (needs/have): {message}");
            }
            Ok(())
        },
    )?;

    // ---- chaos-3: rejoin triggers a lazy re-plan ------------------------
    let epoch2 = step(
        "chaos-3 setup: revive the chaos-2 worker (slow) and reload distributed",
        || {
            start_with(
                survivor.node,
                survivor.mesh_port,
                false,
                SimKnobs {
                    cap_bytes: Some(cap),
                    decode_tps_override: Some(M4_DECODE_SLOW),
                    ..SimKnobs::default()
                },
            )?;
            wait_connected(
                env.a,
                survivor.node,
                env.peer_a,
                survivor.peer,
                "after reviving the chaos-2 worker",
            )?;
            let plan = load_model(env.a, &json!({ "model": env.model_arg }))?;
            assert_distributed(&plan)?;
            if !assignment_node_ids(&plan)
                .iter()
                .any(|id| id == &survivor.peer.id)
            {
                bail!(
                    "the re-established plan does not include {}: {plan}",
                    survivor.node.label
                );
            }
            plan_epoch(&plan)
        },
    )?;

    step(
        "chaos-3: the chaos-1 dead worker restarts -> lazy re-plan reaches a NEW epoch including it",
        || {
            // decode_tps_override 100 vs the survivor's 50: the rejoiner
            // ranks first in the scheduler's score order (equal caps), so
            // the re-planned epoch deterministically picks it
            // (docs/scheduler-v1.md "Placement algorithm" §2).
            start_with(
                victim.node,
                victim.mesh_port,
                false,
                SimKnobs {
                    cap_bytes: Some(cap),
                    decode_tps_override: Some(M4_DECODE_FAST),
                    ..SimKnobs::default()
                },
            )?;
            let plan = wait_for_plan(
                env.a,
                REJOIN_REPLAN_TIMEOUT,
                "a new epoch including the rejoined worker",
                |p| {
                    plan_epoch(p).ok() != Some(epoch2)
                        && assignment_node_ids(p).iter().any(|id| id == &victim_id)
                },
            )?;
            println!(
                "       rejoin re-plan: epoch {epoch2} -> {}",
                plan_epoch(&plan).unwrap_or(0)
            );
            Ok(())
        },
    )?;

    // ---- chaos-4: polite drain via `onebrain stop` ----------------------
    step(
        "chaos-4: `onebrain stop` on the idle worker; it leaves connected on the head",
        || {
            let out = survivor.node.onebrain(&["stop"])?;
            if !out.status.success() {
                bail!(
                    "`onebrain stop` on {} failed: {}",
                    survivor.node.label,
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }
            let deadline = Instant::now() + STOP_TIMEOUT;
            while survivor.node.try_status().is_ok() {
                if Instant::now() >= deadline {
                    bail!(
                        "{} still answering {STOP_TIMEOUT:?} after `onebrain stop`",
                        survivor.node.label
                    );
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            env.a.wait_peer(
                &survivor.peer.id,
                PEER_LOSS_TIMEOUT,
                "the stopped worker leaving connected",
                |p| p["state"] != "connected",
            )?;
            Ok(())
        },
    )?;

    step(
        "chaos-4: the head's next plan excludes the drained worker",
        || {
            let plan = load_model(env.a, &json!({ "model": env.model_arg, "explain": true }))?;
            assert_distributed(&plan)?;
            let ids = assignment_node_ids(&plan);
            if ids.iter().any(|id| id == &survivor.peer.id) {
                bail!(
                    "the new plan still includes the stopped worker {}: {plan}",
                    survivor.node.label
                );
            }
            if !ids.iter().any(|id| id == &victim_id) {
                bail!(
                    "the new plan does not include the remaining live worker {}: {plan}",
                    victim.node.label
                );
            }
            Ok(())
        },
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Chaos support: SSE stream watching, kill-window logic, error shape checks
// ---------------------------------------------------------------------------

/// What one chaos SSE stream produced, fed event by event through
/// [`StreamWatch::observe`] (pure logic — unit-tested with fixtures).
#[derive(Debug, Default)]
struct StreamWatch {
    /// Kill trigger: fire after this many non-empty content chunks.
    kill_after: usize,
    /// Non-empty content chunks, in arrival order.
    pieces: Vec<String>,
    /// `Some(n)`: the kill fired after piece `n` (== `kill_after`).
    kill_at: Option<usize>,
    /// Every non-null finish_reason seen.
    finish: Vec<String>,
    /// Every SSE error event's message (plus unparsable payloads).
    errors: Vec<String>,
    /// `data: [DONE]` arrived.
    done: bool,
}

/// What the driver should do after feeding one event to the watch.
#[derive(Debug, PartialEq, Eq)]
enum StreamAction {
    Continue,
    /// The kill window just opened: hard-kill the target NOW.
    KillNow,
    /// `[DONE]` arrived; stop reading.
    Finished,
}

impl StreamWatch {
    fn new(kill_after: usize) -> StreamWatch {
        StreamWatch {
            kill_after,
            ..StreamWatch::default()
        }
    }

    fn observe(&mut self, payload: &str) -> StreamAction {
        if payload == "[DONE]" {
            self.done = true;
            return StreamAction::Finished;
        }
        let Ok(event) = serde_json::from_str::<Value>(payload) else {
            self.errors
                .push(format!("unparsable SSE payload: {payload}"));
            return StreamAction::Continue;
        };
        if let Some(message) = event.pointer("/error/message").and_then(Value::as_str) {
            self.errors.push(message.to_string());
            return StreamAction::Continue;
        }
        if let Some(reason) = event
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
        {
            self.finish.push(reason.to_string());
        }
        if let Some(content) = event
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            if !content.is_empty() {
                self.pieces.push(content.to_string());
                if self.kill_at.is_none() && self.pieces.len() >= self.kill_after {
                    self.kill_at = Some(self.pieces.len());
                    return StreamAction::KillNow;
                }
            }
        }
        StreamAction::Continue
    }

    /// All streamed text in order (the retry never re-sends sent pieces).
    fn text(&self) -> String {
        self.pieces.concat()
    }

    fn pieces_after_kill(&self) -> usize {
        self.kill_at.map_or(0, |at| self.pieces.len() - at)
    }
}

/// Open a streaming OpenAI chat completion (temperature 0, fixed prompt,
/// [`CHAOS_MAX_TOKENS`]) and read its SSE events on a collector thread:
/// each `data:` payload arrives on the returned channel; the thread ends
/// with the stream. Non-netem only (the chaos section guards this).
fn open_chat_stream(node: &Node, model: &str) -> Result<std::sync::mpsc::Receiver<String>> {
    let body = json!({
        "model": model,
        "messages": [{ "role": "user", "content": PROMPT }],
        "stream": true,
        "temperature": 0,
        "max_tokens": CHAOS_MAX_TOKENS,
    });
    let resp = node
        .client
        .post(node.url("/v1/chat/completions"))
        .bearer_auth(node.token()?)
        .json(&body)
        .timeout(CHAOS_STREAM_TIMEOUT)
        .send()
        .with_context(|| {
            format!(
                "POST /v1/chat/completions (stream) on {} failed",
                node.label
            )
        })?;
    if !resp.status().is_success() {
        let code = resp.status();
        let text = resp.text().unwrap_or_default();
        bail!(
            "streaming chat on {} answered HTTP {code}: {text}",
            node.label
        );
    }
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        use std::io::BufRead;
        for line in std::io::BufReader::new(resp).lines() {
            let Ok(line) = line else { break };
            let Some(payload) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = payload.trim();
            if payload.is_empty() {
                continue;
            }
            if tx.send(payload.to_string()).is_err() {
                break;
            }
        }
    });
    Ok(rx)
}

/// Drive one chaos stream: collect chunks, invoke `kill` once the window
/// opens ([`CHAOS_KILL_AFTER_CHUNKS`] content chunks seen), and read on to
/// `[DONE]` / stream end. The caller judges the returned [`StreamWatch`].
fn run_chaos_stream(
    node: &Node,
    model: &str,
    kill: impl FnOnce() -> Result<()>,
) -> Result<StreamWatch> {
    let rx = open_chat_stream(node, model)?;
    let mut watch = StreamWatch::new(CHAOS_KILL_AFTER_CHUNKS);
    let mut kill = Some(kill);
    loop {
        match rx.recv_timeout(CHAOS_EVENT_TIMEOUT) {
            Ok(payload) => match watch.observe(&payload) {
                StreamAction::Continue => {}
                StreamAction::KillNow => {
                    if let Some(kill) = kill.take() {
                        kill()?;
                    }
                }
                StreamAction::Finished => break,
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => bail!(
                "no SSE event within {CHAOS_EVENT_TIMEOUT:?} ({} content chunks so far, kill \
                 fired: {}, errors: {:?})",
                watch.pieces.len(),
                watch.kill_at.is_some(),
                watch.errors
            ),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(watch)
}

/// One non-streaming greedy chat completion with `max_tokens`; returns
/// (text, usage.completion_tokens, finish_reason) — the chaos control run.
fn chat_full(
    node: &Node,
    model: &str,
    max_tokens: u32,
) -> Result<(String, Option<u64>, Option<String>)> {
    let body = json!({
        "model": model,
        "messages": [{ "role": "user", "content": PROMPT }],
        "stream": false,
        "temperature": 0,
        "max_tokens": max_tokens,
    });
    let v = node.post_json("/v1/chat/completions", &body, GEN_TIMEOUT)?;
    let text = v
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .with_context(|| format!("chat response lacks choices[0].message.content: {v}"))?
        .to_string();
    let tokens = v
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64);
    let finish = v
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok((text, tokens, finish))
}

/// The daemon's pid from its `daemon.json` (read at kill time).
fn daemon_pid(node: &Node) -> Result<u32> {
    Ok(node.daemon_json()?["pid"]
        .as_u64()
        .with_context(|| format!("{}: daemon.json has no numeric `pid`", node.label))?
        as u32)
}

/// Poll the head's active plan until `pred` holds; the timeout error
/// carries the last plan view. A temporarily missing plan (mid-swap) keeps
/// polling rather than failing.
fn wait_for_plan(
    node: &Node,
    window: Duration,
    what: &str,
    pred: impl Fn(&Value) -> bool,
) -> Result<Value> {
    let deadline = Instant::now() + window;
    let mut last;
    loop {
        match active_plan(node) {
            Ok(plan) if pred(&plan) => return Ok(plan),
            Ok(plan) => last = plan.to_string(),
            Err(e) => last = format!("(status/plan fetch failed: {e:#})"),
        }
        if Instant::now() >= deadline {
            bail!(
                "{}: no {what} within {window:?}; last plan view: {last}",
                node.label
            );
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// The plan's epoch number (Epoch serializes as its inner u64).
fn plan_epoch(plan: &Value) -> Result<u64> {
    plan["epoch"]
        .as_u64()
        .with_context(|| format!("plan lacks a numeric `epoch`: {plan}"))
}

/// Node ids of every assignment, in stage order (missing/odd shapes yield
/// an empty list — the callers' membership checks then fail loudly).
fn assignment_node_ids(plan: &Value) -> Vec<String> {
    plan["assignments"]
        .as_array()
        .map(|asgs| {
            asgs.iter()
                .filter_map(|a| a["node"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Which of the known workers hold an assignment in this plan. Empty = a
/// loud error (a distributed plan must involve at least one worker).
fn epoch_worker_ids(plan: &Value, worker_ids: &[&str]) -> Result<Vec<String>> {
    let in_epoch: Vec<String> = assignment_node_ids(plan)
        .into_iter()
        .filter(|id| worker_ids.contains(&id.as_str()))
        .collect();
    if in_epoch.is_empty() {
        bail!(
            "no known worker appears in the plan's assignments (workers: {worker_ids:?}): {plan}"
        );
    }
    Ok(in_epoch)
}

/// A distributed plan: PipelineParallel with at least two assignments (the
/// chaos section's id-based membership checks carry the specifics).
fn assert_distributed(plan: &Value) -> Result<()> {
    if norm_strategy(plan) != "pipelineparallel" {
        bail!(
            "plan strategy is {}, expected PipelineParallel: {plan}",
            plan["strategy"]
        );
    }
    if plan.assignments()?.len() < 2 {
        bail!("distributed plan has fewer than 2 assignments: {plan}");
    }
    Ok(())
}

/// Count `<digits>[ ]MB` figures in an error message — the structured-loss
/// and DoesNotFit contracts both carry exactly two (needs X MB, have Y MB).
fn count_mb_figures(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut count = 0;
    let mut from = 0;
    while let Some(pos) = text[from..].find("MB") {
        let at = from + pos;
        let mut j = at;
        if j > 0 && bytes[j - 1] == b' ' {
            j -= 1;
        }
        if j > 0 && bytes[j - 1].is_ascii_digit() {
            count += 1;
        }
        from = at + 2;
    }
    count
}

/// The chaos-2 stream must end with the structured error of the failure
/// lifecycle (docs/resilience.md step 3): it names the lost node (by name
/// or id) and carries BOTH MB figures.
fn structured_loss_check(text: &str, node_name: &str, node_id: &str) -> Result<()> {
    if !text.contains("lost") {
        bail!("the error does not say the node was lost: {text:?}");
    }
    let short_id: String = node_id.chars().take(8).collect();
    if !(text.contains(node_name) || text.contains(&short_id)) {
        bail!("the error does not name the lost node {node_name:?} (id {short_id}…): {text:?}");
    }
    let figures = count_mb_figures(text);
    if figures < 2 {
        bail!(
            "the error carries {figures} MB figure(s), expected both (needs X MB, have Y MB): \
             {text:?}"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// M4 scenario support: model dims from the GGUF header, cap math, and the
// score prediction (docs/scheduler-v1.md)
// ---------------------------------------------------------------------------

/// The two context lengths of the KV-shift scenario.
const M4_CTX_SMALL: u32 = 2048;
const M4_CTX_BIG: u32 = 16384;
/// `[debug] decode_tps_override` values: A fast, B slow (contract: "100.0
/// on A vs 50.0 on B").
const M4_DECODE_FAST: f64 = 100.0;
const M4_DECODE_SLOW: f64 = 50.0;
/// The fixed per-node compute/graph reserve the v1 scheduler subtracts
/// from usable memory (docs/scheduler-v1.md "Placement algorithm" §1;
/// `onebrain_scheduler::OVERHEAD_RESERVE_BYTES` — mirrored here because
/// xtask deliberately depends on no workspace crates).
const V1_OVERHEAD_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// What the M4 scenarios need to know about the model, read from the GGUF
/// header with the scheduler's own rules (crates/onebrain-scheduler/src/
/// dims.rs; docs/scheduler-v1.md "Placement algorithm" §1):
///
/// ```text
/// kv_rate    = 2 (K+V) × n_embd_kv × 2 bytes (f16)   per layer per token
/// n_embd_kv  = n_head_kv × (n_embd / n_head)          fallbacks -> n_embd
/// weights    = the whole tensor-data section (file minus header)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
struct SimModelDims {
    n_layers: u64,
    /// KV bytes one layer accrues per context token.
    kv_rate: u64,
    /// Total tensor bytes (every layer's weights, embedding and output
    /// included — they ride on their host layers in the scheduler too).
    total_weight_bytes: u64,
}

impl SimModelDims {
    /// Mean weight bytes per layer, rounded up — the scheduler's
    /// uniform-layer approximation (`ModelDims::mean_weight_bytes_per_layer`).
    fn mean_weight(&self) -> u64 {
        self.total_weight_bytes.div_ceil(self.n_layers.max(1))
    }

    /// Total memory at `ctx`: all weights + KV for every layer (the
    /// scheduler's auto-solo requirement, `ModelDims::total_required_bytes`).
    fn required_bytes(&self, ctx: u32) -> u64 {
        self.total_weight_bytes + self.n_layers * self.kv_rate * ctx as u64
    }

    /// Cost of one layer at `ctx` under the uniform-layer approximation.
    fn per_layer_cost(&self, ctx: u32) -> u64 {
        self.mean_weight() + self.kv_rate * ctx as u64
    }
}

/// The equal per-node BUDGET (bytes past the overhead reserve) for the M4
/// scenarios: 80% of the way from the ctx-2048 requirement to the ctx-16384
/// one. Below the 16k requirement (solo must fail there) yet far above the
/// 2k one (solo must succeed there), and high enough that per-node layer
/// capacity at 16k comfortably covers the biggest tilted share — the ±1
/// prediction assumes pure proportions, so capacity must never clamp
/// ([`check_m4_scenario`] verifies all of this against the real dims).
fn m4_budget(dims: &SimModelDims) -> u64 {
    let low = dims.required_bytes(M4_CTX_SMALL);
    let high = dims.required_bytes(M4_CTX_BIG);
    low + (high - low) * 8 / 10
}

/// The `[debug] usable_memory_override_bytes` value: the budget plus the
/// v1 overhead reserve the scheduler will subtract back out.
fn m4_cap(dims: &SimModelDims) -> u64 {
    V1_OVERHEAD_RESERVE_BYTES + m4_budget(dims)
}

/// The score prediction for the asymmetric split, from the contract's own
/// formula (docs/scheduler-v1.md "Placement algorithm" §2): shares are
/// proportional to `capacity_layers × (0.5 + 0.5 × decode/max_decode)`.
/// The caps are EQUAL, so the capacity factor cancels and only the decode
/// tilt remains: fast 1.0 vs slow 0.75. Returns fractional (fast, slow)
/// layer quotas.
fn expected_split(n_layers: u64) -> (f64, f64) {
    let f_fast = 0.5 + 0.5 * M4_DECODE_FAST / M4_DECODE_FAST;
    let f_slow = 0.5 + 0.5 * M4_DECODE_SLOW / M4_DECODE_FAST;
    let total = f_fast + f_slow;
    (
        n_layers as f64 * f_fast / total,
        n_layers as f64 * f_slow / total,
    )
}

/// Integer (fast, slow) counts the scheduler's largest-remainder rounding
/// produces from [`expected_split`] (ties on the fractional remainder go to
/// the higher score, i.e. fast) — used only to verify the scenario is
/// DECISIVE before running it.
fn predicted_counts(n_layers: u64) -> (u64, u64) {
    let (qf, qs) = expected_split(n_layers);
    let (mut f, mut s) = (qf.floor() as u64, qs.floor() as u64);
    if f + s < n_layers {
        // One leftover layer at most with two nodes; it goes to the larger
        // fractional remainder, ties to the higher score (fast).
        if qs - s as f64 > qf - f as f64 {
            s += 1;
        } else {
            f += 1;
        }
    }
    (f, s)
}

/// Every invariant the M4 scenarios rely on, checked against the real
/// model before any daemon restarts — so a failure is one clear message,
/// not a flaky assertion later.
fn check_m4_scenario(dims: &SimModelDims) -> Result<()> {
    if dims.n_layers < 2 {
        bail!(
            "model has {} transformer layer(s); the M4 scenarios need at least 2 to split",
            dims.n_layers
        );
    }
    if dims.kv_rate == 0 {
        bail!("model KV rate is 0 bytes/token/layer; ctx cannot shift its memory need");
    }
    let budget = m4_budget(dims);
    let low = dims.required_bytes(M4_CTX_SMALL);
    let high = dims.required_bytes(M4_CTX_BIG);
    if low > budget {
        bail!("budget {budget} B cannot hold the ctx-{M4_CTX_SMALL} requirement {low} B solo");
    }
    if high <= budget {
        bail!("budget {budget} B still holds the ctx-{M4_CTX_BIG} requirement {high} B solo");
    }
    let (fast, slow) = predicted_counts(dims.n_layers);
    let cap_layers = budget / dims.per_layer_cost(M4_CTX_BIG);
    if cap_layers < fast || 2 * cap_layers < dims.n_layers {
        bail!(
            "per-node capacity at ctx {M4_CTX_BIG} is {cap_layers} layers; the predicted \
             split {fast}/{slow} of {} would be capacity-clamped, so the ±1 score \
             assertion would not be testing the tilt",
            dims.n_layers
        );
    }
    if fast <= slow {
        bail!(
            "decode override {M4_DECODE_FAST} vs {M4_DECODE_SLOW} rounds to the indecisive \
             split {fast}/{slow} for a {}-layer model, so 'A takes MORE layers' cannot be \
             asserted; the scenario is calibrated for stories260K (5 layers) — delete \
             target-smoke/ and re-run so `cargo xtask sim` downloads it again",
            dims.n_layers
        );
    }
    Ok(())
}

/// The layer counts of the two-assignment M4 plan, matched to the daemons
/// by the node ids learned at pairing time: `(A's layers, B's layers)`.
fn layers_by_node(plan: &Value, a_id: &str, b_id: &str) -> Result<(u64, u64)> {
    let mut a_layers = None;
    let mut b_layers = None;
    for asg in plan.assignments()? {
        let node = asg["node"]
            .as_str()
            .with_context(|| format!("assignment lacks a string `node`: {asg}"))?;
        let layers = assignment_layers(asg)?;
        if node == a_id {
            a_layers = Some(layers);
        } else if node == b_id {
            b_layers = Some(layers);
        } else {
            bail!("assignment names unknown node {node} (A is {a_id}, B is {b_id}): {plan}");
        }
    }
    Ok((
        a_layers.with_context(|| format!("no assignment for A ({a_id}): {plan}"))?,
        b_layers.with_context(|| format!("no assignment for B ({b_id}): {plan}"))?,
    ))
}

/// Layer count of one assignment (`layers.end - layers.start`).
fn assignment_layers(asg: &Value) -> Result<u64> {
    match (
        asg.pointer("/layers/start").and_then(Value::as_u64),
        asg.pointer("/layers/end").and_then(Value::as_u64),
    ) {
        (Some(start), Some(end)) if end > start => Ok(end - start),
        _ => bail!("assignment has an empty or missing layer range: {asg}"),
    }
}

/// The plan must have been computed at the ctx the restart configured —
/// pins that the config rewrite actually moved the daemon's ctx_len.
fn assert_plan_ctx(plan: &Value, ctx: u32) -> Result<()> {
    match plan["ctx_len"].as_u64() {
        Some(got) if got == ctx as u64 => Ok(()),
        other => bail!("plan carries ctx_len {other:?}, expected {ctx}: {plan}"),
    }
}

/// Tiny extension trait so the M4 assertions read off the `assignments`
/// array without repeating the shape check.
trait PlanExt {
    fn assignments(&self) -> Result<&Vec<Value>>;
}

impl PlanExt for Value {
    fn assignments(&self) -> Result<&Vec<Value>> {
        self["assignments"]
            .as_array()
            .with_context(|| format!("plan lacks an `assignments` array: {self}"))
    }
}

// ---------------------------------------------------------------------------
// Minimal GGUF metadata reader (M4 scenario support)
// ---------------------------------------------------------------------------
//
// xtask deliberately depends on no workspace crates (it builds them), so
// the few header facts the M4 cap math needs are read with a standalone
// ~parser mirroring `onebrain-models::gguf` + `onebrain-scheduler::dims`:
// scalar metadata + the header length, nothing else. Spec:
// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

/// Read [`SimModelDims`] from a GGUF file. Starts with a 1 MiB prefix and
/// doubles until the header fits (the same strategy as the daemon's own
/// header reads).
fn read_gguf_dims(path: &Path) -> Result<SimModelDims> {
    use std::io::Read;
    let file_len = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    let mut want: u64 = (1u64 << 20).min(file_len);
    loop {
        let mut bytes = Vec::with_capacity(want as usize);
        std::fs::File::open(path)
            .and_then(|f| f.take(want).read_to_end(&mut bytes))
            .with_context(|| format!("reading {}", path.display()))?;
        match parse_gguf_dims(&bytes, file_len) {
            Ok(dims) => return Ok(dims),
            Err(e) if want < file_len && format!("{e:#}").contains(GGUF_TRUNCATED) => {
                want = (want * 2).min(file_len);
            }
            Err(e) => {
                return Err(e.context(format!("parsing the GGUF header of {}", path.display())))
            }
        }
    }
}

/// Marker in truncation errors so [`read_gguf_dims`] knows a longer prefix
/// may still succeed (vs real corruption, which never will).
const GGUF_TRUNCATED: &str = "gguf header truncated";

/// Byte cursor over a GGUF header prefix.
struct GgufCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> GgufCursor<'a> {
    fn take(&mut self, n: u64) -> Result<&'a [u8]> {
        let n = usize::try_from(n).ok().with_context(|| GGUF_TRUNCATED)?;
        let end = self.pos.checked_add(n).with_context(|| GGUF_TRUNCATED)?;
        if end > self.bytes.len() {
            bail!(GGUF_TRUNCATED);
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// GGUF string: u64 length + UTF-8 bytes (lossy: keys are ASCII).
    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }

    /// Read one metadata value of `ty`, returning `Some(v)` for the
    /// integer/bool types (the only ones the dims math needs) and `None`
    /// for everything else (floats, strings, arrays — skipped over).
    fn value(&mut self, ty: u32) -> Result<Option<u64>> {
        Ok(match ty {
            0 | 7 => Some(self.take(1)?[0] as u64),          // u8, bool
            1 => u64::try_from(self.take(1)?[0] as i8).ok(), // i8
            2 => Some(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => u64::try_from(i16::from_le_bytes(self.take(2)?.try_into().unwrap())).ok(),
            4 => Some(self.u32()? as u64),
            5 => u64::try_from(self.u32()? as i32).ok(),
            6 => {
                self.take(4)?; // f32
                None
            }
            8 => {
                let len = self.u64()?;
                self.take(len)?;
                None
            }
            9 => {
                let elem_ty = self.u32()?;
                let count = self.u64()?;
                match gguf_fixed_width(elem_ty) {
                    // Fixed-width elements: skip the whole block at once.
                    Some(width) => {
                        let total = count.checked_mul(width).with_context(|| GGUF_TRUNCATED)?;
                        self.take(total)?;
                    }
                    // Strings / nested arrays: element by element.
                    None => {
                        for _ in 0..count {
                            self.value(elem_ty)?;
                        }
                    }
                }
                None
            }
            10 => Some(self.u64()?),
            11 => u64::try_from(self.u64()? as i64).ok(),
            12 => {
                self.take(8)?; // f64
                None
            }
            other => bail!("unknown GGUF metadata value type {other}; the file may be corrupt"),
        })
    }
}

/// Byte width of a fixed-size GGUF metadata value type (None for string /
/// array).
fn gguf_fixed_width(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Parse the dims out of a GGUF header prefix. `file_len` closes the
/// tensor-data section (total weights = file minus the aligned header).
fn parse_gguf_dims(bytes: &[u8], file_len: u64) -> Result<SimModelDims> {
    let mut cur = GgufCursor { bytes, pos: 0 };
    let magic = cur.u32()?;
    if magic != 0x4655_4747 {
        bail!("not a GGUF file (magic {magic:#010x})");
    }
    let version = cur.u32()?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version} (this reader handles v2/v3)");
    }
    let tensor_count = cur.u64()?;
    let kv_count = cur.u64()?;

    let mut arch: Option<String> = None;
    let mut scalars: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for _ in 0..kv_count {
        let key = cur.string()?;
        let ty = cur.u32()?;
        if ty == 8 && key == "general.architecture" {
            arch = Some(cur.string()?);
            continue;
        }
        if let Some(v) = cur.value(ty)? {
            scalars.insert(key, v);
        }
    }
    for _ in 0..tensor_count {
        let _name = cur.string()?;
        let n_dims = cur.u32()?;
        if n_dims > 8 {
            bail!("tensor declares {n_dims} dimensions; the file may be corrupt");
        }
        for _ in 0..n_dims {
            cur.u64()?;
        }
        cur.u32()?; // ggml type
        cur.u64()?; // offset
    }

    // Tensor data starts at the header end rounded up to the alignment.
    let alignment = scalars
        .get("general.alignment")
        .copied()
        .filter(|a| *a > 0)
        .unwrap_or(32);
    let data_offset = (cur.pos as u64).div_ceil(alignment) * alignment;
    if file_len < data_offset {
        bail!("file is shorter ({file_len} B) than its own header ({data_offset} B)");
    }

    let arch = arch.context("header declares no general.architecture")?;
    let get = |suffix: &str| scalars.get(&format!("{arch}.{suffix}")).copied();
    let n_layers = get("block_count")
        .filter(|n| *n > 0)
        .with_context(|| format!("header lacks {arch}.block_count"))?;
    let n_embd = get("embedding_length")
        .filter(|n| *n > 0)
        .with_context(|| format!("header lacks {arch}.embedding_length"))?;
    // GQA-aware KV width with the scheduler's conservative fallbacks
    // (crates/onebrain-scheduler/src/dims.rs): anything missing degrades
    // toward n_embd.
    let n_embd_kv = match get("attention.head_count").filter(|n| *n > 0) {
        Some(n_head) => {
            let head_dim = n_embd / n_head;
            let n_head_kv = get("attention.head_count_kv")
                .filter(|n| *n > 0)
                .unwrap_or(n_head);
            (n_head_kv * head_dim).min(n_embd).max(1)
        }
        None => n_embd,
    };

    Ok(SimModelDims {
        n_layers,
        kv_rate: 2 * 2 * n_embd_kv, // K+V, f16
        total_weight_bytes: file_len - data_offset,
    })
}

// ---------------------------------------------------------------------------
// Cap math and sandbox config
// ---------------------------------------------------------------------------

/// The ctx the distribute phase plans at: the daemon's `ctx_len` default
/// (onebraind::config; the phase's sandbox configs set no ctx override).
const M3_DEFAULT_CTX: u32 = 4096;

/// Per-node BUDGET (bytes past the v1 overhead reserve) for the distribute
/// phase: exactly `n_layers - 1` layer-costs at the default ctx. One layer
/// short of solo — the head cannot hold all layers (weights + KV always
/// exceed `n-1` mean layer costs), so auto-distribution must engage — while
/// two such nodes pool `2(n-1) >= n` layers, and each node's `ceil(n/2)`
/// share fits its own capacity for every `n >= 2`.
fn m3_distribute_budget(dims: &SimModelDims) -> u64 {
    (dims.n_layers - 1) * dims.per_layer_cost(M3_DEFAULT_CTX)
}

/// The `[debug] usable_memory_override_bytes` value for the distribute
/// phase: the budget plus the v1 overhead reserve the scheduler subtracts
/// back out (docs/scheduler-v1.md "Placement algorithm" §1).
fn m3_distribute_cap(dims: &SimModelDims) -> u64 {
    V1_OVERHEAD_RESERVE_BYTES + m3_distribute_budget(dims)
}

/// Optional sandbox-config knobs a scenario phase turns on (`None` renders
/// nothing, keeping the config byte-identical to the pre-M4 one). All
/// `[debug]` knobs are test-only (docs/distributed.md, docs/scheduler-v1.md):
/// they change what the node REPORTS and BUDGETS, never real allocation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct SimKnobs {
    /// `[debug] usable_memory_override_bytes`: the memory cap.
    cap_bytes: Option<u64>,
    /// `[debug] decode_tps_override`: replaces the measured decode
    /// throughput in this node's profile (M4 asymmetric scenario).
    decode_tps_override: Option<f64>,
    /// Top-level `ctx_len`: the context the daemon plans and loads at (the
    /// load body carries no ctx override, so ctx moves via config+restart).
    ctx_len: Option<u32>,
    /// `[debug] decode_delay_ms`: the engine host sleeps this long per
    /// emitted token (docs/resilience.md sim hooks — keeps a tiny model's
    /// stream open long enough to kill a worker mid-generation).
    decode_delay_ms: Option<u64>,
}

impl SimKnobs {
    /// The M3 shape: a memory cap and nothing else.
    fn capped(cap_bytes: u64) -> SimKnobs {
        SimKnobs {
            cap_bytes: Some(cap_bytes),
            ..SimKnobs::default()
        }
    }
}

/// The sandbox `config.toml`: the pair-sim determinism switches plus
/// whatever [`SimKnobs`] the current scenario phase needs.
fn render_sim_config(
    name: &str,
    port: u16,
    mesh_port: u16,
    netem: bool,
    knobs: SimKnobs,
) -> String {
    // The mesh UDP port is pinned so peers' stored addresses stay valid
    // across the restart scenario (with mDNS/relays off there is no other
    // way for a restarted daemon to be found). Under netem each daemon
    // lives in its own namespace whose loopback is ISOLATED — the mesh
    // must bind all interfaces so the veth address carries the traffic;
    // loopback-only is fine (and tighter) on a shared host.
    let mesh_host = if netem { "0.0.0.0" } else { "127.0.0.1" };
    let mut cfg = format!(
        "node_name = \"{name}\"\n\
         api_bind = \"127.0.0.1:{port}\"\n"
    );
    if let Some(ctx) = knobs.ctx_len {
        // Top-level key: must precede the [mesh] table (TOML validity).
        cfg.push_str(&format!("ctx_len = {ctx}\n"));
    }
    cfg.push_str(&format!(
        "\n[mesh]\n\
         enable_mdns = false\n\
         enable_relays = false\n\
         bind_addr = \"{mesh_host}:{mesh_port}\"\n"
    ));
    if knobs.cap_bytes.is_some()
        || knobs.decode_tps_override.is_some()
        || knobs.decode_delay_ms.is_some()
    {
        cfg.push_str("\n[debug]\n");
        if let Some(cap) = knobs.cap_bytes {
            cfg.push_str(&format!("usable_memory_override_bytes = {cap}\n"));
        }
        if let Some(decode) = knobs.decode_tps_override {
            // {:?} keeps the decimal point (`100.0`), which TOML requires
            // for a float value.
            cfg.push_str(&format!("decode_tps_override = {decode:?}\n"));
        }
        if let Some(delay) = knobs.decode_delay_ms {
            cfg.push_str(&format!("decode_delay_ms = {delay}\n"));
        }
    }
    cfg
}

fn write_config(node: &Node, mesh_port: u16, netem: bool, knobs: SimKnobs) -> Result<()> {
    let path = node.home.join("config").join("config.toml");
    std::fs::write(
        &path,
        render_sim_config(node.name, node.port, mesh_port, netem, knobs),
    )
    .with_context(|| format!("writing {}", path.display()))
}

// ---------------------------------------------------------------------------
// Daemon driving
// ---------------------------------------------------------------------------

/// Pair A (window host) and B (joiner); returns (A as seen by B, B as seen
/// by A), i.e. the ids each side lists the other under.
fn pair(a: &Node, b: &Node) -> Result<(PeerRef, PeerRef)> {
    let mut stream = a.pair_start()?;
    let first = stream.next(WINDOW_TIMEOUT, "window")?;
    if first["status"] != "window" {
        bail!("first pair/start event is not `window`: {first}");
    }
    let code = first["code"]
        .as_str()
        .with_context(|| format!("window event lacks `code`: {first}"))?
        .to_string();
    let ticket = first["ticket"]
        .as_str()
        .with_context(|| format!("window event lacks `ticket`: {first}"))?
        .to_string();

    let peer_a = peer_ref(&b.post_json(
        "/api/internal/pair/join",
        &json!({ "target": &ticket, "code": &code }),
        JOIN_TIMEOUT,
    )?)?;

    let peer_b = loop {
        let event = stream.next(PAIRED_TIMEOUT, "paired")?;
        match event["status"].as_str() {
            Some("attempt") => continue,
            Some("paired") => break peer_ref(&event)?,
            other => bail!("expected a `paired` event, got {other:?}: {event}"),
        }
    };
    Ok((peer_a, peer_b))
}

/// Both sides must list the other as `connected` (heartbeats flowing).
fn wait_connected(
    a: &Node,
    b: &Node,
    peer_a: &PeerRef,
    peer_b: &PeerRef,
    when: &str,
) -> Result<()> {
    for (node, other) in [(a, peer_b), (b, peer_a)] {
        node.wait_peer(
            &other.id,
            CONNECTED_TIMEOUT,
            &format!("state=connected {when}"),
            |p| p["state"] == "connected",
        )?;
    }
    Ok(())
}

/// `POST /api/internal/load`, read the whole NDJSON stream, require the
/// `planning` line, the terminal `ready`, and return the `plan` object from
/// the `{"status":"plan"}` line.
fn load_model(node: &Node, body: &Value) -> Result<Value> {
    let (code, text) = node.http("POST", "/api/internal/load", Some(body), LOAD_TIMEOUT)?;
    if code != 200 {
        bail!("load on {} answered HTTP {code}: {text}", node.label);
    }
    let lines: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let statuses: Vec<&str> = lines.iter().filter_map(|l| l["status"].as_str()).collect();
    let last = lines
        .last()
        .with_context(|| format!("load stream carried no parsable NDJSON lines; body: {text}"))?;
    match last["status"].as_str() {
        Some("ready") => {}
        Some("error") => bail!(
            "load failed: {}",
            last["message"].as_str().unwrap_or("(no message)")
        ),
        other => {
            bail!("load stream ended with status {other:?}, expected `ready`; saw {statuses:?}")
        }
    }
    if !statuses.contains(&"planning") {
        bail!("load stream had no {{\"status\":\"planning\"}} line; saw {statuses:?}");
    }
    lines
        .iter()
        .find(|l| l["status"] == "plan")
        .map(|l| l["plan"].clone())
        .with_context(|| {
            format!("load stream had no {{\"status\":\"plan\"}} line; saw {statuses:?}")
        })
}

/// The active plan from `GET /api/internal/status` (either field spelling).
fn active_plan(node: &Node) -> Result<Value> {
    let status = node.get_json("/api/internal/status")?;
    let plan = status
        .get("plan")
        .or_else(|| status.get("active_plan"))
        .cloned()
        .unwrap_or(Value::Null);
    if plan.is_null() {
        bail!("/api/internal/status reports no active plan: {status}");
    }
    // The daemon reports a view object {role, plan: {strategy, ...},
    // explanation}; flatten it so callers read strategy/assignments and
    // explanation off one level.
    if plan.get("strategy").is_none() {
        if let Some(inner) = plan.get("plan").filter(|v| v.is_object()) {
            let mut merged = inner.clone();
            for key in ["explanation", "role"] {
                if let Some(v) = plan.get(key) {
                    merged[key] = v.clone();
                }
            }
            return Ok(merged);
        }
    }
    Ok(plan)
}

/// The loaded model's name as the daemon reports it (`local:<stem>` for
/// path loads) — generation requests must use this name.
fn model_name(node: &Node) -> Result<String> {
    let status = node.get_json("/api/internal/status")?;
    status
        .pointer("/model/name")
        .and_then(|n| n.as_str())
        .map(str::to_string)
        .with_context(|| format!("status reports no model.name after load: {status}"))
}

/// One OpenAI non-streaming chat completion, greedy (temperature 0), fixed
/// prompt, 12 tokens — the text both phases must agree on byte-for-byte.
fn chat_text(node: &Node, model: &str) -> Result<String> {
    let body = json!({
        "model": model,
        "messages": [{ "role": "user", "content": PROMPT }],
        "stream": false,
        "temperature": 0,
        "max_tokens": 12
    });
    let v = node.post_json("/v1/chat/completions", &body, GEN_TIMEOUT)?;
    let text = v
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .with_context(|| format!("chat response lacks choices[0].message.content: {v}"))?;
    if text.is_empty() {
        bail!("chat completion returned empty text: {v}");
    }
    Ok(text.to_string())
}

/// The Ollama dialect must stream NDJSON ending `done:true` too.
fn ollama_streams(node: &Node, model: &str) -> Result<()> {
    let body = json!({
        "model": model,
        "prompt": PROMPT,
        "stream": true,
        "options": { "num_predict": 8 }
    });
    let (code, text) = node.http("POST", "/api/generate", Some(&body), GEN_TIMEOUT)?;
    if code != 200 {
        bail!("/api/generate answered HTTP {code}: {text}");
    }
    let last = text
        .lines()
        .rfind(|l| !l.trim().is_empty())
        .context("empty NDJSON body from /api/generate")?;
    let v: Value = serde_json::from_str(last)
        .with_context(|| format!("final NDJSON line unparsable: {last}"))?;
    if v["done"] != Value::Bool(true) {
        bail!("final NDJSON line lacks done:true: {v}");
    }
    Ok(())
}

/// Stop, rewrite the config with the given knobs, start again, wait
/// healthy. `SimKnobs::default()` is the uncapped restart of the M3 steps.
fn restart_with(node: &Node, mesh_port: u16, netem: bool, knobs: SimKnobs) -> Result<()> {
    let out = node.onebrain(&["stop"])?;
    if !out.status.success() {
        bail!(
            "`onebrain stop` on {} failed: {}",
            node.label,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let deadline = Instant::now() + STOP_TIMEOUT;
    while node.try_status().is_ok() {
        if Instant::now() >= deadline {
            bail!(
                "{} still answering {STOP_TIMEOUT:?} after `onebrain stop`",
                node.label
            );
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    start_with(node, mesh_port, netem, knobs)
}

/// Write the sandbox config and start a daemon that is NOT currently
/// running (a fresh node, or one that was killed -9 / stopped) — the
/// bottom half of [`restart_with`], and the chaos section's revive path.
fn start_with(node: &Node, mesh_port: u16, netem: bool, knobs: SimKnobs) -> Result<()> {
    write_config(node, mesh_port, netem, knobs)?;
    let out = node.onebrain(&["up"])?;
    node.wait_healthy().map_err(|e| {
        anyhow!(
            "{e:#}\n`onebrain up` exit code {:?}\nstdout: {}\nstderr: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout).trim(),
            String::from_utf8_lossy(&out.stderr).trim()
        )
    })
}

// ---------------------------------------------------------------------------
// Plan assertions (pure over the plan JSON)
// ---------------------------------------------------------------------------

/// Strategy string normalized for comparison ("PipelineParallel",
/// "pipeline_parallel", "pipeline-parallel" all → "pipelineparallel").
fn norm_strategy(plan: &Value) -> String {
    plan["strategy"]
        .as_str()
        .unwrap_or_default()
        .to_ascii_lowercase()
        .replace(['_', '-', ' '], "")
}

fn assert_pipeline(plan: &Value) -> Result<()> {
    if norm_strategy(plan) != "pipelineparallel" {
        bail!(
            "plan strategy is {}, expected PipelineParallel: {plan}",
            plan["strategy"]
        );
    }
    let assignments = plan["assignments"]
        .as_array()
        .with_context(|| format!("plan lacks an `assignments` array: {plan}"))?;
    if assignments.len() != 2 {
        bail!(
            "plan has {} assignments, expected 2 (one per node): {plan}",
            assignments.len()
        );
    }
    for a in assignments {
        let start = a.pointer("/layers/start").and_then(|v| v.as_u64());
        let end = a.pointer("/layers/end").and_then(|v| v.as_u64());
        if !matches!((start, end), (Some(s), Some(e)) if e > s) {
            bail!("assignment has an empty or missing layer range: {a}");
        }
    }
    Ok(())
}

fn assert_solo(plan: &Value) -> Result<()> {
    if norm_strategy(plan) != "solo" {
        bail!(
            "plan strategy is {}, expected Solo: {plan}",
            plan["strategy"]
        );
    }
    let n = plan["assignments"].as_array().map_or(0, Vec::len);
    if n != 1 {
        bail!("solo plan has {n} assignments, expected exactly 1: {plan}");
    }
    Ok(())
}

/// `explain: true` was in every load body, so every plan line must carry
/// non-empty prose.
fn assert_explained(plan: &Value) -> Result<()> {
    match plan["explanation"].as_str() {
        Some(s) if !s.trim().is_empty() => Ok(()),
        _ => bail!("plan lacks a non-empty `explanation` (explain:true was requested): {plan}"),
    }
}

// ---------------------------------------------------------------------------
// Socket scan (contract scenario 3)
// ---------------------------------------------------------------------------

/// One listening TCP socket attributed to a pid.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Listener {
    pid: u32,
    host: String,
    port: u16,
}

/// Scan both daemons: every TCP listener of each pid must be loopback; only
/// the api bind may persist outside a distributed session.
/// `allow_extra_loopback` = a distributed session is active, so the head's
/// per-epoch loopback rpc-bridge listener is legitimate (accept-loop, ADR
/// 0004 amendment) — non-loopback stays forbidden ALWAYS.
fn socket_scan(nodes: &[&Node], allow_extra_loopback: bool) -> Result<()> {
    for node in nodes {
        let pid = node.daemon_json()?["pid"]
            .as_u64()
            .with_context(|| format!("{}: daemon.json has no numeric `pid`", node.label))?
            as u32;
        let listeners = scan_listeners(node, pid)?;
        check_listeners(&listeners, pid, node.port, node.label, allow_extra_loopback)?;
    }
    Ok(())
}

/// Enumerate listening TCP sockets visible to this node (in netem mode the
/// scan runs inside the daemon's namespace via `ip netns exec`).
fn scan_listeners(node: &Node, pid: u32) -> Result<Vec<Listener>> {
    if cfg!(windows) {
        let out = node
            .wrap(OsStr::new("netstat"))
            .args(["-ano"])
            .output()
            .context("failed to spawn netstat")?;
        if !out.status.success() {
            bail!(
                "netstat -ano failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(parse_netstat(&String::from_utf8_lossy(&out.stdout)))
    } else if cfg!(target_os = "macos") {
        let out = node
            .wrap(OsStr::new("lsof"))
            .args(["-nP", "-iTCP", "-sTCP:LISTEN", "-a", "-p", &pid.to_string()])
            .output()
            .context("failed to spawn lsof")?;
        // lsof exits 1 when nothing matches; an empty result is judged by
        // check_listeners (the api listener must exist), not here.
        Ok(parse_lsof(&String::from_utf8_lossy(&out.stdout)))
    } else {
        let out = node
            .wrap(OsStr::new("ss"))
            .args(["-ltnp"])
            .output()
            .context("failed to spawn ss (iproute2)")?;
        if !out.status.success() {
            bail!(
                "ss -ltnp failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(parse_ss(&String::from_utf8_lossy(&out.stdout)))
    }
}

/// The scan verdict, pure over the listener list. The daemon's own api
/// listener must be found (proving the scan and pid attribution work at
/// all); beyond it, any non-loopback listener or any extra loopback
/// listener is a contract violation.
fn check_listeners(
    listeners: &[Listener],
    pid: u32,
    api_port: u16,
    label: &str,
    allow_extra_loopback: bool,
) -> Result<()> {
    let mine: Vec<&Listener> = listeners.iter().filter(|l| l.pid == pid).collect();
    let api_seen = mine
        .iter()
        .any(|l| l.port == api_port && is_loopback(&l.host));
    if !api_seen {
        bail!(
            "socket scan for {label} (pid {pid}) did not find its own api listener on \
             loopback:{api_port} — the scan is incomplete or the daemon bound elsewhere; \
             listeners attributed to the pid: {mine:?}"
        );
    }
    for l in &mine {
        if !is_loopback(&l.host) {
            bail!(
                "{label} (pid {pid}) LISTENS on non-loopback {}:{} — forbidden \
                 (docs/distributed.md: no raw TCP listener anywhere)",
                l.host,
                l.port
            );
        }
        if l.port != api_port && !allow_extra_loopback {
            bail!(
                "{label} (pid {pid}) has an unexpected loopback listener on {}:{} — outside a \
                 distributed session only the api bind :{api_port} may persist (per-epoch rpc \
                 bridge listeners must be gone after teardown)",
                l.host,
                l.port
            );
        }
    }
    Ok(())
}

/// Windows `netstat -ano`: `  TCP    127.0.0.1:11435   0.0.0.0:0   LISTENING   4242`.
/// Windows prints "TCP" for IPv6 rows too (the address form differs).
fn parse_netstat(text: &str) -> Vec<Listener> {
    text.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 5 || f[0] != "TCP" || f[3] != "LISTENING" {
                return None;
            }
            let (host, port) = split_host_port(f[1])?;
            let pid = f[4].parse().ok()?;
            Some(Listener { pid, host, port })
        })
        .collect()
}

/// Linux `ss -ltnp`:
/// `LISTEN 0 128 127.0.0.1:38471 0.0.0.0:* users:(("onebrain",pid=4242,fd=9))`.
/// A socket may name several pids; one Listener per pid.
fn parse_ss(text: &str) -> Vec<Listener> {
    let mut out = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 4 || f[0] != "LISTEN" {
            continue;
        }
        let Some((host, port)) = split_host_port(f[3]) else {
            continue;
        };
        for pid in extract_pids(line) {
            out.push(Listener {
                pid,
                host: host.clone(),
                port,
            });
        }
    }
    out
}

/// Every `pid=<digits>` occurrence in an `ss -p` process blob.
fn extract_pids(line: &str) -> Vec<u32> {
    let mut pids = Vec::new();
    let mut rest = line;
    while let Some(idx) = rest.find("pid=") {
        rest = &rest[idx + 4..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        if let Ok(pid) = digits.parse() {
            pids.push(pid);
        }
    }
    pids
}

/// macOS `lsof -nP -iTCP -sTCP:LISTEN -a -p <pid>`:
/// `onebrain 4242 user 9u IPv4 0x0 0t0 TCP 127.0.0.1:52000 (LISTEN)`.
fn parse_lsof(text: &str) -> Vec<Listener> {
    text.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 4 || f.last() != Some(&"(LISTEN)") || f[0] == "COMMAND" {
                return None;
            }
            let pid = f[1].parse().ok()?;
            let (host, port) = split_host_port(f[f.len() - 2])?;
            Some(Listener { pid, host, port })
        })
        .collect()
}

/// `"[::1]:80"` → `("::1", 80)`; `"127.0.0.1:80"` → `("127.0.0.1", 80)`;
/// scope suffixes (`%lo`) stripped. `None` when the port isn't numeric
/// (e.g. `*:*` UDP rows).
fn split_host_port(s: &str) -> Option<(String, u16)> {
    let (host, port) = s.rsplit_once(':')?;
    let port: u16 = port.parse().ok()?;
    // Scope suffix first (it may follow a closing bracket), brackets second.
    let host = host.split('%').next().unwrap_or(host);
    let host = host.trim_start_matches('[').trim_end_matches(']');
    Some((host.to_string(), port))
}

/// Loopback per the contract: 127.0.0.0/8 or ::1. Wildcards (`0.0.0.0`,
/// `::`, `*`) are NOT loopback.
fn is_loopback(host: &str) -> bool {
    host == "::1" || host == "localhost" || host.starts_with("127.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m3_cap_makes_solo_fail_and_pooled_fit_under_v1_budgets() {
        // The primary smoke model plus a sweep of other shapes: for each,
        // the (n-1)-layer-cost budget must fail solo (weights + KV exceed
        // it) while two such nodes pool enough capacity, with each node's
        // ceil(n/2) share within its own capacity.
        let shapes = [
            stories260k_dims(),
            SimModelDims {
                n_layers: 2,
                kv_rate: 1152,
                total_weight_bytes: 8_300_000,
            },
            SimModelDims {
                n_layers: 6,
                kv_rate: 1152,
                total_weight_bytes: 8_300_000,
            },
            SimModelDims {
                n_layers: 32,
                kv_rate: 0, // no KV growth at all still distributes
                total_weight_bytes: 750_000_000,
            },
        ];
        for dims in shapes {
            let budget = m3_distribute_budget(&dims);
            let capacity = budget / dims.per_layer_cost(M3_DEFAULT_CTX);
            assert!(
                budget < dims.required_bytes(M3_DEFAULT_CTX),
                "{dims:?}: budget {budget} must NOT hold the model solo"
            );
            assert_eq!(capacity, dims.n_layers - 1, "{dims:?}");
            assert!(2 * capacity >= dims.n_layers, "{dims:?}: pooled must fit");
            assert!(
                capacity >= dims.n_layers.div_ceil(2),
                "{dims:?}: the bigger half-share must fit one node"
            );
            assert_eq!(m3_distribute_cap(&dims), 536_870_912 + budget);
        }
        // Pinned numbers for stories260K at the daemon's default ctx 4096:
        // cost = 234,240 + 128×4096 = 758,528; budget = 4 × cost.
        let dims = stories260k_dims();
        assert_eq!(dims.per_layer_cost(M3_DEFAULT_CTX), 758_528);
        assert_eq!(m3_distribute_budget(&dims), 3_034_112);
        assert_eq!(dims.required_bytes(M3_DEFAULT_CTX), 3_792_640);
    }

    #[test]
    fn sim_config_caps_only_when_asked() {
        let capped = render_sim_config("sim-a", 12345, 23456, false, SimKnobs::capped(830_000));
        assert!(capped.contains("node_name = \"sim-a\""));
        assert!(capped.contains("api_bind = \"127.0.0.1:12345\""));
        assert!(capped.contains("enable_mdns = false"));
        assert!(capped.contains("enable_relays = false"));
        assert!(capped.contains("bind_addr = \"127.0.0.1:23456\""));
        assert!(capped.contains("[debug]"));
        assert!(capped.contains("usable_memory_override_bytes = 830000"));
        // The M4 knobs stay out unless asked for.
        assert!(!capped.contains("decode_tps_override"));
        assert!(!capped.contains("ctx_len"));
        // Top-level keys stay above the tables (TOML validity).
        assert!(capped.find("node_name").unwrap() < capped.find("[mesh]").unwrap());

        let uncapped = render_sim_config("sim-b", 1, 2, false, SimKnobs::default());
        assert!(!uncapped.contains("[debug]"));
        assert!(!uncapped.contains("usable_memory_override_bytes"));
        // The pinned mesh port survives the uncapped rewrite (restart
        // scenario: stored peer addresses must stay valid).
        assert!(uncapped.contains("bind_addr = \"127.0.0.1:2\""));
    }

    #[test]
    fn sim_config_renders_the_m4_knobs() {
        let cfg = render_sim_config(
            "sim-a",
            12345,
            23456,
            false,
            SimKnobs {
                cap_bytes: Some(546_692_864),
                decode_tps_override: Some(100.0),
                ctx_len: Some(16384),
                ..SimKnobs::default()
            },
        );
        // ctx_len is a top-level key: it must precede the [mesh] table.
        assert!(cfg.contains("ctx_len = 16384\n"));
        assert!(cfg.find("ctx_len").unwrap() < cfg.find("[mesh]").unwrap());
        // Float syntax: TOML requires the decimal point.
        assert!(cfg.contains("decode_tps_override = 100.0\n"));
        assert!(cfg.contains("usable_memory_override_bytes = 546692864\n"));
        // Both [debug] keys live under one [debug] header.
        assert_eq!(cfg.matches("[debug]").count(), 1);
        assert!(cfg.find("[mesh]").unwrap() < cfg.find("[debug]").unwrap());

        // A decode override alone still opens the [debug] table.
        let cfg = render_sim_config(
            "sim-b",
            1,
            2,
            false,
            SimKnobs {
                decode_tps_override: Some(50.0),
                ..SimKnobs::default()
            },
        );
        assert!(cfg.contains("[debug]\ndecode_tps_override = 50.0\n"));
        assert!(!cfg.contains("usable_memory_override_bytes"));
    }

    // ---- M4 scenario math ------------------------------------------------

    /// stories260K.gguf, the sim's pinned first-choice model: 5 layers,
    /// n_embd 64, 8 heads / 4 KV heads -> n_embd_kv 32 -> kv_rate 128
    /// B/token/layer; tensor-data section 1,171,200 B of the 1,185,376 B
    /// file. These constants pin the whole M4 cap derivation end to end.
    fn stories260k_dims() -> SimModelDims {
        SimModelDims {
            n_layers: 5,
            kv_rate: 128,
            total_weight_bytes: 1_171_200,
        }
    }

    #[test]
    fn m4_budget_math_pins_the_stories260k_numbers() {
        let dims = stories260k_dims();
        assert_eq!(dims.mean_weight(), 234_240);
        assert_eq!(dims.required_bytes(M4_CTX_SMALL), 2_481_920);
        assert_eq!(dims.required_bytes(M4_CTX_BIG), 11_656_960);
        assert_eq!(dims.per_layer_cost(M4_CTX_BIG), 2_331_392);
        // 80% of the way from the 2k to the 16k requirement.
        assert_eq!(m4_budget(&dims), 9_821_952);
        assert_eq!(m4_cap(&dims), 536_870_912 + 9_821_952);
        // Per-node layer capacity at 16k: 4 — holds the tilted 3-layer
        // share without clamping, and two nodes pool 8 >= 5.
        assert_eq!(m4_budget(&dims) / dims.per_layer_cost(M4_CTX_BIG), 4);
        check_m4_scenario(&dims).unwrap();
    }

    #[test]
    fn m4_score_prediction_tilts_three_to_two() {
        // docs/scheduler-v1.md §2 with equal caps: factors 1.0 vs 0.75,
        // quotas 5/1.75 and 5×0.75/1.75.
        let (fast, slow) = expected_split(5);
        assert!((fast - 2.857_142_857).abs() < 1e-6);
        assert!((slow - 2.142_857_143).abs() < 1e-6);
        assert_eq!(predicted_counts(5), (3, 2));
        // Largest remainder can also favor the slow node (6 layers:
        // 3.43/2.57 -> remainders 0.43 vs 0.57).
        assert_eq!(predicted_counts(6), (3, 3));
        assert_eq!(predicted_counts(7), (4, 3));
    }

    #[test]
    fn m4_check_rejects_indecisive_or_unshiftable_models() {
        // 6 layers: the 100/50 tilt rounds to 3/3 — MORE cannot be
        // asserted, so the check must refuse with the remedy.
        let six = SimModelDims {
            n_layers: 6,
            ..stories260k_dims()
        };
        let err = check_m4_scenario(&six).unwrap_err().to_string();
        assert!(err.contains("indecisive"), "got: {err}");

        // KV rate 0: ctx cannot move the memory need at all.
        let flat = SimModelDims {
            kv_rate: 0,
            ..stories260k_dims()
        };
        let err = check_m4_scenario(&flat).unwrap_err().to_string();
        assert!(err.contains("ctx cannot shift"), "got: {err}");

        // One layer cannot split.
        let one = SimModelDims {
            n_layers: 1,
            ..stories260k_dims()
        };
        assert!(check_m4_scenario(&one).is_err());
    }

    #[test]
    fn m4_plan_helpers_read_the_wire_shape() {
        let plan = serde_json::json!({
            "epoch": 9,
            "strategy": "PipelineParallel",
            "ctx_len": 16384,
            "assignments": [
                {"node": "bbbb", "layers": {"start": 0, "end": 2}, "stage": 0},
                {"node": "aaaa", "layers": {"start": 2, "end": 5}, "stage": 1}
            ]
        });
        assert_plan_ctx(&plan, 16384).unwrap();
        assert!(assert_plan_ctx(&plan, 2048).is_err());
        let (a_layers, b_layers) = layers_by_node(&plan, "aaaa", "bbbb").unwrap();
        assert_eq!((a_layers, b_layers), (3, 2));
        // Unknown node ids and missing assignments are loud errors.
        assert!(layers_by_node(&plan, "aaaa", "cccc").is_err());
        let solo = serde_json::json!({
            "ctx_len": 2048,
            "assignments": [{"node": "aaaa", "layers": {"start": 0, "end": 5}, "stage": 0}]
        });
        assert!(layers_by_node(&solo, "aaaa", "bbbb").is_err());
        assert_eq!(
            assignment_layers(&solo.assignments().unwrap()[0]).unwrap(),
            5
        );
        assert!(assignment_layers(&serde_json::json!({"layers": {"start": 3, "end": 3}})).is_err());
    }

    // ---- Minimal GGUF reader ---------------------------------------------

    /// Synthetic GGUF v3 header builder (mirrors the one in
    /// onebrain-scheduler's dims tests; that builder is test-private).
    struct Gguf {
        tensor_count: u64,
        kv_count: u64,
        kvs: Vec<u8>,
        tensors: Vec<u8>,
    }

    impl Gguf {
        fn new() -> Gguf {
            Gguf {
                tensor_count: 0,
                kv_count: 0,
                kvs: Vec::new(),
                tensors: Vec::new(),
            }
        }

        fn string_into(out: &mut Vec<u8>, s: &str) {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s.as_bytes());
        }

        fn kv_str(mut self, key: &str, val: &str) -> Gguf {
            Self::string_into(&mut self.kvs, key);
            self.kvs.extend(8u32.to_le_bytes()); // string
            Self::string_into(&mut self.kvs, val);
            self.kv_count += 1;
            self
        }

        fn kv_u32(mut self, key: &str, val: u32) -> Gguf {
            Self::string_into(&mut self.kvs, key);
            self.kvs.extend(4u32.to_le_bytes()); // u32
            self.kvs.extend(val.to_le_bytes());
            self.kv_count += 1;
            self
        }

        /// An array-of-strings KV (like tokenizer.ggml.tokens) — exercises
        /// the element-by-element array skip.
        fn kv_str_array(mut self, key: &str, vals: &[&str]) -> Gguf {
            Self::string_into(&mut self.kvs, key);
            self.kvs.extend(9u32.to_le_bytes()); // array
            self.kvs.extend(8u32.to_le_bytes()); // of string
            self.kvs.extend((vals.len() as u64).to_le_bytes());
            for v in vals {
                Self::string_into(&mut self.kvs, v);
            }
            self.kv_count += 1;
            self
        }

        /// An array-of-f32 KV (like tokenizer scores) — exercises the
        /// fixed-width block skip.
        fn kv_f32_array(mut self, key: &str, n: u64) -> Gguf {
            Self::string_into(&mut self.kvs, key);
            self.kvs.extend(9u32.to_le_bytes()); // array
            self.kvs.extend(6u32.to_le_bytes()); // of f32
            self.kvs.extend(n.to_le_bytes());
            self.kvs
                .extend(std::iter::repeat(0u8).take((n * 4) as usize));
            self.kv_count += 1;
            self
        }

        fn tensor(mut self, name: &str, offset: u64) -> Gguf {
            Self::string_into(&mut self.tensors, name);
            self.tensors.extend(1u32.to_le_bytes()); // n_dims
            self.tensors.extend(1u64.to_le_bytes()); // dim[0]
            self.tensors.extend(0u32.to_le_bytes()); // ggml type
            self.tensors.extend(offset.to_le_bytes());
            self.tensor_count += 1;
            self
        }

        fn build(self) -> Vec<u8> {
            let mut buf = Vec::new();
            buf.extend(0x4655_4747u32.to_le_bytes()); // "GGUF"
            buf.extend(3u32.to_le_bytes());
            buf.extend(self.tensor_count.to_le_bytes());
            buf.extend(self.kv_count.to_le_bytes());
            buf.extend(&self.kvs);
            buf.extend(&self.tensors);
            buf
        }
    }

    /// A llama-flavored 2-layer header matching the scheduler's own dims
    /// test: n_embd 4096, 32 heads / 8 KV heads -> kv_rate 4096.
    fn two_layer_header() -> Vec<u8> {
        Gguf::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 2)
            .kv_u32("llama.embedding_length", 4096)
            .kv_u32("llama.attention.head_count", 32)
            .kv_u32("llama.attention.head_count_kv", 8)
            .kv_str_array("tokenizer.ggml.tokens", &["<s>", "</s>", "hi"])
            .kv_f32_array("tokenizer.ggml.scores", 3)
            .tensor("token_embd.weight", 0)
            .tensor("blk.0.attn_q.weight", 4096)
            .tensor("blk.1.attn_q.weight", 8192)
            .tensor("output.weight", 12288)
            .build()
    }

    #[test]
    fn gguf_reader_matches_the_scheduler_kv_formula() {
        let header = two_layer_header();
        // Data starts at the header end aligned up to 32.
        let data_offset = (header.len() as u64).div_ceil(32) * 32;
        let dims = parse_gguf_dims(&header, data_offset + 16384).unwrap();
        assert_eq!(dims.n_layers, 2);
        // 8 KV heads × (4096/32) head_dim = 1024; × 2 (K+V) × 2 (f16).
        assert_eq!(dims.kv_rate, 4096);
        assert_eq!(dims.total_weight_bytes, 16384);
        assert_eq!(dims.mean_weight(), 8192);
        // At ctx 2048 one layer's KV is 8 MiB (scheduler dims test parity).
        assert_eq!(dims.kv_rate * 2048, 8 << 20);
    }

    #[test]
    fn gguf_reader_falls_back_toward_n_embd_without_gqa_keys() {
        // No head_count_kv: n_head_kv = n_head, kv_rate = 4×4096 = 16384.
        let header = Gguf::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 1)
            .kv_u32("llama.embedding_length", 4096)
            .kv_u32("llama.attention.head_count", 32)
            .tensor("blk.0.w", 0)
            .build();
        let data_offset = (header.len() as u64).div_ceil(32) * 32;
        let dims = parse_gguf_dims(&header, data_offset + 64).unwrap();
        assert_eq!(dims.kv_rate, 16384);

        // No head_count at all: n_embd_kv = n_embd.
        let header = Gguf::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 1)
            .kv_u32("llama.embedding_length", 4096)
            .tensor("blk.0.w", 0)
            .build();
        let data_offset = (header.len() as u64).div_ceil(32) * 32;
        let dims = parse_gguf_dims(&header, data_offset + 64).unwrap();
        assert_eq!(dims.kv_rate, 16384);
    }

    #[test]
    fn gguf_reader_reads_the_real_smoke_model_when_cached() {
        // Guards the pinned stories260K constants against the actual file
        // whenever the smoke cache is present; silently skips otherwise
        // (CI runs `cargo xtask sim`, which exercises this path anyway).
        let path = crate::workspace_root()
            .join("target-smoke")
            .join("stories260K.gguf");
        if !path.exists() {
            eprintln!(
                "skipping: {} not cached (run `cargo xtask smoke` to fetch it)",
                path.display()
            );
            return;
        }
        assert_eq!(read_gguf_dims(&path).unwrap(), stories260k_dims());
    }

    #[test]
    fn gguf_reader_flags_truncation_and_garbage_differently() {
        let header = two_layer_header();
        // A cut-off prefix errors with the truncation marker (so the file
        // reader knows to fetch a longer prefix)...
        let err = parse_gguf_dims(&header[..header.len() / 2], 1 << 20).unwrap_err();
        assert!(format!("{err:#}").contains(GGUF_TRUNCATED), "got: {err:#}");
        // ...while a wrong magic is terminal.
        let err = parse_gguf_dims(b"not a gguf file at all", 1 << 20).unwrap_err();
        assert!(format!("{err:#}").contains("magic"), "got: {err:#}");
        // Missing required metadata is terminal too, naming the key.
        let no_embd = Gguf::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 1)
            .tensor("blk.0.w", 0)
            .build();
        let len = no_embd.len() as u64 + 4096;
        let err = parse_gguf_dims(&no_embd, len).unwrap_err();
        assert!(
            format!("{err:#}").contains("llama.embedding_length"),
            "got: {err:#}"
        );
    }

    #[test]
    fn netstat_parsing_keeps_tcp_listeners_only() {
        let fixture = "\
Active Connections\n\
\n\
  Proto  Local Address          Foreign Address        State           PID\n\
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1096\n\
  TCP    127.0.0.1:11435        0.0.0.0:0              LISTENING       4242\n\
  TCP    192.168.1.7:50022      52.1.2.3:443           ESTABLISHED     4242\n\
  TCP    [::]:135               [::]:0                 LISTENING       1096\n\
  TCP    [::1]:11436            [::]:0                 LISTENING       4243\n\
  UDP    0.0.0.0:5353           *:*                                    2044\n";
        let got = parse_netstat(fixture);
        assert_eq!(
            got,
            vec![
                Listener {
                    pid: 1096,
                    host: "0.0.0.0".into(),
                    port: 135
                },
                Listener {
                    pid: 4242,
                    host: "127.0.0.1".into(),
                    port: 11435
                },
                Listener {
                    pid: 1096,
                    host: "::".into(),
                    port: 135
                },
                Listener {
                    pid: 4243,
                    host: "::1".into(),
                    port: 11436
                },
            ]
        );
    }

    #[test]
    fn ss_parsing_attributes_pids_and_strips_scopes() {
        let fixture = "\
State  Recv-Q Send-Q Local Address:Port  Peer Address:Port Process\n\
LISTEN 0      128        127.0.0.1:38471      0.0.0.0:*    users:((\"onebrain\",pid=4242,fd=9))\n\
LISTEN 0      4096         0.0.0.0:22           0.0.0.0:*    users:((\"sshd\",pid=1,fd=3),(\"sshd\",pid=2,fd=4))\n\
LISTEN 0      64        [::1]%lo:5000         [::]:*    users:((\"other\",pid=77,fd=5))\n\
LISTEN 0      64           [::]:80              [::]:*\n";
        let got = parse_ss(fixture);
        assert_eq!(
            got,
            vec![
                Listener {
                    pid: 4242,
                    host: "127.0.0.1".into(),
                    port: 38471
                },
                Listener {
                    pid: 1,
                    host: "0.0.0.0".into(),
                    port: 22
                },
                Listener {
                    pid: 2,
                    host: "0.0.0.0".into(),
                    port: 22
                },
                Listener {
                    pid: 77,
                    host: "::1".into(),
                    port: 5000
                },
            ]
        );
    }

    #[test]
    fn lsof_parsing_reads_pid_and_name_columns() {
        let fixture = "\
COMMAND   PID  USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME\n\
onebrain 4242 user    9u  IPv4 0xdeadbeef      0t0  TCP 127.0.0.1:52000 (LISTEN)\n\
onebrain 4242 user   10u  IPv6 0xdeadbeef      0t0  TCP [::1]:52000 (LISTEN)\n\
onebrain 4242 user   11u  IPv4 0xdeadbeef      0t0  TCP 10.0.0.5:52001->1.2.3.4:443 (ESTABLISHED)\n";
        let got = parse_lsof(fixture);
        assert_eq!(
            got,
            vec![
                Listener {
                    pid: 4242,
                    host: "127.0.0.1".into(),
                    port: 52000
                },
                Listener {
                    pid: 4242,
                    host: "::1".into(),
                    port: 52000
                },
            ]
        );
    }

    #[test]
    fn loopback_classification() {
        assert!(is_loopback("127.0.0.1"));
        assert!(is_loopback("127.1.2.3"));
        assert!(is_loopback("::1"));
        assert!(is_loopback("localhost"));
        assert!(!is_loopback("0.0.0.0"));
        assert!(!is_loopback("::"));
        assert!(!is_loopback("*"));
        assert!(!is_loopback("192.168.1.7"));
        assert!(!is_loopback("10.0.0.1"));
    }

    #[test]
    fn host_port_splitting_handles_brackets_and_wildcards() {
        assert_eq!(
            split_host_port("127.0.0.1:80"),
            Some(("127.0.0.1".into(), 80))
        );
        assert_eq!(split_host_port("[::1]:8080"), Some(("::1".into(), 8080)));
        assert_eq!(split_host_port("[::]:22"), Some(("::".into(), 22)));
        assert_eq!(split_host_port("*:9"), Some(("*".into(), 9)));
        assert_eq!(split_host_port("[::1]%lo:5000"), Some(("::1".into(), 5000)));
        assert_eq!(split_host_port("*:*"), None);
        assert_eq!(split_host_port("no-port-here"), None);
    }

    #[test]
    fn listener_check_enforces_the_contract() {
        let api = Listener {
            pid: 7,
            host: "127.0.0.1".into(),
            port: 1000,
        };
        let other_pid = Listener {
            pid: 8,
            host: "0.0.0.0".into(),
            port: 22,
        };

        // Clean: only the api listener (other pids ignored).
        check_listeners(&[api.clone(), other_pid.clone()], 7, 1000, "A", false).unwrap();

        // Missing api listener = scan judged incomplete.
        assert!(check_listeners(std::slice::from_ref(&other_pid), 7, 1000, "A", false).is_err());

        // Non-loopback listener on our pid = violation, session or not.
        let bad = Listener {
            pid: 7,
            host: "0.0.0.0".into(),
            port: 5001,
        };
        let err = check_listeners(&[api.clone(), bad.clone()], 7, 1000, "A", false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-loopback"), "got: {err}");
        assert!(check_listeners(&[api.clone(), bad], 7, 1000, "A", true).is_err());

        // Extra loopback listener beyond the api bind: violation outside a
        // distributed session, legitimate during one (the head's per-epoch
        // rpc bridge listener; accept-loop, ADR 0004 amendment).
        let extra = Listener {
            pid: 7,
            host: "127.0.0.1".into(),
            port: 5002,
        };
        let err = check_listeners(&[api.clone(), extra.clone()], 7, 1000, "A", false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unexpected loopback listener"), "got: {err}");
        check_listeners(&[api, extra], 7, 1000, "A", true).unwrap();
    }

    // ---- M5 chaos helpers ------------------------------------------------

    fn content_chunk(piece: &str) -> String {
        serde_json::json!({
            "choices": [{ "index": 0, "delta": { "content": piece }, "finish_reason": null }]
        })
        .to_string()
    }

    #[test]
    fn stream_watch_kill_window_fires_once_at_the_third_chunk() {
        let mut w = StreamWatch::new(3);
        let role = serde_json::json!({
            "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }]
        })
        .to_string();
        assert_eq!(w.observe(&role), StreamAction::Continue);
        assert_eq!(w.observe(&content_chunk("Once")), StreamAction::Continue);
        assert_eq!(w.observe(&content_chunk(" upon")), StreamAction::Continue);
        // Empty content chunks do not count toward the kill window.
        assert_eq!(w.observe(&content_chunk("")), StreamAction::Continue);
        assert_eq!(w.observe(&content_chunk(" a")), StreamAction::KillNow);
        assert_eq!(w.kill_at, Some(3));
        // Later chunks never re-trigger the kill.
        assert_eq!(w.observe(&content_chunk(" time")), StreamAction::Continue);
        let final_chunk = serde_json::json!({
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "length" }]
        })
        .to_string();
        assert_eq!(w.observe(&final_chunk), StreamAction::Continue);
        assert_eq!(w.observe("[DONE]"), StreamAction::Finished);
        assert!(w.done);
        assert_eq!(w.text(), "Once upon a time");
        assert_eq!(w.pieces_after_kill(), 1);
        assert_eq!(w.finish, vec!["length".to_string()]);
        assert!(w.errors.is_empty());
    }

    #[test]
    fn stream_watch_records_error_events_and_unparsable_payloads() {
        let mut w = StreamWatch::new(3);
        assert_eq!(w.observe(&content_chunk("Hi")), StreamAction::Continue);
        let err = serde_json::json!({
            "error": { "message": "the node 'sim-c' was lost mid-generation", "type": "api_error" }
        })
        .to_string();
        assert_eq!(w.observe(&err), StreamAction::Continue);
        assert_eq!(w.observe("not json"), StreamAction::Continue);
        assert_eq!(w.observe("[DONE]"), StreamAction::Finished);
        assert_eq!(w.errors.len(), 2);
        assert!(w.errors[0].contains("lost"));
        assert!(w.finish.is_empty());
        // The kill window never opened; nothing counts as "after the kill".
        assert_eq!(w.kill_at, None);
        assert_eq!(w.pieces_after_kill(), 0);
    }

    #[test]
    fn mb_figure_counting() {
        // The failure-lifecycle message (docs/resilience.md step 3).
        let contract = "the node 'sim-b' was lost mid-generation and the remaining nodes \
                        cannot hold the model (needs 1210 MB, have 890 MB); reconnect the \
                        node or choose a smaller model";
        assert_eq!(count_mb_figures(contract), 2);
        // The scheduler's DoesNotFit display (onebrain-scheduler).
        let planning = "model needs 1210 MB pooled but the cluster has 890 MB usable; add a \
                        node, choose a smaller quant, or lower the context length";
        assert_eq!(count_mb_figures(planning), 2);
        assert_eq!(count_mb_figures("no figures here"), 0);
        assert_eq!(count_mb_figures("MB MB MB"), 0);
        assert_eq!(count_mb_figures("5MB and 12 MB"), 2);
        assert_eq!(count_mb_figures(""), 0);
    }

    #[test]
    fn structured_loss_check_enforces_name_and_both_figures() {
        let id = "deadbeefcafe0123";
        let good = "the node 'sim-c' was lost mid-generation and the remaining nodes cannot \
                    hold the model (needs 1210 MB, have 890 MB); reconnect the node or \
                    choose a smaller model";
        structured_loss_check(good, "sim-c", id).unwrap();
        // Naming by (shortened) id instead of name is accepted too.
        let by_id = good.replace("sim-c", "deadbeef");
        structured_loss_check(&by_id, "sim-c", id).unwrap();
        // Not marked as lost.
        assert!(structured_loss_check(
            "something else about sim-c (needs 1 MB, have 2 MB)",
            "sim-c",
            id
        )
        .is_err());
        // Wrong node name.
        assert!(structured_loss_check(&good.replace("sim-c", "sim-x"), "sim-c", id).is_err());
        // Only one MB figure.
        assert!(structured_loss_check(
            "the node 'sim-c' was lost mid-generation (needs 1210 MB)",
            "sim-c",
            id
        )
        .is_err());
    }

    #[test]
    fn chaos_plan_helpers_identify_epoch_workers() {
        let plan = serde_json::json!({
            "epoch": 12,
            "strategy": "PipelineParallel",
            "assignments": [
                {"node": "bbbb", "layers": {"start": 0, "end": 3}, "stage": 0},
                {"node": "aaaa", "layers": {"start": 3, "end": 5}, "stage": 1}
            ]
        });
        assert_eq!(plan_epoch(&plan).unwrap(), 12);
        assert_eq!(
            assignment_node_ids(&plan),
            vec!["bbbb".to_string(), "aaaa".to_string()]
        );
        assert_eq!(
            epoch_worker_ids(&plan, &["bbbb", "cccc"]).unwrap(),
            vec!["bbbb".to_string()]
        );
        // No known worker in the plan is a loud error.
        assert!(epoch_worker_ids(&plan, &["cccc", "dddd"]).is_err());
        // A plan without a numeric epoch is a loud error.
        assert!(plan_epoch(&serde_json::json!({"assignments": []})).is_err());
        assert_distributed(&plan).unwrap();
        assert!(assert_distributed(&serde_json::json!({
            "strategy": "Solo",
            "assignments": [{"node": "aaaa", "layers": {"start": 0, "end": 5}, "stage": 0}]
        }))
        .is_err());
    }

    #[test]
    fn sim_config_renders_the_chaos_decode_delay() {
        let cfg = render_sim_config(
            "sim-a",
            1,
            2,
            false,
            SimKnobs {
                cap_bytes: Some(1_000_000),
                decode_delay_ms: Some(150),
                ..SimKnobs::default()
            },
        );
        assert!(cfg.contains("[debug]"));
        assert!(cfg.contains("decode_delay_ms = 150\n"));
        assert!(cfg.contains("usable_memory_override_bytes = 1000000\n"));
        // The delay alone still opens the [debug] table.
        let cfg = render_sim_config(
            "sim-a",
            1,
            2,
            false,
            SimKnobs {
                decode_delay_ms: Some(150),
                ..SimKnobs::default()
            },
        );
        assert!(cfg.contains("[debug]\ndecode_delay_ms = 150\n"));
        // And stays out entirely when unset (the M3/M4 configs).
        assert!(
            !render_sim_config("sim-a", 1, 2, false, SimKnobs::default())
                .contains("decode_delay_ms")
        );
    }

    #[test]
    fn plan_assertions_accept_contract_shapes() {
        let pipeline = serde_json::json!({
            "epoch": 3,
            "strategy": "PipelineParallel",
            "assignments": [
                {"node": "aa", "layers": {"start": 0, "end": 3}, "stage": 0},
                {"node": "bb", "layers": {"start": 3, "end": 5}, "stage": 1}
            ],
            "explanation": "because neither node fits it alone"
        });
        assert_pipeline(&pipeline).unwrap();
        assert_explained(&pipeline).unwrap();
        assert!(assert_solo(&pipeline).is_err());

        let solo = serde_json::json!({
            "epoch": 4,
            "strategy": "Solo",
            "assignments": [
                {"node": "aa", "layers": {"start": 0, "end": 5}, "stage": 0}
            ],
            "explanation": "auto-solo"
        });
        assert_solo(&solo).unwrap();
        assert!(assert_pipeline(&solo).is_err());

        // Snake-case spelling tolerated; empty layer range is not.
        let snake = serde_json::json!({
            "strategy": "pipeline_parallel",
            "assignments": [
                {"node": "aa", "layers": {"start": 0, "end": 0}, "stage": 0},
                {"node": "bb", "layers": {"start": 0, "end": 5}, "stage": 1}
            ]
        });
        assert!(assert_pipeline(&snake)
            .unwrap_err()
            .to_string()
            .contains("layer range"));

        // One assignment is not a 2-node pipeline; no explanation fails.
        let one = serde_json::json!({
            "strategy": "PipelineParallel",
            "assignments": [{"node": "aa", "layers": {"start": 0, "end": 5}, "stage": 0}]
        });
        assert!(assert_pipeline(&one).is_err());
        assert!(assert_explained(&one).is_err());
        assert!(assert_explained(&serde_json::json!({"explanation": "  "})).is_err());
    }
}
