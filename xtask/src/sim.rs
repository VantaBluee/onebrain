//! `cargo xtask sim`: the M3 distributed-inference Definition-of-Done
//! rehearsal (docs/distributed.md, "Tests / DoD hooks").
//!
//! Spawns TWO sandboxed daemons on this host (reusing the pair-sim
//! machinery: separate `ONEBRAIN_HOME`s, distinct API ports, mDNS/relays
//! off) and walks the four contract scenarios in one run:
//!
//! 1. **Distribute** — both daemons capped via `[debug]
//!    usable_memory_override_bytes` so the tiny model fits neither alone
//!    (cap = 70% of the file size: the scheduler's 85% budget of that is
//!    ~59.5% < 100%, so solo fails; pooled ~119% > 100%, so two nodes fit).
//!    A `load` WITHOUT `--nodes` must auto-engage `PipelineParallel` across
//!    2 nodes, streaming `planning` + `plan` NDJSON lines; both API dialects
//!    answer through the distributed plan, and the OpenAI completion
//!    (temperature 0, max_tokens 12, fixed prompt) is captured.
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
//! `--netem` (Linux, root only — SKIP + exit 0 anywhere else): the same
//! scenario inside the pair-sim network namespaces, shaped to
//! 1 Gbit / 0.5 ms per direction.
//!
//! One `[PASS]`/`[FAIL]` checklist line per step; daemon-log tails are
//! dumped on failure; `OB_E2E_SKIP_BUILD=1` skips the inner build.

use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::e2e::{dump_daemon_log, locate_onebrain_binary, step};
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

    let (model_path, cap) = step("model: tiny GGUF + caps (solo fails, pooled fits)", || {
        let cache = root.join("target-smoke");
        std::fs::create_dir_all(&cache).with_context(|| format!("creating {}", cache.display()))?;
        let path = crate::smoke::ensure_model(&cache)?;
        let size = std::fs::metadata(&path)
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        let cap = distribute_cap(size);
        let budget = scheduler_budget(cap);
        println!(
            "  model {} ({size} B); per-node cap {cap} B \
             (85% budget {budget} B < {size} B solo; pooled {} B > {size} B)",
            path.display(),
            2 * budget
        );
        Ok((path, cap))
    })?;

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
        binary,
        netem.then_some(NS_B),
    )?;
    // Node::new wrote the plain pair-sim config; overwrite with the same
    // switches PLUS the memory cap and pinned mesh port before the daemons
    // ever start.
    write_config(&a, mesh_a, netem, Some(cap))?;
    write_config(&b, mesh_b, netem, Some(cap))?;
    println!(
        "sandbox A: {} (api {port_a}, mesh {mesh_a})",
        a.home.display()
    );
    println!(
        "sandbox B: {} (api {port_b}, mesh {mesh_b})",
        b.home.display()
    );

    let outcome = scenario(&a, &b, &model_path, mesh_a, mesh_b, netem);
    if outcome.is_err() {
        dump_daemon_log(&a.home);
        dump_daemon_log(&b.home);
    }
    cleanup(&[&a, &b], netem);
    outcome?;
    println!("sim: all steps passed");
    Ok(())
}

/// The whole rehearsal. Steps abort on first failure (later ones depend on
/// earlier ones); `cleanup` runs in `run` regardless.
fn scenario(
    a: &Node,
    b: &Node,
    model_path: &Path,
    mesh_a: u16,
    mesh_b: u16,
    netem: bool,
) -> Result<()> {
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
                restart_uncapped(node, mesh_port, netem)?;
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

    Ok(())
}

// ---------------------------------------------------------------------------
// Cap math and sandbox config
// ---------------------------------------------------------------------------

/// Per-node `usable_memory_override_bytes` for the distribute scenario:
/// 70% of the model file size. The scheduler budgets 85% of usable memory,
/// so one node offers ~59.5% of the file (solo fails: the GGUF's weight
/// bytes are within a few KB of the file size for these tiny models) while
/// two pooled nodes offer ~119% (pipeline split fits, ~half per node).
fn distribute_cap(model_file_bytes: u64) -> u64 {
    (model_file_bytes as u128 * 7 / 10) as u64
}

/// What the M3 scheduler will actually budget on a node with `usable` bytes
/// (the flat 85% utilization ceiling from docs/distributed.md).
fn scheduler_budget(usable: u64) -> u64 {
    (usable as u128 * 85 / 100) as u64
}

/// The sandbox `config.toml`: the pair-sim determinism switches plus the
/// `[debug]` memory-cap knob when capping (docs/distributed.md, test-only:
/// the value reported in NodeStatus / used by the head for itself; it never
/// touches real allocation).
fn render_sim_config(
    name: &str,
    port: u16,
    mesh_port: u16,
    netem: bool,
    cap_bytes: Option<u64>,
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
         api_bind = \"127.0.0.1:{port}\"\n\
         \n\
         [mesh]\n\
         enable_mdns = false\n\
         enable_relays = false\n\
         bind_addr = \"{mesh_host}:{mesh_port}\"\n"
    );
    if let Some(cap) = cap_bytes {
        cfg.push_str(&format!(
            "\n[debug]\nusable_memory_override_bytes = {cap}\n"
        ));
    }
    cfg
}

fn write_config(node: &Node, mesh_port: u16, netem: bool, cap_bytes: Option<u64>) -> Result<()> {
    let path = node.home.join("config").join("config.toml");
    std::fs::write(
        &path,
        render_sim_config(node.name, node.port, mesh_port, netem, cap_bytes),
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

/// Stop, rewrite the config without the cap, start again, wait healthy.
fn restart_uncapped(node: &Node, mesh_port: u16, netem: bool) -> Result<()> {
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
    write_config(node, mesh_port, netem, None)?;
    let out = node.onebrain(&["up"])?;
    node.wait_healthy().map_err(|e| {
        anyhow!(
            "{e:#}\n`onebrain up` (uncapped restart) exit code {:?}\nstdout: {}\nstderr: {}",
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
    fn cap_makes_solo_fail_and_pooled_fit() {
        // stories260K is ~1.2 MB; sweep a broad size range anyway.
        for size in [123_456u64, 1_185_376, 5_000_000, 1 << 20, 750_000_000] {
            let cap = distribute_cap(size);
            let budget = scheduler_budget(cap);
            assert!(
                budget < size,
                "size {size}: one node's 85% budget {budget} must NOT fit the model"
            );
            assert!(
                2 * budget > size,
                "size {size}: two pooled budgets {} must fit the model",
                2 * budget
            );
            // Each node's ~half share fits its own budget (equal caps ⇒
            // near-equal layer split).
            assert!(
                budget > size / 2,
                "size {size}: half the model must fit one node's budget {budget}"
            );
        }
    }

    #[test]
    fn sim_config_caps_only_when_asked() {
        let capped = render_sim_config("sim-a", 12345, 23456, false, Some(830_000));
        assert!(capped.contains("node_name = \"sim-a\""));
        assert!(capped.contains("api_bind = \"127.0.0.1:12345\""));
        assert!(capped.contains("enable_mdns = false"));
        assert!(capped.contains("enable_relays = false"));
        assert!(capped.contains("bind_addr = \"127.0.0.1:23456\""));
        assert!(capped.contains("[debug]"));
        assert!(capped.contains("usable_memory_override_bytes = 830000"));
        // Top-level keys stay above the tables (TOML validity).
        assert!(capped.find("node_name").unwrap() < capped.find("[mesh]").unwrap());

        let uncapped = render_sim_config("sim-b", 1, 2, false, None);
        assert!(!uncapped.contains("[debug]"));
        assert!(!uncapped.contains("usable_memory_override_bytes"));
        // The pinned mesh port survives the uncapped rewrite (restart
        // scenario: stored peer addresses must stay valid).
        assert!(uncapped.contains("bind_addr = \"127.0.0.1:2\""));
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
