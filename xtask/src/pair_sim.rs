//! `cargo xtask pair-sim`: the M2 mesh-and-pairing Definition-of-Done
//! rehearsal (docs/mesh.md, "Tests / DoD hooks").
//!
//! Default mode (all OSes): builds the workspace, spawns TWO sandboxed
//! daemons (separate `ONEBRAIN_HOME`s, distinct API ports, mDNS and relays
//! disabled in `[mesh]` for determinism — hosted runners have unreliable
//! multicast, and relays would leak traffic off-box). Daemon A opens a
//! pairing window through `POST /api/internal/pair/start` (NDJSON stream),
//! daemon B joins with the ticket + code via `POST /api/internal/pair/join`.
//! Asserts: A's stream reaches `paired`; both `/api/internal/peers` list the
//! other peer as `connected` with `rtt_ms` (within 15 s) and
//! `bandwidth_mbps` (within 30 s); after `POST /api/internal/unpair` on A,
//! A stops listing B and B's entry for A leaves `connected` within 15 s.
//! One `[PASS]`/`[FAIL]` checklist line per step, like `cargo xtask e2e`.
//!
//! `--netem` (Linux, root only — SKIP + exit 0 anywhere else): the same
//! scenario with each daemon inside its own network namespace, joined by a
//! veth pair shaped with `tc netem` to 1 Gbit / 0.5 ms per direction
//! (~1 ms RTT), plus sanity assertions that the probed bandwidth lands in
//! [500, 1100] Mbps and the heartbeat RTT in [0.4, 3.0] ms — wide bands per
//! the contract: the point is that measurement happens and is sane, not
//! calibration. The daemons' loopback API lives inside the namespaces, so
//! HTTP goes through `ip netns exec <ns> curl` instead of reqwest.

use std::ffi::OsStr;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::e2e::{kill_hard, locate_onebrain_binary, step};

/// `onebrain up` must reach healthy within this (same slack as e2e).
const HEALTHY_TIMEOUT: Duration = Duration::from_secs(15);
/// The first NDJSON line (`status: "window"`) must arrive within this.
const WINDOW_TIMEOUT: Duration = Duration::from_secs(10);
/// Budget for `pair/join` (dials, SPAKE2, confirm, introduce, persist).
const JOIN_TIMEOUT: Duration = Duration::from_secs(60);
/// A's stream must report `paired` within this once the join returned.
const PAIRED_TIMEOUT: Duration = Duration::from_secs(15);
/// Freshly persisted peers must show up in `/api/internal/peers` fast.
const LISTED_TIMEOUT: Duration = Duration::from_secs(5);
/// Heartbeats must drive both sides to `connected` + `rtt_ms` within this.
const CONNECTED_TIMEOUT: Duration = Duration::from_secs(15);
/// The on-connect bulk probe must have reported `bandwidth_mbps` by then.
const BANDWIDTH_TIMEOUT: Duration = Duration::from_secs(30);
/// After unpair on A, B's entry for A must leave `connected` within this.
const DEGRADE_TIMEOUT: Duration = Duration::from_secs(15);
/// The pair/start stream stays open up to the 120 s window; cap it anyway.
const STREAM_TIMEOUT: Duration = Duration::from_secs(180);
/// Budget for plain request/response internal calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Network namespace / veth names for `--netem` (fixed: leftovers from an
/// aborted run are deleted at setup, so reruns are idempotent).
const NS_A: &str = "onebrain-pairsim-a";
const NS_B: &str = "onebrain-pairsim-b";
const VETH_A: &str = "ob-veth-a";
const VETH_B: &str = "ob-veth-b";

pub fn run(netem: bool) -> Result<()> {
    if netem && !cfg!(target_os = "linux") {
        println!(
            "[SKIP] pair-sim --netem needs Linux (network namespaces + tc netem); \
             nothing to do on this OS"
        );
        return Ok(());
    }
    if netem && !is_root() {
        println!("[SKIP] pair-sim --netem needs root for `ip netns`/`tc`; rerun under sudo -E");
        return Ok(());
    }

    let root = crate::workspace_root();
    println!(
        "== cargo xtask pair-sim{}: M2 mesh & pairing rehearsal ==",
        if netem { " --netem" } else { "" }
    );

    // Build first (streams cargo's own output), then locate the binary.
    // xtask itself is excluded for the same reason as e2e: rebuilding the
    // xtask binary that is running this command fails on Windows.
    let binary = step("build: cargo build --workspace", || {
        if std::env::var("OB_E2E_SKIP_BUILD").as_deref() == Ok("1") {
            println!("  (skipping inner build: OB_E2E_SKIP_BUILD=1)");
            return crate::e2e::locate_onebrain_binary(&root);
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

    if netem {
        step(
            "netem: namespaces + veth pair shaped to 1gbit / 0.5ms per direction",
            netem_setup,
        )?;
    }

    // Two distinct API ports, picked by binding-then-dropping listeners (the
    // standard cross-platform "free port" trade-off; the race window before
    // the daemons bind is tiny). In netem mode each namespace has its own
    // loopback, so any port is free there — using the same picks is fine.
    let (port_a, port_b) = two_free_ports()?;
    let base = std::env::temp_dir();
    let run_id = std::process::id();
    let a = Node::new(
        "daemon A (host)",
        "sim-a",
        base.join(format!("onebrain-pairsim-{run_id}-a")),
        port_a,
        binary.clone(),
        netem.then_some(NS_A),
    )?;
    let b = Node::new(
        "daemon B (joiner)",
        "sim-b",
        base.join(format!("onebrain-pairsim-{run_id}-b")),
        port_b,
        binary,
        netem.then_some(NS_B),
    )?;
    println!("sandbox A: {} (port {port_a})", a.home.display());
    println!("sandbox B: {} (port {port_b})", b.home.display());

    let outcome = scenario(&a, &b, netem);
    if outcome.is_err() {
        crate::e2e::dump_daemon_log(&a.home);
        crate::e2e::dump_daemon_log(&b.home);
    }
    cleanup(&[&a, &b], netem);
    outcome?;
    println!("pair-sim: all steps passed");
    Ok(())
}

/// One sandboxed daemon: its `ONEBRAIN_HOME`, API port, and (in netem mode)
/// the network namespace every process and HTTP call must run inside.
struct Node {
    /// Human label for error messages.
    label: &'static str,
    /// `node_name` written into the sandbox `config.toml` (introduced to the
    /// peer at pairing time).
    name: &'static str,
    home: PathBuf,
    port: u16,
    binary: PathBuf,
    /// `Some(ns)`: wrap subprocesses in `ip netns exec <ns>` and do HTTP via
    /// in-namespace curl (the daemon's loopback is inside the namespace).
    netns: Option<&'static str>,
    client: reqwest::blocking::Client,
}

impl Node {
    fn new(
        label: &'static str,
        name: &'static str,
        home: PathBuf,
        port: u16,
        binary: PathBuf,
        netns: Option<&'static str>,
    ) -> Result<Node> {
        // Cleared first in case a previous run with this pid left debris.
        let _ = std::fs::remove_dir_all(&home);
        let config_dir = home.join("config");
        std::fs::create_dir_all(&config_dir)
            .with_context(|| format!("creating sandbox {}", home.display()))?;
        // mDNS and relays off per the M2 contract's CI notes: multicast is
        // unreliable on hosted runners and relays would go off-box. Pairing
        // works purely over the ticket's direct addresses.
        std::fs::write(config_dir.join("config.toml"), render_config(name, port))
            .with_context(|| format!("writing sandbox config for {label}"))?;
        Ok(Node {
            label,
            name,
            home,
            port,
            binary,
            netns,
            client: reqwest::blocking::Client::builder()
                .build()
                .context("building HTTP client")?,
        })
    }

    /// A `Command` for `program`, wrapped in `ip netns exec` when namespaced.
    fn wrap(&self, program: &OsStr) -> Command {
        match self.netns {
            None => Command::new(program),
            Some(ns) => {
                let mut cmd = Command::new("ip");
                cmd.args(["netns", "exec", ns]);
                cmd.arg(program);
                cmd
            }
        }
    }

    /// Run `onebrain <args>` against this sandbox, capturing output.
    fn onebrain(&self, args: &[&str]) -> Result<std::process::Output> {
        self.wrap(self.binary.as_os_str())
            .args(args)
            .env("ONEBRAIN_HOME", &self.home)
            .output()
            .with_context(|| {
                format!(
                    "failed to spawn `onebrain {}` for {}",
                    args.join(" "),
                    self.label
                )
            })
    }

    fn token(&self) -> Result<String> {
        let path = self.home.join("config").join("api-token");
        Ok(std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?
            .trim()
            .to_string())
    }

    fn daemon_json(&self) -> Result<Value> {
        let path = self.home.join("data").join("run").join("daemon.json");
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{}", self.port, path)
    }

    /// One internal-API request (always bearer-auth'd) → (status, body).
    fn http(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        timeout: Duration,
    ) -> Result<(u16, String)> {
        let token = self.token()?;
        let url = self.url(path);
        match self.netns {
            None => {
                let rb = match method {
                    "GET" => self.client.get(&url),
                    _ => self.client.post(&url),
                };
                let mut rb = rb.bearer_auth(&token).timeout(timeout);
                if let Some(b) = body {
                    rb = rb.json(b);
                }
                let resp = rb
                    .send()
                    .with_context(|| format!("{method} {url} on {} failed", self.label))?;
                let code = resp.status().as_u16();
                let text = resp
                    .text()
                    .with_context(|| format!("reading {method} {url} body"))?;
                Ok((code, text))
            }
            Some(_) => {
                // curl prints the body, then "\n<code>" (-w trailer).
                let mut args: Vec<String> = vec![
                    "-sS".into(),
                    "-o".into(),
                    "-".into(),
                    "-w".into(),
                    "\n%{http_code}".into(),
                    "--max-time".into(),
                    timeout.as_secs().max(1).to_string(),
                    "-X".into(),
                    method.into(),
                    "-H".into(),
                    format!("Authorization: Bearer {token}"),
                ];
                if let Some(b) = body {
                    args.push("-H".into());
                    args.push("Content-Type: application/json".into());
                    args.push("-d".into());
                    args.push(b.to_string());
                }
                args.push(url);
                let out = self
                    .wrap(OsStr::new("curl"))
                    .args(&args)
                    .output()
                    .context("failed to spawn `ip netns exec ... curl` (is curl installed?)")?;
                if !out.status.success() {
                    bail!(
                        "curl {method} {path} on {} failed: {}",
                        self.label,
                        String::from_utf8_lossy(&out.stderr).trim()
                    );
                }
                split_status_trailer(&String::from_utf8_lossy(&out.stdout))
            }
        }
    }

    fn get_json(&self, path: &str) -> Result<Value> {
        let (code, text) = self.http("GET", path, None, REQUEST_TIMEOUT)?;
        if !(200..300).contains(&code) {
            bail!("GET {path} on {} answered HTTP {code}: {text}", self.label);
        }
        serde_json::from_str(&text).with_context(|| format!("GET {path} body is not JSON: {text}"))
    }

    fn post_json(&self, path: &str, body: &Value, timeout: Duration) -> Result<Value> {
        let (code, text) = self.http("POST", path, Some(body), timeout)?;
        if !(200..300).contains(&code) {
            bail!("POST {path} on {} answered HTTP {code}: {text}", self.label);
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).with_context(|| format!("POST {path} body is not JSON: {text}"))
    }

    /// One health probe: the token file must exist and status answer 200.
    fn try_status(&self) -> Result<()> {
        let (code, text) =
            self.http("GET", "/api/internal/status", None, Duration::from_secs(2))?;
        if code != 200 {
            bail!("status answered HTTP {code}: {text}");
        }
        Ok(())
    }

    fn wait_healthy(&self) -> Result<()> {
        let deadline = Instant::now() + HEALTHY_TIMEOUT;
        loop {
            let err = match self.try_status() {
                Ok(()) => return Ok(()),
                Err(e) => e,
            };
            if Instant::now() >= deadline {
                bail!(
                    "{} not healthy within {HEALTHY_TIMEOUT:?}; last probe: {err:#}",
                    self.label
                );
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// `GET /api/internal/peers` → the `peers` array.
    fn peers(&self) -> Result<Vec<Value>> {
        let v = self.get_json("/api/internal/peers")?;
        v["peers"]
            .as_array()
            .cloned()
            .with_context(|| format!("/api/internal/peers body lacks a `peers` array: {v}"))
    }

    /// Poll until the peer entry with `id` satisfies `pred`; returns the
    /// matching entry. The timeout error carries the last peers snapshot.
    fn wait_peer(
        &self,
        id: &str,
        window: Duration,
        what: &str,
        pred: impl Fn(&Value) -> bool,
    ) -> Result<Value> {
        let deadline = Instant::now() + window;
        // Assigned on every iteration before the deadline check reads it.
        let mut last;
        loop {
            match self.peers() {
                Ok(peers) => {
                    if let Some(entry) = peers.iter().find(|p| p["id"].as_str() == Some(id)) {
                        if pred(entry) {
                            return Ok(entry.clone());
                        }
                    }
                    last = serde_json::to_string(&peers).unwrap_or_default();
                }
                Err(e) => last = format!("(peers fetch failed: {e:#})"),
            }
            if Instant::now() >= deadline {
                bail!(
                    "{} did not report {what} within {window:?}; last peers view: {last}",
                    self.label
                );
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// Poll until NO peer entry with `id` remains.
    fn wait_peer_gone(&self, id: &str, window: Duration) -> Result<()> {
        let deadline = Instant::now() + window;
        // Assigned on every iteration before the deadline check reads it.
        let mut last;
        loop {
            match self.peers() {
                Ok(peers) => {
                    if !peers.iter().any(|p| p["id"].as_str() == Some(id)) {
                        return Ok(());
                    }
                    last = serde_json::to_string(&peers).unwrap_or_default();
                }
                Err(e) => last = format!("(peers fetch failed: {e:#})"),
            }
            if Instant::now() >= deadline {
                bail!(
                    "{} still lists peer {id} after {window:?}; last peers view: {last}",
                    self.label
                );
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// `POST /api/internal/pair/start` → live NDJSON event stream.
    fn pair_start(&self) -> Result<EventStream> {
        let token = self.token()?;
        let url = self.url("/api/internal/pair/start");
        let (tx, rx) = mpsc::channel::<Result<Value>>();
        match self.netns {
            None => {
                let resp = self
                    .client
                    .post(&url)
                    .bearer_auth(&token)
                    .timeout(STREAM_TIMEOUT)
                    .send()
                    .with_context(|| format!("POST {url} failed"))?;
                if !resp.status().is_success() {
                    let code = resp.status();
                    let text = resp.text().unwrap_or_default();
                    bail!("pair/start answered HTTP {code}: {text}");
                }
                std::thread::spawn(move || read_ndjson_into(BufReader::new(resp), &tx));
                Ok(EventStream { rx, child: None })
            }
            Some(_) => {
                // -N disables buffering so events arrive as they are written.
                let mut child = self
                    .wrap(OsStr::new("curl"))
                    .args([
                        "-sN",
                        "-X",
                        "POST",
                        "--max-time",
                        &STREAM_TIMEOUT.as_secs().to_string(),
                        "-H",
                        &format!("Authorization: Bearer {token}"),
                        &url,
                    ])
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .context("failed to spawn `ip netns exec ... curl` (is curl installed?)")?;
                let stdout = child
                    .stdout
                    .take()
                    .context("curl child has no piped stdout")?;
                std::thread::spawn(move || read_ndjson_into(BufReader::new(stdout), &tx));
                Ok(EventStream {
                    rx,
                    child: Some(child),
                })
            }
        }
    }
}

/// Reader half of a pair/start stream: parse each non-empty line as JSON and
/// forward it; stop on stream end, receiver hangup, or a parse error.
fn read_ndjson_into<R: std::io::Read>(reader: BufReader<R>, tx: &mpsc::Sender<Result<Value>>) {
    for line in reader.lines() {
        let Ok(line) = line else { break }; // stream closed
        if line.trim().is_empty() {
            continue;
        }
        let parsed = serde_json::from_str::<Value>(&line)
            .with_context(|| format!("pair/start NDJSON line failed to parse: {line}"));
        let stop = parsed.is_err();
        if tx.send(parsed).is_err() || stop {
            break;
        }
    }
}

/// A live NDJSON event stream from `POST /api/internal/pair/start`.
struct EventStream {
    rx: mpsc::Receiver<Result<Value>>,
    /// The in-namespace curl child (netem mode); killed on drop so an early
    /// failure never leaves it holding the stream open.
    child: Option<Child>,
}

impl EventStream {
    /// The next event, or a descriptive error on timeout / stream end.
    fn next(&mut self, window: Duration, what: &str) -> Result<Value> {
        match self.rx.recv_timeout(window) {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(anyhow!("{e:#}")),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                bail!("no {what} event on the pair/start stream within {window:?}")
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("the pair/start stream ended before a {what} event")
            }
        }
    }
}

impl Drop for EventStream {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Name + id of a peer as reported by the pairing API. Accepts both the
/// nested `{"peer":{"name","id"}}` shape (paired event, join response per
/// the contract's `200 {peer}`) and a bare `{"name","id"}` object.
struct PeerRef {
    name: String,
    id: String,
}

fn peer_ref(v: &Value) -> Result<PeerRef> {
    let obj = if v["peer"].is_object() { &v["peer"] } else { v };
    Ok(PeerRef {
        name: obj["name"]
            .as_str()
            .with_context(|| format!("peer object lacks `name`: {v}"))?
            .to_string(),
        id: obj["id"]
            .as_str()
            .with_context(|| format!("peer object lacks `id`: {v}"))?
            .to_string(),
    })
}

/// First 8 chars of an endpoint id, the way `onebrain status` shortens them.
fn short(id: &str) -> &str {
    &id[..id.len().min(8)]
}

/// The sandbox `config.toml`: explicit API port, deterministic node name,
/// and the `[mesh]` determinism switches from the M2 contract.
fn render_config(name: &str, port: u16) -> String {
    format!(
        "node_name = \"{name}\"\n\
         api_bind = \"127.0.0.1:{port}\"\n\
         \n\
         [mesh]\n\
         enable_mdns = false\n\
         enable_relays = false\n"
    )
}

/// A 6-digit pairing code (leading zeros allowed).
fn is_six_digit_code(code: &str) -> bool {
    code.len() == 6 && code.bytes().all(|b| b.is_ascii_digit())
}

/// Split curl's `-w "\n%{http_code}"` trailer off a response body.
fn split_status_trailer(text: &str) -> Result<(u16, String)> {
    let cut = text
        .rfind('\n')
        .context("curl output is missing the status-code trailer")?;
    let code: u16 = text[cut + 1..]
        .trim()
        .parse()
        .with_context(|| format!("curl status trailer is not numeric: {:?}", &text[cut + 1..]))?;
    Ok((code, text[..cut].to_string()))
}

/// Bind two ephemeral listeners simultaneously (guaranteeing distinct ports),
/// read the ports, drop the listeners.
fn two_free_ports() -> Result<(u16, u16)> {
    let l1 = TcpListener::bind("127.0.0.1:0").context("binding a port-probe listener")?;
    let l2 = TcpListener::bind("127.0.0.1:0").context("binding a port-probe listener")?;
    Ok((l1.local_addr()?.port(), l2.local_addr()?.port()))
}

/// Effective-uid check via `id -u` (only consulted on Linux).
fn is_root() -> bool {
    Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .is_some_and(|s| s.trim() == "0")
}

/// The pairing story. Steps abort on first failure (later ones depend on
/// earlier ones); `cleanup` runs in `run` regardless.
fn scenario(a: &Node, b: &Node, netem: bool) -> Result<()> {
    step("up: both daemons healthy within 15s", || {
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

    let (mut stream, code, ticket) = step("window: pair/start on A streams code + ticket", || {
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
        if !is_six_digit_code(&code) {
            bail!("pairing code {code:?} is not 6 digits");
        }
        Ok((stream, code, ticket))
    })?;

    let peer_a = step(
        "join: pair/join on B (ticket + code) returns A's peer info",
        || {
            let resp = b.post_json(
                "/api/internal/pair/join",
                &json!({ "target": &ticket, "code": &code }),
                JOIN_TIMEOUT,
            )?;
            peer_ref(&resp)
        },
    )?;

    let peer_b = step("paired: A's window stream reports paired", || loop {
        let event = stream.next(PAIRED_TIMEOUT, "paired")?;
        match event["status"].as_str() {
            Some("attempt") => continue,
            Some("paired") => return peer_ref(&event),
            other => bail!("expected a `paired` event, got {other:?}: {event}"),
        }
    })?;
    drop(stream);
    println!(
        "       A paired with {} ({}); B paired with {} ({})",
        peer_b.name,
        short(&peer_b.id),
        peer_a.name,
        short(&peer_a.id)
    );
    note_name_mismatch(&peer_b, b.name);
    note_name_mismatch(&peer_a, a.name);

    step("peers: each side lists exactly the other peer", || {
        for (node, other) in [(a, &peer_b), (b, &peer_a)] {
            node.wait_peer(&other.id, LISTED_TIMEOUT, "the new peer listed", |_| true)?;
            let peers = node.peers()?;
            if peers.len() != 1 {
                bail!(
                    "{} lists {} peers, expected exactly 1: {}",
                    node.label,
                    peers.len(),
                    serde_json::to_string(&peers).unwrap_or_default()
                );
            }
        }
        Ok(())
    })?;

    step(
        "connected: both sides reach state=connected with rtt_ms within 15s",
        || {
            for (node, other) in [(a, &peer_b), (b, &peer_a)] {
                node.wait_peer(
                    &other.id,
                    CONNECTED_TIMEOUT,
                    "state=connected with rtt_ms",
                    |p| p["state"] == "connected" && p["rtt_ms"].as_f64().is_some(),
                )?;
            }
            Ok(())
        },
    )?;

    let (link_a, link_b) = step(
        "bandwidth: bandwidth_mbps measured on both sides within 30s",
        || {
            let has_bw = |p: &Value| {
                p["bandwidth_mbps"].as_f64().is_some() && p["rtt_ms"].as_f64().is_some()
            };
            let la = a.wait_peer(
                &peer_b.id,
                BANDWIDTH_TIMEOUT,
                "bandwidth_mbps present",
                has_bw,
            )?;
            let lb = b.wait_peer(
                &peer_a.id,
                BANDWIDTH_TIMEOUT,
                "bandwidth_mbps present",
                has_bw,
            )?;
            Ok((la, lb))
        },
    )?;
    println!(
        "       measured: A->B rtt {:.2} ms / {:.0} Mbps; B->A rtt {:.2} ms / {:.0} Mbps",
        link_a["rtt_ms"].as_f64().unwrap_or(f64::NAN),
        link_a["bandwidth_mbps"].as_f64().unwrap_or(f64::NAN),
        link_b["rtt_ms"].as_f64().unwrap_or(f64::NAN),
        link_b["bandwidth_mbps"].as_f64().unwrap_or(f64::NAN),
    );

    if netem {
        step(
            "shaped: bandwidth in [500, 1100] Mbps and rtt in [0.4, 3.0] ms on both sides",
            || {
                for (direction, entry) in [("A->B", &link_a), ("B->A", &link_b)] {
                    let bw = entry["bandwidth_mbps"]
                        .as_f64()
                        .context("bandwidth_mbps missing")?;
                    let rtt = entry["rtt_ms"].as_f64().context("rtt_ms missing")?;
                    if !(500.0..=1100.0).contains(&bw) {
                        bail!(
                            "{direction} bandwidth {bw:.0} Mbps is outside [500, 1100] \
                             (link shaped to 1 Gbit)"
                        );
                    }
                    if !(0.4..=3.0).contains(&rtt) {
                        bail!(
                            "{direction} rtt {rtt:.2} ms is outside [0.4, 3.0] \
                             (link shaped to 0.5 ms per direction, ~1 ms RTT)"
                        );
                    }
                }
                Ok(())
            },
        )?;
    }

    step("unpair: A revokes B and stops listing it", || {
        a.post_json(
            "/api/internal/unpair",
            &json!({ "name": &peer_b.name }),
            REQUEST_TIMEOUT,
        )?;
        a.wait_peer_gone(&peer_b.id, LISTED_TIMEOUT)
    })?;

    let degraded = step(
        "degrade: B's entry for A leaves state=connected within 15s",
        || {
            b.wait_peer(&peer_a.id, DEGRADE_TIMEOUT, "state != connected", |p| {
                p["state"] != "connected"
            })
        },
    )?;
    println!(
        "       B now reports {} as {}",
        peer_a.name, degraded["state"]
    );

    Ok(())
}

/// Peer names should equal the configured `node_name` of the other side
/// (contract: names default to the introduced `node_name`). A mismatch is
/// worth a loud note but is not what this rehearsal gates on.
fn note_name_mismatch(peer: &PeerRef, expected: &str) {
    if peer.name != expected {
        eprintln!(
            "note: peer introduced as {:?}, expected the configured node_name {expected:?}",
            peer.name
        );
    }
}

/// Best-effort teardown: stop/kill both daemons, drop the sandboxes, delete
/// the namespaces. Never turns a green run red.
fn cleanup(nodes: &[&Node], netem: bool) {
    for node in nodes {
        let _ = node.onebrain(&["stop"]);
        // Hard-kill whatever still answers; a stale daemon.json pid that no
        // longer serves is left alone (the OS may have recycled it).
        if let Ok(dj) = node.daemon_json() {
            if let Some(pid) = dj["pid"].as_u64() {
                if node.try_status().is_ok() {
                    let _ = kill_hard(pid as u32);
                }
            }
        }
    }
    for node in nodes {
        // Windows can hold file handles for a beat after process death.
        for _ in 0..10 {
            if std::fs::remove_dir_all(&node.home).is_ok() {
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        if node.home.exists() {
            eprintln!(
                "note: could not remove sandbox {}; delete it manually",
                node.home.display()
            );
        }
    }
    if netem {
        // Deleting a namespace also removes the veth end that lives in it
        // (and the peer end dies with it).
        let _ = sh(&["ip", "netns", "del", NS_A]);
        let _ = sh(&["ip", "netns", "del", NS_B]);
    }
}

/// Build the two namespaces, the veth pair between them, and the netem
/// shaping: 1 Gbit / 0.5 ms egress on each end => ~1 ms RTT end to end.
fn netem_setup() -> Result<()> {
    // Leftovers from an aborted run.
    let _ = sh(&["ip", "netns", "del", NS_A]);
    let _ = sh(&["ip", "netns", "del", NS_B]);
    // Hosted runners usually ship sch_netem but do not load it by default.
    let _ = sh(&["modprobe", "sch_netem"]);
    sh(&["ip", "netns", "add", NS_A])?;
    sh(&["ip", "netns", "add", NS_B])?;
    sh(&[
        "ip", "link", "add", VETH_A, "type", "veth", "peer", "name", VETH_B,
    ])?;
    sh(&["ip", "link", "set", VETH_A, "netns", NS_A])?;
    sh(&["ip", "link", "set", VETH_B, "netns", NS_B])?;
    sh(&[
        "ip",
        "-n",
        NS_A,
        "addr",
        "add",
        "10.99.77.1/24",
        "dev",
        VETH_A,
    ])?;
    sh(&[
        "ip",
        "-n",
        NS_B,
        "addr",
        "add",
        "10.99.77.2/24",
        "dev",
        VETH_B,
    ])?;
    // The daemons' HTTP API binds loopback inside each namespace.
    sh(&["ip", "-n", NS_A, "link", "set", "lo", "up"])?;
    sh(&["ip", "-n", NS_B, "link", "set", "lo", "up"])?;
    sh(&["ip", "-n", NS_A, "link", "set", VETH_A, "up"])?;
    sh(&["ip", "-n", NS_B, "link", "set", VETH_B, "up"])?;
    sh(&[
        "ip", "netns", "exec", NS_A, "tc", "qdisc", "add", "dev", VETH_A, "root", "netem", "rate",
        "1gbit", "delay", "0.5ms",
    ])?;
    sh(&[
        "ip", "netns", "exec", NS_B, "tc", "qdisc", "add", "dev", VETH_B, "root", "netem", "rate",
        "1gbit", "delay", "0.5ms",
    ])?;
    Ok(())
}

/// Run one admin command, failing loudly with its stderr.
fn sh(argv: &[&str]) -> Result<()> {
    let out = Command::new(argv[0])
        .args(&argv[1..])
        .output()
        .with_context(|| format!("failed to spawn `{}`", argv.join(" ")))?;
    if !out.status.success() {
        bail!(
            "`{}` failed: {}",
            argv.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_ref_accepts_nested_and_bare_shapes() {
        let nested = json!({ "status": "paired", "peer": { "name": "sim-b", "id": "abc123" } });
        let p = peer_ref(&nested).unwrap();
        assert_eq!(p.name, "sim-b");
        assert_eq!(p.id, "abc123");

        let wrapped = json!({ "peer": { "name": "sim-a", "id": "def456" } });
        let p = peer_ref(&wrapped).unwrap();
        assert_eq!(p.name, "sim-a");

        let bare = json!({ "name": "sim-a", "id": "def456" });
        let p = peer_ref(&bare).unwrap();
        assert_eq!(p.id, "def456");

        assert!(peer_ref(&json!({ "status": "paired" })).is_err());
    }

    #[test]
    fn six_digit_codes_allow_leading_zeros_only_digits() {
        assert!(is_six_digit_code("012345"));
        assert!(is_six_digit_code("000000"));
        assert!(!is_six_digit_code("12345"));
        assert!(!is_six_digit_code("1234567"));
        assert!(!is_six_digit_code("12a456"));
        assert!(!is_six_digit_code(""));
    }

    #[test]
    fn curl_trailer_split_handles_bodies_and_empty_bodies() {
        let (code, body) = split_status_trailer("{\"peers\":[]}\n200").unwrap();
        assert_eq!(code, 200);
        assert_eq!(body, "{\"peers\":[]}");

        let (code, body) = split_status_trailer("\n401").unwrap();
        assert_eq!(code, 401);
        assert_eq!(body, "");

        assert!(split_status_trailer("no trailer here").is_err());
        assert!(split_status_trailer("body\nnot-a-code").is_err());
    }

    #[test]
    fn sandbox_config_disables_mdns_and_relays() {
        let cfg = render_config("sim-a", 12345);
        assert!(cfg.contains("node_name = \"sim-a\""));
        assert!(cfg.contains("api_bind = \"127.0.0.1:12345\""));
        assert!(cfg.contains("[mesh]"));
        assert!(cfg.contains("enable_mdns = false"));
        assert!(cfg.contains("enable_relays = false"));
    }

    #[test]
    fn short_id_is_at_most_eight_chars() {
        assert_eq!(short("abcdefgh1234"), "abcdefgh");
        assert_eq!(short("abc"), "abc");
        assert_eq!(short(""), "");
    }

    #[test]
    fn free_ports_are_distinct() {
        let (p1, p2) = two_free_ports().unwrap();
        assert_ne!(p1, p2);
        assert!(p1 > 0 && p2 > 0);
    }
}
