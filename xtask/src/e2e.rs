//! `cargo xtask e2e`: the M1 Definition-of-Done rehearsal.
//!
//! Builds the workspace, sandboxes daemon state under a throwaway
//! `ONEBRAIN_HOME`, then walks the full single-node story against the real
//! binary: `up` → `run <tiny model>` → OpenAI SSE → Ollama NDJSON → model
//! listings → kill -9 + clean restart → graceful `stop` + lock reacquire.
//! One `[PASS]`/`[FAIL]` checklist line per step; nonzero exit on any FAIL.
//!
//! Everything runtime-shaped follows `docs/internal-api.md` (daemon.json,
//! api-token, `/api/internal/status`, model-name rules).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

/// `onebrain up` must reach healthy within this window (contract gives the
/// CLI 10 s; we allow slack for cold antivirus/filesystem on CI).
const HEALTHY_TIMEOUT: Duration = Duration::from_secs(15);
/// After `onebrain stop`, the status endpoint must be gone within this.
const STOP_TIMEOUT: Duration = Duration::from_secs(5);
/// After kill -9, the port must stop answering within this.
const KILL_TIMEOUT: Duration = Duration::from_secs(10);
const POLL_INTERVAL: Duration = Duration::from_millis(200);
/// Per-request budget for generation calls (tiny model, 8 tokens — but the
/// first request may pay one-time backend warmup).
const GEN_TIMEOUT: Duration = Duration::from_secs(120);
/// Per-request budget for polling/listing calls.
const SHORT_TIMEOUT: Duration = Duration::from_secs(2);

pub fn run() -> Result<()> {
    let root = crate::workspace_root();
    println!("== cargo xtask e2e: M1 definition-of-done rehearsal ==");

    // 1. Build (streams cargo's own output), then locate the binary the way
    //    dist.rs does: a no-op re-invocation with JSON compiler messages.
    let binary = step("build: cargo build --workspace", || {
        // xtask itself is excluded: workspace-level feature unification can
        // force a rebuild of the xtask binary that is running this very
        // command, which fails on Windows ("Access is denied" replacing a
        // running .exe). xtask is trivially already built anyway.
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

    // Tiny GGUF for `onebrain run` (shared cache with `cargo xtask smoke`).
    let model_path = step("model: tiny GGUF present in target-smoke/", || {
        let cache = root.join("target-smoke");
        std::fs::create_dir_all(&cache).with_context(|| format!("creating {}", cache.display()))?;
        crate::smoke::ensure_model(&cache)
    })?;

    // 2. Sandbox: a throwaway ONEBRAIN_HOME so the rehearsal never touches
    //    real user state. Cleared first in case a previous run left debris.
    let home = std::env::temp_dir().join(format!("onebrain-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    std::fs::create_dir_all(&home)
        .with_context(|| format!("creating sandbox {}", home.display()))?;
    println!("sandbox ONEBRAIN_HOME: {}", home.display());

    let ctx = Ctx {
        home,
        binary,
        client: reqwest::blocking::Client::builder()
            .build()
            .context("building HTTP client")?,
    };

    // 3. Scenario, with best-effort cleanup even on failure.
    let outcome = scenario(&ctx, &model_path);
    cleanup(&ctx);
    outcome?;
    println!("e2e: all steps passed");
    Ok(())
}

/// Handle to the sandboxed installation.
struct Ctx {
    home: PathBuf,
    binary: PathBuf,
    client: reqwest::blocking::Client,
}

impl Ctx {
    /// Run `onebrain <args>` against the sandbox, capturing output.
    fn onebrain(&self, args: &[&str]) -> Result<std::process::Output> {
        Command::new(&self.binary)
            .args(args)
            .env("ONEBRAIN_HOME", &self.home)
            .output()
            .with_context(|| format!("failed to spawn `onebrain {}`", args.join(" ")))
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

    /// GET /api/internal/status with a short timeout. Internal endpoints are
    /// always token-auth'd (no localhost exemption), so the token is required.
    fn get_status(&self, port: u16, token: &str) -> reqwest::Result<reqwest::blocking::Response> {
        self.client
            .get(format!("http://127.0.0.1:{port}/api/internal/status"))
            .bearer_auth(token)
            .timeout(SHORT_TIMEOUT)
            .send()
    }

    /// One health probe: daemon.json → port, api-token → token, status 200.
    fn try_health(&self) -> Result<(u16, String, Value)> {
        let dj = self.daemon_json()?;
        let port = dj["port"]
            .as_u64()
            .context("daemon.json has no numeric `port`")? as u16;
        let token = self.token()?;
        let resp = self.get_status(port, &token)?;
        if !resp.status().is_success() {
            bail!("status endpoint answered HTTP {}", resp.status());
        }
        let body: Value = resp.json().context("status body is not JSON")?;
        Ok((port, token, body))
    }

    /// Poll [`Self::try_health`] until it succeeds or the window closes.
    fn wait_healthy(&self, window: Duration) -> Result<(u16, String, Value)> {
        let deadline = Instant::now() + window;
        loop {
            let err = match self.try_health() {
                Ok(ok) => return Ok(ok),
                Err(e) => e,
            };
            if Instant::now() >= deadline {
                bail!("daemon not healthy within {window:?}; last probe: {err:#}");
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// Poll until the status endpoint stops answering (connection error).
    fn wait_gone(&self, port: u16, token: &str, window: Duration, what: &str) -> Result<()> {
        let deadline = Instant::now() + window;
        loop {
            if self.get_status(port, token).is_err() {
                return Ok(()); // connection refused/reset: listener is gone
            }
            if Instant::now() >= deadline {
                bail!("status endpoint on port {port} still answering {window:?} after {what}");
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }
}

/// Steps a–g of the scenario (h, cleanup, runs in `run` regardless).
fn scenario(ctx: &Ctx, model_path: &Path) -> Result<()> {
    // a. onebrain up → healthy.
    let (port, token, _status) = step("up: daemon healthy within 15s", || {
        let out = ctx.onebrain(&["up"])?;
        ctx.wait_healthy(HEALTHY_TIMEOUT).map_err(|e| {
            anyhow!(
                "{e:#}\n`onebrain up` exit code {:?}\nstdout: {}\nstderr: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stdout).trim(),
                String::from_utf8_lossy(&out.stderr).trim()
            )
        })
    })?;

    // b. onebrain run <abs path to tiny gguf> → exit 0.
    step("run: `onebrain run <tiny gguf>` exits 0", || {
        let model_arg = model_path
            .to_str()
            .context("model path is not valid UTF-8")?;
        let out = ctx.onebrain(&["run", model_arg])?;
        if !out.status.success() {
            bail!(
                "exit code {:?}\nstdout: {}\nstderr: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stdout).trim(),
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(())
    })?;

    // The robust model name is whatever the daemon reports; the contract's
    // rule for local paths (`local:<file stem>`) is only the fallback.
    let model_name = step("status: model reported by /api/internal/status", || {
        let (_p, _t, status) = ctx.try_health()?;
        match status["model"]["name"].as_str() {
            Some(name) => Ok(name.to_string()),
            None => {
                let stem = model_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .context("model path has no file stem")?;
                eprintln!("note: status.model is null after run; falling back to local:{stem}");
                Ok(format!("local:{stem}"))
            }
        }
    })?;

    // c. OpenAI dialect: streaming SSE chat completion.
    step("openai: /v1/chat/completions streams SSE to [DONE]", || {
        let body = serde_json::json!({
            "model": model_name,
            "messages": [{"role": "user", "content": "Once upon a time"}],
            "stream": true,
            "max_tokens": 8
        });
        let resp = ctx
            .client
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .bearer_auth(&token)
            .json(&body)
            .timeout(GEN_TIMEOUT)
            .send()
            .context("POST /v1/chat/completions failed")?;
        let status = resp.status();
        let ctype = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let text = resp.text().context("reading SSE body")?;
        if status.as_u16() != 200 {
            bail!("HTTP {status}; body: {text}");
        }
        if !ctype.starts_with("text/event-stream") {
            bail!("content-type is {ctype:?}, expected text/event-stream");
        }
        let events: Vec<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(str::trim)
            .filter(|e| !e.is_empty())
            .collect();
        if events.is_empty() {
            bail!("no `data:` events in SSE body:\n{text}");
        }
        if events.last().copied() != Some("[DONE]") {
            bail!(
                "stream did not terminate with [DONE]; last event: {:?}",
                events.last()
            );
        }
        let has_delta_content = events.iter().any(|e| {
            serde_json::from_str::<Value>(e)
                .ok()
                .and_then(|v| {
                    v["choices"][0]["delta"]["content"]
                        .as_str()
                        .map(|s| !s.is_empty())
                })
                .unwrap_or(false)
        });
        if !has_delta_content {
            bail!("no chunk carried non-empty choices[0].delta.content; body:\n{text}");
        }
        Ok(())
    })?;

    // d. Ollama dialect: NDJSON generate.
    step(
        "ollama: /api/generate streams NDJSON ending done:true",
        || {
            let body = serde_json::json!({
                "model": model_name,
                "prompt": "Once upon a time",
                "stream": true,
                "options": {"num_predict": 8}
            });
            let resp = ctx
                .client
                .post(format!("http://127.0.0.1:{port}/api/generate"))
                .bearer_auth(&token)
                .json(&body)
                .timeout(GEN_TIMEOUT)
                .send()
                .context("POST /api/generate failed")?;
            let status = resp.status();
            let text = resp.text().context("reading NDJSON body")?;
            if status.as_u16() != 200 {
                bail!("HTTP {status}; body: {text}");
            }
            let mut parsed = Vec::new();
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                parsed.push(
                    serde_json::from_str::<Value>(line)
                        .with_context(|| format!("NDJSON line failed to parse: {line}"))?,
                );
            }
            let last = parsed.last().context("empty NDJSON response body")?;
            if last["done"] != Value::Bool(true) {
                bail!("final NDJSON line lacks done:true: {last}");
            }
            Ok(())
        },
    )?;

    // e. Model listings in both dialects.
    step("list: /v1/models and /api/tags include the model", || {
        let v1: Value = ctx
            .client
            .get(format!("http://127.0.0.1:{port}/v1/models"))
            .bearer_auth(&token)
            .timeout(SHORT_TIMEOUT)
            .send()
            .context("GET /v1/models failed")?
            .error_for_status()
            .context("GET /v1/models returned an error status")?
            .json()
            .context("/v1/models body is not JSON")?;
        let in_v1 = v1["data"].as_array().is_some_and(|d| {
            d.iter()
                .any(|m| m["id"].as_str() == Some(model_name.as_str()))
        });
        if !in_v1 {
            bail!("/v1/models does not list {model_name}: {v1}");
        }
        let tags: Value = ctx
            .client
            .get(format!("http://127.0.0.1:{port}/api/tags"))
            .bearer_auth(&token)
            .timeout(SHORT_TIMEOUT)
            .send()
            .context("GET /api/tags failed")?
            .error_for_status()
            .context("GET /api/tags returned an error status")?
            .json()
            .context("/api/tags body is not JSON")?;
        let in_tags = tags["models"].as_array().is_some_and(|d| {
            d.iter().any(|m| {
                m["name"]
                    .as_str()
                    .is_some_and(|n| n == model_name || n.starts_with(model_name.as_str()))
            })
        });
        if !in_tags {
            bail!("/api/tags does not list {model_name}: {tags}");
        }
        Ok(())
    })?;

    // f. kill -9 → port dies → `up` again comes back healthy, proving the
    //    fs4 lock died with the process (never trust the stale pid file).
    let (port, token) = step("kill9: hard kill frees the lock; restart is clean", || {
        let dj = ctx.daemon_json()?;
        let pid = dj["pid"]
            .as_u64()
            .context("daemon.json has no numeric `pid`")? as u32;
        kill_hard(pid)?;
        ctx.wait_gone(port, &token, KILL_TIMEOUT, &format!("kill -9 of pid {pid}"))?;
        let out = ctx.onebrain(&["up"])?;
        let (new_port, new_token, _s) = ctx.wait_healthy(HEALTHY_TIMEOUT).map_err(|e| {
            anyhow!(
                "restart after kill -9 did not reach healthy: {e:#}\n\
                 `onebrain up` exit code {:?}\nstdout: {}\nstderr: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stdout).trim(),
                String::from_utf8_lossy(&out.stderr).trim()
            )
        })?;
        Ok((new_port, new_token))
    })?;

    // g1. Graceful stop: endpoint gone within 5 s.
    step("stop: graceful shutdown; endpoint gone within 5s", || {
        let out = ctx.onebrain(&["stop"])?;
        if !out.status.success() {
            bail!(
                "`onebrain stop` exit code {:?}\nstdout: {}\nstderr: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stdout).trim(),
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        ctx.wait_gone(port, &token, STOP_TIMEOUT, "`onebrain stop`")
    })?;

    // g2. The lock must be acquirable again: a second up/stop cycle proves it
    //     without reimplementing the daemon's locking here.
    step(
        "lock: freed after stop (second up/stop cycle succeeds)",
        || {
            let out = ctx.onebrain(&["up"])?;
            let (p2, t2, _s) = ctx.wait_healthy(HEALTHY_TIMEOUT).map_err(|e| {
                anyhow!(
                    "second `onebrain up` did not reach healthy — lock likely still held: {e:#}\n\
                 exit code {:?}\nstdout: {}\nstderr: {}",
                    out.status.code(),
                    String::from_utf8_lossy(&out.stdout).trim(),
                    String::from_utf8_lossy(&out.stderr).trim()
                )
            })?;
            let out = ctx.onebrain(&["stop"])?;
            if !out.status.success() {
                bail!(
                    "final `onebrain stop` exit code {:?}\nstderr: {}",
                    out.status.code(),
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }
            ctx.wait_gone(p2, &t2, STOP_TIMEOUT, "final `onebrain stop`")
        },
    )?;

    Ok(())
}

/// One checklist entry. On failure the detailed cause is printed here and a
/// terse error propagates (aborting the scenario — later steps depend on
/// earlier ones), which makes the process exit nonzero.
fn step<T>(name: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    match f() {
        Ok(v) => {
            println!("[PASS] {name}");
            Ok(v)
        }
        Err(e) => {
            println!("[FAIL] {name}\n       {e:#}");
            Err(anyhow!("e2e failed at step: {name}"))
        }
    }
}

/// Step h: kill any leftover daemon, then drop the sandbox. Best effort —
/// never turns a green run red.
fn cleanup(ctx: &Ctx) {
    if let (Ok(dj), Ok(token)) = (ctx.daemon_json(), ctx.token()) {
        if let (Some(pid), Some(port)) = (dj["pid"].as_u64(), dj["port"].as_u64()) {
            // Only kill the recorded pid if something is actually still
            // serving — a stale daemon.json pid may have been recycled by
            // the OS for an unrelated process.
            let alive = ctx
                .get_status(port as u16, &token)
                .is_ok_and(|r| r.status().is_success());
            if alive {
                let _ = kill_hard(pid as u32);
            }
        }
    }
    // Windows can hold file handles for a beat after process death.
    for _ in 0..10 {
        if std::fs::remove_dir_all(&ctx.home).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    if ctx.home.exists() {
        eprintln!(
            "note: could not remove sandbox {}; delete it manually",
            ctx.home.display()
        );
    }
}

/// SIGKILL-equivalent hard kill (no graceful shutdown path).
fn kill_hard(pid: u32) -> Result<()> {
    #[cfg(windows)]
    let output = Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .output()
        .context("failed to invoke taskkill")?;
    #[cfg(not(windows))]
    let output = Command::new("kill")
        .args(["-9", &pid.to_string()])
        .output()
        .context("failed to invoke kill")?;
    if !output.status.success() {
        bail!(
            "hard kill of pid {pid} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Locate the built `onebrain` executable the way dist.rs does: re-invoke
/// cargo (a no-op after the workspace build) with JSON compiler messages and
/// read the compiler-artifact record for the bin target named "onebrain".
fn locate_onebrain_binary(root: &Path) -> Result<PathBuf> {
    let output = Command::new("cargo")
        .current_dir(root)
        .args([
            "build",
            "-p",
            "onebrain-cli",
            "--message-format=json-render-diagnostics",
        ])
        .output()
        .context("failed to invoke cargo build for artifact discovery")?;
    if !output.status.success() {
        bail!(
            "artifact-discovery build failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Ok(msg) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let is_bin = msg["target"]["kind"]
            .as_array()
            .is_some_and(|k| k.iter().any(|v| v == "bin"));
        if msg["reason"] == "compiler-artifact"
            && is_bin
            && msg["target"]["name"] == "onebrain"
            && !msg["executable"].is_null()
        {
            return Ok(PathBuf::from(msg["executable"].as_str().unwrap()));
        }
    }
    bail!("cargo did not report an executable for the `onebrain` bin target")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SSE/NDJSON parsing rules used by the scenario, exercised without
    /// a daemon (the live path needs the CLI siblings from M1).
    #[test]
    fn sse_event_extraction_matches_contract() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n\
                    data: [DONE]\n\n";
        let events: Vec<&str> = body
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(str::trim)
            .filter(|e| !e.is_empty())
            .collect();
        assert_eq!(events.len(), 3);
        assert_eq!(events.last().copied(), Some("[DONE]"));
        let has_content = events.iter().any(|e| {
            serde_json::from_str::<Value>(e)
                .ok()
                .and_then(|v| {
                    v["choices"][0]["delta"]["content"]
                        .as_str()
                        .map(|s| !s.is_empty())
                })
                .unwrap_or(false)
        });
        assert!(has_content);
    }

    #[test]
    fn ndjson_done_detection_matches_contract() {
        let body = "{\"model\":\"m\",\"response\":\"Hi\",\"done\":false}\n\
                    {\"model\":\"m\",\"response\":\"\",\"done\":true,\"eval_count\":8}\n";
        let parsed: Vec<Value> = body
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(parsed.last().unwrap()["done"], Value::Bool(true));
        assert_eq!(parsed[0]["done"], Value::Bool(false));
    }

    #[test]
    fn compiler_artifact_parsing_finds_onebrain_bin() {
        // Shape taken from real `cargo build --message-format=json` output.
        let exe = if cfg!(windows) {
            "C:\\t\\debug\\onebrain.exe"
        } else {
            "/t/debug/onebrain"
        };
        let lines = format!(
            "{}\n{}\n",
            r#"{"reason":"compiler-artifact","target":{"kind":["lib"],"name":"onebraind"},"executable":null}"#,
            serde_json::json!({
                "reason": "compiler-artifact",
                "target": {"kind": ["bin"], "name": "onebrain"},
                "executable": exe
            })
        );
        let mut found = None;
        for line in lines.lines() {
            let Ok(msg) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let is_bin = msg["target"]["kind"]
                .as_array()
                .is_some_and(|k| k.iter().any(|v| v == "bin"));
            if msg["reason"] == "compiler-artifact"
                && is_bin
                && msg["target"]["name"] == "onebrain"
                && !msg["executable"].is_null()
            {
                found = Some(PathBuf::from(msg["executable"].as_str().unwrap()));
            }
        }
        assert_eq!(found, Some(PathBuf::from(exe)));
    }

    #[test]
    fn local_model_name_fallback_uses_file_stem() {
        // '/' separates on every OS; the drive-letter form is Windows-only.
        let p = Path::new("models/stories260K.gguf");
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap();
        assert_eq!(format!("local:{stem}"), "local:stories260K");
    }
}
