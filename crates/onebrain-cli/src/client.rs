//! Blocking HTTP client for the daemon's internal control API
//! (`/api/internal/*`, contract in `docs/internal-api.md`).
//!
//! Connection details come straight from disk: the run state file
//! (`<data_dir>/run/daemon.json`) names the port, and the API token file
//! (`<config_dir>/api-token`) is readable because the CLI runs as the same
//! user. The internal endpoints are always token-authenticated — the
//! localhost exemption of the public API does not apply — so every request
//! carries `Authorization: Bearer <token>`.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Duration;

use serde::Deserialize;

use onebraind::paths::AppPaths;

/// Contents of `<data_dir>/run/daemon.json`. Informational only — the fs4
/// lock is the liveness authority — so callers must confirm with
/// [`DaemonClient::status`] before trusting it. Unknown fields are
/// tolerated for forward compatibility.
#[derive(Debug, Clone, Deserialize)]
pub struct DaemonState {
    pub pid: u32,
    pub port: u16,
    pub started_unix: u64,
    pub version: String,
}

/// `<data_dir>/run/daemon.json`.
pub fn run_state_path(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("run").join("daemon.json")
}

/// `<config_dir>/api-token`.
pub fn token_path(paths: &AppPaths) -> PathBuf {
    paths.config_dir.join("api-token")
}

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("daemon is not running (no run state at {path}); run `onebrain up` to start it")]
    NotRunning { path: String },
    #[error(
        "daemon run state at {path} is unreadable ({detail}); run `onebrain up` to refresh it"
    )]
    BadState { path: String, detail: String },
    #[error("could not read the API token at {path} ({source}); run `onebrain up` to create it")]
    Token {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to initialize the HTTP client ({0}); re-run, and report a bug if it persists")]
    Init(String),
    #[error(
        "could not reach the daemon at {url} ({detail}); run `onebrain up` if it is not running"
    )]
    Unreachable { url: String, detail: String },
    #[error("daemon rejected the request with HTTP {status}: {message}")]
    Api { status: u16, message: String },
    #[error("lost the daemon's response stream ({0}); check `onebrain status` and retry")]
    Stream(String),
    #[error(
        "the daemon's {what} stream ended without a terminal status; \
         check the daemon log under the data dir (`onebrain doctor` prints it)"
    )]
    NoTerminal { what: &'static str },
}

/// Options for [`DaemonClient::load`] beyond the model reference — mirrors
/// the `/api/internal/load` body (`--nodes`/`--explain` per
/// docs/distributed.md; `--speculative`/`--draft` per docs/perf.md §5).
#[derive(Debug, Default, Clone)]
pub struct LoadOptions<'a> {
    /// Requested context length (forwarded; the daemon may ignore it).
    pub ctx: Option<u32>,
    /// Force a node count (`1` = solo) instead of the auto-plan.
    pub nodes: Option<u32>,
    /// Ask for the scheduler's prose explanation on the `plan` line.
    pub explain: bool,
    /// Load a speculative draft model alongside the target (the daemon
    /// picks `[perf] draft_model` when `draft` is not given).
    pub speculative: bool,
    /// Explicit draft-model reference (implies speculative daemon-side).
    pub draft: Option<&'a str>,
}

/// A connected view of the local daemon's internal API.
#[derive(Debug)]
pub struct DaemonClient {
    base_url: String,
    token: String,
    state: DaemonState,
    http: reqwest::blocking::Client,
}

impl DaemonClient {
    /// Build a client from the on-disk run state + token. Succeeding here
    /// only means the files parse; call [`DaemonClient::status`] to learn
    /// whether the daemon actually answers.
    pub fn from_paths(paths: &AppPaths) -> Result<DaemonClient, ClientError> {
        let state_path = run_state_path(paths);
        let display = state_path.display().to_string();
        let raw = match std::fs::read_to_string(&state_path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ClientError::NotRunning { path: display });
            }
            Err(e) => {
                return Err(ClientError::BadState {
                    path: display,
                    detail: e.to_string(),
                });
            }
        };
        let state: DaemonState = serde_json::from_str(&raw).map_err(|e| ClientError::BadState {
            path: display,
            detail: e.to_string(),
        })?;

        let tpath = token_path(paths);
        let token = std::fs::read_to_string(&tpath)
            .map_err(|source| ClientError::Token {
                path: tpath.display().to_string(),
                source,
            })?
            .trim()
            .to_string();

        // No overall timeout: model loads legitimately stream for minutes.
        // Quick calls (status/shutdown) set per-request timeouts instead.
        let http = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Option::<Duration>::None)
            .build()
            .map_err(|e| ClientError::Init(e.to_string()))?;

        Ok(DaemonClient {
            base_url: format!("http://127.0.0.1:{}", state.port),
            token,
            state,
            http,
        })
    }

    pub fn state(&self) -> &DaemonState {
        &self.state
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn token(&self) -> &str {
        &self.token
    }

    /// `GET /api/internal/status` — the health check and topology snapshot.
    pub fn status(&self) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}/api/internal/status", self.base_url);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.token)
            .timeout(Duration::from_secs(3))
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        Self::success_json(resp)
    }

    /// `POST /api/internal/load` — NDJSON progress stream. `on_progress`
    /// sees every intermediate line (`downloading`, `planning`, `plan`,
    /// `loading`); the terminal line (`ready` or `error`) is returned
    /// instead. The options mirror the request body: `nodes`/`explain` per
    /// docs/distributed.md (M3), `speculative`/`draft` per docs/perf.md §5.
    pub fn load(
        &self,
        model: &str,
        opts: &LoadOptions<'_>,
        on_progress: impl FnMut(&serde_json::Value),
    ) -> Result<serde_json::Value, ClientError> {
        let mut body = serde_json::json!({ "model": model });
        if let Some(n) = opts.ctx {
            // The daemon may ignore unknown fields in M1; sent anyway for
            // forward compatibility.
            body["ctx"] = n.into();
        }
        if let Some(n) = opts.nodes {
            body["nodes"] = n.into();
        }
        if opts.explain {
            body["explain"] = true.into();
        }
        if opts.speculative {
            body["speculative"] = true.into();
        }
        if let Some(reference) = opts.draft {
            // An explicit draft implies speculative daemon-side; sent as
            // its own field so the daemon owns that rule.
            body["draft"] = reference.into();
        }
        let url = format!("{}/api/internal/load", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Self::api_error(status.as_u16(), resp.text().ok()));
        }

        Self::stream_ndjson(resp, "load", &["ready", "error"], on_progress)
    }

    /// `POST /api/internal/pair/start` — open a pairing window and stream
    /// its NDJSON events. `on_event` sees every non-terminal line (`window`
    /// with the code + ticket, then `attempt`); the terminal line
    /// (`paired`, `expired`, or `failed`) is returned instead.
    pub fn pair_start(
        &self,
        on_event: impl FnMut(&serde_json::Value),
    ) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}/api/internal/pair/start", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Self::api_error(status.as_u16(), resp.text().ok()));
        }
        Self::stream_ndjson(resp, "pairing", &["paired", "expired", "failed"], on_event)
    }

    /// `POST /api/internal/pair/join` — join a pairing window elsewhere.
    /// `target` is a ticket or a bare 6-digit code; `code` accompanies a
    /// ticket. Returns the daemon's `{peer}` response. No client-side
    /// timeout: a code-only join may probe several LAN candidates.
    pub fn pair_join(
        &self,
        target: &str,
        code: Option<&str>,
    ) -> Result<serde_json::Value, ClientError> {
        let mut body = serde_json::json!({ "target": target });
        if let Some(code) = code {
            body["code"] = code.into();
        }
        let url = format!("{}/api/internal/pair/join", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        Self::success_json(resp)
    }

    /// `POST /api/internal/bench` — re-run this node's profile (compute
    /// microbench + disk probe) and every connected peer's link probe (M4,
    /// docs/scheduler-v1.md "`onebrain bench`"). Returns
    /// `{ "profile": { prefill_tps, decode_tps, disk_mbps,
    /// usable_memory_bytes, measured_unix }, "links": [{ peer, rtt_ms,
    /// bandwidth_mbps, loss }] }`. Deliberately long timeout: the
    /// microbench alone has a ~10 s budget and the registry test model may
    /// need a pull on the first run.
    pub fn bench(&self) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}/api/internal/bench", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .timeout(Duration::from_secs(120))
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        Self::success_json(resp)
    }

    /// `POST /api/internal/bench/peers` — ask every Connected peer to run
    /// its microbench on demand (docs/perf.md §10). Returns
    /// `{ "peers": [{ peer, id, available, prefill_tps?, decode_tps?,
    /// disk_mbps?, measured_unix?, error? }] }` — throughput fields are
    /// present only when `available` is true. Generous timeout: the daemon
    /// queries peers concurrently but each bench may take tens of seconds
    /// (the mesh bounds each reply at 60 s).
    pub fn bench_peers(&self) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}/api/internal/bench/peers", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .timeout(Duration::from_secs(90))
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        Self::success_json(resp)
    }

    /// `POST /api/internal/perf` — read or override the runtime-togglable
    /// `[perf]` levers (docs/perf.md §10). `None` leaves a lever unchanged,
    /// so two `None`s just read the current values; overrides take effect
    /// at the daemon's NEXT model load (which is how `bench --cluster`
    /// constructs the M3 baseline: flip, reload, measure, flip back).
    /// Returns `{ "prefill_overlap": bool, "kv_reuse": bool, ... }`.
    pub fn set_perf_toggles(
        &self,
        prefill_overlap: Option<bool>,
        kv_reuse: Option<bool>,
    ) -> Result<serde_json::Value, ClientError> {
        let mut body = serde_json::json!({});
        if let Some(v) = prefill_overlap {
            body["prefill_overlap"] = v.into();
        }
        if let Some(v) = kv_reuse {
            body["kv_reuse"] = v.into();
        }
        let url = format!("{}/api/internal/perf", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .json(&body)
            .timeout(Duration::from_secs(10))
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        Self::success_json(resp)
    }

    /// `POST /api/generate` — the PUBLIC Ollama dialect, `stream:false`,
    /// greedy sampling and a fixed budget: the timed end-to-end measurement
    /// `bench --cluster` reads `prompt_eval_duration`/`eval_duration`
    /// (nanoseconds, docs/perf.md §1) from. Deliberately generous timeout:
    /// a large model on a slow cluster legitimately decodes for minutes.
    pub fn generate_timed(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}/api/generate", self.base_url);
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            // Greedy (temperature 0) so repeated benches are comparable;
            // the fixed seed is inert at temperature 0 but pins sampled
            // paths should a dialect default ever change.
            "options": { "temperature": 0.0, "num_predict": max_tokens, "seed": 0 },
        });
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .json(&body)
            .timeout(Duration::from_secs(600))
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        Self::success_json(resp)
    }

    /// `GET /api/internal/peers` — paired peers with live link state.
    pub fn peers(&self) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}/api/internal/peers", self.base_url);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.token)
            .timeout(Duration::from_secs(5))
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        Self::success_json(resp)
    }

    /// `POST /api/internal/unpair` — revoke a pairing by peer name.
    pub fn unpair(&self, name: &str) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}/api/internal/unpair", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .json(&serde_json::json!({ "name": name }))
            .timeout(Duration::from_secs(10))
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        Self::success_json(resp)
    }

    /// `POST /api/internal/models/pin` (or `/unpin`) — set or clear the
    /// cache pin flag for one model (docs/logistics.md "LRU GC + pinning").
    /// `id` is the CACHE id exactly as `onebrain ls` prints it — the
    /// daemon's manifest writes key off that directory name.
    pub fn set_model_pin(&self, id: &str, pinned: bool) -> Result<serde_json::Value, ClientError> {
        let verb = if pinned { "pin" } else { "unpin" };
        let url = format!("{}/api/internal/models/{verb}", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .json(&serde_json::json!({ "model": id }))
            .timeout(Duration::from_secs(10))
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        Self::success_json(resp)
    }

    /// Read an NDJSON body line by line: events whose `status` is in
    /// `terminal` are returned, everything else goes to `on_event`.
    /// Malformed lines (keep-alives) are tolerated.
    fn stream_ndjson(
        resp: reqwest::blocking::Response,
        what: &'static str,
        terminal: &[&str],
        mut on_event: impl FnMut(&serde_json::Value),
    ) -> Result<serde_json::Value, ClientError> {
        let reader = BufReader::new(resp);
        for line in reader.lines() {
            let line = line.map_err(|e| ClientError::Stream(e.to_string()))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
                continue; // tolerate malformed keep-alives
            };
            match event.get("status").and_then(|s| s.as_str()) {
                Some(status) if terminal.contains(&status) => return Ok(event),
                _ => on_event(&event),
            }
        }
        Err(ClientError::NoTerminal { what })
    }

    /// `POST /api/internal/shutdown` — asks the daemon to exit gracefully.
    pub fn shutdown(&self) -> Result<(), ClientError> {
        let url = format!("{}/api/internal/shutdown", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .timeout(Duration::from_secs(5))
            .send()
            .map_err(|e| ClientError::Unreachable {
                url: url.clone(),
                detail: e.to_string(),
            })?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Self::api_error(status.as_u16(), resp.text().ok()));
        }
        Ok(())
    }

    fn success_json(resp: reqwest::blocking::Response) -> Result<serde_json::Value, ClientError> {
        let status = resp.status();
        if !status.is_success() {
            return Err(Self::api_error(status.as_u16(), resp.text().ok()));
        }
        resp.json().map_err(|e| ClientError::Stream(e.to_string()))
    }

    /// Pull the human message out of the gateway's error envelope when
    /// possible; fall back to the raw body.
    fn api_error(status: u16, body: Option<String>) -> ClientError {
        let body = body.unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| {
                v.pointer("/error/message")
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
            })
            .unwrap_or(body);
        ClientError::Api { status, message }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_daemon_json_fixture() {
        let raw =
            r#"{ "pid": 4242, "port": 11435, "started_unix": 1756200000, "version": "0.1.0" }"#;
        let state: DaemonState = serde_json::from_str(raw).unwrap();
        assert_eq!(state.pid, 4242);
        assert_eq!(state.port, 11435);
        assert_eq!(state.started_unix, 1_756_200_000);
        assert_eq!(state.version, "0.1.0");
    }

    #[test]
    fn daemon_json_tolerates_unknown_fields() {
        // Forward compatibility: a newer daemon may add fields; the CLI
        // must keep parsing the ones it knows.
        let raw =
            r#"{ "pid": 1, "port": 2, "started_unix": 3, "version": "9.9.9", "mesh_port": 4000 }"#;
        let state: DaemonState = serde_json::from_str(raw).unwrap();
        assert_eq!(state.port, 2);
    }

    #[test]
    fn missing_run_state_means_not_running() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths {
            config_dir: dir.path().join("config"),
            data_dir: dir.path().join("data"),
        };
        match DaemonClient::from_paths(&paths) {
            Err(ClientError::NotRunning { .. }) => {}
            other => panic!("expected NotRunning, got {other:?}"),
        }
    }

    #[test]
    fn from_paths_reads_port_and_token() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths {
            config_dir: dir.path().join("config"),
            data_dir: dir.path().join("data"),
        };
        std::fs::create_dir_all(paths.data_dir.join("run")).unwrap();
        std::fs::create_dir_all(&paths.config_dir).unwrap();
        std::fs::write(
            run_state_path(&paths),
            r#"{"pid":7,"port":23456,"started_unix":0,"version":"0.1.0"}"#,
        )
        .unwrap();
        std::fs::write(token_path(&paths), "aabbccdd\n").unwrap();

        let client = DaemonClient::from_paths(&paths).unwrap();
        assert_eq!(client.base_url(), "http://127.0.0.1:23456");
        assert_eq!(client.token(), "aabbccdd");
        assert_eq!(client.state().pid, 7);
    }

    #[test]
    fn corrupt_run_state_is_a_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths {
            config_dir: dir.path().join("config"),
            data_dir: dir.path().join("data"),
        };
        std::fs::create_dir_all(paths.data_dir.join("run")).unwrap();
        std::fs::write(run_state_path(&paths), "not json at all").unwrap();
        match DaemonClient::from_paths(&paths) {
            Err(ClientError::BadState { .. }) => {}
            other => panic!("expected BadState, got {other:?}"),
        }
    }

    #[test]
    fn missing_token_is_a_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths {
            config_dir: dir.path().join("config"),
            data_dir: dir.path().join("data"),
        };
        std::fs::create_dir_all(paths.data_dir.join("run")).unwrap();
        std::fs::write(
            run_state_path(&paths),
            r#"{"pid":7,"port":23456,"started_unix":0,"version":"0.1.0"}"#,
        )
        .unwrap();
        match DaemonClient::from_paths(&paths) {
            Err(ClientError::Token { .. }) => {}
            other => panic!("expected Token error, got {other:?}"),
        }
    }
}
