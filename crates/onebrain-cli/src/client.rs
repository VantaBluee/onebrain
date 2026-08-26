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
        "the daemon's load stream ended without a terminal status; \
         check the daemon log under the data dir (`onebrain doctor` prints it)"
    )]
    NoTerminal,
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
    /// sees every intermediate line (`downloading`, `loading`); the
    /// terminal line (`ready` or `error`) is returned instead.
    pub fn load(
        &self,
        model: &str,
        ctx: Option<u32>,
        mut on_progress: impl FnMut(&serde_json::Value),
    ) -> Result<serde_json::Value, ClientError> {
        let mut body = serde_json::json!({ "model": model });
        if let Some(n) = ctx {
            // The daemon may ignore unknown fields in M1; sent anyway for
            // forward compatibility.
            body["ctx"] = n.into();
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
                Some("ready") | Some("error") => return Ok(event),
                _ => on_progress(&event),
            }
        }
        Err(ClientError::NoTerminal)
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
