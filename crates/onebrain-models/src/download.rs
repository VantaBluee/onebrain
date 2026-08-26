//! Async resumable full-file downloads with BLAKE3 integrity manifests.
//!
//! M1 scope: whole files only (range/shard fetch arrives in M3). The wire
//! protocol is plain HTTPS `GET` with `Range: bytes=<n>-` resume:
//!
//! - bytes stream into `<dest_dir>/<file_name>.part`;
//! - an existing `.part` resumes where it stopped (server answers 206; a
//!   server that ignores the range and answers 200 restarts from zero —
//!   the stale partial is truncated);
//! - on completion the file is renamed to its final name, hashed from byte
//!   zero (correctness over cleverness — resumes never trust an incremental
//!   hash), and `manifest.json` is written next to it.
//!
//! Transient failures (connect errors, dropped connections, 5xx) are retried
//! up to [`MAX_RETRIES`] times with exponential backoff; every retry resumes
//! from the bytes already on disk. Cancelling (dropping) the future leaves
//! the `.part` file in place for the next call to resume.

use std::path::{Path, PathBuf};
use std::time::Duration;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use crate::registry::DownloadSpec;

/// Name of the integrity manifest written next to each downloaded file
/// (`<data_dir>/models/<id>/manifest.json`, per docs/internal-api.md).
pub const MANIFEST_FILE: &str = "manifest.json";

/// Retries after the first attempt.
pub const MAX_RETRIES: u32 = 3;

/// Integrity record for one downloaded file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub url: String,
    pub size_bytes: u64,
    /// Lowercase hex BLAKE3 of the complete file.
    pub blake3: String,
}

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error(
        "could not initialize the HTTPS client: {0}; \
         this is unexpected — retry, and file a bug if it persists"
    )]
    Client(#[source] reqwest::Error),
    #[error(
        "server answered HTTP {status} for {url}; \
         check the model reference (the file may have moved) or try again later"
    )]
    HttpStatus { status: u16, url: String },
    #[error(
        "download from {url} failed after {attempts} attempts (last error: {last_error}); \
         check your network and rerun — the partial file is kept and the download \
         resumes where it stopped"
    )]
    Exhausted {
        url: String,
        attempts: u32,
        last_error: String,
    },
    #[error("i/o error on {path}: {source}; check free disk space and write permissions")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "integrity check failed for {path}: expected blake3 {expected}, computed {actual}; \
         delete the file and download it again"
    )]
    HashMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error(
        "size mismatch for {path}: manifest records {expected} bytes but the file has {actual}; \
         delete the file and download it again"
    )]
    SizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("cannot read manifest {path}: {reason}; download the model again to regenerate it")]
    BadManifest { path: PathBuf, reason: String },
}

/// Path of the manifest inside a model's cache directory.
pub fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(MANIFEST_FILE)
}

/// Read and parse `<dir>/manifest.json`.
pub fn read_manifest(dir: &Path) -> Result<Manifest, DownloadError> {
    let path = manifest_path(dir);
    let bytes = std::fs::read(&path).map_err(|e| DownloadError::BadManifest {
        path: path.clone(),
        reason: e.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|e| DownloadError::BadManifest {
        path,
        reason: e.to_string(),
    })
}

/// Recompute size and BLAKE3 of `path` and compare against `manifest`.
pub async fn verify(path: &Path, manifest: &Manifest) -> Result<(), DownloadError> {
    let (size, hash) = hash_file(path).await?;
    if size != manifest.size_bytes {
        return Err(DownloadError::SizeMismatch {
            path: path.to_path_buf(),
            expected: manifest.size_bytes,
            actual: size,
        });
    }
    if hash != manifest.blake3 {
        return Err(DownloadError::HashMismatch {
            path: path.to_path_buf(),
            expected: manifest.blake3.clone(),
            actual: hash,
        });
    }
    Ok(())
}

/// Download `spec` into `dest_dir` (created if missing), resuming any
/// existing `.part`. Returns the final file path.
///
/// `progress(completed_bytes, total_bytes)` is called as bytes land;
/// `total_bytes` is `0` while the server has not told us the size.
pub async fn download(
    spec: &DownloadSpec,
    dest_dir: &Path,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf, DownloadError> {
    let final_path = dest_dir.join(&spec.file_name);
    let part_path = dest_dir.join(format!("{}.part", spec.file_name));

    tokio::fs::create_dir_all(dest_dir)
        .await
        .map_err(|e| DownloadError::Io {
            path: dest_dir.to_path_buf(),
            source: e,
        })?;

    // Fast path: this exact file already completed earlier.
    if let Ok(manifest) = read_manifest(dest_dir) {
        if manifest.url == spec.url {
            if let Ok(meta) = tokio::fs::metadata(&final_path).await {
                if meta.len() == manifest.size_bytes {
                    progress(manifest.size_bytes, manifest.size_bytes);
                    return Ok(final_path);
                }
            }
        }
    }

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(DownloadError::Client)?;

    let mut last_error = String::from("no attempt made");
    let attempts = MAX_RETRIES + 1;
    for attempt in 0..attempts {
        if attempt > 0 {
            let backoff = Duration::from_millis(250u64 << attempt);
            tracing::warn!(
                url = %spec.url,
                attempt,
                error = %last_error,
                "download attempt failed; retrying after {backoff:?}"
            );
            tokio::time::sleep(backoff).await;
        }
        match run_attempt(&client, spec, &part_path, &mut progress).await {
            Ok(()) => {
                tokio::fs::rename(&part_path, &final_path)
                    .await
                    .map_err(|e| DownloadError::Io {
                        path: part_path.clone(),
                        source: e,
                    })?;
                // Hash the completed file from byte zero — resumes make an
                // incremental hash untrustworthy, and a full re-read is the
                // simple, always-correct option.
                let (size_bytes, hash) = hash_file(&final_path).await?;
                let manifest = Manifest {
                    url: spec.url.clone(),
                    size_bytes,
                    blake3: hash,
                };
                write_manifest(dest_dir, &manifest).await?;
                return Ok(final_path);
            }
            Err(AttemptError::Fatal(e)) => return Err(e),
            Err(AttemptError::Transient(msg)) => last_error = msg,
        }
    }
    Err(DownloadError::Exhausted {
        url: spec.url.clone(),
        attempts,
        last_error,
    })
}

enum AttemptError {
    /// Do not retry (client errors, local i/o failures).
    Fatal(DownloadError),
    /// Worth retrying; bytes already on disk are kept for resume.
    Transient(String),
}

async fn run_attempt(
    client: &reqwest::Client,
    spec: &DownloadSpec,
    part_path: &Path,
    progress: &mut impl FnMut(u64, u64),
) -> Result<(), AttemptError> {
    let existing = tokio::fs::metadata(part_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);

    let mut request = client.get(&spec.url);
    if existing > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={existing}-"));
    }
    let response = match request.send().await {
        Ok(r) => r,
        Err(e) => return Err(AttemptError::Transient(format!("request failed: {e}"))),
    };

    let status = response.status();
    let (mut written, total, restart) =
        if existing > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT {
            // Server honored the resume.
            let total = content_range_total(response.headers())
                .or_else(|| response.content_length().map(|l| l + existing));
            (existing, total, false)
        } else if status == reqwest::StatusCode::OK {
            // Fresh download — or a server that ignored our Range header and
            // restarted from zero; either way the stale partial is discarded.
            (0, response.content_length(), existing > 0)
        } else if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            // The .part is longer than the remote file (changed upstream?).
            // Drop it and let the next attempt start clean.
            let _ = tokio::fs::remove_file(part_path).await;
            return Err(AttemptError::Transient(format!(
                "server rejected resume from byte {existing} (HTTP 416); restarting from zero"
            )));
        } else if status.is_server_error() {
            return Err(AttemptError::Transient(format!("HTTP {status}")));
        } else {
            return Err(AttemptError::Fatal(DownloadError::HttpStatus {
                status: status.as_u16(),
                url: spec.url.clone(),
            }));
        };

    let io_err = |e: std::io::Error| {
        AttemptError::Fatal(DownloadError::Io {
            path: part_path.to_path_buf(),
            source: e,
        })
    };
    // A restart (server ignored our Range) truncates the stale partial;
    // truncation needs a write-mode handle — append-mode handles lack the
    // access right for it on Windows.
    let mut options = tokio::fs::OpenOptions::new();
    options.create(true);
    if restart {
        tracing::debug!(url = %spec.url, "server ignored Range; truncating stale partial");
        options.write(true).truncate(true);
    } else {
        options.append(true);
    }
    let mut file = options.open(part_path).await.map_err(io_err)?;

    progress(written, total.unwrap_or(0));
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                return Err(AttemptError::Transient(format!(
                    "connection interrupted after {written} bytes: {e}"
                )))
            }
        };
        file.write_all(&chunk).await.map_err(io_err)?;
        // Flush through to the OS so a cancelled (dropped) future leaves the
        // bytes on disk for the next resume.
        file.flush().await.map_err(io_err)?;
        written += chunk.len() as u64;
        progress(written, total.unwrap_or(0));
    }

    if let Some(total) = total {
        if written < total {
            return Err(AttemptError::Transient(format!(
                "connection closed early at {written}/{total} bytes"
            )));
        }
    }
    file.sync_all().await.map_err(io_err)?;
    Ok(())
}

/// Parse the total from a `Content-Range: bytes <a>-<b>/<total>` header.
fn content_range_total(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::CONTENT_RANGE)?
        .to_str()
        .ok()?
        .rsplit('/')
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Size and lowercase-hex BLAKE3 of a file, computed off the async runtime.
pub async fn hash_file(path: &Path) -> Result<(u64, String), DownloadError> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut file = std::fs::File::open(&path).map_err(|e| DownloadError::Io {
            path: path.clone(),
            source: e,
        })?;
        let mut hasher = blake3::Hasher::new();
        let size = std::io::copy(&mut file, &mut hasher).map_err(|e| DownloadError::Io {
            path: path.clone(),
            source: e,
        })?;
        Ok((size, hasher.finalize().to_hex().to_string()))
    })
    .await
    .expect("hashing task panicked")
}

async fn write_manifest(dir: &Path, manifest: &Manifest) -> Result<(), DownloadError> {
    let path = manifest_path(dir);
    let json = serde_json::to_vec_pretty(manifest).expect("manifest serialization is infallible");
    tokio::fs::write(&path, json)
        .await
        .map_err(|e| DownloadError::Io { path, source: e })
}
