//! The range-level model cache (docs/logistics.md "Range-level cache").
//!
//! A cache entry may hold, instead of (or before) a full model file, a set
//! of `ranges/<start>-<end>` files plus a `ranges.json` manifest
//! `{ url, total_size, header_len, ranges: [{start, end, blake3}] }`.
//! Ranges are TENSOR-ALIGNED: one range per tensor (from
//! [`GgufHeader::tensor_ranges`]) plus one `0-<data_offset>` header range,
//! so a node assigned layers L..R fetches only the header plus its layers'
//! tensors, and every range file doubles as a BLAKE3-addressed blob for
//! P2P sharing (the manifest hash IS the blob hash).
//!
//! Reads are served from a range file when one covers the request, or by
//! an offset read from a full model file when the entry has one — a full
//! file implicitly holds every range, so nothing is ever stored twice.
//! Fetches are resumable at range granularity (and, via `.part` files,
//! within a range), and nothing re-downloads if the bytes already exist
//! locally.
//!
//! For split-GGUF models each part is its own download directory
//! (`cache::split_part_dir`), so every function here takes the *download
//! directory* — the entry dir for single-file models, the part dir for a
//! split part.

use std::collections::BTreeSet;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use crate::cache::{self, CacheError};
use crate::download::{self, DownloadError, MAX_RETRIES};
use crate::gguf::{GgufError, GgufHeader};

/// Subdirectory of a download dir holding `<start>-<end>` range files.
pub const RANGES_DIR: &str = "ranges";

/// Name of the range manifest inside a download dir.
pub const RANGES_MANIFEST_FILE: &str = "ranges.json";

/// A half-open `[start, end)` byte range within the model file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByteRange {
    pub start: u64,
    /// Exclusive.
    pub end: u64,
}

impl ByteRange {
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One locally-verified range in `ranges.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeEntry {
    pub start: u64,
    /// Exclusive.
    pub end: u64,
    /// Lowercase hex BLAKE3 of the range's bytes (also the P2P blob hash).
    pub blake3: String,
}

/// The on-disk `ranges.json` (docs/logistics.md).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeManifest {
    pub url: String,
    /// Size of the complete remote file in bytes.
    pub total_size: u64,
    /// Byte length of the stored header range (`0-<data_offset>` — the
    /// GGUF header padded to its alignment boundary).
    pub header_len: u64,
    pub ranges: Vec<RangeEntry>,
}

/// The range set one node needs for its assigned layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangePlan {
    pub total_size: u64,
    /// See [`RangeManifest::header_len`].
    pub header_len: u64,
    /// Sorted by start; tensor-aligned, plus the leading header range.
    pub ranges: Vec<ByteRange>,
}

#[derive(Debug, thiserror::Error)]
pub enum RangeError {
    #[error(transparent)]
    Download(#[from] DownloadError),
    #[error("gguf parse error: {0}")]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Cache(#[from] CacheError),
    #[error(
        "invalid byte range {start}..{end}: ranges are half-open [start, end) \
         and must lie inside the file; this is a planner bug — please report it"
    )]
    InvalidRange { start: u64, end: u64 },
    #[error(
        "bytes {start}..{end} are not in the local cache under {dir}; \
         fetch the range (or the full file) first: onebrain pull"
    )]
    NotPresent { dir: PathBuf, start: u64, end: u64 },
    #[error(
        "{url} does not support HTTP Range requests (it answered 200 to a \
         mid-file range); download the full file instead"
    )]
    RangeUnsupported { url: String },
    #[error(
        "no complete model file under {dir} to index; download it first, \
         or fetch ranges instead"
    )]
    NoFullFile { dir: PathBuf },
    #[error("i/o error on {path}: {source}; check free disk space and write permissions")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Layer index a tensor belongs to, following the llama.cpp naming
/// convention: per-layer tensors are `blk.<n>.<suffix>`; everything else
/// (token embeddings, output head, norms, …) is header/shared weight that
/// every node needs regardless of its layer assignment.
pub fn tensor_layer(name: &str) -> Option<u64> {
    let rest = name.strip_prefix("blk.")?;
    let digits = rest.split('.').next()?;
    digits.parse().ok()
}

/// Plan the range set a node needs given the layers it owns: the header
/// range (`0-<data_offset>`), every shared (non-`blk.*`) tensor, and every
/// tensor of an owned layer. `file_size` is the total remote file size.
pub fn plan_ranges(
    header: &GgufHeader,
    file_size: u64,
    owned_layers: &BTreeSet<u64>,
) -> Result<RangePlan, GgufError> {
    let tensor_ranges = header.tensor_ranges(file_size)?;
    let mut ranges = vec![ByteRange {
        start: 0,
        end: header.data_offset,
    }];
    for tr in &tensor_ranges {
        let owned = match tensor_layer(&tr.name) {
            None => true, // shared weights: always fetched
            Some(layer) => owned_layers.contains(&layer),
        };
        if owned && tr.start < tr.end {
            ranges.push(ByteRange {
                start: tr.start,
                end: tr.end,
            });
        }
    }
    ranges.sort();
    Ok(RangePlan {
        total_size: file_size,
        header_len: header.data_offset,
        ranges,
    })
}

/// Path of the range manifest inside a download dir.
pub fn ranges_manifest_path(dir: &Path) -> PathBuf {
    dir.join(RANGES_MANIFEST_FILE)
}

fn range_file_path(dir: &Path, r: ByteRange) -> PathBuf {
    dir.join(RANGES_DIR).join(format!("{}-{}", r.start, r.end))
}

/// Read `ranges.json` leniently: missing is an empty store; a corrupt file
/// is logged and treated as missing (the hashes regenerate on the next
/// fetch, so the store self-heals).
fn load_range_manifest(dir: &Path) -> Option<RangeManifest> {
    let path = ranges_manifest_path(dir);
    let bytes = std::fs::read(&path).ok()?;
    match serde_json::from_slice(&bytes) {
        Ok(m) => Some(m),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "corrupt ranges.json ignored; hashes will be recomputed on the next fetch"
            );
            None
        }
    }
}

fn store_range_manifest(dir: &Path, manifest: &RangeManifest) -> Result<(), RangeError> {
    let path = ranges_manifest_path(dir);
    let json =
        serde_json::to_vec_pretty(manifest).expect("range manifest serialization is infallible");
    std::fs::write(&path, json).map_err(|e| RangeError::Io { path, source: e })
}

fn upsert_range(manifest: &mut RangeManifest, r: ByteRange, blake3: String) {
    match manifest
        .ranges
        .iter_mut()
        .find(|e| e.start == r.start && e.end == r.end)
    {
        Some(entry) => entry.blake3 = blake3,
        None => {
            manifest.ranges.push(RangeEntry {
                start: r.start,
                end: r.end,
                blake3,
            });
            manifest.ranges.sort_by_key(|e| (e.start, e.end));
        }
    }
}

/// Fetch every range in `plan` from `url` into `dir` over HTTP Range
/// requests. Ranges already on disk (verified against their recorded
/// BLAKE3) are skipped; a full model file of the right size satisfies the
/// whole plan without any network traffic. Interrupted fetches resume —
/// completed ranges are never refetched, and a partial range continues
/// from its `.part` bytes.
///
/// `progress(completed, total)` counts bytes of the requested plan
/// (locally-present ranges count immediately). Returns the number of
/// bytes actually fetched over the network (`0` = fully served locally),
/// which is exactly the number the zero-WAN economics in spec §6 care
/// about.
pub async fn fetch_ranges(
    url: &str,
    dir: &Path,
    plan: &RangePlan,
    mut progress: impl FnMut(u64, u64),
) -> Result<u64, RangeError> {
    for r in &plan.ranges {
        if r.start >= r.end || r.end > plan.total_size {
            return Err(RangeError::InvalidRange {
                start: r.start,
                end: r.end,
            });
        }
    }
    let total: u64 = plan.ranges.iter().map(ByteRange::len).sum();
    let mut completed = 0u64;
    progress(completed, total);

    // A full-file entry implicitly holds every range — nothing to fetch.
    if let Some((_, size)) = cache::find_model_file(dir)? {
        if size == plan.total_size {
            progress(total, total);
            return Ok(0);
        }
    }

    let ranges_dir = dir.join(RANGES_DIR);
    tokio::fs::create_dir_all(&ranges_dir)
        .await
        .map_err(|e| RangeError::Io {
            path: ranges_dir.clone(),
            source: e,
        })?;

    let mut manifest = match load_range_manifest(dir) {
        Some(mut m) if m.total_size == plan.total_size => {
            m.url = url.to_string();
            m.header_len = plan.header_len;
            m
        }
        Some(m) => {
            // The remote file changed size: every stored range is stale.
            tracing::warn!(
                dir = %dir.display(),
                old_total = m.total_size,
                new_total = plan.total_size,
                "remote file size changed; discarding the stale range store"
            );
            let _ = tokio::fs::remove_dir_all(&ranges_dir).await;
            tokio::fs::create_dir_all(&ranges_dir)
                .await
                .map_err(|e| RangeError::Io {
                    path: ranges_dir.clone(),
                    source: e,
                })?;
            fresh_manifest(url, plan)
        }
        None => fresh_manifest(url, plan),
    };

    let client = download::http_client()?;
    let mut network_bytes = 0u64;

    for r in &plan.ranges {
        let final_path = range_file_path(dir, *r);
        let expected = manifest
            .ranges
            .iter()
            .find(|e| e.start == r.start && e.end == r.end)
            .map(|e| e.blake3.clone());

        // Already on disk? Verify before trusting — a corrupt range must
        // be caught here and refetched, not handed to the engine or a peer.
        if let Ok(meta) = tokio::fs::metadata(&final_path).await {
            if meta.len() == r.len() {
                let (_, actual) = download::hash_file(&final_path).await?;
                match expected.as_deref() {
                    Some(exp) if exp != actual => {
                        tracing::warn!(
                            range = %format!("{}-{}", r.start, r.end),
                            "cached range failed BLAKE3 verification; refetching"
                        );
                        tokio::fs::remove_file(&final_path)
                            .await
                            .map_err(|e| RangeError::Io {
                                path: final_path.clone(),
                                source: e,
                            })?;
                    }
                    _ => {
                        if expected.is_none() {
                            // A crash landed the file before its manifest
                            // entry; adopt it.
                            upsert_range(&mut manifest, *r, actual);
                            store_range_manifest(dir, &manifest)?;
                        }
                        completed += r.len();
                        progress(completed, total);
                        continue;
                    }
                }
            } else {
                // Wrong length: junk from an interrupted rename or an older
                // layout — refetch from scratch.
                let _ = tokio::fs::remove_file(&final_path).await;
            }
        }

        let mut range_progress = |done: u64| progress(completed + done, total);
        let fetched = fetch_one_range(
            &client,
            url,
            *r,
            dir,
            expected.as_deref(),
            &mut range_progress,
        )
        .await?;
        network_bytes += fetched.network_bytes;
        upsert_range(&mut manifest, *r, fetched.blake3);
        // Persist after every range so an interrupted run resumes with all
        // completed ranges verified and none refetched.
        store_range_manifest(dir, &manifest)?;
        completed += r.len();
        progress(completed, total);
    }
    Ok(network_bytes)
}

fn fresh_manifest(url: &str, plan: &RangePlan) -> RangeManifest {
    RangeManifest {
        url: url.to_string(),
        total_size: plan.total_size,
        header_len: plan.header_len,
        ranges: Vec::new(),
    }
}

struct FetchedRange {
    blake3: String,
    network_bytes: u64,
}

enum AttemptOutcome {
    /// Do not retry (client errors, local i/o failures).
    Fatal(RangeError),
    /// Worth retrying; `.part` bytes already on disk are kept for resume.
    Transient(String),
}

async fn fetch_one_range(
    client: &reqwest::Client,
    url: &str,
    r: ByteRange,
    dir: &Path,
    expected_blake3: Option<&str>,
    progress: &mut dyn FnMut(u64),
) -> Result<FetchedRange, RangeError> {
    let final_path = range_file_path(dir, r);
    let part_path = dir
        .join(RANGES_DIR)
        .join(format!("{}-{}.part", r.start, r.end));

    let mut last_error = String::from("no attempt made");
    let attempts = MAX_RETRIES + 1;
    let mut network_bytes = 0u64;
    for attempt in 0..attempts {
        if attempt > 0 {
            let backoff = download::backoff_delay(attempt);
            tracing::warn!(
                url,
                range = %format!("{}-{}", r.start, r.end),
                attempt,
                error = %last_error,
                "range fetch attempt failed; retrying after {backoff:?}"
            );
            tokio::time::sleep(backoff).await;
        }
        match range_attempt(client, url, r, &part_path, &mut network_bytes, progress).await {
            Ok(()) => {
                let (_, actual) = download::hash_file(&part_path).await?;
                if let Some(expected) = expected_blake3 {
                    if expected != actual {
                        // The server handed us different bytes than the
                        // verified copy we once held — refetch from zero.
                        let _ = tokio::fs::remove_file(&part_path).await;
                        last_error = format!(
                            "range {}-{} failed BLAKE3 verification \
                             (expected {expected}, got {actual})",
                            r.start, r.end
                        );
                        continue;
                    }
                }
                tokio::fs::rename(&part_path, &final_path)
                    .await
                    .map_err(|e| RangeError::Io {
                        path: part_path.clone(),
                        source: e,
                    })?;
                return Ok(FetchedRange {
                    blake3: actual,
                    network_bytes,
                });
            }
            Err(AttemptOutcome::Fatal(e)) => return Err(e),
            Err(AttemptOutcome::Transient(msg)) => last_error = msg,
        }
    }
    Err(DownloadError::Exhausted {
        url: url.to_string(),
        attempts,
        last_error,
    }
    .into())
}

async fn range_attempt(
    client: &reqwest::Client,
    url: &str,
    r: ByteRange,
    part_path: &Path,
    network_bytes: &mut u64,
    progress: &mut dyn FnMut(u64),
) -> Result<(), AttemptOutcome> {
    let io_err = |e: std::io::Error| {
        AttemptOutcome::Fatal(RangeError::Io {
            path: part_path.to_path_buf(),
            source: e,
        })
    };
    let mut existing = tokio::fs::metadata(part_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    if existing >= r.len() {
        // An overlong partial is a crash artifact — restart the range clean.
        tokio::fs::remove_file(part_path).await.map_err(io_err)?;
        existing = 0;
    }

    let from = r.start + existing;
    let mut request = client.get(url).header(
        reqwest::header::RANGE,
        format!("bytes={from}-{}", r.end - 1),
    );
    if let Some(auth) = download::hf_bearer_for(url) {
        request = request.header(reqwest::header::AUTHORIZATION, auth);
    }
    let response = match request.send().await {
        Ok(r) => r,
        Err(e) => return Err(AttemptOutcome::Transient(format!("request failed: {e}"))),
    };

    let status = response.status();
    let restart = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        false
    } else if status == reqwest::StatusCode::OK {
        // The server ignored our Range header and is sending the whole
        // file from byte zero. For a range starting at zero we can take
        // the prefix and drop the rest; for a mid-file range the only
        // honest outcome is an error — silently downloading the entire
        // file would defeat the point of a range plan.
        if r.start > 0 {
            return Err(AttemptOutcome::Fatal(RangeError::RangeUnsupported {
                url: url.to_string(),
            }));
        }
        true // body starts at file byte 0 == range start; discard stale .part
    } else if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let _ = tokio::fs::remove_file(part_path).await;
        return Err(AttemptOutcome::Transient(format!(
            "server rejected range {from}-{} (HTTP 416); restarting the range",
            r.end - 1
        )));
    } else if status.is_server_error() {
        return Err(AttemptOutcome::Transient(format!("HTTP {status}")));
    } else {
        return Err(AttemptOutcome::Fatal(
            DownloadError::HttpStatus {
                status: status.as_u16(),
                url: url.to_string(),
            }
            .into(),
        ));
    };

    let mut options = tokio::fs::OpenOptions::new();
    options.create(true);
    if restart {
        existing = 0;
        options.write(true).truncate(true);
    } else {
        options.append(true);
    }
    let mut file = options.open(part_path).await.map_err(io_err)?;

    let mut written = existing;
    progress(written);
    // Cap what we keep at the range length: a 200 response carries the
    // whole file, and even a 206 server is not trusted to stop exactly at
    // our end byte.
    let mut remaining = r.len() - existing;
    let mut stream = response.bytes_stream();
    while remaining > 0 {
        let Some(chunk) = stream.next().await else {
            break;
        };
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                return Err(AttemptOutcome::Transient(format!(
                    "connection interrupted after {written} bytes: {e}"
                )))
            }
        };
        let take = (chunk.len() as u64).min(remaining) as usize;
        file.write_all(&chunk[..take]).await.map_err(io_err)?;
        // Flush through to the OS so a cancelled (dropped) future leaves
        // the bytes on disk for the next resume (same rationale as the
        // full-file downloader).
        file.flush().await.map_err(io_err)?;
        written += take as u64;
        remaining -= take as u64;
        *network_bytes += take as u64;
        progress(written);
    }
    if remaining > 0 {
        return Err(AttemptOutcome::Transient(format!(
            "connection closed early at {written}/{} bytes of range {}-{}",
            r.len(),
            r.start,
            r.end
        )));
    }
    file.sync_all().await.map_err(io_err)?;
    Ok(())
}

/// Read bytes `[start, end)` of the model from the local cache: from a
/// range file that covers the request, or by an offset read from a full
/// model file. Errors with [`RangeError::NotPresent`] when neither holds
/// the bytes.
pub fn read_range(dir: &Path, start: u64, end: u64) -> Result<Vec<u8>, RangeError> {
    if start >= end {
        return Err(RangeError::InvalidRange { start, end });
    }
    let len = (end - start) as usize;
    if let Some(manifest) = load_range_manifest(dir) {
        for e in &manifest.ranges {
            if e.start <= start && end <= e.end {
                let path = range_file_path(
                    dir,
                    ByteRange {
                        start: e.start,
                        end: e.end,
                    },
                );
                if let Some(bytes) = read_slice(&path, start - e.start, len)? {
                    return Ok(bytes);
                }
            }
        }
    }
    if let Some((path, size)) = cache::find_model_file(dir)? {
        if size >= end {
            if let Some(bytes) = read_slice(&path, start, len)? {
                return Ok(bytes);
            }
        }
    }
    Err(RangeError::NotPresent {
        dir: dir.to_path_buf(),
        start,
        end,
    })
}

/// `len` bytes at `offset` of `path`; `None` when the file is missing or
/// too short (the caller falls back to another source).
fn read_slice(path: &Path, offset: u64, len: usize) -> Result<Option<Vec<u8>>, RangeError> {
    let io_err = |e: std::io::Error| RangeError::Io {
        path: path.to_path_buf(),
        source: e,
    };
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io_err(e)),
    };
    file.seek(SeekFrom::Start(offset)).map_err(io_err)?;
    let mut buf = vec![0u8; len];
    match file.read_exact(&mut buf) {
        Ok(()) => Ok(Some(buf)),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(io_err(e)),
    }
}

/// The ranges locally present with their BLAKE3 hashes — the P2P
/// `RangeInventory` payload. A range counts as present when its range file
/// exists at the recorded size, or when a full model file covers it.
/// Hashes come from `ranges.json`; ranges never recorded there (e.g. a
/// full file that was never indexed — see [`index_full_file`]) are not
/// reported, because an inventory without hashes is useless to a peer.
pub fn present_ranges(dir: &Path) -> Result<Vec<RangeEntry>, RangeError> {
    let Some(manifest) = load_range_manifest(dir) else {
        return Ok(Vec::new());
    };
    let full_len = cache::find_model_file(dir)?.map(|(_, size)| size);
    let mut out: Vec<RangeEntry> = manifest
        .ranges
        .into_iter()
        .filter(|e| {
            let r = ByteRange {
                start: e.start,
                end: e.end,
            };
            let file_ok = std::fs::metadata(range_file_path(dir, r))
                .map(|m| m.len() == r.len())
                .unwrap_or(false);
            let full_ok = full_len.is_some_and(|size| size >= e.end);
            file_ok || full_ok
        })
        .collect();
    out.sort_by_key(|e| (e.start, e.end));
    Ok(out)
}

/// Build (or refresh) `ranges.json` for an entry that holds a complete
/// model file: parse the local GGUF header, derive every tensor range plus
/// the header range, and record each range's BLAKE3 — all served by offset
/// reads, no range files written. This is how a full-file node advertises
/// its ranges to peers.
pub async fn index_full_file(dir: &Path, url: &str) -> Result<RangeManifest, RangeError> {
    let (path, size) = cache::find_model_file(dir)?.ok_or_else(|| RangeError::NoFullFile {
        dir: dir.to_path_buf(),
    })?;
    let url = url.to_string();
    let manifest = tokio::task::spawn_blocking(move || -> Result<RangeManifest, RangeError> {
        let header = parse_local_header(&path, size)?;
        let plan = plan_all_ranges(&header, size)?;
        let io_err = |e: std::io::Error| RangeError::Io {
            path: path.clone(),
            source: e,
        };
        let mut file = std::fs::File::open(&path).map_err(io_err)?;
        let mut ranges = Vec::with_capacity(plan.ranges.len());
        let mut buf = vec![0u8; 4 << 20];
        for r in &plan.ranges {
            file.seek(SeekFrom::Start(r.start)).map_err(io_err)?;
            let mut hasher = blake3::Hasher::new();
            let mut remaining = r.len();
            while remaining > 0 {
                let take = remaining.min(buf.len() as u64) as usize;
                file.read_exact(&mut buf[..take]).map_err(io_err)?;
                hasher.update(&buf[..take]);
                remaining -= take as u64;
            }
            ranges.push(RangeEntry {
                start: r.start,
                end: r.end,
                blake3: hasher.finalize().to_hex().to_string(),
            });
        }
        Ok(RangeManifest {
            url,
            total_size: size,
            header_len: plan.header_len,
            ranges,
        })
    })
    .await
    .expect("indexing task panicked")?;
    store_range_manifest(dir, &manifest)?;
    Ok(manifest)
}

/// Every range of the file: header plus all tensors (no layer filter).
fn plan_all_ranges(header: &GgufHeader, file_size: u64) -> Result<RangePlan, GgufError> {
    let mut all_layers = BTreeSet::new();
    for t in &header.tensors {
        if let Some(layer) = tensor_layer(&t.name) {
            all_layers.insert(layer);
        }
    }
    plan_ranges(header, file_size, &all_layers)
}

/// Parse the GGUF header from a growing prefix of a local file.
fn parse_local_header(path: &Path, size: u64) -> Result<GgufHeader, RangeError> {
    let io_err = |e: std::io::Error| RangeError::Io {
        path: path.to_path_buf(),
        source: e,
    };
    let mut want = (1u64 << 20).min(size);
    loop {
        let mut file = std::fs::File::open(path).map_err(io_err)?;
        let mut buf = vec![0u8; want as usize];
        file.read_exact(&mut buf).map_err(io_err)?;
        match GgufHeader::parse(&buf) {
            Ok(h) => return Ok(h),
            Err(GgufError::NeedMoreData { need_hint }) if want < size => {
                want = need_hint.max(want * 2).min(size);
            }
            Err(e) => return Err(e.into()),
        }
    }
}

/// Fetch and parse the GGUF header of a remote model, returning the header
/// plus the file's total size — everything [`plan_ranges`] needs before a
/// single weight byte moves. Fetches a small prefix over HTTP Range and
/// grows it if the header turns out longer.
pub async fn fetch_remote_header(url: &str) -> Result<(GgufHeader, u64), RangeError> {
    let client = download::http_client()?;
    fetch_remote_header_with(&client, url, 256 * 1024).await
}

/// Test seam: `initial_want` sizes the first prefix request so tests can
/// exercise the grow-and-retry path with tiny files.
pub(crate) async fn fetch_remote_header_with(
    client: &reqwest::Client,
    url: &str,
    initial_want: u64,
) -> Result<(GgufHeader, u64), RangeError> {
    let mut want = initial_want.max(64);
    loop {
        let (bytes, total) = fetch_prefix(client, url, want).await?;
        match GgufHeader::parse(&bytes) {
            Ok(h) => return Ok((h, total)),
            Err(GgufError::NeedMoreData { need_hint }) => {
                if bytes.len() as u64 >= total {
                    // The whole file is in hand and the header still runs
                    // past its end — the remote file is broken.
                    return Err(
                        GgufError::Malformed("header extends past the end of the file").into(),
                    );
                }
                want = need_hint.max(want * 2).min(total);
            }
            Err(e) => return Err(e.into()),
        }
    }
}

/// The first `want` bytes of `url` (fewer if the file is shorter) plus the
/// file's total size, with the downloader's retry/backoff discipline.
async fn fetch_prefix(
    client: &reqwest::Client,
    url: &str,
    want: u64,
) -> Result<(Vec<u8>, u64), RangeError> {
    let mut last_error = String::from("no attempt made");
    let attempts = MAX_RETRIES + 1;
    for attempt in 0..attempts {
        if attempt > 0 {
            let backoff = download::backoff_delay(attempt);
            tracing::warn!(
                url,
                attempt,
                error = %last_error,
                "header fetch attempt failed; retrying after {backoff:?}"
            );
            tokio::time::sleep(backoff).await;
        }
        match prefix_attempt(client, url, want).await {
            Ok(result) => return Ok(result),
            Err(AttemptOutcome::Fatal(e)) => return Err(e),
            Err(AttemptOutcome::Transient(msg)) => last_error = msg,
        }
    }
    Err(DownloadError::Exhausted {
        url: url.to_string(),
        attempts,
        last_error,
    }
    .into())
}

async fn prefix_attempt(
    client: &reqwest::Client,
    url: &str,
    want: u64,
) -> Result<(Vec<u8>, u64), AttemptOutcome> {
    let mut request = client.get(url).header(
        reqwest::header::RANGE,
        format!("bytes=0-{}", want.saturating_sub(1)),
    );
    if let Some(auth) = download::hf_bearer_for(url) {
        request = request.header(reqwest::header::AUTHORIZATION, auth);
    }
    let response = match request.send().await {
        Ok(r) => r,
        Err(e) => return Err(AttemptOutcome::Transient(format!("request failed: {e}"))),
    };
    let status = response.status();
    let total = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        download::content_range_total(response.headers())
    } else if status == reqwest::StatusCode::OK {
        // The server ignored the Range header; the body is the whole file.
        response.content_length()
    } else if status.is_server_error() {
        return Err(AttemptOutcome::Transient(format!("HTTP {status}")));
    } else {
        return Err(AttemptOutcome::Fatal(
            DownloadError::HttpStatus {
                status: status.as_u16(),
                url: url.to_string(),
            }
            .into(),
        ));
    };
    let Some(total) = total else {
        return Err(AttemptOutcome::Transient(
            "server did not report the file's total size".to_string(),
        ));
    };
    let needed = want.min(total);
    let mut bytes = Vec::with_capacity(needed as usize);
    let mut stream = response.bytes_stream();
    while (bytes.len() as u64) < needed {
        let Some(chunk) = stream.next().await else {
            break;
        };
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                return Err(AttemptOutcome::Transient(format!(
                    "connection interrupted after {} bytes: {e}",
                    bytes.len()
                )))
            }
        };
        let take = (chunk.len() as u64).min(needed - bytes.len() as u64) as usize;
        bytes.extend_from_slice(&chunk[..take]);
    }
    if (bytes.len() as u64) < needed {
        return Err(AttemptOutcome::Transient(format!(
            "connection closed early at {}/{needed} header bytes",
            bytes.len()
        )));
    }
    Ok((bytes, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn header_with(tensors: &[(&str, u64, u64)]) -> (GgufHeader, u64) {
        // Build a GgufHeader directly (fields are public); offsets are
        // relative to the data section, sizes give the next offset.
        let mut infos = Vec::new();
        for (name, offset, _size) in tensors {
            infos.push(crate::gguf::TensorInfo {
                name: (*name).to_string(),
                dims: vec![1],
                ggml_type: 0,
                offset: *offset,
            });
        }
        let data_len: u64 = tensors
            .iter()
            .map(|(_, offset, size)| offset + size)
            .max()
            .unwrap_or(0);
        let header = GgufHeader {
            version: 3,
            metadata: BTreeMap::new(),
            tensors: infos,
            header_len: 96,
            alignment: 32,
            data_offset: 96,
        };
        (header, 96 + data_len)
    }

    #[test]
    fn tensor_layer_follows_llama_cpp_naming() {
        assert_eq!(tensor_layer("blk.0.attn_q.weight"), Some(0));
        assert_eq!(tensor_layer("blk.17.ffn_down.weight"), Some(17));
        assert_eq!(tensor_layer("token_embd.weight"), None);
        assert_eq!(tensor_layer("output.weight"), None);
        assert_eq!(tensor_layer("output_norm.weight"), None);
        assert_eq!(tensor_layer("blk.x.weight"), None); // non-numeric
        assert_eq!(tensor_layer("blkx.0.weight"), None); // wrong prefix
    }

    #[test]
    fn plan_includes_header_shared_and_owned_layers_only() {
        let (header, file_size) = header_with(&[
            ("token_embd.weight", 0, 64),
            ("blk.0.ffn.weight", 64, 32),
            ("blk.1.ffn.weight", 96, 32),
            ("output.weight", 128, 64),
        ]);
        let owned: BTreeSet<u64> = [1].into_iter().collect();
        let plan = plan_ranges(&header, file_size, &owned).unwrap();
        assert_eq!(plan.total_size, file_size);
        assert_eq!(plan.header_len, 96);
        assert_eq!(
            plan.ranges,
            vec![
                ByteRange { start: 0, end: 96 }, // header
                ByteRange {
                    start: 96,
                    end: 160
                }, // token_embd (shared)
                ByteRange {
                    start: 192,
                    end: 224
                }, // blk.1
                ByteRange {
                    start: 224,
                    end: 288
                }, // output (shared)
            ],
            "blk.0 must be excluded; header + shared always included"
        );
    }

    #[test]
    fn plan_with_all_layers_covers_the_whole_file() {
        let (header, file_size) = header_with(&[
            ("token_embd.weight", 0, 64),
            ("blk.0.ffn.weight", 64, 32),
            ("blk.1.ffn.weight", 96, 32),
        ]);
        let owned: BTreeSet<u64> = [0, 1].into_iter().collect();
        let plan = plan_ranges(&header, file_size, &owned).unwrap();
        let covered: u64 = plan.ranges.iter().map(ByteRange::len).sum();
        assert_eq!(covered, file_size, "no gaps when every layer is owned");
        for w in plan.ranges.windows(2) {
            assert_eq!(w[0].end, w[1].start, "ranges must be contiguous");
        }
    }

    #[test]
    fn read_range_rejects_inverted_ranges() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_range(dir.path(), 10, 10).unwrap_err();
        assert!(matches!(err, RangeError::InvalidRange { .. }), "got {err}");
    }
}
