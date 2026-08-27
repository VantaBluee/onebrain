//! Range-store integration tests against a local axum server (no WAN),
//! following the download_tests.rs patterns. The server implements real
//! bounded `Range`/206 semantics and counts every body byte it serves, so
//! tests can prove not just byte-exactness but also that nothing was
//! fetched twice (the spec §6 economics).

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;

use onebrain_models::download::{download, read_manifest};
use onebrain_models::gguf::GgufHeader;
use onebrain_models::ranges::{
    fetch_ranges, fetch_remote_header, index_full_file, plan_ranges, present_ranges, read_range,
    RangeError, RangePlan,
};
use onebrain_models::registry::DownloadSpec;

// ---------------------------------------------------------------- server

#[derive(Clone)]
struct RangeServer {
    data: Arc<Vec<u8>>,
    honor_range: bool,
    /// Total body bytes written across all responses.
    served: Arc<AtomicU64>,
}

async fn serve_blob(State(server): State<RangeServer>, headers: HeaderMap) -> impl IntoResponse {
    let data = &server.data;
    let total = data.len();
    if server.honor_range {
        if let Some((start, end_incl)) = parse_range(&headers) {
            if start >= total {
                return (
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    [(header::CONTENT_RANGE, format!("bytes */{total}"))],
                    Vec::new(),
                )
                    .into_response();
            }
            let end = end_incl.map(|e| (e + 1).min(total)).unwrap_or(total);
            let body = data[start..end].to_vec();
            server.served.fetch_add(body.len() as u64, Ordering::SeqCst);
            return (
                StatusCode::PARTIAL_CONTENT,
                [(
                    header::CONTENT_RANGE,
                    format!("bytes {start}-{}/{total}", end - 1),
                )],
                body,
            )
                .into_response();
        }
    }
    server.served.fetch_add(total as u64, Ordering::SeqCst);
    (StatusCode::OK, data.as_ref().clone()).into_response()
}

/// Parse `Range: bytes=<start>-` or `bytes=<start>-<end>` (inclusive end).
fn parse_range(headers: &HeaderMap) -> Option<(usize, Option<usize>)> {
    let value = headers.get(header::RANGE)?.to_str().ok()?;
    let (start, end) = value.strip_prefix("bytes=")?.split_once('-')?;
    let start = start.parse().ok()?;
    let end = if end.is_empty() {
        None
    } else {
        Some(end.parse().ok()?)
    };
    Some((start, end))
}

async fn start_server(data: Arc<Vec<u8>>, honor_range: bool) -> (String, Arc<AtomicU64>) {
    let served = Arc::new(AtomicU64::new(0));
    let app = Router::new()
        .route("/model.gguf", get(serve_blob))
        .with_state(RangeServer {
            data,
            honor_range,
            served: served.clone(),
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}/model.gguf"), served)
}

// ------------------------------------------------------- synthetic model

/// Deterministic pseudo-random bytes (xorshift64) — no `rand` dependency.
fn random_blob(len: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// Minimal GGUF writer (v3, little-endian) — mirrors the builder the gguf
/// unit tests use, but emits a complete file with a data section.
struct Builder {
    kv_count: u64,
    tensor_count: u64,
    kvs: Vec<u8>,
    tensors: Vec<u8>,
}

const T_U32: u32 = 4;
const T_STRING: u32 = 8;

impl Builder {
    fn new() -> Self {
        Builder {
            kv_count: 0,
            tensor_count: 0,
            kvs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn string_into(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    fn kv_str(mut self, key: &str, val: &str) -> Self {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(T_STRING.to_le_bytes());
        Self::string_into(&mut self.kvs, val);
        self.kv_count += 1;
        self
    }

    fn kv_u32(mut self, key: &str, val: u32) -> Self {
        Self::string_into(&mut self.kvs, key);
        self.kvs.extend(T_U32.to_le_bytes());
        self.kvs.extend(val.to_le_bytes());
        self.kv_count += 1;
        self
    }

    fn tensor(mut self, name: &str, offset: u64) -> Self {
        Self::string_into(&mut self.tensors, name);
        self.tensors.extend(2u32.to_le_bytes());
        for d in [16u64, 16u64] {
            self.tensors.extend(d.to_le_bytes());
        }
        self.tensors.extend(0u32.to_le_bytes()); // ggml type f32 (unused)
        self.tensors.extend(offset.to_le_bytes());
        self.tensor_count += 1;
        self
    }

    fn build(self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend(0x4655_4747_u32.to_le_bytes()); // "GGUF"
        buf.extend(3u32.to_le_bytes());
        buf.extend(self.tensor_count.to_le_bytes());
        buf.extend(self.kv_count.to_le_bytes());
        buf.extend(&self.kvs);
        buf.extend(&self.tensors);
        buf
    }
}

const DATA_LEN: usize = 7936;
const TENSORS: [(&str, u64); 5] = [
    ("token_embd.weight", 0),
    ("blk.0.ffn_gate.weight", 1024),
    ("blk.1.ffn_gate.weight", 3072),
    ("blk.2.ffn_gate.weight", 3584),
    ("output.weight", 7680),
];

/// A complete synthetic GGUF: parseable header, aligned data section,
/// deterministic weight bytes. `meta_pad` inflates the header with a large
/// metadata string (to exercise the grow-and-refetch header path).
fn synthetic_model(meta_pad: usize) -> Vec<u8> {
    let mut b = Builder::new()
        .kv_str("general.architecture", "llama")
        .kv_u32("llama.block_count", 3);
    if meta_pad > 0 {
        b = b.kv_str("general.description", &"x".repeat(meta_pad));
    }
    for (name, offset) in TENSORS {
        b = b.tensor(name, offset);
    }
    let mut file = b.build();
    let header = GgufHeader::parse(&file).expect("synthetic header must parse");
    file.resize(header.data_offset as usize, 0);
    file.extend(random_blob(DATA_LEN));
    file
}

fn parse_header(file: &[u8]) -> GgufHeader {
    GgufHeader::parse(file).unwrap()
}

fn plan_for(file: &[u8], layers: &[u64]) -> RangePlan {
    let header = parse_header(file);
    let owned: BTreeSet<u64> = layers.iter().copied().collect();
    plan_ranges(&header, file.len() as u64, &owned).unwrap()
}

fn plan_bytes(plan: &RangePlan) -> u64 {
    plan.ranges.iter().map(|r| r.end - r.start).sum()
}

/// Reassemble every planned range through `read_range` and compare against
/// the source blob — THE byte-exactness proof.
fn assert_ranges_byte_exact(dir: &Path, plan: &RangePlan, blob: &[u8]) {
    for r in &plan.ranges {
        let bytes = read_range(dir, r.start, r.end).unwrap();
        assert_eq!(
            bytes,
            &blob[r.start as usize..r.end as usize],
            "range {}-{} differs from the source",
            r.start,
            r.end
        );
    }
}

// ------------------------------------------------------------------ tests

#[tokio::test]
async fn range_fetch_assembles_byte_exact_vs_full_download() {
    let blob = Arc::new(synthetic_model(0));
    let (url, served) = start_server(blob.clone(), true).await;

    // Reference: a full-file download of the same model.
    let full_dir = tempfile::tempdir().unwrap();
    let spec = DownloadSpec {
        cache_key: "full".into(),
        url: url.clone(),
        file_name: "model.gguf".into(),
    };
    let full_path = download(&spec, full_dir.path(), |_, _| {}).await.unwrap();
    let full_bytes = std::fs::read(&full_path).unwrap();

    // Range fetch of every layer into a separate entry.
    let plan = plan_for(&blob, &[0, 1, 2]);
    assert_eq!(
        plan_bytes(&plan),
        blob.len() as u64,
        "all layers = whole file"
    );
    let dir = tempfile::tempdir().unwrap();
    let mut seen = Vec::new();
    let before = served.load(Ordering::SeqCst);
    let network = fetch_ranges(&url, dir.path(), &plan, |c, t| seen.push((c, t)))
        .await
        .unwrap();
    assert_eq!(network, blob.len() as u64);
    assert_eq!(served.load(Ordering::SeqCst) - before, blob.len() as u64);
    assert_eq!(
        *seen.last().unwrap(),
        (blob.len() as u64, blob.len() as u64)
    );

    // Byte-exact against the full download, range by range and reassembled.
    assert_eq!(full_bytes, **blob);
    assert_ranges_byte_exact(dir.path(), &plan, &full_bytes);
    let mut reassembled = Vec::new();
    for r in &plan.ranges {
        reassembled.extend(read_range(dir.path(), r.start, r.end).unwrap());
    }
    assert_eq!(reassembled, full_bytes, "concatenated ranges = the file");

    // ranges.json follows the documented contract shape.
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("ranges.json")).unwrap()).unwrap();
    assert_eq!(json["url"], url.as_str());
    assert_eq!(json["total_size"], blob.len() as u64);
    assert!(json["header_len"].as_u64().unwrap() > 0);
    assert!(json["ranges"][0]["blake3"].is_string());

    // The inventory reports every range with the right hashes.
    let inventory = present_ranges(dir.path()).unwrap();
    assert_eq!(inventory.len(), plan.ranges.len());
    for e in &inventory {
        let expected = blake3::hash(&blob[e.start as usize..e.end as usize]);
        assert_eq!(e.blake3, expected.to_hex().to_string());
    }

    // A second fetch is a no-op: zero network bytes, server untouched.
    let before = served.load(Ordering::SeqCst);
    let network = fetch_ranges(&url, dir.path(), &plan, |_, _| {})
        .await
        .unwrap();
    assert_eq!(network, 0, "everything is local; nothing may re-download");
    assert_eq!(served.load(Ordering::SeqCst), before);
}

#[tokio::test]
async fn partial_plan_fetches_only_owned_layers_and_replan_reuses_disk() {
    let blob = Arc::new(synthetic_model(0));
    let (url, served) = start_server(blob.clone(), true).await;
    let dir = tempfile::tempdir().unwrap();

    let header = parse_header(&blob);
    // Owned layer 1 only: header + shared tensors + blk.1.
    let plan1 = plan_for(&blob, &[1]);
    let expected1 = header.data_offset /* header */
        + 1024 /* token_embd */ + 512 /* blk.1 */ + 256 /* output */;
    assert_eq!(plan_bytes(&plan1), expected1);

    let network = fetch_ranges(&url, dir.path(), &plan1, |_, _| {})
        .await
        .unwrap();
    assert_eq!(network, expected1);
    assert_ranges_byte_exact(dir.path(), &plan1, &blob);

    // Unowned layers are absent locally.
    let blk0 = tensor_range_of(&blob, "blk.0.ffn_gate.weight");
    let err = read_range(dir.path(), blk0.0, blk0.1).unwrap_err();
    assert!(matches!(err, RangeError::NotPresent { .. }), "got {err}");
    assert!(err.to_string().contains("onebrain pull"), "remedy: {err}");

    // Re-plan now owning layers {0, 1}: only blk.0's bytes hit the network.
    let plan2 = plan_for(&blob, &[0, 1]);
    let before = served.load(Ordering::SeqCst);
    let network = fetch_ranges(&url, dir.path(), &plan2, |_, _| {})
        .await
        .unwrap();
    assert_eq!(network, 2048, "only blk.0 (2048 bytes) is new");
    assert_eq!(served.load(Ordering::SeqCst) - before, 2048);
    assert_ranges_byte_exact(dir.path(), &plan2, &blob);
}

/// Absolute byte range of a named tensor in the synthetic model.
fn tensor_range_of(blob: &[u8], name: &str) -> (u64, u64) {
    let header = parse_header(blob);
    let ranges = header.tensor_ranges(blob.len() as u64).unwrap();
    let r = ranges.iter().find(|r| r.name == name).unwrap();
    (r.start, r.end)
}

#[tokio::test]
async fn interrupted_range_download_resumes_byte_exact() {
    let blob = Arc::new(synthetic_model(0));
    let (url, served) = start_server(blob.clone(), true).await;
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_for(&blob, &[0, 1, 2]);
    let total = plan_bytes(&plan);

    // Simulate an interruption mid-range: the biggest tensor's range has
    // its first half already on disk as a `.part` (exactly the state a
    // dropped fetch future leaves behind).
    let (s, e) = tensor_range_of(&blob, "blk.2.ffn_gate.weight");
    let half = (e - s) / 2;
    std::fs::create_dir_all(dir.path().join("ranges")).unwrap();
    std::fs::write(
        dir.path().join("ranges").join(format!("{s}-{e}.part")),
        &blob[s as usize..(s + half) as usize],
    )
    .unwrap();

    let before = served.load(Ordering::SeqCst);
    let network = fetch_ranges(&url, dir.path(), &plan, |_, _| {})
        .await
        .unwrap();
    assert_eq!(
        network,
        total - half,
        "resume must continue from the .part bytes, not restart the range"
    );
    assert_eq!(served.load(Ordering::SeqCst) - before, total - half);
    assert_ranges_byte_exact(dir.path(), &plan, &blob);
}

#[tokio::test]
async fn cancelled_fetch_resumes_and_skips_completed_ranges() {
    let blob = Arc::new(synthetic_model(0));
    let (url, _served) = start_server(blob.clone(), true).await;
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_for(&blob, &[0, 1, 2]);
    let total = plan_bytes(&plan);

    // Cancel (drop the future) once at least the header and first tensor
    // ranges have completed.
    let threshold = plan.ranges[0].end - plan.ranges[0].start + 1024;
    let notify = Arc::new(tokio::sync::Notify::new());
    let notifier = notify.clone();
    let fut = fetch_ranges(&url, dir.path(), &plan, move |completed, _| {
        if completed >= threshold {
            notifier.notify_one();
        }
    });
    tokio::select! {
        result = fut => panic!("fetch completed before cancellation: {result:?}"),
        _ = notify.notified() => {} // fut dropped here
    }

    // Resume: completed ranges must not refetch; the result is byte-exact.
    let network = fetch_ranges(&url, dir.path(), &plan, |_, _| {})
        .await
        .unwrap();
    assert!(
        network < total,
        "resume refetched everything ({network}/{total} bytes)"
    );
    assert_ranges_byte_exact(dir.path(), &plan, &blob);
}

#[tokio::test]
async fn corrupt_range_is_caught_by_blake3_and_refetched() {
    let blob = Arc::new(synthetic_model(0));
    let (url, served) = start_server(blob.clone(), true).await;
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_for(&blob, &[0, 1, 2]);
    fetch_ranges(&url, dir.path(), &plan, |_, _| {})
        .await
        .unwrap();

    // Flip one byte inside a stored range (same length — only the hash
    // can catch this).
    let (s, e) = tensor_range_of(&blob, "blk.1.ffn_gate.weight");
    let path = dir.path().join("ranges").join(format!("{s}-{e}"));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[7] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let before = served.load(Ordering::SeqCst);
    let network = fetch_ranges(&url, dir.path(), &plan, |_, _| {})
        .await
        .unwrap();
    assert_eq!(network, e - s, "exactly the corrupt range must refetch");
    assert_eq!(served.load(Ordering::SeqCst) - before, e - s);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        &blob[s as usize..e as usize],
        "the refetched range must be byte-exact again"
    );
}

#[tokio::test]
async fn full_file_entry_answers_range_reads_byte_exact() {
    let blob = Arc::new(synthetic_model(0));
    let (url, served) = start_server(blob.clone(), true).await;
    let dir = tempfile::tempdir().unwrap();
    let spec = DownloadSpec {
        cache_key: "m".into(),
        url: url.clone(),
        file_name: "model.gguf".into(),
    };
    download(&spec, dir.path(), |_, _| {}).await.unwrap();

    // Arbitrary offsets, tensor ranges, the whole file — all served by
    // offset reads from the full file, no range files involved.
    let plan = plan_for(&blob, &[0, 1, 2]);
    assert_ranges_byte_exact(dir.path(), &plan, &blob);
    assert_eq!(
        read_range(dir.path(), 5, 100).unwrap(),
        &blob[5..100],
        "sub-tensor reads work too"
    );
    assert!(
        !dir.path().join("ranges").exists(),
        "no duplication on disk"
    );

    // A range fetch against a full-file entry never touches the network.
    let before = served.load(Ordering::SeqCst);
    let network = fetch_ranges(&url, dir.path(), &plan, |_, _| {})
        .await
        .unwrap();
    assert_eq!(network, 0);
    assert_eq!(served.load(Ordering::SeqCst), before);

    // Without an index there are no advertised hashes; indexing the full
    // file advertises every range.
    assert_eq!(present_ranges(dir.path()).unwrap(), Vec::new());
    let manifest = index_full_file(dir.path(), &url).await.unwrap();
    assert_eq!(manifest.total_size, blob.len() as u64);
    let inventory = present_ranges(dir.path()).unwrap();
    assert_eq!(inventory.len(), plan.ranges.len());
    for e in &inventory {
        let expected = blake3::hash(&blob[e.start as usize..e.end as usize]);
        assert_eq!(e.blake3, expected.to_hex().to_string());
    }
    // The manifest survives next to manifest.json without disturbing it.
    assert!(read_manifest(dir.path()).is_ok());
}

#[tokio::test]
async fn server_ignoring_range_is_fatal_for_mid_file_ranges() {
    let blob = Arc::new(synthetic_model(0));
    let (url, _served) = start_server(blob.clone(), false).await; // always 200
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_for(&blob, &[0]);
    let err = fetch_ranges(&url, dir.path(), &plan, |_, _| {})
        .await
        .unwrap_err();
    assert!(
        matches!(err, RangeError::RangeUnsupported { .. }),
        "got {err}"
    );
    assert!(err.to_string().contains("Range requests"), "remedy: {err}");
}

#[tokio::test]
async fn fetch_remote_header_parses_and_reports_total_size() {
    // meta_pad inflates the header past the 256 KiB initial prefix, so the
    // grow-and-refetch path runs against a real server.
    let blob = Arc::new(synthetic_model(400 * 1024));
    let (url, _) = start_server(blob.clone(), true).await;
    let (header, total) = fetch_remote_header(&url).await.unwrap();
    assert_eq!(total, blob.len() as u64);
    assert_eq!(header.tensors.len(), 5);
    assert_eq!(header.block_count(), Some(3));

    // A server that ignores Range (answers 200) still works for headers.
    let (url200, _) = start_server(blob.clone(), false).await;
    let (header, total) = fetch_remote_header(&url200).await.unwrap();
    assert_eq!(total, blob.len() as u64);
    assert_eq!(header.tensors.len(), 5);
}
