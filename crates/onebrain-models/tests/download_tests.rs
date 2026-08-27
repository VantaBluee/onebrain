//! Downloader integration tests against a local axum server (no network).
//! The server implements real `Range`/206 semantics in-test; a variant that
//! ignores `Range` exercises the 200-with-restart path.

use std::path::Path;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;

use onebrain_models::download::{download, read_manifest, verify, Manifest};
use onebrain_models::registry::DownloadSpec;

#[derive(Clone)]
struct BlobServer {
    data: Arc<Vec<u8>>,
    honor_range: bool,
}

async fn serve_blob(State(server): State<BlobServer>, headers: HeaderMap) -> impl IntoResponse {
    let data = &server.data;
    let total = data.len();
    if server.honor_range {
        if let Some(start) = parse_range_start(&headers) {
            if start >= total {
                return (
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    [(header::CONTENT_RANGE, format!("bytes */{total}"))],
                    Vec::new(),
                )
                    .into_response();
            }
            return (
                StatusCode::PARTIAL_CONTENT,
                [(
                    header::CONTENT_RANGE,
                    format!("bytes {start}-{}/{total}", total - 1),
                )],
                data[start..].to_vec(),
            )
                .into_response();
        }
    }
    (StatusCode::OK, data.as_ref().clone()).into_response()
}

/// Parse `Range: bytes=<start>-` (open-ended, the only form the downloader sends).
fn parse_range_start(headers: &HeaderMap) -> Option<usize> {
    headers
        .get(header::RANGE)?
        .to_str()
        .ok()?
        .strip_prefix("bytes=")?
        .strip_suffix('-')?
        .parse()
        .ok()
}

async fn start_server(data: Arc<Vec<u8>>, honor_range: bool) -> String {
    let app = Router::new()
        .route("/model.gguf", get(serve_blob))
        .with_state(BlobServer { data, honor_range });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}/model.gguf")
}

/// Deterministic pseudo-random bytes (xorshift64) — no `rand` dependency.
fn random_blob(len: usize) -> Vec<u8> {
    let mut state = 0x243F_6A88_85A3_08D3_u64;
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

fn spec_for(url: String) -> DownloadSpec {
    DownloadSpec {
        cache_key: "test-model".to_string(),
        url,
        file_name: "model.gguf".to_string(),
    }
}

fn hex_hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

const BLOB_LEN: usize = 2 * 1024 * 1024;

#[tokio::test]
async fn full_download_writes_file_and_manifest() {
    let data = Arc::new(random_blob(BLOB_LEN));
    let url = start_server(data.clone(), true).await;
    let dir = tempfile::tempdir().unwrap();
    let spec = spec_for(url.clone());

    let mut seen: Vec<(u64, u64)> = Vec::new();
    let path = download(&spec, dir.path(), |c, t| seen.push((c, t)))
        .await
        .unwrap();

    assert_eq!(path, dir.path().join("model.gguf"));
    assert_eq!(std::fs::read(&path).unwrap(), **data);
    assert!(
        !dir.path().join("model.gguf.part").exists(),
        "the .part file must be renamed away on completion"
    );

    let manifest = read_manifest(dir.path()).unwrap();
    assert_eq!(manifest.url, url);
    assert_eq!(manifest.size_bytes, BLOB_LEN as u64);
    assert_eq!(manifest.blake3, hex_hash(&data));
    verify(&path, &manifest).await.unwrap();

    let last = *seen.last().unwrap();
    assert_eq!(last, (BLOB_LEN as u64, BLOB_LEN as u64));
    assert!(seen.iter().all(|&(c, t)| c <= t));
}

#[tokio::test]
async fn already_complete_download_returns_without_refetch() {
    let data = Arc::new(random_blob(BLOB_LEN));
    let url = start_server(data.clone(), true).await;
    let dir = tempfile::tempdir().unwrap();
    let spec = spec_for(url);

    let path = download(&spec, dir.path(), |_, _| {}).await.unwrap();
    let manifest_before = read_manifest(dir.path()).unwrap();

    // Second call must short-circuit on the manifest (byte-identical result).
    let path2 = download(&spec, dir.path(), |_, _| {}).await.unwrap();
    assert_eq!(path, path2);
    assert_eq!(read_manifest(dir.path()).unwrap(), manifest_before);
    assert_eq!(std::fs::read(&path2).unwrap(), **data);
}

#[tokio::test]
async fn resume_after_cancel_is_byte_exact() {
    let data = Arc::new(random_blob(BLOB_LEN));
    let url = start_server(data.clone(), true).await;
    let dir = tempfile::tempdir().unwrap();
    let spec = spec_for(url);

    // First run: cancel (drop the future) once ~256 KiB have been flushed.
    let notify = Arc::new(tokio::sync::Notify::new());
    let notifier = notify.clone();
    let fut = download(&spec, dir.path(), move |completed, _| {
        if completed >= 256 * 1024 {
            notifier.notify_one();
        }
    });
    tokio::select! {
        result = fut => panic!("download completed before cancellation: {result:?}"),
        _ = notify.notified() => {} // fut is dropped here
    }

    let part = dir.path().join("model.gguf.part");
    assert!(
        part.exists(),
        "a cancelled download must leave its .part file"
    );
    let part_len = std::fs::metadata(&part).unwrap().len();
    assert!(
        part_len > 0 && part_len < BLOB_LEN as u64,
        "partial length out of range: {part_len}"
    );

    // Second run: must resume from the partial, not restart.
    let mut first_progress = None;
    let path = download(&spec, dir.path(), |completed, _| {
        if first_progress.is_none() {
            first_progress = Some(completed);
        }
    })
    .await
    .unwrap();
    // Not an equality check: dropping a tokio::fs::File lets an in-flight
    // write op complete in the background, so the .part can legitimately
    // grow a little after our metadata read. The invariant is that resume
    // starts from at least what we saw on disk (never from zero) — the
    // byte-exact final hash below is the true correctness proof.
    let resumed_from = first_progress.expect("resume must report progress");
    assert!(
        resumed_from >= part_len && resumed_from < BLOB_LEN as u64,
        "resume must continue from the bytes already on disk \
         (resumed at {resumed_from}, saw {part_len} at cancel)"
    );

    // Byte-exact result: same BLAKE3 as a straight copy of the source blob.
    let final_bytes = std::fs::read(&path).unwrap();
    assert_eq!(final_bytes.len(), BLOB_LEN);
    assert_eq!(hex_hash(&final_bytes), hex_hash(&data));
    let manifest = read_manifest(dir.path()).unwrap();
    assert_eq!(manifest.blake3, hex_hash(&data));
    assert!(!part.exists());
}

#[tokio::test]
async fn server_ignoring_range_restarts_from_zero() {
    let data = Arc::new(random_blob(BLOB_LEN));
    let url = start_server(data.clone(), false).await; // always answers 200
    let dir = tempfile::tempdir().unwrap();
    let spec = spec_for(url);

    // Seed a stale .part full of garbage that must NOT leak into the result.
    std::fs::create_dir_all(dir.path()).unwrap();
    std::fs::write(dir.path().join("model.gguf.part"), vec![0xAB; 100 * 1024]).unwrap();

    let path = download(&spec, dir.path(), |_, _| {}).await.unwrap();
    let final_bytes = std::fs::read(&path).unwrap();
    assert_eq!(final_bytes.len(), BLOB_LEN);
    assert_eq!(hex_hash(&final_bytes), hex_hash(&data));
    assert_eq!(read_manifest(dir.path()).unwrap().blake3, hex_hash(&data));
}

#[tokio::test]
async fn download_preserves_cache_state_fields_in_manifest() {
    let data = Arc::new(random_blob(64 * 1024));
    let url = start_server(data.clone(), true).await;
    let dir = tempfile::tempdir().unwrap();
    let spec = spec_for(url);

    // A user pinned the entry before (or while) the download ran — the
    // completion manifest write must merge, not clobber.
    std::fs::create_dir_all(dir.path()).unwrap();
    std::fs::write(
        dir.path().join("manifest.json"),
        br#"{ "pinned": true, "last_used_unix": 42 }"#,
    )
    .unwrap();

    download(&spec, dir.path(), |_, _| {}).await.unwrap();

    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(json["pinned"], true, "pin state lost: {json}");
    assert_eq!(json["last_used_unix"], 42, "LRU state lost: {json}");
    // And the integrity fields landed as usual.
    let manifest = read_manifest(dir.path()).unwrap();
    assert_eq!(manifest.size_bytes, 64 * 1024);
    assert_eq!(manifest.blake3, hex_hash(&data));
}

#[tokio::test]
async fn corrupted_file_fails_verify() {
    let dir = tempfile::tempdir().unwrap();
    let payload = random_blob(64 * 1024);
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, &payload).unwrap();
    let manifest = Manifest {
        url: "https://example.invalid/model.gguf".to_string(),
        size_bytes: payload.len() as u64,
        blake3: hex_hash(&payload),
    };
    verify(&path, &manifest).await.unwrap();

    // Flip one byte (same length): must fail as a hash mismatch.
    let mut corrupted = payload.clone();
    corrupted[12345] ^= 0xFF;
    std::fs::write(&path, &corrupted).unwrap();
    let err = verify(&path, &manifest).await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("integrity check failed"), "got: {msg}");
    assert!(msg.contains("download it again"), "remedy missing: {msg}");

    // Truncate: must fail as a size mismatch.
    std::fs::write(&path, &payload[..payload.len() - 1]).unwrap();
    let err = verify(&path, &manifest).await.unwrap_err();
    assert!(err.to_string().contains("size mismatch"), "got: {err}");
}

#[tokio::test]
async fn missing_file_yields_404_error_with_remedy() {
    let data = Arc::new(random_blob(1024));
    let url = start_server(data, true).await;
    let dir = tempfile::tempdir().unwrap();
    let spec = DownloadSpec {
        cache_key: "test-model".to_string(),
        url: url.replace("model.gguf", "no-such-file.gguf"),
        file_name: "no-such-file.gguf".to_string(),
    };
    let err = download(&spec, dir.path(), |_, _| {}).await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("404"), "got: {msg}");
    assert!(
        msg.contains("check the model reference"),
        "remedy missing: {msg}"
    );
    assert_no_part_files(dir.path());
}

fn assert_no_part_files(dir: &Path) {
    let leftovers: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".part"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "unexpected .part files: {leftovers:?}"
    );
}
