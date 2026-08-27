//! M6 model logistics (docs/logistics.md): LAN-first downloads over the
//! mesh blob store, the worker's range fetch on plan adoption, RPC
//! tensor-cache pre-seeding with its LRU reaper, and the cache GC trigger.
//!
//! # Wire model keys
//!
//! `RangeQuery { model }` needs an identity both sides derive the same way.
//! The key is the download directory's path relative to the cache root:
//! `<id>` for a single-file entry, `<id>/parts/<part-stem>` for one part of
//! a split model. Both sides derive it from the same `DownloadSpec`, so the
//! keys always agree; [`resolve_model_dir`] validates incoming keys before
//! touching the filesystem (a hostile peer must not name `..`).
//!
//! # What a node advertises and serves
//!
//! - Every on-disk range file recorded in `ranges.json` (each one shared
//!   into the mesh blob store after it lands — the manifest hash IS the
//!   blob address).
//! - A completed full file as ONE blob spanning `0..total_size`, addressed
//!   by the `manifest.json` BLAKE3 (already computed at download time). A
//!   full-file entry holds every range by offset reads, but the blob store
//!   shares whole files — so the full file travels as a single blob rather
//!   than materializing per-range duplicates on disk.
//!
//! # Grep-stable log lines (the sim depends on these exact shapes)
//!
//! - `logistics: fetched {X} bytes p2p, {Y} bytes wan for {model}`
//! - `rpc-cache: pre-seeded {N} tensors ({M} bytes) for epoch {E}`
//! - `rpc-cache: {N} tensors already present`

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use onebrain_engine::rpc_cache::{rpc_cache_filename, RPC_HASH_THRESHOLD};
use onebrain_mesh::{MeshHandle, PeerState, RangeInventorySource};
use onebrain_models::gguf::GgufHeader;
use onebrain_models::registry::{DownloadSpec, ModelRef, Resolved};
use onebrain_models::split::{parse_split_name, sibling_url};
use onebrain_models::{cache, download, ranges};

// ---------------------------------------------------------------------------
// Model keys and directory resolution
// ---------------------------------------------------------------------------

/// One path component of a model key: the same alphabet the registry's
/// `sanitize_component` emits, so every real cache id and part stem passes
/// and every traversal attempt (`..`, separators, drive colons) fails.
fn safe_component(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
}

/// Wire key for a single-file entry's download directory.
pub fn model_key_for_entry(id: &str) -> String {
    id.to_string()
}

/// Wire key for one part of a split entry (`<id>/parts/<part-stem>`).
pub fn model_key_for_part(id: &str, part_file_name: &str) -> String {
    let stem = part_file_name
        .strip_suffix(".gguf")
        .or_else(|| part_file_name.strip_suffix(".GGUF"))
        .unwrap_or(part_file_name);
    format!("{id}/{}/{stem}", cache::PARTS_DIR)
}

/// Resolve a wire model key to its download directory under `root`, or
/// `None` when the key is not one this node would ever have produced —
/// which covers both hostile keys (`..`, absolute paths) and non-cacheable
/// references (`local:<stem>` model names).
pub fn resolve_model_dir(root: &Path, model: &str) -> Option<PathBuf> {
    let parts: Vec<&str> = model.split('/').collect();
    match parts.as_slice() {
        [id] if safe_component(id) => Some(root.join(id)),
        [id, mid, stem]
            if *mid == cache::PARTS_DIR && safe_component(id) && safe_component(stem) =>
        {
            Some(root.join(id).join(cache::PARTS_DIR).join(stem))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The local range inventory (answers peers' RangeQuery)
// ---------------------------------------------------------------------------

/// [`RangeInventorySource`] over the models-crate cache: what THIS node can
/// serve over blobs. Wired into `MeshConfig::range_source` by the runtime,
/// so this daemon answers peers' queries with the same inventory shape it
/// expects from them.
pub struct LocalRangeInventory {
    root: PathBuf,
}

impl LocalRangeInventory {
    pub fn new(root: PathBuf) -> LocalRangeInventory {
        LocalRangeInventory { root }
    }
}

impl RangeInventorySource for LocalRangeInventory {
    fn inventory(&self, model: &str) -> Option<(u64, Vec<(u64, u64, [u8; 32])>)> {
        let dir = resolve_model_dir(&self.root, model)?;
        inventory_of_dir(&dir)
    }
}

/// Decode a lowercase-hex BLAKE3 into the wire's fixed array.
fn hex_hash(hex_str: &str) -> Option<[u8; 32]> {
    hex::decode(hex_str).ok()?.try_into().ok()
}

/// One advertised range on the wire: `(start, end, blake3)` — the mesh
/// crate's `RangeEntry` tuple shape.
type WireRange = (u64, u64, [u8; 32]);

/// The servable inventory of one download directory: range FILES recorded
/// in `ranges.json` (fetchable as individual blobs) plus, when a completed
/// full file with a manifest hash exists, the whole file as one
/// `0..total_size` range addressed by the manifest BLAKE3.
fn inventory_of_dir(dir: &Path) -> Option<(u64, Vec<WireRange>)> {
    let mut out: Vec<WireRange> = Vec::new();
    let mut total = 0u64;
    if let Some(manifest) = read_range_manifest(dir) {
        total = manifest.total_size;
        for e in &manifest.ranges {
            // Only ranges backed by a real range file are advertised: they
            // are what was shared into the blob store. Ranges merely covered
            // by a full file cannot be served as individual blobs.
            let file_ok = std::fs::metadata(range_file_path(dir, e.start, e.end))
                .map(|m| m.len() == e.end.saturating_sub(e.start))
                .unwrap_or(false);
            if !file_ok {
                continue;
            }
            let Some(hash) = hex_hash(&e.blake3) else {
                continue;
            };
            out.push((e.start, e.end, hash));
        }
    }
    if let Ok(manifest) = download::read_manifest(dir) {
        let complete = find_model_file(dir)
            .is_some_and(|(_, size)| size == manifest.size_bytes && manifest.size_bytes > 0);
        if complete {
            if let Some(hash) = hex_hash(&manifest.blake3) {
                total = manifest.size_bytes;
                out.push((0, manifest.size_bytes, hash));
            }
        }
    }
    if out.is_empty() {
        return None;
    }
    out.sort_by_key(|(s, e, _)| (*s, *e));
    Some((total, out))
}

/// Read `ranges.json` leniently (missing or corrupt = none).
fn read_range_manifest(dir: &Path) -> Option<ranges::RangeManifest> {
    let bytes = std::fs::read(ranges::ranges_manifest_path(dir)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Path of one range file, matching the models crate's on-disk layout
/// (`ranges/<start>-<end>`, the contract's documented naming).
fn range_file_path(dir: &Path, start: u64, end: u64) -> PathBuf {
    dir.join(ranges::RANGES_DIR).join(format!("{start}-{end}"))
}

/// The completed model file in a download directory — the models crate's
/// rule (largest regular file that is no manifest and no `.part`),
/// replicated here because the original is crate-private.
fn find_model_file(dir: &Path) -> Option<(PathBuf, u64)> {
    let mut best: Option<(PathBuf, u64)> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == download::MANIFEST_FILE
            || name == ranges::RANGES_MANIFEST_FILE
            || name.ends_with(".part")
        {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        if best
            .as_ref()
            .map(|(_, size)| meta.len() > *size)
            .unwrap_or(true)
        {
            best = Some((entry.path(), meta.len()));
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Peer offers (RangeQuery fan-out)
// ---------------------------------------------------------------------------

/// What the connected peers can serve for one model key.
#[derive(Debug, Default)]
struct PeerOffers {
    /// Total size of the model file as the offering peers know it.
    total_size: u64,
    /// `(start, end)` → `(peer id, blob hash)`; the first offering peer
    /// wins (peer order is the mesh's deterministic name-sorted order).
    ranges: HashMap<(u64, u64), (String, [u8; 32])>,
}

impl PeerOffers {
    fn full_file(&self) -> Option<&(String, [u8; 32])> {
        if self.total_size == 0 {
            return None;
        }
        self.ranges.get(&(0, self.total_size))
    }
}

/// Ask every Connected peer what it holds for `model_key`. Failures are
/// logged and treated as "peer has nothing" — LAN-first is an optimization,
/// never a gate.
async fn query_peer_offers(mesh: &MeshHandle, model_key: &str) -> PeerOffers {
    let mut offers = PeerOffers::default();
    let peers = mesh.peers().await.unwrap_or_default();
    for peer in peers.iter().filter(|p| p.state == PeerState::Connected) {
        let inventory = match mesh.range_query(&peer.id, model_key).await {
            Ok(inv) => inv,
            Err(err) => {
                tracing::debug!(peer = %peer.name, model = model_key, error = %err,
                    "range query failed; treating the peer as empty");
                continue;
            }
        };
        if inventory.ranges.is_empty() || inventory.total_size == 0 {
            continue;
        }
        if offers.total_size == 0 {
            offers.total_size = inventory.total_size;
        } else if offers.total_size != inventory.total_size {
            // The peer knows a different file version; mixing would corrupt.
            tracing::warn!(peer = %peer.name, model = model_key,
                "peer reports a different total size; ignoring its ranges");
            continue;
        }
        for (start, end, hash) in inventory.ranges {
            offers
                .ranges
                .entry((start, end))
                .or_insert_with(|| (peer.id.clone(), hash));
        }
    }
    offers
}

// ---------------------------------------------------------------------------
// LAN-first fetch core
// ---------------------------------------------------------------------------

/// Bytes moved by one logistics operation, split by source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FetchOutcome {
    pub p2p_bytes: u64,
    pub wan_bytes: u64,
}

/// The per-download summary line the sim greps — keep the format stable.
fn log_transfer_summary(model_key: &str, outcome: FetchOutcome) {
    tracing::info!(
        "logistics: fetched {} bytes p2p, {} bytes wan for {}",
        outcome.p2p_bytes,
        outcome.wan_bytes,
        model_key
    );
}

/// Is the whole `[start, end)` range already served locally — by its own
/// range file, a recorded covering range, or a full file?
fn locally_present(dir: &Path, start: u64, end: u64) -> bool {
    if std::fs::metadata(range_file_path(dir, start, end))
        .map(|m| m.len() == end.saturating_sub(start))
        .unwrap_or(false)
    {
        return true;
    }
    if let Some(manifest) = read_range_manifest(dir) {
        for e in &manifest.ranges {
            if e.start <= start
                && end <= e.end
                && std::fs::metadata(range_file_path(dir, e.start, e.end))
                    .map(|m| m.len() == e.end - e.start)
                    .unwrap_or(false)
            {
                return true;
            }
        }
    }
    find_model_file(dir).is_some_and(|(_, size)| size >= end)
}

/// Fetch every missing range of `plan`: peer blobs first, then the WAN
/// remainder through the models crate's verifying fetcher (which also
/// adopts the peer-fetched files into `ranges.json`). With an empty `url`,
/// WAN is unavailable — any range still missing after the peer pass is an
/// error naming the remedy.
async fn fetch_plan_core(
    mesh: &MeshHandle,
    url: &str,
    dir: &Path,
    offers: &PeerOffers,
    plan: &ranges::RangePlan,
) -> Result<FetchOutcome, String> {
    let mut p2p_bytes = 0u64;
    let ranges_dir = dir.join(ranges::RANGES_DIR);
    for r in &plan.ranges {
        if locally_present(dir, r.start, r.end) {
            continue;
        }
        let Some((peer, hash)) = offers.ranges.get(&(r.start, r.end)) else {
            continue;
        };
        if let Err(err) = tokio::fs::create_dir_all(&ranges_dir).await {
            return Err(format!(
                "cannot create {}: {err}; check free disk space and permissions",
                ranges_dir.display()
            ));
        }
        let target = range_file_path(dir, r.start, r.end);
        match mesh.fetch_blob(peer, *hash, &target).await {
            Ok(bytes) => p2p_bytes += bytes,
            Err(err) => {
                // The peer may have evicted the file since it advertised it;
                // the WAN pass below covers the gap.
                tracing::warn!(
                    range = %format!("{}-{}", r.start, r.end),
                    error = %err,
                    "peer blob fetch failed; falling back to WAN for this range"
                );
            }
        }
    }
    if url.is_empty() {
        // No WAN route (unresolvable model reference): everything must have
        // arrived from peers or already been local.
        if let Some(r) = plan
            .ranges
            .iter()
            .find(|r| !locally_present(dir, r.start, r.end))
        {
            return Err(format!(
                "bytes {}..{} of the model are neither in the local cache nor \
                 available from any connected peer, and the model reference \
                 does not resolve to a download URL; pull the model on this \
                 node (or the head) first: onebrain pull",
                r.start, r.end
            ));
        }
    }
    // Verification + manifest adoption + the WAN remainder. With everything
    // local this issues no requests. An empty url only reaches here when
    // every range is local (checked above).
    let effective_url = if url.is_empty() {
        read_range_manifest(dir).map(|m| m.url).unwrap_or_default()
    } else {
        url.to_string()
    };
    let wan_bytes = fetch_ranges_off_thread(effective_url, dir.to_path_buf(), plan.clone()).await?;
    Ok(FetchOutcome {
        p2p_bytes,
        wan_bytes,
    })
}

/// Drive `ranges::fetch_ranges` on the blocking pool with its own small
/// runtime: its future is not `Send` (it threads a `&mut dyn FnMut`
/// progress sink internally), so it cannot ride a spawned task directly.
async fn fetch_ranges_off_thread(
    url: String,
    dir: PathBuf,
    plan: ranges::RangePlan,
) -> Result<u64, String> {
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("building the range-fetch runtime failed: {e}; retry"))?;
        rt.block_on(ranges::fetch_ranges(&url, &dir, &plan, |_, _| {}))
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|_| "the range fetch task failed unexpectedly; retry the download".to_string())?
}

/// Share every on-disk range file of `plan` into the mesh blob store so
/// paired peers can fetch them (the manifest hash doubles as the address).
/// Best-effort: a failed import only costs future P2P savings.
async fn share_plan_ranges(mesh: &MeshHandle, dir: &Path, plan: &ranges::RangePlan) {
    for r in &plan.ranges {
        let path = range_file_path(dir, r.start, r.end);
        if std::fs::metadata(&path)
            .map(|m| m.len() == r.end - r.start)
            .unwrap_or(false)
        {
            if let Err(err) = mesh.share_blob(&path).await {
                tracing::debug!(path = %path.display(), error = %err,
                    "could not share range file into the blob store");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Header acquisition
// ---------------------------------------------------------------------------

/// Parse a GGUF header from a growing prefix of a local file.
fn parse_local_header(path: &Path, size: u64) -> Result<GgufHeader, String> {
    use std::io::Read;
    let mut want = (1u64 << 20).min(size);
    loop {
        let mut bytes = Vec::with_capacity(want as usize);
        std::fs::File::open(path)
            .and_then(|f| f.take(want).read_to_end(&mut bytes))
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        match GgufHeader::parse(&bytes) {
            Ok(header) => return Ok(header),
            Err(onebrain_models::gguf::GgufError::NeedMoreData { need_hint }) if want < size => {
                want = need_hint.max(want.saturating_mul(2)).min(size);
            }
            Err(e) => return Err(format!("cannot parse {}: {e}", path.display())),
        }
    }
}

/// Obtain the model's GGUF header and total file size, cheapest source
/// first: a local full file, the locally cached header range, a peer's
/// header-range blob, then the WAN (when a URL exists).
async fn obtain_header(
    mesh: &MeshHandle,
    dir: &Path,
    url: &str,
    offers: &PeerOffers,
) -> Result<(GgufHeader, u64, u64 /* p2p bytes spent */), String> {
    // 1. A completed local full file.
    if let Some((path, size)) = find_model_file(dir) {
        if let Ok(header) = parse_local_header(&path, size) {
            return Ok((header, size, 0));
        }
    }
    // 2. The locally cached header range (`0-<header_len>` per ranges.json).
    if let Some(manifest) = read_range_manifest(dir) {
        if manifest.header_len > 0 {
            if let Ok(bytes) = ranges::read_range(dir, 0, manifest.header_len) {
                if let Ok(header) = GgufHeader::parse(&bytes) {
                    return Ok((header, manifest.total_size, 0));
                }
            }
        }
    }
    // 3. A peer's header range: the smallest advertised range starting at
    // byte 0 that is not the whole file.
    let peer_header = offers
        .ranges
        .iter()
        .filter(|((start, end), _)| *start == 0 && *end < offers.total_size)
        .min_by_key(|((_, end), _)| *end);
    if let Some((&(start, end), (peer, hash))) = peer_header {
        let ranges_dir = dir.join(ranges::RANGES_DIR);
        if tokio::fs::create_dir_all(&ranges_dir).await.is_ok() {
            let target = range_file_path(dir, start, end);
            match mesh.fetch_blob(peer, *hash, &target).await {
                Ok(bytes) => {
                    if let Ok(raw) = std::fs::read(&target) {
                        if let Ok(header) = GgufHeader::parse(&raw) {
                            return Ok((header, offers.total_size, bytes));
                        }
                    }
                    tracing::warn!(peer = %peer, "peer header range did not parse; trying WAN");
                }
                Err(err) => {
                    tracing::debug!(error = %err, "peer header fetch failed; trying WAN");
                }
            }
        }
    }
    // 4. The WAN.
    if !url.is_empty() {
        let (header, total) = ranges::fetch_remote_header(url)
            .await
            .map_err(|e| e.to_string())?;
        return Ok((header, total, 0));
    }
    Err(
        "cannot obtain the model's GGUF header: no local copy, no peer holds it, and the \
         model reference does not resolve to a download URL; pull the model first: \
         onebrain pull"
            .to_string(),
    )
}

/// Every layer named by the header's tensors (for a full-coverage plan).
fn all_layers(header: &GgufHeader) -> BTreeSet<u64> {
    header
        .tensors
        .iter()
        .filter_map(|t| ranges::tensor_layer(&t.name))
        .collect()
}

// ---------------------------------------------------------------------------
// Worker range fetch (plan adoption)
// ---------------------------------------------------------------------------

/// The result of a worker's range fetch: what moved, plus the parsed header
/// the RPC pre-seed step reuses.
pub struct LayerFetch {
    pub outcome: FetchOutcome,
    pub header: GgufHeader,
    pub total_size: u64,
    /// The download directory the ranges live in.
    pub dir: PathBuf,
}

/// Fetch ONLY the header range plus the given layers' tensor ranges of
/// `model` (a plan's model id), LAN-first (docs/logistics.md). Returns
/// `Ok(None)` when the model is not range-fetchable on this node — a
/// `local:` path reference or a split entry — in which case the head's
/// weight push covers the worker exactly as in M3. Re-invocations reuse
/// every range already on disk; nothing re-downloads.
pub async fn fetch_layer_ranges(
    mesh: &MeshHandle,
    cache_root: &Path,
    model: &str,
    layers: &BTreeSet<u64>,
) -> Result<Option<LayerFetch>, String> {
    let Some(dir) = resolve_model_dir(cache_root, model) else {
        tracing::debug!(model, "not a cacheable model id; skipping the range fetch");
        return Ok(None);
    };
    // Resolve a WAN URL when the model id round-trips through the registry
    // (`hf--…` cache keys do not — those rely on peers, usually the head).
    let (url, file_name, is_split) = match model.parse::<ModelRef>().map(|r| r.resolve()) {
        Ok(Ok(Resolved::Remote(spec))) => {
            let split = parse_split_name(&spec.file_name).is_some();
            (spec.url, spec.file_name, split)
        }
        _ => (String::new(), String::new(), false),
    };
    if is_split || dir.join(cache::PARTS_DIR).is_dir() {
        // Split sets are fetched per part by the pull path; mapping layers
        // to parts needs every part's header, which is not worth the WAN
        // spend here — the head pushes the weights either way.
        tracing::debug!(model, "split model; worker range fetch not attempted");
        return Ok(None);
    }
    if let Err(err) = tokio::fs::create_dir_all(&dir).await {
        return Err(format!(
            "cannot create {}: {err}; check permissions",
            dir.display()
        ));
    }

    let offers = query_peer_offers(mesh, model).await;
    let (header, total_size, header_p2p) = match obtain_header(mesh, &dir, &url, &offers).await {
        Ok(found) => found,
        Err(message) => {
            // Last resort: a peer offers the whole file as one blob (a
            // full-file head that never indexed sub-ranges). Fetching it
            // costs more disk than the layer subset but zero WAN.
            if let Some((peer, hash)) = offers.full_file().cloned() {
                let name = if file_name.is_empty() {
                    "model.gguf".to_string()
                } else {
                    file_name.clone()
                };
                let bytes =
                    fetch_full_blob(mesh, &dir, &peer, hash, offers.total_size, &name, &url)
                        .await?;
                let outcome = FetchOutcome {
                    p2p_bytes: bytes,
                    wan_bytes: 0,
                };
                log_transfer_summary(model, outcome);
                let (path, size) = find_model_file(&dir)
                    .ok_or_else(|| "fetched model file vanished".to_string())?;
                let header = parse_local_header(&path, size)?;
                return Ok(Some(LayerFetch {
                    outcome,
                    header,
                    total_size: size,
                    dir,
                }));
            }
            return Err(message);
        }
    };

    let plan = ranges::plan_ranges(&header, total_size, layers).map_err(|e| e.to_string())?;
    let mut outcome = fetch_plan_core(mesh, &url, &dir, &offers, &plan).await?;
    outcome.p2p_bytes += header_p2p;
    log_transfer_summary(model, outcome);
    // What this node now holds, its peers can fetch (spec §6: bytes spread
    // through the cluster without re-downloads).
    share_plan_ranges(mesh, &dir, &plan).await;
    Ok(Some(LayerFetch {
        outcome,
        header,
        total_size,
        dir,
    }))
}

/// Fetch a peer's whole-file blob into `dir/<file_name>` and record its
/// manifest (the blob hash IS the file's BLAKE3, bao-verified in transit).
async fn fetch_full_blob(
    mesh: &MeshHandle,
    dir: &Path,
    peer: &str,
    hash: [u8; 32],
    total_size: u64,
    file_name: &str,
    url: &str,
) -> Result<u64, String> {
    let final_path = dir.join(file_name);
    let bytes = mesh
        .fetch_blob(peer, hash, &final_path)
        .await
        .map_err(|e| e.to_string())?;
    let size = std::fs::metadata(&final_path)
        .map(|m| m.len())
        .map_err(|e| format!("cannot stat {}: {e}", final_path.display()))?;
    if size != total_size {
        return Err(format!(
            "peer blob for {} is {size} bytes but the inventory promised {total_size}; \
             retry the download",
            final_path.display()
        ));
    }
    write_merged_manifest(dir, url, size, &hex::encode(hash))?;
    Ok(bytes)
}

/// Merge-write `manifest.json` (url/size/blake3) preserving any pin/LRU
/// state fields already present — the same read-modify-write discipline the
/// models crate applies from its side.
fn write_merged_manifest(dir: &Path, url: &str, size: u64, blake3_hex: &str) -> Result<(), String> {
    let path = download::manifest_path(dir);
    let mut map = std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|v| match v {
            serde_json::Value::Object(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default();
    map.insert("url".into(), serde_json::Value::from(url));
    map.insert("size_bytes".into(), serde_json::Value::from(size));
    map.insert("blake3".into(), serde_json::Value::from(blake3_hex));
    let json = serde_json::to_vec_pretty(&serde_json::Value::Object(map))
        .expect("manifest serialization is infallible");
    std::fs::write(&path, json)
        .map_err(|e| format!("cannot write {}: {e}; check permissions", path.display()))
}

// ---------------------------------------------------------------------------
// Full downloads (pull / load), LAN-first
// ---------------------------------------------------------------------------

/// A remote model made local: every part path in load order.
pub struct FetchedModel {
    pub paths: Vec<PathBuf>,
    pub size_bytes: u64,
}

/// Make `spec` fully local, LAN-first, handling split sets (every part is
/// its own download directory, `llama_split_path` naming). Emits the
/// per-download summary line for every part that actually fetched. After
/// each completed file the entry is indexed for range serving and shared
/// into the blob store, and its LRU stamp is touched.
pub async fn ensure_remote_local(
    mesh: &MeshHandle,
    cache_root: &Path,
    spec: &DownloadSpec,
    mut progress: impl FnMut(u64, u64) + Send,
) -> Result<FetchedModel, String> {
    if let Some(split) = parse_split_name(&spec.file_name) {
        let mut done = 0u64;
        for part_name in split.part_file_names() {
            let dir = cache::split_part_dir(cache_root, &spec.cache_key, &part_name)
                .map_err(|e| e.to_string())?;
            let part_spec = DownloadSpec {
                cache_key: spec.cache_key.clone(),
                url: sibling_url(&spec.url, &part_name),
                file_name: part_name.clone(),
            };
            let key = model_key_for_part(&spec.cache_key, &part_name);
            // Per-part progress rides the cumulative byte count; the total
            // is unknown until every part has reported, so it stays 0 (the
            // progress contract's "size not known yet").
            let path = download_one_lan_first(mesh, &dir, &key, &part_spec, &mut |c, _| {
                progress(done + c, 0)
            })
            .await?;
            done += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        }
    } else {
        let dir = cache_root.join(&spec.cache_key);
        download_one_lan_first(mesh, &dir, &spec.cache_key, spec, &mut progress).await?;
    }
    if let Err(err) = cache::touch(cache_root, &spec.cache_key) {
        tracing::debug!(id = %spec.cache_key, error = %err, "could not touch the cache entry");
    }
    let paths = cache::split_part_paths(cache_root, &spec.cache_key).map_err(|e| e.to_string())?;
    let size_bytes = paths
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    Ok(FetchedModel { paths, size_bytes })
}

/// One file (a single-file model or one split part), LAN-first:
///
/// 1. cached already → done, no queries, no summary line;
/// 2. a peer offers the whole file as one blob → fetch it P2P (zero WAN);
/// 3. peers offer sub-ranges → fetch those P2P, the remainder over WAN
///    ranges, then assemble the file locally;
/// 4. no peer has anything → the plain M1 resumable WAN download.
async fn download_one_lan_first(
    mesh: &MeshHandle,
    dir: &Path,
    model_key: &str,
    spec: &DownloadSpec,
    progress: &mut (dyn FnMut(u64, u64) + Send),
) -> Result<PathBuf, String> {
    let final_path = dir.join(&spec.file_name);
    // Fast path (same rule as the models downloader): this exact file
    // already completed — not a download, no summary line. A pre-M6 entry
    // (never indexed) is indexed and shared exactly once here, so a head
    // that cached the model before this build still answers its workers.
    if let Ok(manifest) = download::read_manifest(dir) {
        if manifest.url == spec.url {
            if let Ok(meta) = std::fs::metadata(&final_path) {
                if meta.len() == manifest.size_bytes {
                    let never_indexed = ranges::present_ranges(dir)
                        .map(|r| r.is_empty())
                        .unwrap_or(true);
                    if never_indexed {
                        finish_completed_file(mesh, dir, &spec.url).await;
                    }
                    progress(manifest.size_bytes, manifest.size_bytes);
                    return Ok(final_path);
                }
            }
        }
    }

    if let Err(err) = tokio::fs::create_dir_all(dir).await {
        return Err(format!(
            "cannot create {}: {err}; check permissions",
            dir.display()
        ));
    }
    let offers = query_peer_offers(mesh, model_key).await;

    // Whole-file blob from a peer: the zero-WAN pull (spec §6 DoD).
    if let Some((peer, hash)) = offers.full_file().cloned() {
        match fetch_full_blob(
            mesh,
            dir,
            &peer,
            hash,
            offers.total_size,
            &spec.file_name,
            &spec.url,
        )
        .await
        {
            Ok(bytes) => {
                progress(offers.total_size, offers.total_size);
                let outcome = FetchOutcome {
                    p2p_bytes: bytes,
                    wan_bytes: 0,
                };
                log_transfer_summary(model_key, outcome);
                finish_completed_file(mesh, dir, &spec.url).await;
                return Ok(final_path);
            }
            Err(err) => {
                tracing::warn!(error = %err,
                    "whole-file peer fetch failed; falling back to ranges/WAN");
            }
        }
    }

    // Sub-ranges from peers + WAN remainder, assembled locally.
    if !offers.ranges.is_empty() {
        match assemble_from_ranges(mesh, dir, model_key, spec, &offers).await {
            Ok(path) => {
                progress(offers.total_size, offers.total_size);
                finish_completed_file(mesh, dir, &spec.url).await;
                return Ok(path);
            }
            Err(err) => {
                tracing::warn!(error = %err,
                    "range assembly failed; falling back to the full WAN download");
            }
        }
    }

    // Plain WAN download (the M1 path, resumable). WAN bytes exclude any
    // `.part` bytes a previous run already banked.
    let resumed = std::fs::metadata(dir.join(format!("{}.part", spec.file_name)))
        .map(|m| m.len())
        .unwrap_or(0);
    let path = download::download(spec, dir, progress)
        .await
        .map_err(|e| e.to_string())?;
    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let outcome = FetchOutcome {
        p2p_bytes: 0,
        wan_bytes: size.saturating_sub(resumed),
    };
    log_transfer_summary(model_key, outcome);
    finish_completed_file(mesh, dir, &spec.url).await;
    Ok(path)
}

/// Fetch every tensor-aligned range (peers first, WAN remainder) and
/// assemble the complete file from them.
async fn assemble_from_ranges(
    mesh: &MeshHandle,
    dir: &Path,
    model_key: &str,
    spec: &DownloadSpec,
    offers: &PeerOffers,
) -> Result<PathBuf, String> {
    let (header, total_size, header_p2p) = obtain_header(mesh, dir, &spec.url, offers).await?;
    let layers = all_layers(&header);
    let plan = ranges::plan_ranges(&header, total_size, &layers).map_err(|e| e.to_string())?;
    // The plan must tile the whole file for assembly to be exact.
    let mut expect = 0u64;
    for r in &plan.ranges {
        if r.start != expect {
            return Err(format!(
                "tensor ranges leave a gap at byte {expect}; cannot assemble from ranges"
            ));
        }
        expect = r.end;
    }
    if expect != total_size {
        return Err(format!(
            "tensor ranges cover {expect} of {total_size} bytes; cannot assemble from ranges"
        ));
    }

    let mut outcome = fetch_plan_core(mesh, &spec.url, dir, offers, &plan).await?;
    outcome.p2p_bytes += header_p2p;
    log_transfer_summary(model_key, outcome);

    // Concatenate the verified range files into the final file, then swap
    // the range files for the (implicit) full-file coverage.
    let dir_owned = dir.to_path_buf();
    let file_name = spec.file_name.clone();
    let url = spec.url.clone();
    let plan_ranges: Vec<ranges::ByteRange> = plan.ranges.clone();
    let final_path = tokio::task::spawn_blocking(move || -> Result<PathBuf, String> {
        use std::io::{Read, Write};
        let assemble = dir_owned.join(format!("{file_name}.assemble.part"));
        let io_err = |e: std::io::Error| {
            format!(
                "i/o error assembling {}: {e}; check free disk space",
                assemble.display()
            )
        };
        {
            let mut out =
                std::io::BufWriter::new(std::fs::File::create(&assemble).map_err(io_err)?);
            let mut buf = vec![0u8; 4 << 20];
            for r in &plan_ranges {
                let path = range_file_path(&dir_owned, r.start, r.end);
                let mut file = std::fs::File::open(&path).map_err(io_err)?;
                loop {
                    let n = file.read(&mut buf).map_err(io_err)?;
                    if n == 0 {
                        break;
                    }
                    out.write_all(&buf[..n]).map_err(io_err)?;
                }
            }
            out.flush().map_err(io_err)?;
        }
        let final_path = dir_owned.join(&file_name);
        std::fs::rename(&assemble, &final_path).map_err(io_err)?;
        Ok(final_path)
    })
    .await
    .map_err(|_| "the assembly task failed unexpectedly; retry the download".to_string())??;

    let (size, hash) = download::hash_file(&final_path)
        .await
        .map_err(|e| e.to_string())?;
    if size != total_size {
        return Err(format!(
            "assembled file is {size} bytes, expected {total_size}; retry the download"
        ));
    }
    write_merged_manifest(dir, &url, size, &hash)?;
    // The full file now implicitly holds every range (offset reads); the
    // separate range files would double the disk footprint.
    let _ = tokio::fs::remove_dir_all(dir.join(ranges::RANGES_DIR)).await;
    Ok(final_path)
}

/// Post-completion bookkeeping for a directory that now holds a full model
/// file: index its ranges (so peers can be answered range-by-range) and
/// share the file into the blob store (so peers can fetch it whole).
/// Best-effort — failures cost future P2P savings, never the download.
async fn finish_completed_file(mesh: &MeshHandle, dir: &Path, url: &str) {
    let needs_index = ranges::present_ranges(dir)
        .map(|r| r.is_empty())
        .unwrap_or(true);
    if needs_index {
        if let Err(err) = ranges::index_full_file(dir, url).await {
            tracing::warn!(dir = %dir.display(), error = %err,
                "could not index the model file for range serving");
        }
    }
    if let Some((path, _)) = find_model_file(dir) {
        if let Err(err) = mesh.share_blob(&path).await {
            tracing::debug!(path = %path.display(), error = %err,
                "could not share the model file into the blob store");
        }
    }
}

// ---------------------------------------------------------------------------
// RPC tensor-cache pre-seeding + reaper (ADR 0004 payoff)
// ---------------------------------------------------------------------------

/// What one pre-seed pass did.
#[derive(Debug, Default)]
pub struct PreseedStats {
    /// Cache files newly written.
    pub seeded: u32,
    pub seeded_bytes: u64,
    /// Cache files that already existed (LRU-bumped instead).
    pub present: u32,
    /// Every cache file name the epoch's plan references — the reaper must
    /// never evict these while the epoch is active.
    pub protected: HashSet<String>,
}

/// Pre-seed `<data_dir>/rpc-cache/` for one adopted plan: every assigned
/// tensor payload strictly larger than [`RPC_HASH_THRESHOLD`] is written
/// under its FNV-1a-64 name (the RPC protocol's lookup key — see
/// `onebrain_engine::rpc_cache`), read straight from the range store. The
/// serve session then answers `SET_TENSOR_HASH` from these files and the
/// head's push skips the transfers. Integrity is BLAKE3-at-download; FNV is
/// only the protocol's lookup key.
///
/// Upstream's threshold check is strictly `>` on both ends, so `>` here
/// avoids seeding an exactly-threshold tensor that would never be looked up
/// (the contract's "≥ 10 MiB" wording is satisfied either way — see the
/// note on `RPC_HASH_THRESHOLD`).
///
/// Emits the grep-stable lines
/// `rpc-cache: pre-seeded {N} tensors ({M} bytes) for epoch {E}` (something
/// new was written) or `rpc-cache: {N} tensors already present` (an
/// identical plan re-adopted). Blocking file I/O — call from a blocking
/// context.
pub fn preseed_rpc_cache(
    rpc_cache_dir: &Path,
    model_dir: &Path,
    header: &GgufHeader,
    total_size: u64,
    layers: &BTreeSet<u64>,
    epoch: u64,
) -> Result<PreseedStats, String> {
    std::fs::create_dir_all(rpc_cache_dir).map_err(|e| {
        format!(
            "cannot create the rpc cache dir {}: {e}; check permissions",
            rpc_cache_dir.display()
        )
    })?;
    let tensor_ranges = header
        .tensor_ranges(total_size)
        .map_err(|e| e.to_string())?;
    let mut stats = PreseedStats::default();
    for tr in &tensor_ranges {
        let owned = matches!(ranges::tensor_layer(&tr.name), Some(l) if layers.contains(&l));
        if !owned || tr.end.saturating_sub(tr.start) <= RPC_HASH_THRESHOLD {
            continue;
        }
        let payload = match ranges::read_range(model_dir, tr.start, tr.end) {
            Ok(bytes) => bytes,
            Err(err) => {
                // The range never landed locally (fetch failed, local: model
                // on the head, …). The head's push covers the tensor; only
                // the skip saving is lost.
                tracing::debug!(tensor = %tr.name, error = %err,
                    "tensor bytes not local; skipping its pre-seed");
                continue;
            }
        };
        let name = rpc_cache_filename(&payload);
        let path = rpc_cache_dir.join(&name);
        let already = std::fs::metadata(&path)
            .map(|m| m.len() == payload.len() as u64)
            .unwrap_or(false);
        if already {
            stats.present += 1;
            // Bump the LRU stamp so a hot file never looks reap-worthy.
            if let Ok(file) = std::fs::File::options().write(true).open(&path) {
                let now = std::fs::FileTimes::new().set_modified(SystemTime::now());
                let _ = file.set_times(now);
            }
        } else {
            let tmp = rpc_cache_dir.join(format!("{name}.tmp"));
            std::fs::write(&tmp, &payload)
                .and_then(|()| std::fs::rename(&tmp, &path))
                .map_err(|e| {
                    format!(
                        "cannot write rpc cache file {}: {e}; check free disk space",
                        path.display()
                    )
                })?;
            stats.seeded += 1;
            stats.seeded_bytes += payload.len() as u64;
        }
        stats.protected.insert(name);
    }
    if stats.seeded > 0 {
        tracing::info!(
            "rpc-cache: pre-seeded {} tensors ({} bytes) for epoch {}",
            stats.seeded,
            stats.seeded_bytes,
            epoch
        );
    } else {
        tracing::info!("rpc-cache: {} tensors already present", stats.present);
    }
    Ok(stats)
}

/// LRU-cap the rpc-cache at `max_bytes`: remove oldest-modified files first
/// until the directory fits, never touching `protected` names (the ACTIVE
/// epoch's tensors). Returns the bytes freed. Blocking file I/O.
pub fn reap_rpc_cache(rpc_cache_dir: &Path, max_bytes: u64, protected: &HashSet<String>) -> u64 {
    let entries = match std::fs::read_dir(rpc_cache_dir) {
        Ok(entries) => entries,
        Err(_) => return 0, // nothing cached yet
    };
    let mut files: Vec<(String, PathBuf, u64, SystemTime)> = Vec::new();
    let mut total = 0u64;
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        total += meta.len();
        files.push((
            entry.file_name().to_string_lossy().into_owned(),
            entry.path(),
            meta.len(),
            meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        ));
    }
    if total <= max_bytes {
        return 0;
    }
    // Oldest first; name tiebreak keeps the order deterministic.
    files.sort_by(|a, b| a.3.cmp(&b.3).then_with(|| a.0.cmp(&b.0)));
    let mut freed = 0u64;
    let mut removed = 0u32;
    for (name, path, len, _) in files {
        if total <= max_bytes {
            break;
        }
        if protected.contains(&name) {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => {
                tracing::debug!(file = %name, bytes = len, "rpc-cache file reaped");
                total -= len;
                freed += len;
                removed += 1;
            }
            Err(err) => {
                tracing::warn!(file = %name, error = %err, "could not reap rpc-cache file");
            }
        }
    }
    if freed > 0 {
        tracing::info!(
            removed,
            freed_bytes = freed,
            max_bytes,
            "rpc-cache reaped to fit its cap"
        );
    }
    freed
}

// ---------------------------------------------------------------------------
// Cache GC (LRU + pinning)
// ---------------------------------------------------------------------------

/// The post-download GC trigger (docs/logistics.md "LRU GC + pinning"):
/// when the cache exceeds `max_bytes` (0 = disabled), evict LRU entries via
/// the models-crate candidates — never pinned (the candidates exclude
/// them), never an id in `protected_ids` (currently loaded models and the
/// entry just downloaded). Each eviction is logged with its freed bytes.
/// Blocking file I/O — call from a blocking context.
pub fn run_cache_gc(cache_root: &Path, max_bytes: u64, protected_ids: &HashSet<String>) -> u64 {
    if max_bytes == 0 {
        return 0;
    }
    let mut total = match cache::total_cache_bytes(cache_root) {
        Ok(total) => total,
        Err(err) => {
            tracing::warn!(error = %err, "cache gc could not size the cache; skipping");
            return 0;
        }
    };
    if total <= max_bytes {
        return 0;
    }
    let candidates = match cache::eviction_candidates(cache_root) {
        Ok(candidates) => candidates,
        Err(err) => {
            tracing::warn!(error = %err, "cache gc could not list candidates; skipping");
            return 0;
        }
    };
    let mut freed_total = 0u64;
    for candidate in candidates {
        if total <= max_bytes {
            break;
        }
        if protected_ids.contains(&candidate.id) {
            continue;
        }
        match cache::evict_entry(cache_root, &candidate.id) {
            Ok(freed) => {
                tracing::info!(
                    id = %candidate.id,
                    freed_bytes = freed,
                    cache_max_bytes = max_bytes,
                    "cache gc evicted a model over the cache cap"
                );
                total = total.saturating_sub(freed);
                freed_total += freed;
            }
            Err(err) => {
                // Pinned cannot appear here; InUse can (a load raced the
                // GC) — skip it, the next trigger retries.
                tracing::warn!(id = %candidate.id, error = %err, "cache gc eviction failed");
            }
        }
    }
    freed_total
}

#[cfg(test)]
mod tests {
    use super::*;
    use onebrain_models::gguf::TensorInfo;
    use std::collections::BTreeMap;

    // ------------------------------------------------------------------
    // Model keys
    // ------------------------------------------------------------------

    #[test]
    fn model_keys_roundtrip_through_resolution() {
        let root = Path::new("root");
        let entry = model_key_for_entry("qwen3-0.6b");
        assert_eq!(
            resolve_model_dir(root, &entry),
            Some(root.join("qwen3-0.6b"))
        );
        let part = model_key_for_part("big-model", "m-00002-of-00003.gguf");
        assert_eq!(part, "big-model/parts/m-00002-of-00003");
        assert_eq!(
            resolve_model_dir(root, &part),
            Some(
                root.join("big-model")
                    .join("parts")
                    .join("m-00002-of-00003")
            )
        );
    }

    #[test]
    fn hostile_model_keys_do_not_resolve() {
        let root = Path::new("root");
        for bad in [
            "",
            ".",
            "..",
            "../evil",
            "a/../b",
            "a/b",         // wrong shape (2 components, not parts)
            "a/parts/..",  // traversal in the stem
            "a\\b",        // backslash separator
            "C:",          // drive colon
            "local:model", // local reference names are never cache dirs
            "a/parts/b/c", // too deep
            "a/portz/b",   // wrong middle component
        ] {
            assert_eq!(
                resolve_model_dir(root, bad),
                None,
                "key {bad:?} must not resolve"
            );
        }
    }

    // ------------------------------------------------------------------
    // Inventory
    // ------------------------------------------------------------------

    #[test]
    fn inventory_reports_range_files_and_full_files() {
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        // A range file with a recorded hash…
        std::fs::create_dir_all(dir.join(ranges::RANGES_DIR)).unwrap();
        let range_bytes = vec![7u8; 64];
        std::fs::write(dir.join(ranges::RANGES_DIR).join("0-64"), &range_bytes).unwrap();
        let range_hash = blake3::hash(&range_bytes);
        let manifest = ranges::RangeManifest {
            url: "https://example.invalid/m.gguf".into(),
            total_size: 256,
            header_len: 64,
            ranges: vec![
                ranges::RangeEntry {
                    start: 0,
                    end: 64,
                    blake3: range_hash.to_hex().to_string(),
                },
                // …and one recorded range whose file is MISSING (covered
                // only implicitly): it must not be advertised.
                ranges::RangeEntry {
                    start: 64,
                    end: 128,
                    blake3: range_hash.to_hex().to_string(),
                },
            ],
        };
        std::fs::write(
            ranges::ranges_manifest_path(dir),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let (total, inv) = inventory_of_dir(dir).expect("inventory present");
        assert_eq!(total, 256);
        assert_eq!(inv.len(), 1, "only the file-backed range: {inv:?}");
        assert_eq!((inv[0].0, inv[0].1), (0, 64));
        assert_eq!(&inv[0].2, range_hash.as_bytes());

        // Adding a completed full file with its manifest advertises the
        // whole file as one blob addressed by the manifest hash.
        let full = vec![9u8; 256];
        std::fs::write(dir.join("m.gguf"), &full).unwrap();
        let full_hash = blake3::hash(&full);
        std::fs::write(
            dir.join(download::MANIFEST_FILE),
            serde_json::to_vec(&download::Manifest {
                url: "https://example.invalid/m.gguf".into(),
                size_bytes: 256,
                blake3: full_hash.to_hex().to_string(),
            })
            .unwrap(),
        )
        .unwrap();
        let (total, inv) = inventory_of_dir(dir).expect("inventory present");
        assert_eq!(total, 256);
        assert!(inv.contains(&(0, 256, *full_hash.as_bytes())), "{inv:?}");
    }

    #[test]
    fn empty_dirs_have_no_inventory() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(inventory_of_dir(dir.path()), None);
        // A stale manifest whose file is gone advertises nothing either.
        std::fs::write(
            dir.path().join(download::MANIFEST_FILE),
            serde_json::to_vec(&download::Manifest {
                url: "u".into(),
                size_bytes: 10,
                blake3: blake3::hash(b"x").to_hex().to_string(),
            })
            .unwrap(),
        )
        .unwrap();
        assert_eq!(inventory_of_dir(dir.path()), None);
    }

    // ------------------------------------------------------------------
    // Pre-seed + reaper
    // ------------------------------------------------------------------

    /// A synthetic single-layer-per-tensor GGUF header whose data section
    /// starts at `data_offset` and whose tensors are laid out back-to-back.
    fn synthetic_header(tensors: &[(&str, u64)]) -> (GgufHeader, u64) {
        let mut infos = Vec::new();
        let mut offset = 0u64;
        for (name, size) in tensors {
            infos.push(TensorInfo {
                name: (*name).to_string(),
                dims: vec![1],
                ggml_type: 0,
                offset,
            });
            offset += size;
        }
        let header = GgufHeader {
            version: 3,
            metadata: BTreeMap::new(),
            tensors: infos,
            header_len: 96,
            alignment: 32,
            data_offset: 96,
        };
        (header, 96 + offset)
    }

    /// Deterministic pseudo-random content so hashes differ per tensor.
    fn patterned(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u64).wrapping_mul(31).wrapping_add(seed as u64) as u8)
            .collect()
    }

    const BIG: u64 = RPC_HASH_THRESHOLD + 4096; // strictly over the threshold

    /// Build a model dir holding a FULL file for the synthetic header (the
    /// range reader serves pre-seed payloads by offset from it).
    fn seed_model_dir(dir: &Path, header_and_size: &(GgufHeader, u64)) -> Vec<u8> {
        let (header, total) = header_and_size;
        let mut file = vec![0u8; *total as usize];
        for (i, t) in header.tensors.iter().enumerate() {
            let start = (header.data_offset + t.offset) as usize;
            let len = if i + 1 < header.tensors.len() {
                (header.tensors[i + 1].offset - t.offset) as usize
            } else {
                *total as usize - start
            };
            file[start..start + len].copy_from_slice(&patterned(len, i as u8 + 1));
        }
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model.gguf"), &file).unwrap();
        file
    }

    #[test]
    fn preseed_writes_fnv_named_files_for_owned_big_tensors_only() {
        let spec = synthetic_header(&[
            ("blk.0.big.weight", BIG),
            ("blk.1.big.weight", BIG),
            ("blk.0.small.weight", 64), // under the threshold
            ("token_embd.weight", BIG), // shared: never "assigned"
        ]);
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        let file = seed_model_dir(&model_dir, &spec);
        let cache_dir = tmp.path().join("rpc-cache");

        let layers: BTreeSet<u64> = [0].into_iter().collect();
        let stats = preseed_rpc_cache(&cache_dir, &model_dir, &spec.0, spec.1, &layers, 7).unwrap();
        assert_eq!(stats.seeded, 1, "only blk.0.big is owned AND big");
        assert_eq!(stats.seeded_bytes, BIG);
        assert_eq!(stats.present, 0);

        // The file is named EXACTLY by the engine's FNV-1a-64 rule over the
        // tensor's payload bytes, holds those bytes, and is protected.
        let payload = &file[96..96 + BIG as usize];
        let expected_name = rpc_cache_filename(payload);
        assert!(stats.protected.contains(&expected_name));
        let on_disk = std::fs::read(cache_dir.join(&expected_name)).unwrap();
        assert_eq!(
            on_disk, payload,
            "payload bytes must match the tensor range"
        );
        assert_eq!(
            std::fs::read_dir(&cache_dir).unwrap().count(),
            1,
            "small and shared tensors must not be seeded"
        );

        // A second identical plan finds everything present (the sim greps
        // the 'already present' line this path emits).
        let stats2 =
            preseed_rpc_cache(&cache_dir, &model_dir, &spec.0, spec.1, &layers, 8).unwrap();
        assert_eq!(stats2.seeded, 0);
        assert_eq!(stats2.present, 1);
        assert!(stats2.protected.contains(&expected_name));

        // Owning both layers seeds the second big tensor too.
        let both: BTreeSet<u64> = [0, 1].into_iter().collect();
        let stats3 = preseed_rpc_cache(&cache_dir, &model_dir, &spec.0, spec.1, &both, 9).unwrap();
        assert_eq!(stats3.seeded, 1);
        assert_eq!(stats3.present, 1);
        assert_eq!(stats3.protected.len(), 2);
    }

    #[test]
    fn reaper_is_lru_and_never_touches_the_active_epochs_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let write = |name: &str, len: usize, age_secs: u64| {
            let path = dir.join(name);
            std::fs::write(&path, vec![0u8; len]).unwrap();
            let mtime = SystemTime::now() - std::time::Duration::from_secs(age_secs);
            let file = std::fs::File::options().write(true).open(&path).unwrap();
            file.set_times(std::fs::FileTimes::new().set_modified(mtime))
                .unwrap();
        };
        write("oldest-protected", 100, 300);
        write("old", 100, 200);
        write("newer", 100, 100);
        write("newest", 100, 0);

        let protected: HashSet<String> = ["oldest-protected".to_string()].into_iter().collect();
        // Cap at 250 bytes: 400 on disk, must free >= 150 → "old" (oldest
        // unprotected) then "newer" go; "newest" and the protected survive.
        let freed = reap_rpc_cache(dir, 250, &protected);
        assert_eq!(freed, 200, "two 100-byte files reaped");
        assert!(dir.join("oldest-protected").exists(), "protected survives");
        assert!(!dir.join("old").exists(), "oldest unprotected reaped first");
        assert!(!dir.join("newer").exists());
        assert!(dir.join("newest").exists(), "newest survives under the cap");

        // Under the cap: a no-op.
        assert_eq!(reap_rpc_cache(dir, 1 << 20, &HashSet::new()), 0);
        // Missing dir: a no-op.
        assert_eq!(reap_rpc_cache(&dir.join("ghost"), 1, &HashSet::new()), 0);
    }

    // ------------------------------------------------------------------
    // Cache GC
    // ------------------------------------------------------------------

    /// Seed one cache entry with a model file and raw manifest state.
    fn seed_entry(root: &Path, id: &str, len: usize, pinned: bool, last_used: u64) {
        let dir = root.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.gguf"), vec![0u8; len]).unwrap();
        std::fs::write(
            dir.join(download::MANIFEST_FILE),
            serde_json::to_vec(&serde_json::json!({
                "url": format!("https://example.invalid/{id}.gguf"),
                "size_bytes": len,
                "blake3": "00",
                "pinned": pinned,
                "last_used_unix": last_used,
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn cache_gc_honors_pin_loaded_and_lru_order() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_entry(root, "pinned-old", 1000, true, 10);
        seed_entry(root, "loaded-old", 1000, false, 20);
        seed_entry(root, "oldest", 1000, false, 30);
        seed_entry(root, "newest", 1000, false, 40);

        let protected: HashSet<String> = ["loaded-old".to_string()].into_iter().collect();
        // Everything on disk is a bit over 4000 bytes (manifests included);
        // cap so exactly one eviction suffices.
        let total = cache::total_cache_bytes(root).unwrap();
        let cap = total - 500;
        let freed = run_cache_gc(root, cap, &protected);
        assert!(freed >= 1000, "one full entry must go: {freed}");
        assert!(root.join("pinned-old").exists(), "pinned never evicted");
        assert!(root.join("loaded-old").exists(), "loaded never evicted");
        assert!(
            !root.join("oldest").exists(),
            "LRU: oldest unprotected goes"
        );
        assert!(root.join("newest").exists());
        assert!(cache::total_cache_bytes(root).unwrap() <= cap);
    }

    #[test]
    fn cache_gc_zero_cap_disables_and_under_cap_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        seed_entry(tmp.path(), "m", 100, false, 0);
        assert_eq!(run_cache_gc(tmp.path(), 0, &HashSet::new()), 0);
        assert!(
            tmp.path().join("m").exists(),
            "0 = disabled, nothing evicted"
        );
        assert_eq!(run_cache_gc(tmp.path(), 1 << 30, &HashSet::new()), 0);
        assert!(tmp.path().join("m").exists());
    }

    #[test]
    fn merged_manifest_preserves_pin_state() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(
            dir.join(download::MANIFEST_FILE),
            serde_json::to_vec(&serde_json::json!({
                "url": "old", "size_bytes": 1, "blake3": "aa",
                "pinned": true, "last_used_unix": 123,
            }))
            .unwrap(),
        )
        .unwrap();
        write_merged_manifest(dir, "https://new.invalid/m.gguf", 42, "bb").unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join(download::MANIFEST_FILE)).unwrap())
                .unwrap();
        assert_eq!(value["url"], "https://new.invalid/m.gguf");
        assert_eq!(value["size_bytes"], 42);
        assert_eq!(value["blake3"], "bb");
        assert_eq!(value["pinned"], true, "pin state must survive the merge");
        assert_eq!(value["last_used_unix"], 123);
    }
}
