//! The on-disk model cache: `<root>/<cache_key>/<file>.gguf` plus
//! `manifest.json` (docs/internal-api.md). The caller supplies the root
//! (`<data_dir>/models` in the daemon) — this crate never guesses paths.
//!
//! M6 additions (docs/logistics.md):
//!
//! - the entry manifest carries LRU + pin state (`last_used_unix`,
//!   `pinned`) next to the download integrity fields; manifests written
//!   before M6 lack the fields and parse as unpinned/never-used;
//! - split-GGUF entries store each part as its own download directory
//!   under `<root>/<id>/parts/<part-stem>/`, so the M1 downloader (and the
//!   range store) work per part without any manifest collisions;
//! - GC primitives: [`total_cache_bytes`], [`eviction_candidates`]
//!   (oldest-first, never pinned) and [`evict_entry`]. The config-driven GC
//!   loop itself lives in the daemon — this module only supplies the
//!   mechanism.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::download::{read_manifest, MANIFEST_FILE};
use crate::ranges::RANGES_MANIFEST_FILE;

/// Subdirectory of a cache entry holding split-GGUF part downloads.
pub const PARTS_DIR: &str = "parts";

/// One completed model in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedModel {
    /// The cache directory name (registry id or hf cache key).
    pub id: String,
    /// Path to the model file itself (for split models: the first part).
    pub path: PathBuf,
    /// For split models: the summed size of all complete local parts.
    pub size_bytes: u64,
    /// From `manifest.json` when present and consistent with the file size.
    /// Split models report `None` here — each part carries its own manifest.
    pub blake3: Option<String>,
    /// Number of complete local part files (`1` for single-file entries).
    pub parts: u32,
    /// Pinned entries are never eviction candidates.
    pub pinned: bool,
    /// Seconds since the Unix epoch of the last load/open (`0` = never).
    pub last_used_unix: u64,
}

/// LRU + pin state stored inside the entry manifest (`manifest.json`)
/// alongside the download integrity fields. Manifests from before M6 have
/// neither field and deserialize to the defaults (unpinned, never used).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EntryState {
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub last_used_unix: u64,
}

/// One entry the GC may evict, cheapest-to-lose first once sorted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionCandidate {
    pub id: String,
    /// Total bytes the eviction would free (the whole entry directory,
    /// including range files and split parts).
    pub bytes: u64,
    pub last_used_unix: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("model {id:?} is not in the local cache; `onebrain ls` lists what is")]
    NotCached { id: String },
    #[error(
        "invalid model id {id:?}: cache ids are single path components; \
         `onebrain ls` shows valid ids"
    )]
    InvalidId { id: String },
    #[error(
        "cannot remove model {id:?} while a process holds it open \
         (it is probably loaded); close it first: onebrain stop"
    )]
    InUse {
        id: String,
        #[source]
        source: std::io::Error,
    },
    #[error("model {id:?} is pinned; unpin it first: onebrain unpin {id}")]
    Pinned { id: String },
    #[error(
        "split model {id:?} has {found} of {expected} parts in the local \
         cache; run `onebrain pull` again to fetch the missing parts"
    )]
    SplitIncomplete {
        id: String,
        found: u32,
        expected: u32,
    },
    #[error("i/o error under {path}: {source}; check the directory's permissions")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Scan `<root>` for completed models, sorted by id. A missing root is an
/// empty cache, not an error. Directories holding only `.part` files (an
/// interrupted download) or only range files are skipped.
pub fn list(root: &Path) -> Result<Vec<CachedModel>, CacheError> {
    let mut out = Vec::new();
    for (id, dir) in entry_dirs(root)? {
        if let Some(model) = scan_entry(&dir, id)? {
            out.push(model);
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// The `(id, dir)` of every entry directory under `root` (missing root =
/// no entries), in directory order.
fn entry_dirs(root: &Path) -> Result<Vec<(String, PathBuf)>, CacheError> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => {
            return Err(CacheError::Io {
                path: root.to_path_buf(),
                source: e,
            })
        }
    };
    for entry in entries {
        let entry = entry.map_err(|e| CacheError::Io {
            path: root.to_path_buf(),
            source: e,
        })?;
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().into_owned();
        out.push((id, dir));
    }
    Ok(out)
}

fn scan_entry(dir: &Path, id: String) -> Result<Option<CachedModel>, CacheError> {
    let state = state_of_dir(dir);
    if let Some((path, size_bytes)) = find_model_file(dir)? {
        let blake3 = read_manifest(dir)
            .ok()
            .and_then(|m| (m.size_bytes == size_bytes).then_some(m.blake3));
        return Ok(Some(CachedModel {
            id,
            path,
            size_bytes,
            blake3,
            parts: 1,
            pinned: state.pinned,
            last_used_unix: state.last_used_unix,
        }));
    }
    // No top-level model file: this may be a split entry whose parts live
    // in `parts/<stem>/` download directories.
    let parts = complete_parts(dir)?;
    let Some((first, _)) = parts.first().cloned() else {
        return Ok(None);
    };
    Ok(Some(CachedModel {
        id,
        path: first,
        size_bytes: parts.iter().map(|(_, s)| s).sum(),
        blake3: None,
        parts: parts.len() as u32,
        pinned: state.pinned,
        last_used_unix: state.last_used_unix,
    }))
}

/// The completed model file in a download directory: the largest regular
/// file that is neither a manifest (`manifest.json`, `ranges.json`) nor an
/// in-flight `.part`. Also used by the range store, which serves offset
/// reads from a full file when one exists.
pub(crate) fn find_model_file(dir: &Path) -> Result<Option<(PathBuf, u64)>, CacheError> {
    let io_err = |e: std::io::Error| CacheError::Io {
        path: dir.to_path_buf(),
        source: e,
    };
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io_err(e)),
    };
    // There should be exactly one candidate; if something odd left several,
    // take the largest.
    let mut model_file: Option<(PathBuf, u64)> = None;
    for file in entries {
        let file = file.map_err(io_err)?;
        let name = file.file_name().to_string_lossy().into_owned();
        if name == MANIFEST_FILE || name == RANGES_MANIFEST_FILE || name.ends_with(".part") {
            continue;
        }
        let meta = file.metadata().map_err(io_err)?;
        if !meta.is_file() {
            continue;
        }
        let candidate = (file.path(), meta.len());
        model_file = match model_file {
            Some(prev) if prev.1 >= candidate.1 => Some(prev),
            _ => Some(candidate),
        };
    }
    Ok(model_file)
}

/// Delete one cached model (its whole `<root>/<id>/` directory).
pub fn remove(root: &Path, id: &str) -> Result<(), CacheError> {
    validate_id(id)?;
    let dir = root.join(id);
    if !dir.is_dir() {
        return Err(CacheError::NotCached { id: id.to_string() });
    }
    std::fs::remove_dir_all(&dir).map_err(|e| {
        if is_sharing_violation(&e) {
            CacheError::InUse {
                id: id.to_string(),
                source: e,
            }
        } else {
            CacheError::Io {
                path: dir,
                source: e,
            }
        }
    })
}

/// Pin (or unpin) a cache entry. Pinned entries are never returned by
/// [`eviction_candidates`] and [`evict_entry`] refuses them.
pub fn set_pinned(root: &Path, id: &str, pinned: bool) -> Result<(), CacheError> {
    validate_id(id)?;
    let dir = root.join(id);
    if !dir.is_dir() {
        return Err(CacheError::NotCached { id: id.to_string() });
    }
    update_entry_state(&dir, |map| {
        map.insert("pinned".to_string(), serde_json::Value::Bool(pinned));
    })?;
    tracing::debug!(id, pinned, "cache entry pin state changed");
    Ok(())
}

/// Record a load/open of the entry: sets `last_used_unix` to now. The
/// daemon calls this whenever a model is loaded so LRU eviction order
/// reflects actual use.
pub fn touch(root: &Path, id: &str) -> Result<(), CacheError> {
    validate_id(id)?;
    let dir = root.join(id);
    if !dir.is_dir() {
        return Err(CacheError::NotCached { id: id.to_string() });
    }
    update_entry_state(&dir, |map| {
        map.insert(
            "last_used_unix".to_string(),
            serde_json::Value::from(now_unix()),
        );
    })
}

/// Total bytes under `root` (every entry, including `.part` files, range
/// stores and split parts). A missing root holds zero bytes.
pub fn total_cache_bytes(root: &Path) -> Result<u64, CacheError> {
    match std::fs::metadata(root) {
        Ok(_) => dir_size_bytes(root),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(CacheError::Io {
            path: root.to_path_buf(),
            source: e,
        }),
    }
}

/// Entries the GC may evict: every unpinned entry (including interrupted
/// downloads and range-only entries), least-recently-used first. Ties break
/// on id for a deterministic order. "Never evict the currently loaded
/// model" is the daemon's rule to enforce — it knows what is loaded.
pub fn eviction_candidates(root: &Path) -> Result<Vec<EvictionCandidate>, CacheError> {
    let mut out = Vec::new();
    for (id, dir) in entry_dirs(root)? {
        let state = state_of_dir(&dir);
        if state.pinned {
            continue;
        }
        out.push(EvictionCandidate {
            id,
            bytes: dir_size_bytes(&dir)?,
            last_used_unix: state.last_used_unix,
        });
    }
    out.sort_by(|a, b| {
        a.last_used_unix
            .cmp(&b.last_used_unix)
            .then_with(|| a.id.cmp(&b.id))
    });
    Ok(out)
}

/// Evict one entry, returning the bytes freed. Refuses pinned entries;
/// each successful eviction is logged with the freed byte count (the GC
/// audit trail docs/logistics.md asks for).
pub fn evict_entry(root: &Path, id: &str) -> Result<u64, CacheError> {
    validate_id(id)?;
    let dir = root.join(id);
    if !dir.is_dir() {
        return Err(CacheError::NotCached { id: id.to_string() });
    }
    if state_of_dir(&dir).pinned {
        return Err(CacheError::Pinned { id: id.to_string() });
    }
    let freed = dir_size_bytes(&dir)?;
    remove(root, id)?;
    tracing::info!(id, freed_bytes = freed, "evicted model cache entry");
    Ok(freed)
}

/// Download directory for one part of a split model:
/// `<root>/<id>/parts/<part-stem>/`. Each part is a self-contained
/// download dir (own `manifest.json`, own range store), so the M1
/// downloader and the range fetcher work on it unchanged.
pub fn split_part_dir(root: &Path, id: &str, part_file_name: &str) -> Result<PathBuf, CacheError> {
    validate_id(id)?;
    validate_id(part_file_name)?;
    Ok(root
        .join(id)
        .join(PARTS_DIR)
        .join(part_stem(part_file_name)))
}

/// The ordered list of local model-file paths for a cached model: all
/// complete split parts in load order, or the single file for a non-split
/// entry. Errors with [`CacheError::SplitIncomplete`] when the part names
/// declare more parts than are complete locally — loading a partial set
/// would corrupt the model.
pub fn split_part_paths(root: &Path, id: &str) -> Result<Vec<PathBuf>, CacheError> {
    validate_id(id)?;
    let dir = root.join(id);
    if !dir.is_dir() {
        return Err(CacheError::NotCached { id: id.to_string() });
    }
    let parts = complete_parts(&dir)?;
    if parts.is_empty() {
        return match find_model_file(&dir)? {
            Some((path, _)) => Ok(vec![path]),
            None => Err(CacheError::NotCached { id: id.to_string() }),
        };
    }
    // The declared part count is embedded in every part's file name
    // (`-of-%05d`); an incomplete set must not be handed to the engine.
    let found = parts.len() as u32;
    let declared = parts
        .iter()
        .filter_map(|(p, _)| p.file_name())
        .filter_map(|n| crate::split::parse_split_name(&n.to_string_lossy()))
        .map(|s| s.count)
        .max();
    if let Some(expected) = declared {
        if found < expected {
            return Err(CacheError::SplitIncomplete {
                id: id.to_string(),
                found,
                expected,
            });
        }
    }
    Ok(parts.into_iter().map(|(p, _)| p).collect())
}

/// `(path, size)` of every complete part under `<dir>/parts/`, in load
/// order (the `%05d` naming makes lexicographic order the load order).
/// Part directories without a completed file are skipped.
fn complete_parts(dir: &Path) -> Result<Vec<(PathBuf, u64)>, CacheError> {
    let parts_root = dir.join(PARTS_DIR);
    let entries = match std::fs::read_dir(&parts_root) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(CacheError::Io {
                path: parts_root,
                source: e,
            })
        }
    };
    let mut dirs: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| CacheError::Io {
            path: parts_root.clone(),
            source: e,
        })?;
        if entry.path().is_dir() {
            dirs.push(entry.path());
        }
    }
    dirs.sort();
    let mut out = Vec::new();
    for part_dir in dirs {
        if let Some(found) = find_model_file(&part_dir)? {
            out.push(found);
        }
    }
    Ok(out)
}

/// Directory name for a part download: the file name minus its `.gguf`
/// extension (matched case-insensitively).
fn part_stem(part_file_name: &str) -> &str {
    if part_file_name.to_ascii_lowercase().ends_with(".gguf") {
        &part_file_name[..part_file_name.len() - ".gguf".len()]
    } else {
        part_file_name
    }
}

fn validate_id(id: &str) -> Result<(), CacheError> {
    if id.is_empty() || id.contains('/') || id.contains('\\') || id == "." || id == ".." {
        return Err(CacheError::InvalidId { id: id.to_string() });
    }
    Ok(())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Read the entry state leniently: a missing or unparsable manifest is an
/// unpinned, never-used entry (exactly what a pre-M6 cache looks like).
fn state_of_dir(dir: &Path) -> EntryState {
    std::fs::read(dir.join(MANIFEST_FILE))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

/// Read-modify-write the entry manifest as a JSON object so the state
/// fields coexist with the download integrity fields (and with fields
/// future versions may add) without either side wiping the other.
fn update_entry_state(
    dir: &Path,
    mutate: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) -> Result<(), CacheError> {
    let path = dir.join(MANIFEST_FILE);
    let mut map = std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|v| match v {
            serde_json::Value::Object(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default();
    mutate(&mut map);
    let json = serde_json::to_vec_pretty(&serde_json::Value::Object(map))
        .expect("manifest serialization is infallible");
    std::fs::write(&path, json).map_err(|e| CacheError::Io {
        path: path.clone(),
        source: e,
    })
}

/// Recursive size of a directory in bytes.
fn dir_size_bytes(dir: &Path) -> Result<u64, CacheError> {
    let io_err = |e: std::io::Error| CacheError::Io {
        path: dir.to_path_buf(),
        source: e,
    };
    let mut total = 0u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = match std::fs::read_dir(&current) {
            Ok(entries) => entries,
            // A concurrently evicted subdirectory is not an error.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(io_err(e)),
        };
        for entry in entries {
            let entry = entry.map_err(io_err)?;
            let meta = entry.metadata().map_err(io_err)?;
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    Ok(total)
}

/// Did deletion fail because another process holds the file open?
/// Windows: ERROR_SHARING_VIOLATION (32) / ERROR_LOCK_VIOLATION (33).
/// Unix: EBUSY (16) — rare, since unlinking open files normally succeeds.
fn is_sharing_violation(e: &std::io::Error) -> bool {
    if cfg!(windows) {
        matches!(e.raw_os_error(), Some(32) | Some(33))
    } else {
        e.raw_os_error() == Some(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::Manifest;

    fn seed_model(root: &Path, id: &str, bytes: &[u8], with_manifest: bool) -> PathBuf {
        let dir = root.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.gguf");
        std::fs::write(&file, bytes).unwrap();
        if with_manifest {
            let manifest = Manifest {
                url: format!("https://example.invalid/{id}.gguf"),
                size_bytes: bytes.len() as u64,
                blake3: blake3::hash(bytes).to_hex().to_string(),
            };
            std::fs::write(
                dir.join(MANIFEST_FILE),
                serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();
        }
        file
    }

    /// Write raw state fields into an entry manifest, merging like the
    /// production paths do.
    fn seed_state(root: &Path, id: &str, pinned: bool, last_used_unix: u64) {
        update_entry_state(&root.join(id), |map| {
            map.insert("pinned".into(), serde_json::Value::Bool(pinned));
            map.insert(
                "last_used_unix".into(),
                serde_json::Value::from(last_used_unix),
            );
        })
        .unwrap();
    }

    #[test]
    fn missing_root_lists_empty() {
        let dir = tempfile::tempdir().unwrap();
        let ghost = dir.path().join("never-created");
        assert_eq!(list(&ghost).unwrap(), Vec::new());
        assert_eq!(total_cache_bytes(&ghost).unwrap(), 0);
        assert_eq!(eviction_candidates(&ghost).unwrap(), Vec::new());
    }

    #[test]
    fn list_reports_completed_models_and_skips_partials() {
        let root = tempfile::tempdir().unwrap();
        let root = root.path();
        seed_model(root, "beta-model", b"0123456789", true);
        seed_model(root, "alpha-model", b"abc", false);
        // An interrupted download: directory with only a .part file.
        let partial = root.join("partial-model");
        std::fs::create_dir_all(&partial).unwrap();
        std::fs::write(partial.join("model.gguf.part"), b"xx").unwrap();

        let listed = list(root).unwrap();
        assert_eq!(listed.len(), 2, "partial-model must be skipped: {listed:?}");
        assert_eq!(listed[0].id, "alpha-model"); // sorted
        assert_eq!(listed[0].size_bytes, 3);
        assert_eq!(listed[0].blake3, None);
        assert_eq!(listed[0].parts, 1);
        assert_eq!(listed[1].id, "beta-model");
        assert_eq!(listed[1].size_bytes, 10);
        assert_eq!(
            listed[1].blake3.as_deref(),
            Some(blake3::hash(b"0123456789").to_hex().to_string().as_str())
        );
        assert!(listed[1].path.ends_with("model.gguf"));
    }

    #[test]
    fn stale_manifest_size_drops_the_hash() {
        let root = tempfile::tempdir().unwrap();
        let file = seed_model(root.path(), "m", b"0123456789", true);
        std::fs::write(&file, b"short").unwrap(); // file no longer matches
        let listed = list(root.path()).unwrap();
        assert_eq!(listed[0].blake3, None);
    }

    #[test]
    fn remove_deletes_the_entry() {
        let root = tempfile::tempdir().unwrap();
        seed_model(root.path(), "m", b"abc", true);
        remove(root.path(), "m").unwrap();
        assert!(!root.path().join("m").exists());
        assert_eq!(list(root.path()).unwrap(), Vec::new());
    }

    #[test]
    fn remove_unknown_id_is_not_cached() {
        let root = tempfile::tempdir().unwrap();
        let err = remove(root.path(), "nope").unwrap_err();
        assert!(matches!(err, CacheError::NotCached { .. }), "got {err}");
        assert!(err.to_string().contains("onebrain ls"));
    }

    #[test]
    fn remove_rejects_path_traversal_ids() {
        let root = tempfile::tempdir().unwrap();
        for bad in ["", ".", "..", "a/b", "a\\b"] {
            let err = remove(root.path(), bad).unwrap_err();
            assert!(
                matches!(err, CacheError::InvalidId { .. }),
                "id {bad:?} gave {err}"
            );
        }
    }

    #[test]
    fn old_manifests_parse_as_unpinned_and_never_used() {
        // Byte-for-byte what M1 wrote: url + size_bytes + blake3, nothing else.
        let root = tempfile::tempdir().unwrap();
        seed_model(root.path(), "m", b"weights", true);
        let listed = list(root.path()).unwrap();
        assert_eq!(listed.len(), 1);
        assert!(!listed[0].pinned);
        assert_eq!(listed[0].last_used_unix, 0);
        // And the integrity fields still round-trip through read_manifest.
        let m = read_manifest(&root.path().join("m")).unwrap();
        assert_eq!(m.size_bytes, 7);
    }

    #[test]
    fn pin_touch_roundtrip_preserves_integrity_fields() {
        let root = tempfile::tempdir().unwrap();
        seed_model(root.path(), "m", b"weights", true);

        set_pinned(root.path(), "m", true).unwrap();
        touch(root.path(), "m").unwrap();

        let listed = list(root.path()).unwrap();
        assert!(listed[0].pinned);
        assert!(listed[0].last_used_unix > 0, "touch must set last_used");
        // The state write must not clobber the download manifest fields.
        let m = read_manifest(&root.path().join("m")).unwrap();
        assert_eq!(m.blake3, blake3::hash(b"weights").to_hex().to_string());
        assert!(m.url.contains("example.invalid"));

        set_pinned(root.path(), "m", false).unwrap();
        assert!(!list(root.path()).unwrap()[0].pinned);
    }

    #[test]
    fn pin_and_touch_on_entries_without_manifests_create_state() {
        let root = tempfile::tempdir().unwrap();
        seed_model(root.path(), "m", b"abc", false); // no manifest.json
        set_pinned(root.path(), "m", true).unwrap();
        let listed = list(root.path()).unwrap();
        assert!(listed[0].pinned);
        assert_eq!(listed[0].blake3, None, "state-only manifest has no hash");

        let err = set_pinned(root.path(), "ghost", true).unwrap_err();
        assert!(matches!(err, CacheError::NotCached { .. }), "got {err}");
    }

    #[test]
    fn eviction_candidates_are_lru_ordered_and_never_pinned() {
        let root = tempfile::tempdir().unwrap();
        seed_model(root.path(), "old-pinned", b"aaaa", true);
        seed_model(root.path(), "oldest", b"bb", true);
        seed_model(root.path(), "newer", b"cccccc", true);
        seed_model(root.path(), "never-used", b"d", true);
        seed_state(root.path(), "old-pinned", true, 10);
        seed_state(root.path(), "oldest", false, 20);
        seed_state(root.path(), "newer", false, 30);
        // "never-used" keeps last_used_unix = 0 → evicted first.

        let candidates = eviction_candidates(root.path()).unwrap();
        let ids: Vec<&str> = candidates.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(
            ids,
            ["never-used", "oldest", "newer"],
            "pinned excluded, oldest first"
        );
        assert!(candidates.iter().all(|c| c.bytes > 0));
    }

    #[test]
    fn evict_entry_frees_bytes_and_refuses_pins() {
        let root = tempfile::tempdir().unwrap();
        seed_model(root.path(), "victim", b"0123456789", true);
        seed_model(root.path(), "keeper", b"xyz", true);
        set_pinned(root.path(), "keeper", true).unwrap();

        let before = total_cache_bytes(root.path()).unwrap();
        let freed = evict_entry(root.path(), "victim").unwrap();
        assert!(freed >= 10, "freed must count the whole entry: {freed}");
        assert!(!root.path().join("victim").exists());
        assert_eq!(total_cache_bytes(root.path()).unwrap(), before - freed);

        let err = evict_entry(root.path(), "keeper").unwrap_err();
        assert!(matches!(err, CacheError::Pinned { .. }), "got {err}");
        assert!(err.to_string().contains("onebrain unpin"));
        assert!(root.path().join("keeper").exists());
    }

    #[test]
    fn total_cache_bytes_counts_ranges_and_parts() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("m");
        std::fs::create_dir_all(dir.join("ranges")).unwrap();
        std::fs::write(dir.join("ranges").join("0-100"), vec![0u8; 100]).unwrap();
        std::fs::create_dir_all(dir.join("parts").join("p-00001-of-00002")).unwrap();
        std::fs::write(
            dir.join("parts")
                .join("p-00001-of-00002")
                .join("p-00001-of-00002.gguf"),
            vec![1u8; 50],
        )
        .unwrap();
        assert_eq!(total_cache_bytes(root.path()).unwrap(), 150);
        // Range-only entries never list as completed models…
        assert_eq!(list(root.path()).unwrap().len(), 1); // (the part makes this one listable)
                                                         // …but they are evictable.
        let candidates = eviction_candidates(root.path()).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].bytes, 150);
    }

    #[test]
    fn split_entries_list_as_one_model_and_enumerate_in_order() {
        let root = tempfile::tempdir().unwrap();
        let id = "split-model";
        for (i, bytes) in [b"aaaa".as_slice(), b"bb", b"c"].iter().enumerate() {
            let name = format!("m-{:05}-of-00003.gguf", i + 1);
            let dir = split_part_dir(root.path(), id, &name).unwrap();
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(&name), bytes).unwrap();
        }

        let listed = list(root.path()).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].parts, 3);
        assert_eq!(listed[0].size_bytes, 7); // 4 + 2 + 1
        assert!(listed[0].path.ends_with("m-00001-of-00003.gguf"));

        let paths = split_part_paths(root.path(), id).unwrap();
        assert_eq!(paths.len(), 3);
        for (i, p) in paths.iter().enumerate() {
            assert!(
                p.ends_with(format!("m-{:05}-of-00003.gguf", i + 1)),
                "part {i} out of order: {p:?}"
            );
        }
    }

    #[test]
    fn incomplete_split_sets_refuse_to_enumerate() {
        let root = tempfile::tempdir().unwrap();
        let id = "half-split";
        // Part 2 of 3 is missing entirely; part 3 is only a .part file.
        let name = "m-00001-of-00003.gguf";
        let dir = split_part_dir(root.path(), id, name).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(name), b"x").unwrap();
        let d3 = split_part_dir(root.path(), id, "m-00003-of-00003.gguf").unwrap();
        std::fs::create_dir_all(&d3).unwrap();
        std::fs::write(d3.join("m-00003-of-00003.gguf.part"), b"y").unwrap();

        let err = split_part_paths(root.path(), id).unwrap_err();
        match err {
            CacheError::SplitIncomplete {
                found, expected, ..
            } => {
                assert_eq!((found, expected), (1, 3));
            }
            other => panic!("expected SplitIncomplete, got {other}"),
        }
        assert!(err.to_string().contains("onebrain pull"));
    }

    #[test]
    fn single_file_entries_enumerate_as_one_part() {
        let root = tempfile::tempdir().unwrap();
        let file = seed_model(root.path(), "solo", b"abc", true);
        assert_eq!(split_part_paths(root.path(), "solo").unwrap(), vec![file]);
        let err = split_part_paths(root.path(), "ghost").unwrap_err();
        assert!(matches!(err, CacheError::NotCached { .. }), "got {err}");
    }

    #[cfg(windows)]
    #[test]
    fn remove_while_file_is_open_names_the_remedy() {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 0x1;
        const FILE_SHARE_WRITE: u32 = 0x2; // deliberately no FILE_SHARE_DELETE

        let root = tempfile::tempdir().unwrap();
        let file = seed_model(root.path(), "loaded-model", b"weights", true);
        let _held = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .open(&file)
            .unwrap();

        let err = remove(root.path(), "loaded-model").unwrap_err();
        assert!(matches!(err, CacheError::InUse { .. }), "got {err}");
        assert!(err.to_string().contains("onebrain stop"));
    }
}
