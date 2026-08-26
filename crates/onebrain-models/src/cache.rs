//! The on-disk model cache: `<root>/<cache_key>/<file>.gguf` plus
//! `manifest.json` (docs/internal-api.md). The caller supplies the root
//! (`<data_dir>/models` in the daemon) — this crate never guesses paths.

use std::path::{Path, PathBuf};

use crate::download::{read_manifest, MANIFEST_FILE};

/// One completed model in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedModel {
    /// The cache directory name (registry id or hf cache key).
    pub id: String,
    /// Path to the model file itself.
    pub path: PathBuf,
    pub size_bytes: u64,
    /// From `manifest.json` when present and consistent with the file size.
    pub blake3: Option<String>,
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
    #[error("i/o error under {path}: {source}; check the directory's permissions")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Scan `<root>` for completed models, sorted by id. A missing root is an
/// empty cache, not an error. Directories holding only `.part` files (an
/// interrupted download) are skipped.
pub fn list(root: &Path) -> Result<Vec<CachedModel>, CacheError> {
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
        if let Some(model) = scan_entry(&dir, id)? {
            out.push(model);
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

fn scan_entry(dir: &Path, id: String) -> Result<Option<CachedModel>, CacheError> {
    let io_err = |e: std::io::Error| CacheError::Io {
        path: dir.to_path_buf(),
        source: e,
    };
    // The completed model file: not the manifest, not a `.part`. There
    // should be exactly one; if something odd left several, take the largest.
    let mut model_file: Option<(PathBuf, u64)> = None;
    for file in std::fs::read_dir(dir).map_err(io_err)? {
        let file = file.map_err(io_err)?;
        let name = file.file_name().to_string_lossy().into_owned();
        if name == MANIFEST_FILE || name.ends_with(".part") {
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
    let Some((path, size_bytes)) = model_file else {
        return Ok(None);
    };
    let blake3 = read_manifest(dir)
        .ok()
        .and_then(|m| (m.size_bytes == size_bytes).then_some(m.blake3));
    Ok(Some(CachedModel {
        id,
        path,
        size_bytes,
        blake3,
    }))
}

/// Delete one cached model (its whole `<root>/<id>/` directory).
pub fn remove(root: &Path, id: &str) -> Result<(), CacheError> {
    if id.is_empty() || id.contains('/') || id.contains('\\') || id == "." || id == ".." {
        return Err(CacheError::InvalidId { id: id.to_string() });
    }
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

    #[test]
    fn missing_root_lists_empty() {
        let dir = tempfile::tempdir().unwrap();
        let ghost = dir.path().join("never-created");
        assert_eq!(list(&ghost).unwrap(), Vec::new());
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
