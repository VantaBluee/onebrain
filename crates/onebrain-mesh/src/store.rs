//! Persistent peer store: `<config_dir>/peers.toml`.
//!
//! Format per the M2 contract:
//!
//! ```toml
//! [peers.<endpoint_id>]
//! name = "gaming-pc"
//! added_unix = 1789000000
//! ```
//!
//! Names default to the peer's introduced `node_name`, deduplicated with
//! `-2`, `-3` suffixes. The store is re-read from disk on every mesh accept,
//! so `unpair` takes effect without a restart.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::MeshError;

/// One persisted peer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerRecord {
    /// Human-readable name (unique within the store).
    pub name: String,
    /// Unix timestamp of when the pairing completed.
    pub added_unix: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct StoreFile {
    #[serde(default)]
    peers: BTreeMap<String, PeerRecord>,
}

/// Handle to `peers.toml`. Cheap to clone; every operation reads the file
/// fresh so concurrent daemon tasks always see the latest state.
#[derive(Debug, Clone)]
pub struct PeerStore {
    path: PathBuf,
}

impl PeerStore {
    /// A store backed by the given `peers.toml` path (need not exist yet).
    pub fn new(path: PathBuf) -> Self {
        PeerStore { path }
    }

    /// Read all peers. A missing file is an empty store.
    pub fn load(&self) -> Result<BTreeMap<String, PeerRecord>, MeshError> {
        let raw = match std::fs::read_to_string(&self.path) {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
            Err(source) => {
                return Err(MeshError::StoreRead {
                    path: self.path.clone(),
                    source,
                })
            }
        };
        let file: StoreFile = toml::from_str(&raw).map_err(|source| MeshError::StoreParse {
            path: self.path.clone(),
            source: Box::new(source),
        })?;
        Ok(file.peers)
    }

    /// Whether an endpoint id is paired. Reads the file fresh — this is the
    /// check the mesh accept path runs on every connection.
    pub fn contains(&self, endpoint_id: &str) -> Result<bool, MeshError> {
        Ok(self.load()?.contains_key(endpoint_id))
    }

    /// Add a peer under a deduplicated name and persist. Returns the final
    /// name. Re-adding an already-paired id keeps its existing name.
    pub fn add(&self, endpoint_id: &str, wanted_name: &str) -> Result<String, MeshError> {
        let mut peers = self.load()?;
        if let Some(existing) = peers.get(endpoint_id) {
            return Ok(existing.name.clone());
        }
        let base = {
            let trimmed = wanted_name.trim();
            if trimmed.is_empty() {
                "peer".to_string()
            } else {
                trimmed.to_string()
            }
        };
        let mut name = base.clone();
        let mut n = 2u32;
        while peers.values().any(|record| record.name == name) {
            name = format!("{base}-{n}");
            n += 1;
        }
        let added_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        peers.insert(
            endpoint_id.to_string(),
            PeerRecord {
                name: name.clone(),
                added_unix,
            },
        );
        self.save(&peers)?;
        Ok(name)
    }

    /// Remove a peer by name and persist. Returns the removed endpoint id.
    /// An unknown name errors with the list of known names.
    pub fn remove_by_name(&self, name: &str) -> Result<String, MeshError> {
        let mut peers = self.load()?;
        let id = peers
            .iter()
            .find(|(_, record)| record.name == name)
            .map(|(id, _)| id.clone());
        match id {
            Some(id) => {
                peers.remove(&id);
                self.save(&peers)?;
                Ok(id)
            }
            None => {
                let mut known: Vec<String> =
                    peers.values().map(|record| record.name.clone()).collect();
                known.sort();
                Err(MeshError::UnknownPeerName {
                    name: name.to_string(),
                    known,
                })
            }
        }
    }

    /// Persist the given peer set (write-temp-then-rename, best effort
    /// atomic).
    pub fn save(&self, peers: &BTreeMap<String, PeerRecord>) -> Result<(), MeshError> {
        let file = StoreFile {
            peers: peers.clone(),
        };
        let body = toml::to_string_pretty(&file).map_err(|err| MeshError::StoreWrite {
            path: self.path.clone(),
            source: std::io::Error::other(err),
        })?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| MeshError::StoreWrite {
                path: self.path.clone(),
                source,
            })?;
        }
        let tmp = self.path.with_extension("toml.tmp");
        std::fs::write(&tmp, body).map_err(|source| MeshError::StoreWrite {
            path: self.path.clone(),
            source,
        })?;
        // Windows cannot rename over an existing file; remove first.
        if self.path.exists() {
            std::fs::remove_file(&self.path).map_err(|source| MeshError::StoreWrite {
                path: self.path.clone(),
                source,
            })?;
        }
        std::fs::rename(&tmp, &self.path).map_err(|source| MeshError::StoreWrite {
            path: self.path.clone(),
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(dir: &tempfile::TempDir) -> PeerStore {
        PeerStore::new(dir.path().join("peers.toml"))
    }

    #[test]
    fn missing_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(store(&dir).load().unwrap().is_empty());
    }

    #[test]
    fn add_and_reload_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(&dir);
        let name = s.add("aaaa", "gaming-pc").unwrap();
        assert_eq!(name, "gaming-pc");
        let peers = s.load().unwrap();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers["aaaa"].name, "gaming-pc");
        assert!(peers["aaaa"].added_unix > 0);
    }

    #[test]
    fn duplicate_names_get_suffixes() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(&dir);
        assert_eq!(s.add("aaaa", "laptop").unwrap(), "laptop");
        assert_eq!(s.add("bbbb", "laptop").unwrap(), "laptop-2");
        assert_eq!(s.add("cccc", "laptop").unwrap(), "laptop-3");
    }

    #[test]
    fn re_adding_same_id_keeps_name() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(&dir);
        assert_eq!(s.add("aaaa", "laptop").unwrap(), "laptop");
        assert_eq!(s.add("aaaa", "renamed").unwrap(), "laptop");
        assert_eq!(s.load().unwrap().len(), 1);
    }

    #[test]
    fn empty_wanted_name_defaults_to_peer() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(&dir);
        assert_eq!(s.add("aaaa", "  ").unwrap(), "peer");
    }

    #[test]
    fn remove_by_name_and_unknown_lists_known() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(&dir);
        s.add("aaaa", "laptop").unwrap();
        s.add("bbbb", "desktop").unwrap();
        assert_eq!(s.remove_by_name("laptop").unwrap(), "aaaa");
        let err = s.remove_by_name("laptop").unwrap_err();
        match err {
            MeshError::UnknownPeerName { name, known } => {
                assert_eq!(name, "laptop");
                assert_eq!(known, vec!["desktop".to_string()]);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn malformed_toml_is_a_parse_error() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(&dir);
        std::fs::write(dir.path().join("peers.toml"), "not = [valid").unwrap();
        assert!(matches!(s.load(), Err(MeshError::StoreParse { .. })));
    }
}
