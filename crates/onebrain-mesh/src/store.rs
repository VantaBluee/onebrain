//! Persistent peer store: `<config_dir>/peers.toml`.
//!
//! Format per the M2 contract, extended with last-known addressing so
//! paired daemons can redial each other after a restart:
//!
//! ```toml
//! [peers.<endpoint_id>]
//! name = "gaming-pc"
//! added_unix = 1789000000
//! direct_addrs = ["192.168.1.9:4567"]   # optional, absent in old stores
//! relay_url = "https://relay.example/"  # optional, absent in old stores
//! ```
//!
//! Names default to the peer's introduced `node_name`, deduplicated with
//! `-2`, `-3` suffixes. The store is re-read from disk on every mesh accept,
//! so `unpair` takes effect without a restart. The addressing fields are
//! serde-defaulted: stores written before they existed load fine.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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
    /// Last-known direct socket addresses, refreshed at pairing time and on
    /// every established mesh session. Used to redial the peer after a
    /// daemon restart. Empty for stores written before this field existed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub direct_addrs: Vec<SocketAddr>,
    /// Last-known relay URL, if the peer was ever reached via a relay.
    /// Stored as a string so a hand-edited value can never make the whole
    /// store unreadable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_url: Option<String>,
    /// The peer's OneBrain version from its last mesh `Hello`, refreshed on
    /// every hello exchange (M8: the metrics endpoint and doctor compute
    /// version skew from STORED hello data, so it must survive the
    /// handshake — and the daemon restart). Absent for stores written
    /// before this field existed or peers not heard from since.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product_version: Option<String>,
    /// The peer's engine build id (llama.cpp commit + backend flags + proto
    /// version) from its last `Hello`. Recorded even when the handshake is
    /// judged incompatible — that is exactly when skew advice matters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_build: Option<String>,
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
    /// Serializes every load-modify-save cycle across clones. The service
    /// task and per-session runners write concurrently (pairing persists,
    /// address refreshes, unpair); without this, two interleaved writers
    /// can save from stale snapshots and silently drop a just-added peer.
    write_lock: Arc<Mutex<()>>,
}

impl PeerStore {
    /// A store backed by the given `peers.toml` path (need not exist yet).
    pub fn new(path: PathBuf) -> Self {
        PeerStore {
            path,
            write_lock: Arc::new(Mutex::new(())),
        }
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
        let _guard = self.write_lock.lock().expect("peer store lock poisoned");
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
                direct_addrs: Vec::new(),
                relay_url: None,
                product_version: None,
                engine_build: None,
            },
        );
        self.save(&peers)?;
        Ok(name)
    }

    /// Record a peer's last-known addressing and persist. An unknown id is a
    /// no-op returning `false`: the peer may have been unpaired concurrently
    /// and an address update must never resurrect it. A non-empty
    /// `direct_addrs` REPLACES the stored set (stale ports from before a
    /// peer restart must not accumulate); an empty set keeps what is stored.
    /// `Some` relay replaces, `None` keeps. Returns whether the file
    /// changed.
    pub fn update_addrs(
        &self,
        endpoint_id: &str,
        direct_addrs: Vec<SocketAddr>,
        relay_url: Option<String>,
    ) -> Result<bool, MeshError> {
        let _guard = self.write_lock.lock().expect("peer store lock poisoned");
        let mut peers = self.load()?;
        let Some(record) = peers.get_mut(endpoint_id) else {
            return Ok(false);
        };
        let mut direct = direct_addrs;
        direct.sort();
        direct.dedup();
        let mut changed = false;
        if !direct.is_empty() && record.direct_addrs != direct {
            record.direct_addrs = direct;
            changed = true;
        }
        if relay_url.is_some() && record.relay_url != relay_url {
            record.relay_url = relay_url;
            changed = true;
        }
        if changed {
            self.save(&peers)?;
        }
        Ok(changed)
    }

    /// Record what a peer's `Hello` introduced it as — product version and
    /// engine build — and persist (M8 metrics/doctor: version skew is
    /// computed from stored hello data, docs/product.md §1). Same shape as
    /// [`PeerStore::update_addrs`]: an unknown id is a no-op returning
    /// `false` (an unpair racing the session must never resurrect the
    /// peer), and an unchanged pair skips the write. Returns whether the
    /// file changed.
    pub fn update_hello(
        &self,
        endpoint_id: &str,
        product_version: String,
        engine_build: String,
    ) -> Result<bool, MeshError> {
        let _guard = self.write_lock.lock().expect("peer store lock poisoned");
        let mut peers = self.load()?;
        let Some(record) = peers.get_mut(endpoint_id) else {
            return Ok(false);
        };
        if record.product_version.as_deref() == Some(product_version.as_str())
            && record.engine_build.as_deref() == Some(engine_build.as_str())
        {
            return Ok(false);
        }
        record.product_version = Some(product_version);
        record.engine_build = Some(engine_build);
        self.save(&peers)?;
        Ok(true)
    }

    /// Remove a peer by name and persist. Returns the removed endpoint id.
    /// An unknown name errors with the list of known names.
    pub fn remove_by_name(&self, name: &str) -> Result<String, MeshError> {
        let _guard = self.write_lock.lock().expect("peer store lock poisoned");
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
        // Rename over the destination in one step: Rust's `rename` maps to
        // `MoveFileExW(MOVEFILE_REPLACE_EXISTING)` on Windows, so this
        // replaces an existing `peers.toml` atomically. Removing the file
        // first would open a window in which a concurrent `load` sees
        // NotFound and reports an empty store — a paired daemon flickering
        // to "no peers" mid-query.
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
    fn store_addresses_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(&dir);
        s.add("aaaa", "laptop").unwrap();

        let addrs: Vec<SocketAddr> = vec![
            "192.168.1.9:4567".parse().unwrap(),
            "127.0.0.1:4567".parse().unwrap(),
        ];
        assert!(s
            .update_addrs(
                "aaaa",
                addrs.clone(),
                Some("https://relay.example/".to_string()),
            )
            .unwrap());

        let peers = s.load().unwrap();
        let mut expect = addrs;
        expect.sort();
        assert_eq!(peers["aaaa"].direct_addrs, expect);
        assert_eq!(
            peers["aaaa"].relay_url.as_deref(),
            Some("https://relay.example/")
        );

        // Re-writing identical addressing is a no-op.
        assert!(!s
            .update_addrs("aaaa", peers["aaaa"].direct_addrs.clone(), None)
            .unwrap());
        // An empty direct set keeps the stored addresses (a relay-only
        // session must not erase the last-known direct route).
        assert!(!s.update_addrs("aaaa", Vec::new(), None).unwrap());
        assert_eq!(s.load().unwrap()["aaaa"].direct_addrs.len(), 2);
        // New addressing replaces the old set outright.
        let fresh: Vec<SocketAddr> = vec!["127.0.0.1:9999".parse().unwrap()];
        assert!(s.update_addrs("aaaa", fresh.clone(), None).unwrap());
        assert_eq!(s.load().unwrap()["aaaa"].direct_addrs, fresh);
        // Unknown ids never resurrect an unpaired peer.
        assert!(!s
            .update_addrs("bbbb", vec!["127.0.0.1:1".parse().unwrap()], None)
            .unwrap());
        assert!(!s.load().unwrap().contains_key("bbbb"));
    }

    #[test]
    fn legacy_entry_without_addressing_loads_and_upgrades() {
        let dir = tempfile::tempdir().unwrap();
        // A store written before the addressing fields existed.
        std::fs::write(
            dir.path().join("peers.toml"),
            "[peers.aaaa]\nname = \"gaming-pc\"\nadded_unix = 1789000000\n",
        )
        .unwrap();
        let s = store(&dir);
        let peers = s.load().unwrap();
        assert_eq!(peers["aaaa"].name, "gaming-pc");
        assert!(peers["aaaa"].direct_addrs.is_empty());
        assert!(peers["aaaa"].relay_url.is_none());

        // The legacy entry upgrades in place.
        let addr: SocketAddr = "127.0.0.1:4567".parse().unwrap();
        assert!(s.update_addrs("aaaa", vec![addr], None).unwrap());
        let peers = s.load().unwrap();
        assert_eq!(peers["aaaa"].direct_addrs, vec![addr]);
        assert_eq!(peers["aaaa"].name, "gaming-pc");
    }

    #[test]
    fn concurrent_loads_never_see_a_paired_peer_vanish() {
        // Regression: `save` used to remove `peers.toml` before renaming the
        // temp file into place (believing Windows could not rename over an
        // existing file). A `load` racing into that gap saw NotFound and
        // reported an EMPTY store — pair-sim caught the joiner listing zero
        // peers moments after pairing. Writers now rename atomically, so a
        // reader must always observe the paired peer.
        let dir = tempfile::tempdir().unwrap();
        let s = store(&dir);
        s.add("aaaa", "laptop").unwrap();

        let writer = {
            let s = s.clone();
            std::thread::spawn(move || {
                for port in 1..=300u16 {
                    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
                    s.update_addrs("aaaa", vec![addr], None).unwrap();
                }
            })
        };
        while !writer.is_finished() {
            let peers = s.load().unwrap();
            assert!(
                peers.contains_key("aaaa"),
                "load observed the paired peer missing mid-save"
            );
        }
        writer.join().unwrap();
        assert!(s.load().unwrap().contains_key("aaaa"));
    }

    /// M8: hello data (product version + engine build) persists per peer,
    /// refreshes in place, skips no-op rewrites, and never resurrects an
    /// unpaired id — the metrics/doctor version-skew rules read it back
    /// from the store, so this roundtrip is their whole data path.
    #[test]
    fn hello_data_roundtrips_and_unknown_ids_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(&dir);
        s.add("aaaa", "laptop").unwrap();
        // Absent until a hello lands (legacy stores load the same way).
        assert!(s.load().unwrap()["aaaa"].product_version.is_none());
        assert!(s.load().unwrap()["aaaa"].engine_build.is_none());

        assert!(s
            .update_hello("aaaa", "0.1.0".into(), "llama.cpp-abc/cpu".into())
            .unwrap());
        let peers = s.load().unwrap();
        assert_eq!(peers["aaaa"].product_version.as_deref(), Some("0.1.0"));
        assert_eq!(
            peers["aaaa"].engine_build.as_deref(),
            Some("llama.cpp-abc/cpu")
        );
        // Identical data skips the write; changed data replaces in place.
        assert!(!s
            .update_hello("aaaa", "0.1.0".into(), "llama.cpp-abc/cpu".into())
            .unwrap());
        assert!(s
            .update_hello("aaaa", "0.2.0".into(), "llama.cpp-abc/cpu".into())
            .unwrap());
        assert_eq!(
            s.load().unwrap()["aaaa"].product_version.as_deref(),
            Some("0.2.0")
        );
        // Unknown ids never resurrect an unpaired peer.
        assert!(!s
            .update_hello("bbbb", "0.1.0".into(), "build".into())
            .unwrap());
        assert!(!s.load().unwrap().contains_key("bbbb"));
    }

    #[test]
    fn malformed_toml_is_a_parse_error() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(&dir);
        std::fs::write(dir.path().join("peers.toml"), "not = [valid").unwrap();
        assert!(matches!(s.load(), Err(MeshError::StoreParse { .. })));
    }
}
