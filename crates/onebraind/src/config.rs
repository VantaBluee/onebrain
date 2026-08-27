//! The daemon's persisted configuration (`config.toml` in the platform
//! config dir). Everything here has a working default: a fresh install runs
//! with no config file at all.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::DaemonError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Human-readable node name shown to peers (defaults to the hostname at
    /// first run; persisted so renaming the machine doesn't re-identify it).
    pub node_name: Option<String>,
    /// Exempt localhost API clients from bearer auth (default true, §7).
    /// There is no switch that disables auth for non-loopback clients.
    pub localhost_auth_exempt: bool,
    /// Battery percentage below which this node advertises "draining,
    /// deprioritize" and drains out of new plans (§5).
    pub battery_drain_threshold: u8,
    /// Address the HTTP API gateway binds. Loopback-only in M1; the
    /// internal-api contract fixes the default port at 11435.
    pub api_bind: String,
    /// Context length for the inference session created at model load.
    pub ctx_len: u32,
    /// Mesh transport switches (`[mesh]` table, docs/mesh.md).
    pub mesh: MeshSection,
    /// Test-only knobs (`[debug]` table, docs/distributed.md).
    pub debug: DebugSection,
}

/// The `[debug]` table: test-only knobs for the cluster simulator. Nothing
/// here changes real resource allocation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DebugSection {
    /// TEST-ONLY (docs/distributed.md "Tests / DoD hooks"): when set, this
    /// value replaces the measured usable memory this node reports in
    /// `NodeStatus` and budgets for itself as head — so the cluster sim can
    /// cap nodes without cgroups. It never touches real allocation; the
    /// engine will still use whatever memory the load actually needs.
    pub usable_memory_override_bytes: Option<u64>,
}

/// The `[mesh]` table: switches passed to `onebrain-mesh::MeshConfig`.
/// Both default on — the mesh is useful out of the box; these exist for
/// locked-down networks (no multicast, no third-party relays).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MeshSection {
    /// Advertise and discover peers on the LAN via mDNS.
    pub enable_mdns: bool,
    /// Use n0's relay + pkarr infrastructure for dial-by-key across
    /// networks. Off, only direct addresses (tickets, mDNS) can connect.
    pub enable_relays: bool,
    /// Pin the mesh's UDP socket to a fixed address (e.g.
    /// "0.0.0.0:4711", or "127.0.0.1:<port>" in tests). Unset = an
    /// ephemeral port chosen by iroh. Pinning keeps peers' stored
    /// addresses valid across daemon restarts and simplifies firewall
    /// rules.
    pub bind_addr: Option<String>,
}

impl Default for MeshSection {
    fn default() -> Self {
        MeshSection {
            enable_mdns: true,
            enable_relays: true,
            bind_addr: None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            node_name: None,
            localhost_auth_exempt: true,
            battery_drain_threshold: 25,
            api_bind: "127.0.0.1:11435".to_string(),
            ctx_len: 4096,
            mesh: MeshSection::default(),
            debug: DebugSection::default(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Config, DaemonError> {
        let display = path.display().to_string();
        if !path.exists() {
            return Ok(Config::default());
        }
        let raw = std::fs::read_to_string(path).map_err(|source| DaemonError::ConfigRead {
            path: display.clone(),
            source,
        })?;
        toml::from_str(&raw).map_err(|source| DaemonError::ConfigParse {
            path: display,
            source: Box::new(source),
        })
    }

    pub fn save(&self, path: &Path) -> Result<(), DaemonError> {
        let display = path.display().to_string();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| DaemonError::ConfigWrite {
                path: display.clone(),
                source,
            })?;
        }
        let raw = toml::to_string_pretty(self).expect("config serializes");
        std::fs::write(path, raw).map_err(|source| DaemonError::ConfigWrite {
            path: display,
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_secure() {
        let c = Config::default();
        // Localhost exemption is the only auth relaxation that exists.
        assert!(c.localhost_auth_exempt);
        assert_eq!(c.battery_drain_threshold, 25);
        // The API listens on loopback only in M1 (internal-api contract).
        assert_eq!(c.api_bind, "127.0.0.1:11435");
        assert_eq!(c.ctx_len, 4096);
        // Mesh defaults on: pairing works out of the box (docs/mesh.md).
        assert!(c.mesh.enable_mdns);
        assert!(c.mesh.enable_relays);
    }

    #[test]
    fn roundtrips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let c = Config {
            node_name: Some("gaming-pc".into()),
            api_bind: "127.0.0.1:0".into(),
            ctx_len: 8192,
            ..Default::default()
        };
        c.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap(), c);
    }

    #[test]
    fn partial_file_fills_new_fields_with_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        // A pre-M1 config file lacks the M1 fields; they must default in.
        std::fs::write(&path, "localhost_auth_exempt = false\n").unwrap();
        let c = Config::load(&path).unwrap();
        assert!(!c.localhost_auth_exempt);
        assert_eq!(c.api_bind, "127.0.0.1:11435");
        assert_eq!(c.ctx_len, 4096);
        assert_eq!(c.mesh, MeshSection::default());
    }

    #[test]
    fn mesh_section_parses_and_defaults_missing_switches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[mesh]\nenable_mdns = false\n").unwrap();
        let c = Config::load(&path).unwrap();
        assert!(!c.mesh.enable_mdns);
        // Unset switches in a partial [mesh] table default on.
        assert!(c.mesh.enable_relays);
    }

    #[test]
    fn unknown_keys_in_mesh_section_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[mesh]\ndisable_auth = true\n").unwrap();
        assert!(matches!(
            Config::load(&path),
            Err(DaemonError::ConfigParse { .. })
        ));
    }

    #[test]
    fn debug_section_defaults_to_no_override() {
        let c = Config::default();
        assert_eq!(c.debug.usable_memory_override_bytes, None);
        // A config file without a [debug] table parses to the same default.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "ctx_len = 2048\n").unwrap();
        assert_eq!(Config::load(&path).unwrap().debug, DebugSection::default());
    }

    #[test]
    fn debug_override_parses() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[debug]\nusable_memory_override_bytes = 1073741824\n",
        )
        .unwrap();
        let c = Config::load(&path).unwrap();
        assert_eq!(c.debug.usable_memory_override_bytes, Some(1 << 30));
    }

    #[test]
    fn unknown_keys_in_debug_section_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[debug]\nusable_memory_override = 5\n").unwrap();
        assert!(matches!(
            Config::load(&path),
            Err(DaemonError::ConfigParse { .. })
        ));
    }

    #[test]
    fn debug_section_roundtrips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let c = Config {
            debug: DebugSection {
                usable_memory_override_bytes: Some(256 * 1024 * 1024),
            },
            ..Default::default()
        };
        c.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap(), c);
    }

    #[test]
    fn missing_file_yields_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let c = Config::load(&dir.path().join("nope.toml")).unwrap();
        assert_eq!(c, Config::default());
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "insecure_mode = true\n").unwrap();
        // There is no insecure mode; a typo'd or hostile config must fail
        // loudly rather than silently ignoring the key (§1.3).
        assert!(matches!(
            Config::load(&path),
            Err(DaemonError::ConfigParse { .. })
        ));
    }
}
