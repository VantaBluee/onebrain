//! Platform-appropriate directories for config, state, and the model cache.
//! `onebrain doctor` prints all of these (product spec §3).

use std::path::PathBuf;

use directories::ProjectDirs;

use crate::DaemonError;

/// Resolved storage locations for this installation.
#[derive(Debug, Clone)]
pub struct AppPaths {
    /// `config.toml`, device key, paired-peers store.
    pub config_dir: PathBuf,
    /// Model cache (content-addressed ranges), logs, run state.
    pub data_dir: PathBuf,
}

impl AppPaths {
    /// Platform dirs, unless `ONEBRAIN_HOME` is set — then everything lives
    /// under it (`<home>/config`, `<home>/data`). The override exists for
    /// tests and sandboxed runs; never default it in shipped code paths.
    pub fn resolve() -> Result<AppPaths, DaemonError> {
        if let Some(home) = std::env::var_os("ONEBRAIN_HOME") {
            let home = PathBuf::from(home);
            return Ok(AppPaths {
                config_dir: home.join("config"),
                data_dir: home.join("data"),
            });
        }
        let dirs =
            ProjectDirs::from("ai", "onebrain", "onebrain").ok_or(DaemonError::NoConfigDir)?;
        Ok(AppPaths {
            config_dir: dirs.config_dir().to_path_buf(),
            data_dir: dirs.data_dir().to_path_buf(),
        })
    }

    pub fn config_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }

    pub fn model_cache_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_resolve_on_this_platform() {
        let p = AppPaths::resolve().expect("platform dirs must resolve");
        assert!(p.config_file().ends_with("config.toml"));
        assert!(p.model_cache_dir().ends_with("models"));
    }
}
