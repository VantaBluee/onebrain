//! Single-instance enforcement and run state.
//!
//! The OS advisory lock on `<run_dir>/daemon.lock` is the liveness
//! authority: kill -9 releases it via the OS, so it is never stale.
//! `daemon.json` next to it is informational only (pid/port for the CLI)
//! and must never be trusted over the lock (internal-api contract).

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::DaemonError;

/// Held for the daemon's lifetime; dropping it (or the process dying)
/// releases the OS lock.
#[derive(Debug)]
pub struct DaemonLock {
    file: File,
}

impl DaemonLock {
    /// Acquire the exclusive daemon lock, creating `run_dir` as needed.
    /// Fails with [`DaemonError::AlreadyRunning`] when another daemon holds
    /// it.
    pub fn acquire(run_dir: &Path) -> Result<DaemonLock, DaemonError> {
        std::fs::create_dir_all(run_dir).map_err(|source| DaemonError::LockIo {
            path: run_dir.display().to_string(),
            source,
        })?;
        let path = lock_path(run_dir);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|source| DaemonError::LockIo {
                path: path.display().to_string(),
                source,
            })?;
        // Fully qualified: std::fs::File grew an inherent `try_lock` in Rust
        // 1.89 which would otherwise shadow the fs4 trait method.
        match fs4::FileExt::try_lock(&file) {
            Ok(()) => Ok(DaemonLock { file }),
            Err(fs4::TryLockError::WouldBlock) => Err(DaemonError::AlreadyRunning),
            Err(fs4::TryLockError::Error(source)) => Err(DaemonError::LockIo {
                path: path.display().to_string(),
                source,
            }),
        }
    }
}

impl Drop for DaemonLock {
    fn drop(&mut self) {
        // Belt and braces: closing the handle releases the lock anyway.
        let _ = fs4::FileExt::unlock(&self.file);
    }
}

fn lock_path(run_dir: &Path) -> PathBuf {
    run_dir.join("daemon.lock")
}

fn run_info_path(run_dir: &Path) -> PathBuf {
    run_dir.join("daemon.json")
}

/// Contents of `daemon.json` (internal-api contract).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunInfo {
    pub pid: u32,
    pub port: u16,
    pub started_unix: u64,
    pub version: String,
}

/// Write `daemon.json`, overwriting any stale file from a killed daemon.
pub fn write_run_info(run_dir: &Path, info: &RunInfo) -> Result<(), DaemonError> {
    let path = run_info_path(run_dir);
    let raw = serde_json::to_string_pretty(info).expect("RunInfo serializes");
    std::fs::write(&path, raw).map_err(|source| DaemonError::RunStateWrite {
        path: path.display().to_string(),
        source,
    })
}

/// Read `daemon.json` if present. `Ok(None)` means no daemon has written
/// run state (or it was cleaned up on shutdown).
pub fn read_run_info(run_dir: &Path) -> Result<Option<RunInfo>, DaemonError> {
    let path = run_info_path(run_dir);
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(DaemonError::RunStateRead {
                path: path.display().to_string(),
                source,
            })
        }
    };
    serde_json::from_str(&raw)
        .map(Some)
        .map_err(|source| DaemonError::RunStateParse {
            path: path.display().to_string(),
            source,
        })
}

/// Best-effort removal on clean shutdown; a leftover file is harmless
/// because the lock, not this file, is the liveness authority.
pub fn remove_run_info(run_dir: &Path) {
    let _ = std::fs::remove_file(run_info_path(run_dir));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_drop_acquire_works() {
        let dir = tempfile::tempdir().unwrap();
        let lock = DaemonLock::acquire(dir.path()).unwrap();
        drop(lock);
        // A clean release must allow immediate reacquisition (restart path).
        let _again = DaemonLock::acquire(dir.path()).unwrap();
    }

    #[test]
    fn acquire_creates_run_dir() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path().join("data").join("run");
        let _lock = DaemonLock::acquire(&run_dir).unwrap();
        assert!(run_dir.join("daemon.lock").exists());
    }

    #[test]
    fn run_info_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let info = RunInfo {
            pid: 4242,
            port: 11435,
            started_unix: 1_756_200_000,
            version: "0.1.0".into(),
        };
        write_run_info(dir.path(), &info).unwrap();
        assert_eq!(read_run_info(dir.path()).unwrap(), Some(info));
        remove_run_info(dir.path());
        assert_eq!(read_run_info(dir.path()).unwrap(), None);
    }

    #[test]
    fn corrupt_run_info_is_a_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("daemon.json"), "{not json").unwrap();
        assert!(matches!(
            read_run_info(dir.path()),
            Err(DaemonError::RunStateParse { .. })
        ));
    }
}
