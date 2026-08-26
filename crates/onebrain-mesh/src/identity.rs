//! Device identity: the iroh `SecretKey` persisted at `<config_dir>/device-key`.
//!
//! The key is generated from the OS RNG at first daemon start and stored as
//! 64 lowercase hex characters (file mode 0600 on Unix). It never leaves the
//! machine and is never printed. A malformed file is an error — the key is
//! NEVER silently regenerated, because regenerating changes the device's
//! endpoint id and invalidates every existing pairing.

use std::io::Write;
use std::path::Path;

use iroh::SecretKey;

use crate::MeshError;

/// File name of the device key inside the config directory.
pub const DEVICE_KEY_FILE: &str = "device-key";

/// Load the device secret key from `<config_dir>/device-key`, creating it
/// from the OS RNG on first run.
///
/// - Missing file → a new key is generated and written (0600 on Unix).
/// - Present but malformed (not 64 hex chars of 32 bytes) →
///   [`MeshError::IdentityMalformed`]; the caller must not regenerate.
pub fn load_or_create(config_dir: &Path) -> Result<SecretKey, MeshError> {
    let path = config_dir.join(DEVICE_KEY_FILE);
    if path.exists() {
        let raw = std::fs::read_to_string(&path).map_err(|source| MeshError::IdentityRead {
            path: path.clone(),
            source,
        })?;
        let trimmed = raw.trim();
        let decoded = hex::decode(trimmed)
            .map_err(|_| MeshError::IdentityMalformed { path: path.clone() })?;
        let bytes: [u8; 32] = decoded
            .as_slice()
            .try_into()
            .map_err(|_| MeshError::IdentityMalformed { path: path.clone() })?;
        Ok(SecretKey::from_bytes(&bytes))
    } else {
        std::fs::create_dir_all(config_dir).map_err(|source| MeshError::IdentityWrite {
            path: path.clone(),
            source,
        })?;
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).map_err(|e| MeshError::Rng(e.to_string()))?;
        write_new_key_file(&path, &hex::encode(bytes)).map_err(|source| {
            MeshError::IdentityWrite {
                path: path.clone(),
                source,
            }
        })?;
        tracing::info!(path = %path.display(), "generated new device identity");
        Ok(SecretKey::from_bytes(&bytes))
    }
}

/// Create the key file exclusively (never truncating an existing one) with
/// mode 0600 on Unix.
fn write_new_key_file(path: &Path, hex_key: &str) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(hex_key.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_then_loads_same_identity() {
        let dir = tempfile::tempdir().unwrap();
        let first = load_or_create(dir.path()).unwrap();
        let second = load_or_create(dir.path()).unwrap();
        assert_eq!(first.public(), second.public());
        let raw = std::fs::read_to_string(dir.path().join(DEVICE_KEY_FILE)).unwrap();
        assert_eq!(raw.trim().len(), 64);
        assert!(raw.trim().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn malformed_key_is_an_error_not_a_regenerate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DEVICE_KEY_FILE);
        std::fs::write(&path, "not-hex-at-all\n").unwrap();
        let err = load_or_create(dir.path()).unwrap_err();
        assert!(matches!(err, MeshError::IdentityMalformed { .. }));
        // The file must be untouched.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "not-hex-at-all\n");
    }

    #[test]
    fn wrong_length_hex_is_malformed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(DEVICE_KEY_FILE), "abcd1234\n").unwrap();
        let err = load_or_create(dir.path()).unwrap_err();
        assert!(matches!(err, MeshError::IdentityMalformed { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn key_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        load_or_create(dir.path()).unwrap();
        let mode = std::fs::metadata(dir.path().join(DEVICE_KEY_FILE))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
