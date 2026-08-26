//! The API bearer token: `<config_dir>/api-token`, 64 lowercase hex chars
//! (32 random bytes), created at first daemon start. The CLI reads the file
//! directly (same user); API clients get the value from `onebrain status`.

use crate::paths::AppPaths;
use crate::DaemonError;

/// Byte length of the raw token (64 hex chars once encoded).
const TOKEN_BYTES: usize = 32;

/// Load the token, generating and persisting a fresh one when the file does
/// not exist yet. A malformed file is an error (with the delete-to-reset
/// remedy) rather than being silently rotated: rotation would invalidate
/// clients that already hold the old value.
pub fn load_or_create(paths: &AppPaths) -> Result<String, DaemonError> {
    let path = paths.config_dir.join("api-token");
    let display = path.display().to_string();

    if path.exists() {
        let raw = std::fs::read_to_string(&path).map_err(|source| DaemonError::TokenRead {
            path: display.clone(),
            source,
        })?;
        let token = raw.trim().to_string();
        if !is_valid_token(&token) {
            return Err(DaemonError::TokenInvalid { path: display });
        }
        return Ok(token);
    }

    std::fs::create_dir_all(&paths.config_dir).map_err(|source| DaemonError::TokenWrite {
        path: display.clone(),
        source,
    })?;
    let mut bytes = [0u8; TOKEN_BYTES];
    getrandom::fill(&mut bytes).map_err(|e| DaemonError::Entropy(e.to_string()))?;
    let token = hex::encode(bytes); // hex::encode is lowercase
    std::fs::write(&path, &token).map_err(|source| DaemonError::TokenWrite {
        path: display.clone(),
        source,
    })?;
    restrict_permissions(&path);
    Ok(token)
}

/// 64 lowercase hex characters, nothing else.
fn is_valid_token(token: &str) -> bool {
    token.len() == TOKEN_BYTES * 2
        && token
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Best-effort owner-only mode on Unix; Windows config dirs are already
/// per-user (`%APPDATA%`).
#[cfg(unix)]
fn restrict_permissions(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &std::path::Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_paths(dir: &std::path::Path) -> AppPaths {
        AppPaths {
            config_dir: dir.join("config"),
            data_dir: dir.join("data"),
        }
    }

    #[test]
    fn creates_a_64_char_lowercase_hex_token() {
        let dir = tempfile::tempdir().unwrap();
        let token = load_or_create(&temp_paths(dir.path())).unwrap();
        assert_eq!(token.len(), 64);
        assert!(token
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
    }

    #[test]
    fn is_stable_across_calls() {
        let dir = tempfile::tempdir().unwrap();
        let paths = temp_paths(dir.path());
        let first = load_or_create(&paths).unwrap();
        let second = load_or_create(&paths).unwrap();
        assert_eq!(first, second, "an existing token must be reused");
    }

    #[test]
    fn trailing_newline_is_tolerated() {
        let dir = tempfile::tempdir().unwrap();
        let paths = temp_paths(dir.path());
        let token = load_or_create(&paths).unwrap();
        std::fs::write(paths.config_dir.join("api-token"), format!("{token}\n")).unwrap();
        assert_eq!(load_or_create(&paths).unwrap(), token);
    }

    #[test]
    fn malformed_file_is_a_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let paths = temp_paths(dir.path());
        std::fs::create_dir_all(&paths.config_dir).unwrap();
        std::fs::write(paths.config_dir.join("api-token"), "DEADBEEF").unwrap();
        assert!(matches!(
            load_or_create(&paths),
            Err(DaemonError::TokenInvalid { .. })
        ));
    }
}
