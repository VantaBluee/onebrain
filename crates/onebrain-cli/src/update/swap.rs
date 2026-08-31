//! Getting the verified new binary onto disk: pulling the executable out
//! of whatever shape the release asset has, then the executable swap —
//! including the Windows rename-running-exe dance.

use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::UpdateError;

/// `onebrain.exe` on Windows, `onebrain` elsewhere.
pub fn exe_name() -> String {
    format!("onebrain{}", std::env::consts::EXE_SUFFIX)
}

/// Pull the `onebrain` executable out of a downloaded release asset into
/// `out_dir`. Bare binaries pass through; `.tar.gz`/`.tgz` and `.zip`
/// archives are searched (at any depth) for an entry named exactly like
/// this platform's executable.
pub fn extract_executable(
    asset_name: &str,
    asset_path: &Path,
    out_dir: &Path,
) -> Result<PathBuf, UpdateError> {
    let target = out_dir.join(exe_name());
    if asset_name.ends_with(".tar.gz") || asset_name.ends_with(".tgz") {
        extract_tar_gz(asset_name, asset_path, &target)?;
    } else if asset_name.ends_with(".zip") {
        extract_zip(asset_name, asset_path, &target)?;
    } else {
        // Asset selection only lets bare executables through here.
        fs::copy(asset_path, &target)
            .map_err(|e| extract_err(asset_name, format!("copying the binary failed: {e}")))?;
    }
    make_executable(&target)
        .map_err(|e| extract_err(asset_name, format!("marking it executable failed: {e}")))?;
    Ok(target)
}

fn extract_tar_gz(asset: &str, path: &Path, target: &Path) -> Result<(), UpdateError> {
    let file =
        fs::File::open(path).map_err(|e| extract_err(asset, format!("opening it failed: {e}")))?;
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
    let wanted = exe_name();
    let entries = archive
        .entries()
        .map_err(|e| extract_err(asset, format!("reading the archive failed: {e}")))?;
    for entry in entries {
        let mut entry =
            entry.map_err(|e| extract_err(asset, format!("reading an entry failed: {e}")))?;
        let is_wanted = entry
            .path()
            .ok()
            .is_some_and(|p| p.file_name() == Some(OsStr::new(&wanted)));
        if is_wanted {
            let mut out = fs::File::create(target)
                .map_err(|e| extract_err(asset, format!("writing the binary failed: {e}")))?;
            io::copy(&mut entry, &mut out)
                .map_err(|e| extract_err(asset, format!("writing the binary failed: {e}")))?;
            return Ok(());
        }
    }
    Err(extract_err(
        asset,
        format!("no `{wanted}` inside the archive"),
    ))
}

fn extract_zip(asset: &str, path: &Path, target: &Path) -> Result<(), UpdateError> {
    let file =
        fs::File::open(path).map_err(|e| extract_err(asset, format!("opening it failed: {e}")))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| extract_err(asset, format!("reading the archive failed: {e}")))?;
    let wanted = exe_name();
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| extract_err(asset, format!("reading an entry failed: {e}")))?;
        if entry.is_dir() {
            continue;
        }
        let is_wanted = entry
            .enclosed_name()
            .is_some_and(|p| p.file_name() == Some(OsStr::new(&wanted)));
        if !is_wanted {
            continue;
        }
        let mut out = fs::File::create(target)
            .map_err(|e| extract_err(asset, format!("writing the binary failed: {e}")))?;
        io::copy(&mut entry, &mut out)
            .map_err(|e| extract_err(asset, format!("writing the binary failed: {e}")))?;
        return Ok(());
    }
    Err(extract_err(
        asset,
        format!("no `{wanted}` inside the archive"),
    ))
}

/// The parking spot the previous binary is renamed to during the swap:
/// the current path with `.old` appended (`onebrain.exe.old`).
pub fn old_path(current: &Path) -> PathBuf {
    let mut parked = current.as_os_str().to_owned();
    parked.push(".old");
    PathBuf::from(parked)
}

/// Swap `new_exe` into place at `current`.
///
/// The dance exists for Windows, which lets a RUNNING executable be
/// renamed but never deleted or overwritten: the old binary is renamed
/// aside to [`old_path`], the staged new one renamed in, and the parked
/// file removed — which fails (harmlessly) while the old image is still
/// executing, so the caller gets its path back to tell the user, and the
/// next self-update sweeps it. Unix needs none of that but the same steps
/// are valid there, so one code path serves all platforms.
///
/// Staging happens in `current`'s own directory: `rename` is only atomic
/// (and on Windows only possible) within one volume, and temp dirs are
/// routinely elsewhere.
pub fn swap_executable(new_exe: &Path, current: &Path) -> Result<Option<PathBuf>, UpdateError> {
    let dir = current
        .parent()
        .ok_or_else(|| swap_err("the current executable has no parent directory".to_string()))?;
    let staged = dir.join(format!(".{}.new", exe_name()));
    fs::copy(new_exe, &staged).map_err(|e| {
        swap_err(format!(
            "staging next to the current executable failed: {e}"
        ))
    })?;
    if let Err(e) = make_executable(&staged) {
        let _ = fs::remove_file(&staged);
        return Err(swap_err(format!(
            "marking the staged binary executable failed: {e}"
        )));
    }

    let old = old_path(current);
    let _ = fs::remove_file(&old);
    if let Err(e) = fs::rename(current, &old) {
        let _ = fs::remove_file(&staged);
        return Err(swap_err(format!(
            "moving the current executable aside failed: {e}"
        )));
    }
    if let Err(e) = fs::rename(&staged, current) {
        // Roll back so the user still has a working binary at the old path.
        let _ = fs::rename(&old, current);
        let _ = fs::remove_file(&staged);
        return Err(swap_err(format!(
            "moving the new executable into place failed: {e}"
        )));
    }

    Ok(match fs::remove_file(&old) {
        Ok(()) => None,
        Err(_) => Some(old),
    })
}

/// chmod 755 where the OS has an execute bit; a no-op elsewhere.
fn make_executable(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn extract_err(name: &str, detail: String) -> UpdateError {
    UpdateError::Extract {
        name: name.to_string(),
        detail,
    }
}

fn swap_err(detail: String) -> UpdateError {
    UpdateError::Swap { detail }
}

/// Test helper (also used by the self-update flow tests in [`super`]): a
/// gzipped tarball holding `pkg/<exe>` plus a decoy, like a real release
/// tarball with README/LICENSE alongside the binary.
#[cfg(test)]
pub(crate) fn tar_gz_with_exe(payload: &[u8]) -> Vec<u8> {
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    let mut builder = tar::Builder::new(encoder);
    for (name, bytes) in [
        ("pkg/README.md".to_string(), b"decoy".as_slice()),
        (format!("pkg/{}", exe_name()), payload),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        builder.append_data(&mut header, name, bytes).unwrap();
    }
    builder.into_inner().unwrap().finish().unwrap()
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    const PAYLOAD: &[u8] = b"pretend-new-binary";

    fn zip_with_exe(payload: &[u8]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("pkg/README.md", options).unwrap();
        writer.write_all(b"decoy").unwrap();
        writer
            .start_file(format!("pkg/{}", exe_name()), options)
            .unwrap();
        writer.write_all(payload).unwrap();
        writer.finish().unwrap().into_inner()
    }

    #[test]
    fn extracts_the_exe_from_a_tar_gz() {
        let dir = tempfile::tempdir().unwrap();
        let asset = dir.path().join("release.tar.gz");
        std::fs::write(&asset, tar_gz_with_exe(PAYLOAD)).unwrap();
        let out = extract_executable("release.tar.gz", &asset, dir.path()).unwrap();
        assert_eq!(std::fs::read(out).unwrap(), PAYLOAD);
    }

    #[test]
    fn extracts_the_exe_from_a_zip() {
        let dir = tempfile::tempdir().unwrap();
        let asset = dir.path().join("release.zip");
        std::fs::write(&asset, zip_with_exe(PAYLOAD)).unwrap();
        let out = extract_executable("release.zip", &asset, dir.path()).unwrap();
        assert_eq!(std::fs::read(out).unwrap(), PAYLOAD);
    }

    #[test]
    fn a_bare_binary_asset_passes_through() {
        let dir = tempfile::tempdir().unwrap();
        let asset = dir.path().join("onebrain-bare");
        std::fs::write(&asset, PAYLOAD).unwrap();
        let out = extract_executable("onebrain-bare", &asset, dir.path()).unwrap();
        assert_eq!(std::fs::read(out).unwrap(), PAYLOAD);
    }

    #[test]
    fn an_archive_without_the_exe_names_what_was_missing() {
        let dir = tempfile::tempdir().unwrap();
        let asset = dir.path().join("empty.tar.gz");
        let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(5);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "pkg/README.md", b"decoy".as_slice())
            .unwrap();
        std::fs::write(&asset, builder.into_inner().unwrap().finish().unwrap()).unwrap();

        let err = extract_executable("empty.tar.gz", &asset, dir.path()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(&exe_name()), "got: {msg}");
        assert!(msg.contains("empty.tar.gz"), "got: {msg}");
    }

    #[test]
    fn swap_replaces_the_binary_and_cleans_the_parked_copy() {
        // Nothing here is a running image, so even Windows can delete the
        // parked `.old` — the kept-path branch is exercised in production
        // only, by design (it depends on the OS lock on a live exe).
        let dir = tempfile::tempdir().unwrap();
        let current = dir.path().join(exe_name());
        std::fs::write(&current, b"old-binary").unwrap();
        let incoming = dir.path().join("incoming");
        std::fs::write(&incoming, PAYLOAD).unwrap();

        let kept = swap_executable(&incoming, &current).unwrap();
        assert_eq!(std::fs::read(&current).unwrap(), PAYLOAD);
        assert!(kept.is_none());
        assert!(!old_path(&current).exists());
        assert!(
            !dir.path().join(format!(".{}.new", exe_name())).exists(),
            "the staging file must not survive the swap"
        );
    }

    #[test]
    fn old_path_appends_the_suffix() {
        let parked = old_path(Path::new("C:/bin/onebrain.exe"));
        assert!(parked.to_string_lossy().ends_with("onebrain.exe.old"));
    }
}
