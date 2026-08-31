//! GitHub releases API shapes (parsed tolerantly), platform asset
//! selection, and SHA256SUMS parsing. Everything here is pure so the tests
//! never touch the network; the HTTP calls live in [`super`].

use serde::Deserialize;

/// One release, as `GET /repos/{owner}/{repo}/releases/latest` returns it.
/// Unknown fields are ignored and missing ones default: the API is
/// additive and this consumer must keep working against any revision.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct Release {
    pub tag_name: String,
    pub assets: Vec<Asset>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
    pub size: u64,
}

/// Suffixes that are never the executable payload: detached signatures,
/// checksums, and the OS package formats a package manager (not
/// self-update) owns.
const NON_PAYLOAD_SUFFIXES: &[&str] = &[
    ".sha256", ".sig", ".pem", ".cert", ".crt", ".asc", ".msi", ".deb", ".rpm", ".txt", ".json",
    ".sbom",
];

/// Pick the release asset self-update should install on `triple`.
///
/// Candidates must be named for this platform (contain the target triple,
/// or `universal-apple-darwin` for macOS builds shipped fat) and start
/// with `onebrain` (the staging name `cargo xtask dist` produces). Among
/// candidates the least-work shape wins — see [`payload_class`].
pub fn select_asset<'a>(assets: &'a [Asset], triple: &str) -> Option<&'a Asset> {
    let mut best: Option<(u8, &Asset)> = None;
    for asset in assets {
        let name = asset.name.as_str();
        let for_this_platform = name.contains(triple)
            || (triple.ends_with("apple-darwin") && name.contains("universal-apple-darwin"));
        if !name.starts_with("onebrain") || !for_this_platform {
            continue;
        }
        if NON_PAYLOAD_SUFFIXES.iter().any(|s| name.ends_with(s)) {
            continue;
        }
        let Some(class) = payload_class(name, triple) else {
            continue;
        };
        if best.is_none_or(|(b, _)| class < b) {
            best = Some((class, asset));
        }
    }
    best.map(|(_, asset)| asset)
}

/// Rank runnable payload shapes: a bare executable (0) needs no extraction
/// at all, tarballs (1) are the documented release shape (docs/product.md
/// §4/§5), zips (2) are what the M0 CI stub produced. `None` means "not a
/// shape self-update can install".
fn payload_class(name: &str, triple: &str) -> Option<u8> {
    if name.ends_with(".exe") {
        return Some(0);
    }
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        return Some(1);
    }
    if name.ends_with(".zip") {
        return Some(2);
    }
    // A unix binary published bare has nothing after the platform part —
    // an extension check would trip on the dots in `v0.1.0`.
    if name.ends_with(triple) || name.ends_with("universal-apple-darwin") {
        return Some(0);
    }
    None
}

/// Find an asset by exact name (`SHA256SUMS`, `SHA256SUMS.sig`, ...).
pub fn find_named<'a>(assets: &'a [Asset], name: &str) -> Option<&'a Asset> {
    assets.iter().find(|a| a.name == name)
}

/// Find the hex digest recorded for `asset` in a SHA256SUMS body.
///
/// Lines are sha256sum's `<64-hex>  <name>`; a leading `*` marks binary
/// mode and names may carry a staging-directory prefix (the M0 xtask wrote
/// per-folder sums), so matching is by trailing file name. Asset names
/// never contain whitespace, which keeps the split honest.
pub fn digest_for(sums: &str, asset: &str) -> Option<String> {
    for line in sums.lines() {
        let mut parts = line.split_whitespace();
        let (Some(digest), Some(name)) = (parts.next(), parts.next()) else {
            continue;
        };
        if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
            continue;
        }
        let name = name.trim_start_matches('*');
        let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
        if file == asset {
            return Some(digest.to_ascii_lowercase());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIPLE: &str = "x86_64-pc-windows-msvc";

    fn asset(name: &str) -> Asset {
        Asset {
            name: name.to_string(),
            browser_download_url: format!("http://127.0.0.1:1/assets/{name}"),
            size: 1,
        }
    }

    #[test]
    fn release_parse_tolerates_unknown_and_missing_fields() {
        // Real API responses carry dozens of fields we never model; a
        // minimal response must parse too (missing fields default).
        let release: Release = serde_json::from_str(
            r#"{
                "tag_name": "v0.2.0",
                "prerelease": false,
                "html_url": "https://example.invalid",
                "assets": [
                    { "name": "SHA256SUMS", "browser_download_url": "u", "id": 7,
                      "uploader": { "login": "x" } }
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(release.tag_name, "v0.2.0");
        assert_eq!(release.assets.len(), 1);
        assert_eq!(release.assets[0].size, 0);

        let empty: Release = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.tag_name, "");
        assert!(empty.assets.is_empty());
    }

    #[test]
    fn select_prefers_bare_exe_then_tarball_then_zip() {
        let zip = asset("onebrain-v0.2.0-x86_64-pc-windows-msvc.zip");
        let tarball = asset("onebrain-v0.2.0-x86_64-pc-windows-msvc.tar.gz");
        let exe = asset("onebrain-v0.2.0-x86_64-pc-windows-msvc.exe");

        let all = vec![zip.clone(), tarball.clone(), exe.clone()];
        assert_eq!(select_asset(&all, TRIPLE).unwrap().name, exe.name);

        let no_exe = vec![zip.clone(), tarball.clone()];
        assert_eq!(select_asset(&no_exe, TRIPLE).unwrap().name, tarball.name);

        let only_zip = vec![zip.clone()];
        assert_eq!(select_asset(&only_zip, TRIPLE).unwrap().name, zip.name);
    }

    #[test]
    fn select_skips_signatures_checksums_and_installers() {
        let assets = vec![
            asset("SHA256SUMS"),
            asset("onebrain-v0.2.0-x86_64-pc-windows-msvc.zip.sig"),
            asset("onebrain-v0.2.0-x86_64-pc-windows-msvc.msi"),
            asset("onebrain-v0.2.0-aarch64-apple-darwin.tar.gz"),
        ];
        assert!(select_asset(&assets, TRIPLE).is_none());
    }

    #[test]
    fn select_accepts_a_bare_unix_binary_and_universal_macos_builds() {
        let bare = vec![asset("onebrain-v0.2.0-x86_64-unknown-linux-gnu")];
        assert!(select_asset(&bare, "x86_64-unknown-linux-gnu").is_some());

        let universal = vec![asset("onebrain-v0.2.0-universal-apple-darwin.tar.gz")];
        assert!(select_asset(&universal, "aarch64-apple-darwin").is_some());
        assert!(
            select_asset(&universal, TRIPLE).is_none(),
            "universal builds are a darwin fallback only"
        );
    }

    #[test]
    fn digest_for_matches_plain_starred_and_pathed_names() {
        let digest = "a".repeat(64);
        let sums = format!(
            "{digest}  one.tar.gz\n{}  *two.zip\n{}  dist/pkg/three.exe\nnot a sums line\n",
            "b".repeat(64),
            "c".repeat(64),
        );
        assert_eq!(digest_for(&sums, "one.tar.gz").as_deref(), Some(&*digest));
        assert_eq!(
            digest_for(&sums, "two.zip").as_deref(),
            Some(&*"b".repeat(64))
        );
        assert_eq!(
            digest_for(&sums, "three.exe").as_deref(),
            Some(&*"c".repeat(64))
        );
        assert!(digest_for(&sums, "absent.tar.gz").is_none());
        // A malformed digest column never matches anything.
        assert!(digest_for("zz  one.tar.gz", "one.tar.gz").is_none());
    }
}
