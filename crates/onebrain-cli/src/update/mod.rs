//! `onebrain self-update` (docs/product.md §3): query the repo's GitHub
//! releases API, pick this platform's asset, verify it against the
//! release's SHA256SUMS (plus a cosign keyless check when a cosign binary
//! happens to be on PATH — optional, never required), and atomically swap
//! the current executable.
//!
//! Every network touchpoint goes through [`UpdateConfig::api_base`] so the
//! tests drive the complete flow against a local fixture server — no real
//! network, ever. The daemon-running refusal lives in the command layer
//! ([`crate::commands::self_update`]); this module assumes it may swap.

pub mod github;
pub mod swap;
pub mod version;

use std::path::{Path, PathBuf};
use std::time::Duration;

use github::{digest_for, find_named, select_asset, Release};
use version::Version;

/// Everything an update decision depends on, injectable so tests never
/// touch the real releases API, the real PATH, or the running binary.
pub struct UpdateConfig {
    /// API origin, `https://api.github.com` in production.
    pub api_base: String,
    /// `owner/repo` on GitHub.
    pub repo: String,
    /// Target triple used to pick the platform asset.
    pub triple: String,
    /// The version this binary was built as (compared against the tag).
    pub current_version: String,
    /// The executable to replace.
    pub current_exe: PathBuf,
    /// Install even when the release is older than `current_version`.
    pub allow_downgrade: bool,
    /// A cosign binary to verify SHA256SUMS with, when one exists.
    pub cosign: Option<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    #[error("failed to initialize the HTTP client ({0}); re-run, and report a bug if it persists")]
    Init(String),
    #[error("release query failed: {url} ({detail}); check network access and retry")]
    Api { url: String, detail: String },
    #[error("could not parse `{value}` as a version ({detail}); report a bug if this is a real release tag")]
    BadVersion { value: String, detail: String },
    #[error("release {tag} has no runnable asset for this platform ({triple}); install manually from the release page")]
    NoAsset { tag: String, triple: String },
    #[error("release {tag} publishes no SHA256SUMS; refusing an unverifiable update — install manually after checking the release page")]
    NoChecksums { tag: String },
    #[error("SHA256SUMS has no entry for {asset}; refusing — the release may still be uploading, retry in a minute")]
    ChecksumMissing { asset: String },
    #[error("checksum mismatch for {asset}: SHA256SUMS says {expected}, the download hashed to {actual}; refusing — retry, and report it if it persists")]
    ChecksumMismatch {
        asset: String,
        expected: String,
        actual: String,
    },
    #[error("cosign refused SHA256SUMS ({detail}); refusing the update — verify the release manually before trusting it")]
    Cosign { detail: String },
    #[error(
        "{tag} is older than the installed v{current}; pass --allow-downgrade to install it anyway"
    )]
    Downgrade { tag: String, current: String },
    #[error("could not unpack {name} ({detail}); retry, and install manually if it persists")]
    Extract { name: String, detail: String },
    #[error("could not swap the executable ({detail}); the previous binary is still in place")]
    Swap { detail: String },
    #[error("i/o during update ({0}); retry")]
    Io(String),
}

/// What `--check` learned. All comparison outcomes are surfaced (not just
/// "newer exists"): report-only must be honest about downgrades too.
#[derive(Debug)]
pub struct CheckOutcome {
    pub current: String,
    pub latest: String,
    pub update_available: bool,
    pub downgrade: bool,
    /// The asset an update would install, when one matches this platform.
    pub asset: Option<String>,
}

/// Result of an actual update run.
#[derive(Debug)]
pub enum UpdateOutcome {
    UpToDate { current: String },
    Installed(InstallOutcome),
}

#[derive(Debug)]
pub struct InstallOutcome {
    pub from: String,
    pub to: String,
    pub asset: String,
    pub exe: PathBuf,
    /// The parked previous binary, when the OS would not let it be deleted
    /// yet (Windows keeps the running image locked).
    pub old_kept: Option<PathBuf>,
    pub cosign: CosignStatus,
}

#[derive(Debug)]
pub enum CosignStatus {
    Verified,
    /// Skipped, with the honest reason — cosign is optional (§3), so a
    /// skip is reported, never fatal.
    Skipped(String),
}

/// Best-effort host target triple. `rustc -vV` is a build-time luxury and
/// build-script env is not forwarded to the binary, so the release
/// pipeline's triple vocabulary is reconstructed from the runtime consts;
/// the unit test pins the mapping for every platform we ship.
pub fn host_triple() -> String {
    triple_for(std::env::consts::OS, std::env::consts::ARCH)
}

fn triple_for(os: &str, arch: &str) -> String {
    match os {
        "windows" => format!("{arch}-pc-windows-msvc"),
        "macos" => format!("{arch}-apple-darwin"),
        "linux" => format!("{arch}-unknown-linux-gnu"),
        other => format!("{arch}-unknown-{other}"),
    }
}

/// A tiny `which`: locate `name` (plus the platform's executable suffix)
/// on PATH. Enough for the optional cosign probe.
pub fn find_in_path(name: &str) -> Option<PathBuf> {
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join(&file))
        .find(|candidate| candidate.is_file())
}

/// Report what an update would do, changing nothing (`--check`).
pub fn check(cfg: &UpdateConfig) -> Result<CheckOutcome, UpdateError> {
    let client = http_client()?;
    let release = fetch_release(&client, cfg)?;
    let latest = parse_version(&release.tag_name)?;
    let current = parse_version(&cfg.current_version)?;
    Ok(CheckOutcome {
        current: current.to_string(),
        latest: latest.to_string(),
        update_available: latest > current,
        downgrade: latest < current,
        asset: select_asset(&release.assets, &cfg.triple).map(|a| a.name.clone()),
    })
}

/// Run the full update: fetch, verify, swap. `note` receives progress
/// lines for humans; machine consumers read the returned outcome.
pub fn perform(
    cfg: &UpdateConfig,
    mut note: impl FnMut(&str),
) -> Result<UpdateOutcome, UpdateError> {
    // Finish a previous dance: a parked `.old` from the last update can
    // usually be deleted now that this process runs the new binary.
    let _ = std::fs::remove_file(swap::old_path(&cfg.current_exe));

    let client = http_client()?;
    let release = fetch_release(&client, cfg)?;
    let latest = parse_version(&release.tag_name)?;
    let current = parse_version(&cfg.current_version)?;
    if latest == current {
        return Ok(UpdateOutcome::UpToDate {
            current: current.to_string(),
        });
    }
    if latest < current {
        if !cfg.allow_downgrade {
            return Err(UpdateError::Downgrade {
                tag: release.tag_name.clone(),
                current: cfg.current_version.clone(),
            });
        }
        note(&format!(
            "downgrading to {} over v{} (--allow-downgrade)",
            release.tag_name, cfg.current_version
        ));
    }

    let asset = select_asset(&release.assets, &cfg.triple).ok_or_else(|| UpdateError::NoAsset {
        tag: release.tag_name.clone(),
        triple: cfg.triple.clone(),
    })?;
    let sums_asset =
        find_named(&release.assets, "SHA256SUMS").ok_or_else(|| UpdateError::NoChecksums {
            tag: release.tag_name.clone(),
        })?;

    let tmp = tempfile::tempdir().map_err(|e| UpdateError::Io(e.to_string()))?;
    note(&format!(
        "downloading {} ({})",
        asset.name,
        crate::commands::human_bytes(asset.size)
    ));
    let asset_path = tmp.path().join(&asset.name);
    download_to(&client, &asset.browser_download_url, &asset_path)?;

    let sums = fetch_text(&client, &sums_asset.browser_download_url)?;
    let expected = digest_for(&sums, &asset.name).ok_or_else(|| UpdateError::ChecksumMissing {
        asset: asset.name.clone(),
    })?;
    let actual = sha256_hex(&asset_path)?;
    if !expected.eq_ignore_ascii_case(&actual) {
        return Err(UpdateError::ChecksumMismatch {
            asset: asset.name.clone(),
            expected,
            actual,
        });
    }
    note("sha256: verified against SHA256SUMS");

    let cosign = verify_cosign(cfg, &client, &release, &sums, tmp.path())?;
    if matches!(cosign, CosignStatus::Verified) {
        note("cosign: SHA256SUMS signature verified");
    }

    let new_exe = swap::extract_executable(&asset.name, &asset_path, tmp.path())?;
    let old_kept = swap::swap_executable(&new_exe, &cfg.current_exe)?;

    Ok(UpdateOutcome::Installed(InstallOutcome {
        from: current.to_string(),
        to: latest.to_string(),
        asset: asset.name.clone(),
        exe: cfg.current_exe.clone(),
        old_kept,
        cosign,
    }))
}

fn parse_version(text: &str) -> Result<Version, UpdateError> {
    Version::parse(text).map_err(|detail| UpdateError::BadVersion {
        value: text.to_string(),
        detail,
    })
}

/// Verify the SHA256SUMS blob with cosign when everything needed exists:
/// a cosign binary, the detached `.sig`, and the signing certificate the
/// keyless flow published (docs/product.md §5). Any missing piece is a
/// reported skip; an actual verification FAILURE is fatal — a bad
/// signature must never be shrugged off.
fn verify_cosign(
    cfg: &UpdateConfig,
    client: &reqwest::blocking::Client,
    release: &Release,
    sums: &str,
    tmp: &Path,
) -> Result<CosignStatus, UpdateError> {
    let Some(cosign_bin) = &cfg.cosign else {
        return Ok(CosignStatus::Skipped(
            "no cosign binary on PATH (optional)".to_string(),
        ));
    };
    let Some(sig) = find_named(&release.assets, "SHA256SUMS.sig") else {
        return Ok(CosignStatus::Skipped(
            "the release publishes no SHA256SUMS.sig".to_string(),
        ));
    };
    let Some(cert) = ["SHA256SUMS.pem", "SHA256SUMS.cert", "SHA256SUMS.crt"]
        .iter()
        .find_map(|name| find_named(&release.assets, name))
    else {
        return Ok(CosignStatus::Skipped(
            "the release publishes no signing certificate".to_string(),
        ));
    };

    let sums_path = tmp.join("SHA256SUMS");
    std::fs::write(&sums_path, sums).map_err(|e| UpdateError::Io(e.to_string()))?;
    let sig_path = tmp.join(&sig.name);
    download_to(client, &sig.browser_download_url, &sig_path)?;
    let cert_path = tmp.join(&cert.name);
    download_to(client, &cert.browser_download_url, &cert_path)?;

    let args = cosign_args(&sums_path, &sig_path, &cert_path, &cfg.repo);
    let output = std::process::Command::new(cosign_bin)
        .args(&args)
        .output()
        .map_err(|e| UpdateError::Cosign {
            detail: format!("could not run cosign: {e}"),
        })?;
    if !output.status.success() {
        return Err(UpdateError::Cosign {
            detail: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(CosignStatus::Verified)
}

/// Arguments for keyless `cosign verify-blob`: release.yml signs with the
/// repo's GitHub Actions OIDC identity (docs/product.md §5), so the
/// certificate identity is pinned to workflows of this repo and the
/// issuer to GitHub Actions. Pure, so the test can pin the contract.
fn cosign_args(sums: &Path, sig: &Path, cert: &Path, repo: &str) -> Vec<String> {
    vec![
        "verify-blob".to_string(),
        "--signature".to_string(),
        sig.display().to_string(),
        "--certificate".to_string(),
        cert.display().to_string(),
        "--certificate-identity-regexp".to_string(),
        format!("^https://github.com/{repo}/"),
        "--certificate-oidc-issuer".to_string(),
        "https://token.actions.githubusercontent.com".to_string(),
        sums.display().to_string(),
    ]
}

fn http_client() -> Result<reqwest::blocking::Client, UpdateError> {
    // GitHub's API refuses requests without a User-Agent. No overall
    // timeout: asset downloads legitimately run for minutes on slow
    // links; per-request timeouts cover the small calls.
    reqwest::blocking::Client::builder()
        .user_agent(format!("onebrain/{}", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(10))
        .timeout(Option::<Duration>::None)
        .build()
        .map_err(|e| UpdateError::Init(e.to_string()))
}

fn fetch_release(
    client: &reqwest::blocking::Client,
    cfg: &UpdateConfig,
) -> Result<Release, UpdateError> {
    let url = format!(
        "{}/repos/{}/releases/latest",
        cfg.api_base.trim_end_matches('/'),
        cfg.repo
    );
    let resp = client
        .get(&url)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .timeout(Duration::from_secs(30))
        .send()
        .map_err(|e| UpdateError::Api {
            url: url.clone(),
            detail: e.to_string(),
        })?;
    let status = resp.status();
    if status.as_u16() == 404 {
        return Err(UpdateError::Api {
            url,
            detail: "HTTP 404 — no published releases yet (or the repository moved)".to_string(),
        });
    }
    if !status.is_success() {
        return Err(UpdateError::Api {
            url,
            detail: format!("HTTP {status}"),
        });
    }
    resp.json().map_err(|e| UpdateError::Api {
        url,
        detail: format!("unparseable response: {e}"),
    })
}

fn fetch_text(client: &reqwest::blocking::Client, url: &str) -> Result<String, UpdateError> {
    let resp = client
        .get(url)
        .timeout(Duration::from_secs(30))
        .send()
        .map_err(|e| UpdateError::Api {
            url: url.to_string(),
            detail: e.to_string(),
        })?;
    let status = resp.status();
    if !status.is_success() {
        return Err(UpdateError::Api {
            url: url.to_string(),
            detail: format!("HTTP {status}"),
        });
    }
    resp.text().map_err(|e| UpdateError::Api {
        url: url.to_string(),
        detail: e.to_string(),
    })
}

fn download_to(
    client: &reqwest::blocking::Client,
    url: &str,
    path: &Path,
) -> Result<(), UpdateError> {
    let mut resp = client
        .get(url)
        .timeout(Duration::from_secs(600))
        .send()
        .map_err(|e| UpdateError::Api {
            url: url.to_string(),
            detail: e.to_string(),
        })?;
    let status = resp.status();
    if !status.is_success() {
        return Err(UpdateError::Api {
            url: url.to_string(),
            detail: format!("HTTP {status}"),
        });
    }
    let mut file = std::fs::File::create(path).map_err(|e| UpdateError::Io(e.to_string()))?;
    resp.copy_to(&mut file).map_err(|e| UpdateError::Api {
        url: url.to_string(),
        detail: e.to_string(),
    })?;
    Ok(())
}

fn sha256_hex(path: &Path) -> Result<String, UpdateError> {
    use sha2::Digest;
    let mut file = std::fs::File::open(path).map_err(|e| UpdateError::Io(e.to_string()))?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher).map_err(|e| UpdateError::Io(e.to_string()))?;
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use axum::extract::{Path as AxPath, State};
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::get;
    use axum::Router;

    use super::*;

    const REPO: &str = "VantaBluee/onebrain";
    const NEW_BINARY: &[u8] = b"the-new-binary-payload";

    /// Serve a fixture GitHub releases API + asset store on a loopback
    /// port (mirrors the onebrain-models download tests). The release
    /// document lists every file with `browser_download_url`s pointing
    /// back at this server. Returns the base URL for `api_base`.
    fn start_fixture(tag: &str, files: Vec<(String, Vec<u8>)>) -> String {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let listener = runtime
            .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
            .unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());

        let assets_json: Vec<serde_json::Value> = files
            .iter()
            .map(|(name, bytes)| {
                serde_json::json!({
                    "name": name,
                    "browser_download_url": format!("{base}/assets/{name}"),
                    "size": bytes.len(),
                    "unknown_field": { "ignored": true },
                })
            })
            .collect();
        let release = serde_json::json!({
            "tag_name": tag,
            "prerelease": false,
            "assets": assets_json,
        })
        .to_string();

        let store: Arc<HashMap<String, Vec<u8>>> = Arc::new(files.into_iter().collect());
        let app = Router::new()
            .route(
                "/repos/VantaBluee/onebrain/releases/latest",
                get(move || {
                    let release = release.clone();
                    async move { release }
                }),
            )
            .route("/assets/{name}", get(serve_asset))
            .with_state(store);

        std::thread::spawn(move || {
            runtime.block_on(async move {
                axum::serve(listener, app).await.unwrap();
            });
        });
        base
    }

    async fn serve_asset(
        State(store): State<Arc<HashMap<String, Vec<u8>>>>,
        AxPath(name): AxPath<String>,
    ) -> axum::response::Response {
        match store.get(&name) {
            Some(bytes) => bytes.clone().into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        }
    }

    fn sha256_of(bytes: &[u8]) -> String {
        use sha2::Digest;
        hex::encode(sha2::Sha256::digest(bytes))
    }

    /// A config whose "current executable" is a plain file in `dir`
    /// holding `old-binary` — nothing running, everything observable.
    fn test_cfg(dir: &std::path::Path, base: &str, allow_downgrade: bool) -> UpdateConfig {
        let current_exe = dir.join(swap::exe_name());
        std::fs::write(&current_exe, b"old-binary").unwrap();
        UpdateConfig {
            api_base: base.to_string(),
            repo: REPO.to_string(),
            triple: host_triple(),
            current_version: "0.1.0".to_string(),
            current_exe,
            allow_downgrade,
            cosign: None,
        }
    }

    /// Release fixture: `[tarball asset for this host, SHA256SUMS]`.
    fn release_files(version: &str) -> (String, Vec<(String, Vec<u8>)>) {
        let asset_name = format!("onebrain-v{version}-{}.tar.gz", host_triple());
        let tarball = swap::tar_gz_with_exe(NEW_BINARY);
        let sums = format!("{}  {asset_name}\n", sha256_of(&tarball));
        (
            asset_name.clone(),
            vec![
                (asset_name, tarball),
                ("SHA256SUMS".to_string(), sums.into_bytes()),
            ],
        )
    }

    #[test]
    fn full_update_downloads_verifies_and_swaps() {
        let (asset_name, files) = release_files("9.9.9");
        let base = start_fixture("v9.9.9", files);
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_cfg(dir.path(), &base, false);

        let mut notes = Vec::new();
        let outcome = perform(&cfg, |line| notes.push(line.to_string())).unwrap();
        let UpdateOutcome::Installed(installed) = outcome else {
            panic!("expected an install");
        };
        assert_eq!(installed.from, "0.1.0");
        assert_eq!(installed.to, "9.9.9");
        assert_eq!(installed.asset, asset_name);
        assert!(matches!(installed.cosign, CosignStatus::Skipped(_)));
        // The binary on disk IS the new payload, and the dance left no
        // debris (nothing was a running image here, so `.old` deletes).
        assert_eq!(std::fs::read(&cfg.current_exe).unwrap(), NEW_BINARY);
        assert!(installed.old_kept.is_none());
        assert!(!swap::old_path(&cfg.current_exe).exists());
        assert!(
            notes.iter().any(|n| n.contains("sha256: verified")),
            "{notes:?}"
        );
    }

    #[test]
    fn checksum_mismatch_refuses_and_leaves_the_binary_alone() {
        let asset_name = format!("onebrain-v9.9.9-{}.tar.gz", host_triple());
        let tarball = swap::tar_gz_with_exe(NEW_BINARY);
        // SHA256SUMS lists a digest of DIFFERENT bytes.
        let sums = format!("{}  {asset_name}\n", sha256_of(b"not the tarball"));
        let base = start_fixture(
            "v9.9.9",
            vec![
                (asset_name, tarball),
                ("SHA256SUMS".to_string(), sums.into_bytes()),
            ],
        );
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_cfg(dir.path(), &base, false);

        let err = perform(&cfg, |_| {}).unwrap_err();
        assert!(
            matches!(err, UpdateError::ChecksumMismatch { .. }),
            "got {err}"
        );
        assert_eq!(std::fs::read(&cfg.current_exe).unwrap(), b"old-binary");
    }

    #[test]
    fn up_to_date_is_reported_without_downloading() {
        // No assets at all: reaching selection or download would error,
        // so success proves the early exit.
        let base = start_fixture("v0.1.0", Vec::new());
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_cfg(dir.path(), &base, false);
        let outcome = perform(&cfg, |_| {}).unwrap();
        assert!(matches!(outcome, UpdateOutcome::UpToDate { current } if current == "0.1.0"));
        assert_eq!(std::fs::read(&cfg.current_exe).unwrap(), b"old-binary");
    }

    #[test]
    fn downgrades_need_the_flag() {
        let (_, files) = release_files("0.0.1");
        let base = start_fixture("v0.0.1", files);
        let dir = tempfile::tempdir().unwrap();

        let cfg = test_cfg(dir.path(), &base, false);
        let err = perform(&cfg, |_| {}).unwrap_err();
        assert!(matches!(err, UpdateError::Downgrade { .. }), "got {err}");
        assert!(err.to_string().contains("--allow-downgrade"));
        assert_eq!(std::fs::read(&cfg.current_exe).unwrap(), b"old-binary");

        let cfg = test_cfg(dir.path(), &base, true);
        let outcome = perform(&cfg, |_| {}).unwrap();
        assert!(matches!(outcome, UpdateOutcome::Installed(_)));
        assert_eq!(std::fs::read(&cfg.current_exe).unwrap(), NEW_BINARY);
    }

    #[test]
    fn a_release_without_a_platform_asset_names_the_triple() {
        let base = start_fixture(
            "v9.9.9",
            vec![
                (
                    "onebrain-v9.9.9-armv7-unknown-fantasy.tar.gz".to_string(),
                    vec![1],
                ),
                ("SHA256SUMS".to_string(), b"junk".to_vec()),
            ],
        );
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_cfg(dir.path(), &base, false);
        let err = perform(&cfg, |_| {}).unwrap_err();
        assert!(matches!(err, UpdateError::NoAsset { .. }), "got {err}");
        assert!(err.to_string().contains(&host_triple()), "got {err}");
    }

    #[test]
    fn a_release_without_sha256sums_is_refused() {
        let asset_name = format!("onebrain-v9.9.9-{}.tar.gz", host_triple());
        let base = start_fixture(
            "v9.9.9",
            vec![(asset_name, swap::tar_gz_with_exe(NEW_BINARY))],
        );
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_cfg(dir.path(), &base, false);
        let err = perform(&cfg, |_| {}).unwrap_err();
        assert!(matches!(err, UpdateError::NoChecksums { .. }), "got {err}");
    }

    #[test]
    fn check_reports_both_directions_and_never_touches_the_binary() {
        let (asset_name, files) = release_files("9.9.9");
        let base = start_fixture("v9.9.9", files);
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_cfg(dir.path(), &base, false);

        let outcome = check(&cfg).unwrap();
        assert_eq!(outcome.current, "0.1.0");
        assert_eq!(outcome.latest, "9.9.9");
        assert!(outcome.update_available);
        assert!(!outcome.downgrade);
        assert_eq!(outcome.asset.as_deref(), Some(asset_name.as_str()));
        assert_eq!(std::fs::read(&cfg.current_exe).unwrap(), b"old-binary");

        let base = start_fixture("v0.0.1", Vec::new());
        let cfg = test_cfg(dir.path(), &base, false);
        let outcome = check(&cfg).unwrap();
        assert!(!outcome.update_available);
        assert!(outcome.downgrade);
        assert!(outcome.asset.is_none());
    }

    #[test]
    fn cosign_args_pin_the_keyless_verification_contract() {
        let args = cosign_args(
            Path::new("SHA256SUMS"),
            Path::new("SHA256SUMS.sig"),
            Path::new("SHA256SUMS.pem"),
            REPO,
        );
        assert_eq!(args[0], "verify-blob");
        assert!(args.contains(&"^https://github.com/VantaBluee/onebrain/".to_string()));
        assert!(args.contains(&"https://token.actions.githubusercontent.com".to_string()));
        // The blob itself is the final positional argument.
        assert_eq!(args.last().unwrap(), "SHA256SUMS");
    }

    #[test]
    fn host_triple_mapping_covers_shipped_platforms() {
        assert_eq!(triple_for("windows", "x86_64"), "x86_64-pc-windows-msvc");
        assert_eq!(triple_for("macos", "aarch64"), "aarch64-apple-darwin");
        assert_eq!(triple_for("macos", "x86_64"), "x86_64-apple-darwin");
        assert_eq!(triple_for("linux", "x86_64"), "x86_64-unknown-linux-gnu");
        assert_eq!(triple_for("linux", "aarch64"), "aarch64-unknown-linux-gnu");
    }

    #[test]
    fn find_in_path_misses_cleanly() {
        assert!(find_in_path("definitely-not-a-real-tool-name-onebrain-test").is_none());
    }

    #[test]
    fn bad_release_tags_are_a_typed_error() {
        let base = start_fixture("not-a-version", Vec::new());
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_cfg(dir.path(), &base, false);
        let err = perform(&cfg, |_| {}).unwrap_err();
        assert!(matches!(err, UpdateError::BadVersion { .. }), "got {err}");
    }

    #[test]
    fn test_cfg_paths_are_isolated() {
        // Guard against the helper accidentally touching the real binary.
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_cfg(dir.path(), "http://127.0.0.1:1", false);
        assert!(cfg.current_exe.starts_with(dir.path()));
        assert_ne!(std::env::current_exe().ok(), Some(cfg.current_exe.clone()));
    }
}
