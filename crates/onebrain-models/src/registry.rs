//! The embedded model registry and model-reference parsing.
//!
//! `models.toml` (crate root) ships inside the binary via `include_str!` and
//! is parsed once at first use. A user-supplied model reference (per
//! `docs/internal-api.md` "Model references") is one of:
//!
//! - a registry id (`qwen3-0.6b`),
//! - `hf:<org>/<repo>/<file>.gguf` — direct Hugging Face fetch,
//! - a local path to a `.gguf` file (loaded in place, never copied).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::OnceLock;

use serde::Deserialize;

const EMBEDDED_REGISTRY: &str = include_str!("../models.toml");

/// One curated registry entry (see `models.toml` for field semantics).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryEntry {
    pub display_name: String,
    pub url: String,
    pub file_name: String,
    pub license: String,
    pub min_memory_mb: u64,
    pub ctx_recommended: u32,
}

/// The embedded registry, parsed on first access.
///
/// The TOML is compiled into the binary, so a parse failure is a packaging
/// defect, not a runtime condition — it panics (and a unit test parses the
/// file so CI catches it before any user can).
pub fn registry() -> &'static BTreeMap<String, RegistryEntry> {
    static REG: OnceLock<BTreeMap<String, RegistryEntry>> = OnceLock::new();
    REG.get_or_init(|| {
        toml::from_str(EMBEDDED_REGISTRY)
            .expect("embedded models.toml is malformed — rebuild from a clean checkout")
    })
}

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error(
        "unknown model {id:?}; available models: {available}. \
         For models outside the registry use `hf:<org>/<repo>/<file>.gguf` \
         or the path to a local .gguf file"
    )]
    UnknownId { id: String, available: String },
    #[error(
        "invalid Hugging Face reference {given:?}; \
         write it as hf:<org>/<repo>/<file>.gguf"
    )]
    BadHfRef { given: String },
}

/// A parsed model reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    /// An id from the embedded registry.
    Registry(String),
    /// `hf:<org>/<repo>/<file>` — fetched from huggingface.co.
    Hf {
        org: String,
        repo: String,
        file: String,
    },
    /// A local `.gguf` loaded in place (no download, no copy).
    Local(PathBuf),
}

impl FromStr for ModelRef {
    type Err = RegistryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(rest) = s.strip_prefix("hf:") {
            let mut parts = rest.splitn(3, '/');
            return match (parts.next(), parts.next(), parts.next()) {
                (Some(org), Some(repo), Some(file))
                    if !org.is_empty()
                        && !repo.is_empty()
                        && file.to_ascii_lowercase().ends_with(".gguf") =>
                {
                    Ok(ModelRef::Hf {
                        org: org.to_string(),
                        repo: repo.to_string(),
                        file: file.to_string(),
                    })
                }
                _ => Err(RegistryError::BadHfRef {
                    given: s.to_string(),
                }),
            };
        }
        let looks_like_path =
            (s.contains('/') || s.contains('\\')) && s.to_ascii_lowercase().ends_with(".gguf");
        if looks_like_path || Path::new(s).exists() {
            return Ok(ModelRef::Local(PathBuf::from(s)));
        }
        if registry().contains_key(s) {
            Ok(ModelRef::Registry(s.to_string()))
        } else {
            let available = registry()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            Err(RegistryError::UnknownId {
                id: s.to_string(),
                available,
            })
        }
    }
}

/// What the downloader needs for one remote model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadSpec {
    /// Cache directory name under `<data_dir>/models/` (a single path
    /// component, filesystem-safe).
    pub cache_key: String,
    pub url: String,
    pub file_name: String,
}

/// The result of resolving a [`ModelRef`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolved {
    /// Needs downloading (or is already cached under `cache_key`).
    Remote(DownloadSpec),
    /// Already on disk; skip the download pipeline entirely.
    Local(PathBuf),
}

impl ModelRef {
    /// Resolve to a concrete download spec (or a local path).
    pub fn resolve(&self) -> Result<Resolved, RegistryError> {
        match self {
            ModelRef::Registry(id) => {
                let entry = registry().get(id).ok_or_else(|| RegistryError::UnknownId {
                    id: id.clone(),
                    available: registry()
                        .keys()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                        .join(", "),
                })?;
                Ok(Resolved::Remote(DownloadSpec {
                    cache_key: id.clone(),
                    url: entry.url.clone(),
                    file_name: entry.file_name.clone(),
                }))
            }
            ModelRef::Hf { org, repo, file } => {
                let url = format!("https://huggingface.co/{org}/{repo}/resolve/main/{file}");
                // `file` may carry subdirectories (hf:org/repo/dir/f.gguf);
                // only the last component becomes the on-disk name.
                let file_name = file.rsplit('/').next().unwrap_or(file).to_string();
                let stem = file_name
                    .strip_suffix(".gguf")
                    .unwrap_or(&file_name)
                    .to_string();
                let cache_key = sanitize_component(&format!("hf--{org}--{repo}--{stem}"));
                Ok(Resolved::Remote(DownloadSpec {
                    cache_key,
                    url,
                    file_name,
                }))
            }
            ModelRef::Local(path) => Ok(Resolved::Local(path.clone())),
        }
    }
}

/// Make a string safe as a single directory name: ASCII alphanumerics,
/// `.`, `-` and `_` pass through; everything else becomes `-`.
fn sanitize_component(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_registry_parses_with_required_entries() {
        let reg = registry();
        assert!(reg.len() >= 4, "expected at least 4 registry entries");
        for id in ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "tinystories-260k"] {
            let entry = reg.get(id).unwrap_or_else(|| panic!("missing {id}"));
            assert!(entry.url.starts_with("https://"));
            assert!(entry.file_name.ends_with(".gguf"));
            assert!(entry.min_memory_mb > 0);
            assert!(entry.ctx_recommended > 0);
        }
        assert_eq!(reg["qwen3-0.6b"].license, "Apache-2.0");
    }

    #[test]
    fn registry_id_parses_and_resolves() {
        let r: ModelRef = "tinystories-260k".parse().unwrap();
        assert_eq!(r, ModelRef::Registry("tinystories-260k".to_string()));
        match r.resolve().unwrap() {
            Resolved::Remote(spec) => {
                assert_eq!(spec.cache_key, "tinystories-260k");
                assert_eq!(spec.file_name, "stories260K.gguf");
                assert_eq!(
                    spec.url,
                    "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
                );
            }
            other => panic!("expected Remote, got {other:?}"),
        }
    }

    #[test]
    fn hf_ref_parses_and_resolves() {
        let r: ModelRef = "hf:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
            .parse()
            .unwrap();
        assert_eq!(
            r,
            ModelRef::Hf {
                org: "Qwen".to_string(),
                repo: "Qwen3-0.6B-GGUF".to_string(),
                file: "Qwen3-0.6B-Q8_0.gguf".to_string(),
            }
        );
        match r.resolve().unwrap() {
            Resolved::Remote(spec) => {
                assert_eq!(
                    spec.url,
                    "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
                );
                assert_eq!(spec.file_name, "Qwen3-0.6B-Q8_0.gguf");
                assert_eq!(spec.cache_key, "hf--Qwen--Qwen3-0.6B-GGUF--Qwen3-0.6B-Q8_0");
                assert!(!spec.cache_key.contains('/'));
            }
            other => panic!("expected Remote, got {other:?}"),
        }
    }

    #[test]
    fn hf_ref_with_subdirectory_keeps_last_component_as_file_name() {
        let r: ModelRef = "hf:ggml-org/models/tinyllamas/stories260K.gguf"
            .parse()
            .unwrap();
        match r.resolve().unwrap() {
            Resolved::Remote(spec) => {
                assert_eq!(spec.file_name, "stories260K.gguf");
                assert_eq!(
                    spec.url,
                    "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
                );
            }
            other => panic!("expected Remote, got {other:?}"),
        }
    }

    #[test]
    fn bad_hf_refs_are_rejected() {
        for bad in [
            "hf:",
            "hf:onlyorg",
            "hf:org/repo",
            "hf:org/repo/notgguf.bin",
        ] {
            let err = bad.parse::<ModelRef>().unwrap_err();
            assert!(
                matches!(err, RegistryError::BadHfRef { .. }),
                "{bad} should be BadHfRef, got {err}"
            );
            assert!(err.to_string().contains("hf:<org>/<repo>/<file>.gguf"));
        }
    }

    #[test]
    fn existing_path_parses_as_local() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("weights.bin");
        std::fs::write(&file, b"x").unwrap();
        let s = file.to_string_lossy().into_owned();
        let r: ModelRef = s.parse().unwrap();
        assert_eq!(r, ModelRef::Local(PathBuf::from(&s)));
        assert_eq!(r.resolve().unwrap(), Resolved::Local(PathBuf::from(&s)));
    }

    #[test]
    fn nonexistent_gguf_path_with_separators_parses_as_local() {
        for s in [
            "C:\\models\\does-not-exist.gguf",
            "./somewhere/does-not-exist.GGUF",
            "/opt/models/does-not-exist.gguf",
        ] {
            let r: ModelRef = s.parse().unwrap();
            assert_eq!(r, ModelRef::Local(PathBuf::from(s)), "for input {s}");
        }
    }

    #[test]
    fn unknown_id_error_lists_available_ids() {
        let err = "no-such-model".parse::<ModelRef>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no-such-model"));
        assert!(msg.contains("tinystories-260k"), "message was: {msg}");
        assert!(msg.contains("qwen3-0.6b"), "message was: {msg}");
        assert!(msg.contains("hf:<org>/<repo>/<file>.gguf"));
    }
}
