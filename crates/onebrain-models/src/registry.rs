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
///
/// The M6 fields are optional in the TOML (docs/logistics.md "Registry v1")
/// so pre-M6 entries — and third-party forks of the file — keep parsing:
/// absent means "dense single-file model, default context".
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryEntry {
    pub display_name: String,
    pub url: String,
    pub file_name: String,
    pub license: String,
    /// Minimum usable memory on ONE node to load solo at `recommended_ctx`.
    pub min_memory_mb: u64,
    /// Recommended context length in tokens (default 8192 when absent).
    #[serde(default = "default_recommended_ctx")]
    pub recommended_ctx: u32,
    /// Split-GGUF part count; 1 (the default) = a single-file model. For
    /// split entries `url`/`file_name` name the first part and the
    /// downloader derives the siblings (`crate::split`).
    #[serde(default = "default_parts")]
    pub parts: u32,
    /// MoE total parameter count; `None` = dense model.
    #[serde(default)]
    pub moe_total_params: Option<u64>,
    /// MoE parameters active per token; `None` = dense model.
    #[serde(default)]
    pub moe_active_params: Option<u64>,
    /// Minimum POOLED memory across the whole cluster to load at
    /// `recommended_ctx`; `None` for small models where solo == pooled is
    /// the only sensible reading.
    #[serde(default)]
    pub min_pooled_memory_mb: Option<u64>,
}

/// Spec §6 default: 8192 tokens serves chat well without ballooning the
/// KV cache; entries that need more (reasoning models) say so explicitly.
fn default_recommended_ctx() -> u32 {
    8192
}

fn default_parts() -> u32 {
    1
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
                let url = format!("{}/{org}/{repo}/resolve/main/{file}", hf_base_url());
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

/// Base URL `hf:` references resolve against. TEST-ONLY seam: the cluster
/// sim's zero-WAN proof (docs/logistics.md "DoD hooks") points
/// `OB_HF_BASE_URL` at a byte-counting local server standing in for
/// huggingface.co; real installs never set it. `$HF_TOKEN` stays safe
/// either way — `download::hf_bearer_for` sends it to huggingface.co
/// hosts only, regardless of what the resolved URL's host is.
fn hf_base_url() -> String {
    std::env::var("OB_HF_BASE_URL")
        .ok()
        .map(|v| v.trim().trim_end_matches('/').to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
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
        assert!(
            reg.len() >= 11,
            "expected the full v1 registry, got {}",
            reg.len()
        );
        for id in [
            "qwen3-0.6b",
            "qwen3-1.7b",
            "qwen3-4b",
            "qwen3-32b",
            "qwen3-30b-a3b",
            "glm-4.5-air",
            "gpt-oss-120b",
            "deepseek-r1-distill-qwen-7b",
            "deepseek-r1-distill-qwen-14b",
            "llama-3.3-70b-instruct",
            "tinystories-260k",
        ] {
            assert!(reg.contains_key(id), "missing curated entry {id}");
        }
        assert_eq!(reg["qwen3-0.6b"].license, "Apache-2.0");
    }

    /// Registry v1 structural invariants (docs/logistics.md): every entry —
    /// present and future — must satisfy these, so the test iterates the
    /// whole map instead of naming ids.
    #[test]
    fn every_entry_is_structurally_sound() {
        for (id, entry) in registry() {
            // Ids double as cache directory names, so they must be single
            // filesystem-safe path components (and BTreeMap keys are unique
            // by construction — TOML rejects duplicate tables at parse).
            assert!(!id.is_empty());
            assert_eq!(
                sanitize_component(id),
                *id,
                "id {id:?} is not filesystem-safe"
            );
            assert!(entry.url.starts_with("https://"), "{id}: non-https url");
            assert!(
                entry.url.ends_with(&entry.file_name),
                "{id}: url must end in file_name so the downloader's naming holds"
            );
            assert!(
                entry.file_name.to_ascii_lowercase().ends_with(".gguf"),
                "{id}: file_name must be a .gguf"
            );
            assert!(!entry.display_name.is_empty(), "{id}: empty display_name");
            assert!(!entry.license.is_empty(), "{id}: license must be recorded");
            assert!(entry.min_memory_mb > 0, "{id}: min_memory_mb");
            assert!(entry.recommended_ctx > 0, "{id}: recommended_ctx");
            assert!(entry.parts >= 1, "{id}: parts");
            // MoE fields come as a pair, active strictly below total.
            match (entry.moe_total_params, entry.moe_active_params) {
                (None, None) => {}
                (Some(total), Some(active)) => {
                    assert!(active < total, "{id}: active params must be < total");
                }
                _ => panic!("{id}: moe_total_params and moe_active_params must both be set"),
            }
            // Pooled memory can never be below the solo floor.
            if let Some(pooled) = entry.min_pooled_memory_mb {
                assert!(
                    pooled >= entry.min_memory_mb,
                    "{id}: min_pooled_memory_mb below min_memory_mb"
                );
            }
        }
    }

    #[test]
    fn split_entry_declares_parts_matching_its_file_name() {
        let reg = registry();
        let glm = &reg["glm-4.5-air"];
        assert!(glm.parts > 1, "glm-4.5-air must be a split entry");
        // The file name itself must parse as part 1 of `parts`, otherwise
        // the downloader cannot derive the sibling part URLs.
        let split = crate::split::parse_split_name(&glm.file_name)
            .expect("glm-4.5-air file_name must follow the -%05d-of-%05d.gguf convention");
        assert_eq!(split.index, 1, "registry entries must name the FIRST part");
        assert_eq!(split.count, glm.parts);
        // Every single-file entry must NOT look like a split part.
        for (id, entry) in reg {
            if entry.parts == 1 {
                assert!(
                    crate::split::parse_split_name(&entry.file_name).is_none(),
                    "{id}: single-file entry has a split-style file_name"
                );
            }
        }
    }

    #[test]
    fn moe_entries_carry_both_param_counts() {
        let reg = registry();
        for id in ["qwen3-30b-a3b", "glm-4.5-air", "gpt-oss-120b"] {
            let entry = &reg[id];
            assert!(entry.moe_total_params.is_some(), "{id}: moe_total_params");
            assert!(entry.moe_active_params.is_some(), "{id}: moe_active_params");
        }
        // And the curated big entries all state a pooled-memory floor.
        for id in [
            "qwen3-32b",
            "qwen3-30b-a3b",
            "glm-4.5-air",
            "gpt-oss-120b",
            "deepseek-r1-distill-qwen-7b",
            "deepseek-r1-distill-qwen-14b",
            "llama-3.3-70b-instruct",
        ] {
            assert!(
                reg[id].min_pooled_memory_mb.is_some(),
                "{id}: min_pooled_memory_mb"
            );
        }
    }

    #[test]
    fn entry_without_optional_fields_parses_with_defaults() {
        // Backward compatibility: a pre-M6 entry (no optional fields at
        // all) must keep parsing — absent means dense/single-file/8192.
        let toml = r#"
            ["old-style"]
            display_name = "Old Style"
            url = "https://example.invalid/old.gguf"
            file_name = "old.gguf"
            license = "MIT"
            min_memory_mb = 1024
        "#;
        let reg: BTreeMap<String, RegistryEntry> = toml::from_str(toml).unwrap();
        let entry = &reg["old-style"];
        assert_eq!(entry.recommended_ctx, 8192);
        assert_eq!(entry.parts, 1);
        assert_eq!(entry.moe_total_params, None);
        assert_eq!(entry.moe_active_params, None);
        assert_eq!(entry.min_pooled_memory_mb, None);
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
