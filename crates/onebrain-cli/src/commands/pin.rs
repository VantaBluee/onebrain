//! `onebrain pin <model>` / `onebrain unpin <model>`: set or clear the
//! cache pin flag via `POST /api/internal/models/{pin,unpin}`
//! (docs/logistics.md "LRU GC + pinning"). Pinned models are never
//! LRU-evicted. The daemon owns the cache manifests while it runs — its GC
//! races anyone else writing them — so these verbs go through it rather
//! than mutating files behind its back, same rule as `onebrain unpair`.
//!
//! Accepts the same references as `pull`/`rm` (registry id, `hf:` ref) plus
//! raw cache ids exactly as `onebrain ls` prints them.

use std::str::FromStr;

use onebrain_models::registry::{ModelRef, Resolved};
use onebraind::paths::AppPaths;

use super::CliError;
use crate::client::{ClientError, DaemonClient};

pub fn run(reference: &str, pin: bool, json: bool) -> Result<(), CliError> {
    let id = cache_id(reference, pin)?;

    let paths = AppPaths::resolve()?;
    let client = DaemonClient::from_paths(&paths).map_err(not_running)?;
    if let Err(e) = client.status() {
        return Err(not_running(e));
    }

    match client.set_model_pin(&id, pin) {
        Ok(_) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": if pin { "pinned" } else { "unpinned" },
                        "id": id,
                    })
                );
            } else if pin {
                println!("pinned {id}; the cache GC will never evict it (unpin to undo).");
            } else {
                println!("unpinned {id}; the cache GC may evict it again when space runs low.");
            }
            Ok(())
        }
        // Strip the "HTTP nnn" wrapper: the daemon's message already reads
        // well on its own (unknown models point at `onebrain ls`).
        Err(ClientError::Api { message, .. }) if !message.trim().is_empty() => {
            Err(CliError(message))
        }
        Err(e) => Err(e.into()),
    }
}

/// Map a user reference to the cache id the daemon's pin endpoints key on —
/// the same fallback chain as `onebrain rm`: parse as a model reference,
/// else treat the raw string as a literal cache id from `onebrain ls`.
fn cache_id(reference: &str, pin: bool) -> Result<String, CliError> {
    let verb = if pin { "pin" } else { "unpin" };
    match ModelRef::from_str(reference) {
        Ok(ModelRef::Local(path)) => Err(CliError(format!(
            "{} is a local file, not a cached model; local models are loaded in place and \
             never evicted, so there is nothing to {verb}",
            path.display()
        ))),
        Ok(model_ref) => match model_ref.resolve() {
            Ok(Resolved::Remote(spec)) => Ok(spec.cache_key),
            _ => Ok(reference.to_string()),
        },
        Err(_) => Ok(reference.to_string()),
    }
}

/// Pinning needs the daemon (it owns the cache manifests while running);
/// explain how to get one rather than mutating files behind its back.
fn not_running(e: ClientError) -> CliError {
    CliError(format!(
        "daemon not running; run `onebrain up`, then retry ({e})"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_id_maps_to_itself() {
        assert_eq!(
            cache_id("tinystories-260k", true).unwrap(),
            "tinystories-260k"
        );
    }

    #[test]
    fn hf_ref_maps_to_its_cache_key() {
        assert_eq!(
            cache_id("hf:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", true).unwrap(),
            "hf--Qwen--Qwen3-0.6B-GGUF--Qwen3-0.6B-Q8_0"
        );
    }

    #[test]
    fn raw_cache_id_passes_through() {
        // Not a registry id, not an hf: ref, not a path — exactly what
        // `onebrain ls` prints for an hf download.
        assert_eq!(
            cache_id("hf--Qwen--Qwen3-0.6B-GGUF--Qwen3-0.6B-Q8_0", false).unwrap(),
            "hf--Qwen--Qwen3-0.6B-GGUF--Qwen3-0.6B-Q8_0"
        );
    }

    #[test]
    fn local_path_is_refused_with_the_verb_in_the_message() {
        let err = cache_id("C:\\models\\weights.gguf", false).unwrap_err();
        assert!(err.to_string().contains("nothing to unpin"), "{err}");
    }
}
