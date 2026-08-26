//! `onebrain rm <ref>`: delete one model from the local cache. Accepts the
//! same references as `pull`/`run` (registry id, `hf:` ref) plus raw cache
//! ids exactly as `onebrain ls` prints them.

use std::str::FromStr;

use onebrain_models::registry::{ModelRef, Resolved};
use onebraind::paths::AppPaths;

use super::CliError;

pub fn run(reference: &str, json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;

    // Map the reference to a cache id. Anything that doesn't parse as a
    // model reference may still be a literal cache id from `onebrain ls`
    // (e.g. `hf--Qwen--...`), so fall back to the raw string and let the
    // cache's own typed errors speak.
    let id = match ModelRef::from_str(reference) {
        Ok(ModelRef::Local(path)) => {
            return Err(CliError(format!(
                "{} is a local file, not a cached model; local models are loaded in place, \
                 so delete the file yourself if you want it gone",
                path.display()
            )));
        }
        Ok(model_ref) => match model_ref.resolve() {
            Ok(Resolved::Remote(spec)) => spec.cache_key,
            _ => reference.to_string(),
        },
        Err(_) => reference.to_string(),
    };

    onebrain_models::cache::remove(&paths.model_cache_dir(), &id)
        .map_err(|e| CliError(e.to_string()))?;

    if json {
        println!("{}", serde_json::json!({ "status": "removed", "id": id }));
    } else {
        println!("removed {id} from the model cache.");
    }
    Ok(())
}
