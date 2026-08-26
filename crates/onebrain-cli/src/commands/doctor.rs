//! `onebrain doctor` v0: paths, engine identity, and what the compiled
//! engine supports. Grows GPU/driver/firewall diagnosis in M1 and self-
//! update in M8.

use serde::Serialize;

use onebraind::paths::AppPaths;

use super::CliError;

#[derive(Serialize)]
struct DoctorReport {
    product_version: &'static str,
    engine_build: String,
    llama_version: String,
    engine_system_info: String,
    config_dir: String,
    config_file: String,
    data_dir: String,
    model_cache_dir: String,
}

pub fn run(json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve().map_err(|e| CliError(e.to_string()))?;
    let report = DoctorReport {
        product_version: env!("CARGO_PKG_VERSION"),
        engine_build: onebrain_engine::engine_build_hash().0,
        llama_version: onebrain_engine::llama_version(),
        engine_system_info: onebrain_engine::system_info().trim().to_string(),
        config_dir: paths.config_dir.display().to_string(),
        config_file: paths.config_file().display().to_string(),
        data_dir: paths.data_dir.display().to_string(),
        model_cache_dir: paths.model_cache_dir().display().to_string(),
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serializes")
        );
    } else {
        println!(
            "{} {}",
            onebrain_proto::PRODUCT_NAME,
            report.product_version
        );
        println!();
        println!("engine build : {}", report.engine_build);
        println!("llama.cpp    : {}", report.llama_version);
        println!("capabilities : {}", report.engine_system_info);
        println!();
        println!("config dir   : {}", report.config_dir);
        println!("config file  : {}", report.config_file);
        println!("data dir     : {}", report.data_dir);
        println!("model cache  : {}", report.model_cache_dir);
    }
    Ok(())
}
