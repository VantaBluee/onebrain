//! `onebrain --version`: product, engine build id, vendored llama.cpp.

use serde::Serialize;

#[derive(Serialize)]
struct VersionInfo {
    product: &'static str,
    version: &'static str,
    engine_build: String,
    llama_commit: &'static str,
    proto_version: u16,
}

pub fn run(json: bool) {
    let info = VersionInfo {
        product: onebrain_proto::PRODUCT_NAME,
        version: env!("CARGO_PKG_VERSION"),
        engine_build: onebrain_engine::engine_build_hash().0,
        llama_commit: onebrain_engine::llama_commit(),
        proto_version: onebrain_proto::PROTO_VERSION,
    };
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&info).expect("serializes")
        );
    } else {
        println!("{} {}", info.product, info.version);
        println!("engine: {}", info.engine_build);
        println!("llama.cpp: {}", info.llama_commit);
        println!("protocol: v{}", info.proto_version);
    }
}
