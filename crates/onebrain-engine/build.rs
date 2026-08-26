//! Builds the vendored llama.cpp (static libs) plus the C shim, and stamps
//! the engine build hash (llama.cpp commit + backend feature set + target)
//! that nodes exchange at handshake.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // Note: no canonicalize() here — on Windows it yields \\?\-prefixed
    // extended paths that MSBuild's compile steps cannot open.
    let vendor: PathBuf = manifest_dir
        .join("../../vendor/llama.cpp")
        .components()
        .filter(|c| !matches!(c, std::path::Component::CurDir))
        .fold(PathBuf::new(), |mut acc, c| {
            if matches!(c, std::path::Component::ParentDir)
                && acc.components().count() > 1
                && !matches!(
                    acc.components().next_back(),
                    Some(std::path::Component::ParentDir)
                )
            {
                acc.pop();
            } else {
                acc.push(c);
            }
            acc
        });

    if !vendor.join("CMakeLists.txt").exists() {
        panic!(
            "vendored llama.cpp not found at {}.\n\
             Run: git submodule update --init --recursive",
            vendor.display()
        );
    }

    apply_vendor_patches(&manifest_dir, &vendor);

    println!("cargo:rerun-if-changed=shim/ob_shim.c");
    println!("cargo:rerun-if-changed=shim/ob_shim.h");
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir
            .join("../../patches/0001-rpc-serve-fd.patch")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor.join("include/llama.h").display()
    );
    println!("cargo:rerun-if-env-changed=OB_GGML_NATIVE");
    println!("cargo:rerun-if-env-changed=OB_CUDA_ARCHS");

    let features = enabled_backends();

    // ---- 1. llama.cpp static libraries via CMake ----
    let mut cfg = cmake::Config::new(&vendor);
    cfg.profile("Release") // Debug ggml is unusably slow; engine is always optimized
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_IS_DEV", "OFF")
        // ggml's own threadpool keeps linking portable; revisit in the M7
        // performance program if profiling shows it matters.
        .define("GGML_OPENMP", "OFF")
        // The RPC backend is the distributed-inference engine substrate
        // (M3); its socket is never exposed — sessions run over caller-owned
        // fds bridged to authenticated mesh streams (patches/0001).
        .define("GGML_RPC", "ON")
        // RDMA auto-negotiation probes fds with getsockname and drags in
        // ibverbs; meaningless over bridged sockets.
        .define("GGML_RPC_RDMA", "OFF");

    // -march=native binaries can't be shipped; opt in for local perf runs.
    let native = env::var("OB_GGML_NATIVE")
        .map(|v| v == "1")
        .unwrap_or(false);
    cfg.define("GGML_NATIVE", if native { "ON" } else { "OFF" });

    // llama.cpp enables Metal by default on Apple; our backend set is
    // driven only by cargo features (build-hash determinism), so force it
    // off unless requested — otherwise CPU-only mac builds compile Metal
    // objects that build.rs then doesn't link frameworks for.
    if !features.iter().any(|f| f == "metal") {
        cfg.define("GGML_METAL", "OFF");
    }

    for f in &features {
        match f.as_str() {
            "metal" => {
                cfg.define("GGML_METAL", "ON");
                cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
            }
            "cuda" => {
                cfg.define("GGML_CUDA", "ON");
                // Full multi-arch CUDA builds take >90 min on CI runners;
                // compile-proof jobs constrain to one architecture. Unset =
                // llama.cpp's default (broad) arch list for real builds.
                if let Ok(archs) = env::var("OB_CUDA_ARCHS") {
                    cfg.define("CMAKE_CUDA_ARCHITECTURES", archs);
                }
            }
            "vulkan" => {
                cfg.define("GGML_VULKAN", "ON");
            }
            "rocm" => {
                cfg.define("GGML_HIP", "ON");
            }
            _ => {}
        }
    }

    let dst = cfg.build();

    // ---- 2. the C shim, compiled against the vendored headers ----
    cc::Build::new()
        .file(manifest_dir.join("shim/ob_shim.c"))
        .include(vendor.join("include"))
        .include(vendor.join("ggml/include"))
        .compile("ob_shim");

    // ---- 3. linking ----
    for libdir in ["lib", "lib64"] {
        let p = dst.join(libdir);
        if p.exists() {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }

    // Enumerate what the vendor build actually produced instead of hardcoding
    // the backend lib set. Order matters for GNU ld: dependents first —
    // llama → ggml (registry) → backend libs → ggml-base.
    let produced = installed_static_libs(&dst);
    let mut ordered: Vec<String> = Vec::new();
    for name in ["llama", "ggml"] {
        if produced.iter().any(|l| l == name) {
            ordered.push(name.to_string());
        }
    }
    for lib in &produced {
        if lib != "llama" && lib != "ggml" && lib != "ggml-base" {
            ordered.push(lib.clone());
        }
    }
    if produced.iter().any(|l| l == "ggml-base") {
        ordered.push("ggml-base".to_string());
    }
    for lib in &ordered {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" => {
            // Frameworks must go through rustc-link-lib (framework kind):
            // unlike rustc-link-lib, rustc-link-arg from a library's build
            // script does NOT propagate to the dependent binary's link.
            println!("cargo:rustc-link-lib=c++");
            for fw in ["Foundation", "Accelerate"] {
                println!("cargo:rustc-link-lib=framework={fw}");
            }
            if features.iter().any(|f| f == "metal") {
                for fw in ["Metal", "MetalKit"] {
                    println!("cargo:rustc-link-lib=framework={fw}");
                }
            }
        }
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            if features.iter().any(|f| f == "vulkan") {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
        "windows" => {
            // MSVC objects carry /DEFAULTLIB directives for the C++ runtime;
            // only non-default deps need listing. ggml-cpu reads CPU info
            // from the registry -> advapi32.
            println!("cargo:rustc-link-lib=advapi32");
            if features.iter().any(|f| f == "vulkan") {
                println!("cargo:rustc-link-lib=vulkan-1");
            }
        }
        _ => {}
    }

    // ---- 4. engine build hash ----
    let commit = git_head(&vendor).unwrap_or_else(|| "unknown".to_string());
    let short = &commit[..commit.len().min(12)];
    let mut feat = features.clone();
    feat.insert(0, "cpu".to_string());
    let target = env::var("TARGET").unwrap_or_default();
    println!("cargo:rustc-env=OB_LLAMA_COMMIT={commit}");
    println!(
        "cargo:rustc-env=OB_ENGINE_BUILD_ID=llama.cpp-{short}+{}+{target}",
        feat.join(",")
    );
}

/// Apply the maintained vendor patches (patches/*.patch) to the submodule
/// working tree when not already applied. Idempotence is checked by the
/// presence of the patched symbol, not git state, so a locally pre-patched
/// tree (or a re-run) is a no-op.
fn apply_vendor_patches(manifest_dir: &Path, vendor: &Path) {
    let header = vendor.join("ggml/include/ggml-rpc.h");
    let already = std::fs::read_to_string(&header)
        .map(|s| s.contains("ggml_backend_rpc_serve_fd"))
        .unwrap_or(false);
    if already {
        return;
    }
    let patch = manifest_dir.join("../../patches/0001-rpc-serve-fd.patch");
    let status = Command::new("git")
        .arg("-C")
        .arg(vendor)
        .arg("apply")
        .arg(&patch)
        .status();
    match status {
        Ok(s) if s.success() => {}
        other => panic!(
            "failed to apply vendor patch {} ({other:?}).\n\
             The vendored llama.cpp tree may be dirty; run \
             `git -C vendor/llama.cpp checkout -- .` and rebuild.",
            patch.display()
        ),
    }
}

fn enabled_backends() -> Vec<String> {
    ["metal", "cuda", "vulkan", "rocm"]
        .iter()
        .filter(|f| env::var(format!("CARGO_FEATURE_{}", f.to_uppercase())).is_ok())
        .map(|f| f.to_string())
        .collect()
}

fn installed_static_libs(dst: &Path) -> Vec<String> {
    let mut libs = Vec::new();
    for libdir in ["lib", "lib64"] {
        let dir = dst.join(libdir);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let stem = name
                .strip_suffix(".lib")
                .or_else(|| name.strip_suffix(".a").and_then(|s| s.strip_prefix("lib")));
            if let Some(stem) = stem {
                if stem == "llama" || stem.starts_with("ggml") {
                    libs.push(stem.to_string());
                }
            }
        }
    }
    libs.sort();
    libs.dedup();
    libs
}

fn git_head(vendor: &Path) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(vendor)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim().to_string();
    (!s.is_empty()).then_some(s)
}
