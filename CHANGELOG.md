# Changelog

All notable changes to OneBrain will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

M0 (skeleton & CI) in progress.

### Added

- Rust workspace (edition 2021, rust-version 1.80) with crates
  `onebrain-{proto,mesh,engine,scheduler,models,api,dash}`, `onebraind`,
  `onebrain-cli` (binary name `onebrain`), and `xtask`.
- Vendored llama.cpp as a pinned git submodule
  (`11cd98842874cc1b87ac274bd2d5cceb38102bb2`), built from `build.rs` via CMake
  as static Release libraries, with a minimal C shim
  (`crates/onebrain-engine/shim/ob_shim.c`) instead of bindgen. Backend cargo
  features: `metal`, `cuda`, `vulkan`, `rocm`; CPU always on.
- GGUF header parser in `onebrain-models`.
- Versioned wire-protocol types in `onebrain-proto` (postcard encoding):
  handshake, capability flags, placement-plan and message types.
- CLI skeleton: `onebrain --version` and `onebrain doctor`.
- `cargo xtask dist`: release build staged into
  `dist/onebrain-v<version>-<target>/` with `SHA256SUMS`.
- `cargo xtask smoke`: downloads a tiny GGUF and runs the engine smoke test
  (`smoke_generate_greedy`, self-skips without `OB_SMOKE_MODEL`).
- CI: cross-OS GitHub Actions matrix with fmt/clippy/tests under
  `RUSTFLAGS=-Dwarnings`.
- Dual license: MIT OR Apache-2.0.
