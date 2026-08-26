# Changelog

All notable changes to OneBrain will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

M0 (skeleton & CI) complete locally; M1 (excellent single-node) implemented,
cross-OS CI proof in flight.

### Added — M2

- Mesh & pairing on iroh: per-device Ed25519 identities, `onebrain pair`
  with a 6-digit code (authenticated via SPAKE2 with direction-bound
  confirmation MACs — the code never crosses the wire), ticket + QR for
  cross-network pairing, mDNS candidate discovery on the LAN.
- Persistent peer store with human names; `onebrain unpair`; unpaired
  connection attempts to the mesh are rejected and test-asserted.
- Per-link telemetry: 2 s heartbeats (suspect at 3 missed, down at 10 s),
  RTT EWMA, and a bandwidth probe — surfaced in `onebrain status`'s new
  PEERS table and `/api/internal/peers`.
- `cargo xtask pair-sim`: two-daemon pairing rehearsal in CI on all three
  OSes, plus a Linux netem variant (network namespaces shaped to
  1 Gbit / 0.5 ms) asserting measured bandwidth and RTT land in sane bands.

### Added — M1

- The daemon: `onebrain up/status/stop`, single-instance file lock that a
  kill -9 releases cleanly, auto-generated API bearer token, engine-host
  thread, and an always-authenticated internal control API.
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`,
  `/v1/models`) with SSE streaming and usage accounting; `/v1/embeddings`
  responds 501 with a remedy until it lands.
- Ollama-compatible API (`/api/generate`, `/api/chat`, `/api/tags`,
  `/api/show`, `/api/ps`, `/api/pull`, `/api/version`) streaming NDJSON by
  default — point an unmodified Ollama client at the endpoint.
- Model logistics: embedded registry (Qwen3 0.6B/1.7B/4B + a tiny test
  model, URLs verified), resumable downloads with BLAKE3 manifests,
  `onebrain pull/ls/rm`.
- `onebrain run <model>`: ensures the daemon, downloads if needed, loads,
  and prints the endpoint + token + example calls.
- `onebrain doctor` v1: compute devices with memory, daemon state, config
  validity — every finding with a remedy.
- Engine: device enumeration, session reset, configurable sampler chains,
  chat-template rendering with an explicit fallback path.
- `cargo xtask e2e`: the milestone rehearsal (sandboxed daemon, both
  dialects streaming, kill -9 recovery, graceful stop) — also runs in CI
  on all three OSes.

### Fixed

- `onebrain up` on Windows no longer leaks its caller's pipe handles into
  the detached daemon (captured-output callers would hang until the daemon
  exited).

### Added — M0

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
