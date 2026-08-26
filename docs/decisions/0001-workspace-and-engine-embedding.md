# ADR 0001: One workspace, embedded engine via C shim, postcard wire protocol, per-session roles

Status: accepted (M0, 2026-08-26)

## Context

OneBrain ships one static-ish binary per OS with no Python, no Docker, and no
system package dependencies beyond GPU drivers. It must embed llama.cpp (the
only runtime proven on Metal + CUDA + Vulkan + ROCm + CPU across macOS,
Windows, and Linux), speak a versioned inter-node protocol from day one, and
let the same install act as head or worker depending on the cluster session.

## Decision

- **One Rust workspace, one product binary.** All functionality lives in
  `crates/onebrain-*` plus the `onebraind` daemon and the thin `onebrain-cli`
  front end. Head and worker are per-session roles of the same binary, not
  separate installs or build variants.
- **Engine embedded as vendored static libraries.** `onebrain-engine/build.rs`
  builds `vendor/llama.cpp` via CMake into static libs linked into our binary.
  No dynamic llama.cpp dependency, no separate engine process for users to
  manage or expose.
- **FFI through a minimal hand-written C shim** (`shim/ob_shim.c`, compiled
  with the `cc` crate) plus hand-maintained `extern "C"` declarations, instead
  of bindgen. The shim references the upstream headers directly, so if a
  pinned-vendor bump changes a signature or struct the shim fails to *compile*
  — drift is caught at build time rather than at runtime — and we avoid a
  libclang/bindgen toolchain dependency on every contributor machine and CI
  runner. The shim exposes only the narrow surface OneBrain needs.
- **postcard for the wire protocol** (`onebrain-proto`). A compact,
  no-alloc-friendly, pure-Rust serde format: no protoc/codegen step, small
  messages on the latency-sensitive path. Every message carries a protocol
  version field and capability bits; compatibility across versions is enforced
  by the handshake (see the build-hash handshake in the spec), not by silent
  best-effort decoding.

## Consequences

- Users get one binary; contributors get one `cargo build` (plus CMake and a
  C/C++ compiler for the vendored engine).
- The shim surface must be extended by hand as engine features land — a
  deliberate, reviewable cost that replaces an unbounded generated API.
- postcard is Rust-only; that is acceptable because both ends of the wire are
  OneBrain. If a non-Rust client ever needs the internode protocol (none is
  planned), this ADR must be revisited.
- Compile times include a full llama.cpp CMake build on first compile; see
  ADR 0002 for the caching/profile choices that keep this tolerable.
