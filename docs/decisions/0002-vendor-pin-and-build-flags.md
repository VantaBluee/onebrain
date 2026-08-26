# ADR 0002: llama.cpp vendor pin and engine build flags

Status: accepted (M0, 2026-08-26)

## Context

`vendor/llama.cpp` is a git submodule the engine builds from source. Upstream
llama.cpp does not cut stable release branches — master moves fast and the
project's own RPC docs require identical builds on every machine — so OneBrain
must decide exactly which commit it ships and which build flags produce
portable, distributable artifacts on macOS, Windows, and Linux.

## Decision

- **Pin to commit `11cd98842874cc1b87ac274bd2d5cceb38102bb2`** (upstream
  master as of 2026-08-26). Because upstream has no stable branches, we
  pin-and-bump deliberately: bumps are explicit commits that update the
  submodule, rerun the full CI matrix and smoke inference, and note anything
  the shim had to adapt to (compile-time drift detection, ADR 0001). The
  current pin is always recorded here.
- **`GGML_OPENMP=OFF`.** Avoids a runtime dependency on an OpenMP library,
  which varies by OS/toolchain and undermines the one-static-binary goal.
  Costs some CPU prefill throughput; revisit in the M7 performance program
  with measurements.
- **`GGML_NATIVE=OFF` by default; `OB_GGML_NATIVE=1` opt-in.** Default
  artifacts must run on machines other than the build host, so no
  `-march=native`. Developers benchmarking locally can opt in via the env var.
- **Engine always built Release, even under `cargo build` (dev profile).**
  Debug-built ggml is unusably slow (orders of magnitude), which would make
  every dev-loop smoke test misleading. The Rust side keeps normal dev/release
  profiles.
- **`LLAMA_BUILD_COMMON/TESTS/TOOLS/EXAMPLES/SERVER/APP=OFF`.** We link
  libraries only; upstream's CLI tools, server, tests, and examples are dead
  weight and extra build time, and shipping upstream's unauthenticated server
  would contradict the security posture.

## Consequences

- Reproducible engine bytes on every machine, which the build-hash handshake
  depends on; version skew between nodes is detected, never silently tolerated.
- We track upstream manually. Fixes and new model architectures arrive only
  when we bump the pin — an accepted cost for controlled breakage. Any local
  patches must be maintained as `.patch` files with an upstreaming note, never
  a fork.
- Default builds leave some single-machine CPU performance on the table
  (no `-march=native`, no OpenMP); both are explicit M7 revisit items.
- First builds are slower because the engine compiles at full optimization
  even for dev profiles.
