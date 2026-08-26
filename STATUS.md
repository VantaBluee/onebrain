# OneBrain — build status

Working file for resuming work across sessions. Update on every milestone
item completed. Spec: the build prompt (§8 milestones); decisions live in
`docs/decisions/`.

## Current: M0 — Skeleton & CI (in progress)

- [x] Workspace scaffolded per §3 (proto, mesh, engine, scheduler, models,
      api, dash, onebraind, onebrain-cli, xtask)
- [x] llama.cpp vendored as pinned submodule
      (`11cd98842874cc1b87ac274bd2d5cceb38102bb2`, upstream master 2026-08-26)
- [x] Engine embedding: CMake static build from build.rs + C shim
      (`crates/onebrain-engine/shim/`) — no bindgen, drift = compile error
- [x] Engine build-hash stamped (`OB_ENGINE_BUILD_ID`) for the handshake
- [x] Proto crate: envelope/handshake/capabilities/plan types + tests
- [x] GGUF header parser (v2/v3) with synthetic-file tests (onebrain-models)
- [x] CLI skeleton: `--version` (+ `--json`), `doctor` v0; all other
      commands name the milestone that brings them
- [x] Daemon crate: platform paths, config load/save (secure defaults,
      unknown keys rejected)
- [x] `cargo xtask dist` (stage binary + SHA256SUMS), `cargo xtask smoke`
      (tiny-GGUF download + engine smoke test)
- [x] Workspace builds clean on Windows host (debug + release)
- [x] Engine smoke test passes locally (`cargo xtask smoke`: stories260K
      GGUF, real CPU generation, greedy-determinism assertion)
- [x] `onebrain --version` / `onebrain doctor` verified on the staged
      release binary; `cargo xtask dist` stages binary + SHA256SUMS + licenses
- [x] fmt clean; clippy clean under `RUSTFLAGS=-Dwarnings`
- [x] CI workflows committed (.github/workflows/ci.yml + release.yml,
      dependabot.yml)
- [x] Docs: README, CHANGELOG, ADRs 0001–0003, dual licenses
- [x] Initial commits (5, conventional; note: the vendor submodule pin rode
      in the scaffolding commit)
- [ ] GitHub repository + first push + green CI matrix on all three OSes
      (**needs the user**: repo creation/authorization; the macOS/Linux legs
      of the DoD can only be proven by CI)

### M0 Definition of Done
Green Actions matrix (macos-14/15, ubuntu-22.04/24.04, windows-2022):
fmt, clippy, tests, CPU smoke inference with a tiny GGUF on all three OSes;
`cargo xtask dist` artifacts with `onebrain --version` working per OS.

## M1 — Excellent single-node (implementation complete; CI proof pending)

- [x] Engine: device enumeration, session reset, sampler chains, chat
      templating (with clean no-template fallback), streaming generate
- [x] Daemon: single-instance fs lock (kill -9 safe), API token, engine-host
      thread, internal control API (status/load/shutdown, always-auth'd)
- [x] OpenAI dialect (`/v1/*`): chat + text completions (SSE streaming),
      models; embeddings endpoint present (501 until later in M1.x)
- [x] Ollama dialect (`/api/*`): generate/chat (NDJSON streaming default),
      tags/show/ps/pull/version
- [x] Bearer auth everywhere; loopback exemption (configurable off) never
      applies to internal endpoints
- [x] Model logistics: embedded registry (URL-verified entries), resumable
      full-file downloads w/ BLAKE3 manifests, cache ls/rm
- [x] CLI: up/run/status/stop/pull/ls/rm wired; doctor v1 (devices, daemon
      state, config findings); hidden `__daemon`
- [x] `cargo xtask e2e` DoD rehearsal green on Windows: daemon up, load,
      OpenAI SSE → [DONE], Ollama NDJSON → done:true, kill -9 clean
      restart, graceful stop, lock release (11/11 steps)
- [x] Fixed on the way: Windows handle-inheritance leak in `onebrain up`
      (captured-output callers would hang forever; see up.rs)
- [ ] e2e green on macOS + Linux via CI (step added to the test job)
- [ ] Manual: unmodified OpenAI SDK script (scripts/check_openai_sdk.py)
      against a real model — run per OS before tagging
- [ ] /v1/embeddings implementation (deferred within M1; endpoint returns a
      clean 501 with remedy until then)

## Upcoming

- **M2 — Mesh & pairing** (iroh, pair codes, discovery, link prober)
- **M3 — Distributed inference v1** (pipeline split over authed mesh)
- M4 scheduler v1 · M5 resilience · M6 model logistics · M7 performance ·
  M8 product polish. Details in the spec.

## Dev environment notes (this machine: Windows 11)

- Toolchain bootstrapped 2026-08-26: rustc 1.98.0 (MSVC), VS 2022 Build
  Tools 17.14 (MSVC 14.44), portable CMake 4.4.3 + Ninja 1.13.2 in
  `%USERPROFILE%\dev-tools` (on user PATH).
- `CARGO_TARGET_DIR=C:\Users\0x9fa\.cargo-target` (user env var): the
  checkout lives in OneDrive; build artifacts must stay out of synced dirs.
- Gotcha: never pass `canonicalize()`d (`\\?\…`) paths to CMake/MSBuild.

## Manual hardware checklists (§9)

None run yet — first becomes relevant at M3 (Mac+Windows, Mac+Linux pairs).
