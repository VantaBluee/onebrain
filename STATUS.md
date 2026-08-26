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
- [ ] Workspace builds clean on Windows host (in progress — first full build
      running; llama.cpp path fix for MSBuild `\\?\` issue applied)
- [ ] Engine smoke test passes locally (`cargo xtask smoke`)
- [ ] fmt/clippy clean
- [ ] CI workflows committed (.github/) — being authored
- [ ] Docs: README, CHANGELOG, ADRs 0001–0003, dual licenses — being authored
- [ ] Initial commit + GitHub repository + green CI matrix on all three OSes
      (**blocked on a GitHub repo** — needs the user to create/authorize one;
      everything else is local)

### M0 Definition of Done
Green Actions matrix (macos-14/15, ubuntu-22.04/24.04, windows-2022):
fmt, clippy, tests, CPU smoke inference with a tiny GGUF on all three OSes;
`cargo xtask dist` artifacts with `onebrain --version` working per OS.

## Upcoming

- **M1 — Excellent single-node**: daemon up/stop, backend autodetection,
  OpenAI + Ollama streaming APIs with bearer auth, registry + resumable
  downloads, doctor v1.
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
