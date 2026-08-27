# OneBrain — build status

Working file for resuming work across sessions. Update on every milestone
item completed. Spec: the build prompt (§8 milestones); decisions live in
`docs/decisions/`.

## M0 — Skeleton & CI (DoD MET 2026-08-27: green matrix run 33045749556)

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
- [x] GitHub repository (github.com/VantaBluee/onebrain, public) + green
      CI matrix on all three OSes (run 33045749556, 2026-08-27)

### M0 Definition of Done
Green Actions matrix (macos-14/15, ubuntu-22.04/24.04, windows-2022):
fmt, clippy, tests, CPU smoke inference with a tiny GGUF on all three OSes;
`cargo xtask dist` artifacts with `onebrain --version` working per OS.

## M1 — Excellent single-node (DoD components proven in CI 2026-08-27)

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
- [x] e2e green on all three OSes via CI
- [ ] Manual: unmodified OpenAI SDK script (scripts/check_openai_sdk.py)
      against a real model — run per OS before tagging
- [ ] /v1/embeddings implementation (deferred within M1; endpoint returns a
      clean 501 with remedy until then)

## M2 — Mesh & pairing (CI proof met 2026-08-27, incl. netem leg)

- [x] Device identity: Ed25519 iroh key at `<config_dir>/device-key`,
      never regenerated silently, never leaves the machine
- [x] Pairing: 6-digit code through SPAKE2 (code never on the wire) with
      direction-bound key-confirmation MACs (reflection-proof both ways),
      120 s window, 3-attempt budget; ticket for cross-network, code-only
      via mDNS candidates on the LAN
- [x] Peer store (`peers.toml`), name dedup, `onebrain unpair`
- [x] Mesh service: accept-by-ALPN, unpaired connections closed (code 1)
      — the §10 guarantee, asserted by tests; Hello handshake with
      engine-build judgment; 2 s heartbeats (suspect/down per §5 timings);
      RTT EWMA + 4 MiB bandwidth probe per link
- [x] Internal API: pair/start (NDJSON window), pair/join, peers, unpair,
      status peers_summary; CLI `pair` (code + ticket + terminal QR),
      `unpair`, `status` peers table
- [x] `cargo xtask pair-sim`: two sandboxed daemons pair via ticket+code,
      both report connected with RTT/bandwidth, unpair degrades — 10/10
      steps green on Windows; netem variant (Linux namespaces, 1 Gbit /
      0.5 ms shaping) wired into CI
- [x] pair-sim green on all three OSes via CI; netem leg green
- [ ] Manual two-machine checklist (incl. real mDNS discovery + daemon
      restart survival): run before tagging
- [x] (resolved) Windows CI e2e SSE timeout — gone once the redundant
      inner rebuild was skipped; restart survival hardened in M3 with the
      reconnect loop

## M3 — Distributed inference v1 (CI proof met 2026-08-27: sim green on all three OSes + netem)

- [x] Engine substrate: vendored llama.cpp patched (additive ~84 lines,
      `patches/0001-rpc-serve-fd.patch`, upstreaming note) so GGML RPC
      sessions run over caller-owned sockets — no listener anywhere;
      GGML_RPC on, RDMA negotiation off
- [x] Engine surface: `ob_rpc_serve_fd` / `RemoteServer` registration /
      `Model::load_distributed` with explicit devices + tensor_split; the
      in-process loopback test proves distributed greedy tokens ==
      solo ground truth
- [x] Scheduler v1-lite: auto-solo short-circuit, memory-proportional
      contiguous ranges (largest-remainder, zero-layer drop), `--nodes`,
      `--explain` prose with binding-constraint naming
- [x] Mesh typed streams (`StreamHeader{kind, epoch}`), NodeStatus budgets,
      epoch fencing (close code 4), plan proposal/ack over control streams
- [x] Daemon orchestration: worker ServeShard (socketpair + serve thread
      per rpc stream), head accept-loop bridges (one mesh stream per RPC
      client connection — empirical fix over the contracted accept-once,
      recorded in docs + ADR 0004), role-correct teardown ordering
- [x] Restart resilience found + fixed en route (M2 DoD hardening):
      persisted peer addressing + reconnect loop with backoff; pinned
      `[mesh] bind_addr` config; integration-tested
- [x] `cargo xtask sim`: distribute (auto-engage) → socket scans →
      byte-identical §9 correctness vs solo → forced `--nodes 2`; wired
      into CI on all three OSes + the Linux netem leg
- [x] Sim green on macOS + Linux + Windows via CI (run 33045749556)
- [ ] Manual two-machine checklists (Mac+Windows, Mac+Linux) — documented,
      to run when hardware is available (recorded here per the M3 DoD)
- [ ] Shard-only weight fetch deferred to M6 by design (ADR 0004: GGML RPC
      is head-push; M6 pre-seeds worker tensor caches for transfer economy)

## M4 — Scheduler v1 (CI proof met 2026-08-27: run 33049383364 green)

- [x] Measured device profiles: engine microbench (prefill/decode medians),
      disk read, persisted profile.toml, shared via NodeStatus (proto v2)
- [x] KV budgeting from real GGUF metadata at the requested ctx — proven
      live in the sim: ctx 2048 auto-solos, ctx 16384 forces distribution,
      every assignment shrinks
- [x] Memory-and-compute scoring (asymmetric split within ±1 layer of
      prediction, sim-asserted with decode_tps_override 100/50)
- [x] Boundary-on-fastest-link stage ordering (exact ≤8, RTTs in --explain)
- [x] Additional-node rule (≥5% predicted gain or infeasible-without) —
      unit-tested inclusion/exclusion
- [x] `onebrain bench`: node + links report, internal endpoint, --json
- [x] CI green on all three OSes (run 33049383364)

## M5 — Resilience (implementation landed; integration gate in progress)

- [x] Vendor patch 0002: RPC client failures become error returns (18
      RPC_STATUS_ASSERT sites converted; dead-socket registry; documented
      residual aborts unreachable in our flows) — torn-bridge engine test
      proves Decode error + clean model free, no aborts
- [x] Proto v3: NodeStatus.draining; scheduler excludes draining nodes
      unless infeasible without them; mesh peer-events stream + Draining
      peer state
- [x] Daemon supervisor: one transparent retry via prefix re-prefill into
      the same client stream; death/drain epoch teardown; lazy rejoin
      re-plan; worker drain notice on stop (3 s grace)
- [x] Power: SleepInhibitor + BatteryProbe per OS (Windows/macOS/Linux),
      battery policy pure-tested, doctor + status surfacing, inhibitor
      watcher in the runtime
- [x] Chaos sim green locally (all 4 scenarios: kill mid-generation → same
      stream completes byte-identical to control; no-fallback typed error
      with node + MB figures, daemon stays healthy; rejoin → new epoch;
      drain → excluded from next plan). Confirm-before-send added to the
      engine loop en route: a token streams only after its own decode
      succeeds, so a dying node can never leak a corrupted token.
- [ ] Chaos sim green in CI (rides the M5 push)

## Upcoming
- M6 model logistics · M7 performance · M8 product polish. Details in the
  spec.

## Dev environment notes (this machine: Windows 11)

- Toolchain bootstrapped 2026-08-26: rustc 1.98.0 (MSVC), VS 2022 Build
  Tools 17.14 (MSVC 14.44), portable CMake 4.4.3 + Ninja 1.13.2 in
  `%USERPROFILE%\dev-tools` (on user PATH).
- `CARGO_TARGET_DIR=C:\Users\0x9fa\.cargo-target` (user env var): the
  checkout lives in OneDrive; build artifacts must stay out of synced dirs.
- Gotcha: never pass `canonicalize()`d (`\\?\…`) paths to CMake/MSBuild.

## Manual hardware checklists (§9)

None run yet — first becomes relevant at M3 (Mac+Windows, Mac+Linux pairs).
