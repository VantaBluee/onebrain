# Changelog

All notable changes to OneBrain will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

M0 (skeleton & CI) complete locally; M1 (excellent single-node) implemented,
cross-OS CI proof in flight.

### Added — M7

- Distributed prefill overlaps: an additive vendor patch teaches the
  GGML RPC client async submission (pending-ack ledger; the server's
  serial in-order processing is the correctness argument, so the wire
  format is untouched), which finally lets llama.cpp's pipeline-parallel
  scheduler engage over RPC — chunks of a long prompt now compute on one
  node while another node's chunk is in flight (sim DoD: ≥25% faster
  than the sequential path on the 1 Gbit profile, asserted in CI's
  netem leg).
- Requests no longer queue single-file: up to `max_concurrent_requests`
  generations share one unified KV cache with per-step token batching,
  admission control, and an honest 429 (with remedy) when the queue is
  full. A stalled client only stalls itself. Status queries answer
  instantly during generation.
- A repeated prompt prefix skips its own prefill: completed requests
  retain their KV, and a new request decodes only the divergent suffix
  (byte-identical output, greedy-proven), which makes chat-style
  system-prompt reuse nearly free.
- `--speculative`: a small draft model on the head proposes K tokens and
  the (possibly sharded) target verifies them in one batch — greedy
  output is byte-identical to non-speculative (sim-proven solo and
  distributed), and node loss mid-verify recovers and keeps
  speculating.
- Scheduler v2-lite searches candidate splits (memory/compute tilts,
  slow-node underweighting) with a bandwidth-aware transfer term and
  MoE-aware model dims; `--explain` reports the candidates and why the
  winner won.
- `onebrain bench --cluster`: every paired machine's microbench over
  the mesh plus timed end-to-end runs — as configured, vs the
  pre-overlap baseline, vs solo — in one reproducible markdown table.
- Real timing everywhere: the Ollama dialect now returns genuine
  `*_duration` fields, and every generation logs prefill/decode/TTFT.

### Added — M6

- Weights now move at range granularity: the model cache stores
  tensor-aligned, per-range-BLAKE3-verified ranges, so a worker assigned
  layers L..R fetches only the file header plus those layers' tensors
  (resumable mid-range; a full file on disk implicitly answers every
  range read with no duplication, and re-plans never re-download bytes
  already present).
- Paired machines share weights LAN-first over iroh-blobs riding the
  existing authenticated endpoint (paired peers only, no new sockets):
  before any WAN byte, a downloader asks every connected peer for its
  range inventory (`RangeQuery`/`RangeInventory`, proto v4) and fetches
  what a peer holds as bao-verified blobs — sim-proven: the second
  machine's pull moves ZERO new WAN bytes.
- Distributed loads stop re-pushing big weights (ADR 0004 payoff):
  workers pre-seed the RPC tensor cache from their on-disk ranges at plan
  adoption, so the head's push skips every tensor over the 10 MiB hash
  threshold on reloads; the pre-seed dir is LRU-capped
  (`rpc_cache_max_bytes`, default 20 GiB) and never evicts the active
  epoch's tensors.
- Split GGUFs (`-00001-of-000NN.gguf`) download, cache, and load as one
  model (`llama_model_load_from_splits` via the shim); `onebrain ls`
  shows one row with a parts count.
- Cache management: `onebrain pin`/`unpin` protect models from the new
  LRU GC (`cache_max_bytes`, default off) that runs after downloads and
  never evicts pinned or loaded models; `ls` shows pin state and age.
- Registry v1: eight curated, URL-verified entries from 2.5 GB
  (Qwen3-4B) to 73.5 GB (GLM-4.5-Air, split; MoE metadata recorded), with
  pooled-memory floors and recommended contexts; `HF_TOKEN` is sent to
  huggingface.co (only) when set.

### Fixed — M6

- A pre-existing peer-store race (M2 era): saving `peers.toml` deleted
  the old file before renaming the replacement in, so a concurrent read
  in that window saw an empty store (a just-paired daemon could
  transiently report zero peers); saves now rename atomically and
  read-modify-write cycles are serialized.

### Added — M5

- A lost node no longer kills anything: RPC transport failures became
  error returns (vendor patch `0002`, with a dead-socket registry), the
  daemon fails the in-flight request internally, re-plans without the dead
  node, and transparently retries once by re-prefilling the prompt plus
  everything already generated — the client's stream simply continues, and
  greedy output stays byte-identical to an uninterrupted run
  (chaos-sim-proven). When the survivors can't hold the model, the stream
  ends with a typed error naming the lost node and both memory figures.
- The engine only streams a token after its own decode succeeds
  (confirm-before-send), so a dying node can never leak a corrupted token
  into the output or the retry prefix.
- Node lifecycle: peers rejoining trigger a lazy re-plan to a new epoch;
  `onebrain stop` on a worker sends a polite drain notice the head honors.
- Laptop realities: sleep is inhibited while a model is active or a shard
  is being served (per-OS implementations), and a battery discharging
  below the configured threshold advertises "draining" — such nodes join
  new plans only when nothing fits without them.

### Added — M4

- Scheduler v1: placement now budgets KV cache from the model's real GGUF
  metadata at the requested context length (16k contexts visibly shrink
  per-node layer counts, and can force distribution a 2k context avoids),
  weighs nodes by measured compute (decode tok/s) on top of memory, and
  orders pipeline stages to put boundaries on the fastest links.
- Device profiles: `onebrain bench` measures prefill/decode throughput on
  a tiny model, disk read speed, and every paired link's RTT/bandwidth —
  one-page report, persisted, and shared with peers for planning.
- Plans report a predicted relative cost and `--explain` now names KV
  budgets, per-boundary RTTs, and why each node was included or excluded.
- Wire protocol v2 (profile fields in NodeStatus; same-build clusters only,
  enforced by the engine build-hash handshake since M2).

### Added — M3

- Distributed inference v1: pipeline layer-split across paired nodes with
  GGML RPC tunneled through the authenticated mesh — no TCP listener is
  ever exposed (socket-scan-asserted); a minimal additive vendor patch
  (`patches/0001`) serves RPC sessions over caller-owned sockets.
- Auto-solo placement: models that fit one node never distribute; `onebrain
  run --nodes N` forces a split and `--explain` prints why the plan looks
  the way it does, node by node.
- Plan epochs end-to-end: proposals, acks, fencing of stale epochs, and
  clean teardown; workers preempt local models to serve shards.
- Correctness proof in CI: distributed greedy generation is byte-identical
  to single-node output on the same prompt (`cargo xtask sim`, all OSes,
  plus a netem-shaped Linux leg).
- Paired daemons now reconnect after restarts: peer addresses persist in
  the store and a backoff reconnect loop re-forms links without re-pairing;
  `[mesh] bind_addr` can pin the mesh port.

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
