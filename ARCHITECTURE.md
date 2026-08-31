# OneBrain architecture

A guided tour of how OneBrain works and where the code lives. The
per-milestone contracts under `docs/` are *binding* — when this overview and
a contract disagree, the contract wins:
[internal-api](docs/internal-api.md) (M1), [mesh](docs/mesh.md) (M2),
[distributed](docs/distributed.md) (M3), [scheduler-v1](docs/scheduler-v1.md)
(M4), [resilience](docs/resilience.md) (M5), [logistics](docs/logistics.md)
(M6), [perf](docs/perf.md) (M7), [product](docs/product.md) (M8).
Decisions with rejected alternatives live in `docs/decisions/` (ADRs).

## The shape of the system

One Rust workspace, one product binary. The same install acts as **head**
(API gateway, scheduler, dashboard) or **worker** (shard executor) per
cluster session — roles are chosen when a model loads, not at install time
(ADR [0001](docs/decisions/0001-workspace-and-engine-embedding.md)).

```
onebrain (CLI) ──HTTP──▶ onebraind (daemon)
                            ├─ onebrain-api      OpenAI /v1/* + Ollama /api/* (bearer auth)
                            ├─ internal router   /api/internal/* (always token-auth'd)
                            ├─ onebrain-dash     dashboard SPA at / (embedded assets)
                            ├─ engine host       one OS thread owning the llama.cpp model/session
                            ├─ supervisor        concurrent jobs, retry lifecycle, epochs
                            ├─ onebrain-scheduler placement plans from measured profiles
                            └─ onebrain-mesh     iroh QUIC: pairing, heartbeats, RPC tunnels, blobs
                                    │
                            (authenticated QUIC, no raw TCP)
                                    │
                         peer daemons (same binary, worker role)
```

## Crate map

| Crate | What it owns | Start reading |
|---|---|---|
| `onebrain-cli` | The `onebrain` binary; every verb is a `commands/*.rs` module; self-update lives in `update/` | `crates/onebrain-cli/src/main.rs` |
| `onebraind` | Daemon runtime: paths, config, single-instance lock, engine host, supervisor, cluster/epoch logic, power policy, request log, advisor | `crates/onebraind/src/runtime.rs` |
| `onebrain-api` | axum routers for both public dialects + bearer auth; dialect-agnostic `backend` trait the daemon implements | `crates/onebrain-api/src/lib.rs` |
| `onebrain-dash` | The dashboard: static assets embedded via rust-embed, served by `router()` | `crates/onebrain-dash/src/lib.rs`, `assets/` |
| `onebrain-mesh` | iroh endpoint: identity, pairing, peer store, typed streams, heartbeats/probing, peer events, blobs provider | `crates/onebrain-mesh/src/lib.rs` (the module doc is the map) |
| `onebrain-engine` | llama.cpp FFI through a hand-written C shim; device enumeration; sessions/batches; RPC serve + client registration | `crates/onebrain-engine/src/lib.rs`, `shim/` |
| `onebrain-scheduler` | Device/link profiles, placement search, KV budgeting, plan types | `crates/onebrain-scheduler/src/` |
| `onebrain-models` | Registry (`models.toml`), GGUF header parsing, range cache, downloads | `crates/onebrain-models/src/` |
| `onebrain-proto` | postcard wire types: envelope, handshake, pair, plan; `PROTO_VERSION` + capability bits | `crates/onebrain-proto/src/` |
| `xtask` | `cargo xtask dist / smoke / e2e / pair-sim / sim` — the CI rehearsals | `xtask/src/main.rs` |

## Wire protocol

`onebrain-proto` defines every inter-node message as postcard-serialized
Rust types (ADR 0001: no protoc, no codegen; both ends of the wire are
OneBrain). Every envelope carries `PROTO_VERSION` (currently 5) and
capability bits. Compatibility is enforced, never guessed: the `Hello`
exchange at mesh connect judges the protocol version **and the engine build
hash** (a stamp over the vendored llama.cpp build, `OB_ENGINE_BUILD_ID`) and
closes incompatible connections with a remedy in the log. Identical builds
imply identical GGML RPC protocol, which pre-empts upstream's
silent-version-skew failure mode entirely.

## Mesh and pairing (M2 — docs/mesh.md)

`onebrain-mesh` owns the only network surface: one iroh QUIC endpoint
(Ed25519 device identity generated at first start, stored at
`<config_dir>/device-key`, never leaves the machine). Three ALPNs:

- `onebrain/pair/1` — accepted from anyone, but **only while a pairing
  window is open**. Pairing is SPAKE2 (symmetric PAKE over the 6-digit
  code) with direction-bound key-confirmation MACs; the code never crosses
  the wire; the host budget is 3 failed attempts per 120 s window. Tickets
  (serialized endpoint addresses) let the joiner dial across networks;
  code-only joins discover candidates via mDNS on the LAN.
- `onebrain/mesh/1` — all paired traffic. On accept, the remote endpoint id
  must be in the peer store (`<config_dir>/peers.toml`), else the
  connection closes with code 1 (`unpaired`). This is the §10 guarantee,
  asserted by integration tests.
- The iroh-blobs ALPN (M6) — same endpoint, same paired-only accept rule.

Every mesh bi-stream starts with a `StreamHeader { kind, epoch }` frame:
`control` (Hello, heartbeats, plan traffic, range/bench queries), `rpc`
(one tunneled GGML RPC session), `probe`. Heartbeats every 2 s drive peer
states (`Connected` → 3 missed = `Suspect` → 10 s silent = `Down`), plus an
RTT EWMA and a 4 MiB bandwidth probe per link. A reconnect loop redials
stored peers from persisted addressing, so clusters survive restarts.
Code: `crates/onebrain-mesh/src/service.rs` (the service task),
`pairing.rs`, `store.rs`, `blobs.rs`.

## Engine embedding and the vendor-patch regime

`vendor/llama.cpp` is a pinned submodule built by
`crates/onebrain-engine/build.rs` via CMake into static libs (pin and build
flags: ADR [0002](docs/decisions/0002-vendor-pin-and-build-flags.md)). FFI
goes through a minimal hand-written C shim (`crates/onebrain-engine/shim/`)
— no bindgen, so upstream drift breaks the *compile*, not the runtime.

OneBrain never forks llama.cpp. Three additive patches under `patches/`
(each with an upstreaming note in [patches/README.md](patches/README.md),
applied idempotently by `build.rs`):

1. **0001-rpc-serve-fd** — serve one GGML RPC session over a caller-owned
   socket, so workers need **no listener at all**.
2. **0002-rpc-client-error-returns** — client-side transport failures become
   error returns instead of process aborts (the M5 resilience enabler),
   with a dead-socket registry so torn streams fail fast.
3. **0003-rpc-client-async-pipeline** — client-side async submission
   (pending-ack ledger; the server's serial in-order processing is the
   correctness argument, so the wire format is untouched), which flips the
   RPC device's `async`/`events` caps and lets llama.cpp's own
   pipeline-parallel scheduler engage over RPC (the M7 prefill overlap).

## Distributed inference: RPC over QUIC (M3 — ADR 0004)

GGML RPC sessions are tunneled through authenticated mesh streams; **no raw
TCP listener exists anywhere**, verified by a socket-scan test on every CI
run. Worker side: each accepted `rpc` stream gets a connected local socket
pair, one end served by `ob_rpc_serve_fd` on a dedicated thread, the other
pumped 1:1 to the mesh stream. Head side: a per-epoch loopback-only
listener bridges the RPC client's connections into fresh mesh streams (an
accept-*loop*, amended from accept-once after the client empirically dials
~6 times per load — see ADR
[0004](docs/decisions/0004-rpc-transport-and-weight-flow.md) and
docs/distributed.md).

Weight flow is **head-push** (ADR 0004 decision 2): the head mmaps the
GGUF and pushes remote tensors over the tunnel at load; M6's tensor-cache
pre-seeding then makes re-pushes free (below). Code:
`crates/onebraind/src/cluster.rs` and the engine's `rpc` module.

## Plans and epochs

A load computes a plan: `Solo` when `weights + KV` fit the head's usable
memory (auto-solo, spec §1.4), else `PipelineParallel` with contiguous
layer ranges per node. The head sends `PlanProposal` to each participant,
collects `PlanAck`, and the plan becomes the active **epoch** — a
monotonically increasing fence: streams and ops for epochs ≠ active are
rejected (close code 4). Teardown is role-asymmetric (head frees the model
first, then closes bridges; workers' serve sessions end with their
streams). `--nodes N` forces a node count; `--explain` renders the
reasoning as prose.

## Scheduler (M4 v1 → M7 v2-lite)

Placement inputs are **measured, never assumed**
(`crates/onebrain-scheduler`):

- per-node profiles: usable memory (measured free minus OS reserve — never
  total RAM), prefill/decode tok/s and disk MB/s from a ~10 s microbench
  (`onebrain bench`), persisted in `profile.toml` and shared via
  `NodeStatus`;
- per-link RTT EWMA + probed bandwidth from the mesh.

KV is budgeted from real GGUF metadata at the requested context (per-layer
K+V bytes; weights per layer from actual tensor ranges), replacing any flat
ceiling. Layer shares are proportional to memory capacity tilted by
measured decode throughput; stage order minimizes summed boundary RTTs
(exact search ≤ 8 nodes, head pinned last so sampling stays local). An
extra node joins only when the plan is infeasible without it or predicted
time-per-token improves ≥ 5% — and the explanation says why either way.
M7's v2-lite adds a candidate search (tilt family × stage orders) with a
bandwidth-aware transfer term; every candidate considered appears in
`--explain`.

## Resilience (M5 — docs/resilience.md)

Failure lifecycle on the head: mesh peer events or a decode error mark the
epoch failed → the in-flight request is *not* surfaced yet → tear down,
re-plan from live peers → **one transparent retry**: re-prefill
prompt + already-generated tokens (greedy determinism makes this exact) and
keep streaming into the *same* client response — already-sent pieces are
never re-sent. No feasible re-plan ⇒ a typed error naming the lost node and
both MB figures. A returning peer triggers a lazy re-plan at idle. Workers
drain politely on `onebrain stop` (a `Draining` notice + 3 s grace).

The engine loop is confirm-before-send: a token is streamed only after its
own decode succeeded, so a dying node can never leak a corrupted token.
Power realities live in `crates/onebraind/src/power.rs`: a per-OS sleep
inhibitor held while serving, and a battery probe that flags a node
`draining` below the configured threshold so new plans avoid it.

## Model logistics (M6 — docs/logistics.md)

The model cache (`crates/onebrain-models`) is range-level and
content-addressed: tensor-aligned ranges with per-range BLAKE3, resumable
HTTP-Range fetch, and full files answering range reads by offset. Paired
machines share ranges LAN-first over iroh-blobs on the existing endpoint:
before any WAN byte, a downloader asks every connected peer
(`RangeQuery`/`RangeInventory`), so a model pulled once exists for the
whole cluster (sim-proven: the second node's pull moves zero new WAN
bytes). Workers fetch only the header plus their assigned layers at plan
adoption and pre-seed the RPC tensor cache (`<data_dir>/rpc-cache/`,
FNV-named per the RPC protocol), so the head's weight push skips every
≥ 10 MiB tensor it already holds. Split GGUFs load as one model; an LRU GC
with `onebrain pin`/`unpin` keeps the cache bounded when configured.

## Performance program (M7 — docs/perf.md)

Measurement lands before optimization; every lever ships with its
instrument and an off-switch that reconstructs the pre-M7 baseline
(`[perf]` in `config.toml` — see `crates/onebraind/src/config.rs`):

- timing everywhere: prefill/decode/TTFT per generation, real Ollama
  duration fields, one grep-stable `perf:` log line;
- overlapped chunked prefill (patch 0003; `prefill_overlap`);
- cross-request KV prefix reuse — a shared prefix skips its own prefill,
  byte-identical to cold (`kv_reuse`);
- speculative decoding (`onebrain run <model> --speculative [--draft <ref>]`):
  a draft model on the head proposes K tokens, the target verifies them in
  one batch; greedy output is byte-identical to non-speculative,
  sim-asserted solo and distributed;
- micro-batched decode: up to `max_concurrent_requests` generations share
  one unified-KV session with admission control and a typed 429;
- `onebrain bench --cluster`: peer microbenches over the mesh plus a timed
  end-to-end run vs the constructed baseline vs solo.

## Dashboard, metrics, advisor (M8 — docs/product.md)

`GET /api/internal/metrics` (token-auth'd, additive-stable JSON) is the one
document the dashboard polls: node, peers (with per-link RTT/bandwidth and
Hello-derived versions), the active plan, a 50-entry request ring (counts
and timings — **never prompt text**), and server-side advisor findings
(pure functions in `crates/onebraind/src/advisor.rs`, each backed by a
measurement: slow link, memory-starved node, draining node in plan,
version/engine skew, solo-because-infeasible, battery-draining worker).

The dashboard itself (`crates/onebrain-dash`) is a hand-written static page
— semantic HTML + vanilla JS + CSS, embedded via rust-embed, no framework,
no build step, no CDN, no external fonts (ADR
[0005](docs/decisions/0005-dashboard-no-framework.md)). The daemon serves
it at `/`; the shell is the only Bearer-exempt page (it contains no data)
and asks for the API token once.

`onebrain doctor` adds per-OS firewall posture, GPU-backend hints, and
cross-node version-skew findings (remedy: `onebrain self-update`);
`onebrain self-update` swaps the binary atomically after verifying the
release's `SHA256SUMS` (and its cosign signature when cosign is
installed). The release pipeline is documented in
[RELEASING.md](RELEASING.md).

## The test harness (how claims stay true)

- `cargo test --workspace` — unit/integration tests per crate.
- `cargo xtask smoke` — real CPU inference on a tiny GGUF, greedy
  determinism asserted.
- `cargo xtask e2e` — sandboxed daemon, both dialects streaming, kill -9
  recovery, graceful stop.
- `cargo xtask pair-sim` — two sandboxed daemons pair (ticket + code),
  report RTT/bandwidth, unpair degrades.
- `cargo xtask sim` — the cluster simulator: distributed vs solo
  byte-identity (§9), socket scans (§10), scheduler assertions, the M5
  chaos scenarios (kill mid-generation → same stream completes), M6
  zero-WAN + pre-seed proofs, M7 overlap/speculative/reuse/micro-batch
  proofs. The Linux `--netem` legs run the same scenarios over a shaped
  1 Gbit / 0.5 ms link, where the timing asserts fire.

All of it runs in CI on every commit (`.github/workflows/ci.yml`), on
macOS, Linux, and Windows.
