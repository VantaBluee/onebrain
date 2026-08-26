# OneBrain

**One logical machine for local AI, made from the computers you already own.**

OneBrain connects your own devices — a MacBook, a Windows gaming PC, a Linux box, in any combination — into a single machine for running local AI models. If a model doesn't fit in one device's memory, OneBrain shards it across your paired devices and serves it through one API endpoint. If it *does* fit on one device, OneBrain runs it on that device alone and stays out of the way.

The value is **capacity, not speed**: OneBrain lets you *run models that don't fit on any one of your machines*. Adding devices does not make a single request faster — autoregressive decoding is latency-bound, and every cross-device boundary adds network round-trips. OneBrain is honest about that trade, measures it, and never distributes when it doesn't have to.

> ## Status: pre-release — not usable yet
>
> OneBrain is under active development and has not shipped a release. What exists today is the **M0 skeleton**: the workspace, the vendored engine, and the build/CI plumbing. Everything below that describes running models, pairing, or dashboards is the *plan*, not the present.
>
> - [ ] **M0 — Skeleton & CI** *(in progress)* — workspace, vendored llama.cpp building on all three OSes, CI matrix, smoke inference, `cargo xtask dist`
> - [ ] M1 — Excellent single-node (`up`/`pull`/`run`/`status`/`stop`, OpenAI + Ollama APIs, bearer auth)
> - [ ] M2 — Mesh & pairing (device identities, pair codes, LAN discovery, link probing)
> - [ ] M3 — Distributed inference v1 (pipeline layer-split over the authenticated mesh, auto-solo)
> - [ ] M4 — Scheduler v1 (real microbenchmarks, asymmetric splits, KV budgeting, plan epochs)
> - [ ] M5 — Resilience (death detection, transparent retry, sleep/battery policies)
> - [ ] M6 — Model logistics (P2P range sharing, resumable downloads, curated registry)
> - [ ] M7 — Performance program (chunked prefill overlap, speculative decoding, MoE placement)
> - [ ] M8 — Polish (dashboard, installers, docs, v0.1.0 release)

## Quickstart (build from source)

There are no installers yet — those arrive at M8 (Homebrew, `.msi`, `curl | sh`, `.deb`/`.rpm`). Until then, build from source with a Rust toolchain (1.80+), CMake, and a C/C++ compiler:

```sh
git clone --recursive https://github.com/VantaBluee/onebrain
cd onebrain
cargo build
cargo xtask smoke   # downloads a tiny GGUF and runs a CPU smoke inference
```

If you cloned without `--recursive`, run `git submodule update --init` to fetch the vendored llama.cpp. `cargo xtask dist` builds a release binary and stages a per-OS distribution directory with checksums.

## Platform support

All three platforms are first-class. Every feature works on all of them unless physically impossible, and CI proves it on every commit.

| Platform | Arch | CPU | GPU backends |
|---|---|---|---|
| macOS | arm64 | always on | Metal (`metal` feature) |
| Windows | x64 | always on | CUDA, Vulkan (`cuda`, `vulkan` features) |
| Linux | x64, arm64 | always on | CUDA, Vulkan, ROCm (`cuda`, `vulkan`, `rocm` features) |

Mixed OSes, mixed GPU vendors, mixed performance classes, and any node count are the intended normal case — asymmetric splits are the norm, not an error.

## Security posture

Secure by default, with no insecure mode — planned and enforced from the first networked milestone:

- **Paired-only mesh.** Devices join by an explicit pairing step (short code / ticket). Unpaired connection attempts are rejected. Each device holds an Ed25519 identity generated at init that never leaves the machine.
- **Everything encrypted.** All inter-node traffic runs inside mutually authenticated QUIC (iroh). No raw TCP listeners; the embedded engine's RPC endpoint is never reachable off-box.
- **Authenticated API.** The HTTP API requires a bearer token except on localhost (and the localhost exemption is configurable off).
- **No "insecure mode" flag exists**, and none will be added.

## Architecture

One Rust workspace, one product binary (plus a thin CLI). The same binary acts as **head** (scheduler, API gateway, dashboard) or **worker** (executor) per cluster session — roles are not fixed at install time.

| Crate | Purpose |
|---|---|
| `crates/onebrain-cli` | The `onebrain` binary: `up`, `pair`, `run`, `ls`, `pull`, `status`, `bench`, `doctor`, `stop` |
| `crates/onebraind` | Daemon: role logic, supervision, config, single-instance lock |
| `crates/onebrain-mesh` | iroh transport: identity, pairing, discovery, authenticated streams, link probing, heartbeats |
| `crates/onebrain-engine` | FFI to vendored llama.cpp via a minimal C shim; backend detection; build-hash handshake |
| `crates/onebrain-scheduler` | Device/link profiles, placement solver, plan epochs, strategy choice |
| `crates/onebrain-models` | Model registry, GGUF header parsing, range-fetch shard downloads, integrity, resume |
| `crates/onebrain-api` | axum gateway: OpenAI `/v1/*` and Ollama `/api/*` dialects, SSE streaming, bearer auth |
| `crates/onebrain-dash` | Embedded dashboard SPA + metrics endpoints (arrives M8) |
| `crates/onebrain-proto` | Versioned wire protocol types (postcard), capability flags |
| `vendor/llama.cpp` | Pinned git submodule, built from `build.rs` via CMake |
| `xtask` | `cargo xtask dist` (release artifacts), `smoke` (tiny-model inference test), `sim` (cluster simulator) |

## Why not …?

Respect where it's due — OneBrain exists because each of these proved part of the idea:

- **exo** proved the demand (~47k stars) and set the UX bar, but it is macOS-centric in practice: Linux is CPU-only, Windows is unsupported, and its headline numbers require Thunderbolt 5 Macs on recent macOS with full-mesh cabling. OneBrain targets all three OSes as first-class over ordinary networks.
- **llama.cpp RPC** is cross-platform and is the proven engine underneath OneBrain — but its RPC mode is, by its own documentation, an insecure proof of concept: no auth or encryption, same-build-version required on every machine, no reconnection, assembled by hand. OneBrain embeds llama.cpp and supplies exactly the control plane it lacks: pairing, encryption, version handshakes, scheduling, and resilience.
- **Ollama** has the largest installed base in local AI, and multi-machine support has been requested since 2024 without shipping. OneBrain plans to speak Ollama's API dialect so existing tools work by changing only the base URL.

In one line: *exo's product quality, on everyone's hardware, over ordinary networks, secure by default.*

## Contributing

Contribution guidelines, issue templates, and an `ARCHITECTURE.md` arrive at M8. Until then the project is moving too fast for external PRs to land reliably — issues and discussion are welcome.

## License

Dual-licensed under either of

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in OneBrain by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

The vendored [llama.cpp](https://github.com/ggml-org/llama.cpp) submodule is MIT-licensed by its own authors.
