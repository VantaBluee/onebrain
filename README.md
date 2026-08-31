# OneBrain

**One logical machine for local AI, made from the computers you already own.**

OneBrain connects your own devices — a MacBook, a Windows gaming PC, a Linux box, in any combination — into a single machine for running local AI models. If a model doesn't fit in one device's memory, OneBrain shards it across your paired devices and serves it through one API endpoint. If it *does* fit on one device, OneBrain runs it on that device alone and stays out of the way.

The value is **capacity, not speed**: OneBrain lets you *run models that don't fit on any one of your machines*. Adding devices does not make a single request faster — autoregressive decoding is latency-bound, and every cross-device boundary adds a network round trip per token. OneBrain is honest about that trade, measures it, and never distributes when it doesn't have to.

## Install

Every release artifact is checksummed (`SHA256SUMS`) and signed (sigstore/cosign, keyless); the installers verify checksums before touching your system. Verification one-liners are in the [release notes](https://github.com/VantaBluee/onebrain/releases) and in [RELEASING.md](RELEASING.md).

**macOS and Linux** — one line, no root needed (installs to `~/.local/bin`):

```sh
curl -fsSL https://raw.githubusercontent.com/VantaBluee/onebrain/main/install.sh | bash
```

**Homebrew** (macOS and Linux):

```sh
brew install --formula \
  https://raw.githubusercontent.com/VantaBluee/onebrain/main/Formula/onebrain.rb
```

**Windows** — download `onebrain-vX.Y.Z-x86_64-pc-windows-msvc.msi` from the [releases page](https://github.com/VantaBluee/onebrain/releases) and run it: it installs `onebrain.exe` to Program Files, adds it to `PATH`, and registers an uninstall entry.

**Debian/Ubuntu and Fedora/RHEL** — native packages on the releases page:

```sh
sudo apt install ./onebrain_X.Y.Z-1_amd64.deb      # Debian/Ubuntu
sudo dnf install ./onebrain-X.Y.Z-1.x86_64.rpm     # Fedora/RHEL
```

**From source** — Rust 1.91+, CMake, and a C/C++ compiler (the vendored llama.cpp builds from source):

```sh
git clone --recursive https://github.com/VantaBluee/onebrain
cd onebrain
cargo build --release
```

Later, `onebrain self-update` upgrades any installed binary in place from the latest GitHub release — checksum-verified, cosign-checked when a `cosign` binary is on your `PATH`, and it refuses downgrades unless you pass `--allow-downgrade`. `onebrain self-update --check` only reports.

## Quickstart (one machine)

```sh
onebrain up                # start the daemon
onebrain pull qwen3-4b     # download a model into the local cache
onebrain run qwen3-4b      # load it and print how to talk to it
```

`run` ends with everything a client needs:

```
model ready: qwen3-4b (2.5 GB, 36 layers, ctx 4096)
endpoint         http://127.0.0.1:11435
OpenAI base_url  http://127.0.0.1:11435/v1
token            <64-hex API token>

try it:
  curl http://127.0.0.1:11435/api/generate -d '{"model":"qwen3-4b","prompt":"Why is the sky blue?"}'
```

OneBrain speaks two API dialects at once, so existing tools work by changing only the base URL:

- **OpenAI-compatible** clients → `http://127.0.0.1:11435/v1` (chat/text completions, SSE streaming)
- **Ollama-compatible** clients → `http://127.0.0.1:11435` (`/api/generate`, `/api/chat`, NDJSON streaming)

The API requires the bearer token except on localhost (and that exemption is [configurable off](SECURITY.md)). `onebrain status` prints the token; a **dashboard** is served by the daemon at `http://127.0.0.1:11435/` — open it, paste the token once, and watch topology, the active plan, per-node memory, the request log, and advisor findings live.

Model references are registry ids (below), `hf:<org>/<repo>/<file.gguf>` for any GGUF on Hugging Face, or an absolute local path to a `.gguf` file.

## Two laptops, one model — the 90-second demo

Two machines, each too small for the model, running it together. Both need OneBrain installed and on the same LAN (different networks work too, via the ticket).

**1. Start both daemons** — on each machine:

```sh
onebrain up
```

**2. Pair them** — on laptop A:

```sh
onebrain pair
```

```
pairing window open (120 s, up to 3 attempts)

    code:   4 8 2 9 1 7

ticket (works across networks):
<ticket text>
scan to pair:
<QR code>
on the other device:
  onebrain pair <ticket>   any network (asks for the code)
  onebrain pair 482917     same LAN
```

On laptop B, type the 6-digit code:

```sh
onebrain pair 482917
```

Both sides report `paired with <name>`. Pairing is a one-time trust ceremony (a PAKE — the code never crosses the wire; see [SECURITY.md](SECURITY.md)); from now on the two machines find and reconnect to each other automatically.

**3. Pull the model once** — on laptop A only:

```sh
onebrain pull qwen3-32b    # ~20 GB, one WAN download for the whole cluster
```

**4. Run it** — on laptop A:

```sh
onebrain run qwen3-32b --explain
```

```
planning placement...
plan: PipelineParallel across 2 nodes (epoch 1)
  stage 0  ab12cd34  layers 0-38 (39 layers)
  stage 1  ffee0011  layers 39-63 (25 layers)
why: <per-node memory figures and the binding constraint>
loading model into memory...
model ready: qwen3-32b (19.8 GB, 64 layers, ctx 4096)
endpoint         http://127.0.0.1:11435
...
```

Laptop B never touches the internet for this: it fetches the file header plus exactly its assigned layers **from laptop A over the LAN**. That zero-WAN property is proven in CI — the test suite pulls through a byte-counting fake-WAN server and asserts the second node's fetch moves **zero new WAN bytes**. The split is proportional to each machine's measured usable memory and compute, and the layer ranges above will reflect *your* machines.

Point any OpenAI or Ollama client at laptop A's endpoint and use the model as if it were one machine. `onebrain status` on either laptop shows the peer table and the active plan; the dashboard draws the topology with measured RTT and bandwidth on the link.

Model sizes to plan around (from the built-in registry — usable memory, not installed RAM):

| Registry id | Needs (solo or pooled) |
|---|---|
| `qwen3-4b` | 6 GiB |
| `deepseek-r1-distill-qwen-14b` | 13 GiB |
| `qwen3-32b` | 26 GiB |
| `llama-3.3-70b-instruct` | 54 GiB |
| `gpt-oss-120b` (MoE) | 79 GiB |
| `glm-4.5-air` (MoE) | 91 GiB |

## What to expect from performance

Honesty first: **a distributed model is slower per token than the same model on one sufficiently large machine.** Pipeline boundaries cost a network round trip per token, and no software removes that physics. What OneBrain guarantees and measures instead:

- **Correctness is absolute.** Greedy output from a distributed run is byte-identical to a solo run of the same prompt — asserted in CI on every commit, including under failure injection (a worker killed mid-generation is retried transparently into the same response stream, and the final text still matches the uninterrupted control run).
- **Prompt processing overlaps.** Long-prompt prefill pipelines chunks across nodes (an additive llama.cpp patch teaches the RPC client async submission). On CI's shaped 1 Gbit / 0.5 ms link, overlapped prefill is asserted to take **≤ 0.75× the sequential baseline's time** (≥ 25% faster) with no decode regression.
- **Weights move once.** Paired machines share model bytes LAN-first (CI-asserted zero WAN bytes for the second machine), and reloads skip re-pushing any large tensor a worker already holds.
- **Solo stays solo.** If the model fits one machine, the planner short-circuits to a single node and the mesh is not involved in inference at all.
- **Your numbers, measured.** `onebrain bench --cluster` benchmarks every paired machine and times a real end-to-end generation — as configured, against the no-overlap baseline, and against solo — in one reproducible table. Every figure OneBrain shows you is a measurement of your hardware, never a promise.

## Platform support

All three platforms are first-class. Every feature works on all of them unless physically impossible, and CI proves it on every commit.

| Platform | Arch | CPU | GPU backends |
|---|---|---|---|
| macOS | arm64, x86_64 | always on | Metal (`metal` feature) |
| Windows | x64 | always on | CUDA, Vulkan (`cuda`, `vulkan` features) |
| Linux | x64 | always on | CUDA, Vulkan, ROCm (`cuda`, `vulkan`, `rocm` features) |

(The every-commit CI matrix runs Apple-silicon macOS, Linux, and Windows; Intel-mac binaries are built and shipped by the release pipeline.)

Mixed OSes, mixed GPU vendors, mixed performance classes, and any node count are the intended normal case — asymmetric splits are the norm, not an error. `onebrain doctor` diagnoses driver, firewall, and version problems with concrete remedies.

## Security posture

Secure by default, with no insecure mode:

- **Paired-only mesh.** Devices join by an explicit pairing ceremony (6-digit code through a PAKE). Unpaired connection attempts are rejected — a guarantee asserted by tests. Each device holds an Ed25519 identity generated at first start that never leaves the machine.
- **Everything encrypted.** All inter-node traffic runs inside mutually authenticated QUIC (iroh). No raw TCP listeners: the test suite scans every daemon's listening sockets and forbids anything non-loopback.
- **Authenticated API.** Bearer token everywhere; the localhost exemption is configurable off and never applies to internal endpoints.
- **No "insecure mode" flag exists**, and none will be added — a config file that tries to set one fails loudly.
- **Content stays on your machines.** No telemetry, no phone-home; the dashboard is embedded in the binary with zero external assets; the request log keeps counts and timings, never prompt text.

The full threat model — what pairing proves, what a stolen ticket can and cannot do, residual risks — is in [SECURITY.md](SECURITY.md).

## Architecture

One Rust workspace, one product binary (plus a thin CLI). The same binary acts as **head** (scheduler, API gateway, dashboard) or **worker** (executor) per cluster session — roles are not fixed at install time. The full tour, from crate map to plan/epoch lifecycle to the vendor-patch regime, is in [ARCHITECTURE.md](ARCHITECTURE.md).

| Crate | Purpose |
|---|---|
| `crates/onebrain-cli` | The `onebrain` binary: `up`, `pair`, `run`, `status`, `pull`, `ls`, `pin`, `bench`, `doctor`, `self-update`, `stop`, … |
| `crates/onebraind` | Daemon: engine host, supervisor, plans/epochs, power policy, metrics + advisor |
| `crates/onebrain-mesh` | iroh transport: identity, pairing, discovery, authenticated streams, link probing, P2P blob sharing |
| `crates/onebrain-engine` | FFI to vendored llama.cpp via a minimal C shim; additive RPC patches; build-hash handshake |
| `crates/onebrain-scheduler` | Measured device/link profiles, placement search, KV budgeting, plan epochs |
| `crates/onebrain-models` | Registry, GGUF parsing, range-level content-addressed cache, resumable downloads |
| `crates/onebrain-api` | axum gateway: OpenAI `/v1/*` and Ollama `/api/*` dialects, SSE/NDJSON streaming, bearer auth |
| `crates/onebrain-dash` | Embedded dashboard: hand-written static SPA, no framework, no build step, no CDN |
| `crates/onebrain-proto` | Versioned wire protocol (postcard), capability flags |
| `vendor/llama.cpp` | Pinned git submodule, built from `build.rs` via CMake, patched additively (never forked) |
| `xtask` | Repo automation: `dist`, `smoke`, `e2e`, `pair-sim`, `sim` (the cluster simulator CI runs) |

## Why not …?

Respect where it's due — OneBrain exists because each of these proved part of the idea:

- **exo** proved the demand (~47k stars) and set the UX bar, but it is macOS-centric in practice: Linux is CPU-only, Windows is unsupported, and its headline numbers require Thunderbolt 5 Macs on recent macOS with full-mesh cabling. OneBrain targets all three OSes as first-class over ordinary networks.
- **llama.cpp RPC** is cross-platform and is the proven engine underneath OneBrain — but its RPC mode is, by its own documentation, an insecure proof of concept: no auth or encryption, same-build-version required on every machine, no reconnection, assembled by hand. OneBrain embeds llama.cpp and supplies exactly the control plane it lacks: pairing, encryption, version handshakes, scheduling, and resilience.
- **Ollama** has the largest installed base in local AI, and multi-machine support has been requested since 2024 without shipping. OneBrain speaks Ollama's API dialect so existing tools work by changing only the base URL.

In one line: *exo's product quality, on everyone's hardware, over ordinary networks, secure by default.*

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — how it all fits together, with pointers into the code
- [SECURITY.md](SECURITY.md) — threat model, guarantees, and how to report a vulnerability
- [CONTRIBUTING.md](CONTRIBUTING.md) — build prereqs, the CI gate, patch and ADR conventions
- [RELEASING.md](RELEASING.md) — how a tag becomes a signed release, and how to verify one
- `docs/` — the binding per-milestone contracts (mesh, distributed inference, scheduler, resilience, logistics, performance, product polish) and ADRs under `docs/decisions/`

## License

Dual-licensed under either of

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in OneBrain by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

The vendored [llama.cpp](https://github.com/ggml-org/llama.cpp) submodule is MIT-licensed by its own authors.
