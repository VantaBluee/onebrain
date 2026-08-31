# Contributing to OneBrain

Thanks for wanting to help. OneBrain is milestone-driven: each subsystem has
a **binding contract** under `docs/` (mesh, distributed, scheduler-v1,
resilience, logistics, perf, product) and significant decisions are recorded
as ADRs under `docs/decisions/`. Read the contract for the area you're
touching before writing code — when code and contract disagree, that's a bug
in one of them, and the fix starts with deciding which.

## Build prerequisites

Everything is one Rust workspace; the vendored llama.cpp builds from source
via CMake, so you need a C/C++ toolchain:

| | Requirement |
|---|---|
| All OSes | Rust **1.91+** (stable), CMake, git |
| Windows | Visual Studio 2022 Build Tools (MSVC C++ workload). Ninja recommended. |
| macOS | Xcode Command Line Tools (`xcode-select --install`) |
| Linux | `gcc`/`g++` (or clang), `make`/`ninja` |

Clone with the submodule:

```sh
git clone --recursive https://github.com/VantaBluee/onebrain
cd onebrain
cargo build --workspace
```

If you cloned without `--recursive`: `git submodule update --init`.

Notes that save time:

- **The first build is slow.** The engine is always compiled at full
  optimization, even for dev profiles — debug-built ggml is orders of
  magnitude too slow to smoke-test against (ADR 0002).
- **Windows path length**: MSBuild's file tracker still enforces `MAX_PATH`.
  If the engine build fails with FTK1011-class errors, set a short
  `CARGO_TARGET_DIR` (e.g. `D:\t` — CI does exactly this). Also never pass
  `canonicalize()`d (`\\?\…`) paths to CMake/MSBuild.
- GPU backends are opt-in features of `onebrain-engine`
  (`metal`, `cuda`, `vulkan`, `rocm`); plain `cargo build` is CPU-only and
  is what most development needs.

## The gate

CI (`.github/workflows/ci.yml`) enforces all of this on every push and PR,
on macOS, Linux, and Windows. Run it locally before pushing:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets     # CI sets RUSTFLAGS=-Dwarnings
cargo build --workspace
cargo test --workspace
cargo xtask smoke        # downloads a tiny GGUF, real CPU inference
cargo xtask e2e          # sandboxed daemon, both API dialects, kill -9 recovery
cargo xtask pair-sim     # two daemons pair over real loopback iroh
cargo xtask sim          # the cluster simulator: distributed correctness,
                         # socket scans, chaos, logistics, perf proofs
```

Clippy must be clean **with warnings denied** (`RUSTFLAGS=-Dwarnings`) —
CI treats every warning as an error.

Some engine-backed tests self-skip unless `OB_SMOKE_MODEL` points at the
tiny smoke model. After `cargo xtask smoke` has run once:

```sh
# POSIX shells
export OB_SMOKE_MODEL="$PWD/target-smoke/stories260K.gguf"
cargo test --workspace
```

```powershell
# PowerShell
$env:OB_SMOKE_MODEL = "$PWD\target-smoke\stories260K.gguf"
cargo test --workspace
```

On Linux with root, the netem legs run the same rehearsals over a shaped
1 Gbit / 0.5 ms link (this is where the timing assertions fire):

```sh
sudo -E env PATH=$PATH OB_E2E_SKIP_BUILD=1 cargo xtask pair-sim --netem
sudo -E env PATH=$PATH OB_E2E_SKIP_BUILD=1 cargo xtask sim --netem
```

Installer configs (`.msi`/`.deb`/`.rpm`, `install.sh`, the Homebrew
formula) are rehearsed by CI's `release-dry-run` job on every PR — among
other things it runs `shellcheck install.sh` and `ruby -c
Formula/onebrain.rb`, so installer changes fail in your PR, not on tag day.

## Code conventions

- **Errors carry remedies.** Error types are `thiserror` enums whose
  messages tell the user what to *do*, not just what broke ("daemon not
  running; run `onebrain up` to start it"). If a failure has no actionable
  remedy, say what state the user is left in.
- **Logging is `tracing`**, structured where it helps; grep-stable log
  lines that tests assert on (the `perf:` line, the logistics transfer
  summaries) are contracts — don't reword them casually.
- **Doc comments explain *why*.** The codebase leans heavily on module and
  item docs that record intent, contract references, and the reasoning
  behind non-obvious choices. Match that: cite the contract
  (`docs/foo.md §N`) your code implements.
- **Honest UX** (spec §1.6): user-facing copy is about *capacity* — running
  models that don't fit one machine — never speed multiplication. Numbers
  shown to users are measurements, labeled with what was measured.
- **Claims need proofs.** Behavioral guarantees (byte-identity, no
  listeners, zero WAN bytes, overlap speedups) exist because a test asserts
  them. If your change adds a guarantee, add its assertion; if it
  invalidates one, the discussion belongs in the PR description and
  probably an ADR.
- Commit messages follow Conventional Commits
  (`feat(scope): …`, `fix: …`, `docs: …`), matching the existing history.

## The vendor patch regime

`vendor/llama.cpp` is a **pinned submodule, never a fork** (spec §11).
Local changes live as patch files under `patches/`, applied idempotently by
`crates/onebrain-engine/build.rs` (each patch has a marker symbol; the
build fails loudly if a patch no longer applies). Rules:

- Patches are **minimal and additive** wherever possible; the wire format
  and server paths of the RPC protocol are untouched by all three existing
  patches.
- Every patch gets a full entry in [patches/README.md](patches/README.md):
  what it does, the correctness argument, failure-mode composition with the
  other patches, and an **upstreaming note** — the goal is always to
  propose the change to ggml-org/llama.cpp, not to accumulate divergence.
- Vendor bumps are deliberate, standalone commits: update the pin, re-apply
  the patches, rerun the full CI matrix and smoke inference, and record
  anything the shim had to adapt to (ADR 0002). The shim's hand-written
  `extern "C"` declarations make upstream drift a *compile* error by
  design.

## ADR convention

Decisions that reject a plausible alternative are recorded in
`docs/decisions/NNNN-short-title.md` — numbered sequentially, with the
existing files as templates: a **Context** section (the forces), a
**Decision** section (what and why, including the rejected alternative and
why it lost), and a **Consequences** section (costs accepted, revisit
triggers). Status and date at the top. Amendments to a shipped ADR are
edits that say they are amendments (see ADR 0004's accept-loop amendment),
not silent rewrites.

Write one when: you pick between architectures, change a security-relevant
posture, adopt or reject a dependency, or make a call a future contributor
would otherwise re-litigate.

## Issues and pull requests

- Bug reports: use the issue template and include `onebrain doctor --json`
  output plus the daemon log tail (`onebrain doctor` prints your data dir;
  the log is at `<data_dir>/logs/daemon.log`). Redact anything you consider
  private — none of it leaves your machine otherwise.
- Security issues: **not** the issue tracker — see [SECURITY.md](SECURITY.md).
- PRs: keep them scoped, fill in the template, and make the gate green.
  `STATUS.md` and `CHANGELOG.md` are maintained by the milestone process —
  don't edit them in feature PRs unless asked.

## License

Dual-licensed MIT OR Apache-2.0. Unless you explicitly state otherwise, any
contribution intentionally submitted for inclusion in OneBrain by you, as
defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions (inbound = outbound; no CLA).
