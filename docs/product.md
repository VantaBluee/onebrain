# Product polish contract (M8)

Binding contract for spec §8-M8 (+§7 dashboard). The last milestone:
everything here is about a stranger succeeding with OneBrain on their
own machines, honestly (§1.6: the story is capacity, never speed
multiplication).

## 1. Metrics endpoint (feeds the dashboard)

`GET /api/internal/metrics` (token-auth'd like every internal route),
one JSON document, additive-stable schema:

- `node`: name, platform, version, engine build id, memory
  (usable/total), devices, profile (prefill/decode tps, disk),
  battery/draining, sleep-inhibited.
- `peers[]`: name, id-prefix, state (Connected/Suspect/Down/Draining),
  rtt_ms, bandwidth_mbps, loss, memory, profile, version+engine-build
  (from Hello) — version skew computable client- and server-side.
- `plan`: the ActivePlanView (epoch, model, strategy, assignments with
  layer ranges + stage order, predicted_tpt_ms, predicted_prefill_ms)
  or null.
- `requests[]`: ring buffer (last 50, in-memory, head only): id,
  api dialect, model, prompt/completion token counts,
  prefill/decode/ttft ms, drafted/accepted, finish reason, timestamp.
  No prompt text EVER (privacy: §10 — content stays on the machines).
- `advisor[]`: one-line findings computed SERVER-SIDE (pure functions,
  unit-tested): each `{severity, text}`, honest and actionable. v1
  rules (each fires only on measured data): slow-link ("link A↔B
  measures ~X Mbps — a wired connection would lift the pipeline's
  boundary transfer"), memory-starved node (usable ≪ its share of the
  active plan + headroom), draining node in plan, version/engine skew
  between paired nodes, solo-because-infeasible (from selection notes),
  battery-draining worker. No advice without a measurement behind it.

## 2. Dashboard v1 (spec §7)

- Served by the daemon at `/` (and `/dash/*` assets) — same listener,
  Bearer-exempt for the HTML shell and its `/dash/*` static assets ONLY
  (none of which carry data); the shell asks for the token (paste once,
  kept in localStorage) and calls `/api/internal/metrics` with it.
  Loopback exemption applies as configured (§M1 rules).
- ADR (record as docs/decisions/000N): NO framework, NO build step, NO
  CDN — one hand-written static page (semantic HTML + vanilla JS + CSS,
  a few small files) embedded via rust-embed in onebrain-dash. The spec
  suggests Svelte/React "built at release time" but demands
  dependency-light; a zero-dependency SPA removes node_modules from the
  supply chain and the release pipeline entirely. Revisit only if the
  dashboard outgrows ~1.5k lines of JS.
- Renders from `/api/internal/metrics` polled every 2 s: topology graph
  (nodes + links labeled RTT/bandwidth, SVG, no canvas library), plan
  visualization (layer ranges per node in stage order), per-node cards
  (memory bar, tok/s, state), request log table, advisor list on top.
  Copy follows §1.6: the header line is about pooled capacity.
- Dark/light via prefers-color-scheme; usable at laptop widths; no
  external fonts.

## 3. doctor v2 + self-update

- doctor gains (per-OS, best-effort, never fatal): firewall posture
  (Windows: does a Defender rule exist for this binary / will the
  first bind prompt; macOS: local-network permission note; Linux:
  common firewalld/ufw hints), driver/backend hints (GPU present but
  engine built CPU-only, Vulkan/CUDA/Metal availability one-liners),
  and version-skew findings across paired nodes (from stored peer
  Hello data) with the remedy naming `onebrain self-update`.
- `onebrain self-update`: queries the repo's GitHub releases API,
  finds the platform asset, downloads to a temp path, verifies
  SHA256SUMS (and cosign signature when the cosign binary is present —
  optional, never required), atomically swaps the current executable
  (Windows: rename-running-exe dance), `--check` = report only.
  Refuses downgrades unless `--allow-downgrade`. Daemon must be
  stopped (or it refuses with remedy).

## 4. Install paths

- `install.sh`: curl|sh for macOS+Linux — detects OS/arch, downloads
  the latest release tarball + SHA256SUMS from GitHub releases,
  verifies, installs to ~/.local/bin (or /usr/local/bin with sudo),
  prints PATH guidance. Idempotent, no root required by default.
- Windows: `.msi` via WiX v4 (dotnet tool, CI-installable): installs
  onebrain.exe to Program Files, adds to PATH, uninstall entry.
- Linux packages: `cargo-deb` + `cargo-generate-rpm` configs in the
  CLI crate manifest.
- Homebrew: `Formula/onebrain.rb` in-repo (installable via
  `brew install --formula <raw url>` and ready for a tap repo — the
  tap repo itself is a manual follow-up, documented in RELEASING.md).
- All installers get the version from the tag; none phone home.

## 5. Release pipeline

- release.yml (extends M0's stub): on tag `v*` — build dist per OS
  (macos-14 arm64 + x86_64 via macos-13 or universal, ubuntu-22.04
  x86_64, windows-2022 x86_64), `cargo xtask dist` artifacts,
  SHA256SUMS, cosign KEYLESS signing (GitHub OIDC — no cert purchase,
  no secret to manage; publish `.sig` + certificate identity docs),
  .msi + .deb + .rpm + tarballs, then a GitHub release with everything
  attached and verification instructions in the body.
- RELEASING.md: the tag ritual, what CI produces, how a user verifies
  (sha256sum + cosign verify-blob one-liners), manual follow-ups
  (Homebrew tap push).
- Validation: tag `v0.1.0-rc.1` proves the pipeline end-to-end (DoD:
  "release CI publishes signed artifacts"); `v0.1.0` follows once the
  rc run is green. The fresh-machine README walkthrough (DoD's manual
  half) stays a human step, listed in STATUS as such.

## 6. Docs pack

- README: capacity-first pitch, per-OS quickstart (installer one-liner
  → `onebrain up` → `onebrain pull` → `onebrain run`, plus the
  90-second "two laptops, one model" pairing demo with real commands
  and expected output shapes), honest §1.6 framing, link to docs/.
- ARCHITECTURE.md: crate map, the mesh/RPC-over-QUIC design (ADR 0004
  amendments included), plan/epoch lifecycle, M5 resilience story, M6
  logistics, M7 performance — a reader can find the code from the doc.
- SECURITY.md (threat model): §10 guarantees (all inter-node traffic
  mutually authenticated+encrypted, no insecure mode, pairing trust
  ceremony, what a stolen ticket can/can't do, socket posture proven
  by sim scans), reporting contact.
- CONTRIBUTING.md (build prereqs per OS, gate commands, patch regime
  for vendor/llama.cpp, ADR convention) + .github/ISSUE_TEMPLATE
  (bug/feature, asking for `onebrain doctor --json` output) + PR
  template pointing at the gate.

## DoD hooks

- Sim: with the 2-node cluster up, `GET /` serves the dashboard shell
  (contains the app root marker), `/api/internal/metrics` with the
  token returns topology matching the live cluster (both nodes, link
  with measured rtt/bandwidth), the active plan's assignments, and — 
  under netem shaping — the slow-link advisor line fires; without
  auth → 401. Request-log entries appear after a generation with
  nonzero timing and NO prompt text (asserted).
- Unit: advisor rules (each fires/holds on constructed metrics),
  self-update version comparison + SHA verification (local fixture
  server, no real network), installer script shellcheck-clean (xtask
  lint step), WiX/deb/rpm configs validated by building them in the
  release-dry-run CI leg (PR-triggered, no publish).
- CI: a `release-dry-run` job on main builds all installers without
  tagging; the real release.yml runs on tags. Existing test matrix
  must stay green (dashboard adds no runtime deps beyond rust-embed).
- Manual (recorded in STATUS, user-run): fresh-machine README
  walkthrough per OS; two-machine checklists from M3/M2.
