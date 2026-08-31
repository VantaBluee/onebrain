# Security

OneBrain runs an inference protocol that upstream llama.cpp itself documents
as unsafe on open networks. The entire design answer is: that protocol is
never exposed. This document is the threat model — what OneBrain guarantees,
how each guarantee is enforced and tested, and what remains your
responsibility.

## Guarantees (spec §10)

### All inter-node traffic is mutually authenticated and encrypted

Every byte between nodes travels inside QUIC connections (iroh) mutually
authenticated by Ed25519 device identities. Each device generates its key at
first daemon start (`<config_dir>/device-key`, file mode 0600 on Unix); the
key never leaves the machine and is never printed.

A connection on the mesh ALPN from an endpoint that is **not in the peer
store is closed immediately** (application close code 1, `unpaired`) and
logged. The same rule guards the P2P blob-sharing ALPN. This is asserted by
integration tests: an unpaired connect must be rejected with no state
change.

### There is no insecure mode

No flag, config key, or environment variable disables authentication or
encryption, and none will be added. The config parser rejects unknown keys
loudly (`deny_unknown_fields`), so a config file that tries to set
`insecure_mode = true` is a startup *error*, not a silent no-op — there is a
unit test asserting exactly that.

### No raw TCP listeners

The embedded GGML RPC engine is never reachable off-box. Workers serve RPC
sessions over caller-owned socket pairs bridged 1:1 to authenticated QUIC
streams (additive vendor patch `patches/0001-rpc-serve-fd.patch` — no
listener at all). The only TCP listeners a OneBrain daemon ever holds are:

- the HTTP API bind — `127.0.0.1:11435` by default (loopback-only unless
  you change `api_bind`),
- on the head, a **loopback-only** bridge listener that exists only while a
  distributed epoch is active, accepting only within the same machine, torn
  down with the epoch.

This posture is *proven, not promised*: the cluster simulator
(`cargo xtask sim`, run in CI on every commit) enumerates every listening
TCP socket of every daemon process and fails on anything non-loopback, at
all times, including during distributed sessions.

### The API is authenticated

Every HTTP request requires `Authorization: Bearer <token>` (a 64-hex-char
random token created at first start, printed by `onebrain status`). One
deliberate exception exists: **localhost clients** may call the *public*
dialects (`/v1/*`, `/api/*`) without the token, and that exemption is
configurable off (`localhost_auth_exempt = false`). The exemption **never**
applies to `/api/internal/*` — status, load, pairing, metrics, and every
other control endpoint require the token even from loopback. The dashboard
HTML shell at `/` is the one Bearer-exempt page; it contains no data and
asks you for the token, which it then sends on every metrics poll.

### Content stays on your machines

No telemetry, no phone-home. Installers and the binary never call home;
`onebrain self-update` contacts the GitHub releases API only when you run
it. The dashboard is embedded in the binary with zero external assets (no
CDN, no fonts, no framework — ADR 0005), so no third party learns you run
one. The metrics request log records token counts, timings, and finish
reasons — **never prompt or completion text**: the entry type has no field
that could carry text, and a test feeds a sentinel prompt through a
generation and asserts it never reaches the log.
`HF_TOKEN`, when set, is sent only to huggingface.co and its subdomains,
never to other registry mirrors.

## The pairing trust ceremony

Pairing is the security boundary of the whole system, so it is explicit and
human-mediated:

1. The host device opens a 120-second pairing window (`onebrain pair`) and
   shows a 6-digit code from the OS RNG, plus a ticket/QR.
2. The joiner runs SPAKE2 (a balanced PAKE) with the code as the password.
   The code never crosses the wire; a party that doesn't know it learns
   nothing from the exchange and cannot complete it.
3. Key-confirmation MACs are exchanged — joiner first, host verifying in
   constant time before revealing its own, and MACs are direction-bound, so
   a codeless dialer can never reflect a MAC back as its own.
4. The host budget is 3 failed attempts per window, then the window closes.
   On success, both sides persist the peer (`<config_dir>/peers.toml`) and
   only then does the mesh accept it.

**Pairing admits a device to full cluster trust.** A paired peer can serve
and receive model shards, share cached model bytes, and participate in
plans. The GGML RPC protocol that runs *inside* the authenticated tunnels
trusts its peer (that is why it is never exposed to anyone else) — so a
malicious *paired* device could disrupt or corrupt inference on the
cluster. Pair only devices you own and control, and `onebrain unpair
<name>` any device you lose.

## What a stolen ticket can and cannot do

A pairing ticket is serialized *addressing* — the host's public endpoint id
and network addresses. It contains no secret.

Someone holding a ticket **can**: learn the device's endpoint id and
last-known addresses (metadata), and dial it.

They **cannot**: pair without the 6-digit code (each wrong guess burns one
of the host's 3 window attempts, and the PAKE gives a wrong guess zero
information), connect outside a pairing window (the pair ALPN answers only
while a window is open; otherwise close code 2), join the mesh (not in the
peer store ⇒ close code 1), or decrypt any traffic.

The 6-digit code is the actual credential. It is short because it is
single-use, rate-limited (3 attempts), and expires with the 120 s window —
treat it accordingly: don't post a live code+ticket publicly while a window
is open.

## Version skew is refused, never tolerated

At mesh connect, nodes exchange a `Hello` carrying the product version and
an **engine build hash** stamped over the vendored llama.cpp build.
Mismatched builds are marked incompatible and refused (close code 3) with
the remedy in the log — identical builds imply an identical RPC protocol,
which removes upstream RPC's silent cross-version failure mode.
`onebrain doctor` surfaces skew across paired nodes and names the fix
(`onebrain self-update`).

## Supply chain

- `vendor/llama.cpp` is a **pinned** git submodule; bumps are deliberate,
  reviewed commits. Local changes are additive `.patch` files with
  upstreaming notes — never a fork.
- Releases ship `SHA256SUMS` signed with **cosign keyless** (sigstore): the
  signature is bound to this repository's release workflow and the tag via
  GitHub's OIDC — there is no long-lived signing key to steal. Verification
  one-liners are in every release body and in [RELEASING.md](RELEASING.md).
- `install.sh` and `onebrain self-update` verify checksums before
  installing anything (self-update also verifies the cosign signature when
  a `cosign` binary is present, and refuses downgrades by default).
- The dashboard has no JavaScript dependency tree — a few hand-written
  files embedded in the binary, reviewable by reading them.
- CI runs `cargo audit` on every push; dependabot watches the manifests.

## Residual risks (documented, deliberate)

- On the head, the per-epoch RPC bridge listener is loopback-only but
  local: another process on the *head machine itself* could race a connect
  during an active epoch. Same-machine processes running as you are outside
  the threat model (they could equally read your model files).
- On Windows, the worker's socket pair is emulated with a loopback
  listener bound to `127.0.0.1:0` that accepts exactly once and closes —
  a single intra-process accept race, same class as above.
- A paired device is trusted (see the pairing section). Unpair devices you
  no longer control.

## Reporting a vulnerability

Please report vulnerabilities privately via GitHub's security advisory
form: <https://github.com/VantaBluee/onebrain/security/advisories/new>
("Report a vulnerability"). If the form is unavailable, open an issue
asking for a private channel **without** including details of the
vulnerability itself. You should receive an acknowledgement within a few
days; please allow a reasonable window for a fix and a release before
public disclosure.
