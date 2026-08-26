# Mesh & pairing contract (M2)

Binding contract between `onebrain-mesh`, the daemon, and the CLI. The wire
messages live in `onebrain-proto` (`pair`, `message`, `handshake`); this
file fixes behavior, names, and files.

## Identity

- iroh `SecretKey` (Ed25519) generated at first daemon start, stored at
  `<config_dir>/device-key` (64 lowercase hex chars of the 32 secret bytes,
  file mode 0600 on Unix). Never leaves the machine, never printed.
- The public identity is iroh's `EndpointId`; its string form (z-base-32 via
  `Display`) is what `onebrain-proto::plan::NodeId` carries and what users
  see shortened to 8 chars in `status`.

## ALPNs

- `onebrain/pair/1` — pairing exchanges only. Accepted from ANY endpoint
  while (and only while) a pairing window is open.
- `onebrain/mesh/1` — all paired traffic. On accept, `remote_id()` must be
  in the peer store; unknown peers are closed immediately with error code 1
  (`unpaired`) and the event is logged at warn. THIS is the §10 guarantee.

## Pairing flow

Host side (`onebrain pair`, no args):
1. Daemon opens a 120-second pairing window: 6-digit code from the OS RNG
   (leading zeros allowed), at most 3 failed attempts, single success.
2. CLI prints the code, the ticket (`EndpointAddr` serialized via
   `iroh-tickets`' endpoint ticket), and a QR of `onebrain:<ticket>`.
3. On success both stores persist the peer and the CLI reports the name.

Joiner side (`onebrain pair <ticket-or-code>`):
- Ticket → dial it directly (works cross-network via relays).
- 6-digit code alone → discover candidates via mDNS on the LAN and try
  each candidate that answers the pair ALPN (the PAKE makes dialing a wrong
  or hostile candidate safe — it learns nothing and cannot complete).
- Then, per `onebrain-proto::pair`: SPAKE2 (crate `spake2` 0.4, symmetric
  mode, password = the 6-digit code, identity = `PAIR_CONFIRM_CONTEXT`) →
  `Confirm` MACs both ways (`confirm_mac`, constant-time compare) →
  `Introduce` both ways → persist.
- Joiners entering a code get 1 attempt per invocation; the host counts
  every failed confirm against its 3-attempt budget and closes the window
  on exhaustion with a "code guessed wrong too often" error.

## Peer store

`<config_dir>/peers.toml`: `[peers.<endpoint_id>] name = "gaming-pc",
added_unix = 1789...`. Names default to the peer's introduced `node_name`,
deduplicated with `-2`, `-3` suffixes. `onebrain unpair <name>` removes by
name (error lists known names on miss). The store is read on every accept
(no restart needed after unpair).

## Mesh service (daemon)

`onebrain-mesh::MeshService::spawn(secret_key, peer_store_path, node_name,
config) -> MeshHandle`, owning the iroh endpoint (built with the default
n0 preset: relays + pkarr publishing for dial-by-key; mDNS added via
`iroh-mdns-address-lookup`). The handle exposes async:
- `pair_start() -> PairWindow { code, ticket, events: mpsc<PairEvent> }`
- `pair_join(target: PairTarget, code: Option<String>) -> Result<PeerInfo>`
- `peers() -> Vec<PeerStatus>` — store + live state merged
- `unpair(name) -> Result<()>`
- `shutdown()`

Per paired peer, the service maintains (lazily, on first reachability):
- a mesh connection (dial by `EndpointId`; accept side symmetrical; tie
  break concurrent dials by lower-id-wins keeping its outgoing conn),
- proto `Hello` exchange on connect (handshake::judge; mismatch closes
  with the remedy in the log and marks the peer `incompatible`),
- heartbeats: `Envelope(Heartbeat)` every 2s on a dedicated bi stream;
  3 missed → `suspect`, 10s silent → `down` (spec §5 timings),
- link profile: RTT = EWMA of heartbeat round-trips (also exposes iroh's
  `Connection::rtt`), bandwidth = one 4 MiB bulk-stream probe on connect
  (repeatable via `probe()`), loss = missed-heartbeat fraction over the
  last 100.

## Internal API additions (same auth rules as the rest of `/api/internal`)

- `POST /api/internal/pair/start` → NDJSON stream:
  `{"status":"window","code":"123456","ticket":"..."}` then
  `{"status":"attempt"}` / `{"status":"paired","peer":{"name","id"}}` /
  `{"status":"expired"}` / `{"status":"failed","message"}` (terminal).
- `POST /api/internal/pair/join` body `{"target":"<ticket|code>",
  "code":"123456"?}` → 200 `{peer}` or error envelope.
- `GET /api/internal/peers` →
  `{"peers":[{"name","id","state":"connected|reachable|down|incompatible|
  unknown","rtt_ms":f64?,"bandwidth_mbps":f64?,"loss":f32?,
  "last_seen_unix":u64?}]}`.
- `POST /api/internal/unpair` body `{"name"}` → 200 or error.
- `GET /api/internal/status` gains `"peers_summary": {"paired":n,
  "connected":n}`.

## CLI

- `onebrain pair` (host) / `onebrain pair <ticket|code>` (joiner): host
  prints code + ticket + QR (crate `qrcode`, render to terminal with
  Unicode half-blocks) and streams window events; joiner prompts for the
  code via stdin when a ticket was given without `--code`.
- `onebrain unpair <name>`.
- `onebrain status` adds a PEERS table: NAME, ID (8 chars), STATE, RTT,
  BANDWIDTH; `--json` includes the full records.

## Tests / DoD hooks

- onebrain-mesh integration tests (single process, two `MeshService`s over
  real loopback iroh endpoints, temp dirs): pairing happy path via ticket
  (both stores persist, names exchanged), wrong code fails without state
  change and burns an attempt, window expiry, unpaired mesh connect is
  REJECTED (assert closed + nothing in store), heartbeat produces RTT and
  peer state `connected`, bandwidth probe reports > 0.
- `cargo xtask pair-sim`: two sandboxed daemons (ONEBRAIN_HOME) on one
  host; pair via ticket through the real CLI; assert `status` shows the
  peer connected with RTT on both; then `unpair` and assert the mesh
  connection drops. Runs in CI on all three OSes (mDNS is NOT exercised in
  CI — multicast is unreliable on hosted runners; LAN discovery is part of
  the manual two-machine checklist).
- Linux netem leg (`cargo xtask pair-sim --netem`, CI ubuntu only): veth
  pair in namespaces shaped to 1 Gbit / 0.5 ms; assert the probed
  bandwidth lands within [0.5, 1.1]× the shaped rate and RTT within
  [0.4, 3]× — wide bands, the point is that measurement happens and is
  sane, not calibration.
