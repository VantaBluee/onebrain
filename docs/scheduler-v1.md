# Scheduler v1 contract (M4)

Binding contract for the real scheduler (spec §4 v1, §8 M4). Replaces the
M3 "v1-lite" placement (85% flat ceiling, memory-only proportions) while
keeping its API shape (`plan()` → `PlannedPlacement`) and the epoch
machinery unchanged.

## Profiles (measured, never assumed)

`onebrain-scheduler::profile` gains a real `DeviceProfile` producer, run at
pairing time and by `onebrain bench`:

- **Usable memory**: as today (device free minus OS reserve; `[debug]`
  override wins). Per backend when several devices exist.
- **Compute microbench** (~10 s budget): the registry test model
  (`tinystories-260k`, pulled through the normal registry path on first
  bench — never bundled in the binary): one warmup generate, then measured
  prefill tok/s (64-token prompt, 3 reps, median) and decode tok/s
  (32 greedy steps, 3 reps, median).
- **Disk sequential read**: read 64 MiB of the cached model file with a
  fresh handle, MB/s (documented caveat: OS page cache makes this an upper
  bound; used only for relative ordering and disk-offload penalties in M7).
- **Links**: RTT EWMA + probed bandwidth from the M2 mesh prober, refreshed
  by `onebrain bench` via `MeshHandle::probe`.

Profiles persist at `<config_dir>/profile.toml` (`measured_unix` stamp;
`onebrain bench` refreshes, otherwise reused). Workers report them in the
mesh `NodeStatus` message, which gains `prefill_tps`, `decode_tps`,
`disk_mbps` fields — an in-place proto change, legal because the engine
build hash gate means clusters are always same-build; `PROTO_VERSION`
bumps to 2 so the refusal message stays truthful for genuinely mixed
builds.

## Placement algorithm

Inputs as today plus per-node `DeviceProfile` and the link table.

1. **KV budgeting replaces the 85% ceiling.** From the GGUF header:
   `kv_per_layer = 2 (K+V) × ctx × n_embd_kv × 2 bytes (f16)`, where
   `n_embd_kv = n_head_kv × head_dim` (fall back to `n_embd` when GQA keys
   are absent). Node budget = usable − fixed overhead (512 MiB compute/
   graph reserve); a node's layer capacity satisfies
   `layers × (weight_per_layer + kv_per_layer) ≤ budget`. Weights per
   layer from the GGUF tensor ranges (real bytes, embedding/output counted
   on their host).
2. **Memory-and-compute score.** Candidate layer shares are proportional
   to `capacity_layers × (0.5 + 0.5 × decode_tps / max_decode_tps)` —
   memory sets the hard cap, compute tilts the split so a fast node takes
   more than its memory share when memory allows (largest-remainder
   rounding as today; nodes rounding to 0 drop).
3. **Boundary-on-fastest-link.** Every pipeline boundary costs ~1 RTT per
   token regardless of size, so stage ORDER is chosen to minimize the sum
   of RTTs across consecutive stages: exact search over permutations for
   ≤ 8 participants (head pinned last for sampling locality), greedy
   nearest-neighbor beyond. The explanation names the per-boundary RTTs.
4. **Auto-solo unchanged** (now with the real KV budget), `--nodes`
   semantics unchanged. A third node joins the plan only when the
   two-node plan is memory-infeasible OR the predicted time-per-token
   improves ≥ 5% (decode model: `max_stage(layers/decode_tps_node) +
   Σ boundary RTTs`); otherwise the plan states why it was left out.

## `onebrain bench`

CLI command (replaces the M2 stub): runs the local profile, probes every
connected peer's link, and prints the one-page report: node table
(memory, prefill/decode tok/s, disk), link table (RTT, bandwidth, loss),
and the active profile's age. `--json` structured. Internal endpoint
`POST /api/internal/bench` drives it; profiles refresh in the store and
the next NodeStatus.

## DoD sim hooks (extend `cargo xtask sim`)

- **Asymmetric**: fast-big vs slow-small (caps + a decode_tps override in
  `[debug]` — add `decode_tps_override`) ⇒ assert the layer ratio lands
  within ±1 layer of the score prediction.
- **Third node helps only when it helps**: 3-node run where two fit the
  model ⇒ plan stays 2-node and the explanation says why; shrink caps so
  only 3 fit ⇒ 3-node plan.
- **KV shifts with ctx**: same caps, ctx 2k vs 16k ⇒ fewer layers per
  node at 16k (assert), and a capped 16k load that would OOM under the
  M3 flat ceiling plans correctly (no engine OOM in the sim).
