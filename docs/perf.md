# Performance program contract (M7)

Binding contract for spec §8-M7 and the scheduler-v2 note (§"v2 (M7)").
Grounded in a code audit of the pinned vendor tree and our request path;
facts cited here were verified against the vendored sources at contract
time. Ordering rule: measurement lands before optimization — every lever
ships with the instrument that proves it.

## 0. Ground truth the contract is built on

- Upstream `ggml_backend_sched` supports pipeline-parallel execution
  (4 copies + events), but the RPC backend advertises
  `caps.async = false, events = false`, so llama.cpp's
  `pipeline_parallel` gate is ALWAYS false when any RPC device is
  present. Distributed prefill today is strictly sequential: one
  blocking round trip per ubatch per split boundary. M3 "baseline" is
  exactly this path.
- The RPC server processes commands serially, in order, per connection.
  Therefore client-side pipelining (send the next request before reading
  the previous ack) preserves semantics WITHOUT any wire-format change.
- `llama_decode` requires caller-side chunking to `n_batch` and splits
  internally to `n_ubatch` (default 512, not exposed by our shim).
  Steady-state decode graph traffic is 8 bytes/token/remote
  (GRAPH_RECOMPUTE); the real per-ubatch cost is the split-boundary
  activation copy (`4·n_embd·n_tokens` bytes each way, relayed through
  the head for worker↔worker boundaries).
- The C API has complete support for multi-sequence batches
  (`llama_batch_init`, per-token `pos/seq_id/logits`,
  `llama_get_logits_ith`, `LLAMA_MAX_SEQ 256`, `kv_unified`) and KV
  surgery (`llama_memory_seq_rm/cp/keep/pos_max`). Our shim exposes none
  of it — `ob_decode` is `llama_batch_get_one` (seq 0, last-token
  logits). All of M7's engine features are shim-only; no vendor change
  except patch 0003.
- No timing exists anywhere in the product today: `DoneStats` has token
  counts only, Ollama's terminal line omits the duration fields real
  Ollama sends, and profile.toml holds a solo 260K-model microbench.
  The M3 baseline for the before/after table must be CONSTRUCTED via
  config knobs that disable each M7 feature.

## 1. Timing instrumentation (lands first)

- `GenerationStats` (engine) gains wall-clock: `prefill_ms`,
  `decode_ms`, `ttft_ms` (time to first emitted piece), plus
  `drafted`/`accepted` counters (0 until §5). `DoneStats` (api) carries
  them through; the Ollama dialect now emits real
  `total_duration`/`prompt_eval_duration`/`eval_duration` (nanoseconds,
  as real Ollama does); OpenAI `usage` stays counts-only per the
  OpenAI schema.
- The daemon logs one stable line per completed generation:
  `perf: prefill {n}tok {ms}ms decode {n}tok {ms}ms ttft {ms}ms` —
  sim-greppable.

## 2. Engine session/batch substrate (shim-only)

- `ob_session_new` is widened to take an `ob_session_params` struct:
  existing fields + `n_ubatch`, `n_seq_max`, `kv_unified`,
  `flash_attn_type` (AUTO default), `type_k`/`type_v` (F16 default),
  `offload_kqv` (true). Rust `SessionParams` mirrors it with defaults
  preserving today's behavior exactly.
- New shim surface (thin wrappers, no logic): explicit batch API
  (`ob_batch_new/free`, per-token token/pos/seq_id/logits push,
  `ob_decode_batch`), `ob_sample_ith` (sampler over the i-th output),
  and memory ops `ob_memory_seq_rm/seq_cp/seq_keep/seq_pos_max`.
  Position rule (upstream-enforced): sequence positions must stay
  consecutive — rollback is a real `seq_rm`, never a rewound counter.
- `Session::generate` keeps confirm-before-send exactly as documented in
  docs/resilience.md.

## 3. Overlapped chunked prefill (patch 0003 — the DoD headline)

- Vendor patch `0003-rpc-client-async-pipeline.patch` (additive, same
  regime as 0001/0002): the RPC client backend implements
  `graph_compute_async`, `event_new/free/record/wait/synchronize`, and
  the async tensor-copy entry points as a client-side FIFO of pending
  acks per endpoint. `event_record` = FIFO marker; `event_wait`/
  `synchronize` = drain to marker. Device caps flip to
  `async = true, events = true`. NO wire-format change (server
  untouched; in-order serial semantics are the correctness argument —
  document it in the patch header + patches/README.md + upstreaming
  note). Dead-socket behavior must compose with patch 0002: a failed
  pending ack surfaces as an error return on the draining call, and the
  socket registers dead as today.
- With caps flipped, llama.cpp's own gate enables pipeline parallelism
  (verify the remaining conditions hold in our loads: split-mode layer,
  offload_kqv, no tensor overrides; n_gpu_layers already -1). Memory
  note: pipeline parallelism allocates up to 4 compute-buffer copies per
  backend — the scheduler's overhead reserve must account for
  `~4·(4·n_embd·n_ubatch)` per node; fold it into
  `OVERHEAD_RESERVE_BYTES` math with a comment, not a new constant.
- Config: `[perf] prefill_overlap = true` (default). `false` restores
  the exact M3 path (sched created without parallel copies) — this is
  the constructed M3 baseline for benches and the sim proof.
- `n_ubatch` becomes a `[perf]` config knob (default 512); the plan
  explanation mentions the effective value.

## 4. Prefix/KV reuse across requests

- The engine session keeps its token history (prompt + generated) after
  a completed generation instead of resetting. On the next request the
  host computes the longest common token prefix; if ≥ a floor (64
  tokens), it `seq_rm`s the divergent suffix and decodes only the new
  tokens; otherwise full reset. Guarantee (sim + unit asserted): greedy
  output is byte-identical to a cold run, and the second request's
  prefill decodes exactly `len(prompt) - len(shared_prefix)` tokens
  (assert via the perf log line).
- Interactions (binding): epoch teardown, plan change, model swap, and
  the M5 retry path all reset the reuse state (retry semantics
  unchanged — full re-prefill; correctness first). `[perf] kv_reuse =
  true` default, `false` = today's behavior.

## 5. Speculative decoding (`--speculative`)

- Draft placement v1: the draft model loads SOLO on the head (the spec's
  "fastest single node" is the API-serving node in every 2-3 node
  topology we target; revisit with a proto change only if bench data
  ever shows otherwise — note it in the writeup). Target may be solo or
  sharded. New second-model slot in the engine host, explicitly excluded
  from the single-model invariant; unload order: draft before target.
- Loop (per accepted batch): draft K=8 greedy tokens on the draft
  session; verify in ONE target `ob_decode_batch` with per-position
  logits; accept the longest prefix where target greedy == draft token;
  emit accepted tokens (their verifying decode succeeded — this IS
  confirm-before-send, one batch earlier); `seq_rm` rejected positions
  on the target, always resync the draft's KV to the accepted stream.
  GREEDY TOKEN-EQUIVALENCE IS THE DoD: with `--speculative`, greedy
  output must be byte-identical to non-speculative — sim-asserted
  against both a solo and a distributed target.
- Non-greedy sampling with a draft is out of M7 scope (rejection
  sampling deferred; `--speculative` + temperature>0 runs the target
  path with a logged notice — honest UX over silent wrongness).
- Surface: `onebrain run <model> --speculative [--draft <ref>]` and
  config `[perf] draft_model`. No draft ⇒ error naming usable registry
  pairs (same-vocab check at load, typed error on mismatch).
  `drafted`/`accepted` flow into stats + the perf log line.
- M5 interplay: a torn distributed target mid-verify surfaces as
  Interrupted exactly as today (accepted-token bookkeeping feeds
  `generated_tokens`); retry re-prefills and CONTINUES speculating.

## 6. Micro-batched decode (concurrent requests)

- The supervisor stops single-flighting generations: up to
  `[perf] max_concurrent_requests` (default 4) jobs run concurrently in
  the engine host on one session with `kv_unified = true`,
  `n_seq_max = max_concurrent_requests`; each decode step batches one
  token per active sequence (prefills chunk-interleave FCFS; no
  starvation: a sequence's decode never waits on another's prefill
  chunk more than one chunk).
- Admission control (fixes the audited unbounded-queue gap): a request
  that cannot fit (unified-KV headroom check at admission) queues up to
  `queue_depth` (default 8), beyond which the dialects return a typed
  429-equivalent with remedy. Cancellation: a disconnected client's
  sequence is `seq_rm`'d at the next step boundary, not after prefill.
- Correctness assertions: run-to-run determinism (greedy, fixed seeds)
  is a HARD assert; concurrent-vs-alone byte-equality on the sim model
  is the primary assert — if CPU batching empirically breaks it, the
  implementer must record the divergence in this file's appendix with
  the measured delta and downgrade only that assert to run-to-run
  determinism (never silently).
- Status honesty: `HostMsg::Models`-class queries answer from cached
  state, never queue behind generation (fixes "model: null while
  busy").
- Speculative + micro-batch compose only when a single request is
  active (drafting steals batch slots); document the scheduling rule in
  code.

## 7. Scheduler v2-lite (cost-model search)

- `evaluate()` grows a transfer term: per stage boundary,
  `RTT + (4·n_embd·n_ubatch)/measured_bandwidth` for prefill-weighted
  cost alongside the decode term (link bandwidth now flows into
  `PlanRequest` — `LinkRtt` gains `bandwidth_mbps`, fed from
  `PeerStatus`; absent ⇒ today's RTT-only behavior).
- Candidate search in the prima.cpp spirit, kept enumerable: for each
  viable node subset (existing inclusion rules), evaluate the tilt
  family {memory-proportional, current 0.5+0.5 compute tilt, full
  compute-proportional, and each with the slowest node underweighted one
  layer-quantum} × exact stage orders (≤8 nodes as today); pick min
  predicted tpt. Every candidate considered appears in `--explain`
  (count + winner rationale). ±1-layer prediction assertions from M4
  keep holding.
- Disk-offload penalties: N/A — OneBrain does not offload to disk;
  recorded here so the spec item has a documented disposition.
- Tensor-parallel islands: the meta/TP device also reports
  `events=false` upstream and TP-over-RPC is unproven; M7 measures
  nothing here and KEEPS pipeline-parallel, recording that decision +
  the upstream evidence in the writeup (spec allows exactly this
  disposition: "otherwise keep PP and document why").

## 8. MoE awareness (exploration, written up)

- `ModelDims` learns experts: read `{arch}.expert_count` /
  `expert_used_count` from GGUF so KV/compute predictions stop
  mis-costing MoE models (weights math is already tensor-range-driven);
  registry `moe_*` fields become cross-checks in `--explain`.
- Experiment (measurements land as an appendix in this file): a
  synthetic MoE GGUF in the sim (GgufBuilder + `ffn_*_exps` tensors,
  expert metadata) loaded distributed with per-layer expert tensors
  placed via `tensor_buft_overrides` (public RPC buft) vs default
  placement; measure prefill/decode both ways. Overrides disable
  upstream pipeline-parallel — the experiment records that trade-off;
  expert placement ships as a measured writeup + the dims fix only, not
  a default-on scheduler feature.

## 9. int8 activation compression — deferred, with reasons

Deliberately NOT implemented in M7: (a) lossy activation compression
breaks the §9 byte-identity guarantee by construction, so it can never
be default-on; (b) the audit shows decode-path traffic is 8 B/token +
one 16 KiB-class boundary copy — latency-bound, not bandwidth-bound;
prefill bandwidth is addressed by overlap (§3) first. Insertion points
are documented (RPC buffer get/set quantization behind a protocol-bump
patch; worker↔worker direct copy to kill the head double-hop) for a
future milestone if field measurements show sub-1Gbps links binding
after overlap ships. The `--compress-activations` flag intentionally
does not exist yet.

## 10. bench --cluster

- New control messages `Message::BenchRequest { }` /
  `Message::BenchReport { prefill_tps, decode_tps, disk_mbps,
  measured_unix }` (peer runs its §M4 microbench on demand, echo-style
  on one control stream like RangeQuery). PROTO_VERSION → 5;
  CLUSTER_BENCH capability bit.
- `onebrain bench --cluster`: every connected peer's fresh microbench +
  link table + an END-TO-END distributed measurement — timed prefill +
  timed decode of a standard prompt on the ACTIVE plan (or a plan it
  creates on the bench model), reported against (a) the same run with
  `prefill_overlap=false` + `kv_reuse=false` (constructed M3 baseline)
  and (b) solo on the local node. Output: reproducible markdown table
  (`--json` for machines); values are measurements, labeled with model
  + plan + config, never promises (§1.6 honest-UX rule).

## DoD hooks

- Sim (netem leg, 1 Gbit / 0.5 ms — the "1Gbps sim profile"): a perf
  step loads a synthetic model SIZED so per-ubatch boundary transfer is
  a substantial (≥40%) fraction of per-chunk wall time, then measures a
  long-prompt distributed prefill with `prefill_overlap=false` vs
  `true`, median of 3 each: assert `overlap ≤ 0.75 × sequential`
  (the spec's ≥25%) AND decode tok/s within [0.9, ∞) of the
  no-overlap run (no decode regression). Netem-only because hosted
  runners are too noisy for absolute numbers; the non-netem sim runs
  the same step assert-free as a smoke.
- Sim: speculative greedy token-equivalence — same prompt, spec on/off,
  solo AND distributed target, byte-identical streams; acceptance
  counter > 0 asserted via the perf log line.
- Sim: prefix-reuse proof — request B sharing request A's prefix logs a
  prefill of exactly the suffix length and produces byte-identical
  output to a cold-cache control.
- Sim: micro-batch proof — two concurrent requests both stream to
  completion, each byte-identical to its alone-run (or the documented
  fallback), plus the 429-with-remedy path when the queue is full.
- Unit/integration: batch/seq shim wrappers (rollback via seq_rm,
  per-index logits), admission math, LinkRtt bandwidth plumbing, v2
  candidate search determinism, MoE dims parsing, Ollama duration
  fields.
- CI: rides the existing test-job sim + the netem leg; the perf step's
  measured numbers print in CI logs so regressions are diffable.
