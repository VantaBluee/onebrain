# Distributed inference contract (M3)

Binding contract for pipeline layer-split across paired nodes. Grounded in
the RPC recon of the vendored tree (see `patches/README.md`): GGML RPC
sessions are tunneled through authenticated mesh streams; **no raw TCP
listener exists anywhere**, verified by a socket-scan test.

## Transport shape

- Mesh connections multiplex typed bi-streams. First frame on any mesh
  stream is a postcard `StreamHeader { kind, epoch }`
  (`onebrain-proto::message`): kinds `control` (Envelope traffic incl.
  heartbeats — existing behavior), `rpc` (one GGML RPC session), `probe`.
- **Worker side** (`rpc` stream accepted, epoch must equal the worker's
  active epoch, sender must be the epoch's head — else close code 4
  `bad-epoch`): create a connected local socket pair (Unix `socketpair`;
  Windows: loopback listener on 127.0.0.1:0, self-connect, accept once,
  verify the accepted peer is the connecting socket — a foreign local
  connection is rejected and setup fails closed — then close the
  listener). Hand one end to a dedicated OS thread running
  `ob_rpc_serve_fd` (shim over patched `ggml_backend_rpc_serve_fd`, cache
  disabled in M3); pump bytes 1:1 between the other end and the mesh
  stream. Serve returns when the stream closes → session over, thread
  joins.
- **Head side** (per remote node in the plan): bind a loopback listener on
  127.0.0.1:0 and **keep accepting for the epoch's lifetime**, opening one
  fresh mesh `rpc` stream per accepted connection; register the endpoint
  with `ob_rpc_add_server("127.0.0.1:<port>")` → remote devices. (Amended
  from "accept-once", which failed empirically: the RPC client dials the
  endpoint string repeatedly — a registration probe that closes instantly,
  device-property queries during load, then the buffer/compute connection —
  six sequential connections observed per load. The listener is loopback-
  only, exists only while its epoch is active, and its teardown aborts all
  in-flight pumps.) The RPC
  protocol's own HELLO (v5.1) runs inside; version skew is *pre-empted* by
  our engine-build-hash handshake at mesh connect (identical builds ⇒
  identical RPC protocol), so the silent-vanish failure mode upstream has
  cannot occur between paired OneBrain nodes.
- GGML aborts the client process on a torn RPC stream: the bridge treats
  any stream error as fatal to the *plan* (fail the in-flight request,
  tear down all rpc sessions, re-plan — full resilience lands in M5; M3
  returns a structured error).

## Placement (scheduler v1-lite; full v1 in M4)

Inputs: per-node `usable_memory_bytes` (workers send a proto
`NodeStatus` after Hello: usable memory = measured free of the chosen
device minus a fixed OS reserve; never total RAM), the model's total
weight bytes (GGUF), requested `ctx_len`.

- **Auto-solo short-circuit** (spec §1.4): if `weights + kv_estimate(ctx)`
  ≤ head's usable memory ⇒ `Strategy::Solo`, zero mesh involvement.
  `--nodes N` forces distribution (N=1 forces solo; N > paired+1 is a
  typed error naming both numbers).
- Otherwise pipeline-parallel: contiguous layer ranges proportional to
  usable memory, computed over `[workers..., head]` in that order (head
  last ⇒ head owns the tail layers; the input layer is pinned to CPU by
  the engine and the output head lands on the head node, keeping sampling
  local). Ranges are integral layer counts, largest-remainder rounding,
  minimum 1 layer per participating node; nodes that would round to 0 drop
  out of the plan. KV budget in M3 is a flat 0.85 utilization ceiling per
  node (real per-range KV accounting is M4).
- The plan is expressed to the engine as an explicit `devices[]`
  (worker RPC devices in stage order, then local device) plus
  `tensor_split[]` fractions — llama.cpp's own free-memory probing (a live
  RTT per call against RPC devices) is never allowed to drive placement.
- `--explain`: the plan (or solo decision) is rendered as prose: per node —
  layers, weight MB, the binding constraint, and why distribution did or
  did not engage.

## Epoch lifecycle (M3 scope)

Head: compute plan → `PlanProposal` to each participant → collect
`PlanAck{ready}` (workers pre-open their serve state; nothing downloads in
M3 — weights flow through RPC set_tensor from the head, see ADR 0004) →
epoch active → open rpc streams → `ob_model_load_with_devices`. Any nack
or timeout (15 s) aborts activation with a typed error naming the node.
Workers fence: streams and ops for epochs ≠ active are rejected (close 4).
Epoch teardown (new plan, `stop`, model unload) is role-asymmetric: the
**head frees the model first** — freeing sends remote FREE_BUFFER commands
over the still-standing bridges, and GGML aborts the process on a torn
stream — then closes bridges and streams. Workers keep the contract order:
their serve sessions end when the streams close, then threads join.
(Amended: the original "close streams first" wording was head-unsafe.)

## Engine surface additions (shim + safe wrappers)

- `ob_rpc_serve_fd(fd, n_threads, dev_index)` — serve one session over the
  fd on the calling thread, CPU device by default in M3 (`dev_index` into
  `ob_dev_*` enumeration).
- `ob_rpc_add_server(endpoint) -> int32` returning a server slot handle;
  `ob_rpc_server_device_count(slot)`, plus
  `ob_model_load_devices(path, slots[], n_slots, tensor_split[], n_split,
  use_local_device: bool, n_gpu_layers)` — builds the NULL-terminated
  device array (remote slots' devices in order, then the local device when
  requested) and loads with `split_mode = LAYER`.
- Rust: `RpcSession` (owns the bridge socket + serve thread),
  `RemoteServer` (registered endpoint), `Model::load_distributed(...)`.
  All existing solo paths unchanged.

## Daemon & API

- Internal `POST /api/internal/load` body gains optional
  `"nodes": u32` and `"explain": bool`; the NDJSON stream gains
  `{"status":"planning"}` and `{"status":"plan", "plan": {...,
  "explanation": "..."}}` lines before `loading/ready`. `GET
  /api/internal/status` reports the active plan (epoch, strategy,
  assignments).
- Worker daemons need no new public API: plans arrive over the mesh
  control stream; the engine host gains a `ServeShard` mode driven by it.
- CLI `onebrain run` passes `--nodes`/`--explain` through and renders the
  plan lines; `status` shows the active plan.

## Tests / DoD hooks

- `cargo xtask sim`: N sandboxed daemons on one host (mesh loopback,
  relays/mdns off). Memory caps via config `[debug]
  usable_memory_override_bytes` (test-only knob: the value workers report
  in NodeStatus and the head uses for itself; documented as never touching
  real allocation). Scenarios:
  1. **Distribute**: cap both nodes so the tiny model fits neither alone;
     `run` → plan is PipelineParallel across 2 nodes; both API dialects
     stream; **greedy tokens are byte-identical to a solo uncapped run of
     the same prompt** (the correctness property of §9).
  2. **Auto-solo**: uncapped → plan is Solo, no rpc streams opened
     (assert via status + socket scan).
  3. **Socket scan**: enumerate listening TCP sockets of every daemon pid.
     Non-loopback listeners are forbidden at all times. While a distributed
     session is active, the head's per-epoch loopback bridge listener is
     expected; once the session ends, only the api binds may remain.
  4. `--nodes 2` on the uncapped pair forces distribution; `--explain`
     lines present in both modes.
- Linux CI leg reuses the netem namespace machinery from pair-sim at
  1 Gbit / 0.5 ms.
- Manual two-machine checklists (Mac+Windows, Mac+Linux) documented in
  STATUS.md per the M3 DoD (run when hardware is available).
