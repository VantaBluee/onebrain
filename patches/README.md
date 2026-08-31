# Vendor patches

Minimal, additive-only patches applied to `vendor/llama.cpp` at build time
by `crates/onebrain-engine/build.rs` (idempotent: skipped when the patched
symbol already exists). Policy per the product spec §11: never fork —
vendor + pin + minimal patches, each with an upstreaming note.

## 0001-rpc-serve-fd.patch

Adds `ggml_backend_rpc_serve_fd(fd, …)`: serve exactly one GGML RPC session
over an already-connected, caller-owned socket, returning when the peer
closes. Enables embedding the RPC server with **no listener at all** —
OneBrain workers serve inference over sockets bridged 1:1 to authenticated
QUIC streams, so nothing is reachable off-box (spec §1.3/§10). Also exposes
`socket_t::from_fd` and factors backend init out of
`ggml_backend_rpc_start_server` (no behavior change to existing paths).
The signature mirrors `ggml_backend_rpc_start_server` including the
nullable `cache_dir` (upstream's tensor cache); M3 always passed NULL, and
M6 threads it through the shim's `ob_rpc_serve_fd` for tensor-cache
pre-seeding (docs/logistics.md) — no patch change was needed for that.

Upstreaming note: generally useful for anyone embedding the RPC server
behind their own transport/auth (the README itself warns the TCP listener
is insecure). Candidate PR to ggml-org/llama.cpp once OneBrain's usage has
soak time; the patch is additive and should rebase trivially across vendor
bumps. Tripwire on bumps: upstream's own
`static_assert(GGML_OP_COUNT == …)` in `ggml-rpc.h` fires when the op set
changes, forcing a deliberate re-look at the protocol.

## 0002-rpc-client-error-returns.patch

Behavioral patch to the CLIENT paths of `ggml/src/ggml-rpc/ggml-rpc.cpp`
only (the server side and the wire format are untouched): upstream's
`RPC_STATUS_ASSERT` turned any client-side transport failure into
`GGML_ABORT` — a dead worker killed the whole daemon. This patch is the M5
resilience enabler (docs/resilience.md): failures become error returns so a
lost node fails the in-flight request, which the daemon's retry lifecycle
then handles. Applied idempotently by `crates/onebrain-engine/build.rs`
(marker symbol: `ob_rpc_mark_dead` in `ggml-rpc.cpp`).

What it does:

- Adds a client-side dead-socket registry (`ob_rpc_mark_dead` /
  `ob_rpc_is_dead`): the first send/recv failure (or malformed response)
  marks the socket; every later command on it fails fast without touching
  the torn stream. Identity is exact — entries keep a `weak_ptr` so a
  recycled allocation at the same address can never be falsely dead.
- `graph_compute` (both the compute and recompute branches) returns
  `GGML_STATUS_FAILED`, which propagates out of `llama_decode` as `-3` and
  surfaces as `EngineError::Decode`. The cached graph uid is reset on
  failure so a retry never takes the recompute shortcut.
- `get_tensor` zeroes the destination and returns (void signature); the
  socket is now marked dead, so the surrounding decode fails at its next
  remote command. A token sampled from such zeroed logits is never
  delivered: the engine's confirm-before-send loop only streams a token
  after its own decode succeeds (residual: a tear exactly at the FINAL
  budgeted token has no confirming decode — one-token window, Length
  finishes only). `set_tensor` / `memset_tensor` / `buffer_clear` log
  (first failure loud, later ones debug) and return; `cpy_tensor` returns
  false; `init_tensor` returns `GGML_STATUS_FAILED`.
- Alloc/query paths return failure values their callers already handle:
  `alloc_buffer` → nullptr (also covers a null socket), alignment/max-size
  queries → 0 with `ggml_backend_rpc_buffer_type` returning nullptr,
  `get_alloc_size` → local `ggml_nbytes` fallback, device memory → 0/0,
  device count → 0 (registration fails cleanly), HELLO → false.
- Frees tolerate dead sockets silently: `buffer_free_buffer` drops the
  remote round trip on failure (the server frees with the connection) and
  always releases local state — freeing a model over dead bridges neither
  aborts nor hangs (dead sockets short-circuit before any blocking I/O).

Residual abort paths (each unreachable in OneBrain's flows because the
retry design never loads or computes through an already-dead bridge — new
epochs bridge through fresh connections):

1. ~~`ggml_backend_rpc_buffer_get_base` returns nullptr on failure, and
   upstream `ggml_backend_buffer_get_base` (ggml-backend.cpp) asserts
   `base != NULL`.~~ CLOSED by patch 0003: a model can outlive its bridge
   (a tear between requests), and the next session's setup walks existing
   buffers — CI hit exactly this abort. `get_base` now returns a stable,
   well-aligned non-NULL sentinel on a dead socket; an RPC base is a
   remote address used only for pointer arithmetic (never dereferenced
   locally), and every later command on the dead socket fails fast, so
   the failure surfaces as the typed session-creation error.
2. `ggml_backend_rpc_buffer_type` returns nullptr when the
   alignment/max-size queries fail, and several upstream `GGML_ASSERT(buft)`
   call sites (ggml-backend.cpp) would then fire during model LOAD. Left
   as-is deliberately — converting llama.cpp's load pipeline to tolerate
   null buffer types is invasive, and our scheduler only plans loads over
   live, freshly-bridged endpoints.
3. `GGML_ABORT` inside `ggml_backend_rpc_reg_get_device` ("does not have
   enumerated devices") is an API-misuse guard, not a transport path;
   untouched.

Upstreaming note: worth proposing upstream as opt-in behavior — several
upstream issues ask for rpc-client crash tolerance, but upstream may prefer
a `GGML_RPC_NO_ABORT`-style compile flag or a per-connection error callback
over unconditional error returns, and the dead-socket registry would need
their review (it changes retry semantics for long-lived rpc clients like
`llama-server`). Rebase risk on vendor bumps is moderate: the hunks touch
every client entry point, so a bump that reshapes `ggml-rpc.cpp` client
paths forces a deliberate re-port (the build fails loudly via the marker
check + `git apply` if the patch no longer applies).

## 0003-rpc-client-async-pipeline.patch

Client-side async pipelining for the RPC backend (docs/perf.md §3 —
overlapped chunked prefill). Applies on top of 0001 + 0002 (the
`ggml-rpc.cpp` hunks share context with 0002, the additive `ggml-rpc.h`
declaration sits below 0001's `serve_fd`; `build.rs` applies the table in
order). Client paths of `ggml-rpc.cpp`, one scheduler hunk in
`ggml-backend.cpp`, and two hunks in `llama-context.cpp`; the server side
and the wire format are untouched. Applied idempotently by
`crates/onebrain-engine/build.rs` (marker symbol: `ob_rpc_drain_to` in
`ggml-rpc.cpp`).

Mechanism: the RPC protocol already sends `GRAPH_COMPUTE`,
`GRAPH_RECOMPUTE`, `SET_TENSOR`, and `MEMSET_TENSOR` with no response —
submission is fire-and-forget on the wire. What upstream lacked was any way
to *wait* for that work (its `synchronize` was a no-op and events were
unimplemented), so the RPC device advertised `async = false,
events = false` and llama.cpp's `pipeline_parallel` gate could never pass
with an RPC device present: distributed prefill ran one blocking boundary
round trip per ubatch. This patch adds:

- a per-socket pending-command ledger (`ob_rpc_pending`:
  `submitted`/`completed` counters guarded by a per-socket transaction
  mutex) — the client-side pending-ack FIFO, with acks implicit and
  batched (see the correctness argument below). The transaction mutex also
  serializes whole exchanges per socket, so event waits arriving from
  another backend's thread can never interleave bytes on the wire;
- pipelined fences — "ack tickets" (`ob_rpc_fence_begin` /
  `ob_rpc_ticket_pop`): `event_record` puts one `GET_DEVICE_MEMORY`
  REQUEST on the wire immediately and remembers the ledger position; the
  16-byte response stays in the socket buffer until a later wait pops it.
  Because the in-order server answers a response-bearing command only
  after executing everything queued before it — and keeps processing
  afterwards — a popped ticket is a *positional* ack: draining to an old
  event marker never waits for work submitted after that marker. This is
  what keeps a deep ubatch train actually overlapped; a wait-side-only
  fence can never wait for less than the whole queue and re-serializes
  the pipeline to depth one. Tickets are strictly FIFO with the stream,
  so every response-bearing exchange pops all outstanding tickets before
  reading its own response;
- client-side flow control (`ob_rpc_inflight_cap`): the ledger also
  counts wire BYTES, and fire-and-forget sends keep the unproven
  in-flight window at roughly TWO ubatches of traffic (adaptive:
  2.5× the largest command of the current burst + 1 MiB, floor 4 MiB;
  the learned size resets whenever the ledger fully drains, so the
  request-response-paced model load never inflates the cap with weight
  push sizes; `OB_RPC_INFLIGHT_CAP` overrides the floor, `0` disables).
  Rationale, measured the hard way: UNBOUNDED pipelining collapsed on the
  CI netem leg — a client bursting ~4 ubatches (~9 MiB) of unread backlog
  turns the receiver's window reopening into line-rate bursts that
  overflow the bounded netem qdisc (default ~1000 packets ≈ 1.5 MB):
  tail drops, retransmit storms, RTO stalls, overlapped prefill 2.5x
  SLOWER than sequential with huge variance. Two ubatches is exactly deep
  enough that ubatch k+1 streams while ubatch k computes — the entire
  overlap win — and keeps TCP in the same stable regime as the
  sequential baseline (which carries one ~2 MiB burst per ubatch without
  trouble). An over-cap send first pops a ticket, i.e. blocks until the
  server has executed one whole earlier ubatch;
- `synchronize` = drain everything submitted on the backend's socket;
  `event_wait`/`event_synchronize` = drain to the recorded marker
  (tickets first, then — only if the ledger still cannot prove the
  target — one blocking fence round trip);
- async tensor entry points: `set_tensor_async` (a no-response
  `SET_TENSOR`; the data is serialized before returning, so the caller's
  buffer is immediately reusable), `get_tensor_async` (a get MUST observe
  all pending computes on the endpoint — `GET_TENSOR` is response-bearing,
  so the in-order server makes the drain implicit in reading the reply),
  and `cpy_tensor_async` (host→remote becomes a pipelined `SET_TENSOR`,
  gated on a synchronous CPU source backend; same-server copies reuse
  remote `COPY_TENSOR`; anything else falls back to the scheduler's
  synchronized copy path);
- a client-side cache for `GET_ALLOC_SIZE` responses (resolves upstream's
  own `TODO: cache the alloc responses`). The server's answer is a pure
  function of what it deserializes — op, type, shapes, strides, op_params,
  and the same for srcs — so the key is the request bytes with the
  volatile identity fields (client-side addresses) reduced to presence
  bits. Why it matters here: graph allocation issues these queries every
  ubatch (`FLASH_ATTN_EXT` nodes — ~20/ubatch measured), each one a
  response-bearing round trip that must first drain every outstanding
  ticket, i.e. one full pipeline stall per graph. Cached, steady state
  issues no allocation round trips at all. Bounded (4096 entries,
  clear-on-overflow); failed exchanges are never cached;
- `buffer_get_base` on a dead socket returns a stable, well-aligned
  non-NULL sentinel (cached like a real base) instead of NULL, closing a
  patch-0002 gap CI hit: a model can outlive its bridge, and the next
  session's setup walks existing buffers straight into upstream's generic
  `GGML_ASSERT(base != NULL)` abort. An RPC base is a REMOTE address the
  client only ever uses for pointer arithmetic — never dereferenced
  locally — and every later command carrying a sentinel-derived address
  fails fast on the dead socket, so the failure surfaces as the typed
  session-creation error (asserted by
  `session_create_on_torn_model_is_an_error_not_an_abort`). The other
  post-tear escalation sites were audited: `alloc_buffer` → nullptr fails
  graph/context creation cleanly; alignment/max-size live in the cached
  buffer type (never re-queried); `get_alloc_size` falls back to
  `ggml_nbytes`; `init_tensor`'s failed status is tolerated by the
  allocator; device memory 0/0 falls back to host memory in llama;
- device caps flip to `async = true, events = true`, which is exactly what
  `llama_context`'s pipeline-parallel gate checks. With our distributed
  loads (n_devices > 1, split-mode layer, full offload, offload_kqv, no
  tensor overrides) llama.cpp logs
  `llama_context: pipeline parallelism enabled` and schedules up to 4
  in-flight copies per backend;
- `ggml_backend_rpc_pipeline_enable(bool)` (ggml-rpc.h): process-wide
  switch for the caps ADVERTISEMENT, default on — the engine hook behind
  `[perf] prefill_overlap` (surfaced as
  `onebrain_engine::rpc::set_pipeline_overlap`). Disabled, the gate fails
  and schedulers created afterwards run without parallel copies: the
  constructed M3 baseline. Only the advertisement is gated — the ledger,
  drains, and async entry points stay active (correct either way, see
  below), so a live context keeps the mode it was created with; llama
  caches the gate result per context, so a flip affects the next load. A
  deliberate nuance: even disabled, split-boundary input copies still go
  through the (correct) fire-and-forget `cpy_tensor_async` path — the
  baseline restores the M3 *scheduler shape* (no parallel copies, no
  events), which is what the overlap A/B measures. (The alloc-size cache
  is also active either way; it removes identical round trips from both
  sides of the A/B.)

The `ggml-backend.cpp` hunk (scheduler, upstream-worthy on its own): skip
the split-input copy entirely for empty tensors (a zero in `ne`). During
chunked prefill, the ubatches that produce no output rows feed the
final-stage split a zero-row boundary tensor; upstream's synchronizing
copy fallback still fenced the producing backend for that zero-byte copy,
stalling the scheduler once per ubatch and serializing pipeline-parallel
prefill end to end. An empty tensor carries no data and no dependency.

The `llama-context.cpp` hunks:

1. resolve upstream's own TODO at the pipeline_parallel gate ("should we
   ignore ACCEL types too?"): ACCEL devices (BLAS/Accelerate — linked into
   our macOS CPU builds) advertise no async/events and vetoed pipelining
   for the whole context. `ggml_backend_sched` already null-checks
   per-backend events and falls back to blocking synchronization at every
   wait/record site, so an eventless ACCEL backend simply runs its own
   boundaries synchronously while the capable backends still overlap.
2. keep graph REUSE off the pipelined path: under pipeline parallelism the
   reuse branch skips `ggml_backend_sched_alloc_graph` — so the
   scheduler's pipeline copies never rotate — and must fully synchronize
   before `set_inputs` (the in-flight graph reads the very tensors being
   overwritten). A "reused" prefill therefore executes strictly
   sequentially, one ubatch at a time, with the pipeline machinery inert.
   Multi-token (prefill-shaped) ubatches now take the alloc path
   (rotating copies, real overlap); single-token decode keeps graph reuse
   — one ubatch per `llama_decode` has nothing to overlap with, and the
   rebuild would cost decode latency for no gain.

Correctness argument (NO wire change): the RPC server processes commands
strictly serially, in order, per connection. Deferring ack reads therefore
preserves semantics — the response of ANY later response-bearing command on
the same socket proves that every command submitted before it has fully
executed, and a read's own response cannot arrive before prior computes
finished. Ticket responses are ordinary `GET_DEVICE_MEMORY` responses,
read strictly in request order, so request/response pairing stays exact.
Remote write-after-read hazards (the scheduler overwriting an input copy
an in-flight graph may still read) are ordered by the socket itself: the
overwrite is a later command on the same connection.

Failure composition with 0002: a fence or ticket read that fails marks the
socket dead exactly like every other round trip, and the error surfaces as
an error return on whichever status-returning call drains or runs next
(`graph_compute` → `GGML_STATUS_FAILED` fail-fast, `cpy` → false, gets →
zeroed destination). Void paths (`synchronize`, `event_wait`) log loudly
via the 0002 first-failure logger and never abort; teardown with pending
work (tickets included) either drains naturally through the
response-bearing `FREE_BUFFER` round trips (live server) or fails fast
(dead socket) — no hangs either way. The 0002 residuals are unchanged,
including the one-token zeroed-logits window at the final budgeted token.

Upstreaming note: a strong candidate PR — it makes `llama.cpp --rpc`
benefit from upstream's own pipeline parallelism with zero protocol
changes; the implicit-ack ledger, ack tickets, and alloc-size cache are
self-contained, and the empty-copy skip plus both llama-context hunks fix
upstream TODOs/latent issues that bite any async backend, not just RPC.
Upstream review would focus on the fence choice (`GET_DEVICE_MEMORY` as a
semantic no-op vs adding a dedicated lightweight PING command — the
latter is a protocol version bump we deliberately avoided), on the
reuse-vs-pipelining trade in llama-context (upstream may prefer making
reuse rotate copies instead of bypassing it), and on whether
`ggml_backend_rpc_pipeline_enable` should be public API or a build flag
(upstream has no config-knob use case; ours is the measured A/B
baseline). Rebase risk on vendor bumps: moderate — the hunks touch
`send_rpc_cmd` (both overloads), the backend/device interface tables, the
caps struct, the scheduler's input-copy loop, and llama-context's reuse
gate; a bump that reshapes those forces a deliberate re-port (the build
fails loudly via the marker check + `git apply`).
