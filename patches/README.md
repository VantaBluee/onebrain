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

1. `ggml_backend_rpc_buffer_get_base` returns nullptr on failure, and
   upstream `ggml_backend_buffer_get_base` (ggml-backend.cpp) asserts
   `base != NULL`. Only reachable at load/alloc time: the base pointer is
   cached in the buffer context on first use, so mid-generation this RPC
   never happens.
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
