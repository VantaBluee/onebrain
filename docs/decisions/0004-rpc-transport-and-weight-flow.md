# 0004 — RPC-over-mesh transport and M3 weight flow

Date: 2026-08-26 · Status: accepted

## Context

M3 tunnels GGML RPC sessions through the authenticated mesh. Recon of the
vendored tree established: (1) the RPC server's only entry point binds and
listens on TCP with no fd-injection seam (`socket_t`'s fd constructor is
private, `rpc_serve_client` is file-static); (2) the RPC client pushes all
weights to workers via `SET_TENSOR` from the model file the *client*
mmaps — workers never read model bytes from disk themselves; (3) the
protocol trusts peer-supplied pointers and aborts the client process on a
torn stream, which is precisely why upstream documents it as unsafe on
open networks.

## Decision 1: patch the vendor (additive serve-fd entry point)

`patches/0001-rpc-serve-fd.patch` (+84 lines, additive only) adds
`ggml_backend_rpc_serve_fd`: one RPC session over a caller-owned socket,
returning on close. Applied idempotently by the engine build script.

Rejected alternative — loopback listener bridged to the mesh with no
patch: leaves an unauthenticated RPC endpoint (a protocol that trusts
raw pointers) reachable by any local process for the daemon's lifetime,
has no clean shutdown (the serve loop never returns), and adds a
listener the socket-scan test exists to forbid. The patch is smaller than
the bridge workaround and strictly safer. Upstreaming note in
`patches/README.md`.

The head-side client keeps the unpatched path: it *connects out* to a
loopback socket our own process created and accepted exactly once —
client-side exposure is one intra-process accept race on the trusting
party's machine (documented residual, threat model).

## Decision 2: M3 ships head-push weight flow; shard-only fetch moves to M6

The M3 milestone text asks for "per-node shard-only weight fetch". GGML
RPC's architecture is the opposite: the head reads the model file and
pushes every remote tensor over the wire. Making workers load their own
shards from disk would require invasive llama.cpp surgery (loading a
model whose tensors are absent locally), against the §11 no-fork rule.

M3 therefore ships: head holds the full GGUF; workers receive weights
over the authenticated stream at load; correctness and placement are
unaffected. The §6 economics ("never re-download if bytes exist in the
cluster") are delivered in M6 via the RPC tensor cache: workers pre-seed
`cache_dir` files keyed by the protocol's FNV-1a hash — computable on the
worker from range-fetched bytes plus the GGUF layout — so `SET_TENSOR_HASH`
skips the transfer for every ≥10 MiB tensor on reload and re-shard. The
FNV hash is not collision-resistant; integrity continues to come from our
BLAKE3 manifests at download time, and the RPC cache stays disabled until
M6 wires the reaper and pre-seeding.

## Consequences

- Vendor bumps must re-apply/rebase the patch (tracked; additive-only so
  conflicts are unlikely; upstream's own GGML_OP_COUNT static_assert
  tripwires protocol drift).
- A torn mesh stream during distributed inference is fatal to the plan
  (structured error + teardown in M3; transparent retry arrives in M5).
- Worker disk stays empty in M3 (no shard caches to GC until M6).
