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

Upstreaming note: generally useful for anyone embedding the RPC server
behind their own transport/auth (the README itself warns the TCP listener
is insecure). Candidate PR to ggml-org/llama.cpp once OneBrain's usage has
soak time; the patch is additive and should rebase trivially across vendor
bumps. Tripwire on bumps: upstream's own
`static_assert(GGML_OP_COUNT == …)` in `ggml-rpc.h` fires when the op set
changes, forcing a deliberate re-look at the protocol.
