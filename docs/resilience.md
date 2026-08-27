# Resilience contract (M5)

Binding contract for spec §5 / §8-M5. Builds on M3/M4's epochs, mesh peer
states, and plan machinery.

## The enabler: RPC failures must be errors, not aborts

The vendored RPC client aborts the whole process (`GGML_ABORT` via
`RPC_STATUS_ASSERT`, plus null-buft asserts) on any transport failure —
incompatible with "structured mid-request failure". Vendor patch
`patches/0002-rpc-client-error-returns.patch` (additive/behavioral,
upstreaming note required) converts client-side failure paths to error
returns:

- `graph_compute` → `GGML_STATUS_FAILED` on send/recv failure (propagates
  out of `llama_decode` as a fatal-error return code, which the engine
  already maps to `EngineError::Decode`).
- buffer-type/alloc/init/memory queries → nullptr/false/zeroed returns
  where signatures allow; llama.cpp's own null-checks then fail model
  load cleanly instead of asserting where possible. Where a null return
  would hit an upstream assert we cannot reach (e.g.
  `ggml_backend_buft_alloc_buffer`), the patched path may still leave an
  abort — DOCUMENT each such residual in the patch README; the retry
  design below never re-uses a torn session, so residuals should be
  unreachable in our flows.
- Frees/releases during teardown tolerate dead sockets silently (retry
  frees local memory even when workers are gone).

Every hunk stays inside `ggml-rpc.cpp` client paths; the server side and
wire format are untouched.

## Failure lifecycle (head)

1. **Detect**: mesh peer transitions (new `MeshHandle::peer_events()`
   consumer stream: `PeerEvent { peer, state }` on every state change)
   and/or an `EngineError::Decode` from a generation against an active
   distributed epoch. Either marks the epoch **failed**.
2. **Fail structured**: the in-flight request's engine job ends; the
   daemon does NOT surface it yet — it enters the retry path with the
   original prompt tokens and the pieces already streamed to the client.
3. **Re-plan**: tear down the failed epoch (patched frees tolerate dead
   bridges; serve threads on live workers end with their streams). Plan
   again from live Connected peers (dead/suspect/draining excluded). If
   nothing fits pooled: emit the typed error to the client
   (`the node '<name>' was lost mid-generation and the remaining nodes
   cannot hold the model (needs X MB, have Y MB); reconnect the node or
   choose a smaller model`) and mark the model unloaded.
4. **Retry once, transparently**: reload on the new plan, re-prefill
   `prompt_tokens + already_generated_tokens` (greedy determinism makes
   this exact for temp<=0; for sampled runs the seed chain restarts —
   documented, acceptable), continue sampling, and keep streaming into
   the SAME client response (already-sent pieces are never re-sent). One
   retry per request; a second failure surfaces the typed error.
5. **Rejoin**: a peer returning to Connected while a distributed model is
   active schedules a lazy re-plan: when the engine host is idle (no
   queued jobs), the daemon re-plans with the full peer set and swaps to
   the new epoch (proposal/ack/load as usual). In-flight/queued requests
   always finish on the old epoch first.

## Worker-side drain & shutdown

`onebrain stop` on a worker with an active shard: send proto `Draining`
to the epoch's head over control, give in-flight serve traffic a 3 s
grace, then continue normal teardown. The head treats `Draining` exactly
like a death for planning (exclude from new plans; trigger the failure
lifecycle if a request is in flight) but logs it as polite.

## Power realities (per-OS, behind traits)

`onebraind::power` defines:

- `trait SleepInhibitor { fn hold(&mut self, why: &str); fn release(&mut self); }`
  — held while the engine host has any loaded model AND (a request is in
  flight OR a distributed epoch is active); released when idle-with-no-
  epoch. Impls: Windows `SetThreadExecutionState(ES_CONTINUOUS |
  ES_SYSTEM_REQUIRED)` (direct kernel32 extern, no new deps); macOS
  `IOPMAssertionCreateWithName` (IOKit framework externs); Linux: hold a
  `systemd-inhibit --what=sleep --who=onebrain --why=... sleep infinity`
  child (kill to release; absent systemd-inhibit = warn once, no-op).
- `trait BatteryProbe { fn level_percent(&self) -> Option<u8>; fn on_ac(&self) -> Option<bool>; }`
  — Windows `GetSystemPowerStatus` extern; macOS `pmset -g batt` parse;
  Linux `/sys/class/power_supply/*/capacity` + `status`. Desktops report
  `None` (never draining).
- Battery policy: below `config.battery_drain_threshold` (default 25) and
  not on AC → `NodeStatus.draining = true` (proto: add the field,
  **PROTO_VERSION → 3**) → the scheduler excludes draining nodes from new
  plans unless the plan is infeasible without them (then include, and say
  so in the explanation).
- All impls unit-tested behind the traits with mocks; the OS calls
  themselves are smoke-tested only on their own OS (cfg'd tests).

## Sim / DoD hooks (extend `cargo xtask sim` — chaos section)

1. **Kill mid-generation, retry succeeds**: 3 daemons (A head, B+C
   workers), caps such that any 2 of the 3 hold the model. Start a
   distributed generation with a long `max_tokens`; kill -9 one worker
   mid-stream; assert the SAME HTTP stream completes with the full token
   count, the final text equals an uninterrupted 2-node control run
   (greedy), and status shows a NEW epoch excluding the dead node.
2. **Kill with no fallback, typed error**: 2 daemons, caps where both are
   required; kill the worker mid-stream; assert the stream ends with the
   structured error naming the lost node and both MB numbers, and the
   daemon stays healthy (solo loads still work afterwards).
3. **Rejoin new epoch**: restart the killed worker; assert the head
   reaches a NEW epoch including it (lazy re-plan) without any client
   activity beyond a status poll loop.
4. **Drain**: `onebrain stop` on a worker mid-idle-epoch: head excludes
   it from the next plan and logs the polite drain.
Battery/sleep paths are unit-tested (mock traits), not simmed.
