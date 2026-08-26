# Internal control API and daemon runtime contract (M1)

This is the contract between the daemon (`onebraind`), the CLI, and the HTTP
gateway. Public dialects (OpenAI `/v1/*`, Ollama `/api/*`) are specified by
their upstreams; this file covers what OneBrain adds around them.

## Files (under the platform dirs from `onebraind::paths`)

Env override: when `ONEBRAIN_HOME` is set, config lives in
`$ONEBRAIN_HOME/config` and data in `$ONEBRAIN_HOME/data` — used by tests
and the e2e harness so they never touch real user state.

| Path | Contents |
|---|---|
| `<config_dir>/api-token` | 64 lowercase hex chars (32 random bytes). Created at first daemon start. The CLI reads it directly (same user); API clients get it from `onebrain status`. |
| `<config_dir>/config.toml` | See `onebraind::config::Config`. M1 adds `api_bind` (default `"127.0.0.1:11435"`, loopback-only in M1) and `ctx_len` (default 4096). |
| `<data_dir>/run/daemon.lock` | fs4 exclusive lock, held for the daemon's lifetime. Kill -9 releases it via the OS — never trust a pid file over this lock. |
| `<data_dir>/run/daemon.json` | `{ "pid": u32, "port": u16, "started_unix": u64, "version": "0.1.0" }` — written after the listener binds; informational only (the lock is the liveness authority). |
| `<data_dir>/models/` | Model cache: `<id>/<file>.gguf` + `<id>/manifest.json` (`{ "url", "size_bytes", "blake3" }`). |

## Daemon lifecycle

- `onebrain up` → if the lock is free, spawn `onebrain __daemon` **detached**
  (Windows: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`; Unix: fork-style
  spawn + `setsid`), then poll `GET /api/internal/status` (with token) until
  healthy (10 s timeout). If the lock is held: report the existing endpoint.
- `onebrain __daemon` (hidden subcommand) → acquire lock (exit 1 with a
  clear message if held), init token, bind `api_bind`, write `daemon.json`,
  start the engine-host thread, serve until shutdown.
- `onebrain stop` → `POST /api/internal/shutdown`, wait for the lock to
  free (5 s), report. SIGTERM/console-close does a graceful shutdown too.
- Kill -9 then `onebrain up` must start cleanly (stale `daemon.json` is
  overwritten; the dead process's lock is already gone).

## Internal endpoints (mounted under `/api/internal`, ALWAYS token-auth'd —
the localhost exemption does **not** apply to these)

- `GET  /api/internal/status` →
  `{ "version", "engine_build", "port", "uptime_secs",
     "model": null | { "name", "size_bytes", "n_layer", "n_ctx" } }`
- `POST /api/internal/load` body `{ "model": "<ref>" }` → **NDJSON stream**:
  `{"status":"downloading","completed":u64,"total":u64}` (repeated) →
  `{"status":"loading"}` → `{"status":"ready","model":{…as status…}}`,
  or `{"status":"error","message":"…"}` as the terminal line.
  One model loaded at a time; loading a second unloads the first.
- `POST /api/internal/shutdown` → `200 {"status":"stopping"}` then exit.

## Model references

`<id>` from the embedded registry (`onebrain-models/models.toml`), or
`hf:<org>/<repo>/<file.gguf>` (direct Hugging Face fetch), or an absolute
local path to a `.gguf` (no copy; loaded in place, listed as `local:<name>`).

## Engine host (inside the daemon)

One OS thread owning `Model` + one `Session` (created at load with
`config.ctx_len`, `n_batch = 512`). Jobs arrive on a channel and run
serially (queueing is the M1 concurrency model). Per job:
`session.reset()` → sampler from `GenParams` (OpenAI `temperature`/`top_p`
map directly; `seed` honored when given) → chat jobs render through
`Model::apply_chat_template` (fallback when the model has none:
`"<|role|>\n{content}\n"` per turn plus `"<|assistant|>\n"`, and log that
the fallback was used) → stream pieces as `TokenEvent::Token`.

Stop strings: accumulate output; when any stop string matches the
accumulated text, finish with `FinishKind::Stop` **without sending** the
piece that completed the match. Prompt longer than `ctx_len - max_tokens`
→ `ApiError::BadRequest` naming both numbers.

## Public dialect notes

- OpenAI: SSE (`data: {json}\n\n`, terminated by `data: [DONE]\n\n`) when
  `"stream": true`; plain JSON otherwise. `usage` filled from `DoneStats`.
- Ollama: **NDJSON**, not SSE (`{"model","created_at","response","done":false}`
  lines; final line has `"done":true` plus counts). `/api/pull` streams the
  same download-progress NDJSON as internal load (download only, no load).
- Both dialects must work by pointing an unmodified client at
  `http://127.0.0.1:11435` (Ollama clients) / `http://127.0.0.1:11435/v1`
  (OpenAI clients).
