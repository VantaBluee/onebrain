# Model logistics contract (M6)

Binding contract for spec §6 / §8-M6. Builds on M1's downloader (Range
resume, BLAKE3 manifests) and ADR 0004's deferred pieces.

## Range-level cache (content-addressed)

`<data_dir>/models/<cache_key>/` grows a range store:
`ranges/<start>-<end>` files + `ranges.json`
`{ url, total_size, header_len, ranges: [{start, end, blake3}] }`.
Ranges are TENSOR-ALIGNED (from `GgufHeader::tensor_ranges`), plus one
`0-<data_offset>` header range. A node assigned layers L..R fetches only
the header range + its layers' tensor ranges (HTTP Range, resumable at
range granularity, BLAKE3 per range). A full-file cache entry implicitly
holds every range (offset reads; no duplication). Re-plans reuse ranges on
disk; nothing re-downloads if the bytes exist locally (spec §6).

## P2P sharing (LAN-first, iroh-blobs)

- Each range file is an iroh-blobs blob (BLAKE3-addressed — the range
  manifest hash IS the blob hash). The mesh service hosts a blobs
  provider over the existing endpoint (paired peers only: the provider
  accepts only store-listed EndpointIds, same rule as the mesh ALPN).
- New control message `Message::RangeQuery { model: String }` →
  `Message::RangeInventory { model, total_size, ranges: Vec<(u64, u64,
  [u8;32])> }` (empty = none). Before any WAN byte, a downloader asks
  every Connected peer; ranges available from peers fetch via blobs,
  the rest via WAN. PROTO_VERSION → 4.
- DoD proof: the sim's fake-WAN server counts bytes served; after node A
  holds the model, node B's pull completes with ZERO new WAN bytes.

## RPC tensor-cache pre-seeding (ADR 0004 payoff)

- Shim: `ob_rpc_serve_fd` gains `const char * cache_dir` (nullable).
- Workers, on plan adoption, pre-seed `<data_dir>/rpc-cache/` with files
  named by FNV-1a-64 of each ≥10 MiB assigned tensor payload (Rust FNV
  impl over the range bytes; the GGUF layout gives payload = range).
  Integrity remains BLAKE3-at-download; FNV is only the RPC protocol's
  lookup key. Serve sessions then answer `SET_TENSOR_HASH` from cache and
  the head's push skips those transfers. Reaper: rpc-cache LRU-capped at
  `config.rpc_cache_max_bytes` (default 20 GiB), never evicting entries
  for the active epoch.

## Split-GGUF

`hf:` and registry refs may name `-00001-of-000NN.gguf` sets: the
downloader derives all part names (`llama_split_path` convention),
fetches each (full or ranged), and the engine loads via a new
`ob_model_load_splits` (wraps `llama_model_load_from_splits`) whenever a
cache entry has >1 part. `onebrain ls` shows parts as one model.

## LRU GC + pinning

`cache.rs`: per-entry `last_used_unix` (touched on load) + `pinned: bool`
in the entry manifest. `onebrain pin <model>` / `onebrain unpin <model>`
(new CLI verbs); `onebrain ls` shows pin + age. GC triggers after every
download when `config.cache_max_bytes` (default: unlimited, 0 = disabled)
is exceeded: evict LRU entries, never pinned and never currently loaded;
each eviction logged with the freed bytes.

## Registry v1 (curated models.toml)

Entries verified live (URL 200 + license recorded) at implementation
time: Qwen3-4B, Qwen3-32B, Qwen3-30B-A3B (MoE: total/active params
fields), GLM-4.5-Air, gpt-oss-120b, DeepSeek-R1-Distill (Qwen 7B + 14B),
plus the existing small entries. Llama-3.3-70B only if a license-gate-free
mirror is verifiable; otherwise documented as skipped (our downloader
sends `Authorization: Bearer $HF_TOKEN` when the env var is set — add
that passthrough). New optional fields: `moe_total_params`,
`moe_active_params`, `parts` (split count), `min_pooled_memory_mb`,
`recommended_ctx`.

## DoD hooks

- Unit/integration (onebrain-models, local servers, no WAN): range fetch
  assembles byte-exact vs full download; interrupted RANGE download
  resumes byte-exact; corrupt range caught by BLAKE3 and refetched;
  split-GGUF part naming; LRU GC evicts oldest-unpinned only; FNV matches
  a hand-computed vector.
- Sim additions: (a) zero-WAN proof — A pulls from the counting fake-WAN
  server, B pairs then pulls the same model, assert server byte counter
  unchanged and B's manifest byte-exact; (b) distributed load after
  pre-seed: second load of the same plan logs skipped transfers (assert
  via daemon log grep for the skip counter line the daemon emits).
- CI: rides the existing test-job steps (models tests + sim).
