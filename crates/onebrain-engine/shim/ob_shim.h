// OneBrain engine shim: a minimal, stable C ABI over the vendored llama.cpp.
//
// Rust never mirrors llama.cpp structs. This shim is compiled against the
// real vendored headers, so upstream API drift becomes a compile error here
// instead of silent FFI undefined behavior. Only opaque pointers and scalar
// types cross the Rust boundary.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ob_model ob_model;
typedef struct ob_session ob_session;

// ---- process-wide engine lifecycle ----
void ob_backend_init(void);
void ob_backend_free(void);
// Route llama.cpp/ggml logging away from stderr (true = drop everything).
void ob_log_silence(bool silent);
const char * ob_llama_version(void);
const char * ob_system_info(void);

// ---- model ----
// Returns NULL on failure. n_gpu_layers < 0 offloads everything supported.
ob_model * ob_model_load(const char * path, int32_t n_gpu_layers, bool use_mmap);
// Load a model stored as multiple split-GGUF parts (docs/logistics.md
// "Split-GGUF"). `paths` must list every part in split order (part 1 first —
// the `-%05d-of-%05d.gguf` convention); wraps llama_model_load_from_splits,
// so custom part names are accepted as long as the order is right. Same
// params semantics as ob_model_load. Returns NULL on failure (missing part,
// wrong order, or parts from different split sets).
ob_model * ob_model_load_splits(const char ** paths, size_t n_paths,
                                int32_t n_gpu_layers, bool use_mmap);
void ob_model_free(ob_model * m);
int32_t  ob_model_n_layer(const ob_model * m);
int32_t  ob_model_n_embd(const ob_model * m);
int32_t  ob_model_n_ctx_train(const ob_model * m);
uint64_t ob_model_n_params(const ob_model * m);
uint64_t ob_model_size_bytes(const ob_model * m);
// Writes a human-readable description; returns bytes written or -1.
int32_t ob_model_desc(const ob_model * m, char * buf, size_t buf_size);

// ---- tokenization (thread-safe per llama.cpp docs) ----
// Returns token count on success; negative count = required capacity.
int32_t ob_tokenize(const ob_model * m, const char * text, int32_t text_len,
                    int32_t * tokens, int32_t n_tokens_max, bool add_special);
// Renders one token; returns bytes written (no NUL) or negative on failure.
int32_t ob_token_to_piece(const ob_model * m, int32_t token, char * buf, int32_t buf_len);
bool ob_token_is_eog(const ob_model * m, int32_t token);

// ---- compute devices (backend autodetection) ----
// Device type codes: 0 = CPU, 1 = GPU, 2 = integrated GPU, 3 = accelerator,
// 4 = meta; mirrors ggml_backend_dev_type.
int32_t ob_dev_count(void);
// Fills name/description and free/total memory for device i; returns the
// device type code, or -1 if i is out of range.
int32_t ob_dev_info(int32_t i, char * name, size_t name_len,
                    char * desc, size_t desc_len,
                    uint64_t * free_mem, uint64_t * total_mem);

// ---- distributed inference: GGML RPC over caller-owned sockets ----
// (docs/distributed.md, ADR 0004). Sessions are tunneled through
// authenticated mesh streams by the daemon; nothing here binds or listens.

// Serve exactly one GGML RPC session over an already-connected socket
// (Unix fd / Windows SOCKET, widened to long long). Blocks the calling
// thread until the peer closes the connection; the socket is closed before
// returning, so ownership of the handle transfers here. `dev_index` selects
// the serving device from the ob_dev_* enumeration; `n_threads` must be
// >= 1. `cache_dir` (nullable / empty = no cache) points the session at a
// local tensor-cache directory: SET_TENSOR_HASH requests are answered from
// files named by the FNV-1a-64 of the tensor payload (16 lowercase hex
// digits, no extension — see src/rpc_cache.rs), letting a pre-seeded worker
// skip the head's weight push for every tensor over the protocol's 10 MiB
// hash threshold; incoming >threshold SET_TENSOR payloads are also saved
// there. The directory must already exist (the Rust wrapper creates it);
// the C side never creates or evicts — the daemon's reaper owns lifetime
// (docs/logistics.md "RPC tensor-cache pre-seeding"). Returns 0 after a
// served session, or -1 on invalid arguments — the socket is closed on
// that path too, so the bridging peer sees EOF instead of a hang.
int32_t ob_rpc_serve_fd(long long fd, const char * cache_dir,
                        int32_t n_threads, int32_t dev_index);

// Register a remote RPC server endpoint ("host:port"). Returns a slot
// handle >= 0, -1 when the endpoint is unreachable or spoke a different
// protocol (an engine version mismatch is indistinguishable from connect
// failure here; the mesh build-hash handshake pre-empts it), or -2 when the
// slot table (GGML_RPC_MAX_SERVERS) is full. Re-registering an endpoint
// string returns its existing slot; the underlying registration lives for
// the process (upstream caches it), so new epochs should bridge through
// fresh ephemeral ports. NOT thread-safe against itself — the Rust wrapper
// serializes calls.
int32_t ob_rpc_add_server(const char * endpoint);

// Number of devices exposed by a registered server slot, or -1 for an
// invalid slot. Cheap: counts were fetched at registration.
int32_t ob_rpc_server_device_count(int32_t slot);

// Load a model across an explicit device list: each slot's remote devices
// in slots[] order, then — when use_local_device — the best local device
// (GPU, else integrated GPU, else CPU). tensor_split[i] is the layer
// proportion for device i in that order; n_split must equal the total
// device count (NULL/0 lets llama.cpp probe free memory instead, which is
// a live network round trip per remote device — never do that from
// placement code). split_mode is always LAYER; weights are memory-mapped
// locally and pushed to remote devices through the RPC sessions at load.
// Returns NULL on failure.
ob_model * ob_model_load_devices(const char * path,
                                 const int32_t * slots, int32_t n_slots,
                                 const float * tensor_split, int32_t n_split,
                                 bool use_local_device, int32_t n_gpu_layers);

// ob_model_load_devices for a split-GGUF model: identical placement
// contract, but the local weights come from `paths` (every part, in split
// order, as in ob_model_load_splits). n_paths == 1 behaves exactly like
// ob_model_load_devices.
ob_model * ob_model_load_splits_devices(const char ** paths, size_t n_paths,
                                        const int32_t * slots, int32_t n_slots,
                                        const float * tensor_split, int32_t n_split,
                                        bool use_local_device, int32_t n_gpu_layers);

// ---- model metadata & chat template ----
// GGUF metadata value by key; returns length or -1 (see llama_model_meta_val_str).
int32_t ob_model_meta(const ob_model * m, const char * key, char * buf, size_t buf_size);
// Renders a chat through the model's built-in template. Returns the total
// formatted length (call again with a bigger buffer if it exceeds buf_len),
// -1 on template failure, or -2 when the model declares no template.
int32_t ob_chat_apply_template(const ob_model * m,
                               const char ** roles, const char ** contents, size_t n_msgs,
                               bool add_assistant, char * buf, int32_t buf_len);

// ---- session: one llama_context + a sampler chain ----

// Session creation parameters (perf contract, docs/perf.md §2). This struct
// is OWNED BY THE SHIM (not a llama.cpp mirror): sys.rs mirrors it with
// #[repr(C)], and both sides live in this crate, so the two definitions can
// only drift together. Zero-values mean "engine default" for the scalar
// fields noted below; the bool fields carry their value directly, so Rust
// must set them explicitly (its Default mirrors llama.cpp's defaults).
typedef struct ob_session_params {
    uint32_t n_ctx;     // context length
    uint32_t n_batch;   // logical max tokens per llama_decode call
    // Physical micro-batch: llama_decode splits each call into n_ubatch
    // slices (per-slice activations bound distributed transfer cost —
    // docs/perf.md §0/§3). 0 = engine default (512).
    uint32_t n_ubatch;
    // Max concurrent sequences in this context (micro-batched decode,
    // docs/perf.md §6). 0 = engine default (1).
    uint32_t n_seq_max;
    int32_t  n_threads; // <= 0 lets the engine pick
    // Mirrors enum llama_flash_attn_type: -1 AUTO, 0 disabled, 1 enabled.
    int32_t  flash_attn_type;
    // ggml_type codes for the KV cache (0 = F32, 1 = F16, 2 = Q4_0,
    // 8 = Q8_0). llama.cpp's default is F16 for both.
    int32_t  type_k;
    int32_t  type_v;
    // Mirrors enum llama_pooling_type: -1 UNSPECIFIED (the model's own
    // declared pooling; generative models resolve to 0 NONE), 1 MEAN,
    // 2 CLS, 3 LAST. Only meaningful with `embeddings = true`; llama.cpp's
    // default is -1.
    int32_t  pooling_type;
    // One KV buffer shared across sequences: required for the §6 unified-KV
    // admission headroom math. llama.cpp's default is false.
    bool     kv_unified;
    // Offload KQV ops (incl. the KV cache) to GPU. llama.cpp default: true.
    bool     offload_kqv;
    // Extract embeddings from decodes (llama_context_params.embeddings).
    // llama.cpp's default is false; a false session is bit-for-bit the
    // pre-embeddings generation context.
    bool     embeddings;
} ob_session_params;

// Returns NULL on failure (including p == NULL).
ob_session * ob_session_new(ob_model * m, const ob_session_params * p);
void ob_session_free(ob_session * s);
// Decode tokens (appended to the tracked position). 0 = ok; llama_decode
// codes otherwise (1 = no KV slot, 2 = aborted, <0 = error).
// `want_logits = false` requests NO output rows: the KV state advances
// identically but the output head is never gathered or computed. For a
// prompt chunked to n_batch, only the FINAL chunk's logits are ever
// sampled — skipping the rest removes the one hard cross-node sync a
// distributed pipelined prefill would otherwise pay per chunk
// (docs/perf.md §3): the output-row fetch is the only per-chunk command
// that must wait for that chunk's last graph before the next chunk's
// work can be submitted.
int32_t ob_decode(ob_session * s, const int32_t * tokens, int32_t n_tokens,
                  bool want_logits);
// Greedy-sample from the last decoded logits.
int32_t ob_sample_greedy(ob_session * s);

// Clear the session's memory (KV cache) so the next decode starts a fresh
// sequence. Cheaper than recreating the context between requests.
void ob_session_reset(ob_session * s);
// Replace the sampler chain: temp <= 0 selects pure greedy; otherwise
// top-k (k <= 0 disables) -> top-p (p >= 1 disables) -> temp -> dist(seed).
void ob_session_set_sampler(ob_session * s, float temp, float top_p,
                            int32_t top_k, uint32_t seed);
// Sample from the last decoded logits with the configured chain.
int32_t ob_sample(ob_session * s);

// ---- standalone sampler chains (per-sequence sampling) ----
// One chain per generation, owned by the caller instead of the session:
// interleaved sampled sequences each keep their own RNG/chain state, so a
// request behaves identically whether it runs alone or interleaved.
// llama_sampler_chain_init takes only chain params (verified in the pinned
// llama.h) — a chain is context-independent and may be built before any
// session exists and sampled against any context.
typedef struct ob_sampler ob_sampler;

// Build a standalone chain with EXACTLY the ob_session_set_sampler
// construction (one shared builder in the .c, so they cannot drift):
// temp <= 0 selects pure greedy; otherwise top-k (k <= 0 disables) ->
// top-p (p >= 1 disables) -> temp -> dist(seed). Returns NULL on
// allocation failure.
ob_sampler * ob_sampler_new(float temp, float top_p, int32_t top_k, uint32_t seed);
void ob_sampler_free(ob_sampler * smp);
// Reset the chain's internal state (the dist sampler reseeds to its
// creation seed, replaying the draw sequence from the start).
void ob_sampler_reset(ob_sampler * smp);
// ob_sample_ith with THIS chain instead of the session's built-in one:
// sample from the logits of batch token index i of s's most recent decode
// (same index contract; -1 = last output).
int32_t ob_sampler_sample_ith(ob_session * s, ob_sampler * smp, int32_t i);

// ---- explicit multi-sequence batches (docs/perf.md §2) ----
// A reusable token batch with per-token position, sequence id, and logits
// flag — the substrate for KV reuse (§4), speculative verify (§5), and
// micro-batched decode (§6). Thin wrappers over llama_batch_init /
// llama_decode; no policy lives here.
//
// POSITION RULE (upstream-enforced, llama-batch.cpp consistency checks):
// each sequence's positions must stay consecutive — the first position a
// batch adds to a sequence must be exactly seq_pos_max + 1, and positions
// within the batch ascend by 1 per sequence. Rolling back is a real
// ob_memory_seq_rm of the divergent suffix, never a rewound counter.
typedef struct ob_batch ob_batch;

// Allocate a batch holding up to n_tokens_max tokens, each taggable with a
// sequence id in [0, n_seq_max). Returns NULL on invalid args or OOM.
ob_batch * ob_batch_new(int32_t n_tokens_max, int32_t n_seq_max);
void ob_batch_free(ob_batch * b);
// Drop all queued tokens (capacity is retained for reuse).
void ob_batch_clear(ob_batch * b);
// Append one token; `logits` asks the decode to produce logits for this
// position (sample with ob_sample_ith at the index this token got, i.e.
// the batch length before the push). Each token carries exactly one
// sequence id — shared prefixes are expressed via ob_memory_seq_cp, never
// by multi-tagging. Returns false when the batch is full or seq_id is out
// of range.
bool ob_batch_push(ob_batch * b, int32_t token, int32_t pos, int32_t seq_id, bool logits);
int32_t ob_batch_n_tokens(const ob_batch * b);
// Decode the queued tokens in one llama_decode call. Same return codes as
// ob_decode (0 = ok, 1 = no KV slot, 2 = aborted, <0 = error); -1 for an
// empty batch.
int32_t ob_decode_batch(ob_session * s, const ob_batch * b);
// Sample with the session's configured chain from the logits of batch
// token index i (only valid for indexes pushed with logits=true in the
// most recent decode; -1 = last output).
int32_t ob_sample_ith(ob_session * s, int32_t i);

// ---- embeddings (M1 /v1/embeddings) ----
// Only meaningful on a session created with `embeddings = true`; the safe
// Rust wrapper (Session::embed) owns the decode/pooling protocol.

// The session's RESOLVED pooling type (enum llama_pooling_type): the
// creation-time UNSPECIFIED (-1) has been replaced by the model's own
// declared pooling by now (0 NONE for generative models). Lets the caller
// pick the read path before decoding: > 0 means llama pools per sequence
// and ob_embeddings_seq works; 0 means only per-token rows exist.
int32_t ob_session_pooling(const ob_session * s);

// Copy the pooled embedding of `seq_id` from the most recent decode into
// buf (exactly n_embd floats). Returns n_embd on success; -1 when the
// context holds no pooled row for the sequence (pooling NONE, no decode
// yet, or RANK pooling — rerankers emit n_cls_out ranks, not an n_embd
// vector, so they are refused here rather than over-read); -2 when
// buf_len < n_embd (nothing written).
int32_t ob_embeddings_seq(ob_session * s, int32_t seq_id, float * buf, int32_t buf_len);

// Copy the per-token embedding of batch output index `i` from the most
// recent decode into buf (exactly n_embd floats). Same return contract as
// ob_embeddings_seq (-1 covers an invalid index or a context that produced
// no per-token rows).
int32_t ob_embeddings_ith(ob_session * s, int32_t i, float * buf, int32_t buf_len);

// ---- per-sequence KV surgery (docs/perf.md §2) ----
// Thin wrappers over llama_memory_seq_*. p0/p1 follow upstream range
// conventions: p0 < 0 means "from 0", p1 < 0 means "to the end"; the range
// is [p0, p1).
// Remove a position range from one sequence. Returns false when a partial
// range cannot be removed (recurrent/SWA memories); removing a whole
// sequence never fails.
bool ob_memory_seq_rm(ob_session * s, int32_t seq_id, int32_t p0, int32_t p1);
// Copy a position range from one sequence to another (cheap KV sharing).
void ob_memory_seq_cp(ob_session * s, int32_t seq_id_src, int32_t seq_id_dst,
                      int32_t p0, int32_t p1);
// Drop every sequence except seq_id.
void ob_memory_seq_keep(ob_session * s, int32_t seq_id);
// Largest position present for seq_id, or -1 when the sequence is empty.
int32_t ob_memory_seq_pos_max(ob_session * s, int32_t seq_id);

#ifdef __cplusplus
}
#endif
