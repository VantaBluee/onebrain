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
// Returns NULL on failure. n_threads <= 0 lets the engine pick.
ob_session * ob_session_new(ob_model * m, uint32_t n_ctx, uint32_t n_batch, int32_t n_threads);
void ob_session_free(ob_session * s);
// Decode tokens (appended to the tracked position). 0 = ok; llama_decode
// codes otherwise (1 = no KV slot, 2 = aborted, <0 = error).
int32_t ob_decode(ob_session * s, const int32_t * tokens, int32_t n_tokens);
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

#ifdef __cplusplus
}
#endif
