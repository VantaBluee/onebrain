// See ob_shim.h for the contract. Compiled against the vendored llama.cpp
// headers by build.rs; any upstream API change fails here at compile time.
#include "ob_shim.h"

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#else
#include <unistd.h>
#endif

struct ob_model {
    struct llama_model * model;
    const struct llama_vocab * vocab;
};

struct ob_session {
    struct llama_context * ctx;
    struct llama_sampler * sampler;
    // llama_batch_get_one wants a mutable token pointer; keep a growable
    // scratch copy so callers can pass const slices.
    int32_t * scratch;
    int32_t   scratch_cap;
    // All-zero per-token output flags for want_logits = false decodes
    // (llama treats a NULL logits array as "output the last token").
    int8_t  * scratch_logits;
};

static void ob_null_logger(enum ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) text;
    (void) user;
}

void ob_backend_init(void) {
    // Loads dynamic backends when the build uses GGML_BACKEND_DL; harmless
    // no-op for fully static builds.
    ggml_backend_load_all();
    llama_backend_init();
}

void ob_backend_free(void) {
    llama_backend_free();
}

void ob_log_silence(bool silent) {
    llama_log_set(silent ? ob_null_logger : NULL, NULL);
}

const char * ob_llama_version(void) {
    return llama_version();
}

const char * ob_system_info(void) {
    return llama_print_system_info();
}

// Wrap a freshly loaded llama_model in the shim handle; frees the model on
// allocation failure so no load path can leak it.
static ob_model * ob_wrap_model(struct llama_model * model) {
    if (model == NULL) {
        return NULL;
    }
    ob_model * m = calloc(1, sizeof(ob_model));
    if (m == NULL) {
        llama_model_free(model);
        return NULL;
    }
    m->model = model;
    m->vocab = llama_model_get_vocab(model);
    return m;
}

ob_model * ob_model_load(const char * path, int32_t n_gpu_layers, bool use_mmap) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    if (!use_mmap) {
        params.load_mode = LLAMA_LOAD_MODE_NONE;
    }

    return ob_wrap_model(llama_model_load_from_file(path, params));
}

ob_model * ob_model_load_splits(const char ** paths, size_t n_paths,
                                int32_t n_gpu_layers, bool use_mmap) {
    if (paths == NULL || n_paths == 0) {
        return NULL;
    }
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    if (!use_mmap) {
        params.load_mode = LLAMA_LOAD_MODE_NONE;
    }

    return ob_wrap_model(llama_model_load_from_splits(paths, n_paths, params));
}

void ob_model_free(ob_model * m) {
    if (m == NULL) {
        return;
    }
    llama_model_free(m->model);
    free(m);
}

int32_t ob_model_n_layer(const ob_model * m) {
    return llama_model_n_layer(m->model);
}

int32_t ob_model_n_embd(const ob_model * m) {
    return llama_model_n_embd(m->model);
}

int32_t ob_model_n_ctx_train(const ob_model * m) {
    return llama_model_n_ctx_train(m->model);
}

uint64_t ob_model_n_params(const ob_model * m) {
    return llama_model_n_params(m->model);
}

uint64_t ob_model_size_bytes(const ob_model * m) {
    return llama_model_size(m->model);
}

int32_t ob_model_desc(const ob_model * m, char * buf, size_t buf_size) {
    return llama_model_desc(m->model, buf, buf_size);
}

int32_t ob_tokenize(const ob_model * m, const char * text, int32_t text_len,
                    int32_t * tokens, int32_t n_tokens_max, bool add_special) {
    return llama_tokenize(m->vocab, text, text_len, tokens, n_tokens_max,
                          add_special, /*parse_special=*/true);
}

int32_t ob_token_to_piece(const ob_model * m, int32_t token, char * buf, int32_t buf_len) {
    return llama_token_to_piece(m->vocab, token, buf, buf_len, /*lstrip=*/0,
                                /*special=*/false);
}

bool ob_token_is_eog(const ob_model * m, int32_t token) {
    return llama_vocab_is_eog(m->vocab, token);
}

ob_session * ob_session_new(ob_model * m, const ob_session_params * p) {
    if (p == NULL) {
        return NULL;
    }
    // Start from llama.cpp's defaults and override only what
    // ob_session_params carries: a zero scalar keeps the engine default, so
    // the widened struct with default values reproduces the pre-widening
    // session bit-for-bit (docs/perf.md §2).
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = p->n_ctx;
    params.n_batch = p->n_batch;
    if (p->n_ubatch > 0) {
        params.n_ubatch = p->n_ubatch;
    }
    if (p->n_seq_max > 0) {
        params.n_seq_max = p->n_seq_max;
    }
    if (p->n_threads > 0) {
        params.n_threads = p->n_threads;
        params.n_threads_batch = p->n_threads;
    }
    params.flash_attn_type = (enum llama_flash_attn_type) p->flash_attn_type;
    params.type_k = (enum ggml_type) p->type_k;
    params.type_v = (enum ggml_type) p->type_v;
    // Embeddings surface (M1 /v1/embeddings): both values default to
    // llama.cpp's own (-1 UNSPECIFIED / false), so a params struct that
    // does not opt in reproduces the pre-embeddings context bit-for-bit.
    params.pooling_type = (enum llama_pooling_type) p->pooling_type;
    params.embeddings = p->embeddings;
    params.kv_unified = p->kv_unified;
    params.offload_kqv = p->offload_kqv;

    struct llama_context * ctx = llama_init_from_model(m->model, params);
    if (ctx == NULL) {
        return NULL;
    }

    struct llama_sampler * sampler =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (sampler == NULL) {
        llama_free(ctx);
        return NULL;
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    ob_session * s = calloc(1, sizeof(ob_session));
    if (s == NULL) {
        llama_sampler_free(sampler);
        llama_free(ctx);
        return NULL;
    }
    s->ctx = ctx;
    s->sampler = sampler;
    return s;
}

void ob_session_free(ob_session * s) {
    if (s == NULL) {
        return;
    }
    llama_sampler_free(s->sampler);
    llama_free(s->ctx);
    free(s->scratch);
    free(s->scratch_logits);
    free(s);
}

int32_t ob_decode(ob_session * s, const int32_t * tokens, int32_t n_tokens,
                  bool want_logits) {
    if (n_tokens <= 0) {
        return -1;
    }
    if (n_tokens > s->scratch_cap) {
        int32_t * grown = realloc(s->scratch, (size_t) n_tokens * sizeof(int32_t));
        if (grown == NULL) {
            return -1;
        }
        int8_t * grown_logits =
            realloc(s->scratch_logits, (size_t) n_tokens * sizeof(int8_t));
        if (grown_logits == NULL) {
            // scratch already grew; keep the larger buffer but do not
            // advance the cap past what BOTH buffers can hold.
            s->scratch = grown;
            return -1;
        }
        memset(grown_logits, 0, (size_t) n_tokens * sizeof(int8_t));
        s->scratch = grown;
        s->scratch_logits = grown_logits;
        s->scratch_cap = n_tokens;
    }
    memcpy(s->scratch, tokens, (size_t) n_tokens * sizeof(int32_t));

    struct llama_batch batch = llama_batch_get_one(s->scratch, n_tokens);
    if (!want_logits) {
        // An explicit all-zero flag array = no output rows at all; llama
        // treats the NULL default as "output the last token".
        batch.logits = s->scratch_logits;
    }
    return llama_decode(s->ctx, batch);
}

int32_t ob_sample_greedy(ob_session * s) {
    return llama_sampler_sample(s->sampler, s->ctx, -1);
}

void ob_session_reset(ob_session * s) {
    llama_memory_clear(llama_get_memory(s->ctx), /*data=*/true);
    llama_sampler_reset(s->sampler);
}

// The ONE chain construction, shared by the session chain and standalone
// ob_sampler objects so the two can never drift: temp <= 0 is pure greedy;
// otherwise top-k -> top-p -> temp -> dist(seed).
static struct llama_sampler * ob_build_chain(float temp, float top_p,
                                             int32_t top_k, uint32_t seed) {
    struct llama_sampler * chain =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (chain == NULL) {
        return NULL;
    }
    if (temp <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        if (top_k > 0) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
        }
        if (top_p < 1.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
        }
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }
    return chain;
}

void ob_session_set_sampler(ob_session * s, float temp, float top_p,
                            int32_t top_k, uint32_t seed) {
    struct llama_sampler * chain = ob_build_chain(temp, top_p, top_k, seed);
    if (chain == NULL) {
        return; // keep the existing chain rather than crash
    }
    llama_sampler_free(s->sampler);
    s->sampler = chain;
}

// ---- standalone sampler chains (per-sequence sampling) ----

struct ob_sampler {
    struct llama_sampler * chain;
};

ob_sampler * ob_sampler_new(float temp, float top_p, int32_t top_k, uint32_t seed) {
    struct llama_sampler * chain = ob_build_chain(temp, top_p, top_k, seed);
    if (chain == NULL) {
        return NULL;
    }
    ob_sampler * smp = calloc(1, sizeof(ob_sampler));
    if (smp == NULL) {
        llama_sampler_free(chain);
        return NULL;
    }
    smp->chain = chain;
    return smp;
}

void ob_sampler_free(ob_sampler * smp) {
    if (smp == NULL) {
        return;
    }
    llama_sampler_free(smp->chain);
    free(smp);
}

void ob_sampler_reset(ob_sampler * smp) {
    llama_sampler_reset(smp->chain);
}

int32_t ob_sampler_sample_ith(ob_session * s, ob_sampler * smp, int32_t i) {
    return llama_sampler_sample(smp->chain, s->ctx, i);
}

int32_t ob_sample(ob_session * s) {
    return llama_sampler_sample(s->sampler, s->ctx, -1);
}

// ---- explicit multi-sequence batches (docs/perf.md §2) ----

struct ob_batch {
    struct llama_batch batch;
    int32_t capacity;
    int32_t n_seq_max;
};

ob_batch * ob_batch_new(int32_t n_tokens_max, int32_t n_seq_max) {
    if (n_tokens_max <= 0 || n_seq_max <= 0) {
        return NULL;
    }
    ob_batch * b = calloc(1, sizeof(ob_batch));
    if (b == NULL) {
        return NULL;
    }
    // llama_batch_init allocates every per-token array (token/pos/n_seq_id/
    // seq_id/logits) at n_tokens_max and leaves n_tokens = 0.
    b->batch = llama_batch_init(n_tokens_max, /*embd=*/0, n_seq_max);
    b->capacity = n_tokens_max;
    b->n_seq_max = n_seq_max;
    return b;
}

void ob_batch_free(ob_batch * b) {
    if (b == NULL) {
        return;
    }
    llama_batch_free(b->batch);
    free(b);
}

void ob_batch_clear(ob_batch * b) {
    b->batch.n_tokens = 0;
}

bool ob_batch_push(ob_batch * b, int32_t token, int32_t pos, int32_t seq_id, bool logits) {
    if (b->batch.n_tokens >= b->capacity || seq_id < 0 || seq_id >= b->n_seq_max) {
        return false;
    }
    const int32_t i = b->batch.n_tokens;
    b->batch.token[i] = token;
    b->batch.pos[i] = pos;
    // Exactly one sequence id per token: shared prefixes go through
    // ob_memory_seq_cp instead of multi-tagging, so rollback (seq_rm) never
    // has to reason about tokens co-owned by other sequences.
    b->batch.n_seq_id[i] = 1;
    b->batch.seq_id[i][0] = seq_id;
    b->batch.logits[i] = logits ? 1 : 0;
    b->batch.n_tokens = i + 1;
    return true;
}

int32_t ob_batch_n_tokens(const ob_batch * b) {
    return b->batch.n_tokens;
}

int32_t ob_decode_batch(ob_session * s, const ob_batch * b) {
    if (b->batch.n_tokens <= 0) {
        return -1;
    }
    return llama_decode(s->ctx, b->batch);
}

int32_t ob_sample_ith(ob_session * s, int32_t i) {
    return llama_sampler_sample(s->sampler, s->ctx, i);
}

// ---- embeddings (M1 /v1/embeddings) ----

int32_t ob_session_pooling(const ob_session * s) {
    return (int32_t) llama_pooling_type(s->ctx);
}

// Shared copy-out for both embedding readers: full n_embd rows only, no
// partial writes (a truncated embedding is silently wrong, not smaller).
static int32_t ob_copy_embd(const ob_session * s, const float * emb,
                            float * buf, int32_t buf_len) {
    const int32_t n_embd = llama_model_n_embd(llama_get_model(s->ctx));
    if (buf == NULL || buf_len < n_embd || n_embd <= 0) {
        return -2;
    }
    if (emb == NULL) {
        return -1;
    }
    memcpy(buf, emb, (size_t) n_embd * sizeof(float));
    return n_embd;
}

int32_t ob_embeddings_seq(ob_session * s, int32_t seq_id, float * buf, int32_t buf_len) {
    // RANK pooling emits float[n_cls_out] (reranker scores), not an n_embd
    // vector: copying n_embd floats would over-read. Refuse instead —
    // OneBrain never creates RANK contexts, so this only guards models
    // that declare RANK themselves.
    if (llama_pooling_type(s->ctx) == LLAMA_POOLING_TYPE_RANK) {
        return -1;
    }
    return ob_copy_embd(s, llama_get_embeddings_seq(s->ctx, seq_id), buf, buf_len);
}

int32_t ob_embeddings_ith(ob_session * s, int32_t i, float * buf, int32_t buf_len) {
    return ob_copy_embd(s, llama_get_embeddings_ith(s->ctx, i), buf, buf_len);
}

// ---- per-sequence KV surgery (docs/perf.md §2) ----

bool ob_memory_seq_rm(ob_session * s, int32_t seq_id, int32_t p0, int32_t p1) {
    return llama_memory_seq_rm(llama_get_memory(s->ctx), seq_id, p0, p1);
}

void ob_memory_seq_cp(ob_session * s, int32_t seq_id_src, int32_t seq_id_dst,
                      int32_t p0, int32_t p1) {
    llama_memory_seq_cp(llama_get_memory(s->ctx), seq_id_src, seq_id_dst, p0, p1);
}

void ob_memory_seq_keep(ob_session * s, int32_t seq_id) {
    llama_memory_seq_keep(llama_get_memory(s->ctx), seq_id);
}

int32_t ob_memory_seq_pos_max(ob_session * s, int32_t seq_id) {
    return llama_memory_seq_pos_max(llama_get_memory(s->ctx), seq_id);
}

int32_t ob_dev_count(void) {
    return (int32_t) ggml_backend_dev_count();
}

static void ob_copy_str(char * dst, size_t dst_len, const char * src) {
    if (dst == NULL || dst_len == 0) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    size_t n = strlen(src);
    if (n >= dst_len) {
        n = dst_len - 1;
    }
    memcpy(dst, src, n);
    dst[n] = '\0';
}

int32_t ob_dev_info(int32_t i, char * name, size_t name_len,
                    char * desc, size_t desc_len,
                    uint64_t * free_mem, uint64_t * total_mem) {
    if (i < 0 || (size_t) i >= ggml_backend_dev_count()) {
        return -1;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_get((size_t) i);
    ob_copy_str(name, name_len, ggml_backend_dev_name(dev));
    ob_copy_str(desc, desc_len, ggml_backend_dev_description(dev));
    size_t free_b = 0;
    size_t total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    if (free_mem != NULL) {
        *free_mem = (uint64_t) free_b;
    }
    if (total_mem != NULL) {
        *total_mem = (uint64_t) total_b;
    }
    return (int32_t) ggml_backend_dev_type(dev);
}

// ---- distributed inference: GGML RPC over caller-owned sockets ----

static void ob_close_raw_socket(long long fd) {
#ifdef _WIN32
    closesocket((SOCKET) fd);
#else
    close((int) fd);
#endif
}

int32_t ob_rpc_serve_fd(long long fd, const char * cache_dir,
                        int32_t n_threads, int32_t dev_index) {
    if (n_threads < 1 || dev_index < 0 || (size_t) dev_index >= ggml_backend_dev_count()) {
        // Close the socket so the bridging peer sees EOF instead of a hang.
        ob_close_raw_socket(fd);
        return -1;
    }
    ggml_backend_dev_t devices[1] = { ggml_backend_dev_get((size_t) dev_index) };
    // Empty string means "no cache" like NULL does: the serve session builds
    // cache paths by naive concatenation, so "" would resolve hash files
    // against the process cwd.
    if (cache_dir != NULL && cache_dir[0] == '\0') {
        cache_dir = NULL;
    }
    ggml_backend_rpc_serve_fd(fd, cache_dir, (size_t) n_threads, 1, devices);
    return 0;
}

// Registered remote servers. Slots are never reused within a process;
// upstream caches registrations per endpoint string for the process
// lifetime anyway (ggml-rpc.cpp reg_map), so a slot handle can stay a plain
// index. Registration is serialized by the Rust wrapper.
static ggml_backend_reg_t ob_rpc_servers[GGML_RPC_MAX_SERVERS];
static int32_t ob_rpc_servers_len = 0;

int32_t ob_rpc_add_server(const char * endpoint) {
    if (endpoint == NULL) {
        return -1;
    }
    ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoint);
    if (reg == NULL) {
        return -1;
    }
    // The same endpoint string returns the same cached reg: keep slots
    // idempotent instead of burning table entries.
    for (int32_t i = 0; i < ob_rpc_servers_len; i++) {
        if (ob_rpc_servers[i] == reg) {
            return i;
        }
    }
    if (ob_rpc_servers_len >= GGML_RPC_MAX_SERVERS) {
        return -2;
    }
    ob_rpc_servers[ob_rpc_servers_len] = reg;
    return ob_rpc_servers_len++;
}

int32_t ob_rpc_server_device_count(int32_t slot) {
    if (slot < 0 || slot >= ob_rpc_servers_len) {
        return -1;
    }
    return (int32_t) ggml_backend_reg_dev_count(ob_rpc_servers[slot]);
}

// Shared implementation for ob_model_load_devices (n_paths == 1) and
// ob_model_load_splits_devices: build the explicit device list from the
// registered server slots (+ optionally the best local device), then load
// either the single file or the split set across it.
static ob_model * ob_load_devices_impl(const char ** paths, size_t n_paths,
                                       const int32_t * slots, int32_t n_slots,
                                       const float * tensor_split, int32_t n_split,
                                       bool use_local_device, int32_t n_gpu_layers) {
    if (paths == NULL || n_paths == 0 || paths[0] == NULL || n_slots < 0 ||
        (n_slots > 0 && slots == NULL)) {
        return NULL;
    }
    const size_t max_devices = llama_max_devices();

    // devices[] is NULL-terminated per llama_model_params; tensor_split is
    // read to llama_max_devices() entries by the loader, so allocate both
    // at that size.
    ggml_backend_dev_t * devices =
        calloc(max_devices + 1, sizeof(ggml_backend_dev_t));
    float * split = calloc(max_devices, sizeof(float));
    if (devices == NULL || split == NULL) {
        free(devices);
        free(split);
        return NULL;
    }

    size_t n_devices = 0;
    bool ok = true;
    for (int32_t i = 0; ok && i < n_slots; i++) {
        int32_t slot = slots[i];
        if (slot < 0 || slot >= ob_rpc_servers_len) {
            ok = false;
            break;
        }
        ggml_backend_reg_t reg = ob_rpc_servers[slot];
        size_t count = ggml_backend_reg_dev_count(reg);
        for (size_t d = 0; d < count; d++) {
            if (n_devices >= max_devices) {
                ok = false;
                break;
            }
            devices[n_devices++] = ggml_backend_reg_dev_get(reg, d);
        }
    }
    if (ok && use_local_device) {
        ggml_backend_dev_t local = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (local == NULL) {
            local = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
        if (local == NULL) {
            local = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        if (local == NULL || n_devices >= max_devices) {
            ok = false;
        } else {
            devices[n_devices++] = local;
        }
    }
    // The placement contract: tensor_split entries map 1:1 onto devices[].
    if (n_devices == 0 || (n_split > 0 && (size_t) n_split != n_devices)) {
        ok = false;
    }
    if (!ok) {
        free(devices);
        free(split);
        return NULL;
    }
    devices[n_devices] = NULL;

    struct llama_model_params params = llama_model_default_params();
    params.devices = devices;
    params.n_gpu_layers = n_gpu_layers;
    params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    if (n_split > 0) {
        memcpy(split, tensor_split, (size_t) n_split * sizeof(float));
        params.tensor_split = split;
    }

    struct llama_model * model = n_paths == 1
        ? llama_model_load_from_file(paths[0], params)
        : llama_model_load_from_splits(paths, n_paths, params);
    // The loader copies both arrays (devices into model->devices,
    // tensor_split into the model's owned vector) before returning.
    free(devices);
    free(split);
    return ob_wrap_model(model);
}

ob_model * ob_model_load_devices(const char * path,
                                 const int32_t * slots, int32_t n_slots,
                                 const float * tensor_split, int32_t n_split,
                                 bool use_local_device, int32_t n_gpu_layers) {
    const char * paths[1] = { path };
    return ob_load_devices_impl(paths, path == NULL ? 0 : 1, slots, n_slots,
                                tensor_split, n_split, use_local_device, n_gpu_layers);
}

ob_model * ob_model_load_splits_devices(const char ** paths, size_t n_paths,
                                        const int32_t * slots, int32_t n_slots,
                                        const float * tensor_split, int32_t n_split,
                                        bool use_local_device, int32_t n_gpu_layers) {
    return ob_load_devices_impl(paths, n_paths, slots, n_slots,
                                tensor_split, n_split, use_local_device, n_gpu_layers);
}

int32_t ob_model_meta(const ob_model * m, const char * key, char * buf, size_t buf_size) {
    return llama_model_meta_val_str(m->model, key, buf, buf_size);
}

int32_t ob_chat_apply_template(const ob_model * m,
                               const char ** roles, const char ** contents, size_t n_msgs,
                               bool add_assistant, char * buf, int32_t buf_len) {
    const char * tmpl = llama_model_chat_template(m->model, NULL);
    if (tmpl == NULL) {
        return -2;
    }
    struct llama_chat_message * msgs =
        calloc(n_msgs, sizeof(struct llama_chat_message));
    if (msgs == NULL && n_msgs > 0) {
        return -1;
    }
    for (size_t i = 0; i < n_msgs; i++) {
        msgs[i].role = roles[i];
        msgs[i].content = contents[i];
    }
    int32_t n = llama_chat_apply_template(tmpl, msgs, n_msgs, add_assistant, buf, buf_len);
    free(msgs);
    return n;
}
