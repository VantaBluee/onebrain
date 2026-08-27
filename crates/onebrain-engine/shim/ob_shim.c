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

ob_model * ob_model_load(const char * path, int32_t n_gpu_layers, bool use_mmap) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    if (!use_mmap) {
        params.load_mode = LLAMA_LOAD_MODE_NONE;
    }

    struct llama_model * model = llama_model_load_from_file(path, params);
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

ob_session * ob_session_new(ob_model * m, uint32_t n_ctx, uint32_t n_batch, int32_t n_threads) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    if (n_threads > 0) {
        params.n_threads = n_threads;
        params.n_threads_batch = n_threads;
    }

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
    free(s);
}

int32_t ob_decode(ob_session * s, const int32_t * tokens, int32_t n_tokens) {
    if (n_tokens <= 0) {
        return -1;
    }
    if (n_tokens > s->scratch_cap) {
        int32_t * grown = realloc(s->scratch, (size_t) n_tokens * sizeof(int32_t));
        if (grown == NULL) {
            return -1;
        }
        s->scratch = grown;
        s->scratch_cap = n_tokens;
    }
    memcpy(s->scratch, tokens, (size_t) n_tokens * sizeof(int32_t));

    struct llama_batch batch = llama_batch_get_one(s->scratch, n_tokens);
    return llama_decode(s->ctx, batch);
}

int32_t ob_sample_greedy(ob_session * s) {
    return llama_sampler_sample(s->sampler, s->ctx, -1);
}

void ob_session_reset(ob_session * s) {
    llama_memory_clear(llama_get_memory(s->ctx), /*data=*/true);
    llama_sampler_reset(s->sampler);
}

void ob_session_set_sampler(ob_session * s, float temp, float top_p,
                            int32_t top_k, uint32_t seed) {
    struct llama_sampler * chain =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (chain == NULL) {
        return; // keep the existing chain rather than crash
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
    llama_sampler_free(s->sampler);
    s->sampler = chain;
}

int32_t ob_sample(ob_session * s) {
    return llama_sampler_sample(s->sampler, s->ctx, -1);
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

int32_t ob_rpc_serve_fd(long long fd, int32_t n_threads, int32_t dev_index) {
    if (n_threads < 1 || dev_index < 0 || (size_t) dev_index >= ggml_backend_dev_count()) {
        // Close the socket so the bridging peer sees EOF instead of a hang.
        ob_close_raw_socket(fd);
        return -1;
    }
    ggml_backend_dev_t devices[1] = { ggml_backend_dev_get((size_t) dev_index) };
    // cache_dir stays NULL until M6 wires the reaper + pre-seeding (ADR 0004).
    ggml_backend_rpc_serve_fd(fd, /*cache_dir=*/NULL, (size_t) n_threads, 1, devices);
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

ob_model * ob_model_load_devices(const char * path,
                                 const int32_t * slots, int32_t n_slots,
                                 const float * tensor_split, int32_t n_split,
                                 bool use_local_device, int32_t n_gpu_layers) {
    if (path == NULL || n_slots < 0 || (n_slots > 0 && slots == NULL)) {
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

    struct llama_model * model = llama_model_load_from_file(path, params);
    // The loader copies both arrays (devices into model->devices,
    // tensor_split into the model's owned vector) before returning.
    free(devices);
    free(split);
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
