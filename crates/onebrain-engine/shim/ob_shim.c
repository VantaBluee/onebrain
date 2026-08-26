// See ob_shim.h for the contract. Compiled against the vendored llama.cpp
// headers by build.rs; any upstream API change fails here at compile time.
#include "ob_shim.h"

#include "llama.h"
#include "ggml-backend.h"

#include <stdlib.h>
#include <string.h>

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
