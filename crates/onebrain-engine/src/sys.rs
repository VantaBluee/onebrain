//! Raw FFI to the C shim (shim/ob_shim.h). Nothing else in the crate may
//! call these directly; the safe wrappers in `lib.rs` own all invariants.

use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct ObModel {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ObSession {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ObBatch {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ObSampler {
    _private: [u8; 0],
}

/// Mirror of the shim-owned `ob_session_params` (shim/ob_shim.h). This is
/// deliberately NOT a llama.cpp struct: the shim defines it and this crate
/// compiles both sides, so the definitions can only drift together. Field
/// semantics (zero = engine default for scalars, bools carried verbatim)
/// are documented on the C side.
#[repr(C)]
pub struct ObSessionParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: c_int,
    pub n_threads_batch: c_int,
    pub flash_attn_type: i32,
    pub type_k: i32,
    pub type_v: i32,
    pub pooling_type: i32,
    pub kv_unified: bool,
    pub offload_kqv: bool,
    pub embeddings: bool,
}

extern "C" {
    pub fn ob_backend_init();
    pub fn ob_backend_free();
    pub fn ob_log_silence(silent: bool);
    pub fn ob_llama_version() -> *const c_char;
    pub fn ob_system_info() -> *const c_char;

    pub fn ob_model_load(path: *const c_char, n_gpu_layers: i32, use_mmap: bool) -> *mut ObModel;
    pub fn ob_model_load_splits(
        paths: *const *const c_char,
        n_paths: usize,
        n_gpu_layers: i32,
        use_mmap: bool,
    ) -> *mut ObModel;
    pub fn ob_model_free(m: *mut ObModel);
    pub fn ob_model_n_layer(m: *const ObModel) -> i32;
    pub fn ob_model_n_embd(m: *const ObModel) -> i32;
    pub fn ob_model_n_ctx_train(m: *const ObModel) -> i32;
    pub fn ob_model_n_params(m: *const ObModel) -> u64;
    pub fn ob_model_size_bytes(m: *const ObModel) -> u64;
    pub fn ob_model_desc(m: *const ObModel, buf: *mut c_char, buf_size: usize) -> i32;

    pub fn ob_tokenize(
        m: *const ObModel,
        text: *const c_char,
        text_len: i32,
        tokens: *mut i32,
        n_tokens_max: i32,
        add_special: bool,
    ) -> i32;
    pub fn ob_token_to_piece(m: *const ObModel, token: i32, buf: *mut c_char, buf_len: i32) -> i32;
    pub fn ob_token_is_eog(m: *const ObModel, token: i32) -> bool;

    pub fn ob_session_new(m: *mut ObModel, p: *const ObSessionParams) -> *mut ObSession;
    pub fn ob_session_free(s: *mut ObSession);
    pub fn ob_decode(
        s: *mut ObSession,
        tokens: *const i32,
        n_tokens: i32,
        want_logits: bool,
    ) -> i32;
    pub fn ob_sample_greedy(s: *mut ObSession) -> i32;
    pub fn ob_session_reset(s: *mut ObSession);
    pub fn ob_session_set_sampler(s: *mut ObSession, temp: f32, top_p: f32, top_k: i32, seed: u32);
    pub fn ob_sample(s: *mut ObSession) -> i32;

    // Explicit multi-sequence batches + per-sequence KV surgery
    // (docs/perf.md §2). The safe wrappers in lib.rs own the position rule
    // (consecutive per sequence).
    pub fn ob_batch_new(n_tokens_max: i32, n_seq_max: i32) -> *mut ObBatch;
    pub fn ob_batch_free(b: *mut ObBatch);
    pub fn ob_batch_clear(b: *mut ObBatch);
    pub fn ob_batch_push(b: *mut ObBatch, token: i32, pos: i32, seq_id: i32, logits: bool) -> bool;
    pub fn ob_batch_n_tokens(b: *const ObBatch) -> i32;
    pub fn ob_decode_batch(s: *mut ObSession, b: *const ObBatch) -> i32;
    pub fn ob_sample_ith(s: *mut ObSession, i: i32) -> i32;
    // Standalone sampler chains (per-sequence sampling): context-independent
    // chain objects owned by the caller; contracts in ob_shim.h.
    pub fn ob_sampler_new(temp: f32, top_p: f32, top_k: i32, seed: u32) -> *mut ObSampler;
    pub fn ob_sampler_free(smp: *mut ObSampler);
    pub fn ob_sampler_reset(smp: *mut ObSampler);
    pub fn ob_sampler_sample_ith(s: *mut ObSession, smp: *mut ObSampler, i: i32) -> i32;
    // Embeddings (M1 /v1/embeddings). The safe wrapper (Session::embed)
    // owns the decode/pooling protocol; contracts in ob_shim.h.
    pub fn ob_session_pooling(s: *const ObSession) -> i32;
    pub fn ob_embeddings_seq(s: *mut ObSession, seq_id: i32, buf: *mut f32, buf_len: i32) -> i32;
    pub fn ob_embeddings_ith(s: *mut ObSession, i: i32, buf: *mut f32, buf_len: i32) -> i32;

    pub fn ob_memory_seq_rm(s: *mut ObSession, seq_id: i32, p0: i32, p1: i32) -> bool;
    pub fn ob_memory_seq_cp(s: *mut ObSession, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32);
    pub fn ob_memory_seq_keep(s: *mut ObSession, seq_id: i32);
    pub fn ob_memory_seq_pos_max(s: *mut ObSession, seq_id: i32) -> i32;

    pub fn ob_dev_count() -> i32;
    pub fn ob_dev_info(
        i: i32,
        name: *mut c_char,
        name_len: usize,
        desc: *mut c_char,
        desc_len: usize,
        free_mem: *mut u64,
        total_mem: *mut u64,
    ) -> i32;

    // Distributed inference (docs/distributed.md). ob_rpc_add_server is not
    // thread-safe against itself; rpc.rs serializes registration.
    pub fn ob_rpc_serve_fd(
        fd: i64,
        cache_dir: *const c_char,
        n_threads: i32,
        dev_index: i32,
    ) -> i32;
    pub fn ob_rpc_add_server(endpoint: *const c_char) -> i32;
    pub fn ob_rpc_server_device_count(slot: i32) -> i32;
    // Note: the shim also exports single-path ob_model_load_devices; Rust
    // routes every distributed load through the splits variant (n_paths == 1
    // is the single-file case), so only that extern is declared here.
    pub fn ob_model_load_splits_devices(
        paths: *const *const c_char,
        n_paths: usize,
        slots: *const i32,
        n_slots: i32,
        tensor_split: *const f32,
        n_split: i32,
        use_local_device: bool,
        n_gpu_layers: i32,
    ) -> *mut ObModel;

    pub fn ob_model_meta(
        m: *const ObModel,
        key: *const c_char,
        buf: *mut c_char,
        buf_size: usize,
    ) -> i32;
    pub fn ob_chat_apply_template(
        m: *const ObModel,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n_msgs: usize,
        add_assistant: bool,
        buf: *mut c_char,
        buf_len: i32,
    ) -> i32;
}
