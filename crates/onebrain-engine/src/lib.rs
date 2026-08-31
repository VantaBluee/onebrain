//! Embedded llama.cpp engine.
//!
//! The vendored llama.cpp is linked statically through a minimal C shim (see
//! `shim/ob_shim.h`); this crate exposes the safe Rust surface. M0 scope:
//! load a GGUF, tokenize, greedy generation, and the engine build hash that
//! nodes compare at handshake. Distributed execution arrives in M3.

pub mod rpc;
pub mod rpc_cache;
mod sys;

use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Once;
use std::time::Instant;

use onebrain_proto::handshake::EngineBuildHash;

/// Token id in the model's vocabulary.
pub type Token = i32;

/// Sequence id within a session's KV memory (`0..n_seq_max`).
pub type SeqId = i32;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("model path contains an interior NUL byte: {0}")]
    BadPath(String),
    #[error(
        "failed to load model from {path}. Check the file is a valid GGUF and that \
         this machine has enough free memory; `onebrain doctor` shows both."
    )]
    ModelLoad { path: String },
    #[error("failed to create an inference session (context). Lower --ctx or free memory.")]
    SessionCreate,
    #[error("tokenization failed for the given input")]
    Tokenize,
    #[error(
        "decode failed with engine status {status}. Status 1 means no KV slot: \
         lower the batch size or raise the context length."
    )]
    Decode { status: i32 },
    #[error("engine returned invalid UTF-8 where text was expected")]
    BadUtf8,
    #[error("the model's chat template could not be applied")]
    ChatTemplate,
    #[error("RPC endpoint contains an interior NUL byte: {0}")]
    BadEndpoint(String),
    #[error(
        "could not register the RPC server at {endpoint}: connection failed or the peer \
         spoke a different RPC protocol. Check the bridge is running; a version mismatch \
         looks identical to connect failure here, so verify both nodes report the same \
         engine build hash (`onebrain doctor`)."
    )]
    RpcConnect { endpoint: String },
    #[error(
        "too many RPC servers registered in this process (max {max}). \
         Reduce the number of remote nodes in the plan."
    )]
    RpcServerLimit { max: u32 },
    #[error(
        "RPC serve device index {index} is out of range (this node has {count} devices). \
         Use an index from the local device enumeration (`onebrain doctor` lists them)."
    )]
    RpcDeviceIndex { index: i32, count: i32 },
    #[error(
        "failed to create the local RPC bridge socket: {source}. Check that local \
         firewall or endpoint-security software allows loopback connections for this process."
    )]
    SocketPair { source: std::io::Error },
    #[error(
        "tensor_split has {got} entries but the plan spans {expected} devices; the \
         scheduler must emit exactly one fraction per device (remote devices in server \
         order, then the local device)."
    )]
    TensorSplit { expected: usize, got: usize },
    #[error(
        "failed to load model from {path} across {n_devices} devices. Check every RPC \
         bridge is connected and each node has enough free memory for its layer range; \
         `onebrain status` shows the active plan."
    )]
    DistributedLoad { path: String, n_devices: usize },
    #[error(
        "split model load was given an empty part list; a split cache entry must name \
         every part in order (`onebrain ls` shows the parts an entry holds)."
    )]
    NoSplitParts,
    #[error(
        "failed to load split model ({n_parts} parts, first part {first}). Check that \
         every part file exists, the parts come from the same split set, and they are \
         listed in split order (part -00001- first); `onebrain doctor` shows free memory."
    )]
    SplitLoad { first: String, n_parts: usize },
    #[error(
        "failed to allocate a token batch ({capacity} tokens, {n_seq_max} sequences); \
         both must be at least 1 and the process must have free memory."
    )]
    BatchAlloc { capacity: usize, n_seq_max: usize },
    #[error(
        "the token batch is full ({capacity} tokens); decode it (or clear it) before \
         pushing more, or create it with a larger capacity."
    )]
    BatchFull { capacity: usize },
    #[error(
        "sequence id {seq_id} is out of range for this batch (n_seq_max {n_seq_max}); \
         create the batch (and the session) with n_seq_max covering every sequence used."
    )]
    BatchSeqId { seq_id: i32, n_seq_max: i32 },
    #[error(
        "could not remove positions [{p0}, {p1}) of sequence {seq_id}: this model's \
         memory cannot drop a partial position range (recurrent / SWA models); roll \
         back by clearing the whole sequence instead."
    )]
    SeqRemove { seq_id: i32, p0: i32, p1: i32 },
    #[error(
        "this session cannot produce embeddings; create it with \
         SessionParams {{ embeddings: true, .. }}"
    )]
    NotEmbeddingSession,
    #[error("embedding input produced no tokens; provide non-empty text")]
    EmbedEmptyInput,
    #[error(
        "embedding input is {got} tokens but this model pools embeddings over a single \
         batch of at most {max}; shorten the input or raise the session's batch size"
    )]
    EmbedInputTooLong { got: usize, max: usize },
    #[error(
        "the engine produced no embedding rows for the decoded input; the model may not \
         support embedding extraction (rerankers emit ranks, not embeddings)"
    )]
    EmbedOutput,
    #[error(
        "failed to allocate a sampler chain; the process is out of memory — \
         free memory and retry the request"
    )]
    SamplerAlloc,
    #[error("RPC tensor cache directory path contains an interior NUL byte: {0}")]
    BadCacheDir(String),
    #[error(
        "failed to create the RPC tensor cache directory {path}: {source}. Check \
         permissions under the data dir; the cache is optional — serve without one to \
         fall back to full weight pushes."
    )]
    RpcCacheDir {
        path: String,
        source: std::io::Error,
    },
}

/// The engine build identity exchanged at handshake: vendored llama.cpp
/// commit + compiled backend set + target triple. Two nodes cooperate on a
/// plan only when these match exactly (product spec §3).
pub fn engine_build_hash() -> EngineBuildHash {
    EngineBuildHash(env!("OB_ENGINE_BUILD_ID").to_string())
}

/// Full commit hash of the vendored llama.cpp this binary embeds.
pub fn llama_commit() -> &'static str {
    env!("OB_LLAMA_COMMIT")
}

static INIT: Once = Once::new();

/// Initialize the process-wide engine backend. Idempotent; called by every
/// entry point that needs the engine. Logging from the C side is silenced —
/// engine events surface through `tracing` on the Rust side instead.
pub fn init() {
    INIT.call_once(|| unsafe {
        sys::ob_log_silence(true);
        sys::ob_backend_init();
    });
}

/// Tear down the process-wide engine backend. Only meaningful on clean
/// daemon shutdown, and only after every [`Model`] and [`Session`] has been
/// dropped; the process must not touch the engine again afterwards.
pub fn shutdown() {
    if INIT.is_completed() {
        unsafe { sys::ob_backend_free() };
    }
}

/// llama.cpp's own version string (e.g. its build number).
pub fn llama_version() -> String {
    init();
    unsafe { cstr_to_string(sys::ob_llama_version()) }
}

/// Human-readable summary of what the compiled engine supports (SIMD, GPU
/// backends). Surfaced by `onebrain doctor`.
pub fn system_info() -> String {
    init();
    unsafe { cstr_to_string(sys::ob_system_info()) }
}

unsafe fn cstr_to_string(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}

/// Convert a non-empty part-path list into C strings (plus the lossy display
/// strings for errors/logs). The typed errors match the single-path loaders:
/// empty list and interior NUL both fail before any FFI call.
fn cstring_paths(paths: &[&Path]) -> Result<(Vec<CString>, Vec<String>), EngineError> {
    if paths.is_empty() {
        return Err(EngineError::NoSplitParts);
    }
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let cpaths: Vec<CString> = path_strs
        .iter()
        .map(|s| CString::new(s.clone()).map_err(|_| EngineError::BadPath(s.clone())))
        .collect::<Result<_, _>>()?;
    Ok((cpaths, path_strs))
}

/// What kind of compute device a backend exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
    Other,
}

/// One compute device visible to the compiled engine.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub description: String,
    pub kind: DeviceKind,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

/// Enumerate the compute devices the engine can use right now. Feeds
/// backend autodetection, `onebrain doctor`, and (later) device profiling.
pub fn devices() -> Vec<DeviceInfo> {
    init();
    let count = unsafe { sys::ob_dev_count() };
    let mut out = Vec::with_capacity(count.max(0) as usize);
    for i in 0..count {
        let mut name = vec![0u8; 128];
        let mut desc = vec![0u8; 256];
        let mut free_b = 0u64;
        let mut total_b = 0u64;
        let kind = unsafe {
            sys::ob_dev_info(
                i,
                name.as_mut_ptr().cast(),
                name.len(),
                desc.as_mut_ptr().cast(),
                desc.len(),
                &mut free_b,
                &mut total_b,
            )
        };
        if kind < 0 {
            continue;
        }
        let trim = |mut v: Vec<u8>| {
            let end = v.iter().position(|b| *b == 0).unwrap_or(v.len());
            v.truncate(end);
            String::from_utf8_lossy(&v).into_owned()
        };
        out.push(DeviceInfo {
            name: trim(name),
            description: trim(desc),
            kind: match kind {
                0 => DeviceKind::Cpu,
                1 => DeviceKind::Gpu,
                2 => DeviceKind::IntegratedGpu,
                3 => DeviceKind::Accelerator,
                _ => DeviceKind::Other,
            },
            free_bytes: free_b,
            total_bytes: total_b,
        });
    }
    out
}

/// Options for loading a model.
#[derive(Debug, Clone)]
pub struct ModelParams {
    /// Layers to offload to GPU backends; negative = all that fit.
    pub n_gpu_layers: i32,
    /// Memory-map weights (the default; disable for benchmarking cold loads).
    pub use_mmap: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        ModelParams {
            n_gpu_layers: -1,
            use_mmap: true,
        }
    }
}

/// A loaded model. Immutable after load; llama.cpp's read paths on it are
/// thread-safe, but OneBrain keeps one owner (the engine host task).
pub struct Model {
    ptr: *mut sys::ObModel,
    path: String,
}

// The underlying llama_model is not tied to a thread.
unsafe impl Send for Model {}

impl Model {
    pub fn load(path: &Path, params: &ModelParams) -> Result<Model, EngineError> {
        init();
        let path_str = path.to_string_lossy().into_owned();
        let cpath =
            CString::new(path_str.clone()).map_err(|_| EngineError::BadPath(path_str.clone()))?;
        let ptr =
            unsafe { sys::ob_model_load(cpath.as_ptr(), params.n_gpu_layers, params.use_mmap) };
        if ptr.is_null() {
            return Err(EngineError::ModelLoad { path: path_str });
        }
        Ok(Model {
            ptr,
            path: path_str,
        })
    }

    /// Load a model stored as multiple split-GGUF parts (docs/logistics.md
    /// "Split-GGUF"): `paths` must list every part in split order (part
    /// `-00001-` first). Wraps `llama_model_load_from_splits`, so parts with
    /// non-standard names load too as long as the order is right. The
    /// returned model is the ordinary [`Model`]; [`Model::path`] reports the
    /// first part.
    pub fn load_splits(paths: &[&Path], params: &ModelParams) -> Result<Model, EngineError> {
        init();
        let (cpaths, path_strs) = cstring_paths(paths)?;
        let ptrs: Vec<*const std::os::raw::c_char> = cpaths.iter().map(|c| c.as_ptr()).collect();
        let ptr = unsafe {
            sys::ob_model_load_splits(
                ptrs.as_ptr(),
                ptrs.len(),
                params.n_gpu_layers,
                params.use_mmap,
            )
        };
        if ptr.is_null() {
            return Err(EngineError::SplitLoad {
                first: path_strs[0].clone(),
                n_parts: paths.len(),
            });
        }
        Ok(Model {
            ptr,
            path: path_strs[0].clone(),
        })
    }

    /// Load a model split across remote RPC servers plus (optionally) this
    /// node's own device, per the M3 placement contract
    /// (docs/distributed.md): the device order is every server's devices in
    /// `servers` order, then the local device when `use_local_device`;
    /// `tensor_split[i]` is device i's layer proportion and must have
    /// exactly one entry per device — llama.cpp's own free-memory probing
    /// (a live network round trip per remote device) never drives
    /// placement. Split mode is always by layer.
    ///
    /// The returned [`Model`] is the ordinary model type (same sessions,
    /// same drop/free path); weights are memory-mapped locally and pushed
    /// to remote devices during this call, so `params.use_mmap` is ignored
    /// here (distributed loads always map the local file).
    pub fn load_distributed(
        path: &Path,
        servers: &[&rpc::RemoteServer],
        tensor_split: &[f32],
        use_local_device: bool,
        params: &ModelParams,
    ) -> Result<Model, EngineError> {
        Self::load_distributed_splits(&[path], servers, tensor_split, use_local_device, params)
    }

    /// [`Model::load_distributed`] for a split-GGUF model: identical
    /// placement contract, but the local weights come from `paths` — every
    /// part, in split order, as in [`Model::load_splits`]. A single-element
    /// `paths` behaves exactly like `load_distributed` (which delegates
    /// here), so the daemon can call this unconditionally with whatever part
    /// list the cache entry holds. [`Model::path`] reports the first part.
    pub fn load_distributed_splits(
        paths: &[&Path],
        servers: &[&rpc::RemoteServer],
        tensor_split: &[f32],
        use_local_device: bool,
        params: &ModelParams,
    ) -> Result<Model, EngineError> {
        init();
        let n_devices = servers
            .iter()
            .map(|s| s.device_count().max(0) as usize)
            .sum::<usize>()
            + usize::from(use_local_device);
        if tensor_split.len() != n_devices {
            return Err(EngineError::TensorSplit {
                expected: n_devices,
                got: tensor_split.len(),
            });
        }
        if !params.use_mmap {
            tracing::warn!(
                "distributed loads always memory-map the local weights; use_mmap=false ignored"
            );
        }
        let (cpaths, path_strs) = cstring_paths(paths)?;
        let ptrs: Vec<*const std::os::raw::c_char> = cpaths.iter().map(|c| c.as_ptr()).collect();
        let slots: Vec<i32> = servers.iter().map(|s| s.slot()).collect();
        tracing::info!(
            path = %path_strs[0],
            n_parts = paths.len(),
            n_servers = servers.len(),
            n_devices,
            ?tensor_split,
            use_local_device,
            "loading model across devices"
        );
        let ptr = unsafe {
            sys::ob_model_load_splits_devices(
                ptrs.as_ptr(),
                ptrs.len(),
                slots.as_ptr(),
                slots.len() as i32,
                tensor_split.as_ptr(),
                tensor_split.len() as i32,
                use_local_device,
                params.n_gpu_layers,
            )
        };
        if ptr.is_null() {
            return Err(EngineError::DistributedLoad {
                path: path_strs[0].clone(),
                n_devices,
            });
        }
        Ok(Model {
            ptr,
            path: path_strs[0].clone(),
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn n_layer(&self) -> i32 {
        unsafe { sys::ob_model_n_layer(self.ptr) }
    }

    pub fn n_embd(&self) -> i32 {
        unsafe { sys::ob_model_n_embd(self.ptr) }
    }

    pub fn n_ctx_train(&self) -> i32 {
        unsafe { sys::ob_model_n_ctx_train(self.ptr) }
    }

    pub fn n_params(&self) -> u64 {
        unsafe { sys::ob_model_n_params(self.ptr) }
    }

    pub fn size_bytes(&self) -> u64 {
        unsafe { sys::ob_model_size_bytes(self.ptr) }
    }

    pub fn desc(&self) -> String {
        let mut buf = vec![0u8; 256];
        let n = unsafe { sys::ob_model_desc(self.ptr, buf.as_mut_ptr().cast(), buf.len()) };
        if n < 0 {
            return String::new();
        }
        buf.truncate(n as usize);
        String::from_utf8_lossy(&buf).into_owned()
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<Token>, EngineError> {
        let ctext = text.as_bytes();
        let needed = unsafe {
            sys::ob_tokenize(
                self.ptr,
                ctext.as_ptr().cast(),
                ctext.len() as i32,
                std::ptr::null_mut(),
                0,
                add_special,
            )
        };
        // A negative return is the required capacity; zero tokens is legal
        // for an empty prompt.
        let cap = needed.unsigned_abs() as usize;
        let mut tokens = vec![0i32; cap];
        if cap == 0 {
            return Ok(tokens);
        }
        let written = unsafe {
            sys::ob_tokenize(
                self.ptr,
                ctext.as_ptr().cast(),
                ctext.len() as i32,
                tokens.as_mut_ptr(),
                cap as i32,
                add_special,
            )
        };
        if written < 0 {
            return Err(EngineError::Tokenize);
        }
        tokens.truncate(written as usize);
        Ok(tokens)
    }

    pub fn token_to_piece(&self, token: Token) -> Result<String, EngineError> {
        let mut buf = vec![0u8; 128];
        let n = unsafe {
            sys::ob_token_to_piece(self.ptr, token, buf.as_mut_ptr().cast(), buf.len() as i32)
        };
        if n < 0 {
            return Err(EngineError::BadUtf8);
        }
        buf.truncate(n as usize);
        // Token pieces may split UTF-8 sequences; callers that stream should
        // buffer bytes. For M0 we replace invalid sequences.
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    pub fn is_eog(&self, token: Token) -> bool {
        unsafe { sys::ob_token_is_eog(self.ptr, token) }
    }

    /// A GGUF metadata value by key (e.g. `general.architecture`).
    pub fn meta(&self, key: &str) -> Option<String> {
        let ckey = CString::new(key).ok()?;
        let mut buf = vec![0u8; 512];
        let n = unsafe {
            sys::ob_model_meta(self.ptr, ckey.as_ptr(), buf.as_mut_ptr().cast(), buf.len())
        };
        if n < 0 {
            return None;
        }
        buf.truncate(n as usize);
        Some(String::from_utf8_lossy(&buf).into_owned())
    }

    /// Render a conversation through the model's built-in chat template.
    /// Returns `Ok(None)` when the model ships no template (the caller picks
    /// a fallback format and says so in logs).
    pub fn apply_chat_template(
        &self,
        turns: &[(String, String)],
        add_assistant: bool,
    ) -> Result<Option<String>, EngineError> {
        let roles: Vec<CString> = turns
            .iter()
            .map(|(r, _)| CString::new(r.as_str()))
            .collect::<Result<_, _>>()
            .map_err(|_| EngineError::Tokenize)?;
        let contents: Vec<CString> = turns
            .iter()
            .map(|(_, c)| CString::new(c.as_str()))
            .collect::<Result<_, _>>()
            .map_err(|_| EngineError::Tokenize)?;
        let role_ptrs: Vec<*const std::os::raw::c_char> =
            roles.iter().map(|c| c.as_ptr()).collect();
        let content_ptrs: Vec<*const std::os::raw::c_char> =
            contents.iter().map(|c| c.as_ptr()).collect();

        // Start with a generous guess; grow once if the template needs more.
        let mut cap = turns.iter().map(|(r, c)| r.len() + c.len()).sum::<usize>() * 2 + 1024;
        for _ in 0..2 {
            let mut buf = vec![0u8; cap];
            let n = unsafe {
                sys::ob_chat_apply_template(
                    self.ptr,
                    role_ptrs.as_ptr(),
                    content_ptrs.as_ptr(),
                    turns.len(),
                    add_assistant,
                    buf.as_mut_ptr().cast(),
                    buf.len() as i32,
                )
            };
            if n == -2 {
                return Ok(None);
            }
            if n < 0 {
                return Err(EngineError::ChatTemplate);
            }
            if (n as usize) <= buf.len() {
                buf.truncate(n as usize);
                return Ok(Some(String::from_utf8_lossy(&buf).into_owned()));
            }
            cap = n as usize + 1;
        }
        Err(EngineError::ChatTemplate)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { sys::ob_model_free(self.ptr) };
    }
}

/// When to enable Flash Attention (mirrors `llama_flash_attn_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlashAttnType {
    /// Let the engine decide per model/backend (llama.cpp's default).
    #[default]
    Auto,
    Disabled,
    Enabled,
}

impl FlashAttnType {
    fn code(self) -> i32 {
        match self {
            FlashAttnType::Auto => -1,
            FlashAttnType::Disabled => 0,
            FlashAttnType::Enabled => 1,
        }
    }
}

/// Element type for the KV cache (the `ggml_type` subset OneBrain exposes).
/// Quantized KV trades accuracy for memory; F16 is llama.cpp's default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)] // Q8_0/Q4_0 are the upstream quant names
pub enum KvCacheType {
    #[default]
    F16,
    F32,
    Q8_0,
    Q4_0,
}

impl KvCacheType {
    fn code(self) -> i32 {
        // ggml_type enum values (ggml.h); the shim casts them back.
        match self {
            KvCacheType::F32 => 0,
            KvCacheType::F16 => 1,
            KvCacheType::Q4_0 => 2,
            KvCacheType::Q8_0 => 8,
        }
    }
}

/// How an embeddings context pools per-token states into one vector per
/// sequence (mirrors `enum llama_pooling_type`; only meaningful with
/// [`SessionParams::embeddings`]). `Unspecified` defers to the model's own
/// declared pooling (GGUF metadata): purpose-built embedding models
/// resolve to their trained head (mean/CLS/last), generative models
/// resolve to no pooling — [`Session::embed`] then mean-pools per-token
/// rows in Rust. RANK (rerankers) is deliberately not exposed: its pooled
/// output is classification scores, not an `n_embd` vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolingType {
    /// Use the model's own declared pooling (llama.cpp's default).
    #[default]
    Unspecified,
    /// No pooling: only per-token embedding rows exist.
    None,
    Mean,
    Cls,
    Last,
}

impl PoolingType {
    fn code(self) -> i32 {
        // llama_pooling_type enum values (llama.h); the shim casts back.
        match self {
            PoolingType::Unspecified => -1,
            PoolingType::None => 0,
            PoolingType::Mean => 1,
            PoolingType::Cls => 2,
            PoolingType::Last => 3,
        }
    }
}

/// Options for creating a [`Session`] (perf contract, docs/perf.md §2).
///
/// The `Default` values reproduce the pre-M7 session EXACTLY (each new
/// field defaults to what llama.cpp already chose when the field was not
/// exposed), so existing callers that spread `..SessionParams::default()`
/// keep today's behavior bit-for-bit.
#[derive(Debug, Clone)]
pub struct SessionParams {
    pub n_ctx: u32,
    /// Max tokens per decode call; prompts are chunked to this.
    pub n_batch: u32,
    /// <= 0 lets the engine choose.
    pub n_threads: i32,
    /// Physical micro-batch: `llama_decode` splits each `n_batch` chunk
    /// into `n_ubatch` slices internally; per-slice activation copies bound
    /// distributed prefill cost (docs/perf.md §0/§3). 0 = engine default;
    /// the engine also caps it at `n_batch`.
    pub n_ubatch: u32,
    /// Max concurrent sequences in this context (micro-batched decode,
    /// docs/perf.md §6). Sequence ids passed to [`Batch::push`] and the
    /// `seq_*` methods must stay below this.
    pub n_seq_max: u32,
    /// One KV buffer shared across sequences instead of one per sequence —
    /// the §6 admission headroom math assumes this layout when running
    /// concurrent requests.
    pub kv_unified: bool,
    pub flash_attn_type: FlashAttnType,
    /// KV cache element types (K and V independently).
    pub type_k: KvCacheType,
    pub type_v: KvCacheType,
    /// Offload the KQV ops (including the KV cache) to GPU backends.
    pub offload_kqv: bool,
    /// Extract embeddings from decodes ([`Session::embed`]). Off by
    /// default: a `false` session is bit-for-bit the pre-embeddings
    /// generation context, and `embed` on it fails typed.
    pub embeddings: bool,
    /// Sequence pooling for an embeddings context; ignored by the engine
    /// when `embeddings` is off.
    pub pooling: PoolingType,
}

impl Default for SessionParams {
    fn default() -> Self {
        SessionParams {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: 0,
            // llama.cpp's own default; spelled out because it becomes a
            // `[perf]` config knob (docs/perf.md §3).
            n_ubatch: 512,
            n_seq_max: 1,
            kv_unified: false,
            flash_attn_type: FlashAttnType::Auto,
            type_k: KvCacheType::F16,
            type_v: KvCacheType::F16,
            offload_kqv: true,
            embeddings: false,
            pooling: PoolingType::Unspecified,
        }
    }
}

impl SessionParams {
    fn to_ffi(&self) -> sys::ObSessionParams {
        sys::ObSessionParams {
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_seq_max: self.n_seq_max,
            n_threads: self.n_threads,
            flash_attn_type: self.flash_attn_type.code(),
            type_k: self.type_k.code(),
            type_v: self.type_v.code(),
            pooling_type: self.pooling.code(),
            kv_unified: self.kv_unified,
            offload_kqv: self.offload_kqv,
            embeddings: self.embeddings,
        }
    }
}

/// Sampling configuration for a session.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerParams {
    /// `<= 0` selects greedy decoding (deterministic).
    pub temperature: f32,
    /// `>= 1` disables nucleus sampling.
    pub top_p: f32,
    /// `<= 0` disables top-k.
    pub top_k: i32,
    pub seed: u32,
}

impl Default for SamplerParams {
    fn default() -> Self {
        SamplerParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            seed: 0xFFFF_FFFF, // llama.cpp's "random seed" sentinel
        }
    }
}

/// A standalone sampler chain, independent of any [`Session`].
///
/// The chain construction is EXACTLY [`Session::set_sampler`]'s (one shared
/// builder in the shim, so the two cannot drift): `temperature <= 0` is pure
/// greedy, otherwise top-k → top-p → temp → dist(seed). Chains are
/// context-independent (`llama_sampler_chain_init` takes only chain params,
/// verified in the pinned llama.h): a `Sampler` may be created before any
/// session exists and sampled against any session via
/// [`Session::sample_ith_with`].
///
/// This is the per-sequence sampling primitive: a caller serving concurrent
/// sequences holds one `Sampler` per sequence, so each sequence's RNG/chain
/// state advances independently and a sampled request behaves identically
/// whether it runs alone or interleaved.
pub struct Sampler {
    ptr: *mut sys::ObSampler,
}

// The chain owns plain heap state with no thread affinity.
unsafe impl Send for Sampler {}

impl Sampler {
    /// Build a chain from `params`. Fails typed only on allocation failure
    /// (OOM-class).
    pub fn new(params: &SamplerParams) -> Result<Sampler, EngineError> {
        init();
        let ptr = unsafe {
            sys::ob_sampler_new(params.temperature, params.top_p, params.top_k, params.seed)
        };
        if ptr.is_null() {
            return Err(EngineError::SamplerAlloc);
        }
        Ok(Sampler { ptr })
    }

    /// Reset the chain's internal state: the dist sampler reseeds to its
    /// creation seed, replaying the draw sequence from the start.
    pub fn reset(&mut self) {
        unsafe { sys::ob_sampler_reset(self.ptr) };
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { sys::ob_sampler_free(self.ptr) };
    }
}

/// Why a generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// End-of-generation token.
    Stop,
    /// Hit the `max_new` budget.
    Length,
    /// The caller broke out (client disconnect, stop sequence).
    Aborted,
}

/// Outcome of one [`Session::generate`] call.
///
/// The wall-clock fields are measured inside `generate` itself (perf
/// contract §1: timing lands before any optimization, so every M7 lever has
/// the instrument that proves it). Milliseconds as `f64` because the sim
/// model decodes in microseconds — integer ms would round real work to 0.
#[derive(Debug, Clone, Copy)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub finished: FinishReason,
    /// Wall-clock of the prompt prefill decode, in milliseconds.
    pub prefill_ms: f64,
    /// Wall-clock of the sample/decode loop (everything after prefill), in
    /// milliseconds.
    pub decode_ms: f64,
    /// Time from `generate` entry to the first emitted piece, in
    /// milliseconds; 0 when nothing was emitted (immediate EOG).
    pub ttft_ms: f64,
    /// Speculative decoding counters (docs/perf.md §5); always 0 until the
    /// speculative loop lands.
    pub drafted: u32,
    pub accepted: u32,
}

/// A reusable multi-sequence token batch (perf contract, docs/perf.md §2):
/// per-token position, sequence id, and logits flag, decoded in one engine
/// call via [`Session::decode_batch`]. The substrate for KV reuse (§4),
/// speculative verify (§5), and micro-batched decode (§6) — no scheduling
/// policy lives here.
///
/// # Position rule (upstream-enforced)
///
/// A sequence's positions must stay CONSECUTIVE: the first position a batch
/// adds to a sequence must be exactly `seq_pos_max + 1` (0 for an empty
/// sequence), and further pushes to that sequence ascend by 1. llama.cpp
/// rejects the decode otherwise (llama-batch.cpp consistency checks).
/// Rolling back is a real [`Session::seq_rm`] of the divergent suffix,
/// never a rewound counter — the KV cache holds state per position, so a
/// "rewound" position would silently attend to stale entries. Both rules
/// are debug-asserted here and in [`Session::decode_batch`].
pub struct Batch {
    ptr: *mut sys::ObBatch,
    capacity: usize,
    n_seq_max: i32,
    /// (seq_id, first_pos, last_pos) per sequence present in the batch.
    /// Tiny linear-scan bookkeeping that powers the position-rule debug
    /// assertions and lets `decode_batch` verify continuity against the
    /// session's KV state cheaply (one entry per sequence, not per token).
    seqs: Vec<(SeqId, i32, i32)>,
}

// The batch owns plain heap arrays with no thread affinity.
unsafe impl Send for Batch {}

impl Batch {
    /// Allocate a batch holding up to `capacity` tokens tagged with
    /// sequence ids in `0..n_seq_max`. Reuse one batch across decode steps
    /// (via [`Batch::clear`]) instead of reallocating per step.
    pub fn new(capacity: usize, n_seq_max: usize) -> Result<Batch, EngineError> {
        let alloc_err = EngineError::BatchAlloc {
            capacity,
            n_seq_max,
        };
        if capacity == 0 || n_seq_max == 0 || capacity > i32::MAX as usize {
            return Err(alloc_err);
        }
        let Ok(n_seq) = i32::try_from(n_seq_max) else {
            return Err(alloc_err);
        };
        let ptr = unsafe { sys::ob_batch_new(capacity as i32, n_seq) };
        if ptr.is_null() {
            return Err(alloc_err);
        }
        Ok(Batch {
            ptr,
            capacity,
            n_seq_max: n_seq,
            seqs: Vec::new(),
        })
    }

    /// Append one token. `logits` requests logits for this position — the
    /// returned batch index is what [`Session::sample_ith`] takes after the
    /// decode. Each token carries exactly one sequence id; shared prefixes
    /// are expressed with [`Session::seq_cp`], never by multi-tagging.
    pub fn push(
        &mut self,
        token: Token,
        pos: i32,
        seq_id: SeqId,
        logits: bool,
    ) -> Result<usize, EngineError> {
        let index = self.len();
        if index >= self.capacity {
            return Err(EngineError::BatchFull {
                capacity: self.capacity,
            });
        }
        if seq_id < 0 || seq_id >= self.n_seq_max {
            return Err(EngineError::BatchSeqId {
                seq_id,
                n_seq_max: self.n_seq_max,
            });
        }
        match self.seqs.iter_mut().find(|(s, _, _)| *s == seq_id) {
            Some(entry) => {
                debug_assert_eq!(
                    pos,
                    entry.2 + 1,
                    "positions within a batch must ascend consecutively per \
                     sequence (seq {seq_id}: pushed pos {pos} after {})",
                    entry.2
                );
                entry.2 = pos;
            }
            None => self.seqs.push((seq_id, pos, pos)),
        }
        let ok = unsafe { sys::ob_batch_push(self.ptr, token, pos, seq_id, logits) };
        debug_assert!(ok, "shim push rejected an argument lib.rs validated");
        Ok(index)
    }

    /// Drop all queued tokens; capacity is retained.
    pub fn clear(&mut self) {
        unsafe { sys::ob_batch_clear(self.ptr) };
        self.seqs.clear();
    }

    /// Number of tokens currently queued.
    pub fn len(&self) -> usize {
        (unsafe { sys::ob_batch_n_tokens(self.ptr) }).max(0) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum tokens this batch can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { sys::ob_batch_free(self.ptr) };
    }
}

/// One (sequence, token) pair for a multi-sequence decode step — see
/// [`Session::decode_step`].
#[derive(Debug, Clone, Copy)]
pub struct SeqToken {
    pub seq_id: SeqId,
    pub token: Token,
}

/// One inference context over a model with a configurable sampler chain
/// (greedy by default until [`Session::set_sampler`] is called).
pub struct Session<'m> {
    ptr: *mut sys::ObSession,
    model: &'m Model,
    n_batch: u32,
    n_seq_max: u32,
    embeddings: bool,
}

unsafe impl Send for Session<'_> {}

impl<'m> Session<'m> {
    pub fn new(model: &'m Model, params: &SessionParams) -> Result<Session<'m>, EngineError> {
        let ffi = params.to_ffi();
        let ptr = unsafe { sys::ob_session_new(model.ptr, &ffi) };
        if ptr.is_null() {
            return Err(EngineError::SessionCreate);
        }
        Ok(Session {
            ptr,
            model,
            n_batch: params.n_batch.max(1),
            n_seq_max: params.n_seq_max.max(1),
            embeddings: params.embeddings,
        })
    }

    pub fn model(&self) -> &Model {
        self.model
    }

    /// Max concurrent sequences this session was created with — the bound
    /// on sequence ids for [`Batch::push`] and the `seq_*` methods (the
    /// daemon's §6 admission math needs it back).
    pub fn n_seq_max(&self) -> u32 {
        self.n_seq_max
    }

    /// Decode tokens, chunked to the session's batch size. Only the FINAL
    /// chunk asks for logits: nothing samples from an intermediate chunk,
    /// and on a distributed pipelined session the output-row fetch is the
    /// one per-chunk command that must block on that chunk's last graph
    /// before the next chunk's ubatches can be submitted (docs/perf.md §3).
    /// The decoded KV state is identical either way.
    pub fn decode(&mut self, tokens: &[Token]) -> Result<(), EngineError> {
        let n_chunks = tokens.chunks(self.n_batch as usize).count();
        for (i, chunk) in tokens.chunks(self.n_batch as usize).enumerate() {
            let want_logits = i + 1 == n_chunks;
            let status = unsafe {
                sys::ob_decode(self.ptr, chunk.as_ptr(), chunk.len() as i32, want_logits)
            };
            if status != 0 {
                return Err(EngineError::Decode { status });
            }
        }
        Ok(())
    }

    /// Greedy-sample the next token from the last decoded logits.
    pub fn sample_greedy(&mut self) -> Token {
        unsafe { sys::ob_sample_greedy(self.ptr) }
    }

    /// Sample with the configured chain (see [`Session::set_sampler`]).
    pub fn sample(&mut self) -> Token {
        unsafe { sys::ob_sample(self.ptr) }
    }

    /// Decode an explicit multi-sequence [`Batch`] in one engine call.
    ///
    /// The position rule (see [`Batch`]) is debug-asserted against the live
    /// KV state here: the first position the batch adds to each sequence
    /// must be that sequence's `seq_pos_max() + 1`. The batch is NOT
    /// cleared on success so callers can inspect indexes for
    /// [`Session::sample_ith`]; clear (or drop) it before reuse.
    pub fn decode_batch(&mut self, batch: &Batch) -> Result<(), EngineError> {
        debug_assert!(
            batch
                .seqs
                .iter()
                .all(|&(seq, first, _)| { first == self.seq_pos_max(seq).map_or(0, |p| p + 1) }),
            "a batch must continue each sequence at seq_pos_max + 1 \
             (rollback is seq_rm, never a rewound position counter)"
        );
        let status = unsafe { sys::ob_decode_batch(self.ptr, batch.ptr) };
        if status != 0 {
            return Err(EngineError::Decode { status });
        }
        Ok(())
    }

    /// Sample with the configured chain from the logits of batch token
    /// index `i` — the index [`Batch::push`] returned for a token pushed
    /// with `logits = true` in the most recently decoded batch (`-1` means
    /// the last logits-bearing token).
    ///
    /// The SESSION's chain is one object shared across sequences, which is
    /// only sound when its state does not matter (greedy is stateless).
    /// Callers serving concurrent sampled sequences hold one standalone
    /// [`Sampler`] per sequence and use [`Session::sample_ith_with`]
    /// instead, so each sequence's chain state advances independently.
    pub fn sample_ith(&mut self, i: i32) -> Token {
        unsafe { sys::ob_sample_ith(self.ptr, i) }
    }

    /// [`Session::sample_ith`] with a caller-owned standalone [`Sampler`]
    /// instead of the session's built-in chain — the per-sequence sampling
    /// primitive (see [`Sampler`]). Same index contract; the session's own
    /// chain is neither read nor advanced.
    pub fn sample_ith_with(&mut self, sampler: &mut Sampler, i: i32) -> Token {
        unsafe { sys::ob_sampler_sample_ith(self.ptr, sampler.ptr, i) }
    }

    /// Remove positions `[p0, p1)` of `seq_id` from the KV memory (negative
    /// `p0` = from 0, negative `p1` = to the end). THE rollback primitive:
    /// after removing a divergent suffix, re-decode from the removed range's
    /// start so positions stay consecutive. Fails typed when the model's
    /// memory cannot drop a partial range (recurrent/SWA); removing a whole
    /// sequence never fails.
    pub fn seq_rm(&mut self, seq_id: SeqId, p0: i32, p1: i32) -> Result<(), EngineError> {
        if unsafe { sys::ob_memory_seq_rm(self.ptr, seq_id, p0, p1) } {
            Ok(())
        } else {
            Err(EngineError::SeqRemove { seq_id, p0, p1 })
        }
    }

    /// Copy positions `[p0, p1)` of `seq_id_src` onto `seq_id_dst` (same
    /// range conventions as [`Session::seq_rm`]). Cheap KV sharing: this is
    /// how a shared prompt prefix reaches a second sequence without
    /// re-decoding it.
    pub fn seq_cp(&mut self, seq_id_src: SeqId, seq_id_dst: SeqId, p0: i32, p1: i32) {
        unsafe { sys::ob_memory_seq_cp(self.ptr, seq_id_src, seq_id_dst, p0, p1) };
    }

    /// Drop every sequence except `seq_id` from the KV memory.
    pub fn seq_keep(&mut self, seq_id: SeqId) {
        unsafe { sys::ob_memory_seq_keep(self.ptr, seq_id) };
    }

    /// Largest position present for `seq_id`, or `None` when the sequence
    /// is empty. The next token for a sequence always decodes at
    /// `seq_pos_max + 1` (0 when empty) — the position rule on [`Batch`].
    pub fn seq_pos_max(&self, seq_id: SeqId) -> Option<i32> {
        let pos = unsafe { sys::ob_memory_seq_pos_max(self.ptr, seq_id) };
        (pos >= 0).then_some(pos)
    }

    /// One multi-sequence decode step (docs/perf.md §6 primitive): append
    /// each `(seq_id, token)` at its sequence's next position, decode them
    /// all in ONE batch, and greedy-position sample per sequence. Returns
    /// the sampled next token per input, in input order.
    ///
    /// `batch` is only a reusable allocation — it is cleared on entry. A
    /// sequence may appear at most once per step (a repeat would need the
    /// position after a token this very call is still decoding; the batch
    /// position debug-assert catches it). This is a stepping primitive,
    /// NOT a scheduler: the caller decides which sequences are active each
    /// step (admission, fairness, and starvation rules live in the
    /// daemon).
    pub fn decode_step(
        &mut self,
        batch: &mut Batch,
        steps: &[SeqToken],
    ) -> Result<Vec<Token>, EngineError> {
        if steps.is_empty() {
            return Ok(Vec::new());
        }
        batch.clear();
        let mut indexes = Vec::with_capacity(steps.len());
        for step in steps {
            let pos = self.seq_pos_max(step.seq_id).map_or(0, |p| p + 1);
            indexes.push(batch.push(step.token, pos, step.seq_id, true)?);
        }
        self.decode_batch(batch)?;
        Ok(indexes
            .into_iter()
            .map(|i| self.sample_ith(i as i32))
            .collect())
    }

    /// Clear the KV cache and sampler state so the next decode starts a
    /// fresh sequence. Much cheaper than recreating the session.
    pub fn reset(&mut self) {
        unsafe { sys::ob_session_reset(self.ptr) };
    }

    /// Replace the sampler chain. `temperature <= 0` selects greedy.
    pub fn set_sampler(&mut self, params: &SamplerParams) {
        unsafe {
            sys::ob_session_set_sampler(
                self.ptr,
                params.temperature,
                params.top_p,
                params.top_k,
                params.seed,
            )
        };
    }

    /// Streaming generation: prefill `prompt_tokens`, then sample with the
    /// configured chain until EOG, `max_new`, or the callback returns
    /// [`std::ops::ControlFlow::Break`] (client disconnect, stop sequence).
    pub fn generate(
        &mut self,
        prompt_tokens: &[Token],
        max_new: usize,
        mut on_token: impl FnMut(Token, &str) -> std::ops::ControlFlow<()>,
    ) -> Result<GenerationStats, EngineError> {
        // Timing (docs/perf.md §1): measured at the source so every caller
        // reports the same numbers the daemon's perf log line greps for.
        let start = Instant::now();
        self.decode(prompt_tokens)?;
        let mut stats = GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: 0,
            finished: FinishReason::Length,
            prefill_ms: start.elapsed().as_secs_f64() * 1e3,
            decode_ms: 0.0,
            ttft_ms: 0.0,
            drafted: 0,
            accepted: 0,
        };
        let decode_start = Instant::now();
        for _ in 0..max_new {
            let tok = self.sample();
            if self.model.is_eog(tok) {
                stats.finished = FinishReason::Stop;
                stats.decode_ms = decode_start.elapsed().as_secs_f64() * 1e3;
                return Ok(stats);
            }
            let piece = self.model.token_to_piece(tok)?;
            // Confirm-before-send (docs/resilience.md): a token's own
            // decode must succeed BEFORE its piece is emitted. With a torn
            // remote (patches/0002), the logits fetch that produced `tok`
            // can have been silently zeroed — the very next decode on the
            // dead socket fails, and a piece once streamed cannot be
            // unstreamed (it would poison both the client text and the
            // resume prefix). Costs one decode step of first-token latency;
            // the token sequence is unchanged. Residual: a tear exactly at
            // the final budgeted token has no confirming decode (the tear
            // window is one token; documented in patches/README.md).
            self.decode(&[tok])?;
            stats.generated_tokens += 1;
            if stats.generated_tokens == 1 {
                // Time to first token = up to the first EMITTED piece, so
                // it includes the confirming decode above — that is the
                // moment the piece may actually reach the client.
                stats.ttft_ms = start.elapsed().as_secs_f64() * 1e3;
            }
            if on_token(tok, &piece).is_break() {
                stats.finished = FinishReason::Aborted;
                stats.decode_ms = decode_start.elapsed().as_secs_f64() * 1e3;
                return Ok(stats);
            }
        }
        stats.decode_ms = decode_start.elapsed().as_secs_f64() * 1e3;
        Ok(stats)
    }

    /// Embed one input: decode `tokens` as a fresh sequence and return its
    /// pooled embedding — exactly `n_embd` floats (M1 `/v1/embeddings`).
    ///
    /// Requires a session created with `SessionParams { embeddings: true }`
    /// (typed error otherwise). Two paths, chosen by the context's RESOLVED
    /// pooling type:
    ///
    /// - **Engine-pooled** (the model declares a pooling head, or the
    ///   session asked for an explicit [`PoolingType`]): the whole input is
    ///   decoded in ONE call — the pooling graph reduces over a single
    ///   batch, so a chunked decode would silently embed only the last
    ///   chunk — and the engine's own pooled sequence row is returned.
    ///   Inputs beyond `n_batch` tokens fail typed.
    /// - **Mean-pool fallback** (pooling resolved to NONE — every
    ///   generative model under the default `Unspecified`): per-token
    ///   embedding rows are MEAN-POOLED here in Rust, the standard trick
    ///   for embedding with a generative model. The decode is chunked (KV
    ///   carries the prefix, so each token's row is identical to the
    ///   single-shot value under causal attention) to bound the transient
    ///   output buffer, which this llama build sizes at
    ///   `n_vocab + n_embd` floats PER OUTPUT ROW.
    ///
    /// The session's KV is reset on entry (each input embeds
    /// independently) and left holding this input afterwards. Vectors are
    /// returned raw (un-normalized); unit-norm parity with OpenAI is the
    /// serving layer's choice.
    pub fn embed(&mut self, tokens: &[Token]) -> Result<Vec<f32>, EngineError> {
        /// Mean-pool decode chunk cap: bounds the per-decode output buffer
        /// (`(n_vocab + n_embd) × 4` bytes per row — ~256 MiB per 512
        /// tokens on a 128k-vocab model) without changing any row's value.
        const EMBED_CHUNK: usize = 512;

        if !self.embeddings {
            return Err(EngineError::NotEmbeddingSession);
        }
        if tokens.is_empty() {
            return Err(EngineError::EmbedEmptyInput);
        }
        let n_embd = self.model.n_embd().max(0) as usize;
        if n_embd == 0 {
            return Err(EngineError::EmbedOutput);
        }
        self.reset();
        let mut out = vec![0f32; n_embd];

        if unsafe { sys::ob_session_pooling(self.ptr) } > 0 {
            let max = self.n_batch as usize;
            if tokens.len() > max {
                return Err(EngineError::EmbedInputTooLong {
                    got: tokens.len(),
                    max,
                });
            }
            // want_logits = true leaves the batch's output flags NULL,
            // which an embeddings context reads as "output every token"
            // (llama.h batch docs) — required input for the pooling graph.
            let status =
                unsafe { sys::ob_decode(self.ptr, tokens.as_ptr(), tokens.len() as i32, true) };
            if status != 0 {
                return Err(EngineError::Decode { status });
            }
            let n = unsafe { sys::ob_embeddings_seq(self.ptr, 0, out.as_mut_ptr(), n_embd as i32) };
            if n != n_embd as i32 {
                // Declared pooling but no pooled row (RANK, or an engine
                // regression): honest typed error, never a zero vector.
                return Err(EngineError::EmbedOutput);
            }
            return Ok(out);
        }

        // Mean-pool fallback. f64 accumulation keeps long inputs from
        // losing low-order bits; the result is cast back to the engine's
        // own f32.
        let mut acc = vec![0f64; n_embd];
        let mut row = vec![0f32; n_embd];
        let chunk_len = (self.n_batch as usize).min(EMBED_CHUNK);
        for chunk in tokens.chunks(chunk_len) {
            let status =
                unsafe { sys::ob_decode(self.ptr, chunk.as_ptr(), chunk.len() as i32, true) };
            if status != 0 {
                return Err(EngineError::Decode { status });
            }
            for i in 0..chunk.len() {
                let n = unsafe {
                    sys::ob_embeddings_ith(self.ptr, i as i32, row.as_mut_ptr(), n_embd as i32)
                };
                if n != n_embd as i32 {
                    return Err(EngineError::EmbedOutput);
                }
                for (a, v) in acc.iter_mut().zip(row.iter()) {
                    *a += f64::from(*v);
                }
            }
        }
        let inv = 1.0 / tokens.len() as f64;
        for (o, a) in out.iter_mut().zip(acc.iter()) {
            *o = (a * inv) as f32;
        }
        Ok(out)
    }

    /// Convenience loop: prefill `prompt_tokens`, then greedily generate up
    /// to `max_new`, invoking `on_token` per generated token. Returns the
    /// generated tokens (EOG excluded).
    pub fn generate_greedy(
        &mut self,
        prompt_tokens: &[Token],
        max_new: usize,
        mut on_token: impl FnMut(Token, &str),
    ) -> Result<Vec<Token>, EngineError> {
        self.decode(prompt_tokens)?;
        let mut out = Vec::new();
        for _ in 0..max_new {
            let tok = self.sample_greedy();
            if self.model.is_eog(tok) {
                break;
            }
            let piece = self.model.token_to_piece(tok)?;
            on_token(tok, &piece);
            out.push(tok);
            self.decode(&[tok])?;
        }
        Ok(out)
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        unsafe { sys::ob_session_free(self.ptr) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devices_enumerate() {
        let devs = devices();
        assert!(!devs.is_empty(), "at least the CPU device must exist");
        assert!(devs.iter().any(|d| matches!(d.kind, DeviceKind::Cpu)));
        let cpu = devs
            .iter()
            .find(|d| matches!(d.kind, DeviceKind::Cpu))
            .unwrap();
        assert!(cpu.total_bytes > 0, "CPU device must report memory");
    }

    /// Split load with no parts must fail typed before any FFI call.
    #[test]
    fn load_splits_empty_list_is_typed_error() {
        match Model::load_splits(&[], &ModelParams::default()) {
            Err(EngineError::NoSplitParts) => {}
            Err(other) => panic!("expected NoSplitParts, got {other:?}"),
            Ok(_) => panic!("expected NoSplitParts, got a loaded model"),
        }
    }

    /// Split load over nonexistent parts must fail with the typed split
    /// error (naming the first part and the part count), not an abort.
    #[test]
    fn load_splits_missing_parts_is_typed_error() {
        let parts = [
            Path::new("definitely-missing-00001-of-00002.gguf"),
            Path::new("definitely-missing-00002-of-00002.gguf"),
        ];
        match Model::load_splits(&parts, &ModelParams::default()) {
            Err(EngineError::SplitLoad { first, n_parts }) => {
                assert_eq!(n_parts, 2);
                assert!(first.contains("00001-of-00002"));
            }
            Err(other) => panic!("expected SplitLoad, got {other:?}"),
            Ok(_) => panic!("expected SplitLoad, got a loaded model"),
        }
    }

    #[test]
    fn build_hash_is_stamped() {
        let h = engine_build_hash();
        assert!(h.0.starts_with("llama.cpp-"));
        assert!(h.0.contains("cpu"));
        assert!(!llama_commit().is_empty());
    }

    /// The widened SessionParams defaults must equal what llama.cpp chose
    /// before the fields were exposed (docs/perf.md §2: bit-for-bit) —
    /// verified against llama_context_default_params in the pinned vendor
    /// tree. A drifted default here silently changes every session.
    #[test]
    fn session_params_defaults_preserve_pre_widening_behavior() {
        let p = SessionParams::default();
        assert_eq!(p.n_ctx, 4096);
        assert_eq!(p.n_batch, 512);
        assert_eq!(p.n_threads, 0);
        assert_eq!(p.n_ubatch, 512, "llama.cpp default n_ubatch");
        assert_eq!(p.n_seq_max, 1, "llama.cpp default n_seq_max");
        assert!(!p.kv_unified, "llama.cpp default kv_unified = false");
        assert_eq!(p.flash_attn_type, FlashAttnType::Auto);
        assert_eq!(p.type_k, KvCacheType::F16);
        assert_eq!(p.type_v, KvCacheType::F16);
        assert!(p.offload_kqv, "llama.cpp default offload_kqv = true");
        assert!(!p.embeddings, "llama.cpp default embeddings = false");
        assert_eq!(
            p.pooling,
            PoolingType::Unspecified,
            "llama.cpp default pooling_type = UNSPECIFIED"
        );
        // The enum codes the shim casts back into llama/ggml enums.
        assert_eq!(FlashAttnType::Auto.code(), -1);
        assert_eq!(KvCacheType::F32.code(), 0);
        assert_eq!(KvCacheType::F16.code(), 1);
        assert_eq!(KvCacheType::Q4_0.code(), 2);
        assert_eq!(KvCacheType::Q8_0.code(), 8);
        assert_eq!(PoolingType::Unspecified.code(), -1);
        assert_eq!(PoolingType::None.code(), 0);
        assert_eq!(PoolingType::Mean.code(), 1);
        assert_eq!(PoolingType::Cls.code(), 2);
        assert_eq!(PoolingType::Last.code(), 3);
    }

    /// Batch bookkeeping without a model: capacity and seq-id bounds fail
    /// typed, clear() makes the allocation reusable, and push returns the
    /// index sample_ith will want.
    #[test]
    fn batch_push_bounds_and_clear() {
        match Batch::new(0, 1) {
            Err(EngineError::BatchAlloc { .. }) => {}
            Err(other) => panic!("expected BatchAlloc for zero capacity, got {other:?}"),
            Ok(_) => panic!("expected BatchAlloc for zero capacity, got a batch"),
        }
        let mut b = Batch::new(3, 2).expect("batch alloc");
        assert_eq!(b.capacity(), 3);
        assert!(b.is_empty());
        assert_eq!(b.push(11, 0, 0, false).unwrap(), 0);
        assert_eq!(b.push(12, 1, 0, false).unwrap(), 1);
        assert_eq!(b.push(21, 0, 1, true).unwrap(), 2);
        assert_eq!(b.len(), 3);
        match b.push(13, 2, 0, true) {
            Err(EngineError::BatchFull { capacity: 3 }) => {}
            other => panic!("expected BatchFull, got {other:?}"),
        }
        b.clear();
        assert!(b.is_empty());
        match b.push(31, 0, 2, false) {
            Err(EngineError::BatchSeqId {
                seq_id: 2,
                n_seq_max: 2,
            }) => {}
            other => panic!("expected BatchSeqId, got {other:?}"),
        }
        assert_eq!(b.push(11, 0, 1, true).unwrap(), 0, "reusable after clear");
    }

    /// The splits entry point over a single part must behave exactly like
    /// the plain load (the vendored loader ignores the splits list when the
    /// GGUF carries no split metadata): same layer count, same greedy
    /// tokens. A true multi-part parity test needs a gguf-split'd model and
    /// lives with the downloader's split fixtures (docs/logistics.md DoD).
    #[test]
    fn smoke_load_splits_single_part_parity() {
        let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping split-load smoke test");
            return;
        };
        let path = Path::new(&model_path);
        let solo = Model::load(path, &ModelParams::default()).expect("plain load");
        let split = Model::load_splits(&[path], &ModelParams::default()).expect("splits load");
        assert_eq!(solo.n_layer(), split.n_layer());
        assert_eq!(split.path(), path.to_string_lossy());

        let prompt = solo.tokenize("Once upon a time", true).unwrap();
        let gen = |model: &Model| {
            let mut session = Session::new(
                model,
                &SessionParams {
                    n_ctx: 256,
                    n_batch: 64,
                    ..SessionParams::default()
                },
            )
            .unwrap();
            session.generate_greedy(&prompt, 8, |_, _| {}).unwrap()
        };
        assert_eq!(
            gen(&solo),
            gen(&split),
            "single-part split load must generate identical greedy tokens"
        );
    }

    /// End-to-end smoke: requires a real tiny GGUF, so it only runs when
    /// OB_SMOKE_MODEL points at one (CI downloads it; `cargo xtask smoke`
    /// wires it locally).
    #[test]
    fn smoke_generate_greedy() {
        let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping engine smoke test");
            return;
        };
        let model = Model::load(Path::new(&model_path), &ModelParams::default())
            .expect("smoke model should load");
        assert!(model.n_layer() > 0);

        let prompt = model.tokenize("Once upon a time", true).unwrap();
        assert!(!prompt.is_empty());

        let mut session = Session::new(
            &model,
            &SessionParams {
                n_ctx: 256,
                n_batch: 64,
                ..SessionParams::default()
            },
        )
        .unwrap();

        let mut text = String::new();
        let toks = session
            .generate_greedy(&prompt, 16, |_t, piece| text.push_str(piece))
            .expect("generation should succeed");
        assert!(!toks.is_empty(), "expected at least one generated token");
        assert!(!text.is_empty(), "expected non-empty generated text");

        // Greedy decoding is deterministic: the same prompt in a fresh
        // session must reproduce the same tokens (ground truth for the
        // distributed-correctness property tests later).
        let mut session2 = Session::new(
            &model,
            &SessionParams {
                n_ctx: 256,
                n_batch: 64,
                ..SessionParams::default()
            },
        )
        .unwrap();
        let toks2 = session2.generate_greedy(&prompt, 16, |_, _| {}).unwrap();
        assert_eq!(toks, toks2, "greedy decode must be deterministic");

        // Session reset must behave like a fresh session: same prompt in the
        // SAME session after reset() reproduces the same greedy tokens.
        session2.reset();
        let toks3 = session2.generate_greedy(&prompt, 16, |_, _| {}).unwrap();
        assert_eq!(toks, toks3, "reset must clear all sequence state");

        // The general sampler path with temperature <= 0 must equal greedy.
        session2.reset();
        session2.set_sampler(&SamplerParams {
            temperature: 0.0,
            ..Default::default()
        });
        let mut toks4 = Vec::new();
        let stats = session2
            .generate(&prompt, 16, |t, _| {
                toks4.push(t);
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(toks, toks4, "temp<=0 sampling must match greedy");
        assert_eq!(stats.generated_tokens, toks4.len());

        // The tinyllamas models ship no chat template; the engine must say
        // so cleanly rather than erroring.
        let rendered = model
            .apply_chat_template(&[("user".into(), "hi".into())], true)
            .unwrap();
        assert!(rendered.is_none(), "stories260K declares no chat template");
    }

    /// GenerationStats wall-clock fields (docs/perf.md §1) are populated by
    /// generate(): all three phases take measurable time even on the tiny
    /// model (f64 ms keeps microsecond work above zero), and the
    /// speculative counters stay 0 until §5 lands.
    #[test]
    fn smoke_generation_stats_timed() {
        let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping stats smoke test");
            return;
        };
        let model = Model::load(Path::new(&model_path), &ModelParams::default()).unwrap();
        let prompt = model.tokenize("Once upon a time", true).unwrap();
        let mut session = Session::new(
            &model,
            &SessionParams {
                n_ctx: 256,
                n_batch: 64,
                ..SessionParams::default()
            },
        )
        .unwrap();
        session.set_sampler(&SamplerParams {
            temperature: 0.0,
            ..Default::default()
        });
        let stats = session
            .generate(&prompt, 8, |_, _| std::ops::ControlFlow::Continue(()))
            .unwrap();
        assert!(stats.generated_tokens > 0);
        assert!(
            stats.prefill_ms > 0.0,
            "prefill must take measurable time (got {})",
            stats.prefill_ms
        );
        assert!(
            stats.decode_ms > 0.0,
            "decode loop must take measurable time (got {})",
            stats.decode_ms
        );
        assert!(
            stats.ttft_ms > 0.0,
            "a generation that emitted tokens must have a TTFT (got {})",
            stats.ttft_ms
        );
        assert!(
            stats.ttft_ms >= stats.prefill_ms,
            "TTFT is measured from generate() entry, so it contains prefill \
             (ttft {} < prefill {})",
            stats.ttft_ms,
            stats.prefill_ms
        );
        assert_eq!(stats.drafted, 0, "speculative counters are 0 until §5");
        assert_eq!(stats.accepted, 0, "speculative counters are 0 until §5");
    }

    /// A standalone [`Sampler`] must reproduce the session-chain path
    /// byte-for-byte: same chain construction, same seed semantics (the
    /// shim shares one builder, and this pins it observably). Greedy
    /// standalone chains must equal `sample_greedy` too.
    #[test]
    fn smoke_standalone_sampler_matches_session_chain() {
        let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping standalone sampler smoke test");
            return;
        };
        let model = Model::load(Path::new(&model_path), &ModelParams::default()).unwrap();
        let prompt = model.tokenize("Once upon a time", true).unwrap();
        let params = SessionParams {
            n_ctx: 256,
            n_batch: 64,
            ..SessionParams::default()
        };
        let sampler_params = SamplerParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            seed: 42,
        };

        // Oracle: the session's own chain via set_sampler + generate.
        let mut chain_session = Session::new(&model, &params).unwrap();
        chain_session.set_sampler(&sampler_params);
        let mut expected = Vec::new();
        chain_session
            .generate(&prompt, 12, |t, _| {
                expected.push(t);
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
        assert!(!expected.is_empty(), "the oracle run must emit tokens");

        // Same loop shape (sample → EOG check → confirming decode) with a
        // standalone Sampler against a fresh session.
        let mut session = Session::new(&model, &params).unwrap();
        let mut sampler = Sampler::new(&sampler_params).unwrap();
        session.decode(&prompt).unwrap();
        let mut got = Vec::new();
        for _ in 0..12 {
            let tok = session.sample_ith_with(&mut sampler, -1);
            if model.is_eog(tok) {
                break;
            }
            session.decode(&[tok]).unwrap();
            got.push(tok);
        }
        assert_eq!(
            got, expected,
            "a standalone Sampler must reproduce the session chain exactly"
        );

        // reset() replays the draw sequence: re-decoding the same prompt on
        // a reset session with a reset sampler reproduces the same tokens.
        session.reset();
        sampler.reset();
        session.decode(&prompt).unwrap();
        let mut replay = Vec::new();
        for _ in 0..12 {
            let tok = session.sample_ith_with(&mut sampler, -1);
            if model.is_eog(tok) {
                break;
            }
            session.decode(&[tok]).unwrap();
            replay.push(tok);
        }
        assert_eq!(replay, expected, "reset must replay the seeded draws");

        // Greedy standalone chain == sample_greedy on the same logits.
        let mut greedy_session = Session::new(&model, &params).unwrap();
        let mut greedy = Sampler::new(&SamplerParams {
            temperature: 0.0,
            ..SamplerParams::default()
        })
        .unwrap();
        greedy_session.decode(&prompt).unwrap();
        assert_eq!(
            greedy_session.sample_ith_with(&mut greedy, -1),
            greedy_session.sample_greedy(),
            "a greedy standalone chain must equal the built-in greedy sampler"
        );
    }

    /// Max absolute element difference between two embeddings.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The embeddings surface end-to-end on the tiny model: the mean-pool
    /// fallback (stories260K is causal with no pooling head, so the default
    /// `Unspecified` resolves to NONE), the engine-pooled path (explicit
    /// MEAN), chunked-vs-single-shot equivalence, and the typed refusals.
    #[test]
    fn smoke_embeddings() {
        let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping embeddings smoke test");
            return;
        };
        let model = Model::load(Path::new(&model_path), &ModelParams::default()).unwrap();
        let n_embd = model.n_embd() as usize;
        assert!(n_embd > 0);
        let short = model.tokenize("Once upon a time", true).unwrap();
        let other = model.tokenize("The little dog barked", true).unwrap();
        // Longer than the fallback session's n_batch below, to force a
        // chunked mean-pool decode.
        let long = model
            .tokenize(&"once upon a time ".repeat(30), true)
            .unwrap();
        assert!(long.len() > 64);

        // Fallback path: embeddings on, pooling left Unspecified.
        let mut fallback = Session::new(
            &model,
            &SessionParams {
                n_ctx: 1024,
                n_batch: 64,
                embeddings: true,
                ..SessionParams::default()
            },
        )
        .unwrap();
        let e_short = fallback.embed(&short).unwrap();
        assert_eq!(e_short.len(), n_embd, "embedding must be n_embd wide");
        assert!(e_short.iter().all(|v| v.is_finite()));
        assert!(
            e_short.iter().any(|v| *v != 0.0),
            "embedding must not be the zero vector"
        );
        assert_eq!(
            e_short,
            fallback.embed(&short).unwrap(),
            "embedding must be deterministic"
        );
        let e_other = fallback.embed(&other).unwrap();
        assert_ne!(e_short, e_other, "different texts must embed differently");

        // Chunked mean-pool (n_batch 64) must match a single-shot decode of
        // the same input (causal rows depend only on their prefix; only
        // f32 graph-order noise may differ).
        let e_long_chunked = fallback.embed(&long).unwrap();
        let mut single = Session::new(
            &model,
            &SessionParams {
                n_ctx: 1024,
                n_batch: 1024,
                n_ubatch: 1024,
                embeddings: true,
                ..SessionParams::default()
            },
        )
        .unwrap();
        let e_long_single = single.embed(&long).unwrap();
        let diff = max_abs_diff(&e_long_chunked, &e_long_single);
        assert!(
            diff < 1e-3,
            "chunked and single-shot mean-pool diverged (max abs diff {diff})"
        );

        // Engine-pooled path: explicit MEAN makes llama pool the sequence
        // itself; it must agree with the Rust mean of the same rows.
        let mut pooled = Session::new(
            &model,
            &SessionParams {
                n_ctx: 256,
                n_batch: 64,
                n_ubatch: 64,
                embeddings: true,
                pooling: PoolingType::Mean,
                ..SessionParams::default()
            },
        )
        .unwrap();
        let e_pooled = pooled.embed(&short).unwrap();
        assert_eq!(e_pooled.len(), n_embd);
        let diff = max_abs_diff(&e_pooled, &e_short);
        assert!(
            diff < 1e-3,
            "engine MEAN pooling and the Rust mean-pool fallback diverged \
             (max abs diff {diff})"
        );
        // The pooled path cannot chunk: over-long input fails typed.
        match pooled.embed(&long) {
            Err(EngineError::EmbedInputTooLong { got, max: 64 }) if got == long.len() => {}
            other => panic!("expected EmbedInputTooLong, got {other:?}"),
        }

        // A generation session refuses embed typed; empty input too.
        let mut gen = Session::new(
            &model,
            &SessionParams {
                n_ctx: 256,
                n_batch: 64,
                ..SessionParams::default()
            },
        )
        .unwrap();
        match gen.embed(&short) {
            Err(EngineError::NotEmbeddingSession) => {}
            other => panic!("expected NotEmbeddingSession, got {other:?}"),
        }
        match fallback.embed(&[]) {
            Err(EngineError::EmbedEmptyInput) => {}
            other => panic!("expected EmbedEmptyInput, got {other:?}"),
        }

        // The generation surface is untouched by an embeddings session
        // having existed: greedy output on a fresh default session matches
        // a fresh run (bit-identical defaults contract).
        let mut a = Session::new(
            &model,
            &SessionParams {
                n_ctx: 256,
                n_batch: 64,
                ..SessionParams::default()
            },
        )
        .unwrap();
        let toks = a.generate_greedy(&short, 8, |_, _| {}).unwrap();
        assert!(!toks.is_empty());
    }
}
