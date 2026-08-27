//! Embedded llama.cpp engine.
//!
//! The vendored llama.cpp is linked statically through a minimal C shim (see
//! `shim/ob_shim.h`); this crate exposes the safe Rust surface. M0 scope:
//! load a GGUF, tokenize, greedy generation, and the engine build hash that
//! nodes compare at handshake. Distributed execution arrives in M3.

pub mod rpc;
mod sys;

use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Once;

use onebrain_proto::handshake::EngineBuildHash;

/// Token id in the model's vocabulary.
pub type Token = i32;

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
        let path_str = path.to_string_lossy().into_owned();
        let cpath =
            CString::new(path_str.clone()).map_err(|_| EngineError::BadPath(path_str.clone()))?;
        let slots: Vec<i32> = servers.iter().map(|s| s.slot()).collect();
        tracing::info!(
            path = %path_str,
            n_servers = servers.len(),
            n_devices,
            ?tensor_split,
            use_local_device,
            "loading model across devices"
        );
        let ptr = unsafe {
            sys::ob_model_load_devices(
                cpath.as_ptr(),
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
                path: path_str,
                n_devices,
            });
        }
        Ok(Model {
            ptr,
            path: path_str,
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

/// Options for creating a [`Session`].
#[derive(Debug, Clone)]
pub struct SessionParams {
    pub n_ctx: u32,
    /// Max tokens per decode call; prompts are chunked to this.
    pub n_batch: u32,
    /// <= 0 lets the engine choose.
    pub n_threads: i32,
}

impl Default for SessionParams {
    fn default() -> Self {
        SessionParams {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: 0,
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
#[derive(Debug, Clone, Copy)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub finished: FinishReason,
}

/// One inference context over a model with a configurable sampler chain
/// (greedy by default until [`Session::set_sampler`] is called).
pub struct Session<'m> {
    ptr: *mut sys::ObSession,
    model: &'m Model,
    n_batch: u32,
}

unsafe impl Send for Session<'_> {}

impl<'m> Session<'m> {
    pub fn new(model: &'m Model, params: &SessionParams) -> Result<Session<'m>, EngineError> {
        let ptr = unsafe {
            sys::ob_session_new(model.ptr, params.n_ctx, params.n_batch, params.n_threads)
        };
        if ptr.is_null() {
            return Err(EngineError::SessionCreate);
        }
        Ok(Session {
            ptr,
            model,
            n_batch: params.n_batch.max(1),
        })
    }

    pub fn model(&self) -> &Model {
        self.model
    }

    /// Decode tokens, chunked to the session's batch size.
    pub fn decode(&mut self, tokens: &[Token]) -> Result<(), EngineError> {
        for chunk in tokens.chunks(self.n_batch as usize) {
            let status = unsafe { sys::ob_decode(self.ptr, chunk.as_ptr(), chunk.len() as i32) };
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
        self.decode(prompt_tokens)?;
        let mut generated = 0usize;
        for _ in 0..max_new {
            let tok = self.sample();
            if self.model.is_eog(tok) {
                return Ok(GenerationStats {
                    prompt_tokens: prompt_tokens.len(),
                    generated_tokens: generated,
                    finished: FinishReason::Stop,
                });
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
            generated += 1;
            if on_token(tok, &piece).is_break() {
                return Ok(GenerationStats {
                    prompt_tokens: prompt_tokens.len(),
                    generated_tokens: generated,
                    finished: FinishReason::Aborted,
                });
            }
        }
        Ok(GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated,
            finished: FinishReason::Length,
        })
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

    #[test]
    fn build_hash_is_stamped() {
        let h = engine_build_hash();
        assert!(h.0.starts_with("llama.cpp-"));
        assert!(h.0.contains("cpu"));
        assert!(!llama_commit().is_empty());
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
                n_threads: 0,
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
                n_threads: 0,
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
}
