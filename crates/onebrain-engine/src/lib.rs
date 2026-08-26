//! Embedded llama.cpp engine.
//!
//! The vendored llama.cpp is linked statically through a minimal C shim (see
//! `shim/ob_shim.h`); this crate exposes the safe Rust surface. M0 scope:
//! load a GGUF, tokenize, greedy generation, and the engine build hash that
//! nodes compare at handshake. Distributed execution arrives in M3.

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

/// One inference context over a model, with a greedy sampler (M0 scope; the
/// full sampler surface arrives with the API milestone).
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
    }
}
