//! Device profile production: the compute microbench, the disk sequential
//! read probe, and `profile.toml` persistence (docs/scheduler-v1.md
//! "Profiles").
//!
//! The microbench runs against the registry test model (pulled through the
//! normal registry path — never bundled): one warmup generate, then measured
//! prefill (64-token prompt, 3 reps, median) and decode (32 greedy steps,
//! 3 reps, median), timed with [`std::time::Instant`].
//!
//! The disk probe reads 64 MiB (capped at the file size for tiny models)
//! through a fresh handle. Documented caveat: the OS page cache makes the
//! result an upper bound; it is used only for relative ordering and the M7
//! disk-offload penalties, never as an absolute promise.

use std::io::Read;
use std::path::Path;
use std::time::Instant;

use onebrain_engine::{Model, ModelParams, Session, SessionParams, Token};
use serde::{Deserialize, Serialize};

/// Prompt length for the prefill measurement (tokens).
pub const PREFILL_PROMPT_TOKENS: usize = 64;
/// Greedy steps per decode measurement.
pub const DECODE_STEPS: usize = 32;
/// Repetitions per measurement; the median is kept.
pub const MICROBENCH_REPS: usize = 3;
/// Bytes the disk probe reads (capped at the file size).
pub const DISK_PROBE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("microbench engine failure: {0}")]
    Engine(#[from] onebrain_engine::EngineError),
    #[error(
        "microbench could not build a prompt: the model tokenized the seed text to nothing; \
         the test model may be corrupt — re-pull it with `onebrain pull`"
    )]
    EmptyPrompt,
    #[error("cannot read {path}: {source}; check the file exists and is readable")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "profile file {path} is not valid TOML: {detail}; delete it and re-run `onebrain bench`"
    )]
    Parse { path: String, detail: String },
    #[error("cannot serialize the profile: {0}")]
    Serialize(#[from] toml::ser::Error),
}

/// Result of the compute microbench: measured throughput on the registry
/// test model. Only meaningful relative to other nodes' numbers on the same
/// test model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComputeProfile {
    /// Prefill throughput (tokens/sec), median of [`MICROBENCH_REPS`].
    pub prefill_tps: f64,
    /// Decode throughput (tokens/sec), median of [`MICROBENCH_REPS`].
    pub decode_tps: f64,
}

/// The persisted `<config_dir>/profile.toml` payload. `measured_unix` is
/// supplied by the caller (seconds since the epoch when the bench ran) so
/// this module stays clock-free and the stamp is testable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StoredProfile {
    /// Unix seconds when the profile was measured (`onebrain bench` shows
    /// the profile's age from this).
    pub measured_unix: u64,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub disk_mbps: f64,
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("throughput is never NaN"));
    samples[samples.len() / 2]
}

/// Run the compute microbench against `model_path` (the registry test
/// model). Loads the model, does one warmup generate, then measures prefill
/// and decode throughput per the module docs. Blocking and CPU-heavy
/// (~seconds); call from a blocking context.
pub fn measure_compute(model_path: &Path) -> Result<ComputeProfile, ProfileError> {
    let model = Model::load(model_path, &ModelParams::default())?;
    let mut session = Session::new(
        &model,
        &SessionParams {
            n_ctx: 256,
            n_batch: PREFILL_PROMPT_TOKENS as u32,
            n_threads: 0,
        },
    )?;

    let seed = model.tokenize("Once upon a time", true)?;
    if seed.is_empty() {
        return Err(ProfileError::EmptyPrompt);
    }
    // Exactly PREFILL_PROMPT_TOKENS tokens: cycle the seed. Repeated ids are
    // valid vocabulary entries; the output is discarded, only timing counts.
    let prompt: Vec<Token> = seed
        .iter()
        .copied()
        .cycle()
        .take(PREFILL_PROMPT_TOKENS)
        .collect();

    // One warmup generate: first-touch page faults, backend graph building
    // and thread-pool spin-up land here, outside the measured reps.
    session.generate_greedy(&seed, 8, |_, _| {})?;

    let mut prefill = Vec::with_capacity(MICROBENCH_REPS);
    for _ in 0..MICROBENCH_REPS {
        session.reset();
        let start = Instant::now();
        session.decode(&prompt)?;
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        prefill.push(PREFILL_PROMPT_TOKENS as f64 / secs);
    }

    let mut decode = Vec::with_capacity(MICROBENCH_REPS);
    for _ in 0..MICROBENCH_REPS {
        session.reset();
        // Establish a short context (untimed), then time pure single-token
        // decode steps. EOG is deliberately ignored: the step count must be
        // fixed for comparable timing, and feeding an EOG token back is
        // mechanically fine — the text is thrown away.
        session.decode(&seed)?;
        let start = Instant::now();
        for _ in 0..DECODE_STEPS {
            let tok = session.sample_greedy();
            session.decode(&[tok])?;
        }
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        decode.push(DECODE_STEPS as f64 / secs);
    }

    let profile = ComputeProfile {
        prefill_tps: median(prefill),
        decode_tps: median(decode),
    };
    tracing::info!(
        prefill_tps = profile.prefill_tps,
        decode_tps = profile.decode_tps,
        model = %model_path.display(),
        "compute microbench complete"
    );
    Ok(profile)
}

/// Measure sequential disk read throughput in MB/s (decimal megabytes):
/// read [`DISK_PROBE_BYTES`] of `path` — capped at the file size for tiny
/// models — through a fresh handle. Upper bound only (page cache; module
/// docs).
pub fn measure_disk(path: &Path) -> Result<f64, ProfileError> {
    let io_err = |source| ProfileError::Io {
        path: path.display().to_string(),
        source,
    };
    let file_len = std::fs::metadata(path).map_err(io_err)?.len();
    let want = DISK_PROBE_BYTES.min(file_len);
    let mut file = std::fs::File::open(path).map_err(io_err)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut read_total: u64 = 0;
    let start = Instant::now();
    while read_total < want {
        let cap = buf.len().min((want - read_total) as usize);
        let n = file.read(&mut buf[..cap]).map_err(io_err)?;
        if n == 0 {
            break; // File shrank underneath us; measure what we got.
        }
        read_total += n as u64;
    }
    let secs = start.elapsed().as_secs_f64().max(1e-9);
    let mbps = read_total as f64 / 1_000_000.0 / secs;
    tracing::debug!(
        bytes = read_total,
        mbps,
        path = %path.display(),
        "disk sequential read probe complete"
    );
    Ok(mbps)
}

/// Write `profile` to `path` as TOML, creating parent directories.
pub fn save_profile(path: &Path, profile: &StoredProfile) -> Result<(), ProfileError> {
    let io_err = |source| ProfileError::Io {
        path: path.display().to_string(),
        source,
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io_err)?;
    }
    let text = toml::to_string_pretty(profile)?;
    std::fs::write(path, text).map_err(io_err)
}

/// Read a [`StoredProfile`] back from `path`.
pub fn load_profile(path: &Path) -> Result<StoredProfile, ProfileError> {
    let text = std::fs::read_to_string(path).map_err(|source| ProfileError::Io {
        path: path.display().to_string(),
        source,
    })?;
    toml::from_str(&text).map_err(|e| ProfileError::Parse {
        path: path.display().to_string(),
        detail: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_roundtrips_through_toml() {
        let dir = tempfile::tempdir().unwrap();
        // Parent creation is part of the contract (fresh config dirs).
        let path = dir.path().join("cfg").join("profile.toml");
        let stored = StoredProfile {
            measured_unix: 1_756_200_000,
            prefill_tps: 812.53,
            decode_tps: 41.25,
            disk_mbps: 1732.8,
        };
        save_profile(&path, &stored).unwrap();
        let back = load_profile(&path).unwrap();
        assert_eq!(back, stored);
    }

    #[test]
    fn corrupt_profile_reports_path_and_remedy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profile.toml");
        std::fs::write(&path, "measured_unix = \"not a number\"\n").unwrap();
        let err = load_profile(&path).unwrap_err().to_string();
        assert!(err.contains("profile.toml"), "{err}");
        assert!(err.contains("onebrain bench"), "remedy missing: {err}");
    }

    #[test]
    fn missing_profile_is_an_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_profile(&dir.path().join("nope.toml")).unwrap_err();
        assert!(matches!(err, ProfileError::Io { .. }), "{err}");
    }

    #[test]
    fn disk_probe_caps_at_file_size_and_is_positive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.bin");
        std::fs::write(&path, vec![0xABu8; 128 * 1024]).unwrap();
        let mbps = measure_disk(&path).unwrap();
        assert!(mbps > 0.0, "read throughput must be positive: {mbps}");
    }

    /// Engine-backed microbench smoke: needs a real tiny GGUF, so it runs
    /// only when OB_SMOKE_MODEL points at one (same gating as the engine's
    /// own smoke tests).
    #[test]
    fn microbench_smoke_measures_positive_throughput() {
        let Ok(model_path) = std::env::var("OB_SMOKE_MODEL") else {
            eprintln!("OB_SMOKE_MODEL not set; skipping microbench smoke test");
            return;
        };
        let path = Path::new(&model_path);
        let compute = measure_compute(path).expect("microbench should run on the smoke model");
        assert!(
            compute.prefill_tps > 0.0,
            "prefill tok/s must be positive: {compute:?}"
        );
        assert!(
            compute.decode_tps > 0.0,
            "decode tok/s must be positive: {compute:?}"
        );
        let disk = measure_disk(path).expect("disk probe should run on the smoke model");
        assert!(disk > 0.0, "disk MB/s must be positive: {disk}");
    }
}
