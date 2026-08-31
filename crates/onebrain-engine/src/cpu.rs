//! CPU-topology detection for thread-count defaults (docs/perf.md
//! "Thread-count defaults").
//!
//! llama.cpp's own engine default (`GGML_DEFAULT_N_THREADS = 4`) leaves
//! most of a modern CPU idle: measured on a 24-core hybrid machine,
//! decode ran 1.7-2.1x faster at the performance-core count and prefill
//! 3-4x faster at the full physical-core count. The two optima genuinely
//! differ — single-token decode is memory-bandwidth-bound and REGRESSES
//! when efficiency cores join (and collapses under saturated affinity
//! masks, so no pinning is done here or anywhere), while prefill is
//! compute-bound and keeps scaling — hence two recommended counts.
//!
//! Detection is deliberately conservative:
//! - Windows: `GetLogicalProcessorInformationEx(RelationProcessorCore)` —
//!   physical cores (SMT collapses to one), performance cores = cores
//!   whose `EfficiencyClass` equals the maximum present (all cores on a
//!   non-hybrid CPU).
//! - Elsewhere: `std::thread::available_parallelism()` for both counts, an
//!   over-approximation on SMT/hybrid machines, documented as such. (CI
//!   runners have few cores, so the resolved counts there stay near
//!   llama.cpp's own default.)
//!
//! Nothing in the engine consults this module implicitly:
//! [`crate::SessionParams`] defaults stay `0` (= llama.cpp's own choice,
//! the documented bit-for-bit pre-M7 behavior) and callers opt in.

use std::sync::OnceLock;

/// Thread counts recommended for an inference session on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecommendedThreads {
    /// For [`crate::SessionParams::n_threads`] (single-token decode):
    /// the performance-core count. Always >= 1.
    pub n_threads: i32,
    /// For [`crate::SessionParams::n_threads_batch`] (prefill and
    /// multi-token batch steps): the full physical-core count. Always
    /// >= `n_threads`.
    pub n_threads_batch: i32,
}

/// Detect once per process; the topology cannot change under us.
pub fn recommended_threads() -> RecommendedThreads {
    static CACHE: OnceLock<RecommendedThreads> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let (perf, total) = detect_cores().unwrap_or_else(|| {
            let n = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            (n, n)
        });
        let perf = perf.clamp(1, i32::MAX as usize) as i32;
        let total = total.clamp(1, i32::MAX as usize) as i32;
        RecommendedThreads {
            n_threads: perf.min(total),
            n_threads_batch: total.max(perf),
        }
    })
}

/// `(performance_cores, total_physical_cores)`, or `None` when the OS
/// query is unavailable/failed (the caller falls back to
/// `available_parallelism`).
#[cfg(windows)]
fn detect_cores() -> Option<(usize, usize)> {
    // Direct kernel32 externs, no new dependencies (the same pattern as
    // onebraind::power). Layout facts used below, verified against the
    // Windows SDK headers:
    //   SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX {
    //       LOGICAL_PROCESSOR_RELATIONSHIP Relationship; // u32, offset 0
    //       DWORD Size;                                  // u32, offset 4
    //       union { PROCESSOR_RELATIONSHIP Processor; .. } // offset 8
    //   }
    //   PROCESSOR_RELATIONSHIP { BYTE Flags;            // offset 8
    //                            BYTE EfficiencyClass;  // offset 9
    //                            .. }
    // Entries are variable-length; walk by `Size`.
    const RELATION_PROCESSOR_CORE: u32 = 0;
    extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut u8,
            returned_length: *mut u32,
        ) -> i32;
    }

    let mut len: u32 = 0;
    // Sizing call: fails (returns 0) with ERROR_INSUFFICIENT_BUFFER and
    // writes the required length.
    unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len);
    }
    if len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    let ok = unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr(), &mut len)
    };
    if ok == 0 {
        return None;
    }
    let buf = &buf[..len as usize];

    let mut classes: Vec<u8> = Vec::new();
    let mut off = 0usize;
    while off + 10 <= buf.len() {
        let relationship = u32::from_ne_bytes(buf[off..off + 4].try_into().ok()?);
        let size = u32::from_ne_bytes(buf[off + 4..off + 8].try_into().ok()?) as usize;
        if size < 8 || off + size > buf.len() {
            return None; // malformed entry: never guess
        }
        if relationship == RELATION_PROCESSOR_CORE {
            classes.push(buf[off + 9]); // PROCESSOR_RELATIONSHIP.EfficiencyClass
        }
        off += size;
    }
    if classes.is_empty() {
        return None;
    }
    // Higher EfficiencyClass = more performant core; non-hybrid CPUs
    // report one class for every core, so perf == total there.
    let max_class = classes.iter().copied().max().expect("non-empty");
    let perf = classes.iter().filter(|&&c| c == max_class).count();
    Some((perf, classes.len()))
}

#[cfg(not(windows))]
fn detect_cores() -> Option<(usize, usize)> {
    // No portable physical/hybrid-core query without new dependencies:
    // fall back to available_parallelism (documented over-approximation).
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommended_threads_are_sane_and_stable() {
        let r = recommended_threads();
        // Contract: both counts positive, decode <= batch (E-cores and SMT
        // siblings only ever ADD to the batch count).
        assert!(r.n_threads >= 1);
        assert!(r.n_threads_batch >= r.n_threads);
        // A machine that can run this test has bounded parallelism.
        assert!(r.n_threads_batch <= 4096);
        // Cached: two calls agree (OnceLock).
        assert_eq!(r, recommended_threads());
    }

    #[cfg(windows)]
    #[test]
    fn windows_core_query_succeeds() {
        // On Windows the real query must work (the fallback is for other
        // OSes); physical cores never exceed logical processors.
        let (perf, total) = detect_cores().expect("GetLogicalProcessorInformationEx works");
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert!(perf >= 1 && perf <= total);
        assert!(total <= logical);
    }
}
