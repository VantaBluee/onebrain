//! The GGML RPC tensor-cache lookup key (docs/logistics.md "RPC
//! tensor-cache pre-seeding", ADR 0004).
//!
//! When a serve session runs with a cache directory, the RPC protocol lets
//! the head replace a full weight push with a `SET_TENSOR_HASH` round trip:
//! the worker answers from a local file named by the FNV-1a-64 hash of the
//! tensor payload. Workers therefore pre-seed `<data_dir>/rpc-cache/` with
//! files carrying exactly the names the vendored serve loop will look up —
//! this module is the Rust mirror of that naming scheme, byte-for-byte
//! (reference: `fnv_hash` + the `%016PRIx64` cache paths in
//! `vendor/llama.cpp/ggml/src/ggml-rpc/ggml-rpc.cpp`).
//!
//! FNV-1a is NOT collision-resistant and carries no integrity here: payload
//! integrity comes from the BLAKE3 range manifests at download time; FNV is
//! only the protocol's lookup key.

/// Upstream's `HASH_THRESHOLD` (ggml-rpc.cpp): 10 MiB, in bytes.
///
/// The RPC client sends `SET_TENSOR_HASH` only for payloads STRICTLY larger
/// than this (`size > HASH_THRESHOLD`), and a caching serve session saves
/// incoming `SET_TENSOR` payloads under the same strict condition. The
/// logistics contract pre-seeds every tensor >= 10 MiB; the boundary case
/// (exactly 10 MiB) is simply never looked up, so pre-seeding it costs a
/// few idle bytes and can never serve wrong data.
pub const RPC_HASH_THRESHOLD: u64 = 10 * 1024 * 1024;

/// FNV-1a, 64-bit — exactly the `fnv_hash` the vendored GGML RPC protocol
/// uses for tensor-cache keys.
///
/// Algorithm (Fowler–Noll–Vo, variant 1a): start from the 64-bit offset
/// basis `0xcbf29ce484222325`; for each input byte, XOR the byte into the
/// hash, then multiply by the 64-bit FNV prime `0x100000001b3`, wrapping
/// mod 2^64. Hashing zero bytes returns the offset basis itself.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The exact cache file name a caching serve session resolves for `payload`:
/// the FNV-1a-64 hash as 16 lowercase, zero-padded hex digits, no extension
/// (upstream formats it with `snprintf("%016" PRIx64)` and joins it directly
/// onto the cache directory). Pre-seeded files must use this name verbatim
/// or the `SET_TENSOR_HASH` lookup misses and the head falls back to a full
/// push.
pub fn rpc_cache_filename(payload: &[u8]) -> String {
    format!("{:016x}", fnv1a64(payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed vectors, derived independently from the FNV-1a
    /// definition (not by running our own code):
    ///
    /// - `""`: no bytes are processed, so the result is the offset basis
    ///   `0xcbf29ce484222325` by definition.
    /// - `"a"` (one byte, 0x61): step 1 XOR:
    ///   `0xcbf29ce484222325 ^ 0x61 = 0xcbf29ce484222344`; step 2 multiply
    ///   by `0x100000001b3` mod 2^64 = `0xaf63dc4c8601ec8c` — which is also
    ///   the published FNV-1a-64 test vector for "a" (Fowler/Noll/Vo
    ///   reference test suite).
    /// - `"foobar"`: published reference vector `0x85944171f73967e8` from
    ///   the same suite; cross-checked with an independent Python
    ///   implementation of the reference algorithm.
    #[test]
    fn fnv1a64_matches_reference_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
        // NUL bytes are data, not terminators (payloads are binary).
        assert_eq!(fnv1a64(&[0x00]), 0xaf63_bd4c_8601_b7df);
    }

    /// The filename is 16 lowercase hex digits with zero padding and no
    /// extension. `b"baa"` hashes to 0x0039231913392937 (independently
    /// computed) — high byte zero, so it proves the `%016` padding: an
    /// unpadded rendering would produce 14 digits and every serve-side
    /// lookup would miss.
    #[test]
    fn cache_filename_is_padded_16_hex() {
        assert_eq!(rpc_cache_filename(b"baa"), "0039231913392937");
        assert_eq!(rpc_cache_filename(b"a"), "af63dc4c8601ec8c");
        for payload in [&b""[..], b"a", b"baa", b"hello world"] {
            let name = rpc_cache_filename(payload);
            assert_eq!(name.len(), 16);
            assert!(
                name.chars().all(
                    |c| c.is_ascii_hexdigit() && (c.is_ascii_digit() || c.is_ascii_lowercase())
                ),
                "filename must be lowercase hex: {name}"
            );
        }
    }

    #[test]
    fn threshold_matches_upstream() {
        // ggml-rpc.cpp: const size_t HASH_THRESHOLD = 10 * 1024 * 1024;
        assert_eq!(RPC_HASH_THRESHOLD, 10 * 1024 * 1024);
    }
}
