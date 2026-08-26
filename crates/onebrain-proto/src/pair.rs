//! Pairing wire protocol (ALPN `onebrain/pair/1`).
//!
//! Pairing authenticates a short numeric code through a PAKE (SPAKE2,
//! symmetric mode) so that neither a LAN eavesdropper nor a malicious
//! device that races the exchange can complete it without the code: the
//! code never crosses the wire, and each side proves knowledge via a key-
//! confirmation MAC bound to both endpoint identities.
//!
//! Sequence (either side may be the dialer):
//! 1. Both sides send `Pake` (their SPAKE2 message), derive the shared key.
//! 2. Both sides send `Confirm` with `mac = keyed_hash(shared_key,
//!    "onebrain-pair-v1" || sorted(endpoint_id_a, endpoint_id_b))`.
//!    A wrong code yields a different key ⇒ MAC mismatch ⇒ `Rejected`.
//! 3. On success both sides exchange `Introduce` and persist the peer.

use serde::{Deserialize, Serialize};

/// Domain separator for the key-confirmation MAC.
pub const PAIR_CONFIRM_CONTEXT: &str = "onebrain-pair-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PairMessage {
    /// SPAKE2 (Ed25519 group, symmetric) first-flight message.
    Pake { body: Vec<u8> },
    /// Key-confirmation MAC over the pairing transcript.
    Confirm { mac: [u8; 32] },
    /// Sent after both confirms verify; introduces this device.
    Introduce {
        node_name: String,
        product_version: String,
    },
    /// Terminal failure; the connection closes after this.
    Rejected { reason: String },
}

/// Compute the key-confirmation MAC: a BLAKE3 keyed hash over the context
/// and the two endpoint ids in sorted order (so both sides agree without
/// caring who dialed).
pub fn confirm_mac(shared_key: &[u8; 32], id_a: &str, id_b: &str) -> [u8; 32] {
    let (lo, hi) = if id_a <= id_b { (id_a, id_b) } else { (id_b, id_a) };
    let mut hasher = blake3::Hasher::new_keyed(shared_key);
    hasher.update(PAIR_CONFIRM_CONTEXT.as_bytes());
    hasher.update(lo.as_bytes());
    hasher.update(hi.as_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mac_is_order_independent() {
        let key = [7u8; 32];
        assert_eq!(confirm_mac(&key, "aa", "bb"), confirm_mac(&key, "bb", "aa"));
    }

    #[test]
    fn mac_differs_by_key_and_ids() {
        let k1 = [1u8; 32];
        let k2 = [2u8; 32];
        assert_ne!(confirm_mac(&k1, "aa", "bb"), confirm_mac(&k2, "aa", "bb"));
        assert_ne!(confirm_mac(&k1, "aa", "bb"), confirm_mac(&k1, "aa", "cc"));
    }
}
