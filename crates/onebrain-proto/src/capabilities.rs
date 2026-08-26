//! Capability negotiation.
//!
//! Capabilities are a bitset exchanged at handshake. A node may only use a
//! feature against a peer when both sides advertise the bit. This is how the
//! protocol grows without version bumps: `PROTO_VERSION` changes only when an
//! existing message becomes incompatible.

use serde::{Deserialize, Serialize};

/// Bitset of optional protocol features a node supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Capabilities(pub u64);

impl Capabilities {
    /// Pipeline-parallel inference over mesh streams (M3).
    pub const PIPELINE_PARALLEL: u64 = 1 << 0;
    /// Peer-to-peer weight range sharing via blobs (M6).
    pub const BLOB_SHARING: u64 = 1 << 1;
    /// Tensor-parallel islands on sub-millisecond links (M7, engine-gated).
    pub const TENSOR_PARALLEL: u64 = 1 << 2;
    /// Speculative decoding participation (M7).
    pub const SPECULATIVE: u64 = 1 << 3;
    /// int8 activation compression on slow links (M7, flagged).
    pub const ACT_COMPRESSION: u64 = 1 << 4;

    /// The capabilities this build of OneBrain implements.
    pub fn current() -> Self {
        // M0: no distributed features are implemented yet; bits light up as
        // milestones land so that mixed-version clusters degrade gracefully.
        Capabilities(0)
    }

    pub fn supports(&self, bit: u64) -> bool {
        self.0 & bit != 0
    }

    /// Features both peers support.
    pub fn intersect(&self, other: Capabilities) -> Capabilities {
        Capabilities(self.0 & other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_keeps_common_bits() {
        let a = Capabilities(Capabilities::PIPELINE_PARALLEL | Capabilities::BLOB_SHARING);
        let b = Capabilities(Capabilities::PIPELINE_PARALLEL | Capabilities::SPECULATIVE);
        let c = a.intersect(b);
        assert!(c.supports(Capabilities::PIPELINE_PARALLEL));
        assert!(!c.supports(Capabilities::BLOB_SHARING));
        assert!(!c.supports(Capabilities::SPECULATIVE));
    }
}
