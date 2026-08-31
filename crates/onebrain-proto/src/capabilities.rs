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
    /// Wire-level speculative-decoding participation by peers. M7 shipped
    /// head-solo drafting (docs/perf.md §5 "draft placement v1") with no
    /// proto change — no peer-facing feature exists yet, so the bit stays
    /// dark deliberately until one does.
    pub const SPECULATIVE: u64 = 1 << 3;
    /// int8 activation compression on slow links (M7, flagged).
    pub const ACT_COMPRESSION: u64 = 1 << 4;
    /// On-demand cluster benchmarking (M7): answers `BenchRequest` with a
    /// fresh microbench `BenchReport` (docs/perf.md §10).
    pub const CLUSTER_BENCH: u64 = 1 << 5;

    /// The capabilities this build of OneBrain implements.
    pub fn current() -> Self {
        // Bits light up as milestones land so that mixed-version clusters
        // degrade gracefully. M3: pipeline-parallel inference. M6: P2P
        // weight-range sharing over blobs (`RangeQuery`/`RangeInventory`
        // plus the mesh blobs provider, docs/logistics.md). M7: on-demand
        // cluster benchmarking (`BenchRequest`/`BenchReport`, docs/perf.md
        // §10) — every build with the bit answers the request, at minimum
        // with the cannot-bench-now marker. SPECULATIVE, TENSOR_PARALLEL,
        // and ACT_COMPRESSION stay dark deliberately post-M7: they gate
        // wire-level peer features M7 chose not to ship (drafting is
        // head-solo, tensor-parallel is engine-gated, act-compression
        // flagged) — see the bit docs above.
        Capabilities(Self::PIPELINE_PARALLEL | Self::BLOB_SHARING | Self::CLUSTER_BENCH)
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

    #[test]
    fn current_lights_the_landed_milestones_only() {
        let caps = Capabilities::current();
        assert!(caps.supports(Capabilities::PIPELINE_PARALLEL), "M3");
        assert!(caps.supports(Capabilities::BLOB_SHARING), "M6");
        assert!(caps.supports(Capabilities::CLUSTER_BENCH), "M7 bench");
        // Deliberately dark post-M7: these bits gate wire-level features
        // M7 chose not to ship — speculative drafting is head-solo
        // (docs/perf.md §5), tensor-parallel is engine-gated,
        // act-compression flagged. They light only when a proto-level peer
        // feature exists to gate.
        assert!(!caps.supports(Capabilities::TENSOR_PARALLEL));
        assert!(!caps.supports(Capabilities::SPECULATIVE));
        assert!(!caps.supports(Capabilities::ACT_COMPRESSION));
    }
}
