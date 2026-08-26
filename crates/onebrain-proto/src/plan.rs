//! Placement plans: which layers run where, under which strategy.
//!
//! Plans are epoch-numbered. The head computes a plan, broadcasts it, workers
//! ack, and the epoch activates. Any membership change (join/leave/death) is
//! simply a new epoch; workers fence stale epochs by rejecting operations
//! tagged with an older one (§3 of the product spec).

use serde::{Deserialize, Serialize};

/// Monotonically increasing plan generation. Ordering is total per cluster
/// head; workers must reject work items from epochs older than their active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Epoch(pub u64);

/// Stable identifier for a node: its mesh public key, hex-encoded.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

/// Inclusive range of transformer layers assigned to one node.
/// Embedding and output head are modeled as pseudo-layers by the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: u32,
    /// Exclusive end.
    pub end: u32,
}

impl LayerRange {
    pub fn len(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Cross-node execution strategy for a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Strategy {
    /// Single node runs the whole model (auto-solo, §1.4).
    Solo,
    /// Contiguous layer ranges pipelined across nodes; the only cross-node
    /// strategy until tensor-parallel islands prove out (scheduler v2).
    PipelineParallel,
}

/// One node's assignment within a plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assignment {
    pub node: NodeId,
    pub layers: LayerRange,
    /// Position in the pipeline (0 = holds the embedding input side).
    pub stage: u32,
}

/// A complete placement plan broadcast by the head.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub epoch: Epoch,
    /// Content-addressed model identity (manifest hash), not a display name.
    pub model: String,
    pub strategy: Strategy,
    /// Ordered by `stage`.
    pub assignments: Vec<Assignment>,
    /// Requested context length the KV budget was computed for.
    pub ctx_len: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epochs_order() {
        assert!(Epoch(2) > Epoch(1));
    }

    #[test]
    fn layer_range_len() {
        let r = LayerRange { start: 10, end: 22 };
        assert_eq!(r.len(), 12);
        assert!(!r.is_empty());
        assert!(LayerRange { start: 5, end: 5 }.is_empty());
    }

    #[test]
    fn plan_roundtrips_through_postcard() {
        let plan = Plan {
            epoch: Epoch(7),
            model: "blake3:deadbeef".into(),
            strategy: Strategy::PipelineParallel,
            assignments: vec![Assignment {
                node: NodeId("ab12".into()),
                layers: LayerRange { start: 0, end: 16 },
                stage: 0,
            }],
            ctx_len: 8192,
        };
        let bytes = crate::encode(&plan).unwrap();
        let back: Plan = crate::decode(&bytes).unwrap();
        assert_eq!(plan, back);
    }
}
