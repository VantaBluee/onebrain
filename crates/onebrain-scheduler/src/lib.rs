//! Placement: decide which layers run on which nodes.
//!
//! M0 ships the profile vocabulary plus the one rule that is product law
//! from day one: auto-solo (§1.4 — if the model fits one node's usable
//! memory, never distribute). The proportional splitter lands in M3/M4 and
//! the prima.cpp-style cost-model search in M7.

use serde::{Deserialize, Serialize};

use onebrain_proto::plan::NodeId;

#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
    #[error(
        "model needs {required_mb} MB pooled but the cluster has {available_mb} MB usable; \
         add a node, choose a smaller quant, or lower the context length"
    )]
    DoesNotFit { required_mb: u64, available_mb: u64 },
    #[error("no nodes available to plan on")]
    NoNodes,
}

/// A node's measured capabilities, filled by pairing-time profiling and
/// `onebrain bench`. Usable memory is measured free memory minus OS reserve
/// (on Macs, Metal's recommendedMaxWorkingSetSize) — never total RAM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceProfile {
    pub node: NodeId,
    pub usable_memory_bytes: u64,
    /// Prefill throughput from the compute microbench (tokens/sec).
    pub prefill_tps: f64,
    /// Decode throughput from the compute microbench (tokens/sec).
    pub decode_tps: f64,
    pub disk_read_mbps: f64,
}

/// The auto-solo rule: given the model's total memory need (weights + KV at
/// the requested context) and a candidate node, decide if it runs alone.
pub fn fits_solo(profile: &DeviceProfile, required_bytes: u64) -> bool {
    profile.usable_memory_bytes >= required_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(mem: u64) -> DeviceProfile {
        DeviceProfile {
            node: NodeId("n".into()),
            usable_memory_bytes: mem,
            prefill_tps: 100.0,
            decode_tps: 20.0,
            disk_read_mbps: 1000.0,
        }
    }

    #[test]
    fn solo_when_it_fits() {
        assert!(fits_solo(&node(16 << 30), 10 << 30));
    }

    #[test]
    fn distribute_when_it_does_not() {
        assert!(!fits_solo(&node(8 << 30), 10 << 30));
    }
}
