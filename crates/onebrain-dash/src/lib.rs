//! Dashboard: embedded static SPA plus the auth'd metrics endpoints that
//! feed it (topology graph, plan visualization, bottleneck advisor).
//!
//! Implementation is milestone M8; M0 reserves the metric vocabulary.

use serde::{Deserialize, Serialize};

/// One point-in-time snapshot for the dashboard's topology view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    pub node: String,
    pub memory_used_bytes: u64,
    pub memory_usable_bytes: u64,
    pub tokens_per_second: f64,
}
