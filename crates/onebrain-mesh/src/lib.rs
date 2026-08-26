//! Mesh transport: device identities, pairing, discovery, authenticated
//! streams, link probing, and heartbeats.
//!
//! M0 ships only the vocabulary types. M2 implements the transport on iroh:
//! Ed25519 device identities (dial by public key), mDNS LAN discovery, QUIC
//! with mutual authentication, and the RTT/bandwidth prober. Nothing in this
//! crate may ever open an unauthenticated non-loopback listener (§1.3).

use serde::{Deserialize, Serialize};

use onebrain_proto::plan::NodeId;

#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    #[error("peer {0:?} is not paired with this device; run `onebrain pair` on both machines")]
    NotPaired(NodeId),
    #[error("mesh transport is not implemented yet (arrives in milestone M2)")]
    NotImplemented,
}

/// Measured quality of a link between two nodes. Populated by the M2 prober;
/// consumed by the scheduler for boundary placement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinkProfile {
    pub rtt_micros: u64,
    pub bandwidth_mbps: f64,
    /// Fraction of probe packets lost (Wi-Fi warning signal, §1.7).
    pub loss: f32,
}

/// A paired peer as persisted in the peer store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Peer {
    pub id: NodeId,
    /// Human-readable name chosen at pairing time.
    pub name: String,
}
