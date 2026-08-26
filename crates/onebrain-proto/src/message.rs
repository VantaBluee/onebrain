//! Top-level envelope for all internode messages.
//!
//! Every payload rides in an [`Envelope`] carrying the sender's active plan
//! epoch, which lets receivers fence stale traffic uniformly instead of each
//! handler re-implementing the check.

use serde::{Deserialize, Serialize};

use crate::handshake::Hello;
use crate::plan::{Epoch, Plan};

/// All message kinds that cross the mesh. Extended only alongside a
/// capability bit; removing or changing a variant bumps `PROTO_VERSION`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// First frame in both directions on any new connection.
    Hello(Hello),
    /// Head → worker: adopt this plan.
    PlanProposal(Plan),
    /// Worker → head: plan adopted and shards ready (or not).
    PlanAck {
        epoch: Epoch,
        ready: bool,
        /// Human-readable reason when `ready == false`.
        detail: Option<String>,
    },
    /// Liveness probe, both directions (2s interval; 3 missed = suspect,
    /// 10s = dead — §5 of the product spec).
    Heartbeat { epoch: Epoch },
    /// Opaque engine traffic (GGML RPC semantics) for an active epoch.
    /// The mesh delivers these on dedicated streams; the envelope form exists
    /// for control-channel fallback and testing.
    EngineFrame { epoch: Epoch, payload: Vec<u8> },
    /// Polite shutdown notice: node is draining (battery policy, `stop`).
    Draining { epoch: Epoch, reason: String },
}

/// Wire envelope: message plus the sender's view of the active epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub proto_version: u16,
    pub message: Message,
}

impl Envelope {
    pub fn new(message: Message) -> Self {
        Envelope {
            proto_version: crate::PROTO_VERSION,
            message,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Epoch;

    #[test]
    fn envelope_roundtrip() {
        let env = Envelope::new(Message::Heartbeat { epoch: Epoch(3) });
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        assert_eq!(back.proto_version, crate::PROTO_VERSION);
        assert!(matches!(
            back.message,
            Message::Heartbeat { epoch: Epoch(3) }
        ));
    }
}
