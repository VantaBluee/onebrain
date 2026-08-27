//! Top-level envelope for all internode messages, plus the per-stream
//! [`StreamHeader`] that types every mesh bi-stream.
//!
//! Every control payload rides in an [`Envelope`] carrying the sender's
//! active plan epoch, which lets receivers fence stale traffic uniformly
//! instead of each handler re-implementing the check.
//!
//! # Mesh application close codes (authoritative list)
//!
//! Mesh connections and streams are closed with these QUIC application error
//! codes. This list is the single source of truth; `onebrain-mesh` enforces
//! the codes but documents them by reference to here.
//!
//! - `0` — normal close (duplicate connection, shutdown, pairing done).
//! - `1` — `unpaired`: the remote endpoint is not in the peer store.
//! - `2` — `no-pairing-window`: pair-ALPN connection outside an open window.
//! - `3` — `incompatible`: the `Hello` handshake judged the peers
//!   incompatible (protocol or engine build mismatch).
//! - `4` — `bad-epoch`: an `rpc` stream arrived whose [`StreamHeader`]
//!   `epoch` is not the receiving worker's active epoch, or the sender is
//!   not that epoch's head. The stream is refused before any RPC byte is
//!   bridged (the M3 fencing rule in `docs/distributed.md`).

use serde::{Deserialize, Serialize};

use crate::handshake::Hello;
use crate::plan::{Epoch, Plan};

/// Purpose of a mesh bi-stream, declared by its first frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamKind {
    /// Envelope traffic (Hello, plans, heartbeats) — the existing behavior.
    Control,
    /// One tunneled GGML RPC session for the active epoch. Refused with
    /// close code `4` (`bad-epoch`) unless the header's epoch matches the
    /// worker's active epoch and the sender is that epoch's head.
    Rpc,
    /// Link probing (RTT/bandwidth measurement) traffic.
    Probe,
}

/// First frame on any mesh bi-stream, postcard-encoded: what the stream is
/// for and which plan epoch it belongs to. Streams for epochs other than the
/// receiver's active epoch are fenced (close code `4`, `bad-epoch`).
///
/// `Control` and `Probe` streams are not epoch-scoped; senders put their
/// current view of the active epoch (or [`Epoch`]`(0)` before any plan) in
/// `epoch` and receivers ignore it for those kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamHeader {
    pub kind: StreamKind,
    pub epoch: Epoch,
}

/// One compute device on a node, as reported in [`Message::NodeStatus`].
/// Memory figures are a snapshot taken off the hot path — never refreshed
/// per-token (probing an RPC device live is a network round trip).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceBrief {
    /// Backend kind, e.g. `"cpu"`, `"metal"`, `"cuda"`, `"vulkan"`.
    pub kind: String,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

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
    /// Worker → head, sent after `Hello`: the memory the scheduler may
    /// budget against. `usable_memory_bytes` is measured free memory of the
    /// chosen device minus a fixed OS reserve — never total RAM (§3
    /// auto-solo rule). Gated on the `PIPELINE_PARALLEL` capability bit.
    ///
    /// M4 (`PROTO_VERSION` 2) added the microbench profile fields in place —
    /// an exception to the "new fields need a new message kind" rule that is
    /// legal because the engine build-hash gate guarantees both ends of any
    /// cluster run the same build (docs/scheduler-v1.md); the version bump
    /// keeps the handshake refusal message truthful for genuinely mixed
    /// builds. `None` means the node has not run its profile yet (the
    /// scheduler then falls back to memory-only weighting).
    NodeStatus {
        usable_memory_bytes: u64,
        devices: Vec<DeviceBrief>,
        /// Measured prefill throughput (tokens/sec) from the compute
        /// microbench, if profiled.
        prefill_tps: Option<f64>,
        /// Measured decode throughput (tokens/sec) from the compute
        /// microbench, if profiled.
        decode_tps: Option<f64>,
        /// Measured sequential disk read rate (MB/s), if profiled. An upper
        /// bound (OS page cache); used only for relative ordering.
        disk_mbps: Option<f64>,
    },
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

    #[test]
    fn stream_header_roundtrip_all_kinds() {
        for kind in [StreamKind::Control, StreamKind::Rpc, StreamKind::Probe] {
            let header = StreamHeader {
                kind,
                epoch: Epoch(42),
            };
            let bytes = crate::encode(&header).unwrap();
            let back: StreamHeader = crate::decode(&bytes).unwrap();
            assert_eq!(header, back);
        }
    }

    #[test]
    fn node_status_roundtrip() {
        let env = Envelope::new(Message::NodeStatus {
            usable_memory_bytes: 12 << 30,
            devices: vec![
                DeviceBrief {
                    kind: "cuda".into(),
                    free_bytes: 10 << 30,
                    total_bytes: 16 << 30,
                },
                DeviceBrief {
                    kind: "cpu".into(),
                    free_bytes: 20 << 30,
                    total_bytes: 32 << 30,
                },
            ],
            prefill_tps: Some(812.5),
            decode_tps: Some(41.25),
            disk_mbps: Some(1732.0),
        });
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        match back.message {
            Message::NodeStatus {
                usable_memory_bytes,
                devices,
                prefill_tps,
                decode_tps,
                disk_mbps,
            } => {
                assert_eq!(usable_memory_bytes, 12 << 30);
                assert_eq!(devices.len(), 2);
                assert_eq!(devices[0].kind, "cuda");
                assert_eq!(devices[1].free_bytes, 20 << 30);
                assert_eq!(prefill_tps, Some(812.5));
                assert_eq!(decode_tps, Some(41.25));
                assert_eq!(disk_mbps, Some(1732.0));
            }
            other => panic!("expected NodeStatus, got {other:?}"),
        }
    }

    #[test]
    fn node_status_roundtrip_unprofiled() {
        // A node that has not run its microbench yet reports None for every
        // profile field; the scheduler treats that as memory-only weighting.
        let env = Envelope::new(Message::NodeStatus {
            usable_memory_bytes: 8 << 30,
            devices: vec![],
            prefill_tps: None,
            decode_tps: None,
            disk_mbps: None,
        });
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        match back.message {
            Message::NodeStatus {
                usable_memory_bytes,
                prefill_tps,
                decode_tps,
                disk_mbps,
                ..
            } => {
                assert_eq!(usable_memory_bytes, 8 << 30);
                assert_eq!(prefill_tps, None);
                assert_eq!(decode_tps, None);
                assert_eq!(disk_mbps, None);
            }
            other => panic!("expected NodeStatus, got {other:?}"),
        }
    }
}
