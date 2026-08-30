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
    /// The mesh surfaces the arrival as a `Draining` peer event AND forwards
    /// the envelope to the control consumer — the head needs `epoch` to run
    /// the failure lifecycle (docs/resilience.md).
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
    /// scheduler then falls back to memory-only weighting). M5
    /// (`PROTO_VERSION` 3) added `draining` in place under the same rule.
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
        /// `true` while the node's battery policy says new plans should
        /// avoid it (below the drain threshold and not on AC — M5,
        /// docs/resilience.md). The scheduler excludes draining nodes from
        /// new plans unless a plan is infeasible without them.
        #[serde(default)]
        draining: bool,
    },
    /// Ask a peer which byte ranges of `model` it can serve over the blobs
    /// ALPN (M6 P2P weight sharing, docs/logistics.md). Appended as a NEW
    /// message kind (postcard keeps earlier variant tags stable) and gated
    /// on the `BLOB_SHARING` capability bit; `PROTO_VERSION` 4 accompanies
    /// it per the M6 logistics contract. The peer answers on the SAME
    /// control stream with [`Message::RangeInventory`] — the mesh handles
    /// the exchange like a heartbeat echo, so requesters never have to
    /// correlate replies out of the shared control-consumer firehose.
    RangeQuery {
        /// Model reference exactly as it appears in the range manifest
        /// (`ranges.json`) — a cache key both sides derive the same way.
        model: String,
    },
    /// Reply to [`Message::RangeQuery`]: every byte range of `model` the
    /// sender can serve over blobs. Empty `ranges` = the peer has none (the
    /// downloader then falls back to WAN for the whole file). Introduced
    /// with `PROTO_VERSION` 4 / `BLOB_SHARING` alongside `RangeQuery`.
    RangeInventory {
        /// The queried model reference, echoed back.
        model: String,
        /// Total size in bytes of the complete model file, so the receiver
        /// can sanity-check range bounds against its own manifest. `0` when
        /// the peer has nothing.
        total_size: u64,
        /// `(start, end, blake3)` per available range — exclusive `end`,
        /// hash of the range's bytes. Each hash doubles as the iroh-blobs
        /// blob address (both are plain BLAKE3 of content).
        ranges: Vec<(u64, u64, [u8; 32])>,
    },
    /// Ask a peer to run its compute/disk microbench (the M4 `onebrain
    /// bench` measurement) on demand and report the result — M7 `onebrain
    /// bench --cluster`, docs/perf.md §10. Appended as a NEW message kind
    /// (postcard keeps earlier variant tags stable) and gated on the
    /// `CLUSTER_BENCH` capability bit; `PROTO_VERSION` 5 accompanies it per
    /// the M6 precedent. The peer answers on the SAME control stream with
    /// [`Message::BenchReport`] — the mesh handles the exchange like a
    /// heartbeat echo (the `RangeQuery` pattern), so requesters never have
    /// to correlate replies out of the shared control-consumer firehose.
    BenchRequest {},
    /// Reply to [`Message::BenchRequest`]: the sender's fresh microbench
    /// figures. `measured_unix == 0` is the reserved cannot-bench-now
    /// marker (no bench wired up, or the node is busy generating); the
    /// throughput fields are meaningless then and the caller must treat the
    /// peer as unbenchable this round — never as measuring 0 tok/s.
    /// Introduced with `PROTO_VERSION` 5 / `CLUSTER_BENCH` alongside
    /// `BenchRequest`.
    BenchReport {
        /// Measured prefill throughput (tokens/sec).
        prefill_tps: f64,
        /// Measured decode throughput (tokens/sec).
        decode_tps: f64,
        /// Measured sequential disk read rate (MB/s). An upper bound (OS
        /// page cache) used only for relative ordering, exactly like
        /// `NodeStatus.disk_mbps`.
        disk_mbps: f64,
        /// Unix seconds when the microbench finished; `0` = the
        /// cannot-bench-now marker described above.
        measured_unix: u64,
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
            draining: true,
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
                draining,
            } => {
                assert_eq!(usable_memory_bytes, 12 << 30);
                assert_eq!(devices.len(), 2);
                assert_eq!(devices[0].kind, "cuda");
                assert_eq!(devices[1].free_bytes, 20 << 30);
                assert_eq!(prefill_tps, Some(812.5));
                assert_eq!(decode_tps, Some(41.25));
                assert_eq!(disk_mbps, Some(1732.0));
                assert!(draining, "the M5 draining flag must survive roundtrip");
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
            draining: false,
        });
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        match back.message {
            Message::NodeStatus {
                usable_memory_bytes,
                prefill_tps,
                decode_tps,
                disk_mbps,
                draining,
                ..
            } => {
                assert_eq!(usable_memory_bytes, 8 << 30);
                assert_eq!(prefill_tps, None);
                assert_eq!(decode_tps, None);
                assert_eq!(disk_mbps, None);
                assert!(!draining, "draining defaults to false");
            }
            other => panic!("expected NodeStatus, got {other:?}"),
        }
    }

    #[test]
    fn range_query_roundtrip() {
        let env = Envelope::new(Message::RangeQuery {
            model: "blake3:abc123".into(),
        });
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        match back.message {
            Message::RangeQuery { model } => assert_eq!(model, "blake3:abc123"),
            other => panic!("expected RangeQuery, got {other:?}"),
        }
    }

    #[test]
    fn range_inventory_roundtrip() {
        // Non-trivial hashes: every byte position must survive the wire —
        // a truncated hash would address the wrong blob.
        let h1: [u8; 32] = core::array::from_fn(|i| i as u8);
        let h2: [u8; 32] = core::array::from_fn(|i| 255 - i as u8);
        let env = Envelope::new(Message::RangeInventory {
            model: "blake3:abc123".into(),
            total_size: 7 << 30,
            ranges: vec![(0, 4096, h1), (4096, 7 << 30, h2)],
        });
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        match back.message {
            Message::RangeInventory {
                model,
                total_size,
                ranges,
            } => {
                assert_eq!(model, "blake3:abc123");
                assert_eq!(total_size, 7 << 30);
                assert_eq!(ranges, vec![(0, 4096, h1), (4096, 7 << 30, h2)]);
            }
            other => panic!("expected RangeInventory, got {other:?}"),
        }
    }

    #[test]
    fn range_inventory_roundtrip_empty_means_peer_has_none() {
        // The contract's "empty ranges = peer has none" case must encode
        // and decode cleanly (the downloader treats it as all-WAN).
        let env = Envelope::new(Message::RangeInventory {
            model: "blake3:missing".into(),
            total_size: 0,
            ranges: Vec::new(),
        });
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        match back.message {
            Message::RangeInventory {
                model,
                total_size,
                ranges,
            } => {
                assert_eq!(model, "blake3:missing");
                assert_eq!(total_size, 0);
                assert!(ranges.is_empty());
            }
            other => panic!("expected RangeInventory, got {other:?}"),
        }
    }

    #[test]
    fn bench_request_roundtrip() {
        let env = Envelope::new(Message::BenchRequest {});
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        assert!(matches!(back.message, Message::BenchRequest {}));
    }

    #[test]
    fn bench_report_roundtrip() {
        // Values mirror the M4 microbench shape carried by NodeStatus; every
        // field must survive the wire bit-exact (f64 throughputs included).
        let env = Envelope::new(Message::BenchReport {
            prefill_tps: 812.5,
            decode_tps: 41.25,
            disk_mbps: 1732.0,
            measured_unix: 1_756_252_800,
        });
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        match back.message {
            Message::BenchReport {
                prefill_tps,
                decode_tps,
                disk_mbps,
                measured_unix,
            } => {
                assert_eq!(prefill_tps, 812.5);
                assert_eq!(decode_tps, 41.25);
                assert_eq!(disk_mbps, 1732.0);
                assert_eq!(measured_unix, 1_756_252_800);
            }
            other => panic!("expected BenchReport, got {other:?}"),
        }
    }

    #[test]
    fn bench_report_roundtrip_unavailable_marker() {
        // The wire contract's "peer cannot bench right now" reply is
        // measured_unix == 0; it must encode/decode cleanly so callers can
        // distinguish "no data" from a real measurement.
        let env = Envelope::new(Message::BenchReport {
            prefill_tps: 0.0,
            decode_tps: 0.0,
            disk_mbps: 0.0,
            measured_unix: 0,
        });
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        match back.message {
            Message::BenchReport { measured_unix, .. } => {
                assert_eq!(measured_unix, 0, "the marker must survive the wire");
            }
            other => panic!("expected BenchReport, got {other:?}"),
        }
    }

    #[test]
    fn draining_roundtrip_keeps_the_epoch() {
        // The head consumes the epoch field to fence stale drains, so it must
        // survive the wire (docs/resilience.md worker-side drain).
        let env = Envelope::new(Message::Draining {
            epoch: Epoch(7),
            reason: "battery below threshold".into(),
        });
        let bytes = crate::encode(&env).unwrap();
        let back: Envelope = crate::decode(&bytes).unwrap();
        match back.message {
            Message::Draining { epoch, reason } => {
                assert_eq!(epoch, Epoch(7));
                assert_eq!(reason, "battery below threshold");
            }
            other => panic!("expected Draining, got {other:?}"),
        }
    }
}
