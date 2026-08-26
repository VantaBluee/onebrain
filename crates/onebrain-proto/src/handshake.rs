//! Node handshake: the first exchange on any authenticated mesh connection.
//!
//! The handshake carries everything needed to refuse incompatible peers with
//! an actionable error instead of failing mysteriously mid-inference:
//! protocol version, capability bits, and the engine build hash (llama.cpp
//! commit + backend flags + proto version). Cross-version RPC is never
//! attempted (§1.3, §3 of the product spec).

use serde::{Deserialize, Serialize};

use crate::capabilities::Capabilities;

/// Identifies the exact engine a node runs. Built at compile time from the
/// vendored llama.cpp commit, the enabled backend feature set, and
/// `PROTO_VERSION`. Two nodes may cooperate on a plan only if equal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineBuildHash(pub String);

/// Hello message sent by both sides of a fresh connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hello {
    pub proto_version: u16,
    pub capabilities: Capabilities,
    pub engine_build: EngineBuildHash,
    /// Human-readable node name shown in `status` and the dashboard.
    pub node_name: String,
    /// Semver of the OneBrain build, for "update node X" error messages.
    pub product_version: String,
}

/// Outcome of comparing two `Hello`s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandshakeVerdict {
    Compatible,
    /// Peers disagree on wire protocol; names the node that must update.
    ProtoMismatch {
        ours: u16,
        theirs: u16,
    },
    /// Same protocol but different engine builds; distributed inference
    /// would hit undefined tensor semantics, so it is refused outright.
    EngineMismatch {
        ours: String,
        theirs: String,
    },
}

/// Compare handshakes. The caller turns the verdict into a typed error with
/// the concrete remedy ("run `onebrain doctor --self-update` on <node>").
pub fn judge(ours: &Hello, theirs: &Hello) -> HandshakeVerdict {
    if ours.proto_version != theirs.proto_version {
        return HandshakeVerdict::ProtoMismatch {
            ours: ours.proto_version,
            theirs: theirs.proto_version,
        };
    }
    if ours.engine_build != theirs.engine_build {
        return HandshakeVerdict::EngineMismatch {
            ours: ours.engine_build.0.clone(),
            theirs: theirs.engine_build.0.clone(),
        };
    }
    HandshakeVerdict::Compatible
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hello(proto: u16, engine: &str) -> Hello {
        Hello {
            proto_version: proto,
            capabilities: Capabilities::current(),
            engine_build: EngineBuildHash(engine.to_string()),
            node_name: "test".into(),
            product_version: "0.1.0".into(),
        }
    }

    #[test]
    fn same_build_is_compatible() {
        let a = hello(1, "abc");
        assert_eq!(judge(&a, &a.clone()), HandshakeVerdict::Compatible);
    }

    #[test]
    fn proto_mismatch_wins_over_engine_mismatch() {
        let a = hello(1, "abc");
        let b = hello(2, "def");
        assert!(matches!(
            judge(&a, &b),
            HandshakeVerdict::ProtoMismatch { .. }
        ));
    }

    #[test]
    fn engine_mismatch_detected() {
        let a = hello(1, "abc");
        let b = hello(1, "def");
        assert!(matches!(
            judge(&a, &b),
            HandshakeVerdict::EngineMismatch { .. }
        ));
    }
}
