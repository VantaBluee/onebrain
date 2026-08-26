//! Versioned wire protocol types for OneBrain.
//!
//! Every message that crosses a machine boundary is defined here, carries a
//! protocol version, and is encoded with postcard (unknown-field tolerance is
//! handled by capability negotiation at handshake rather than schema evolution
//! within a message: new message kinds require a new capability bit).

pub mod capabilities;
pub mod handshake;
pub mod message;
pub mod plan;

/// Human-visible product name. A rebrand changes this constant, the binary
/// name in `onebrain-cli`, and workspace metadata — nothing else.
pub const PRODUCT_NAME: &str = "OneBrain";

/// Wire protocol version. Bumped on any breaking change to the types in this
/// crate. Nodes with different protocol versions refuse to form a cluster and
/// tell the user which node to update (see `handshake`).
pub const PROTO_VERSION: u16 = 1;

/// Errors produced while encoding or decoding protocol messages.
#[derive(Debug, thiserror::Error)]
pub enum ProtoError {
    #[error("failed to encode message: {0}")]
    Encode(postcard::Error),
    #[error("failed to decode message: {0}")]
    Decode(postcard::Error),
}

/// Encode a protocol message to bytes.
pub fn encode<T: serde::Serialize>(msg: &T) -> Result<Vec<u8>, ProtoError> {
    postcard::to_stdvec(msg).map_err(ProtoError::Encode)
}

/// Decode a protocol message from bytes.
pub fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, ProtoError> {
    postcard::from_bytes(bytes).map_err(ProtoError::Decode)
}
