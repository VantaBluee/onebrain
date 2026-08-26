//! Model logistics: registry, GGUF header parsing, range-fetch shard
//! downloads, peer-to-peer weight sharing, and integrity manifests.
//!
//! M0 ships the GGUF header reader (`gguf`), which everything else builds
//! on: it enumerates every tensor's byte range so a node can fetch only its
//! assigned layers via HTTP range requests (§6). Downloads land in M1,
//! shard-only fetch in M3, p2p sharing in M6.

pub mod gguf;

#[derive(Debug, thiserror::Error)]
pub enum ModelsError {
    #[error("gguf parse error: {0}")]
    Gguf(#[from] gguf::GgufError),
    #[error("model downloads are not implemented yet (arrives in milestone M1)")]
    NotImplemented,
}
