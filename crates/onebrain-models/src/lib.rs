//! Model logistics: registry, GGUF header parsing, range-fetch shard
//! downloads, peer-to-peer weight sharing, and integrity manifests.
//!
//! M0 shipped the GGUF header reader (`gguf`). M1 adds the embedded model
//! registry (`registry`), full-file resumable downloads with BLAKE3
//! manifests (`download`), and the on-disk cache view (`cache`). Shard-only
//! range fetch arrives in M3, p2p sharing in M6.

pub mod cache;
pub mod download;
pub mod gguf;
pub mod registry;

#[derive(Debug, thiserror::Error)]
pub enum ModelsError {
    #[error("gguf parse error: {0}")]
    Gguf(#[from] gguf::GgufError),
    #[error(transparent)]
    Registry(#[from] registry::RegistryError),
    #[error(transparent)]
    Download(#[from] download::DownloadError),
    #[error(transparent)]
    Cache(#[from] cache::CacheError),
}
