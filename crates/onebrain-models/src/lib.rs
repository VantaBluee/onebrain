//! Model logistics: registry, GGUF header parsing, range-fetch shard
//! downloads, peer-to-peer weight sharing, and integrity manifests.
//!
//! M0 shipped the GGUF header reader (`gguf`). M1 added the embedded model
//! registry (`registry`), full-file resumable downloads with BLAKE3
//! manifests (`download`), and the on-disk cache view (`cache`). M6 adds
//! the tensor-aligned range store (`ranges`), split-GGUF part naming
//! (`split`), and the LRU + pin GC primitives in `cache`
//! (docs/logistics.md is the binding contract).

pub mod cache;
pub mod download;
pub mod gguf;
pub mod ranges;
pub mod registry;
pub mod split;

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
    #[error(transparent)]
    Ranges(#[from] ranges::RangeError),
}
