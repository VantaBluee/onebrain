//! P2P weight-range sharing over iroh-blobs (M6, docs/logistics.md).
//!
//! Each shared item is one range file on disk whose iroh-blobs address IS the
//! range manifest hash: iroh-blobs addresses a raw blob by the plain BLAKE3
//! of its content, and the range manifest stores exactly that hash. (Large
//! blobs additionally carry a bao outboard for verified streaming, but that
//! is transfer encoding — the identity stays the content's BLAKE3.)
//!
//! The provider is hosted on the EXISTING mesh endpoint under the iroh-blobs
//! ALPN ([`ALPN_BLOBS`]) — no new sockets, no raw TCP listener (spec §10).
//! The accept path enforces the same rule as the mesh ALPN: connections from
//! endpoint ids missing from the peer store are closed with code 1
//! (`unpaired`) before a single protocol byte is exchanged.
//!
//! Which ranges a peer holds is negotiated over the control plane: the
//! daemon calls [`crate::MeshHandle::range_query`], the peer's mesh service
//! answers from its configured [`RangeInventorySource`], and the actual
//! bytes then move via [`crate::MeshHandle::fetch_blob`].

use std::path::PathBuf;

use iroh::endpoint::Connection;
use iroh::{Endpoint, EndpointAddr};
use iroh_blobs::api::blobs::{AddPathOptions, ImportMode};
use iroh_blobs::api::Store;
use iroh_blobs::store::fs::FsStore;
use iroh_blobs::store::mem::MemStore;
use iroh_blobs::{BlobFormat, BlobsProtocol, Hash};
use tracing::debug;

use crate::MeshError;

/// ALPN of the iroh-blobs provider hosted on the mesh endpoint. Accepts
/// require the remote id to be in the peer store — the same §10 rule as
/// [`crate::ALPN_MESH`]; unknown peers are closed with code 1 (`unpaired`).
pub const ALPN_BLOBS: &[u8] = iroh_blobs::ALPN;

/// One available byte range of a model file: `(start, end, blake3)` with
/// exclusive `end`. The hash is both the range manifest's integrity check
/// and the blob address — the plain-tuple shape is the M6 contract type,
/// shared verbatim with `onebrain_proto::message::Message::RangeInventory`.
pub type RangeEntry = (u64, u64, [u8; 32]);

/// Answers `RangeQuery` control messages: which byte ranges of a model this
/// node can serve over blobs. The daemon wires the models-crate range cache
/// into this; the mesh only relays.
///
/// Implementations should be cheap (a cached manifest read at most): the
/// mesh calls this off the async runtime via `spawn_blocking`, but a peer is
/// actively waiting on the reply.
pub trait RangeInventorySource: Send + Sync {
    /// `(total_size, ranges)` of `model`. `None` when this node holds
    /// nothing for `model` — the mesh then replies with an empty inventory
    /// (the wire contract's "peer has none").
    fn inventory(&self, model: &str) -> Option<(u64, Vec<RangeEntry>)>;
}

/// A peer's answer to [`crate::MeshHandle::range_query`]: the ranges of a
/// model it can serve over blobs. `ranges` empty = the peer has none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerRangeInventory {
    /// Total size in bytes of the complete model file as the peer knows it
    /// (`0` when the peer holds nothing).
    pub total_size: u64,
    /// `(start, end, blake3)` per range the peer can serve — exclusive
    /// `end`; the hash is both the integrity check and the blob address.
    pub ranges: Vec<RangeEntry>,
}

/// The mesh's blob store plus its wire protocol handler. Cloned freely: both
/// halves are handles onto one store actor.
#[derive(Debug, Clone)]
pub(crate) struct BlobStore {
    store: Store,
    protocol: BlobsProtocol,
}

impl BlobStore {
    /// Open the blob store: on-disk (persistent, references shared files in
    /// place) when the daemon configured a directory, in-memory otherwise
    /// (tests, embedders that never share weights).
    pub(crate) async fn open(dir: Option<&PathBuf>) -> Result<Self, MeshError> {
        let store: Store = match dir {
            Some(dir) => FsStore::load(dir)
                .await
                .map_err(|err| MeshError::BlobStore {
                    detail: format!("could not open blob store at {}: {err}", dir.display()),
                })?
                .into(),
            None => MemStore::new().into(),
        };
        let protocol = BlobsProtocol::new(&store, None);
        Ok(BlobStore { store, protocol })
    }

    /// The provider-side protocol handler for accepted [`ALPN_BLOBS`]
    /// connections (pairing is enforced by the caller BEFORE handing a
    /// connection over).
    pub(crate) fn protocol(&self) -> BlobsProtocol {
        self.protocol.clone()
    }

    /// Share one file: import it into the store (referenced in place where
    /// the store supports it — no duplication for on-disk stores) and return
    /// its BLAKE3 hash, which paired peers use as the blob address. The hash
    /// equals `blake3(file contents)`, i.e. the range manifest hash.
    pub(crate) async fn share_file(&self, path: PathBuf) -> Result<[u8; 32], MeshError> {
        let display = path.display().to_string();
        let tag = self
            .store
            .blobs()
            .add_path_with_opts(AddPathOptions {
                path,
                format: BlobFormat::Raw,
                mode: ImportMode::TryReference,
            })
            .await
            .map_err(|err| MeshError::BlobStore {
                detail: format!("could not share {display}: {err}"),
            })?;
        Ok(*tag.hash.as_bytes())
    }

    /// Fetch the blob `hash` from `addr` over [`ALPN_BLOBS`] and write it to
    /// `target`. Returns the bytes read from the network (0 when the blob
    /// was already complete locally); the file lands at `target` either way.
    /// The blob also stays in the local store, so this node can serve it to
    /// further peers — how a model spreads through a cluster without any
    /// node re-downloading from the WAN (spec §6).
    pub(crate) async fn fetch_into(
        &self,
        ep: &Endpoint,
        addr: EndpointAddr,
        hash: [u8; 32],
        target: PathBuf,
    ) -> Result<u64, MeshError> {
        let peer = addr.id.fmt_short().to_string();
        let hash = Hash::from_bytes(hash);
        let conn: Connection =
            ep.connect(addr, ALPN_BLOBS)
                .await
                .map_err(|err| MeshError::BlobFetch {
                    detail: format!("could not reach {peer} on the blobs ALPN: {err}"),
                })?;
        let stats = self
            .store
            .remote()
            .fetch(conn.clone(), hash)
            .await
            .map_err(|err| MeshError::BlobFetch {
                detail: format!("transfer of {hash} from {peer} failed: {err}"),
            })?;
        conn.close(0u32.into(), b"fetch-done");
        self.store
            .blobs()
            .export(hash, &target)
            .await
            .map_err(|err| MeshError::BlobStore {
                detail: format!("could not write {hash} to {}: {err}", target.display()),
            })?;
        debug!(
            %hash,
            peer,
            bytes = stats.total_bytes_read(),
            target = %target.display(),
            "blob fetched from peer"
        );
        Ok(stats.total_bytes_read())
    }

    /// Flush and stop the store actor (called from the mesh service's
    /// shutdown; errors are the caller's to log — shutdown must not fail).
    pub(crate) async fn shutdown(&self) {
        if let Err(err) = self.store.shutdown().await {
            debug!("blob store shutdown reported: {err}");
        }
    }
}
