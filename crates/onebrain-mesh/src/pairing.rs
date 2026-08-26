//! The pairing exchange (ALPN `onebrain/pair/1`) and the length-prefixed
//! frame codec shared with the heartbeat streams.
//!
//! Protocol per `onebrain-proto::pair` and `docs/mesh.md`, over one bi
//! stream opened by the joiner:
//!
//! 1. Both sides send [`PairMessage::Pake`] (SPAKE2 Ed25519, symmetric
//!    mode; password = the 6-digit code, identity = `PAIR_CONFIRM_CONTEXT`).
//! 2. The joiner sends its [`PairMessage::Confirm`] MAC first; the host
//!    verifies it (constant time) BEFORE revealing its own confirm, so a
//!    codeless dialer cannot reflect the host's MAC back at it. A mismatch
//!    is answered with [`PairMessage::Rejected`] and burns one of the
//!    host's three window attempts.
//! 3. On success both sides exchange [`PairMessage::Introduce`] and the
//!    caller persists the peer.
//!
//! MACs are direction-bound (`PairRole::{Host,Joiner}` is hashed into the
//! keyed transcript), so a party that does not know the code can never
//! reflect a received MAC back as its own — in either role. The joiner
//! still confirms first, keeping the host's 3-attempt budget authoritative
//! for guess-rate limiting.

use iroh::endpoint::{Connection, RecvStream, SendStream};
use iroh::EndpointId;
use serde::de::DeserializeOwned;
use serde::Serialize;
use spake2::{Ed25519Group, Identity, Password, Spake2};

use onebrain_proto::pair::{confirm_mac, PairMessage, PairRole, PAIR_CONFIRM_CONTEXT};

use crate::MeshError;

/// Upper bound on a single frame; anything larger is a protocol violation.
pub(crate) const MAX_FRAME_BYTES: u32 = 1 << 20;

/// Map any stream-level failure into [`MeshError::Stream`].
pub(crate) fn stream_err(err: impl std::fmt::Display) -> MeshError {
    MeshError::Stream {
        detail: err.to_string(),
    }
}

/// Write one postcard-encoded frame: u32-le length prefix + payload.
pub(crate) async fn write_frame<T: Serialize>(
    tx: &mut SendStream,
    msg: &T,
) -> Result<(), MeshError> {
    let bytes = onebrain_proto::encode(msg)?;
    if bytes.len() as u64 > u64::from(MAX_FRAME_BYTES) {
        return Err(MeshError::Stream {
            detail: format!("refusing to send oversized frame ({} bytes)", bytes.len()),
        });
    }
    tx.write_all(&(bytes.len() as u32).to_le_bytes())
        .await
        .map_err(stream_err)?;
    tx.write_all(&bytes).await.map_err(stream_err)?;
    Ok(())
}

/// Read one postcard-encoded frame (u32-le length prefix + payload).
pub(crate) async fn read_frame<T: DeserializeOwned>(rx: &mut RecvStream) -> Result<T, MeshError> {
    let mut len_buf = [0u8; 4];
    rx.read_exact(&mut len_buf).await.map_err(stream_err)?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        return Err(MeshError::Stream {
            detail: format!("peer sent oversized frame ({len} bytes)"),
        });
    }
    let mut buf = vec![0u8; len as usize];
    rx.read_exact(&mut buf).await.map_err(stream_err)?;
    Ok(onebrain_proto::decode(&buf)?)
}

/// Generate a uniform 6-digit pairing code (leading zeros allowed) from the
/// OS RNG.
pub(crate) fn generate_code() -> Result<String, MeshError> {
    // Rejection-sample so `% 1_000_000` is uniform.
    const LIMIT: u32 = (u32::MAX / 1_000_000) * 1_000_000;
    loop {
        let v = getrandom::u32().map_err(|e| MeshError::Rng(e.to_string()))?;
        if v < LIMIT {
            return Ok(format!("{:06}", v % 1_000_000));
        }
    }
}

/// Validate joiner input as a 6-digit code.
pub(crate) fn validate_code(code: &str) -> Result<String, MeshError> {
    let trimmed = code.trim();
    if trimmed.len() == 6 && trimmed.bytes().all(|b| b.is_ascii_digit()) {
        Ok(trimmed.to_string())
    } else {
        Err(MeshError::BadPairTarget {
            input: truncate_for_display(code),
        })
    }
}

/// Keep error messages readable when the input was a pasted blob.
pub(crate) fn truncate_for_display(input: &str) -> String {
    const MAX: usize = 48;
    if input.len() <= MAX {
        input.to_string()
    } else {
        let cut = input
            .char_indices()
            .take_while(|(i, _)| *i < MAX)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        format!("{}…", &input[..cut])
    }
}

/// Finish the send half and wait (bounded) until the peer acknowledges all
/// data. The pairing connection is closed right after each attempt, and
/// closing a QUIC connection discards in-flight stream data — without this
/// drain the peer can lose the final `Rejected`/`Introduce` frame.
async fn flush_stream(tx: &mut SendStream) {
    let _ = tx.finish();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), tx.stopped()).await;
}

/// Constant-time 32-byte comparison.
fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// The other device, as learned from a successful pairing.
#[derive(Debug, Clone)]
pub(crate) struct PairOutcome {
    /// Authenticated endpoint id (from the QUIC handshake).
    pub peer_id: EndpointId,
    /// The peer's introduced node name.
    pub node_name: String,
    /// The peer's OneBrain version (for logs).
    pub product_version: String,
}

/// Run SPAKE2 and return the derived 32-byte key, exchanging `Pake` frames.
async fn pake_exchange(
    tx: &mut SendStream,
    rx: &mut RecvStream,
    code: &str,
) -> Result<[u8; 32], MeshError> {
    let (state, outbound) = Spake2::<Ed25519Group>::start_symmetric(
        &Password::new(code.as_bytes()),
        &Identity::new(PAIR_CONFIRM_CONTEXT.as_bytes()),
    );
    write_frame(tx, &PairMessage::Pake { body: outbound }).await?;
    let inbound = match read_frame::<PairMessage>(rx).await? {
        PairMessage::Pake { body } => body,
        PairMessage::Rejected { reason } => return Err(MeshError::PairRejected { reason }),
        _ => {
            return Err(MeshError::PairRejected {
                reason: "peer broke the pairing protocol (expected a key-exchange message)".into(),
            })
        }
    };
    let key = state
        .finish(&inbound)
        .map_err(|_| MeshError::PairRejected {
            reason: "peer sent an invalid key-exchange message".into(),
        })?;
    key.as_slice()
        .try_into()
        .map_err(|_| MeshError::PairRejected {
            reason: "key exchange produced an unexpected key length".into(),
        })
}

/// Host side of one pairing attempt: the joiner dialed us and opens the bi
/// stream. Returns the introduced peer on success; any `Err` burns one of
/// the window's attempts.
pub(crate) async fn host_attempt(
    conn: &Connection,
    code: &str,
    local_id: EndpointId,
    local_name: &str,
) -> Result<PairOutcome, MeshError> {
    let peer_id = conn.remote_id();
    let (mut tx, mut rx) = conn.accept_bi().await.map_err(stream_err)?;
    let key = pake_exchange(&mut tx, &mut rx, code).await?;
    // MACs are role-bound: we send the Host MAC and verify the Joiner MAC,
    // so neither side can ever reflect the other's confirm.
    let mine = confirm_mac(
        &key,
        PairRole::Host,
        &local_id.to_string(),
        &peer_id.to_string(),
    );
    let expected = confirm_mac(
        &key,
        PairRole::Joiner,
        &local_id.to_string(),
        &peer_id.to_string(),
    );

    // The joiner must prove knowledge of the code first.
    let their_mac = match read_frame::<PairMessage>(&mut rx).await? {
        PairMessage::Confirm { mac } => mac,
        PairMessage::Rejected { reason } => return Err(MeshError::PairRejected { reason }),
        _ => {
            return Err(MeshError::PairRejected {
                reason: "peer broke the pairing protocol (expected a confirm)".into(),
            })
        }
    };
    if !ct_eq(&their_mac, &expected) {
        let _ = write_frame(
            &mut tx,
            &PairMessage::Rejected {
                reason: "wrong pairing code".into(),
            },
        )
        .await;
        flush_stream(&mut tx).await;
        return Err(MeshError::PairRejected {
            reason: "wrong pairing code".into(),
        });
    }
    write_frame(&mut tx, &PairMessage::Confirm { mac: mine }).await?;
    write_frame(
        &mut tx,
        &PairMessage::Introduce {
            node_name: local_name.to_string(),
            product_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    )
    .await?;
    let (node_name, product_version) = read_introduce(&mut rx).await?;
    flush_stream(&mut tx).await;
    Ok(PairOutcome {
        peer_id,
        node_name,
        product_version,
    })
}

/// Joiner side of a pairing attempt: dial happened, we open the bi stream.
pub(crate) async fn joiner_attempt(
    conn: &Connection,
    code: &str,
    local_id: EndpointId,
    local_name: &str,
) -> Result<PairOutcome, MeshError> {
    let peer_id = conn.remote_id();
    let (mut tx, mut rx) = conn.open_bi().await.map_err(stream_err)?;
    let key = pake_exchange(&mut tx, &mut rx, code).await?;
    let mine = confirm_mac(
        &key,
        PairRole::Joiner,
        &local_id.to_string(),
        &peer_id.to_string(),
    );
    let expected = confirm_mac(
        &key,
        PairRole::Host,
        &local_id.to_string(),
        &peer_id.to_string(),
    );

    // Joiner confirms first (see module docs).
    write_frame(&mut tx, &PairMessage::Confirm { mac: mine }).await?;
    let their_mac = match read_frame::<PairMessage>(&mut rx).await? {
        PairMessage::Confirm { mac } => mac,
        PairMessage::Rejected { reason } => return Err(MeshError::PairRejected { reason }),
        _ => {
            return Err(MeshError::PairRejected {
                reason: "peer broke the pairing protocol (expected a confirm)".into(),
            })
        }
    };
    if !ct_eq(&their_mac, &expected) {
        let _ = write_frame(
            &mut tx,
            &PairMessage::Rejected {
                reason: "host failed key confirmation".into(),
            },
        )
        .await;
        flush_stream(&mut tx).await;
        return Err(MeshError::PairRejected {
            reason: "the host failed key confirmation; check the code and try again".into(),
        });
    }
    write_frame(
        &mut tx,
        &PairMessage::Introduce {
            node_name: local_name.to_string(),
            product_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    )
    .await?;
    let (node_name, product_version) = read_introduce(&mut rx).await?;
    flush_stream(&mut tx).await;
    Ok(PairOutcome {
        peer_id,
        node_name,
        product_version,
    })
}

async fn read_introduce(rx: &mut RecvStream) -> Result<(String, String), MeshError> {
    match read_frame::<PairMessage>(rx).await? {
        PairMessage::Introduce {
            node_name,
            product_version,
        } => Ok((node_name, product_version)),
        PairMessage::Rejected { reason } => Err(MeshError::PairRejected { reason }),
        _ => Err(MeshError::PairRejected {
            reason: "peer broke the pairing protocol (expected an introduction)".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_codes_are_six_digits() {
        for _ in 0..64 {
            let code = generate_code().unwrap();
            assert_eq!(code.len(), 6);
            assert!(code.bytes().all(|b| b.is_ascii_digit()));
        }
    }

    #[test]
    fn code_validation() {
        assert_eq!(validate_code(" 012345 ").unwrap(), "012345");
        assert!(validate_code("12345").is_err());
        assert!(validate_code("1234567").is_err());
        assert!(validate_code("12a456").is_err());
    }

    #[test]
    fn ct_eq_matches_plain_eq() {
        let a = [7u8; 32];
        let mut b = a;
        assert!(ct_eq(&a, &b));
        b[31] ^= 1;
        assert!(!ct_eq(&a, &b));
    }

    #[test]
    fn truncation_keeps_short_inputs() {
        assert_eq!(truncate_for_display("abc"), "abc");
        let long = "x".repeat(100);
        let shown = truncate_for_display(&long);
        assert!(shown.len() < long.len());
        assert!(shown.ends_with('…'));
    }
}
