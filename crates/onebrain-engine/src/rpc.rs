//! Distributed-inference plumbing: GGML RPC sessions over caller-owned
//! sockets (docs/distributed.md, ADR 0004).
//!
//! Nothing in this module binds a listener that outlives a call: the worker
//! side serves RPC over one end of a process-local socket pair whose other
//! end the daemon pumps into an authenticated mesh stream, and the head
//! side connects out to a loopback endpoint its own process accepted
//! exactly once per connection. The RPC protocol itself trusts its peer
//! (raw pointers, client aborts on torn streams) — which is exactly why it
//! only ever runs inside those authenticated tunnels.

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use crate::{init, sys, EngineError};

/// Platform stream type for the bridge end of a [`SocketPair`]: the end the
/// daemon (or a test pump) reads/writes while the other end is served by
/// [`RpcServeSession`].
#[cfg(unix)]
pub type BridgeStream = std::os::unix::net::UnixStream;
/// Platform stream type for the bridge end of a [`SocketPair`].
#[cfg(windows)]
pub type BridgeStream = TcpStream;

/// A connected pair of process-local sockets: one end is handed to
/// [`RpcServeSession::spawn`] (the GGML RPC server), the other is pumped
/// against a mesh stream.
///
/// Unix uses `socketpair(2)` via [`std::os::unix::net::UnixStream::pair`]
/// (GGML's transport uses `send`/`recv`, which work on unix-domain
/// sockets). Windows has no socketpair, so a loopback TCP pair is built:
/// bind `127.0.0.1:0`, self-connect, accept once, verify the accepted peer
/// is our own connecting socket (which closes the accept race — the
/// documented residual in the threat model), then drop the listener.
pub struct SocketPair {
    serve: BridgeStream,
    bridge: BridgeStream,
}

impl SocketPair {
    pub fn new() -> Result<SocketPair, EngineError> {
        Self::new_inner().map_err(|source| EngineError::SocketPair { source })
    }

    #[cfg(unix)]
    fn new_inner() -> io::Result<SocketPair> {
        let (serve, bridge) = BridgeStream::pair()?;
        Ok(SocketPair { serve, bridge })
    }

    #[cfg(windows)]
    fn new_inner() -> io::Result<SocketPair> {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
        let addr = listener.local_addr()?;
        let bridge = TcpStream::connect(addr)?;
        let (serve, peer) = listener.accept()?;
        // Kill the accept race: the connection we accepted must be the one
        // we just made, not some other local process racing the port.
        if peer != bridge.local_addr()? {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionRefused,
                "loopback socket pair accepted a foreign connection",
            ));
        }
        serve.set_nodelay(true)?;
        bridge.set_nodelay(true)?;
        drop(listener);
        Ok(SocketPair { serve, bridge })
    }

    /// Split into (raw serve-end handle, bridge stream). Ownership of the
    /// raw handle transfers to [`RpcServeSession::spawn`], which closes it.
    pub fn into_parts(self) -> (i64, BridgeStream) {
        (into_raw(self.serve), self.bridge)
    }
}

#[cfg(unix)]
fn into_raw(s: BridgeStream) -> i64 {
    use std::os::fd::IntoRawFd;
    i64::from(s.into_raw_fd())
}

#[cfg(windows)]
fn into_raw(s: BridgeStream) -> i64 {
    use std::os::windows::io::IntoRawSocket;
    s.into_raw_socket() as i64
}

/// Close a raw socket handle we still own (error paths before the C side
/// takes ownership).
fn close_raw(raw: i64) {
    #[cfg(unix)]
    unsafe {
        use std::os::fd::{FromRawFd, OwnedFd};
        drop(OwnedFd::from_raw_fd(raw as std::os::fd::RawFd));
    }
    #[cfg(windows)]
    unsafe {
        use std::os::windows::io::{FromRawSocket, OwnedSocket};
        drop(OwnedSocket::from_raw_socket(
            raw as std::os::windows::io::RawSocket,
        ));
    }
}

/// A dedicated OS thread serving exactly one GGML RPC session over a
/// caller-owned socket. The thread blocks in `ob_rpc_serve_fd` until the
/// peer closes (client model freed, or the bridge shuts the pair down) and
/// the socket is closed by the serve call itself.
pub struct RpcServeSession {
    handle: thread::JoinHandle<i32>,
}

impl RpcServeSession {
    /// Spawn the serve thread. `raw_fd` comes from
    /// [`SocketPair::into_parts`]; ownership transfers here on success and
    /// the handle is closed on every error path too, so the peer never
    /// hangs. `n_threads <= 0` picks the machine's available parallelism;
    /// `dev_index` indexes the local device enumeration
    /// ([`crate::devices`]) — CPU in M3.
    ///
    /// `cache_dir` (None = no cache) points the session at the local RPC
    /// tensor cache (docs/logistics.md "RPC tensor-cache pre-seeding"): the
    /// serve loop answers `SET_TENSOR_HASH` from files named per
    /// [`crate::rpc_cache::rpc_cache_filename`], so a pre-seeded worker
    /// skips the head's push for every tensor over
    /// [`crate::rpc_cache::RPC_HASH_THRESHOLD`]. The directory is created
    /// here if missing; eviction is the daemon reaper's job, never this
    /// session's.
    pub fn spawn(
        raw_fd: i64,
        n_threads: i32,
        dev_index: i32,
        cache_dir: Option<&std::path::Path>,
    ) -> Result<RpcServeSession, EngineError> {
        init();
        let dev_count = unsafe { sys::ob_dev_count() };
        if dev_index < 0 || dev_index >= dev_count {
            close_raw(raw_fd);
            return Err(EngineError::RpcDeviceIndex {
                index: dev_index,
                count: dev_count,
            });
        }
        // Resolve the cache dir to a CString up front (the serve thread
        // outlives this call) and make sure it exists — the C side only
        // opens files inside it, it never creates the directory.
        let cache_dir_c = match cache_dir {
            None => None,
            Some(dir) => {
                let dir_str = dir.to_string_lossy().into_owned();
                let cdir = match std::ffi::CString::new(dir_str.clone()) {
                    Ok(c) => c,
                    Err(_) => {
                        close_raw(raw_fd);
                        return Err(EngineError::BadCacheDir(dir_str));
                    }
                };
                if let Err(source) = std::fs::create_dir_all(dir) {
                    close_raw(raw_fd);
                    return Err(EngineError::RpcCacheDir {
                        path: dir_str,
                        source,
                    });
                }
                Some(cdir)
            }
        };
        let n_threads = if n_threads > 0 {
            n_threads
        } else {
            thread::available_parallelism().map_or(1, |n| n.get() as i32)
        };
        let handle = thread::Builder::new()
            .name("ob-rpc-serve".into())
            .spawn(move || {
                let with_cache = cache_dir_c.is_some();
                tracing::debug!(
                    raw_fd,
                    n_threads,
                    dev_index,
                    with_cache,
                    "rpc serve session starting"
                );
                let cache_ptr = cache_dir_c
                    .as_ref()
                    .map_or(std::ptr::null(), |c| c.as_ptr());
                let status =
                    unsafe { sys::ob_rpc_serve_fd(raw_fd, cache_ptr, n_threads, dev_index) };
                tracing::debug!(raw_fd, status, "rpc serve session ended");
                status
            })
            .map_err(|source| EngineError::SocketPair { source })?;
        Ok(RpcServeSession { handle })
    }

    /// Whether the session has ended (peer closed its side).
    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    /// Block until the session ends; returns the shim status (0 = served).
    pub fn join(self) -> i32 {
        self.handle.join().unwrap_or(-1)
    }

    /// Wait up to `timeout` for the session to end. `Ok(status)` when it
    /// did; `Err(self)` when it is still serving (the caller keeps it).
    pub fn join_timeout(self, timeout: Duration) -> Result<i32, RpcServeSession> {
        let deadline = Instant::now() + timeout;
        while !self.handle.is_finished() {
            if Instant::now() >= deadline {
                return Err(self);
            }
            thread::sleep(Duration::from_millis(10));
        }
        Ok(self.join())
    }
}

/// Worker-side session unit (the contract's `RpcSession`): owns the bridge
/// socket and the serve thread. The daemon pumps [`RpcSession::bridge`]
/// against the accepted mesh `rpc` stream; dropping the session closes the
/// bridge end, which ends the serve thread (epoch teardown closes streams
/// first, then frees the model).
pub struct RpcSession {
    serve: Option<RpcServeSession>,
    bridge: Option<BridgeStream>,
}

impl RpcSession {
    /// Create the socket pair and start serving one RPC session on a
    /// dedicated thread (CPU device by default in M3 — pass its index).
    /// No tensor cache; see [`RpcSession::start_with_cache`].
    pub fn start(n_threads: i32, dev_index: i32) -> Result<RpcSession, EngineError> {
        Self::start_with_cache(n_threads, dev_index, None)
    }

    /// [`RpcSession::start`] with an optional RPC tensor-cache directory
    /// (`<data_dir>/rpc-cache/` in the daemon): the session then answers
    /// `SET_TENSOR_HASH` from pre-seeded files (see [`crate::rpc_cache`])
    /// and the head's weight push skips every cached tensor. The directory
    /// is created if missing.
    pub fn start_with_cache(
        n_threads: i32,
        dev_index: i32,
        cache_dir: Option<&std::path::Path>,
    ) -> Result<RpcSession, EngineError> {
        let (raw, bridge) = SocketPair::new()?.into_parts();
        let serve = RpcServeSession::spawn(raw, n_threads, dev_index, cache_dir)?;
        Ok(RpcSession {
            serve: Some(serve),
            bridge: Some(bridge),
        })
    }

    /// The bridge end to pump against the mesh stream.
    pub fn bridge(&self) -> Option<&BridgeStream> {
        self.bridge.as_ref()
    }

    /// Take ownership of the bridge end (e.g. to hand to [`pump`]). The
    /// session keeps the serve thread; the caller keeps the socket alive
    /// for as long as the session should run.
    pub fn take_bridge(&mut self) -> Option<BridgeStream> {
        self.bridge.take()
    }

    /// Whether the serve thread has ended.
    pub fn is_finished(&self) -> bool {
        self.serve
            .as_ref()
            .map_or(true, RpcServeSession::is_finished)
    }

    /// Close the bridge end (if still held) and wait up to `timeout` for
    /// the serve thread to end. Returns false on timeout.
    pub fn shutdown(mut self, timeout: Duration) -> bool {
        if let Some(bridge) = self.bridge.take() {
            let _ = bridge.shutdown(Shutdown::Both);
            drop(bridge);
        }
        match self.serve.take() {
            Some(serve) => serve.join_timeout(timeout).is_ok(),
            None => true,
        }
    }
}

/// Registration of a remote GGML RPC server (head side). The endpoint is a
/// loopback bridge the daemon owns — never a raw remote address.
#[derive(Debug)]
pub struct RemoteServer {
    slot: i32,
    endpoint: String,
    device_count: i32,
}

/// `ob_rpc_add_server` mutates a static slot table; serialize it.
static REGISTER_LOCK: Mutex<()> = Mutex::new(());

impl RemoteServer {
    /// Register `endpoint` ("host:port") and enumerate its devices. Connect
    /// failure and protocol/version mismatch are indistinguishable at this
    /// layer; the mesh engine-build-hash handshake pre-empts the latter
    /// between paired OneBrain nodes.
    ///
    /// Registrations are cached per endpoint string for the process
    /// lifetime (upstream behavior): re-registering an endpoint returns the
    /// same slot, so new epochs should bridge through fresh ephemeral
    /// ports.
    pub fn register(endpoint: &str) -> Result<RemoteServer, EngineError> {
        init();
        let cendpoint = std::ffi::CString::new(endpoint)
            .map_err(|_| EngineError::BadEndpoint(endpoint.into()))?;
        let guard = REGISTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let slot = unsafe { sys::ob_rpc_add_server(cendpoint.as_ptr()) };
        drop(guard);
        match slot {
            -1 => Err(EngineError::RpcConnect {
                endpoint: endpoint.into(),
            }),
            -2 => Err(EngineError::RpcServerLimit { max: 16 }),
            slot => {
                let device_count = unsafe { sys::ob_rpc_server_device_count(slot) };
                tracing::debug!(endpoint, slot, device_count, "registered rpc server");
                Ok(RemoteServer {
                    slot,
                    endpoint: endpoint.into(),
                    device_count,
                })
            }
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Devices this server exposes (fetched once at registration — reading
    /// it is never a network round trip).
    pub fn device_count(&self) -> i32 {
        self.device_count
    }

    pub(crate) fn slot(&self) -> i32 {
        self.slot
    }
}

/// A duplex byte stream that can be split (cloned) and half-closed — what
/// the byte pump needs. Implemented for [`TcpStream`] and (on Unix)
/// [`std::os::unix::net::UnixStream`].
pub trait Duplex: Read + Write + Send + Sized + 'static {
    fn try_clone_duplex(&self) -> io::Result<Self>;
    fn shutdown_duplex(&self, how: Shutdown) -> io::Result<()>;
}

impl Duplex for TcpStream {
    fn try_clone_duplex(&self) -> io::Result<Self> {
        self.try_clone()
    }
    fn shutdown_duplex(&self, how: Shutdown) -> io::Result<()> {
        self.shutdown(how)
    }
}

#[cfg(unix)]
impl Duplex for std::os::unix::net::UnixStream {
    fn try_clone_duplex(&self) -> io::Result<Self> {
        self.try_clone()
    }
    fn shutdown_duplex(&self, how: Shutdown) -> io::Result<()> {
        self.shutdown(how)
    }
}

/// The two relay threads of a [`pump`]; join to observe both directions
/// drained.
pub struct Pump {
    a_to_b: thread::JoinHandle<()>,
    b_to_a: thread::JoinHandle<()>,
}

impl Pump {
    pub fn join(self) {
        let _ = self.a_to_b.join();
        let _ = self.b_to_a.join();
    }

    pub fn is_finished(&self) -> bool {
        self.a_to_b.is_finished() && self.b_to_a.is_finished()
    }
}

/// Bidirectional byte pump between two duplex streams — the in-process
/// miniature of what the daemon does between a mesh `rpc` stream and a
/// [`SocketPair`] bridge end. Each direction runs on its own thread; EOF
/// (or error) on one side half-closes the other, so both threads terminate
/// once either peer closes.
pub fn pump<A: Duplex, B: Duplex>(a: A, b: B) -> Result<Pump, EngineError> {
    fn relay<R: Duplex, W: Duplex>(mut from: R, to: W) -> impl FnOnce() + Send + 'static {
        move || {
            let mut to_w = to;
            if let Err(e) = io::copy(&mut from, &mut to_w) {
                tracing::debug!(error = %e, "rpc bridge pump direction ended with error");
            }
            let _ = to_w.shutdown_duplex(Shutdown::Write);
        }
    }
    let (a_r, b_w) = (
        a.try_clone_duplex()
            .map_err(|source| EngineError::SocketPair { source })?,
        b.try_clone_duplex()
            .map_err(|source| EngineError::SocketPair { source })?,
    );
    let a_to_b = thread::Builder::new()
        .name("ob-rpc-pump".into())
        .spawn(relay(a_r, b_w))
        .map_err(|source| EngineError::SocketPair { source })?;
    let b_to_a = thread::Builder::new()
        .name("ob-rpc-pump".into())
        .spawn(relay(b, a))
        .map_err(|source| EngineError::SocketPair { source })?;
    Ok(Pump { a_to_b, b_to_a })
}
