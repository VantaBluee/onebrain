//! The daemon runtime: wires paths, config, lock, token, the engine host,
//! and both routers into one serving loop (docs/internal-api.md "Daemon
//! lifecycle"). `onebrain __daemon` calls [`run_blocking`] and nothing else.

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use onebrain_api::auth::AuthConfig;
use onebrain_api::ApiState;
use onebrain_mesh::{identity, MeshConfig, MeshHandle, MeshService};
use tokio::sync::Notify;

use crate::config::Config;
use crate::engine_host::{DaemonBackend, EngineHost, HostMsg};
use crate::lock::{self, DaemonLock, RunInfo};
use crate::paths::AppPaths;
use crate::server::{internal_router, InternalState};
use crate::{token, DaemonError};

/// Run the daemon until shutdown (internal endpoint or Ctrl-C/SIGTERM-
/// equivalent console signal). Blocks the calling thread for the daemon's
/// whole life; returns only on clean exit or startup failure.
pub fn run_blocking() -> Result<(), DaemonError> {
    // Logging first so every later step can explain itself. RUST_LOG wins;
    // `info` is the quiet-but-useful default. try_init: embedding tests may
    // have installed a subscriber already.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    let paths = AppPaths::resolve()?;
    let mut config = Config::load(&paths.config_file())?;
    let run_dir = paths.data_dir.join("run");
    // The lock is the single-instance authority; hold it before touching
    // any shared state. Exits with the already-running remedy if held.
    let lock = DaemonLock::acquire(&run_dir)?;
    let token = token::load_or_create(&paths)?;
    let cache_root = paths.model_cache_dir();
    std::fs::create_dir_all(&cache_root).map_err(|source| DaemonError::DataDir {
        path: cache_root.display().to_string(),
        source,
    })?;

    // Device identity for the mesh: created at first start, never
    // regenerated (docs/mesh.md "Identity"). A malformed key file is a
    // startup error, not a silent re-key.
    let device_key = identity::load_or_create(&paths.config_dir)
        .map_err(|source| DaemonError::Mesh { source })?;
    // Node name shown to peers: config wins; first run derives it from the
    // hostname and persists it so renaming the machine later does not
    // re-identify this node to its peers.
    let node_name = match config.node_name.clone() {
        Some(name) => name,
        None => {
            let name = default_node_name();
            config.node_name = Some(name.clone());
            if let Err(e) = config.save(&paths.config_file()) {
                tracing::warn!(error = %e, "could not persist node_name; using it for this run only");
            }
            name
        }
    };

    let (host, host_thread) = EngineHost::spawn();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|source| DaemonError::Runtime { source })?;

    let mesh_config = MeshConfig {
        enable_mdns: config.mesh.enable_mdns,
        enable_relays: config.mesh.enable_relays,
        engine_build: onebrain_engine::engine_build_hash().0,
        ..MeshConfig::default()
    };
    let served = match runtime.block_on(MeshService::spawn(
        device_key,
        paths.config_dir.join("peers.toml"),
        node_name,
        mesh_config,
    )) {
        Ok(mesh) => {
            let served = runtime.block_on(serve(
                &config,
                token,
                host.clone(),
                &cache_root,
                &run_dir,
                mesh.clone(),
            ));
            // Close mesh connections and the endpoint before tearing down
            // the rest; peers see a clean close instead of a timeout.
            let _ = runtime.block_on(mesh.shutdown());
            served
        }
        Err(source) => Err(DaemonError::Mesh { source }),
    };

    // Teardown in dependency order: stop the engine host (drops Session and
    // Model), only then free the process-wide engine backend.
    let _ = host.send(HostMsg::Shutdown);
    let _ = host_thread.join();
    onebrain_engine::shutdown();
    lock::remove_run_info(&run_dir);
    drop(lock);
    tracing::info!("daemon exited cleanly");
    served
}

/// Hostname fallback for `node_name` when the config does not set one.
fn default_node_name() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "onebrain-node".to_string())
}

async fn serve(
    config: &Config,
    token: String,
    host: EngineHost,
    cache_root: &Path,
    run_dir: &Path,
    mesh: MeshHandle,
) -> Result<(), DaemonError> {
    let product_version: &'static str = env!("CARGO_PKG_VERSION");
    let shutdown = Arc::new(Notify::new());

    let listener = tokio::net::TcpListener::bind(&config.api_bind)
        .await
        .map_err(|source| DaemonError::Bind {
            addr: config.api_bind.clone(),
            source,
        })?;
    let local_addr = listener.local_addr().map_err(|source| DaemonError::Bind {
        addr: config.api_bind.clone(),
        source,
    })?;

    let api_state = ApiState {
        backend: Arc::new(DaemonBackend::new(host.clone(), cache_root.to_path_buf())),
        auth: Arc::new(AuthConfig {
            token: token.clone(),
            localhost_exempt: config.localhost_auth_exempt,
        }),
        product_version,
    };
    let internal_state = Arc::new(InternalState {
        host,
        // Internal endpoints are ALWAYS token-auth'd (contract) — the
        // public localhost exemption is not consulted here.
        auth: AuthConfig {
            token,
            localhost_exempt: false,
        },
        cache_root: cache_root.to_path_buf(),
        ctx_len: config.ctx_len,
        port: local_addr.port(),
        started: Instant::now(),
        product_version,
        shutdown: shutdown.clone(),
        mesh,
    });
    let app = onebrain_api::router(api_state).merge(internal_router(internal_state));

    // daemon.json only after the listener is bound (contract): its port is
    // real, and a kill -9 leftover gets overwritten right here on restart.
    lock::write_run_info(
        run_dir,
        &RunInfo {
            pid: std::process::id(),
            port: local_addr.port(),
            started_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            version: product_version.to_string(),
        },
    )?;
    tracing::info!(addr = %local_addr, "daemon listening");

    let graceful = async move {
        let ctrl_c = async {
            if let Err(e) = tokio::signal::ctrl_c().await {
                tracing::warn!(error = %e, "console-signal handler unavailable; only /api/internal/shutdown will stop this daemon");
                std::future::pending::<()>().await;
            }
        };
        tokio::select! {
            _ = shutdown.notified() => {}
            _ = ctrl_c => tracing::info!("console signal received; shutting down"),
        }
    };
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(graceful)
    .await
    .map_err(|source| DaemonError::Serve { source })
}
