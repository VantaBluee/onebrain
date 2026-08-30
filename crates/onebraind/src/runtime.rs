//! The daemon runtime: wires paths, config, lock, token, the engine host,
//! and both routers into one serving loop (docs/internal-api.md "Daemon
//! lifecycle"). `onebrain __daemon` calls [`run_blocking`] and nothing else.

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use onebrain_api::auth::AuthConfig;
use onebrain_api::ApiState;
use onebrain_mesh::{identity, MeshConfig, MeshHandle, MeshService};
use tokio::sync::Notify;

use crate::cluster::{self, ClusterState};
use crate::config::Config;
use crate::engine_host::{DaemonBackend, EngineHost, HostMsg, HostPerf};
use crate::lock::{self, DaemonLock, RunInfo};
use crate::paths::AppPaths;
use crate::server::{internal_router, InternalState};
use crate::supervisor::{self, SupervisorMsg, SupervisorTx};
use crate::{token, DaemonError};

/// Grace given to in-flight serve traffic after the polite `Draining`
/// notice on `onebrain stop` (docs/resilience.md "Worker-side drain").
const DRAIN_GRACE: Duration = Duration::from_secs(3);

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

    // Test-only `[debug] decode_delay_ms` (docs/resilience.md): a
    // per-piece sleep that gives the chaos sim a deterministic kill window;
    // None (every real deployment) adds no delay anywhere.
    let decode_delay = config.debug.decode_delay_ms.map(Duration::from_millis);
    if decode_delay.is_some() {
        tracing::warn!(
            delay_ms = config.debug.decode_delay_ms,
            "[debug] decode_delay_ms is set; token streaming is artificially slowed (test-only)"
        );
    }
    // [perf] levers the engine host consumes (docs/perf.md §3/§4/§6):
    // session shape (concurrency, micro-batch), the prefill-overlap
    // switch, and cross-request KV reuse. n_batch stays at the pre-M7 512
    // (not a config knob).
    let host_perf = HostPerf {
        max_concurrent: config.perf.max_concurrent_requests.max(1),
        n_ubatch: config.perf.n_ubatch,
        prefill_overlap: config.perf.prefill_overlap,
        kv_reuse: config.perf.kv_reuse,
        ..HostPerf::default()
    };
    let (host, host_thread) = EngineHost::spawn(decode_delay, host_perf);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|source| DaemonError::Runtime { source })?;

    // Persisted device profile (M4, docs/scheduler-v1.md): loaded at
    // startup when present, refreshed by `POST /api/internal/bench`. Shared
    // between the NodeStatus provider (peers see the profile fields) and
    // the planner (the head scores itself with them). A corrupt file warns
    // and is treated as absent — startup must not fail over a stale bench.
    let profile_path = paths.config_dir.join("profile.toml");
    let profile: crate::server::SharedProfile =
        Arc::new(std::sync::Mutex::new(if profile_path.exists() {
            match onebrain_scheduler::load_profile(&profile_path) {
                Ok(stored) => {
                    tracing::info!(
                        measured_unix = stored.measured_unix,
                        decode_tps = stored.decode_tps,
                        "loaded persisted device profile"
                    );
                    Some(stored)
                }
                Err(err) => {
                    tracing::warn!(error = %err, "ignoring unreadable profile.toml");
                    None
                }
            }
        } else {
            None
        }));

    // NodeStatus provider: the schedulable memory this node reports to
    // peers and budgets for itself — the CPU device's free memory minus the
    // OS reserve, or the test-only `[debug]` override (docs/distributed.md)
    // — plus the profile fields (the `[debug] decode_tps_override` wins
    // over the measured decode rate, docs/scheduler-v1.md).
    let debug_override = config.debug.usable_memory_override_bytes;
    let decode_override = config.debug.decode_tps_override;
    let status_profile = profile.clone();
    // Battery policy inputs (docs/resilience.md "Power realities"): one
    // probe shared by the NodeStatus provider and the internal API.
    let battery_probe: Arc<dyn crate::power::BatteryProbe + Send + Sync> =
        Arc::from(crate::power::platform_battery_probe());
    let battery_threshold = config.battery_drain_threshold;
    let status_battery_probe = battery_probe.clone();
    let bind_addrs = match &config.mesh.bind_addr {
        Some(addr) => vec![addr.parse().map_err(|e| DaemonError::ConfigParse {
            path: paths.config_file().display().to_string(),
            source: Box::new(serde::de::Error::custom(format!(
                "[mesh] bind_addr {addr:?} is not a socket address: {e}"
            ))),
        })?],
        None => Vec::new(),
    };
    // Cluster-session state: epoch counter, active plan, plan acks. Created
    // before the mesh config because the bench source below consults it
    // (worker-shard check); the cluster task consumes it further down.
    let cluster = ClusterState::new();
    let mesh_config = MeshConfig {
        enable_mdns: config.mesh.enable_mdns,
        enable_relays: config.mesh.enable_relays,
        bind_addrs,
        engine_build: onebrain_engine::engine_build_hash().0,
        // M7 cluster bench (docs/perf.md §10): answer peers' on-demand
        // `BenchRequest`s with the same microbench `POST /api/internal/bench`
        // runs; busy/shard-serving/uncached-model nodes decline with the
        // wire's cannot-bench-now marker.
        bench_source: Some(Arc::new(crate::server::DaemonBenchSource {
            host: host.clone(),
            cluster: cluster.clone(),
            cache_root: cache_root.clone(),
            profile: profile.clone(),
            profile_path: profile_path.clone(),
        })),
        node_status: Some(Arc::new(move || {
            let (usable_memory_bytes, devices) = cluster::local_node_status(debug_override);
            let stored = *status_profile.lock().expect("profile state poisoned");
            onebrain_mesh::NodeStatusReport {
                usable_memory_bytes,
                devices,
                prefill_tps: stored.map(|p| p.prefill_tps),
                decode_tps: decode_override.or(stored.map(|p| p.decode_tps)),
                disk_mbps: stored.map(|p| p.disk_mbps),
                // Battery policy (docs/resilience.md): probed per status
                // send, so a threshold crossing propagates on the next
                // session establishment or bench push.
                draining: crate::power::battery_status(
                    status_battery_probe.as_ref(),
                    battery_threshold,
                )
                .draining,
            }
        })),
        // M6 logistics (docs/logistics.md): a persistent blob store so
        // shared ranges survive restarts, and the local range inventory so
        // this daemon ANSWERS peers' RangeQuery — the same source shape it
        // queries them with.
        blobs_dir: Some(paths.data_dir.join("blobs")),
        range_source: Some(Arc::new(crate::logistics::LocalRangeInventory::new(
            cache_root.clone(),
        ))),
        ..MeshConfig::default()
    };
    let served = match runtime.block_on(MeshService::spawn(
        device_key,
        paths.config_dir.join("peers.toml"),
        node_name,
        mesh_config,
    )) {
        Ok(mesh) => {
            // Supervisor queue (M5): the gateway backend and the cluster
            // task send into it; the supervisor task itself is spawned by
            // `serve` once the internal state exists (messages queue in the
            // channel until then).
            let (sup_tx, sup_rx) = supervisor::channel();
            // Cluster task: consumes plan traffic, rpc streams, and — M5 —
            // peer events from the mesh (worker path + death/drain/rejoin
            // detection) and records acks (head path).
            let cluster_task = runtime.block_on(async {
                let ctrl_rx = mesh.incoming_control().await?;
                let rpc_rx = mesh.incoming_rpc().await?;
                let events_rx = mesh.peer_events().await?;
                Ok::<_, onebrain_mesh::MeshError>(cluster::spawn_cluster_task(
                    mesh.clone(),
                    host.clone(),
                    cluster.clone(),
                    ctrl_rx,
                    rpc_rx,
                    events_rx,
                    sup_tx.clone(),
                    // M6 worker logistics: range fetch on plan adoption +
                    // rpc-cache pre-seed and reaper (docs/logistics.md).
                    cluster::WorkerLogistics {
                        cache_root: cache_root.clone(),
                        rpc_cache_dir: paths.data_dir.join("rpc-cache"),
                        rpc_cache_max_bytes: config.rpc_cache_max_bytes,
                        cache_max_bytes: config.cache_max_bytes,
                    },
                ))
            });
            let cluster_task = match cluster_task {
                Ok(task) => Some(task),
                Err(err) => {
                    tracing::error!(error = %err, "cluster task unavailable; distributed plans are disabled this run");
                    None
                }
            };
            let served = runtime.block_on(serve(
                &config,
                token,
                host.clone(),
                &cache_root,
                &run_dir,
                mesh.clone(),
                cluster.clone(),
                debug_override,
                profile.clone(),
                profile_path.clone(),
                sup_tx,
                sup_rx,
            ));
            // M5 worker-side drain (docs/resilience.md): with an active
            // shard epoch, tell the head we are draining politely and give
            // in-flight serve traffic a short grace window — the mesh and
            // the serve bridges are still up here.
            if let Some((epoch, head)) = cluster.worker_shard() {
                runtime.block_on(async {
                    let notice = onebrain_proto::message::Envelope::new(
                        onebrain_proto::message::Message::Draining {
                            epoch,
                            reason: "onebrain stop".to_string(),
                        },
                    );
                    match mesh.send_control(&head.0, notice).await {
                        Ok(()) => {
                            tracing::info!(
                                epoch = epoch.0,
                                "sent polite drain notice to the head; granting {DRAIN_GRACE:?} \
                                 for in-flight serve traffic"
                            );
                            tokio::time::sleep(DRAIN_GRACE).await;
                        }
                        Err(err) => tracing::warn!(
                            epoch = epoch.0,
                            error = %err,
                            "could not send the drain notice (head unreachable); \
                             continuing teardown"
                        ),
                    }
                });
            }
            // Teardown ordering (ADR 0004): free any loaded model FIRST —
            // a distributed head model sends remote frees over its rpc
            // bridges, and GGML aborts on a torn stream — then close the
            // mesh (which ends bridges and worker serve sessions).
            let (unload_tx, unload_rx) = tokio::sync::oneshot::channel();
            if host.send(HostMsg::Unload { resp: unload_tx }).is_ok() {
                let _ = runtime.block_on(async {
                    tokio::time::timeout(Duration::from_secs(15), unload_rx).await
                });
            }
            cluster.teardown_head_bridges();
            // Close mesh connections and the endpoint before tearing down
            // the rest; peers see a clean close instead of a timeout.
            let _ = runtime.block_on(mesh.shutdown());
            // The mesh close drops the cluster task's channels; give it a
            // SHORT bounded window to join its worker serve threads. This
            // window sits between "API endpoint gone" and "lock released",
            // and `onebrain stop` callers immediately re-`up` — every
            // second here is user-visible restart latency (a lingering
            // reconnect-loop sleep must never hold the lock for 10s).
            if let Some(task) = cluster_task {
                let _ = runtime
                    .block_on(async { tokio::time::timeout(Duration::from_secs(2), task).await });
            }
            served
        }
        Err(source) => Err(DaemonError::Mesh { source }),
    };

    // Teardown in dependency order: stop the engine host (drops Session and
    // Model), stop the runtime (ends any leftover bridge pumps, closing
    // their sockets), only then free the process-wide engine backend.
    let _ = host.send(HostMsg::Shutdown);
    let _ = host_thread.join();
    runtime.shutdown_timeout(Duration::from_secs(5));
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

#[allow(clippy::too_many_arguments)]
async fn serve(
    config: &Config,
    token: String,
    host: EngineHost,
    cache_root: &Path,
    run_dir: &Path,
    mesh: MeshHandle,
    cluster: Arc<ClusterState>,
    usable_memory_override: Option<u64>,
    profile: crate::server::SharedProfile,
    profile_path: std::path::PathBuf,
    supervisor_tx: SupervisorTx,
    supervisor_rx: tokio::sync::mpsc::UnboundedReceiver<SupervisorMsg>,
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
        backend: Arc::new(DaemonBackend::new(
            host.clone(),
            cache_root.to_path_buf(),
            supervisor_tx,
            mesh.clone(),
            config.cache_max_bytes,
            // Admission control (docs/perf.md §6): at most
            // max_concurrent_requests running + queue_depth waiting.
            config.perf.max_concurrent_requests,
            config.perf.queue_depth,
        )),
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
        n_ubatch: config.perf.n_ubatch,
        port: local_addr.port(),
        started: Instant::now(),
        product_version,
        shutdown: shutdown.clone(),
        mesh,
        cluster,
        usable_memory_override,
        decode_tps_override: config.debug.decode_tps_override,
        profile,
        profile_path,
        // The platform probe is stateless (reads on every call); a second
        // instance here keeps `serve` self-contained.
        battery_probe: Arc::from(crate::power::platform_battery_probe()),
        battery_threshold: config.battery_drain_threshold,
        cache_max_bytes: config.cache_max_bytes,
        // [perf] draft_model (docs/perf.md §5): the default speculative
        // draft for loads that ask for `speculative` without naming one.
        draft_model: config.perf.draft_model.clone(),
        retry: tokio::sync::Mutex::new(crate::supervisor::RetryLedger::default()),
    });
    // M5 supervisor: owns every generation job's lifecycle (transparent
    // retry) plus the death-teardown and lazy rejoin re-plan follow-ups
    // from the cluster task. Ends when its senders drop at teardown; the
    // runtime shutdown reaps it regardless.
    let _supervisor_task = supervisor::spawn(internal_state.clone(), supervisor_rx);
    // M5 sleep inhibitor (docs/resilience.md): held while this node has
    // work sleep would break — a loaded model with a request in flight or
    // a distributed epoch, or (worker role) an adopted shard other nodes
    // depend on. hold/release are idempotent, so edge-free polling is fine.
    {
        let watch = internal_state.clone();
        tokio::spawn(async move {
            let mut inhibitor = crate::power::platform_sleep_inhibitor();
            let mut tick = tokio::time::interval(Duration::from_secs(2));
            loop {
                tick.tick().await;
                let epoch_active = watch.cluster.active().is_some();
                let serving_shard = watch.cluster.worker_shard().is_some();
                let model_loaded = watch.cluster.loaded_source().is_some() || serving_shard;
                let in_flight = !watch.host.is_idle();
                if crate::power::should_hold_sleep(
                    model_loaded,
                    in_flight || serving_shard,
                    epoch_active,
                ) {
                    inhibitor.hold("model active");
                } else {
                    inhibitor.release();
                }
            }
        });
    }
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
