//! `onebrain up`: make sure the daemon is running and print how to reach
//! it. Healthy daemon → just report it. Otherwise spawn `onebrain __daemon`
//! detached (logs to `<data_dir>/logs/daemon.log`) and poll the internal
//! status endpoint until healthy (10 s budget).

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use onebraind::paths::AppPaths;

use super::CliError;
use crate::client::DaemonClient;

/// A healthy daemon, plus whether this call had to start it.
pub struct UpOutcome {
    pub client: DaemonClient,
    pub started: bool,
}

/// Idempotent "make it run": used by `up` and by `run`.
pub fn ensure_up(paths: &AppPaths) -> Result<UpOutcome, CliError> {
    if let Ok(client) = DaemonClient::from_paths(paths) {
        if client.status().is_ok() {
            return Ok(UpOutcome {
                client,
                started: false,
            });
        }
    }
    let (child, log_path) = spawn_daemon(paths)?;
    poll_until_healthy(paths, child, &log_path)
}

pub fn run(json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;
    let outcome = ensure_up(&paths)?;
    let state = outcome.client.state();

    if json {
        println!(
            "{}",
            serde_json::json!({
                "status": "running",
                "started": outcome.started,
                "pid": state.pid,
                "version": state.version,
                "endpoint": outcome.client.base_url(),
                "openai_base_url": format!("{}/v1", outcome.client.base_url()),
                "token": outcome.client.token(),
            })
        );
    } else {
        if outcome.started {
            println!("daemon started (pid {}, v{})", state.pid, state.version);
        } else {
            println!(
                "daemon already running (pid {}, v{})",
                state.pid, state.version
            );
        }
        println!("endpoint  {}", outcome.client.base_url());
        println!("token     {}", outcome.client.token());
        println!();
        println!("hint: `onebrain run <model>` loads a model; `onebrain status` shows this again.");
    }
    Ok(())
}

/// Spawn `onebrain __daemon` fully detached from this terminal, with both
/// output streams appended to the daemon log.
fn spawn_daemon(paths: &AppPaths) -> Result<(Child, PathBuf), CliError> {
    let exe = std::env::current_exe().map_err(|e| {
        CliError(format!(
            "could not locate the onebrain executable ({e}); reinstall onebrain"
        ))
    })?;

    let log_dir = paths.data_dir.join("logs");
    std::fs::create_dir_all(&log_dir).map_err(|e| {
        CliError(format!(
            "could not create the log directory {} ({e}); check permissions on {}",
            log_dir.display(),
            paths.data_dir.display()
        ))
    })?;
    let log_path = log_dir.join("daemon.log");
    let open_log = || {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| {
                CliError(format!(
                    "could not open the daemon log {} ({e}); check permissions",
                    log_path.display()
                ))
            })
    };
    let out = open_log()?;
    let err = open_log()?;

    let mut cmd = Command::new(exe);
    cmd.arg("__daemon")
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err));

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: no console, survives
        // this shell, unaffected by its Ctrl+C.
        cmd.creation_flags(0x0000_0008 | 0x0000_0200);
        unshare_std_handles();
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // New process group: the daemon outlives this shell's signals.
        cmd.process_group(0);
    }

    let child = cmd.spawn().map_err(|e| {
        CliError(format!(
            "failed to start the daemon process ({e}); check that onebrain is installed correctly"
        ))
    })?;
    Ok((child, log_path))
}

/// CreateProcess with `bInheritHandles = TRUE` (which any stdio redirect
/// forces) copies EVERY inheritable handle into the child — including this
/// process's own stdout/stderr, which are inheritable pipe ends whenever a
/// caller captured `onebrain up`'s output. The daemon outlives us, so a
/// leaked write-end would hold the caller's pipe open forever (their read
/// never sees EOF). Strip the inherit flag from our std handles before
/// spawning; the daemon's own stdio is wired to the log file explicitly.
#[cfg(windows)]
fn unshare_std_handles() {
    const STD_INPUT_HANDLE: u32 = -10i32 as u32;
    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    const HANDLE_FLAG_INHERIT: u32 = 1;
    // Stable-forever kernel32 ABI; declared directly to avoid a dependency.
    extern "system" {
        fn GetStdHandle(nstdhandle: u32) -> isize;
        fn SetHandleInformation(handle: isize, mask: u32, flags: u32) -> i32;
    }
    for which in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
        unsafe {
            let handle = GetStdHandle(which);
            if handle != 0 && handle != -1 {
                SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
            }
        }
    }
}

fn poll_until_healthy(
    paths: &AppPaths,
    mut child: Child,
    log_path: &std::path::Path,
) -> Result<UpOutcome, CliError> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        // Re-read daemon.json every attempt: the fresh daemon overwrites
        // any stale state file once its listener binds.
        if let Ok(client) = DaemonClient::from_paths(paths) {
            if client.status().is_ok() {
                return Ok(UpOutcome {
                    client,
                    started: true,
                });
            }
        }
        if let Ok(Some(code)) = child.try_wait() {
            return Err(CliError(format!(
                "the daemon exited during startup ({code}); see the log at {}",
                log_path.display()
            )));
        }
        if Instant::now() >= deadline {
            return Err(CliError(format!(
                "the daemon did not become healthy within 10s; see the log at {}",
                log_path.display()
            )));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}
