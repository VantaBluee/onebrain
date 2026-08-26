//! `onebrain doctor` v1: every check is a finding — (status, message,
//! remedy) — covering identity/paths (kept from v0), compute devices from
//! the engine, daemon state, and config-file validity. Grows firewall and
//! self-update checks in M8.

use serde::Serialize;

use onebrain_engine::DeviceKind;
use onebraind::config::Config;
use onebraind::paths::AppPaths;

use super::{human_bytes, CliError};
use crate::client::{ClientError, DaemonClient};

#[derive(Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Status {
    Ok,
    Warn,
    Fail,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Ok => "ok  ",
            Status::Warn => "warn",
            Status::Fail => "fail",
        }
    }
}

#[derive(Serialize)]
struct Finding {
    status: Status,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    remedy: Option<String>,
}

impl Finding {
    fn ok(message: impl Into<String>) -> Finding {
        Finding {
            status: Status::Ok,
            message: message.into(),
            remedy: None,
        }
    }
    fn warn(message: impl Into<String>, remedy: impl Into<String>) -> Finding {
        Finding {
            status: Status::Warn,
            message: message.into(),
            remedy: Some(remedy.into()),
        }
    }
    fn fail(message: impl Into<String>, remedy: impl Into<String>) -> Finding {
        Finding {
            status: Status::Fail,
            message: message.into(),
            remedy: Some(remedy.into()),
        }
    }
}

pub fn run(json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;
    let mut findings = Vec::new();

    // Identity + paths (the v0 report, as findings).
    findings.push(Finding::ok(format!(
        "product: {} {}",
        onebrain_proto::PRODUCT_NAME,
        env!("CARGO_PKG_VERSION")
    )));
    findings.push(Finding::ok(format!(
        "engine build: {}",
        onebrain_engine::engine_build_hash().0
    )));
    findings.push(Finding::ok(format!(
        "llama.cpp: {}",
        onebrain_engine::llama_version()
    )));
    findings.push(Finding::ok(format!(
        "engine capabilities: {}",
        onebrain_engine::system_info().trim()
    )));
    findings.push(Finding::ok(format!(
        "config dir: {}",
        paths.config_dir.display()
    )));
    findings.push(Finding::ok(format!(
        "data dir: {}",
        paths.data_dir.display()
    )));
    findings.push(Finding::ok(format!(
        "model cache: {}",
        paths.model_cache_dir().display()
    )));
    findings.push(Finding::ok(format!(
        "daemon log: {}",
        paths.data_dir.join("logs").join("daemon.log").display()
    )));

    // Config file validity.
    let config_file = paths.config_file();
    if config_file.exists() {
        match Config::load(&config_file) {
            Ok(_) => findings.push(Finding::ok(format!(
                "config file: {} (valid)",
                config_file.display()
            ))),
            Err(e) => findings.push(Finding::fail(
                format!("config file: {e}"),
                "fix the file or delete it — every setting has a working default",
            )),
        }
    } else {
        findings.push(Finding::ok(format!(
            "config file: {} (absent; defaults in effect)",
            config_file.display()
        )));
    }

    // Compute devices.
    let devices = onebrain_engine::devices();
    if devices.is_empty() {
        findings.push(Finding::fail(
            "the engine reports no compute devices at all",
            "reinstall onebrain; if it persists, report a bug with `onebrain --version` output",
        ));
    }
    for dev in &devices {
        findings.push(Finding::ok(format!(
            "device: {} ({}) — {} free of {}",
            dev.name,
            kind_str(dev.kind),
            human_bytes(dev.free_bytes),
            human_bytes(dev.total_bytes),
        )));
    }
    let has_gpu = devices.iter().any(|d| {
        matches!(
            d.kind,
            DeviceKind::Gpu | DeviceKind::IntegratedGpu | DeviceKind::Accelerator
        )
    });
    if !devices.is_empty() && !has_gpu {
        findings.push(Finding::warn(
            "no GPU visible to the engine; inference will run on CPU",
            "install or update GPU drivers (CUDA/Vulkan/Metal), then re-run `onebrain doctor`",
        ));
    }

    // Daemon state.
    match DaemonClient::from_paths(&paths) {
        Ok(client) => match client.status() {
            Ok(status) => {
                let model = match status.get("model") {
                    Some(m) if !m.is_null() => m
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("?")
                        .to_string(),
                    _ => "none".to_string(),
                };
                findings.push(Finding::ok(format!(
                    "daemon: running (pid {}, endpoint {}, model {})",
                    client.state().pid,
                    client.base_url(),
                    model
                )));
            }
            Err(e) => findings.push(Finding::warn(
                format!("daemon: run state on disk but not answering ({e})"),
                "run `onebrain up` to restart it",
            )),
        },
        Err(ClientError::NotRunning { .. }) => {
            findings.push(Finding::warn("daemon: not running", "run `onebrain up`"))
        }
        Err(e) => findings.push(Finding::warn(
            format!("daemon: {e}"),
            "run `onebrain up` to rewrite the run state",
        )),
    }

    // Report.
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&findings).expect("findings serialize")
        );
        return Ok(());
    }

    let count = |s: Status| findings.iter().filter(|f| f.status == s).count();
    println!(
        "{} doctor — {} checks: {} ok, {} warn, {} fail",
        onebrain_proto::PRODUCT_NAME,
        findings.len(),
        count(Status::Ok),
        count(Status::Warn),
        count(Status::Fail)
    );
    println!();
    for f in &findings {
        println!("[{}] {}", f.status.label(), f.message);
        if let Some(remedy) = &f.remedy {
            println!("       remedy: {remedy}");
        }
    }
    Ok(())
}

fn kind_str(kind: DeviceKind) -> &'static str {
    match kind {
        DeviceKind::Cpu => "cpu",
        DeviceKind::Gpu => "gpu",
        DeviceKind::IntegratedGpu => "integrated gpu",
        DeviceKind::Accelerator => "accelerator",
        DeviceKind::Other => "other",
    }
}
