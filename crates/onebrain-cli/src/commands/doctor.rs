//! `onebrain doctor` v1: every check is a finding — (status, message,
//! remedy) — covering identity/paths (kept from v0), compute devices from
//! the engine, battery/power realities (M5, docs/resilience.md), daemon
//! state, and config-file validity. Grows firewall and self-update checks
//! in M8.

use serde::Serialize;

use onebrain_engine::DeviceKind;
use onebraind::config::Config;
use onebraind::paths::AppPaths;
use onebraind::power::{self, InhibitorSupport};

use super::{human_bytes, CliError};
use crate::client::{ClientError, DaemonClient};

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
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

    // Config file validity. The parsed config is kept: the battery check
    // below reports against the node's real drain threshold (an unreadable
    // file falls back to defaults — the daemon would not start on it
    // anyway, and that is already a `fail` finding here).
    let config_file = paths.config_file();
    let config = if config_file.exists() {
        match Config::load(&config_file) {
            Ok(config) => {
                findings.push(Finding::ok(format!(
                    "config file: {} (valid)",
                    config_file.display()
                )));
                config
            }
            Err(e) => {
                findings.push(Finding::fail(
                    format!("config file: {e}"),
                    "fix the file or delete it — every setting has a working default",
                ));
                Config::default()
            }
        }
    } else {
        findings.push(Finding::ok(format!(
            "config file: {} (absent; defaults in effect)",
            config_file.display()
        )));
        Config::default()
    };

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

    // Battery / power (M5, docs/resilience.md "Power realities"): battery
    // level and AC state against the draining policy, plus whether this
    // node can hold OS sleep during requests.
    let probe = power::platform_battery_probe();
    findings.extend(power_findings(
        probe.level_percent(),
        probe.on_ac(),
        config.battery_drain_threshold,
        &power::sleep_inhibitor_support(),
    ));

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

/// The battery/power findings (M5), pure over already-probed values so the
/// formats are unit-testable on any OS. The draining decision is
/// [`power::battery_verdict`] itself — this report can never disagree with
/// what the daemon advertises in `NodeStatus`.
fn power_findings(
    level: Option<u8>,
    on_ac: Option<bool>,
    threshold: u8,
    inhibit: &InhibitorSupport,
) -> Vec<Finding> {
    let mut findings = Vec::new();

    let verdict = power::battery_verdict(level, on_ac, threshold);
    match level {
        None => findings.push(Finding::ok(
            "battery: none detected (desktop; never drains out of plans)",
        )),
        Some(level) => {
            let ac_desc = match on_ac {
                Some(true) => "on AC",
                Some(false) => "discharging",
                None => "AC state unknown",
            };
            if verdict.draining {
                findings.push(Finding::warn(
                    format!(
                        "battery: {level}% ({ac_desc}) — below the {threshold}% drain threshold"
                    ),
                    "this node advertises draining and is avoided by new plans; plug in to \
                     rejoin them (threshold: battery_drain_threshold in config.toml)",
                ));
            } else {
                findings.push(Finding::ok(format!("battery: {level}% ({ac_desc})")));
            }
        }
    }

    match inhibit {
        InhibitorSupport::Available(mechanism) => {
            findings.push(Finding::ok(format!("sleep inhibit: {mechanism}")));
        }
        InhibitorSupport::Unavailable(reason) => findings.push(Finding::warn(
            format!("sleep inhibit: {reason}"),
            "the OS may sleep mid-generation on this node; install systemd \
             (for systemd-inhibit) or keep it awake manually during long runs",
        )),
    }

    findings
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

#[cfg(test)]
mod tests {
    use super::*;

    const AVAILABLE: InhibitorSupport = InhibitorSupport::Available("test-mechanism");

    #[test]
    fn draining_battery_warns_with_the_threshold_and_a_remedy() {
        let findings = power_findings(Some(19), Some(false), 25, &AVAILABLE);
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].status, Status::Warn);
        assert_eq!(
            findings[0].message,
            "battery: 19% (discharging) — below the 25% drain threshold"
        );
        assert!(findings[0]
            .remedy
            .as_deref()
            .unwrap()
            .contains("battery_drain_threshold"));
    }

    #[test]
    fn healthy_battery_states_are_ok_findings() {
        let on_ac = power_findings(Some(87), Some(true), 25, &AVAILABLE);
        assert_eq!(on_ac[0].status, Status::Ok);
        assert_eq!(on_ac[0].message, "battery: 87% (on AC)");

        // Low but on AC: the policy never drains on AC, so no warning.
        let low_on_ac = power_findings(Some(5), Some(true), 25, &AVAILABLE);
        assert_eq!(low_on_ac[0].status, Status::Ok);

        let discharging_high = power_findings(Some(62), Some(false), 25, &AVAILABLE);
        assert_eq!(discharging_high[0].status, Status::Ok);
        assert_eq!(discharging_high[0].message, "battery: 62% (discharging)");

        let unknown_ac = power_findings(Some(10), None, 25, &AVAILABLE);
        assert_eq!(unknown_ac[0].status, Status::Ok);
        assert_eq!(unknown_ac[0].message, "battery: 10% (AC state unknown)");
    }

    #[test]
    fn desktops_report_no_battery_as_ok() {
        let findings = power_findings(None, Some(true), 25, &AVAILABLE);
        assert_eq!(findings[0].status, Status::Ok);
        assert_eq!(
            findings[0].message,
            "battery: none detected (desktop; never drains out of plans)"
        );
    }

    #[test]
    fn inhibitor_availability_maps_to_ok_or_warn() {
        let ok = power_findings(None, None, 25, &AVAILABLE);
        assert_eq!(ok[1].status, Status::Ok);
        assert_eq!(ok[1].message, "sleep inhibit: test-mechanism");
        assert!(ok[1].remedy.is_none());

        let missing = InhibitorSupport::Unavailable("systemd-inhibit not found on PATH".into());
        let warned = power_findings(None, None, 25, &missing);
        assert_eq!(warned[1].status, Status::Warn);
        assert_eq!(
            warned[1].message,
            "sleep inhibit: systemd-inhibit not found on PATH"
        );
        assert!(warned[1].remedy.as_deref().unwrap().contains("sleep"));
    }
}
