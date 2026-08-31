//! `onebrain doctor`: every check is a finding — (status, message,
//! remedy) — covering identity/paths (v0), compute devices from the
//! engine, battery/power realities (M5, docs/resilience.md), daemon state,
//! and config-file validity. M8 (docs/product.md §3) adds firewall
//! posture, driver/backend hints, and cross-node version-skew via
//! `/api/internal/metrics` — all per-OS, best-effort, never fatal, and
//! runtime-dispatched on the OS name so every per-OS branch compiles on
//! every platform.

use std::process::Command;

use serde::Serialize;

use onebrain_engine::DeviceKind;
use onebraind::config::Config;
use onebraind::paths::AppPaths;
use onebraind::power::{self, InhibitorSupport};

use super::{human_bytes, CliError};
use crate::client::{ClientError, DaemonClient};
use crate::metrics::{MetricsDoc, MetricsPeer};
use crate::update::version::Version;

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
    /// Stable machine id of the check that produced this finding. M8+
    /// findings only — v1 findings predate ids, and retrofitting them
    /// would change the `--json` shape consumers already parse (the field
    /// is simply absent there).
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'static str>,
    status: Status,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    remedy: Option<String>,
}

impl Finding {
    fn ok(message: impl Into<String>) -> Finding {
        Finding {
            id: None,
            status: Status::Ok,
            message: message.into(),
            remedy: None,
        }
    }
    fn warn(message: impl Into<String>, remedy: impl Into<String>) -> Finding {
        Finding {
            id: None,
            status: Status::Warn,
            message: message.into(),
            remedy: Some(remedy.into()),
        }
    }
    fn fail(message: impl Into<String>, remedy: impl Into<String>) -> Finding {
        Finding {
            id: None,
            status: Status::Fail,
            message: message.into(),
            remedy: Some(remedy.into()),
        }
    }
    fn with_id(mut self, id: &'static str) -> Finding {
        self.id = Some(id);
        self
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

    // M8 driver/backend hint (docs/product.md §3): what this BUILD carries
    // vs what the OS says is physically present. The OS-level probe only
    // runs when the engine itself sees no GPU — the case where the hint
    // can say something the device list above cannot.
    let os = std::env::consts::OS;
    let os_gpus = if has_gpu {
        Vec::new()
    } else {
        probe_os_gpus(os)
    };
    findings.extend(driver_findings(
        &onebrain_engine::system_info(),
        has_gpu,
        &os_gpus,
    ));

    // M8 firewall posture, per-OS, best-effort (docs/product.md §3).
    findings.extend(firewall_findings(&probe_firewall()));

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

    // Daemon state. The client is kept when the daemon answers: the M8
    // skew check below reads the metrics document over the same
    // connection details.
    let mut live_client = None;
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
                live_client = Some(client);
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

    // M8 version/engine-build skew across paired nodes, read from
    // `/api/internal/metrics` (docs/product.md §1, §3). Strictly
    // best-effort: a daemon that is down — or one predating the endpoint —
    // degrades to a neutral ok finding, never a failure.
    match &live_client {
        None => findings.push(
            Finding::ok(
                "version skew: not checked (daemon not running; `onebrain up` enables \
                 cross-node checks)",
            )
            .with_id("version-skew"),
        ),
        Some(client) => match client.metrics() {
            Ok(raw) => {
                let doc: MetricsDoc = serde_json::from_value(raw).unwrap_or_default();
                // The document's own identity wins; fall back to this
                // binary's compile-time identity when a field is absent
                // (the endpoint lands in parallel with this consumer).
                let our_version = if doc.node.version.is_empty() {
                    env!("CARGO_PKG_VERSION").to_string()
                } else {
                    doc.node.version.clone()
                };
                let our_build = if doc.node.engine_build.is_empty() {
                    onebrain_engine::engine_build_hash().0
                } else {
                    doc.node.engine_build.clone()
                };
                findings.extend(skew_findings(&our_version, &our_build, &doc.peers));
            }
            Err(_) => findings.push(
                Finding::ok(
                    "version skew: not checked (this daemon does not expose \
                     /api/internal/metrics yet)",
                )
                .with_id("version-skew"),
            ),
        },
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

/// Everything the firewall check observed, separated from the rendering so
/// [`firewall_findings`] is pure and testable on any OS (docs/product.md
/// §3: emit the right text for the OS doctor runs on, compile all paths).
/// `None` always means "probe did not run or could not answer" — the
/// wording degrades accordingly, it never fails.
struct FirewallProbe {
    os: &'static str,
    /// This binary's path, as firewall rules would reference it.
    exe: Option<String>,
    /// Windows: `netsh advfirewall firewall show rule name=all verbose`
    /// output (`verbose` is what includes the `Program:` lines).
    windows_rules: Option<String>,
    /// Linux: `firewall-cmd --state` stdout (zero exit only).
    firewalld_state: Option<String>,
    /// Linux: `ufw status` stdout (zero exit only — needs root, so this
    /// is usually `None` and worded as "not queryable").
    ufw_status: Option<String>,
}

fn probe_firewall() -> FirewallProbe {
    let os = std::env::consts::OS;
    let mut probe = FirewallProbe {
        os,
        exe: std::env::current_exe()
            .ok()
            .map(|p| p.display().to_string()),
        windows_rules: None,
        firewalld_state: None,
        ufw_status: None,
    };
    match os {
        "windows" => {
            probe.windows_rules = run_capture(
                "netsh",
                &[
                    "advfirewall",
                    "firewall",
                    "show",
                    "rule",
                    "name=all",
                    "verbose",
                ],
            )
        }
        "linux" => {
            probe.firewalld_state = run_capture("firewall-cmd", &["--state"]);
            probe.ufw_status = run_capture("ufw", &["status"]);
        }
        // macOS has no CLI to query the Local Network permission — the
        // finding is a note, not a probe result.
        _ => {}
    }
    probe
}

/// Run a probe command; `Some(stdout)` only on a zero exit. Failing to
/// spawn, a non-zero exit (e.g. `ufw` without root, `firewall-cmd` with
/// firewalld stopped) and non-UTF-8 all degrade to `None` — doctor must
/// never die on a missing admin tool.
fn run_capture(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

fn firewall_findings(probe: &FirewallProbe) -> Vec<Finding> {
    let finding = match probe.os {
        "windows" => windows_firewall_finding(probe),
        "macos" => Finding::ok(
            "firewall: macOS asks for Local Network permission the first time this node \
             looks for peers — approve it under System Settings > Privacy & Security > \
             Local Network (and allow incoming connections if the Application Firewall asks)",
        ),
        "linux" => linux_firewall_finding(probe),
        _ => Finding::ok(
            "firewall: no probe for this OS; if pairing stalls, allow onebrain's mesh \
             traffic in your firewall",
        ),
    };
    vec![finding.with_id("firewall")]
}

fn windows_firewall_finding(probe: &FirewallProbe) -> Finding {
    let exe = probe.exe.as_deref().unwrap_or("onebrain.exe");
    match &probe.windows_rules {
        None => Finding::ok(
            "firewall: could not query Windows Defender Firewall rules (netsh unavailable); \
             expect the allow prompt on the daemon's first bind — choose Allow for private \
             networks",
        ),
        Some(rules) if rules.to_lowercase().contains(&exe.to_lowercase()) => Finding::ok(format!(
            "firewall: a Windows Defender Firewall rule exists for {exe}"
        )),
        Some(_) => Finding::warn(
            format!("firewall: no Windows Defender Firewall rule references {exe}"),
            format!(
                "the first `onebrain up` bind pops the Defender prompt — choose Allow for \
                 private networks; or pre-authorize with: netsh advfirewall firewall add rule \
                 name=\"OneBrain\" dir=in action=allow program=\"{exe}\""
            ),
        ),
    }
}

fn linux_firewall_finding(probe: &FirewallProbe) -> Finding {
    if probe
        .firewalld_state
        .as_deref()
        .is_some_and(|s| s.trim() == "running")
    {
        return Finding::warn(
            "firewall: firewalld is active — inbound mesh traffic may be blocked",
            "allow the mesh port (see `onebrain status`): sudo firewall-cmd \
             --add-port=<port>/udp --permanent && sudo firewall-cmd --reload",
        );
    }
    if probe
        .ufw_status
        .as_deref()
        .is_some_and(|s| s.contains("Status: active"))
    {
        return Finding::warn(
            "firewall: ufw is active — inbound mesh traffic may be blocked",
            "allow the mesh port (see `onebrain status`): sudo ufw allow <port>/udp",
        );
    }
    Finding::ok(
        "firewall: firewalld/ufw not detected as active (not every setup is queryable \
         without root); if pairing stalls, allow onebrain's mesh port",
    )
}

/// GPU backends a llama.cpp build advertises in its system-info string.
/// Substring matching is a heuristic — good enough for a hint, and the
/// finding wording stays hedged accordingly.
const GPU_BACKENDS: [&str; 6] = ["CUDA", "Vulkan", "Metal", "HIP", "SYCL", "OpenCL"];

fn gpu_backend_in(system_info: &str) -> Option<&'static str> {
    let lower = system_info.to_lowercase();
    GPU_BACKENDS
        .iter()
        .copied()
        .find(|b| lower.contains(&b.to_lowercase()))
}

/// Best-effort "is a GPU physically present" per OS, runtime-dispatched so
/// all branches compile everywhere. Only consulted when the engine itself
/// sees no GPU; an empty answer just softens the finding. A GPU running
/// only the fallback driver (e.g. Windows' Basic Display Adapter) is
/// deliberately still a hit — that is exactly the missing-driver case the
/// hint exists for.
fn probe_os_gpus(os: &str) -> Vec<String> {
    match os {
        "windows" => run_capture(
            "powershell",
            &[
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name",
            ],
        )
        .map(|out| {
            out.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default(),
        "linux" => {
            let mut gpus = Vec::new();
            if std::path::Path::new("/proc/driver/nvidia/version").exists() {
                gpus.push("NVIDIA GPU (kernel driver loaded)".to_string());
            }
            let render_node = std::fs::read_dir("/dev/dri").ok().is_some_and(|entries| {
                entries
                    .flatten()
                    .any(|e| e.file_name().to_string_lossy().starts_with("renderD"))
            });
            if render_node {
                gpus.push("GPU render node (/dev/dri)".to_string());
            }
            gpus
        }
        // Apple Silicon always has an integrated GPU: if the engine cannot
        // see it, the build is missing Metal. Intel Macs vary, so no claim.
        "macos" if std::env::consts::ARCH == "aarch64" => {
            vec!["Apple Silicon integrated GPU".to_string()]
        }
        _ => Vec::new(),
    }
}

/// The M8 driver/backend hint: one finding relating what the build
/// compiled in (from the system-info string) to what the engine and the
/// OS each see. Pure over probed values, so every quadrant is testable on
/// any machine.
fn driver_findings(system_info: &str, engine_sees_gpu: bool, os_gpus: &[String]) -> Vec<Finding> {
    let backend = gpu_backend_in(system_info);
    let gpu_list = os_gpus.join(", ");
    let finding = match (engine_sees_gpu, backend) {
        (true, Some(b)) => Finding::ok(format!(
            "gpu backend: {b} compiled in and a GPU device is visible"
        )),
        (true, None) => Finding::ok("gpu backend: a GPU device is visible to the engine"),
        (false, Some(b)) if !os_gpus.is_empty() => Finding::warn(
            format!(
                "gpu backend: {b} is compiled in but no GPU is visible to the engine \
                 (OS reports: {gpu_list})"
            ),
            format!("update the GPU driver ({b} runtime), then re-run `onebrain doctor`"),
        ),
        (false, Some(b)) => Finding::ok(format!(
            "gpu backend: {b} compiled in; no GPU detected on this machine"
        )),
        (false, None) if !os_gpus.is_empty() => Finding::warn(
            format!("gpu backend: this build looks CPU-only but the OS reports a GPU ({gpu_list})"),
            "install a GPU-enabled OneBrain build — `onebrain self-update` fetches the \
             standard release; if that stays CPU-only for this platform, build from source \
             with the CUDA/Vulkan/Metal feature",
        ),
        (false, None) => {
            Finding::ok("gpu backend: CPU-only build; no GPU detected on this machine")
        }
    };
    vec![finding.with_id("gpu-backend")]
}

/// Cross-node version / engine-build skew from the metrics document's
/// Hello-reported peer data (docs/product.md §1, §3). Pure over parsed
/// metrics so it is testable without a cluster. The remedy names which
/// side must run `onebrain self-update` when the versions order cleanly,
/// and hedges when they do not parse.
fn skew_findings(our_version: &str, our_build: &str, peers: &[MetricsPeer]) -> Vec<Finding> {
    if peers.is_empty() {
        return vec![
            Finding::ok("version skew: no paired peers to compare").with_id("version-skew")
        ];
    }
    let mut findings = Vec::new();
    let mut comparable = 0usize;
    for peer in peers {
        let name = if peer.name.is_empty() {
            "a paired peer"
        } else {
            peer.name.as_str()
        };
        if !peer.version.is_empty() && peer.version != our_version {
            let remedy = match (Version::parse(&peer.version), Version::parse(our_version)) {
                (Ok(theirs), Ok(ours)) if theirs < ours => format!(
                    "run `onebrain self-update` on {name} — mixed versions refuse to plan \
                     together"
                ),
                (Ok(theirs), Ok(ours)) if theirs > ours => {
                    "run `onebrain self-update` on this machine — mixed versions refuse to \
                     plan together"
                        .to_string()
                }
                _ => format!(
                    "bring this machine and {name} to the same version with \
                     `onebrain self-update` on each"
                ),
            };
            findings.push(
                Finding::warn(
                    format!(
                        "version skew: {name} runs v{}, this node v{our_version}",
                        peer.version
                    ),
                    remedy,
                )
                .with_id("version-skew"),
            );
        } else if !peer.engine_build.is_empty()
            && !our_build.is_empty()
            && peer.engine_build != our_build
        {
            findings.push(
                Finding::warn(
                    format!(
                        "engine build skew: {name} runs {}, this node {our_build}",
                        peer.engine_build
                    ),
                    format!(
                        "same version but different engine builds never cooperate on a plan; \
                         reinstall the release build on {name} or here (`onebrain self-update`)"
                    ),
                )
                .with_id("engine-skew"),
            );
        } else if !peer.version.is_empty() || !peer.engine_build.is_empty() {
            comparable += 1;
        }
        // Peers reporting neither field (an older daemon's document) are
        // skipped silently — claiming "no skew" for them would be a lie.
    }
    if findings.is_empty() {
        findings.push(if comparable == 0 {
            Finding::ok("version skew: peers reported no version data (older daemons?)")
                .with_id("version-skew")
        } else {
            Finding::ok(format!(
                "version skew: none across {comparable} paired node{}",
                if comparable == 1 { "" } else { "s" }
            ))
            .with_id("version-skew")
        });
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

    fn probe(os: &'static str) -> FirewallProbe {
        FirewallProbe {
            os,
            exe: Some("C:\\bin\\onebrain.exe".to_string()),
            windows_rules: None,
            firewalld_state: None,
            ufw_status: None,
        }
    }

    #[test]
    fn windows_firewall_rule_present_absent_and_unqueryable() {
        // Rule mentioning the binary (netsh output is case-mixed) → ok.
        let mut p = probe("windows");
        p.windows_rules =
            Some("Rule Name: x\nProgram: c:\\BIN\\onebrain.EXE\nAction: Allow".into());
        let found = firewall_findings(&p);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, Some("firewall"));
        assert_eq!(found[0].status, Status::Ok);
        assert!(
            found[0].message.contains("rule exists"),
            "{}",
            found[0].message
        );

        // Rules exist, none reference the binary → warn naming the dance.
        p.windows_rules = Some("Rule Name: y\nProgram: C:\\other\\thing.exe".into());
        let found = firewall_findings(&p);
        assert_eq!(found[0].status, Status::Warn);
        let remedy = found[0].remedy.as_deref().unwrap();
        assert!(
            remedy.contains("netsh advfirewall firewall add rule"),
            "{remedy}"
        );
        assert!(remedy.contains("Defender prompt"), "{remedy}");

        // netsh unqueryable → best-effort ok with the first-bind note.
        p.windows_rules = None;
        let found = firewall_findings(&p);
        assert_eq!(found[0].status, Status::Ok);
        assert!(
            found[0].message.contains("first bind"),
            "{}",
            found[0].message
        );
    }

    #[test]
    fn macos_and_linux_firewall_hints() {
        let found = firewall_findings(&probe("macos"));
        assert_eq!(found[0].status, Status::Ok);
        assert!(
            found[0].message.contains("Local Network"),
            "{}",
            found[0].message
        );

        let mut p = probe("linux");
        p.firewalld_state = Some("running\n".into());
        let found = firewall_findings(&p);
        assert_eq!(found[0].status, Status::Warn);
        assert!(found[0].remedy.as_deref().unwrap().contains("firewall-cmd"));

        let mut p = probe("linux");
        p.ufw_status = Some("Status: active\n".into());
        let found = firewall_findings(&p);
        assert_eq!(found[0].status, Status::Warn);
        assert!(found[0].remedy.as_deref().unwrap().contains("ufw allow"));

        // Nothing queryable → honest ok, not a fabricated all-clear.
        let found = firewall_findings(&probe("linux"));
        assert_eq!(found[0].status, Status::Ok);
        assert!(found[0].message.contains("not detected as active"));
    }

    #[test]
    fn driver_findings_cover_all_quadrants() {
        let gpu = vec!["NVIDIA RTX 4070".to_string()];

        // Engine sees a GPU: always ok, backend named when known.
        let f = driver_findings("CUDA : ARCHS = 890 | CPU : AVX = 1", true, &[]);
        assert_eq!(f[0].status, Status::Ok);
        assert!(f[0].message.contains("CUDA"));
        assert_eq!(f[0].id, Some("gpu-backend"));
        let f = driver_findings("CPU : AVX = 1", true, &[]);
        assert_eq!(f[0].status, Status::Ok);

        // Backend compiled, GPU present, engine blind → driver problem.
        let f = driver_findings("Vulkan0 | CPU : AVX = 1", false, &gpu);
        assert_eq!(f[0].status, Status::Warn);
        assert!(f[0].message.contains("NVIDIA RTX 4070"));
        assert!(f[0].remedy.as_deref().unwrap().contains("driver"));

        // CPU-only build on a machine with a GPU → name self-update.
        let f = driver_findings("CPU : AVX = 1", false, &gpu);
        assert_eq!(f[0].status, Status::Warn);
        assert!(f[0].remedy.as_deref().unwrap().contains("self-update"));

        // No GPU anywhere: plain ok statements.
        let f = driver_findings("Metal | CPU", false, &[]);
        assert_eq!(f[0].status, Status::Ok);
        assert!(f[0].message.contains("Metal"));
        let f = driver_findings("CPU : AVX = 1", false, &[]);
        assert_eq!(f[0].status, Status::Ok);
        assert!(f[0].message.contains("CPU-only"));
    }

    fn peer(name: &str, version: &str, build: &str) -> crate::metrics::MetricsPeer {
        crate::metrics::MetricsPeer {
            name: name.to_string(),
            version: version.to_string(),
            engine_build: build.to_string(),
        }
    }

    #[test]
    fn skew_findings_name_the_side_that_must_update() {
        // Peer behind → remedy targets the peer by name.
        let f = skew_findings("0.2.0", "build-a", &[peer("laptop", "0.1.0", "build-b")]);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].status, Status::Warn);
        assert_eq!(f[0].id, Some("version-skew"));
        assert!(f[0].message.contains("laptop runs v0.1.0"));
        assert!(f[0].remedy.as_deref().unwrap().contains("on laptop"));

        // This node behind → remedy targets this machine.
        let f = skew_findings("0.1.0", "build-a", &[peer("laptop", "0.2.0", "build-b")]);
        assert!(f[0].remedy.as_deref().unwrap().contains("this machine"));

        // Unparsable versions → hedged both-sides remedy, still a warn.
        let f = skew_findings("0.1.0", "b", &[peer("laptop", "nightly", "b")]);
        assert_eq!(f[0].status, Status::Warn);
        assert!(f[0].remedy.as_deref().unwrap().contains("on each"));
    }

    #[test]
    fn engine_build_skew_with_matching_versions_is_its_own_finding() {
        let f = skew_findings("0.1.0", "build-a", &[peer("desk", "0.1.0", "build-b")]);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].status, Status::Warn);
        assert_eq!(f[0].id, Some("engine-skew"));
        assert!(f[0].message.contains("engine build skew"));
        assert!(f[0].remedy.as_deref().unwrap().contains("self-update"));
    }

    #[test]
    fn skew_all_clear_and_degraded_paths_stay_ok() {
        let f = skew_findings("0.1.0", "b", &[]);
        assert_eq!(f[0].status, Status::Ok);
        assert!(f[0].message.contains("no paired peers"));

        let f = skew_findings(
            "0.1.0",
            "b",
            &[peer("a", "0.1.0", "b"), peer("c", "0.1.0", "b")],
        );
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].status, Status::Ok);
        assert_eq!(f[0].message, "version skew: none across 2 paired nodes");

        // A peer reporting nothing yields no fabricated all-clear.
        let f = skew_findings("0.1.0", "b", &[peer("mystery", "", "")]);
        assert_eq!(f[0].status, Status::Ok);
        assert!(f[0].message.contains("no version data"));
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
