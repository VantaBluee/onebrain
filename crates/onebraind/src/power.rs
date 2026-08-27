//! Per-OS power realities behind traits (docs/resilience.md "Power
//! realities"): holding OS sleep while the node is doing work, and probing
//! the battery for the draining policy peers see in `NodeStatus`.
//!
//! Two traits, exactly as contracted:
//!
//! - [`SleepInhibitor`] — held while the engine host has any loaded model
//!   AND (a request is in flight OR a distributed epoch is active);
//!   released when idle-with-no-epoch. [`should_hold_sleep`] is that
//!   predicate as a pure function so the wiring site stays trivial.
//! - [`BatteryProbe`] — battery percentage and AC state, `None` where the
//!   platform cannot say (desktops report `None` and therefore never
//!   drain). [`battery_status`] applies the policy: below
//!   `config.battery_drain_threshold` (default 25) AND known to be off AC
//!   ⇒ `draining` — which flows into `NodeStatus` (proto v3) and makes the
//!   scheduler avoid this node in new plans unless infeasible without it.
//!
//! Platform impls (each behind `cfg`, OS calls smoke-tested only on their
//! own OS): Windows `SetThreadExecutionState` + `GetSystemPowerStatus`
//! (direct kernel32 externs, no new deps); macOS
//! `IOPMAssertionCreateWithName`/`IOPMAssertionRelease` (IOKit +
//! CoreFoundation externs, linked by this crate's build.rs) + `pmset -g
//! batt` parsing; Linux a `systemd-inhibit … sleep infinity` child process
//! (missing binary = one warning, then no-op) + `/sys/class/power_supply`
//! scanning. [`mock`] carries the trait mocks the policy tests (and
//! siblings' integration tests) use.
//!
//! # wiring: runtime integration (finisher: apply if runtime.rs still says `draining: false`)
//!
//! The sibling owns `runtime.rs`/`server.rs`; this block is the reviewed
//! patch to apply there if their edit has not landed.
//!
//! 1. `runtime.rs::run_blocking`, just before `let mesh_config = …`:
//!
//!    ```text
//!    let battery_probe = crate::power::platform_battery_probe();
//!    let battery_threshold = config.battery_drain_threshold;
//!    ```
//!
//! 2. In the `node_status` provider closure (it already `move`s; both new
//!    bindings move in), replace `draining: false,` and its placeholder
//!    comment with:
//!
//!    ```text
//!    draining: crate::power::battery_status(battery_probe.as_ref(), battery_threshold)
//!        .draining,
//!    ```
//!
//! 3. `server.rs` has two more placeholder `draining: false` sites that
//!    should use the same call once `InternalState` gains
//!    `battery_probe: std::sync::Arc<dyn crate::power::BatteryProbe + Send + Sync>`
//!    and `battery_threshold: u8` (filled from the same values in
//!    `runtime.rs::serve`): the post-bench `NodeStatus` push (`Message::
//!    NodeStatus { …, draining: false }`) and the planner's own-head
//!    `NodeCaps { …, draining: false }`. Head and workers then apply one
//!    policy.
//!
//! 4. Sleep inhibitor: create `crate::power::platform_sleep_inhibitor()`
//!    where the engine host's activity is visible (the host loop, or a
//!    small watcher owning the box), and on every state edge call:
//!
//!    ```text
//!    if crate::power::should_hold_sleep(model_loaded, request_in_flight, epoch_active) {
//!        inhibitor.hold("model active");
//!    } else {
//!        inhibitor.release();
//!    }
//!    ```
//!
//!    `hold`/`release` are idempotent in every impl, so calling them on
//!    each edge (or even each poll) is safe and cheap.
//!
//! Note on freshness: `NodeStatus` is sent per established session and on
//! bench re-push, so a battery crossing the threshold propagates on the
//! next send — a periodic re-send is part of the sibling's integrate work,
//! not this module.

/// Holds the OS awake while this node is doing work the user would lose to
/// a sleep (docs/resilience.md "Power realities"). Impls are idempotent:
/// `hold` while held and `release` while released are no-ops, and every
/// impl releases on `Drop`.
pub trait SleepInhibitor {
    /// Ask the OS not to sleep. `why` is surfaced to the OS where the
    /// platform allows (IOPM assertion name, systemd-inhibit `--why`).
    fn hold(&mut self, why: &str);
    /// Let the OS sleep again.
    fn release(&mut self);
}

/// Reads this node's battery state. `None` means "the platform cannot
/// say" — a desktop without a battery reports `None` for the level and is
/// therefore never considered draining.
pub trait BatteryProbe {
    /// Battery charge, 0–100. `None`: no battery, or unknown.
    fn level_percent(&self) -> Option<u8>;
    /// `Some(true)` on AC power, `Some(false)` discharging, `None` unknown.
    fn on_ac(&self) -> Option<bool>;
}

/// The battery policy's output: what this node advertises in `NodeStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatteryVerdict {
    /// `true` ⇒ advertise "draining, deprioritize": the scheduler excludes
    /// this node from new plans unless a plan is infeasible without it.
    pub draining: bool,
    /// The probed battery level, passed through for status/doctor display.
    pub level: Option<u8>,
}

/// The battery-drain policy over a probe (docs/resilience.md): draining ⇔
/// the level is known and strictly below `threshold_percent` AND the node
/// is known to be off AC. See [`battery_verdict`] for the exact rules.
pub fn battery_status(probe: &dyn BatteryProbe, threshold_percent: u8) -> BatteryVerdict {
    battery_verdict(probe.level_percent(), probe.on_ac(), threshold_percent)
}

/// [`battery_status`]'s policy core, pure over already-probed values (the
/// doctor reuses it so CLI output and the daemon's advertisement can never
/// disagree). Unknowns never drain: a desktop (level `None`) and an
/// unknown AC state both stay `draining: false` — a node is only excluded
/// from plans on positive evidence that it is discharging below the
/// threshold.
pub fn battery_verdict(
    level: Option<u8>,
    on_ac: Option<bool>,
    threshold_percent: u8,
) -> BatteryVerdict {
    let draining = matches!(
        (level, on_ac),
        (Some(l), Some(false)) if l < threshold_percent
    );
    BatteryVerdict { draining, level }
}

/// The sleep-hold predicate (docs/resilience.md): hold while the engine
/// host has any loaded model AND (a request is in flight OR a distributed
/// epoch is active); release when idle-with-no-epoch.
pub fn should_hold_sleep(model_loaded: bool, request_in_flight: bool, epoch_active: bool) -> bool {
    model_loaded && (request_in_flight || epoch_active)
}

/// Whether this platform's sleep-inhibit mechanism is usable — surfaced by
/// `onebrain doctor`'s power section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InhibitorSupport {
    /// The mechanism [`platform_sleep_inhibitor`] will use.
    Available(&'static str),
    /// Why sleep cannot be held on this node (the daemon still runs; the
    /// OS may just sleep mid-request).
    Unavailable(String),
}

/// Probe sleep-inhibit availability without holding anything. Windows and
/// macOS mechanisms are direct OS calls (always present); Linux depends on
/// the `systemd-inhibit` binary.
pub fn sleep_inhibitor_support() -> InhibitorSupport {
    #[cfg(windows)]
    {
        InhibitorSupport::Available("SetThreadExecutionState (kernel32)")
    }
    #[cfg(target_os = "macos")]
    {
        InhibitorSupport::Available("IOPMAssertionCreateWithName (IOKit)")
    }
    #[cfg(target_os = "linux")]
    {
        use std::process::{Command, Stdio};
        match Command::new("systemd-inhibit")
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
        {
            Ok(s) if s.success() => InhibitorSupport::Available("systemd-inhibit (systemd)"),
            Ok(s) => InhibitorSupport::Unavailable(format!(
                "systemd-inhibit is present but `--version` exited with {s}"
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                InhibitorSupport::Unavailable("systemd-inhibit not found on PATH".to_string())
            }
            Err(e) => InhibitorSupport::Unavailable(format!("systemd-inhibit could not run: {e}")),
        }
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        InhibitorSupport::Unavailable(format!(
            "no sleep-inhibit support on {}",
            std::env::consts::OS
        ))
    }
}

/// This platform's [`SleepInhibitor`]. Unsupported platforms get
/// [`NoopSleepInhibitor`] (the daemon still works; the OS may sleep).
pub fn platform_sleep_inhibitor() -> Box<dyn SleepInhibitor + Send> {
    #[cfg(windows)]
    {
        Box::new(windows_impl::WindowsSleepInhibitor::new())
    }
    #[cfg(target_os = "macos")]
    {
        Box::new(macos_impl::MacSleepInhibitor::new())
    }
    #[cfg(target_os = "linux")]
    {
        Box::new(linux_impl::LinuxSleepInhibitor::new())
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        Box::new(NoopSleepInhibitor)
    }
}

/// This platform's [`BatteryProbe`]. Unsupported platforms get
/// [`UnknownBatteryProbe`] (everything `None` ⇒ never draining).
pub fn platform_battery_probe() -> Box<dyn BatteryProbe + Send + Sync> {
    #[cfg(windows)]
    {
        Box::new(windows_impl::WindowsBatteryProbe)
    }
    #[cfg(target_os = "macos")]
    {
        Box::new(macos_impl::MacBatteryProbe)
    }
    #[cfg(target_os = "linux")]
    {
        Box::new(linux_impl::LinuxBatteryProbe::new())
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        Box::new(UnknownBatteryProbe)
    }
}

/// Inhibitor that holds nothing: unsupported platforms, and tests that
/// need a real trait object without OS effects.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopSleepInhibitor;

impl SleepInhibitor for NoopSleepInhibitor {
    fn hold(&mut self, _why: &str) {}
    fn release(&mut self) {}
}

/// Probe for platforms without a battery API: everything unknown, so the
/// policy never marks the node draining.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnknownBatteryProbe;

impl BatteryProbe for UnknownBatteryProbe {
    fn level_percent(&self) -> Option<u8> {
        None
    }
    fn on_ac(&self) -> Option<bool> {
        None
    }
}

pub mod mock {
    //! Trait mocks (docs/resilience.md: "All impls unit-tested behind the
    //! traits with mocks"). Public so sibling crates' integration tests can
    //! drive the policy without a real battery.

    use super::{BatteryProbe, SleepInhibitor};

    /// Battery probe returning fixed values.
    #[derive(Debug, Clone, Copy)]
    pub struct MockBattery {
        /// What `level_percent` returns.
        pub level: Option<u8>,
        /// What `on_ac` returns.
        pub ac: Option<bool>,
    }

    impl BatteryProbe for MockBattery {
        fn level_percent(&self) -> Option<u8> {
            self.level
        }
        fn on_ac(&self) -> Option<bool> {
            self.ac
        }
    }

    /// Inhibitor that records its calls for assertions.
    #[derive(Debug, Default)]
    pub struct MockInhibitor {
        /// Every `why` passed to `hold`, in order.
        pub holds: Vec<String>,
        /// How many times `release` was called.
        pub releases: usize,
    }

    impl SleepInhibitor for MockInhibitor {
        fn hold(&mut self, why: &str) {
            self.holds.push(why.to_string());
        }
        fn release(&mut self) {
            self.releases += 1;
        }
    }
}

#[cfg(windows)]
pub use windows_impl::{WindowsBatteryProbe, WindowsSleepInhibitor};

#[cfg(windows)]
mod windows_impl {
    //! Windows: `SetThreadExecutionState` holds sleep and
    //! `GetSystemPowerStatus` reads the battery — both direct kernel32
    //! externs (docs/resilience.md: no new deps).

    use std::sync::mpsc;

    use super::{BatteryProbe, SleepInhibitor};

    #[link(name = "kernel32")]
    extern "system" {
        fn SetThreadExecutionState(es_flags: u32) -> u32;
        fn GetSystemPowerStatus(status: *mut SystemPowerStatus) -> i32;
    }

    const ES_CONTINUOUS: u32 = 0x8000_0000;
    const ES_SYSTEM_REQUIRED: u32 = 0x0000_0001;

    /// Win32 `SYSTEM_POWER_STATUS`, field for field.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub(super) struct SystemPowerStatus {
        /// 0 offline, 1 online, 255 unknown.
        pub(super) ac_line_status: u8,
        /// Bit 128 = no system battery; 255 = unknown.
        pub(super) battery_flag: u8,
        /// 0–100, or 255 = unknown (desktops report 255).
        pub(super) battery_life_percent: u8,
        // Present for ABI layout only; never read.
        _system_status_flag: u8,
        _battery_life_time: u32,
        _battery_full_life_time: u32,
    }

    pub(super) fn read_power_status() -> Option<SystemPowerStatus> {
        let mut status = SystemPowerStatus::default();
        // SAFETY: out-pointer to a live, properly sized and aligned struct
        // matching the Win32 SYSTEM_POWER_STATUS layout; the call writes it
        // only on success (nonzero return).
        let ok = unsafe { GetSystemPowerStatus(&mut status) };
        (ok != 0).then_some(status)
    }

    /// `ES_CONTINUOUS` execution state is per-thread and cleared when its
    /// thread exits, so one dedicated keeper thread owns it for the
    /// inhibitor's life — `hold`/`release` may then be called from any
    /// thread (the daemon's state edges happen on several).
    pub struct WindowsSleepInhibitor {
        tx: Option<mpsc::Sender<Cmd>>,
        held: bool,
    }

    enum Cmd {
        Hold,
        Release,
    }

    impl WindowsSleepInhibitor {
        pub fn new() -> WindowsSleepInhibitor {
            let (tx, rx) = mpsc::channel::<Cmd>();
            match std::thread::Builder::new()
                .name("ob-sleep-inhibit".to_string())
                .spawn(move || keeper(rx))
            {
                Ok(_) => WindowsSleepInhibitor {
                    tx: Some(tx),
                    held: false,
                },
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "could not start the sleep-inhibit keeper thread; the OS may sleep mid-request"
                    );
                    WindowsSleepInhibitor {
                        tx: None,
                        held: false,
                    }
                }
            }
        }
    }

    impl Default for WindowsSleepInhibitor {
        fn default() -> Self {
            WindowsSleepInhibitor::new()
        }
    }

    fn keeper(rx: mpsc::Receiver<Cmd>) {
        for cmd in rx {
            let flags = match cmd {
                Cmd::Hold => ES_CONTINUOUS | ES_SYSTEM_REQUIRED,
                Cmd::Release => ES_CONTINUOUS,
            };
            // SAFETY: plain flags call, no pointers.
            let prev = unsafe { SetThreadExecutionState(flags) };
            if prev == 0 {
                tracing::warn!("SetThreadExecutionState failed; the OS may sleep mid-request");
            }
        }
        // Sender dropped: clear before exiting (thread exit would clear the
        // continuous state anyway; this keeps the intent explicit).
        // SAFETY: as above.
        unsafe { SetThreadExecutionState(ES_CONTINUOUS) };
    }

    impl SleepInhibitor for WindowsSleepInhibitor {
        fn hold(&mut self, why: &str) {
            if self.held {
                return;
            }
            if let Some(tx) = &self.tx {
                if tx.send(Cmd::Hold).is_ok() {
                    self.held = true;
                    tracing::debug!(why, "holding OS sleep (SetThreadExecutionState)");
                }
            }
        }

        fn release(&mut self) {
            if !self.held {
                return;
            }
            if let Some(tx) = &self.tx {
                let _ = tx.send(Cmd::Release);
            }
            self.held = false;
            tracing::debug!("released OS sleep hold");
        }
    }

    impl Drop for WindowsSleepInhibitor {
        fn drop(&mut self) {
            self.release();
            // Dropping tx ends the keeper loop, which clears the state.
        }
    }

    /// Battery probe over `GetSystemPowerStatus`.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct WindowsBatteryProbe;

    impl BatteryProbe for WindowsBatteryProbe {
        fn level_percent(&self) -> Option<u8> {
            const NO_SYSTEM_BATTERY: u8 = 128;
            const UNKNOWN: u8 = 255;
            let s = read_power_status()?;
            if s.battery_flag & NO_SYSTEM_BATTERY != 0 || s.battery_life_percent == UNKNOWN {
                return None;
            }
            Some(s.battery_life_percent.min(100))
        }

        fn on_ac(&self) -> Option<bool> {
            match read_power_status()?.ac_line_status {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub use macos_impl::{MacBatteryProbe, MacSleepInhibitor};

#[cfg(target_os = "macos")]
mod macos_impl {
    //! macOS: an IOKit power-management assertion holds sleep; `pmset -g
    //! batt` (parsed by [`super::pmset::parse_pmset`]) reads the battery.
    //! The IOKit and CoreFoundation frameworks are linked by this crate's
    //! build.rs (macOS targets only).

    use std::ffi::{c_char, c_void, CString};

    use super::{BatteryProbe, SleepInhibitor};

    type CFStringRef = *const c_void;
    type CFAllocatorRef = *const c_void;
    type IOPMAssertionID = u32;
    type IOReturn = i32;

    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    /// kIOPMAssertionLevelOn.
    const ASSERTION_LEVEL_ON: u32 = 255;
    /// kIOPMAssertionTypePreventUserIdleSystemSleep: the assertion held by
    /// long-running user work (what `caffeinate -i` takes).
    const ASSERTION_TYPE: &str = "PreventUserIdleSystemSleep";

    extern "C" {
        // CoreFoundation
        fn CFStringCreateWithCString(
            alloc: CFAllocatorRef,
            c_str: *const c_char,
            encoding: u32,
        ) -> CFStringRef;
        fn CFRelease(cf: *const c_void);
        // IOKit
        fn IOPMAssertionCreateWithName(
            assertion_type: CFStringRef,
            assertion_level: u32,
            assertion_name: CFStringRef,
            assertion_id: *mut IOPMAssertionID,
        ) -> IOReturn;
        fn IOPMAssertionRelease(assertion_id: IOPMAssertionID) -> IOReturn;
    }

    /// Holds at most one IOPM assertion; released on `release` and `Drop`.
    #[derive(Debug, Default)]
    pub struct MacSleepInhibitor {
        assertion: Option<IOPMAssertionID>,
    }

    impl MacSleepInhibitor {
        pub fn new() -> MacSleepInhibitor {
            MacSleepInhibitor::default()
        }
    }

    impl SleepInhibitor for MacSleepInhibitor {
        fn hold(&mut self, why: &str) {
            if self.assertion.is_some() {
                return;
            }
            let Ok(type_c) = CString::new(ASSERTION_TYPE) else {
                return;
            };
            let Ok(name_c) = CString::new(format!("{}: {why}", onebrain_proto::PRODUCT_NAME))
            else {
                return; // interior NUL in `why`; skip rather than panic
            };
            // SAFETY: both CFStrings are created from valid NUL-terminated
            // UTF-8 and released before returning (Create rule: we own
            // them); the out-pointer targets a live IOPMAssertionID.
            unsafe {
                let cf_type = CFStringCreateWithCString(
                    std::ptr::null(),
                    type_c.as_ptr(),
                    K_CF_STRING_ENCODING_UTF8,
                );
                let cf_name = CFStringCreateWithCString(
                    std::ptr::null(),
                    name_c.as_ptr(),
                    K_CF_STRING_ENCODING_UTF8,
                );
                if cf_type.is_null() || cf_name.is_null() {
                    if !cf_type.is_null() {
                        CFRelease(cf_type);
                    }
                    if !cf_name.is_null() {
                        CFRelease(cf_name);
                    }
                    tracing::warn!(
                        "could not build IOPM assertion strings; the OS may sleep mid-request"
                    );
                    return;
                }
                let mut id: IOPMAssertionID = 0;
                let ret =
                    IOPMAssertionCreateWithName(cf_type, ASSERTION_LEVEL_ON, cf_name, &mut id);
                CFRelease(cf_type);
                CFRelease(cf_name);
                if ret == 0 {
                    tracing::debug!(why, "holding OS sleep (IOPMAssertion)");
                    self.assertion = Some(id);
                } else {
                    tracing::warn!(
                        ret,
                        "IOPMAssertionCreateWithName failed; the OS may sleep mid-request"
                    );
                }
            }
        }

        fn release(&mut self) {
            if let Some(id) = self.assertion.take() {
                // SAFETY: `id` came from a successful create and is
                // released exactly once (take()).
                unsafe { IOPMAssertionRelease(id) };
                tracing::debug!("released OS sleep hold");
            }
        }
    }

    impl Drop for MacSleepInhibitor {
        fn drop(&mut self) {
            self.release();
        }
    }

    /// Battery probe shelling out to `pmset -g batt` per call — cheap at
    /// NodeStatus frequency (per session / bench push), and the only
    /// dependency-free way to read power state on macOS.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct MacBatteryProbe;

    fn pmset_read() -> (Option<u8>, Option<bool>) {
        let out = std::process::Command::new("pmset")
            .args(["-g", "batt"])
            .output();
        match out {
            Ok(out) if out.status.success() => match String::from_utf8(out.stdout) {
                Ok(text) => super::pmset::parse_pmset(&text),
                Err(_) => (None, None),
            },
            _ => (None, None),
        }
    }

    impl BatteryProbe for MacBatteryProbe {
        fn level_percent(&self) -> Option<u8> {
            pmset_read().0
        }
        fn on_ac(&self) -> Option<bool> {
            pmset_read().1
        }
    }
}

/// `pmset -g batt` parsing, compiled everywhere under test so the parser is
/// unit-tested on every CI OS (the subprocess itself only runs on macOS).
#[cfg(any(target_os = "macos", test))]
mod pmset {
    /// Parse `pmset -g batt` output to (level, on_ac). Typical output:
    ///
    /// ```text
    /// Now drawing from 'Battery Power'
    ///  -InternalBattery-0 (id=1234567)    87%; discharging; 4:32 remaining present: true
    /// ```
    ///
    /// A desktop Mac prints the AC line with no `InternalBattery` rows ⇒
    /// `(None, Some(true))` ⇒ never draining.
    pub(super) fn parse_pmset(text: &str) -> (Option<u8>, Option<bool>) {
        let mut level = None;
        let mut on_ac = None;
        for line in text.lines() {
            if line.contains("Now drawing from") {
                if line.contains("AC Power") {
                    on_ac = Some(true);
                } else if line.contains("Battery Power") {
                    on_ac = Some(false);
                }
            }
            if level.is_none() && line.contains("InternalBattery") {
                level = percent_before_sign(line);
            }
        }
        (level, on_ac)
    }

    /// The digits immediately before the first `%` in `line`, clamped to
    /// 100. `None` when there is no percent token.
    fn percent_before_sign(line: &str) -> Option<u8> {
        let percent = line.find('%')?;
        let digits: &str = line[..percent].trim_end_matches(|c: char| !c.is_ascii_digit());
        let start = digits
            .rfind(|c: char| !c.is_ascii_digit())
            .map(|i| i + 1)
            .unwrap_or(0);
        let digits = &digits[start..];
        let value: u32 = digits.parse().ok()?;
        Some(value.min(100) as u8)
    }
}

#[cfg(target_os = "linux")]
pub use linux_impl::{LinuxBatteryProbe, LinuxSleepInhibitor};

#[cfg(target_os = "linux")]
mod linux_impl {
    //! Linux: a `systemd-inhibit … sleep infinity` child holds sleep (kill
    //! to release); `/sys/class/power_supply` scanning (shared with the
    //! tests via [`super::sysfs`]) reads the battery.

    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};

    use super::{BatteryProbe, SleepInhibitor};

    /// Holds sleep by keeping a `systemd-inhibit` child alive. A missing
    /// binary warns once and degrades to a no-op (docs/resilience.md).
    #[derive(Debug, Default)]
    pub struct LinuxSleepInhibitor {
        child: Option<Child>,
        warned_missing: bool,
    }

    impl LinuxSleepInhibitor {
        pub fn new() -> LinuxSleepInhibitor {
            LinuxSleepInhibitor::default()
        }
    }

    impl SleepInhibitor for LinuxSleepInhibitor {
        fn hold(&mut self, why: &str) {
            if self.child.is_some() {
                return;
            }
            let spawned = Command::new("systemd-inhibit")
                .arg("--what=sleep")
                .arg("--who=onebrain")
                .arg(format!("--why={why}"))
                .arg("--mode=block")
                .arg("sleep")
                .arg("infinity")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn();
            match spawned {
                Ok(child) => {
                    tracing::debug!(why, pid = child.id(), "holding OS sleep (systemd-inhibit)");
                    self.child = Some(child);
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    if !self.warned_missing {
                        self.warned_missing = true;
                        tracing::warn!(
                            "systemd-inhibit not found; OneBrain cannot hold OS sleep — \
                             the machine may sleep mid-request"
                        );
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "failed to spawn systemd-inhibit; the OS may sleep mid-request"
                    );
                }
            }
        }

        fn release(&mut self) {
            if let Some(mut child) = self.child.take() {
                let _ = child.kill();
                let _ = child.wait(); // reap; never leaves a zombie holding the inhibit
                tracing::debug!("released OS sleep hold");
            }
        }
    }

    impl Drop for LinuxSleepInhibitor {
        fn drop(&mut self) {
            self.release();
        }
    }

    /// Battery probe over `/sys/class/power_supply`.
    #[derive(Debug, Clone)]
    pub struct LinuxBatteryProbe {
        root: PathBuf,
    }

    impl LinuxBatteryProbe {
        pub fn new() -> LinuxBatteryProbe {
            LinuxBatteryProbe {
                root: PathBuf::from("/sys/class/power_supply"),
            }
        }
    }

    impl Default for LinuxBatteryProbe {
        fn default() -> Self {
            LinuxBatteryProbe::new()
        }
    }

    impl BatteryProbe for LinuxBatteryProbe {
        fn level_percent(&self) -> Option<u8> {
            super::sysfs::scan_power_supply(&self.root).0
        }
        fn on_ac(&self) -> Option<bool> {
            super::sysfs::scan_power_supply(&self.root).1
        }
    }
}

/// `/sys/class/power_supply` scanning, compiled everywhere under test so
/// the logic is unit-tested (against a fake tree) on every CI OS.
#[cfg(any(target_os = "linux", test))]
mod sysfs {
    use std::path::Path;

    /// Scan a power-supply class directory to (level, on_ac).
    ///
    /// Entries are read in name order (deterministic: `BAT0` before
    /// `BAT1`). The first `type == Battery` supplies `capacity` (level)
    /// and, as an AC fallback, `status` (`Discharging` ⇒ off AC;
    /// `Charging`/`Full`/`Not charging` ⇒ on AC). Any non-battery entry
    /// with an `online` file (Mains, USB-PD, …) is authoritative for AC:
    /// on AC iff any of them is online. Missing tree or no battery ⇒
    /// `None`s (desktop; never draining).
    pub(super) fn scan_power_supply(root: &Path) -> (Option<u8>, Option<bool>) {
        let Ok(entries) = std::fs::read_dir(root) else {
            return (None, None);
        };
        let mut dirs: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        dirs.sort();

        let mut level: Option<u8> = None;
        let mut mains_online: Option<bool> = None;
        let mut battery_says_ac: Option<bool> = None;

        for dir in dirs {
            let read = |name: &str| -> Option<String> {
                std::fs::read_to_string(dir.join(name))
                    .ok()
                    .map(|s| s.trim().to_string())
            };
            let Some(kind) = read("type") else {
                continue;
            };
            if kind == "Battery" {
                if level.is_none() {
                    level = read("capacity")
                        .and_then(|s| s.parse::<u32>().ok())
                        .map(|v| v.min(100) as u8);
                }
                if battery_says_ac.is_none() {
                    battery_says_ac = match read("status").as_deref() {
                        Some("Discharging") => Some(false),
                        Some("Charging") | Some("Full") | Some("Not charging") => Some(true),
                        _ => None,
                    };
                }
            } else if let Some(online) = read("online").and_then(|s| s.parse::<u8>().ok()) {
                mains_online = Some(mains_online.unwrap_or(false) || online != 0);
            }
        }
        (level, mains_online.or(battery_says_ac))
    }
}

#[cfg(test)]
mod tests {
    use super::mock::{MockBattery, MockInhibitor};
    use super::*;

    fn verdict(level: Option<u8>, ac: Option<bool>) -> BatteryVerdict {
        battery_status(&MockBattery { level, ac }, 25)
    }

    #[test]
    fn below_threshold_off_ac_drains() {
        assert_eq!(
            verdict(Some(24), Some(false)),
            BatteryVerdict {
                draining: true,
                level: Some(24)
            }
        );
        // "below" is strict: exactly at the threshold does not drain.
        assert!(!verdict(Some(25), Some(false)).draining);
    }

    #[test]
    fn above_threshold_never_drains() {
        assert!(!verdict(Some(80), Some(false)).draining);
        assert!(!verdict(Some(100), Some(false)).draining);
    }

    #[test]
    fn on_ac_never_drains_even_when_low() {
        let v = verdict(Some(3), Some(true));
        assert!(!v.draining);
        assert_eq!(v.level, Some(3)); // level still surfaces for display
    }

    #[test]
    fn unknowns_never_drain() {
        // Desktop: no battery at all.
        assert_eq!(
            verdict(None, Some(true)),
            BatteryVerdict {
                draining: false,
                level: None
            }
        );
        assert!(!verdict(None, None).draining);
        // Low battery but AC state unknown: no positive evidence of
        // discharge, so the node is not excluded from plans.
        assert!(!verdict(Some(5), None).draining);
    }

    #[test]
    fn threshold_comes_from_config_not_a_constant() {
        let probe = MockBattery {
            level: Some(40),
            ac: Some(false),
        };
        assert!(!battery_status(&probe, 25).draining);
        assert!(battery_status(&probe, 50).draining);
    }

    #[test]
    fn sleep_hold_predicate_truth_table() {
        // Held only with a loaded model AND (in-flight OR epoch).
        assert!(should_hold_sleep(true, true, false));
        assert!(should_hold_sleep(true, false, true));
        assert!(should_hold_sleep(true, true, true));
        // Idle-with-no-epoch releases even with a model loaded.
        assert!(!should_hold_sleep(true, false, false));
        // No model ⇒ never held, whatever else claims to be active.
        assert!(!should_hold_sleep(false, true, true));
        assert!(!should_hold_sleep(false, false, false));
    }

    #[test]
    fn mock_inhibitor_records_calls() {
        let mut inhibitor = MockInhibitor::default();
        inhibitor.hold("generation active");
        inhibitor.hold("epoch active");
        inhibitor.release();
        assert_eq!(inhibitor.holds, vec!["generation active", "epoch active"]);
        assert_eq!(inhibitor.releases, 1);
    }

    #[test]
    fn noop_inhibitor_and_unknown_probe_are_inert() {
        let mut noop = NoopSleepInhibitor;
        noop.hold("anything");
        noop.release();
        let probe = UnknownBatteryProbe;
        assert_eq!(probe.level_percent(), None);
        assert_eq!(probe.on_ac(), None);
        assert!(!battery_status(&probe, 100).draining);
    }

    // --- pmset parser (macOS battery path, testable everywhere) ---

    #[test]
    fn pmset_battery_discharging_parses() {
        let text = "Now drawing from 'Battery Power'\n \
                    -InternalBattery-0 (id=6094947)\t87%; discharging; 4:32 remaining present: true\n";
        assert_eq!(super::pmset::parse_pmset(text), (Some(87), Some(false)));
    }

    #[test]
    fn pmset_on_ac_charging_parses() {
        let text = "Now drawing from 'AC Power'\n \
                    -InternalBattery-0 (id=6094947)\t100%; charged; 0:00 remaining present: true\n";
        assert_eq!(super::pmset::parse_pmset(text), (Some(100), Some(true)));
    }

    #[test]
    fn pmset_desktop_mac_has_ac_but_no_battery() {
        let text = "Now drawing from 'AC Power'\n";
        assert_eq!(super::pmset::parse_pmset(text), (None, Some(true)));
    }

    #[test]
    fn pmset_garbage_yields_unknowns() {
        assert_eq!(super::pmset::parse_pmset(""), (None, None));
        assert_eq!(
            super::pmset::parse_pmset("no such tool output"),
            (None, None)
        );
    }

    // --- /sys/class/power_supply scanner (Linux battery path, testable
    // everywhere against a fake tree) ---

    fn fake_supply(dir: &std::path::Path, name: &str, files: &[(&str, &str)]) {
        let root = dir.join(name);
        std::fs::create_dir_all(&root).unwrap();
        for (file, value) in files {
            std::fs::write(root.join(file), format!("{value}\n")).unwrap();
        }
    }

    #[test]
    fn sysfs_battery_discharging_off_ac() {
        let dir = tempfile::tempdir().unwrap();
        fake_supply(
            dir.path(),
            "BAT0",
            &[
                ("type", "Battery"),
                ("capacity", "17"),
                ("status", "Discharging"),
            ],
        );
        fake_supply(dir.path(), "AC", &[("type", "Mains"), ("online", "0")]);
        assert_eq!(
            super::sysfs::scan_power_supply(dir.path()),
            (Some(17), Some(false))
        );
    }

    #[test]
    fn sysfs_mains_online_wins_over_battery_status() {
        let dir = tempfile::tempdir().unwrap();
        // Charger just plugged in: status file still stale on "Discharging".
        fake_supply(
            dir.path(),
            "BAT0",
            &[
                ("type", "Battery"),
                ("capacity", "55"),
                ("status", "Discharging"),
            ],
        );
        fake_supply(dir.path(), "AC", &[("type", "Mains"), ("online", "1")]);
        assert_eq!(
            super::sysfs::scan_power_supply(dir.path()),
            (Some(55), Some(true))
        );
    }

    #[test]
    fn sysfs_battery_status_is_the_ac_fallback_without_mains() {
        let dir = tempfile::tempdir().unwrap();
        fake_supply(
            dir.path(),
            "BAT0",
            &[
                ("type", "Battery"),
                ("capacity", "90"),
                ("status", "Charging"),
            ],
        );
        assert_eq!(
            super::sysfs::scan_power_supply(dir.path()),
            (Some(90), Some(true))
        );
    }

    #[test]
    fn sysfs_desktop_reports_no_battery() {
        let dir = tempfile::tempdir().unwrap();
        fake_supply(dir.path(), "AC", &[("type", "Mains"), ("online", "1")]);
        assert_eq!(
            super::sysfs::scan_power_supply(dir.path()),
            (None, Some(true))
        );
        // And a machine with no power_supply entries (or no such tree at
        // all) is all-unknown ⇒ never draining.
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(super::sysfs::scan_power_supply(empty.path()), (None, None));
        assert_eq!(
            super::sysfs::scan_power_supply(&empty.path().join("missing")),
            (None, None)
        );
    }

    #[test]
    fn sysfs_first_battery_by_name_wins() {
        let dir = tempfile::tempdir().unwrap();
        fake_supply(
            dir.path(),
            "BAT1",
            &[("type", "Battery"), ("capacity", "99"), ("status", "Full")],
        );
        fake_supply(
            dir.path(),
            "BAT0",
            &[
                ("type", "Battery"),
                ("capacity", "12"),
                ("status", "Discharging"),
            ],
        );
        let (level, _) = super::sysfs::scan_power_supply(dir.path());
        assert_eq!(level, Some(12)); // BAT0 sorts first
    }

    // --- Windows-real smoke tests (own-OS only, per the contract) ---

    #[cfg(windows)]
    #[test]
    fn windows_power_status_is_parseable() {
        // The raw call must succeed on any real Windows box or VM.
        let raw = super::windows_impl::read_power_status()
            .expect("GetSystemPowerStatus failed on a real Windows host");
        assert!(matches!(raw.ac_line_status, 0 | 1 | 255));
        // Percent is 0–100 or the documented 255 = unknown.
        assert!(raw.battery_life_percent <= 100 || raw.battery_life_percent == 255);

        // And through the trait: values are range-checked and coherent.
        let probe = WindowsBatteryProbe;
        if let Some(level) = probe.level_percent() {
            assert!(level <= 100);
        }
        let _ = probe.on_ac(); // any of Some(true|false)/None is valid
    }

    #[cfg(windows)]
    #[test]
    fn windows_inhibitor_hold_release_roundtrip() {
        // Transiently holds ES_SYSTEM_REQUIRED; harmless and instant.
        let mut inhibitor = WindowsSleepInhibitor::new();
        inhibitor.hold("power smoke test");
        inhibitor.hold("idempotent");
        inhibitor.release();
        inhibitor.release(); // idempotent
    }

    #[test]
    fn platform_constructors_and_support_probe_do_not_panic() {
        let _ = platform_sleep_inhibitor();
        let probe = platform_battery_probe();
        if let Some(level) = probe.level_percent() {
            assert!(level <= 100);
        }
        match sleep_inhibitor_support() {
            InhibitorSupport::Available(mechanism) => assert!(!mechanism.is_empty()),
            InhibitorSupport::Unavailable(reason) => assert!(!reason.is_empty()),
        }
    }
}
