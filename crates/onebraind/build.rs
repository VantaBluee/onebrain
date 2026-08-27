//! Links the macOS frameworks behind `onebraind::power` (M5,
//! docs/resilience.md "Power realities"): `IOPMAssertionCreateWithName` /
//! `IOPMAssertionRelease` live in IOKit and the `CFString` helpers in
//! CoreFoundation. Emitted from THIS crate because the externs are declared
//! here, and via `rustc-link-lib` (framework kind) because — unlike
//! `rustc-link-arg` — it propagates to dependent binaries' links (see the
//! same note in crates/onebrain-engine/build.rs).
//!
//! Other targets need nothing: kernel32 is always linked on Windows, and
//! the Linux paths shell out to `systemd-inhibit` / read sysfs.

fn main() {
    // Build-script boilerplate: nothing here depends on source contents.
    println!("cargo:rerun-if-changed=build.rs");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=IOKit");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
    }
}
