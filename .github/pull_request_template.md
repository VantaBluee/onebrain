<!--
Thanks! Two quick pointers before you fill this in:
- CONTRIBUTING.md has the build prereqs, the gate, and the conventions
  (error remedies, tracing, contract citations, vendor-patch rules).
- The contracts under docs/ are binding: name the one your change
  implements or amends.
-->

## What & why

<!-- What the change does, and the problem it solves. Link issues. -->

## Contract / ADR impact

<!-- Which docs/*.md contract covers this area? Does the change amend it,
     or require a new ADR in docs/decisions/? "None" is a valid answer if
     you say why. -->

## The gate

The same checks CI runs — please make them green locally first
(`RUSTFLAGS=-Dwarnings` for clippy):

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets` (warnings denied)
- [ ] `cargo test --workspace`
- [ ] `cargo xtask smoke`
- [ ] `cargo xtask e2e`
- [ ] `cargo xtask pair-sim`
- [ ] `cargo xtask sim`
- [ ] N/A or done: netem legs (Linux + root: `cargo xtask sim --netem`) for
      mesh/RPC/scheduler-timing changes
- [ ] N/A or done: installer configs touched ⇒ `release-dry-run` CI job is
      green on this PR

## Guarantees touched

<!-- OneBrain's headline guarantees are test-asserted: distributed==solo
     byte-identity (§9), loopback-only listeners (§10), zero-WAN P2P reuse,
     transparent retry, prefill-overlap timing. If your change touches one,
     say which assertion covers it (or what new assertion you added). -->

- [ ] No user-visible guarantee changed, or the covering test is named above

<!-- Reminder: STATUS.md and CHANGELOG.md are maintained by the milestone
     process — leave them untouched unless a maintainer asked. -->
