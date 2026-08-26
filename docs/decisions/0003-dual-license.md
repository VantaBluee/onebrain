# ADR 0003: Dual license MIT OR Apache-2.0

Status: accepted (M0, 2026-08-26)

## Context

OneBrain is open source, written in Rust, and vendors MIT-licensed llama.cpp
as a submodule that we statically link. The license must be maximally easy for
users and downstreams, conventional for contributors, and compatible with what
we embed.

## Decision

License the entire workspace as **MIT OR Apache-2.0** (licensee's choice):

- It is the Rust ecosystem convention (rustc, cargo, and most of crates.io),
  so contributors and downstream crates get exactly the terms they expect and
  OneBrain crates can be depended on without license friction.
- **Apache-2.0 contributes an express patent grant**, protecting users and
  contributors in a space (inference runtimes, scheduling) where that matters.
- **MIT keeps compatibility trivially clear with llama.cpp** (MIT) and with
  MIT-only downstreams; the vendored submodule remains under its own upstream
  MIT license and copyright.

Mechanics: `license = "MIT OR Apache-2.0"` in workspace metadata,
`LICENSE-MIT` (copyright "OneBrain contributors") and `LICENSE-APACHE`
(canonical text) at the repo root, and the standard inbound=outbound note in
the README: contributions are dual-licensed under the same terms.

## Consequences

- Downstreams pick either license; nothing in the dependency graph (llama.cpp
  included) conflicts with either choice.
- Contributions need no CLA; the README statement makes inbound=outbound
  explicit.
- Both license files must ship with every distributed artifact, alongside
  llama.cpp's upstream license notice — a release-packaging (M8 `dist`)
  checklist item.
