# 0005 — Dashboard: hand-written static SPA, no framework, no build step, no CDN

Date: 2026-08-31 · Status: accepted

## Context

Spec §7 wants a dashboard served by the daemon itself and suggests
"Svelte/React built at release time" while simultaneously demanding a
dependency-light product. Those pull in opposite directions: a framework
build means a Node toolchain in CI, a `node_modules` tree (hundreds of
transitive packages) in the supply chain, a bundler config to maintain,
and a second artifact pipeline feeding the Rust release build. The
dashboard itself is small: five read-only views over one JSON document
(`GET /api/internal/metrics`), polled every 2 s.

## Decision

The dashboard is a hand-written static page — semantic HTML + vanilla JS
+ CSS, a few small files under `crates/onebrain-dash/assets/` — embedded
into the binary with `rust-embed` and served by `onebrain_dash::router()`
(`GET /` for the shell, `GET /dash/*` for assets). Concretely:

- **No framework, no build step.** The files on disk are the files
  served, byte for byte. Reviewing the dashboard is reading it.
- **No CDN, no external fonts.** A strict same-origin page: works
  air-gapped, keeps §10's "content stays on the machines" true for the
  UI too (no third party learns you run a dashboard), and cannot break
  or be tampered with via a remote asset.
- **Rendering is pure functions** (`render.js`: metrics JSON in, HTML/SVG
  string out) wired to the DOM by a thin `app.js`. There is no Node test
  runner in this repo, so testability comes from purity plus Rust tests
  asserting the embedded assets, the app-root marker, and the routes.
- **No auth logic in the crate.** The shell is the one Bearer-exempt page
  (it contains no data); it asks for the API token once, keeps it in
  `localStorage`, and sends it as `Authorization: Bearer` on every
  metrics poll. Auth enforcement stays where it lives today: the daemon.
- **Budget: ~1.5k lines of JS**, enforced by a unit test. That number is
  the revisit trigger — under it, a framework buys nothing.

Rejected alternatives:

- *Svelte/React built at release time*: adds Node + npm to CI and the
  supply chain for five read-only views; the release pipeline (M8 §5)
  stays Rust-only without it.
- *Framework from a CDN*: violates the offline story, adds a runtime
  third-party dependency to a security-sensitive page, and makes the
  dashboard's behavior change without a release.

## Consequences

- DOM and SVG are hand-rolled; the topology graph is generated SVG text,
  not a charting library. Fine at this size, and the first thing to
  outgrow the budget if views multiply.
- If the JS crosses ~1.5k lines (test fails), that is the signal to
  re-open this ADR, not to delete the test.
- Assets ship with `Cache-Control: no-cache` so an upgraded daemon never
  serves a stale shell against a new metrics schema.
- The renderer must tolerate absent optional fields (additive-stable
  schema, product.md §1) — it degrades to "—", it never throws.
