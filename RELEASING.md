# Releasing OneBrain

How a tag becomes a signed, installable release. The pipeline is
`.github/workflows/release.yml` (M8 §5, docs/product.md); everything it
builds is rehearsed on every PR by ci.yml's `release-dry-run` job, so tag
day should never be the first time an installer config is exercised.

## The tag ritual

The tag IS the version. release.yml refuses to build if the tag does not
equal `v<workspace version>`, so artifacts can never lie about what they
contain.

1. **Bump the version** in the root `Cargo.toml`:

   ```toml
   [workspace.package]
   version = "0.1.0"
   ```

   Every crate inherits it; `onebrain --version`, the archive names, the
   MSI/deb/rpm versions, and install.sh's expectations all follow.

2. **Update `Formula/onebrain.rb`**: bump `version` and the three `url`
   tags to the new `vX.Y.Z`. Leave the sha256 placeholders — the real
   values only exist after CI publishes SHA256SUMS (step "After the tag").

3. **Sanity-check locally** (optional but cheap):

   ```sh
   cargo xtask dist        # stages dist/onebrain-v<version>-<host triple>/
   bash -n install.sh
   ```

4. **Commit, tag, push.** Tag names: `v0.1.0-rc.1` first — a prerelease
   tag (anything with a hyphen) publishes as a GitHub *prerelease*, visible
   for validation but never `latest`, so install.sh keeps resolving the
   last stable release. `v0.1.0` follows once the rc run is green
   end-to-end.

   ```sh
   git commit -am "release: v0.1.0-rc.1"
   git tag v0.1.0-rc.1
   git push origin main v0.1.0-rc.1
   ```

5. **Watch the run.** Four `dist (<target>)` jobs, then `publish release`.
   If the version-check step fails, the tag and Cargo.toml disagree: fix
   Cargo.toml, delete the tag (`git push --delete origin vX.Y.Z`,
   `git tag -d vX.Y.Z`), re-tag the fixed commit.

## What CI produces (artifact inventory)

Attached to the GitHub release, all listed in ONE `SHA256SUMS`:

| Artifact | Contents |
|---|---|
| `onebrain-vX.Y.Z-aarch64-apple-darwin.tar.gz` | binary + README + licenses + per-dir SHA256SUMS |
| `onebrain-vX.Y.Z-x86_64-apple-darwin.tar.gz` | " |
| `onebrain-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` | " |
| `onebrain-vX.Y.Z-x86_64-pc-windows-msvc.zip` | " |
| `onebrain-vX.Y.Z-x86_64-pc-windows-msvc.msi` | WiX v4 installer: Program Files, PATH, uninstall entry |
| `onebrain_X.Y.Z-1_amd64.deb` | Debian/Ubuntu package (cargo-deb; native naming) |
| `onebrain-X.Y.Z-1.x86_64.rpm` | Fedora/RHEL package (cargo-generate-rpm; native naming) |
| `SHA256SUMS` | checksums of every file above |
| `SHA256SUMS.sig`, `SHA256SUMS.pem` | cosign keyless signature + certificate |

Signing is keyless (sigstore): the GitHub OIDC token binds the signature to
this repo's release workflow and the tag via a short-lived certificate.
There is no signing key to store, rotate, or leak.

## How a user verifies

Checksums (macOS: `shasum -a 256 -c` instead of `sha256sum -c`):

```sh
sha256sum --check --ignore-missing SHA256SUMS
```

Signature — proves SHA256SUMS came from this repo's release workflow for
this tag, not from a compromised account uploading by hand:

```sh
cosign verify-blob \
  --certificate SHA256SUMS.pem \
  --signature SHA256SUMS.sig \
  --certificate-identity "https://github.com/VantaBluee/onebrain/.github/workflows/release.yml@refs/tags/vX.Y.Z" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  SHA256SUMS
```

(Substitute the real tag for `vX.Y.Z`. The release body carries both
one-liners pre-substituted.)

## After the tag (manual follow-ups)

1. **Fill in the Homebrew formula.** Download the release's `SHA256SUMS`,
   copy the three tarball hashes into `Formula/onebrain.rb`, replacing the
   zero placeholders (each is labeled with its target). Commit to main.
   Until then the formula fails loudly on install — by design, never
   silently installing unverified bytes.

2. **Homebrew tap** (once, then per release): push the filled-in formula to
   a `VantaBluee/homebrew-onebrain` repo so users get
   `brew install VantaBluee/onebrain/onebrain`. Until the tap exists the
   formula installs directly:

   ```sh
   brew install --formula \
     https://raw.githubusercontent.com/VantaBluee/onebrain/main/Formula/onebrain.rb
   ```

3. **Fresh-machine walkthrough** (the DoD's manual half, tracked in
   STATUS.md): on a machine that has never seen OneBrain, follow the README
   quickstart per OS — installer one-liner → `onebrain up` → `onebrain
   pull` → `onebrain run` — and the two-laptop pairing demo. Record the
   result in STATUS.md.

## Notes and sharp edges

- **Prerelease versions in deb/rpm**: rc tags produce package versions with
  the prerelease suffix normalized by the packaging tools (Debian/RPM
  ordering rules differ from semver). The rc run exists to validate exactly
  this — inspect the produced filenames before cutting the final tag.
- **MSI ProductVersion** is numeric-only; for rc tags the installer's
  internal version is `X.Y.Z` while the .msi *filename* keeps the full tag.
  Two rc MSIs of the same X.Y.Z upgrade over each other
  (`AllowSameVersionUpgrades`).
- **Never change** the WiX `UpgradeCode` in `wix/onebrain.wxs` — it is the
  product's permanent identity; changing it breaks upgrades forever.
- **install.sh resolves `releases/latest`**, which GitHub only points at
  non-prerelease releases. Test an rc explicitly with
  `ONEBRAIN_VERSION=v0.1.0-rc.1 ./install.sh`.
- **Runner images**: Intel macOS builds on `macos-15-intel`
  (`macos-13` was retired by GitHub in Dec 2025). Linux builds stay on
  ubuntu-22.04 so binaries link the oldest supported glibc.
