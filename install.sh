#!/usr/bin/env bash
# OneBrain installer (macOS + Linux) — docs/product.md §4.
#
#   curl -fsSL https://raw.githubusercontent.com/VantaBluee/onebrain/main/install.sh | bash
#
# Detects OS/arch, downloads the release tarball + SHA256SUMS from GitHub
# releases, verifies the checksum, and installs the `onebrain` binary to
# ~/.local/bin (no root needed; /usr/local/bin when run as root). Idempotent:
# re-running replaces the binary atomically with the same result.
#
# Environment overrides:
#   ONEBRAIN_VERSION      tag to install (e.g. v0.1.0 or 0.1.0); default: the
#                         latest non-prerelease GitHub release
#   ONEBRAIN_INSTALL_DIR  install directory; default ~/.local/bin
#                         (/usr/local/bin when running as root)
#
# Windows is served by the .msi on the releases page, not this script.

set -euo pipefail

REPO="VantaBluee/onebrain"

say() { printf '%s\n' "$*" >&2; }
die() {
  say "error: $*"
  exit 1
}
need() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed"
}

need curl
need tar
need mktemp

# --- Detect OS/arch and map to the release target triple ------------------
os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
  Darwin)
    case "$arch" in
      arm64 | aarch64) target="aarch64-apple-darwin" ;;
      x86_64) target="x86_64-apple-darwin" ;;
      *) die "unsupported macOS architecture: $arch" ;;
    esac
    ;;
  Linux)
    case "$arch" in
      x86_64 | amd64) target="x86_64-unknown-linux-gnu" ;;
      *) die "no prebuilt binary for Linux/$arch yet — build from source: https://github.com/${REPO}#install" ;;
    esac
    ;;
  MINGW* | MSYS* | CYGWIN*)
    die "on Windows, use the .msi installer from https://github.com/${REPO}/releases"
    ;;
  *)
    die "unsupported OS: $os"
    ;;
esac

# --- Resolve the release tag ----------------------------------------------
tag="${ONEBRAIN_VERSION:-}"
if [ -z "$tag" ]; then
  json="$(curl -fsSL --proto '=https' --tlsv1.2 \
    "https://api.github.com/repos/${REPO}/releases/latest")" ||
    die "could not reach the GitHub releases API (pin a version with ONEBRAIN_VERSION=vX.Y.Z)"
  case "$json" in
    *'"tag_name"'*) ;;
    *) die "no release found — the project may not have shipped one yet (pin one with ONEBRAIN_VERSION=vX.Y.Z)" ;;
  esac
  # Pipeline-free JSON field grab (jq-less, SIGPIPE-proof under pipefail):
  # cut everything through the first "tag_name" key, then take the next
  # quoted string.
  rest="${json#*\"tag_name\"}"
  rest="${rest#*\"}"
  tag="${rest%%\"*}"
  [ -n "$tag" ] || die "could not parse the latest release tag (pin one with ONEBRAIN_VERSION=vX.Y.Z)"
fi
case "$tag" in
  v*) ;;
  *) tag="v${tag}" ;;
esac

archive="onebrain-${tag}-${target}.tar.gz"
base="https://github.com/${REPO}/releases/download/${tag}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# --- Download and verify ---------------------------------------------------
say "downloading ${archive} (${tag}) ..."
curl -fsSL --proto '=https' --tlsv1.2 -o "${tmp}/${archive}" "${base}/${archive}"
curl -fsSL --proto '=https' --tlsv1.2 -o "${tmp}/SHA256SUMS" "${base}/SHA256SUMS"

# SHA256SUMS covers every release file; extract our line and check it.
grep -F "  ${archive}" "${tmp}/SHA256SUMS" > "${tmp}/expected.sum" ||
  die "${archive} is not listed in SHA256SUMS for ${tag}"
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$tmp" && sha256sum --check --quiet expected.sum) ||
    die "checksum mismatch for ${archive} — refusing to install"
else
  # macOS ships shasum (perl) but not coreutils' sha256sum.
  (cd "$tmp" && shasum -a 256 --check --quiet expected.sum) ||
    die "checksum mismatch for ${archive} — refusing to install"
fi
say "checksum verified"

# --- Install ---------------------------------------------------------------
tar -xzf "${tmp}/${archive}" -C "$tmp"
bin="${tmp}/onebrain-${tag}-${target}/onebrain"
[ -f "$bin" ] || die "unexpected archive layout: ${archive} does not contain onebrain-${tag}-${target}/onebrain"

if [ -n "${ONEBRAIN_INSTALL_DIR:-}" ]; then
  dir="$ONEBRAIN_INSTALL_DIR"
elif [ "$(id -u)" -eq 0 ]; then
  dir="/usr/local/bin"
else
  dir="${HOME}/.local/bin"
fi
mkdir -p "$dir"

# Copy-then-rename: atomic on the same filesystem, and replacing a running
# binary this way never hits ETXTBSY.
staged="${dir}/.onebrain.new.$$"
cp "$bin" "$staged"
chmod 755 "$staged"
mv -f "$staged" "${dir}/onebrain"

"${dir}/onebrain" --version >/dev/null 2>&1 ||
  die "installed ${dir}/onebrain but it failed to run on this machine"
say "installed: ${dir}/onebrain ($("${dir}/onebrain" --version))"

# --- PATH guidance ---------------------------------------------------------
case ":${PATH}:" in
  *":${dir}:"*) ;;
  *)
    say ""
    say "note: ${dir} is not on your PATH. Add it with:"
    say "  export PATH=\"${dir}:\$PATH\""
    say "and put that line in your shell profile (~/.bashrc, ~/.zshrc, ...)."
    ;;
esac

say ""
say "next steps:"
say "  onebrain up      # start the daemon"
say "  onebrain pull    # fetch a model"
say "  onebrain run     # talk to it"
