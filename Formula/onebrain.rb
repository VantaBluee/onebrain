# typed: false
# frozen_string_literal: true

# OneBrain Homebrew formula (docs/product.md section 4). Binary-release
# based: it installs the prebuilt tarball from GitHub releases rather than
# building from source (the vendored llama.cpp build needs CMake + a C++
# toolchain, which a formula should not drag in).
#
# Installable directly from the repo:
#   brew install --formula \
#     https://raw.githubusercontent.com/VantaBluee/onebrain/main/Formula/onebrain.rb
# and ready to be pushed to a tap repo (manual follow-up, see RELEASING.md).
#
# Release ritual (RELEASING.md "After the tag"): after CI publishes a
# release, bump `version`, update the three `url` tags to match, and replace
# each sha256 placeholder below with the matching line from the release's
# SHA256SUMS file. The zeros are deliberate placeholders — brew refuses to
# install while they remain, so a stale formula fails loudly, never wrongly.
class Onebrain < Formula
  desc "One logical machine for local AI, made from computers you already own"
  homepage "https://github.com/VantaBluee/onebrain"
  version "0.1.0"
  license any_of: ["MIT", "Apache-2.0"]

  on_macos do
    on_arm do
      url "https://github.com/VantaBluee/onebrain/releases/download/v0.1.0/onebrain-v0.1.0-aarch64-apple-darwin.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000" # REPLACE: aarch64-apple-darwin line from SHA256SUMS
    end
    on_intel do
      url "https://github.com/VantaBluee/onebrain/releases/download/v0.1.0/onebrain-v0.1.0-x86_64-apple-darwin.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000" # REPLACE: x86_64-apple-darwin line from SHA256SUMS
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/VantaBluee/onebrain/releases/download/v0.1.0/onebrain-v0.1.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000" # REPLACE: x86_64-unknown-linux-gnu line from SHA256SUMS
    end
  end

  def install
    # The tarball unpacks to onebrain-<tag>-<target>/; brew strips the
    # single top-level directory, leaving the binary at the root.
    bin.install "onebrain"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/onebrain --version")
  end
end
