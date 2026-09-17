#!/usr/bin/env bash
# Build the Flatpak (task 1.14b). Output: target/flatpak/LiveInterpreter.flatpak
#
#   scripts/flatpak.sh              # build, install --user, write the bundle
#   scripts/flatpak.sh --no-bundle  # build and install only
#
# Companion to scripts/package.sh, which still builds the .rpm and .deb. Both
# stay until the flatpak is shown to work: the rpm is the only package that has
# ever actually been installed (dnf install, 2026-09-11).
#
# flatpak-builder is not installed on the host and does not need to be -- it
# ships as a flatpak of its own (org.flatpak.Builder), which is what this runs.
#
# Everything lands under target/, and that matters: the manifest's `dir` source
# copies the repository, skipping `target`. A build directory anywhere else in
# the tree would copy itself.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ID="io.github.AquilaWei.LiveInterpreter"
MANIFEST_DIR="$REPO/packaging/flatpak"
MANIFEST="$MANIFEST_DIR/$ID.yml"
OUT="$REPO/target/flatpak"

bundle=1
if [ "${1:-}" = "--no-bundle" ]; then bundle=0; shift; fi

# The AppStream metainfo carries the version a second time, and `flatpak list`
# reads that one rather than the binary. 1.2.2 shipped advertising itself as
# 1.2.1 because this file was not appended to. CHANGELOG says the workspace
# Cargo.toml is the single source of truth for the version, so make the build
# enforce it rather than trusting anyone to remember.
version() { sed -n 's/^version = "\(.*\)"$/\1/p' "$REPO/Cargo.toml" | head -1; }
declared() { sed -n 's/.*<release version="\([^"]*\)".*/\1/p' "$MANIFEST_DIR/$ID.metainfo.xml" | head -1; }
if [ "$(version)" != "$(declared)" ]; then
    echo "metainfo 最新的 <release> 是 $(declared)，但 Cargo.toml 是 $(version)。" >&2
    echo "補一行 <release version=\"$(version)\" date=\"$(date +%F)\" /> 再建。" >&2
    exit 1
fi

mkdir -p "$OUT"

# --ccache matters more here than it looks: flatpak-builder wipes the build
# directory for every run, so whisper.cpp, CTranslate2 and oneDNN are compiled
# from source again on each rebuild. The cache lives in the state dir and
# survives.
flatpak run org.flatpak.Builder \
    --force-clean --ccache \
    --user --install \
    --state-dir "$OUT/state" \
    --repo "$OUT/repo" \
    "$@" \
    "$OUT/build" "$MANIFEST"

if [ "$bundle" -eq 1 ]; then
    # --runtime-repo is what makes the bundle installable on a machine that has
    # never seen org.gnome.Platform//50. It puts a pointer in the bundle, not
    # the 2.3 GB runtime, so `flatpak install ./LiveInterpreter.flatpak` can add
    # the remote and pull the dependency itself. Without it the other machine
    # stops at "No remote refs found", which is the error this very host hit
    # when flathub existed only as a system remote.
    flatpak build-bundle \
        --runtime-repo=https://flathub.org/repo/flathub.flatpakrepo \
        "$OUT/repo" "$OUT/LiveInterpreter.flatpak" "$ID"
    ls -lh "$OUT/LiveInterpreter.flatpak"
fi

cat <<EOF

裝好了。試法：

  flatpak run --command=liveinterpreter $ID --list-devices
  flatpak run --command=liveinterpreter $ID --fetch-models
  flatpak run $ID

Vulkan 有沒有掉到 llvmpipe，看啟動 log 的裝置名 —— 那是會安靜失敗的那一個。
EOF
