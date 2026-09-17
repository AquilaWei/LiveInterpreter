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
MANIFEST="$REPO/packaging/flatpak/$ID.yml"
OUT="$REPO/target/flatpak"

bundle=1
if [ "${1:-}" = "--no-bundle" ]; then bundle=0; shift; fi

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
    flatpak build-bundle "$OUT/repo" "$OUT/LiveInterpreter.flatpak" "$ID"
    ls -lh "$OUT/LiveInterpreter.flatpak"
fi

cat <<EOF

裝好了。試法：

  flatpak run --command=liveinterpreter $ID --list-devices
  flatpak run --command=liveinterpreter $ID --fetch-models
  flatpak run $ID

Vulkan 有沒有掉到 llvmpipe，看啟動 log 的裝置名 —— 那是會安靜失敗的那一個。
EOF
