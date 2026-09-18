#!/usr/bin/env bash
# Install the bundle on a machine that has never seen it.
#
#   scripts/flatpak-test.sh              # the real bundle
#   scripts/flatpak-test.sh --control    # a bundle built WITHOUT --runtime-repo,
#                                        # which must fail -- the negative control
#   scripts/flatpak-test.sh --clean      # remove the image and the cache volume
#
# Debian rather than Fedora on purpose: "a different distribution" is what the
# flatpak exists to prove, and the host is Fedora. The container has no flatpak
# remote, no runtime, and -- just as deliberately -- no CJK font, which is the
# other thing that cannot be tested on a machine that has one.
#
# What this does NOT cover: the GUI (no display), audio (no PulseAudio) and
# Vulkan (no GPU). Install, launch, and font resolution are what it proves.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ID=io.github.AquilaWei.LiveInterpreter
OUT="$REPO/target/flatpak"
DIR="$REPO/packaging/flatpak/test"
IMAGE=li-flatpak-test
VOLUME=li-fp-cache

if [ "${1:-}" = "--clean" ]; then
    docker rmi -f "$IMAGE" >/dev/null 2>&1
    docker volume rm "$VOLUME" >/dev/null 2>&1
    echo "沙盒已刪除（映像與快取 volume）"
    exit 0
fi

BUNDLE="$OUT/LiveInterpreter.flatpak"
if [ "${1:-}" = "--control" ]; then
    # Same build, repackaged without the pointer that tells a stranger's
    # machine where org.gnome.Platform//50 comes from. Must fail.
    BUNDLE="$OUT/NoRuntimeRepo.flatpak"
    flatpak build-bundle "$OUT/repo" "$BUNDLE" "$ID"
fi
[ -f "$BUNDLE" ] || { echo "沒有 $BUNDLE，先跑 scripts/flatpak.sh" >&2; exit 1; }

docker build -q -t "$IMAGE" "$DIR" >/dev/null
docker volume create "$VOLUME" >/dev/null

# --privileged because bubblewrap needs it here; /dev/fuse because ostree's
# checkout does. The container is thrown away either way (--rm).
docker run --rm --privileged \
    --security-opt seccomp=unconfined \
    --device /dev/fuse \
    -v "$VOLUME:/home/tester/.local/share/flatpak" \
    -v "$BUNDLE:/home/tester/$(basename "$BUNDLE"):ro" \
    -v "$DIR/inside.sh:/home/tester/run.sh:ro" \
    -v "$DIR/entry.sh:/entry.sh:ro" \
    --user root "$IMAGE" bash /entry.sh "/home/tester/$(basename "$BUNDLE")"
