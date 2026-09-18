#!/usr/bin/env bash
# Build (or run) with the two things this workspace needs that cargo cannot
# work out for itself. See BUILDING.md.
#
#   scripts/build.sh                    # cargo build --release
#   scripts/build.sh run -p li-cli --   # anything else cargo can do
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${LI_PREFIX:-$HOME/.cache/liveinterpreter/local}"

export PATH="$HOME/.cargo/bin:$PATH"
# whisper.cpp's Vulkan backend needs SPIRV-Headers, which Fedora does not
# package. BUILDING.md says how to put it in $PREFIX.
if [ -d "$PREFIX/include" ]; then
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+$CMAKE_PREFIX_PATH:}$PREFIX"
    export CXXFLAGS="${CXXFLAGS:-} -I$PREFIX/include"
fi

# `onednn-src` hard-codes `-L $OUT_DIR/lib`, but oneDNN installs to
# `$OUT_DIR/lib64` on Fedora and every other lib64 distribution, so the link
# fails with `could not find native static library dnnl`. The
# directory only exists after the build script has run once, which is why this
# is a retry rather than a precondition.
link_lib64() {
    local found=1
    for d in "$REPO"/target/*/build/onednn-src-*/out; do
        if [ -d "$d/lib64" ] && [ ! -e "$d/lib" ]; then
            ln -sfn lib64 "$d/lib"
            echo "scripts/build.sh: linked $d/lib -> lib64" >&2
            found=0
        fi
    done
    return $found
}

link_lib64 || true
if [ $# -eq 0 ]; then set -- build --release; fi
if ! cargo "$@"; then
    link_lib64 || exit 1
    exec cargo "$@"
fi
