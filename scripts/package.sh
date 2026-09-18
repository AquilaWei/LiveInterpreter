#!/usr/bin/env bash
# Build the Linux packages. Output: target/release/bundle/{deb,rpm}/
#
#   scripts/package.sh
#
# Two reasons this exists rather than "just run cargo tauri build":
#
#  1. The build needs `scripts/build.sh`'s environment (SPIRV-Headers prefix,
#     and the oneDNN lib64 retry). Bare cargo does not have it.
#  2. The packages carry the CLI (`liveinterpreter`) as well as the desktop
#     app, and `cargo tauri build` only builds the desktop crate. The bundler
#     copies the CLI binary from target/release, so it has to exist first or
#     bundling fails with a missing-file error.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$REPO/scripts/build.sh" build --release -p li-cli
cd "$REPO/apps/desktop/src-tauri"
"$REPO/scripts/build.sh" tauri build "$@"
