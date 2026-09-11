#!/usr/bin/env bash
# Run LiveInterpreter on this machine. Everything after the script name is
# passed to the program: `scripts/run.sh --list-devices`, `--source mic`, ...
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO/scripts/build.sh" run --release -p li-cli -- "$@"
