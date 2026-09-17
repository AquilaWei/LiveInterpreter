#!/usr/bin/env bash
# flatpak talks to both buses; a bare container has neither.
mkdir -p /run/dbus /home/tester/.local/share/flatpak
dbus-daemon --system --fork 2>/dev/null || true
chown -R tester:tester /home/tester/.local
exec runuser -u tester -- dbus-run-session -- bash /home/tester/run.sh "$@"
