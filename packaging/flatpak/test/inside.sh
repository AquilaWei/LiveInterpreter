#!/usr/bin/env bash
set -uo pipefail
ID=io.github.AquilaWei.LiveInterpreter
BUNDLE="${1:-/home/tester/LiveInterpreter.flatpak}"
say() { printf '\n=== %s ===\n' "$*"; }

say "1. 這台機器的起點"
echo "發行版: $(. /etc/os-release; echo "$PRETTY_NAME")"
echo "--- flatpak --user remotes ---"; flatpak --user remotes || true
echo "--- flatpak remotes (含 system) ---"; flatpak remotes || true
echo "已安裝 runtime: $(flatpak --user list --runtime | wc -l)"
echo "系統 CJK 字型: $(fc-list :lang=zh-tw | wc -l)"

say "2. flatpak install --user -y $(basename "$BUNDLE")"
flatpak install --user -y "$BUNDLE"; rc=$?
echo "exit=$rc"
[ $rc -ne 0 ] && exit $rc

say "3. 裝完之後"
flatpak --user remotes
flatpak --user list --columns=application,version,ref

say "4. CLI"
flatpak run --command=liveinterpreter $ID --version 2>&1

say "5. 中文字型解析（這台主機 0 個 CJK 字型）"
echo -n "Noto Sans TC   -> "; flatpak run --command=fc-match $ID "Noto Sans TC" 2>&1
echo -n "sans:lang=zh-tw-> "; flatpak run --command=fc-match $ID "sans:lang=zh-tw" 2>&1

say "6. 包進去的字型"
flatpak run --command=ls $ID -l /app/share/fonts/ 2>&1
