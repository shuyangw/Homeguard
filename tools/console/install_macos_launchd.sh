#!/bin/bash
# Installs the Homeguard Console as a launchd agent that starts at login.
#   bash tools/console/install_macos_launchd.sh /path/to/env/bin/python
set -euo pipefail

PYTHON="${1:?usage: install_macos_launchd.sh /path/to/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.homeguard.console.plist"

"$PYTHON" -c "import fastapi, uvicorn, jinja2, httpx, boto3" || {
    echo "[-] Missing dependencies; run: $PYTHON -m pip install -r $REPO/tools/console/requirements.txt"
    exit 1
}
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
sed -e "s|__PYTHON__|$PYTHON|g" -e "s|__REPO__|$REPO|g" -e "s|__HOME__|$HOME|g" \
    "$REPO/tools/console/com.homeguard.console.plist" > "$PLIST"
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "[+] Homeguard Console installed; open http://127.0.0.1:8765 (log: ~/Library/Logs/homeguard-console.log)"
