#!/bin/bash
# Idempotent installer for the read-only Homeguard console agent (Phase 1).
# Run ON the instance as ec2-user from the repo root:
#   bash infra/ec2/setup/install_console_agent.sh
set -euo pipefail

REPO_DIR=/home/ec2-user/Homeguard
UNIT=homeguard-console-agent.service
AGENT_PORT=8090
TAILNET_PORT=8443

if ! grep -q '^CONSOLE_OPERATOR_LOGIN=..*' "$REPO_DIR/.env"; then
    echo "[-] CONSOLE_OPERATOR_LOGIN is not set in $REPO_DIR/.env"
    echo "    Add: CONSOLE_OPERATOR_LOGIN=\"<your tailscale login>\""
    exit 1
fi

echo "[+] Installing $UNIT"
sudo cp "$REPO_DIR/infra/ec2/services/$UNIT" "/etc/systemd/system/$UNIT"
sudo systemctl daemon-reload
sudo systemctl enable "$UNIT"
sudo systemctl restart "$UNIT"

for _ in $(seq 1 10); do
    if curl -s -o /dev/null "http://127.0.0.1:$AGENT_PORT/status"; then
        break
    fi
    sleep 1
done
if ! systemctl is-active --quiet "$UNIT"; then
    echo "[-] $UNIT is not active; see: journalctl -u $UNIT -n 50"
    exit 1
fi
echo "  $UNIT active"

echo "[+] Publishing 127.0.0.1:$AGENT_PORT on the tailnet at :$TAILNET_PORT"
tailscale version | head -1
if sudo tailscale serve status 2>/dev/null | grep -q "127.0.0.1:$AGENT_PORT"; then
    echo "  Already published via tailscale serve"
else
    sudo tailscale serve --bg --https="$TAILNET_PORT" "http://127.0.0.1:$AGENT_PORT"
fi
sudo tailscale serve status
