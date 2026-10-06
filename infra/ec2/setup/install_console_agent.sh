#!/bin/bash
# Idempotent installer for the read-only Homeguard console agent (Phase 1).
# Run ON the instance as ec2-user from the repo root:
#   bash infra/ec2/setup/install_console_agent.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ec2-user/Homeguard}"
UNIT=homeguard-console-agent.service
AGENT_PORT=8090
TAILNET_PORT=8443

LOGIN=$(sed -n 's/^CONSOLE_OPERATOR_LOGIN=//p' "$REPO_DIR/.env" 2>/dev/null | tail -1 | tr -d "\"' \r" || true)
if [ -z "$LOGIN" ] || [[ "$LOGIN" == "<"*">" ]]; then
    echo "[-] CONSOLE_OPERATOR_LOGIN is not set to a real Tailscale login in $REPO_DIR/.env"
    echo "    Add: CONSOLE_OPERATOR_LOGIN=\"<your tailscale login>\""
    exit 1
fi

echo "[+] Installing $UNIT"
sudo cp "$REPO_DIR/infra/ec2/services/$UNIT" "/etc/systemd/system/$UNIT"
sudo systemctl daemon-reload
sudo systemctl enable "$UNIT"
sudo systemctl restart "$UNIT"

# A 403 to a request without the login header proves the agent is up and enforcing auth;
# is-active alone can read "active" between the restarts of a crash-looping unit.
code=""
for _ in $(seq 1 15); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$AGENT_PORT/status" || true)
    if [ "$code" = "403" ]; then
        break
    fi
    sleep 1
done
if [ "$code" != "403" ]; then
    echo "[-] $UNIT did not answer 403 on 127.0.0.1:$AGENT_PORT (got '${code:-no response}'); see: journalctl -u $UNIT -n 50"
    exit 1
fi
echo "  $UNIT up and enforcing auth"

echo "[+] Publishing 127.0.0.1:$AGENT_PORT on the tailnet at :$TAILNET_PORT"
tailscale version | head -1
if sudo tailscale serve status 2>/dev/null | grep -q "127.0.0.1:$AGENT_PORT"; then
    echo "  Already published via tailscale serve"
else
    sudo tailscale serve --bg --https="$TAILNET_PORT" "http://127.0.0.1:$AGENT_PORT"
fi
sudo tailscale serve status
