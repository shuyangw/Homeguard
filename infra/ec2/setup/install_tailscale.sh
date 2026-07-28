#!/bin/bash
# Idempotent Tailscale installer for Amazon Linux 2023
set -euo pipefail

echo "[+] Installing Tailscale..."

# Install if not present
if ! command -v tailscale &>/dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
    echo "  Tailscale installed"
else
    echo "  Tailscale already installed"
fi

# Authenticate and connect
# Auth key must be set in environment: TAILSCALE_AUTH_KEY
if [ -z "${TAILSCALE_AUTH_KEY:-}" ]; then
    echo "[-] TAILSCALE_AUTH_KEY not set. Set it in .env and source it."
    echo "    export TAILSCALE_AUTH_KEY=tskey-auth-..."
    exit 1
fi

# Check if already connected
if tailscale status &>/dev/null 2>&1; then
    echo "  Already connected to tailnet"
    tailscale status
else
    sudo tailscale up \
        --authkey="${TAILSCALE_AUTH_KEY}" \
        --hostname=homeguard-ec2 \
        --ssh
    echo "  Connected to tailnet as homeguard-ec2"
fi

# Add systemd override for resource limits
sudo mkdir -p /etc/systemd/system/tailscaled.service.d
cat <<'OVERRIDE' | sudo tee /etc/systemd/system/tailscaled.service.d/limits.conf
[Service]
MemoryMax=100M
OOMScoreAdjust=-100
OVERRIDE
sudo systemctl daemon-reload

# Publish Grafana over the tailnet with a real TLS certificate. Grafana stays
# bound to loopback (see install_grafana.sh); tailscaled terminates TLS and
# proxies to 127.0.0.1:3000. --bg persists in tailscaled state, so this survives
# the EventBridge stop/start cycle without re-running.
#
# Requires "HTTPS Certificates" enabled for the tailnet (admin console -> DNS).
# Non-fatal on failure: tailscale itself is already up and SSH still works, so a
# missing cert should not abort the installer.
if sudo tailscale serve status 2>/dev/null | grep -q '127.0.0.1:3000'; then
    echo "  Grafana already published via tailscale serve"
elif sudo tailscale serve --bg 3000; then
    echo "  Published Grafana via tailscale serve -> 127.0.0.1:3000"
else
    echo "  [!] 'tailscale serve --bg 3000' failed."
    echo "      Most likely cause: HTTPS Certificates not enabled for this tailnet."
    echo "      Enable it in the admin console (DNS -> HTTPS Certificates), then re-run."
fi
sudo tailscale serve status || true

# VictoriaMetrics (8428) and Loki (3100) are deliberately NOT served. Both are
# bound to 127.0.0.1 and are reached through Grafana's datasource proxy
# (access: proxy in config/monitoring/grafana/datasources.yaml), so they never
# need tailnet exposure. Use an SSH tunnel for direct UI access.
TS_FQDN="$(tailscale status --json 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' \
    2>/dev/null || true)"

echo "[+] Tailscale setup complete"
if [ -n "${TS_FQDN}" ]; then
    echo "  Grafana:  https://${TS_FQDN}/         (via tailscale serve -> 127.0.0.1:3000)"
else
    echo "  Grafana:  <tailnet FQDN unresolved>   (via tailscale serve -> 127.0.0.1:3000)"
fi
echo "  VM UI:    ssh -L 8428:127.0.0.1:8428 ec2-user@homeguard-ec2  (not served; loopback only)"
echo "  Loki:     ssh -L 3100:127.0.0.1:3100 ec2-user@homeguard-ec2  (not served; loopback only)"
echo "  SSH:      ssh ec2-user@homeguard-ec2"
