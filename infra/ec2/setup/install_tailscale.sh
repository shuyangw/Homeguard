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

echo "[+] Tailscale setup complete"
echo "  Access Grafana: http://homeguard-ec2:3000"
echo "  Access VM UI:   http://homeguard-ec2:8428/vmui"
echo "  SSH:            ssh ec2-user@homeguard-ec2"
