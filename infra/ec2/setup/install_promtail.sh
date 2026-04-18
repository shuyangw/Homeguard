#!/bin/bash
set -euo pipefail

PROMTAIL_VERSION="3.3.2"
PROMTAIL_URL="https://github.com/grafana/loki/releases/download/v${PROMTAIL_VERSION}/promtail-linux-arm64.zip"

echo "[+] Installing Promtail ${PROMTAIL_VERSION}..."

if [ ! -f /usr/local/bin/promtail ] || ! /usr/local/bin/promtail --version 2>&1 | grep -q "${PROMTAIL_VERSION}"; then
    cd /tmp
    curl -fsSL "${PROMTAIL_URL}" -o promtail.zip
    unzip -o promtail.zip
    sudo mv promtail-linux-arm64 /usr/local/bin/promtail
    sudo chmod +x /usr/local/bin/promtail
    rm -f promtail.zip
fi

sudo mkdir -p /var/lib/promtail
sudo mkdir -p /etc/homeguard
sudo cp ~/Homeguard/config/monitoring/promtail/config.yaml /etc/homeguard/promtail.yaml
sudo cp ~/Homeguard/infra/ec2/services/promtail.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable promtail
sudo systemctl restart promtail

echo "[+] Promtail installed and started"
