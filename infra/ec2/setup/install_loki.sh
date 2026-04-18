#!/bin/bash
set -euo pipefail

LOKI_VERSION="3.3.2"
LOKI_URL="https://github.com/grafana/loki/releases/download/v${LOKI_VERSION}/loki-linux-arm64.zip"

echo "[+] Installing Loki ${LOKI_VERSION}..."

if [ ! -f /usr/local/bin/loki ] || ! /usr/local/bin/loki --version 2>&1 | grep -q "${LOKI_VERSION}"; then
    cd /tmp
    curl -fsSL "${LOKI_URL}" -o loki.zip
    unzip -o loki.zip
    sudo mv loki-linux-arm64 /usr/local/bin/loki
    sudo chmod +x /usr/local/bin/loki
    rm -f loki.zip
fi

if ! id loki &>/dev/null; then
    sudo useradd --system --no-create-home --shell /sbin/nologin loki
fi

sudo mkdir -p /var/lib/loki/{chunks,rules,compactor}
sudo chown -R loki:loki /var/lib/loki

sudo mkdir -p /etc/homeguard
sudo cp ~/Homeguard/config/monitoring/loki/config.yaml /etc/homeguard/loki.yaml
sudo cp ~/Homeguard/infra/ec2/services/loki.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable loki
sudo systemctl restart loki

echo "[+] Loki installed and started on 127.0.0.1:3100"
