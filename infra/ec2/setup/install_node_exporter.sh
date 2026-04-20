#!/bin/bash
# Idempotent node_exporter installer for Amazon Linux 2023 ARM64
set -euo pipefail

NE_VERSION="1.8.2"
NE_URL="https://github.com/prometheus/node_exporter/releases/download/v${NE_VERSION}/node_exporter-${NE_VERSION}.linux-arm64.tar.gz"
INSTALL_DIR="/usr/local/bin"

echo "[+] Installing node_exporter ${NE_VERSION} (ARM64)..."

if [ -f "${INSTALL_DIR}/node_exporter" ]; then
    CURRENT=$("${INSTALL_DIR}/node_exporter" --version 2>&1 | head -1 || echo "unknown")
    echo "  Current: ${CURRENT}"
    if echo "${CURRENT}" | grep -q "${NE_VERSION}"; then
        echo "  Already at ${NE_VERSION}, skipping download"
    else
        echo "  Upgrading to ${NE_VERSION}..."
        cd /tmp
        curl -fsSL "${NE_URL}" -o ne.tar.gz
        tar xzf ne.tar.gz
        sudo mv "node_exporter-${NE_VERSION}.linux-arm64/node_exporter" "${INSTALL_DIR}/"
        rm -rf "node_exporter-${NE_VERSION}.linux-arm64" ne.tar.gz
    fi
else
    cd /tmp
    curl -fsSL "${NE_URL}" -o ne.tar.gz
    tar xzf ne.tar.gz
    sudo mv "node_exporter-${NE_VERSION}.linux-arm64/node_exporter" "${INSTALL_DIR}/"
    rm -rf "node_exporter-${NE_VERSION}.linux-arm64" ne.tar.gz
fi

# Create system user
if ! id node_exporter &>/dev/null; then
    sudo useradd --system --no-create-home --shell /sbin/nologin node_exporter
    echo "  Created node_exporter user"
fi

# Install service
sudo cp ~/Homeguard/infra/ec2/services/node-exporter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable node-exporter
sudo systemctl restart node-exporter

echo "[+] node_exporter installed and started"
echo "  Metrics: http://127.0.0.1:9100/metrics"
