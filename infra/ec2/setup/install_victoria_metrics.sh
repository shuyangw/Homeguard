#!/bin/bash
# Idempotent VictoriaMetrics installer for Amazon Linux 2023 ARM64
set -euo pipefail

VM_VERSION="v1.106.1"
VM_URL="https://github.com/VictoriaMetrics/VictoriaMetrics/releases/download/${VM_VERSION}/victoria-metrics-linux-arm64-${VM_VERSION}.tar.gz"
INSTALL_DIR="/usr/local/bin"
DATA_DIR="/var/lib/victoria-metrics"
CONFIG_DIR="/etc/homeguard"

echo "[+] Installing VictoriaMetrics ${VM_VERSION} (ARM64)..."

# Check if already installed at this version
if [ -f "${INSTALL_DIR}/victoria-metrics-prod" ]; then
    CURRENT=$("${INSTALL_DIR}/victoria-metrics-prod" --version 2>&1 | head -1 || echo "unknown")
    echo "  Current version: ${CURRENT}"
    if echo "${CURRENT}" | grep -q "${VM_VERSION}"; then
        echo "  Already at ${VM_VERSION}, skipping download"
    else
        echo "  Upgrading to ${VM_VERSION}..."
        cd /tmp
        curl -fsSL "${VM_URL}" -o vm.tar.gz
        tar xzf vm.tar.gz
        sudo mv victoria-metrics-prod "${INSTALL_DIR}/"
        rm -f vm.tar.gz
    fi
else
    cd /tmp
    curl -fsSL "${VM_URL}" -o vm.tar.gz
    tar xzf vm.tar.gz
    sudo mv victoria-metrics-prod "${INSTALL_DIR}/"
    rm -f vm.tar.gz
fi

# Create user
if ! id victoria &>/dev/null; then
    sudo useradd --system --no-create-home --shell /sbin/nologin victoria
    echo "  Created victoria user"
fi

# Create data directory
sudo mkdir -p "${DATA_DIR}"
sudo chown victoria:victoria "${DATA_DIR}"

# Copy config
sudo mkdir -p "${CONFIG_DIR}"
sudo cp ~/Homeguard/config/monitoring/victoria-metrics/scrape.yaml "${CONFIG_DIR}/scrape.yaml"

# Install service
sudo cp ~/Homeguard/infra/ec2/services/victoria-metrics.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable victoria-metrics
sudo systemctl restart victoria-metrics

echo "[+] VictoriaMetrics installed and started"
echo "  UI: http://127.0.0.1:8428/vmui"
echo "  Data: ${DATA_DIR}"
