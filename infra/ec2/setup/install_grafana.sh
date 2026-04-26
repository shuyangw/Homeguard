#!/bin/bash
# Idempotent Grafana installer for Amazon Linux 2023 ARM64
set -euo pipefail

GRAFANA_VERSION="11.4.0"
CONFIG_DIR="/etc/grafana"
PROVISIONING_DIR="${CONFIG_DIR}/provisioning"

echo "[+] Installing Grafana ${GRAFANA_VERSION}..."

# Add Grafana yum repo if not present
if [ ! -f /etc/yum.repos.d/grafana.repo ]; then
    cat <<'REPO' | sudo tee /etc/yum.repos.d/grafana.repo
[grafana]
name=grafana
baseurl=https://rpm.grafana.com
repo_gpgcheck=1
enabled=1
gpgcheck=1
gpgkey=https://rpm.grafana.com/gpg.key
sslverify=1
sslcacert=/etc/pki/tls/certs/ca-bundle.crt
REPO
    echo "  Added Grafana yum repo"
fi

# Install or update (pin to target version to avoid surprise upgrades to Grafana 13+)
sudo dnf install -y "grafana-${GRAFANA_VERSION}"

# Workaround: Grafana 11.4.0 aarch64 rpm ships Loki datasource with an empty
# dist/ subdirectory but no dist/plugin.json. The plugin finder silently skips
# the entire plugin directory. Copy plugin.json into dist/ so it gets discovered.
LOKI_PLUGIN_DIR="/usr/share/grafana/public/app/plugins/datasource/loki"
if [ -f "${LOKI_PLUGIN_DIR}/plugin.json" ] && [ ! -f "${LOKI_PLUGIN_DIR}/dist/plugin.json" ]; then
    sudo mkdir -p "${LOKI_PLUGIN_DIR}/dist"
    sudo cp "${LOKI_PLUGIN_DIR}/plugin.json" "${LOKI_PLUGIN_DIR}/dist/plugin.json"
    echo "  Patched Loki plugin.json into dist/ (rpm packaging bug workaround)"
fi

# Configure grafana.ini for localhost binding
sudo sed -i 's/^;http_addr =.*/http_addr = 127.0.0.1/' "${CONFIG_DIR}/grafana.ini"
sudo sed -i 's/^;http_port =.*/http_port = 3000/' "${CONFIG_DIR}/grafana.ini"

# Set admin password from env
GRAFANA_PASS="${GRAFANA_ADMIN_PASSWORD:-admin}"
sudo sed -i "s/^;admin_password =.*/admin_password = ${GRAFANA_PASS}/" "${CONFIG_DIR}/grafana.ini"

# Provision datasources
sudo mkdir -p "${PROVISIONING_DIR}/datasources"
sudo cp ~/Homeguard/config/monitoring/grafana/datasources.yaml \
    "${PROVISIONING_DIR}/datasources/homeguard.yaml"

# Provision dashboards directory
sudo mkdir -p "${PROVISIONING_DIR}/dashboards"
cat <<'DASH' | sudo tee "${PROVISIONING_DIR}/dashboards/homeguard.yaml"
apiVersion: 1
providers:
  - name: 'Homeguard'
    orgId: 1
    folder: 'Homeguard'
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards/homeguard
      foldersFromFilesStructure: false
DASH

# Copy dashboard JSON files (idempotent; reuses the standalone sync script
# so a single source of truth handles both initial install and ongoing updates).
if [ -f ~/Homeguard/infra/ec2/sync_grafana_dashboards.sh ]; then
    bash ~/Homeguard/infra/ec2/sync_grafana_dashboards.sh
fi

# Install service override
sudo cp ~/Homeguard/infra/ec2/services/grafana-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable grafana-server
sudo systemctl restart grafana-server

echo "[+] Grafana installed and started"
echo "  URL: http://127.0.0.1:3000"
echo "  Login: admin / (see GRAFANA_ADMIN_PASSWORD env var)"
