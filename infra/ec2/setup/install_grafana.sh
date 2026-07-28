#!/bin/bash
# Idempotent Grafana installer for Amazon Linux 2023 ARM64
set -euo pipefail

# Pin to specific 12.x version to avoid surprise upgrades to Grafana 13+ (which
# may introduce breaking dashboard schema or plugin loader changes). Bump
# deliberately after testing on a non-prod EC2.
#
# Why 12.x: Grafana 11.x aarch64 RPM ships the Loki datasource with only
# TypeScript source files, no compiled `dist/module.js`. Frontend requests
# /public/plugins/loki/module.js and 404s, breaking any dashboard that uses
# the Logs panel type. 12.x ships the compiled bundle (~670KB module.js).
GRAFANA_VERSION="12.4.3"
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

# Install or update (pin to target version)
sudo dnf install -y "grafana-${GRAFANA_VERSION}"

# Idempotent grafana.ini key setter. The previous `sed 's/^;key =.*/'` form only
# matched the commented default, so it silently no-opped on every re-run after
# the first -- meaning a hand-edited or drifted value was never corrected.
# Safe because http_addr, http_port, root_url, and admin_password each appear
# exactly once in grafana.ini. Escapes &, |, and \ so values containing sed
# replacement metacharacters (notably passwords) are written literally.
set_ini_key() {
    local key="$1" val="$2" file="$3"
    local esc
    esc="$(printf '%s' "$val" | sed -e 's/[&|\\]/\\\\&/g')"
    if sudo grep -qE "^[;#]?[[:space:]]*${key}[[:space:]]*=" "$file"; then
        sudo sed -i -E "s|^[;#]?[[:space:]]*${key}[[:space:]]*=.*|${key} = ${esc}|" "$file"
    else
        printf '%s = %s\n' "$key" "$val" | sudo tee -a "$file" >/dev/null
    fi
}

# Configure grafana.ini for loopback binding. Grafana is NEVER exposed directly:
# `tailscale serve` (see install_tailscale.sh) terminates TLS on the tailnet and
# proxies to 127.0.0.1:3000.
set_ini_key http_addr 127.0.0.1 "${CONFIG_DIR}/grafana.ini"
set_ini_key http_port 3000      "${CONFIG_DIR}/grafana.ini"

# root_url must be the tailnet FQDN, not loopback. Grafana builds absolute links
# from it, so a 127.0.0.1 root_url makes every generated dashboard URL (and the
# MCP `generate_deeplink` tool) emit addresses that are useless off-host.
# python3 rather than jq: python3 ships with Amazon Linux 2023, jq may not.
TS_FQDN="$(tailscale status --json 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' \
    2>/dev/null || true)"
if [ -n "${TS_FQDN}" ]; then
    set_ini_key root_url "https://${TS_FQDN}/" "${CONFIG_DIR}/grafana.ini"
    echo "  root_url = https://${TS_FQDN}/"
else
    echo "  [!] Could not resolve tailnet FQDN; leaving root_url unchanged."
    echo "      Run install_tailscale.sh first, then re-run this script."
fi

# Set admin password from env. Deliberately does NOT default to "admin": this
# script is re-run on a live host, and now that set_ini_key actually matches an
# uncommented key (the old sed silently no-opped), a missing env var would
# overwrite a good password with a known one. Leave it alone instead.
if [ -n "${GRAFANA_ADMIN_PASSWORD:-}" ]; then
    set_ini_key admin_password "${GRAFANA_ADMIN_PASSWORD}" "${CONFIG_DIR}/grafana.ini"
    echo "  admin_password updated from GRAFANA_ADMIN_PASSWORD"
else
    echo "  [!] GRAFANA_ADMIN_PASSWORD not set; leaving admin_password unchanged."
fi

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
if [ -n "${TS_FQDN}" ]; then
    echo "  URL:   https://${TS_FQDN}/  (tailnet, via tailscale serve)"
else
    echo "  URL:   http://127.0.0.1:3000  (loopback only -- tailscale serve not configured)"
fi
echo "  Login: admin / (see GRAFANA_ADMIN_PASSWORD env var)"
