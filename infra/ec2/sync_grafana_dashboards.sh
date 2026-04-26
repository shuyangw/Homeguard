#!/bin/bash
# Sync Grafana dashboard JSONs from the repo to Grafana's provisioning dir.
#
# Idempotent. Safe to run on every git pull. Grafana's file-watcher polls
# the provisioning dir every ~10s by default, so changes go live without a
# service restart.
#
# Usage (run on EC2):
#   bash ~/Homeguard/infra/ec2/sync_grafana_dashboards.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_DIR="${REPO_DIR}/config/monitoring/grafana/dashboards"
TARGET_DIR="/var/lib/grafana/dashboards/homeguard"

if ! ls "${SOURCE_DIR}"/*.json 1>/dev/null 2>&1; then
    echo "[sync-dashboards] No dashboard JSONs found at ${SOURCE_DIR}"
    exit 0
fi

sudo mkdir -p "${TARGET_DIR}"
sudo cp "${SOURCE_DIR}"/*.json "${TARGET_DIR}/"
sudo chown -R grafana:grafana "${TARGET_DIR}"
sudo chmod 644 "${TARGET_DIR}"/*.json

count=$(ls "${SOURCE_DIR}"/*.json | wc -l)
echo "[sync-dashboards] Synced ${count} dashboards to ${TARGET_DIR}"
ls -la "${TARGET_DIR}"/ | tail -n +2
