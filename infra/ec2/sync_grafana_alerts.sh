#!/bin/bash
# Sync Grafana alert-rule provisioning from the repo to Grafana's config dir.
#
# Idempotent. Safe to run on every git pull.
#
# IMPORTANT -- unlike sync_grafana_dashboards.sh, this script RESTARTS Grafana.
# Grafana's file-watcher covers provisioning/dashboards/ but NOT
# provisioning/alerting/, which is read at startup only. The restart is guarded
# on a content comparison so a routine `git pull` with no rule changes does not
# bounce Grafana.
#
# Target is /etc/grafana/provisioning/alerting/ (root:grafana 640), matching the
# datasource-provisioning convention, rather than /var/lib/grafana/ where the
# dashboard JSONs live.
#
# Usage (run on EC2):
#   bash ~/Homeguard/infra/ec2/sync_grafana_alerts.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_DIR="${REPO_DIR}/config/monitoring/grafana/alerting"
TARGET_DIR="/etc/grafana/provisioning/alerting"

if ! ls "${SOURCE_DIR}"/*.yaml 1>/dev/null 2>&1; then
    echo "[sync-alerts] No alert rule YAMLs found at ${SOURCE_DIR}"
    exit 0
fi

sudo mkdir -p "${TARGET_DIR}"

changed=0
for src in "${SOURCE_DIR}"/*.yaml; do
    dest="${TARGET_DIR}/$(basename "${src}")"
    if ! sudo cmp -s "${src}" "${dest}" 2>/dev/null; then
        sudo cp "${src}" "${dest}"
        sudo chown root:grafana "${dest}"
        sudo chmod 640 "${dest}"
        echo "[sync-alerts] Updated $(basename "${src}")"
        changed=1
    fi
done

# Any ${VAR} referenced by a provisioning file must be present in Grafana's
# EnvironmentFile, or Grafana resolves it to an empty string and silently fails
# to deliver. Surface that here rather than leaving it to be discovered by an
# alert that never arrives.
GRAFANA_ENV_FILE="/etc/homeguard/grafana.env"
missing_vars=""
# Strip comments first: these files document the ${VAR} mechanism in their own
# headers, and scanning comment text yields spurious names (e.g. a literal
# "${VAR}" in prose) which makes the warning untrustworthy and easy to ignore.
for var in $(sed 's/#.*//' "${SOURCE_DIR}"/*.yaml 2>/dev/null \
                | grep -ohE '\$\{[A-Z_][A-Z0-9_]*\}' \
                | tr -d '${}' | sort -u); do
    if ! sudo grep -qE "^${var}=.+" "${GRAFANA_ENV_FILE}" 2>/dev/null; then
        missing_vars="${missing_vars} ${var}"
    fi
done
if [ -n "${missing_vars}" ]; then
    echo "[sync-alerts] [!] Referenced but NOT set in ${GRAFANA_ENV_FILE}:${missing_vars}"
    echo "[sync-alerts]     Alert rules will still evaluate, but notifications"
    echo "[sync-alerts]     will NOT be delivered. See"
    echo "[sync-alerts]     config/monitoring/grafana/grafana.env.example"
fi

count=$(ls "${SOURCE_DIR}"/*.yaml | wc -l)
if [ "${changed}" -eq 0 ]; then
    echo "[sync-alerts] ${count} file(s) already current; skipping Grafana restart"
    exit 0
fi

# provisioning/alerting/ has no file watcher, so a restart is required.
echo "[sync-alerts] Rules changed -> restarting grafana-server"
sudo systemctl restart grafana-server
sleep 5
if sudo systemctl is-active --quiet grafana-server; then
    echo "[sync-alerts] Synced ${count} file(s); grafana-server active"
else
    echo "[sync-alerts] [-] grafana-server is NOT active after restart"
    echo "[sync-alerts]     Check: journalctl -u grafana-server -n 50"
    exit 1
fi

# Provisioning errors are non-fatal to Grafana startup: a malformed rule file is
# logged and skipped, so the service comes up "healthy" with zero rules loaded.
# That silent-skip behaviour is exactly how this rule set sat unloaded from
# 2026-04-18 to 2026-07-27, so surface it here rather than trusting is-active.
if sudo journalctl -u grafana-server --since "-1min" --no-pager 2>/dev/null \
        | grep -iE "provisioning.*(error|failed)" ; then
    echo "[sync-alerts] [-] Provisioning errors above -- rules may NOT be loaded"
    exit 1
fi
echo "[sync-alerts] No provisioning errors in the journal"
