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

GRAFANA_ENV_FILE="/etc/homeguard/grafana.env"

# Return the ${VAR} names a provisioning file requires.
#
# Comments are stripped first: these files document the interpolation mechanism
# in their own headers, so scanning comment text yields phantom names (a literal
# "${VAR}" in prose) and a check that reports phantoms is one people ignore.
required_vars_for() {
    sed 's/#.*//' "$1" 2>/dev/null \
        | grep -ohE '\$\{[A-Z_][A-Z0-9_]*\}' \
        | tr -d '${}' | sort -u
}

# A file whose required secret is missing MUST NOT be installed.
#
# This is not a nicety. An unset ${VAR} does not degrade to "provisioned but
# undeliverable" -- Grafana FAILS ITS PROVISIONING MODULE AND REFUSES TO START:
#   Failed to provision alerting: ... could not find webhook url property
#   grafana-server.service: Main process exited, code=exited, status=1/FAILURE
# and then crash-loops, taking down dashboards and rule evaluation with it. So a
# missing secret must mean "this feature is absent", never "monitoring is down".
# Learned the hard way on 2026-07-27.
changed=0
for src in "${SOURCE_DIR}"/*.yaml; do
    name="$(basename "${src}")"
    dest="${TARGET_DIR}/${name}"

    missing=""
    for var in $(required_vars_for "${src}"); do
        if ! sudo grep -qE "^${var}=.+" "${GRAFANA_ENV_FILE}" 2>/dev/null; then
            missing="${missing} ${var}"
        fi
    done

    if [ -n "${missing}" ]; then
        echo "[sync-alerts] [!] SKIPPING ${name}: unset in ${GRAFANA_ENV_FILE}:${missing}"
        echo "[sync-alerts]     Installing it would prevent grafana-server from starting."
        echo "[sync-alerts]     See config/monitoring/grafana/grafana.env.example"
        # Remove a previously-installed copy, otherwise the next restart -- which
        # may be hours later, or a reboot -- fails for a reason nobody connects
        # to this change.
        if sudo test -f "${dest}"; then
            sudo rm -f "${dest}"
            echo "[sync-alerts]     Removed stale ${name} to keep Grafana bootable."
            changed=1
        fi
        continue
    fi

    if ! sudo cmp -s "${src}" "${dest}" 2>/dev/null; then
        sudo cp "${src}" "${dest}"
        sudo chown root:grafana "${dest}"
        sudo chmod 640 "${dest}"
        echo "[sync-alerts] Updated ${name}"
        changed=1
    fi
done

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
