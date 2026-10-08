#!/bin/bash
# Idempotent installer for the Homeguard Console snapshot uploader (Phase 2a).
# Run ON the instance as ec2-user from the repo root:
#   bash infra/ec2/setup/install_console_upload.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ec2-user/Homeguard}"
UNITS=(homeguard-console-upload.service homeguard-console-upload.timer homeguard-console-upload-shutdown.service)

BUCKET=$(sed -n 's/^CONSOLE_SNAPSHOT_BUCKET=//p' "$REPO_DIR/.env" 2>/dev/null | tail -1 | tr -d "\"' \r" || true)
if [ -z "$BUCKET" ] || [[ "$BUCKET" == "<"*">" ]]; then
    echo "[-] CONSOLE_SNAPSHOT_BUCKET is not set to a real bucket in $REPO_DIR/.env"
    echo "    Add: CONSOLE_SNAPSHOT_BUCKET=\"<your bucket>\""
    exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
    echo "[-] The aws CLI is not installed; the uploader needs it"
    exit 1
fi
aws --version

echo "[+] Installing ${UNITS[*]}"
for unit in "${UNITS[@]}"; do
    sudo cp "$REPO_DIR/infra/ec2/services/$unit" "/etc/systemd/system/$unit"
done
sudo systemd-analyze verify "${UNITS[@]/#//etc/systemd/system/}"
sudo systemctl daemon-reload
sudo systemctl enable --now homeguard-console-upload-shutdown.service homeguard-console-upload.timer

echo "[+] Running one upload now"
# systemctl start fails on a non-zero upload; the instance role has PutObject only, so it cannot list the bucket.
sudo systemctl start homeguard-console-upload.service
journalctl -u homeguard-console-upload.service -n 3 --no-pager
systemctl list-timers homeguard-console-upload.timer --no-pager
