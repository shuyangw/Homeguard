#!/bin/bash
# A7 paper-validation counter passive check.
# Reads /var/lib/homeguard/a7_clean_sessions on EC2 and appends to a local log.
# Designed for daily Mon-Fri at 16:10 ET via Windows Task Scheduler.
#
# Output:  $HOME/.homeguard/a7_log/a7_check.log
# Review:  tail -n 50 ~/.homeguard/a7_log/a7_check.log
#
# See docs/operations/A7_MONITORING_SETUP.md for setup instructions.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f infra/ec2/load_env.sh ]; then
    echo "[!] infra/ec2/load_env.sh not found; cannot SSH to EC2" >&2
    exit 1
fi

# shellcheck source=/dev/null
source infra/ec2/load_env.sh

LOG_DIR="$HOME/.homeguard/a7_log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/a7_check.log"

TIMESTAMP=$(date -Iseconds)
SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -i "$EC2_SSH_KEY_PATH")

# Read counter
if COUNTER=$(ssh "${SSH_OPTS[@]}" "$EC2_USER@$EC2_IP" \
        "cat /var/lib/homeguard/a7_clean_sessions 2>/dev/null || echo 0" 2>/dev/null); then
    :
else
    COUNTER="SSH_FAIL"
fi

# Pull recent comparator activity (filtered for A7-relevant lines)
if ACTIVITY=$(ssh "${SSH_OPTS[@]}" "$EC2_USER@$EC2_IP" \
        "sudo journalctl -u homeguard-multi --since '2 hours ago' --no-pager 2>/dev/null | grep -iE 'a7|clean_session|variant.*v11|reset' | tail -10" 2>/dev/null); then
    :
else
    ACTIVITY="(unable to fetch journalctl)"
fi

{
    echo "[$TIMESTAMP] A7 counter: $COUNTER"
    if [ -n "$ACTIVITY" ]; then
        echo "$ACTIVITY" | sed "s/^/[$TIMESTAMP]   /"
    fi
    echo ""
} >> "$LOG_FILE"

echo "[$TIMESTAMP] A7 counter: $COUNTER"
