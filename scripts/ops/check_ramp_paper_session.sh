#!/usr/bin/env bash
# RAMP Phase 4 A7 paper validation session check.
# Designed for EC2-resident execution via homeguard-ramp-paper-check.timer.
#
# Reads data/trading/decisions/_latest/ramp.json (the most recent RAMP
# decision snapshot, updated by the live runner after each rebalance),
# runs the paper validation comparator, and increments or resets the
# consecutive-clean-session counter at /var/lib/homeguard/a7_clean_sessions.
#
# Idempotent per day via /var/lib/homeguard/a7_last_session_date marker.
#
# Emits VM gauge hg_a7_clean_sessions and hg_a7_check_error via the
# node_exporter textfile collector (writes to
# /var/lib/node_exporter/textfile_collector/homeguard_a7.prom).
#
# Exit codes:
#   0 - session is CLEAN (PASS); counter incremented or unchanged (already today)
#   1 - session FAILED comparator; counter reset to 0
#   2 - setup error (decision log missing); counter unchanged, error gauge set

set -u

REPO_ROOT="${REPO_ROOT:-/home/ec2-user/Homeguard}"
PYTHON="${PYTHON:-${REPO_ROOT}/venv/bin/python}"
LATEST_JSON="${LATEST_JSON:-${REPO_ROOT}/data/trading/decisions/_latest/ramp.json}"
COMPARATOR="${REPO_ROOT}/scripts/trading/compare_paper_vs_plan.py"
COUNTER_FILE="${COUNTER_FILE:-/var/lib/homeguard/a7_clean_sessions}"
MARKER_FILE="${MARKER_FILE:-/var/lib/homeguard/a7_last_session_date}"
TEXTFILE_OUT="${TEXTFILE_OUT:-/var/lib/node_exporter/textfile_collector/homeguard_a7.prom}"
REQUIRED_CLEAN=5

# Ensure state directories exist.
mkdir -p "$(dirname "$COUNTER_FILE")"
mkdir -p "$(dirname "$TEXTFILE_OUT")"

write_gauges() {
    local clean_count="$1"
    local error_flag="$2"
    local tmp="${TEXTFILE_OUT}.tmp.$$"
    {
        echo "# HELP hg_a7_clean_sessions Consecutive clean RAMP paper sessions."
        echo "# TYPE hg_a7_clean_sessions gauge"
        echo "hg_a7_clean_sessions ${clean_count}"
        echo "# HELP hg_a7_check_error 1 if the most recent session check errored, 0 otherwise."
        echo "# TYPE hg_a7_check_error gauge"
        echo "hg_a7_check_error ${error_flag}"
    } > "$tmp"
    mv "$tmp" "$TEXTFILE_OUT"
}

read_counter() {
    if [[ -f "$COUNTER_FILE" ]]; then
        local v
        v="$(cat "$COUNTER_FILE" | tr -d '[:space:]')"
        if [[ "$v" =~ ^[0-9]+$ ]]; then
            echo "$v"
            return
        fi
    fi
    echo 0
}

current_counter="$(read_counter)"

# Idempotent-per-day guard.
TODAY="$(date -u +%Y-%m-%d)"
if [[ -f "$MARKER_FILE" ]]; then
    LAST_DATE="$(cat "$MARKER_FILE" | tr -d '[:space:]')"
    if [[ "$LAST_DATE" == "$TODAY" ]]; then
        echo "[SKIP] already processed today ($TODAY); counter unchanged at ${current_counter}"
        write_gauges "$current_counter" 0
        exit 0
    fi
fi

if [[ ! -f "$LATEST_JSON" ]]; then
    echo "[ERROR] decision log snapshot not found: $LATEST_JSON"
    write_gauges "$current_counter" 1
    exit 2
fi

echo "[CHECK] $LATEST_JSON"
"$PYTHON" "$COMPARATOR" "$LATEST_JSON"
RC=$?

if [[ "$RC" -eq 0 ]]; then
    new_counter=$((current_counter + 1))
    echo "$new_counter" > "$COUNTER_FILE"
    echo "$TODAY" > "$MARKER_FILE"
    write_gauges "$new_counter" 0
    echo "[CLEAN] consecutive clean sessions: ${new_counter}/${REQUIRED_CLEAN}"
    if [[ "$new_counter" -ge "$REQUIRED_CLEAN" ]]; then
        echo "[GATE PASSED] Ready for Task 14 production resume."
        echo "Run: bash scripts/ops/ramp_phase4_close_progress_doc.sh"
    fi
    exit 0
elif [[ "$RC" -eq 1 ]]; then
    echo "0" > "$COUNTER_FILE"
    echo "$TODAY" > "$MARKER_FILE"
    write_gauges 0 0
    echo "[FAIL] session diverged from plan; counter reset to 0"
    exit 1
else
    echo "[ERROR] comparator returned unexpected exit code: $RC"
    write_gauges "$current_counter" 1
    exit 2
fi
