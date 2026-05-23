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
#   0 - session is CLEAN (PASS) OR comparator returned VACUOUS (nothing to compare);
#       on PASS the counter is incremented (or unchanged if already counted today);
#       on VACUOUS the counter and marker are left untouched so a later real
#       session today can still count.
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

# Snapshot-date guard: the snapshot's decision timestamp (in UTC) must equal today's UTC date.
# Without this guard, on a day where RAMP doesn't fire (e.g. ramp.enabled=false,
# market holiday, regime SAFE_MODE), the helper would re-process the prior day's
# snapshot and increment the counter vacuously.
SNAPSHOT_DATE="$("$PYTHON" - "$LATEST_JSON" <<'PY' 2>/dev/null
import json, sys
from datetime import datetime, timezone
rec = json.load(open(sys.argv[1]))
ts = rec.get("timestamp") or rec.get("trigger", {}).get("actual_fire_time")
if not ts:
    sys.exit("no timestamp in snapshot")
dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
print(dt.astimezone(timezone.utc).strftime("%Y-%m-%d"))
PY
)"
SNAPSHOT_DATE="$(echo "$SNAPSHOT_DATE" | tr -d '[:space:]')"
if [[ -z "$SNAPSHOT_DATE" ]]; then
    echo "[ERROR] could not parse snapshot timestamp from $LATEST_JSON"
    write_gauges "$current_counter" 1
    exit 2
fi
if [[ "$SNAPSHOT_DATE" != "$TODAY" ]]; then
    echo "[STALE] snapshot date $SNAPSHOT_DATE != today $TODAY; no RAMP rebalance fired today; counter unchanged at ${current_counter}"
    # Do NOT write the marker -- if RAMP fires later today the helper should be able to re-run.
    write_gauges "$current_counter" 0
    exit 0
fi

echo "[CHECK] $LATEST_JSON"
# Module-style invocation so `from src.* import ...` inside the comparator resolves.
# Direct `$PYTHON $COMPARATOR ...` would put the script dir on sys.path instead
# of REPO_ROOT, causing ModuleNotFoundError on `import src.strategies...`.
LEDGER_JSON="${REPO_ROOT}/data/trading/decisions/_latest/ramp_position_state.json"
(cd "$REPO_ROOT" && "$PYTHON" -m scripts.trading.compare_paper_vs_plan \
    "$LATEST_JSON" \
    --position-ledger "$LEDGER_JSON" \
    --variant v11)
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
elif [[ "$RC" -eq 3 ]]; then
    # Comparator had nothing to compare (empty logic_decisions AND empty
    # strategy_inputs). Treat as a no-op: do NOT write the marker so a real
    # later session today can still count, and leave the counter unchanged.
    echo "[VACUOUS] no positions to compare; counter unchanged at ${current_counter}"
    write_gauges "$current_counter" 0
    exit 0
else
    echo "[ERROR] comparator returned unexpected exit code: $RC"
    write_gauges "$current_counter" 1
    exit 2
fi
