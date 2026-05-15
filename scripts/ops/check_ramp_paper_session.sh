#!/usr/bin/env bash
# RAMP Phase 4 A7 paper validation session check.
# Run after each daily paper rebalance.
#
# Usage: bash scripts/ops/check_ramp_paper_session.sh
#
# Reads the most recent decision log entry for RAMP from the _latest/
# snapshot (data/trading/decisions/_latest/ramp.json), runs the paper
# validation comparator, and increments or resets the consecutive-clean-
# session counter.
#
# The comparator (scripts/trading/compare_paper_vs_plan.py) expects a
# single JSON record as the file content -- the _latest/ snapshot is
# exactly that format.
#
# Counter file: docs/progress/.a7_clean_sessions
#
# Exit codes:
#   0 - session is CLEAN (PASS); counter incremented
#   1 - session FAILED comparator; counter reset to 0
#   2 - no decision log entry found / setup error

set -u

PYTHON="${PYTHON:-/c/Users/qwqw1/anaconda3/envs/fintech/python.exe}"
LATEST_JSON="${LATEST_JSON:-data/trading/decisions/_latest/ramp.json}"
COUNTER_FILE="docs/progress/.a7_clean_sessions"
COMPARATOR="scripts/trading/compare_paper_vs_plan.py"
REQUIRED_CLEAN=5

if [[ ! -f "$LATEST_JSON" ]]; then
    echo "ERROR: latest decision log snapshot not found: $LATEST_JSON"
    echo "Either no rebalance has run yet, or set LATEST_JSON to override."
    echo "You can also point LATEST_JSON at a specific JSONL line extracted"
    echo "from data/trading/decisions/ramp_YYYYMMDD.jsonl."
    exit 2
fi

echo "Checking: $LATEST_JSON"
"$PYTHON" "$COMPARATOR" "$LATEST_JSON"
RC=$?

# Read current counter (default 0 if file absent or empty)
CURRENT=0
if [[ -f "$COUNTER_FILE" ]]; then
    CURRENT="$(cat "$COUNTER_FILE" | tr -d '[:space:]')"
    # Guard against non-numeric content
    if ! [[ "$CURRENT" =~ ^[0-9]+$ ]]; then
        CURRENT=0
    fi
fi

if [[ "$RC" -eq 0 ]]; then
    NEW=$((CURRENT + 1))
    echo "$NEW" > "$COUNTER_FILE"
    echo ""
    echo "[CLEAN] Consecutive clean sessions: $NEW / $REQUIRED_CLEAN"
    if [[ "$NEW" -ge "$REQUIRED_CLEAN" ]]; then
        echo "[GATE PASSED] Ready for Task 14 (production resume)."
        echo "Run: bash scripts/ops/ramp_phase4_close_progress_doc.sh"
    fi
    exit 0
else
    echo "0" > "$COUNTER_FILE"
    echo ""
    echo "[FAILED] Consecutive clean sessions counter RESET to 0."
    echo "Investigate divergence, fix, then redeploy and resume validation."
    exit 1
fi
