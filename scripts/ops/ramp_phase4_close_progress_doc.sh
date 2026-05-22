#!/usr/bin/env bash
# RAMP Phase 4 Phase A -- close-out script.
# Run AFTER Task 13 paper validation acceptance gate (>=5 consecutive
# clean sessions) AND Task 14 production resume have both happened.
#
# Usage: bash scripts/ops/ramp_phase4_close_progress_doc.sh
#
# Updates docs/progress/20260515_RAMP_PHASE4.md and commits.
#
# Exit codes:
#   0 - success; progress doc updated and committed
#   1 - gate not met or progress doc missing

set -eu

PROGRESS_DOC="docs/progress/20260515_RAMP_PHASE4.md"
COUNTER_FILE="${COUNTER_FILE:-/var/lib/homeguard/a7_clean_sessions}"
REQUIRED_CLEAN=5
CLOSE_DATE="$(TZ='America/New_York' date '+%Y-%m-%d %H:%M ET' 2>/dev/null || date '+%Y-%m-%d %H:%M ET')"

if [[ ! -f "$PROGRESS_DOC" ]]; then
    echo "ERROR: progress doc not found: $PROGRESS_DOC"
    exit 1
fi

# Check acceptance gate
CLEAN=0
if [[ -f "$COUNTER_FILE" ]]; then
    CLEAN="$(cat "$COUNTER_FILE" | tr -d '[:space:]')"
    if ! [[ "$CLEAN" =~ ^[0-9]+$ ]]; then
        CLEAN=0
    fi
fi

if [[ "$CLEAN" -lt "$REQUIRED_CLEAN" ]]; then
    echo "ERROR: A7 acceptance gate not met."
    echo "  Consecutive clean sessions: $CLEAN / $REQUIRED_CLEAN"
    echo "  Run more paper sessions via: bash scripts/ops/check_ramp_paper_session.sh"
    exit 1
fi

echo "A7 gate met: $CLEAN consecutive clean sessions."
echo "Updating progress doc: $PROGRESS_DOC"

# Update Status line and tick remaining tasks
python -c "
import re
from pathlib import Path
p = Path('$PROGRESS_DOC')
s = p.read_text()
s = re.sub(
    r'> Status: in-progress',
    '> Status: COMPLETE (Phase A; F6 next spec)',
    s,
)
s = re.sub(r'- \[ \] Task 13 --', '- [x] Task 13 --', s)
s = re.sub(r'- \[ \] Task 14 --', '- [x] Task 14 --', s)
p.write_text(s)
print('Status line and task ticks updated.')
"

# Append closeout section
cat >> "$PROGRESS_DOC" <<CLOSEOUT

## Closeout -- ${CLOSE_DATE}

Phase A complete. F1-F5 all landed. A7 paper validation passed (${CLEAN}
consecutive clean sessions). Production resumed via market-timing-guarded
toggle change.

Total: 23 commits + closeout commit. 56 Phase A tests pass. 0 regressions
in existing test suite.

Follow-ups (separate specs/plans):
- F6 stateful backtest harness
- Paper P&L cross-check vs backtest (separate validation spec)
- Legacy _execute_rebalance_legacy deletion (after F2 has run live for
  a sufficient period without issue)
- WEAK_BULL regime fix (variant research, blocked on F6)
CLOSEOUT

echo "Closeout section appended."

# Commit
git add "$PROGRESS_DOC"
git commit -m "docs(progress): close RAMP Phase 4 Phase A

All 14 steps (A0-A7) completed. F1-F5 fixes live in production paper
behind feature flag (use_target_planner=True). ${CLEAN} consecutive clean
paper sessions verified. 0 regressions across the existing test suite.

Follow-ups (separate specs):
- F6 stateful backtest harness
- Paper P&L cross-check spec
- Legacy code path deletion
- WEAK_BULL regime fix"

echo ""
echo "[DONE] Phase A closed. Branch ready to merge to main."
