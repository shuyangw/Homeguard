# RAMP Phase 4 Phase A -- Deploy Runbook

> Companion to docs/progress/20260515_RAMP_PHASE4.md
> Branch: ramp-phase4-turnover-regime-research

## Pre-flight (already done in this branch)

- F1 planner, F2 target-aware execution, F3 parity tests, F4 safe mode,
  F5 decision log enrichment all landed and tested locally.
- scripts/trading/run_live_paper_trading.py now sets use_target_planner=True
  for the RAMPLiveAdapter constructor (line ~1128, commit ec4a01c).
- Local config/trading/strategy_toggle.yaml shows ramp.enabled: false
  (force-committed for audit; gitignored so this commit does NOT propagate
  to EC2 automatically -- see Step 1).
- All 56 Phase A tests pass. 0 regressions in existing test suite.
- Runner compiles cleanly (py_compile confirmed, commit 936d7e5 context).


## Step 1: Pause production paper on EC2

The strategy_toggle.yaml file is gitignored. Production reads the EC2-side
copy. The local force-committed version is an audit point only.

To pause production on EC2:

    ssh <ec2-host>
    cd ~/Homeguard
    # Edit config/trading/strategy_toggle.yaml
    # Set: ramp: enabled: false
    # Verify by reading the file back

Confirm in journald that the next scheduled rebalance does NOT fire
(wait until after the 15:55 ET rebalance window):

    journalctl -u homeguard-multi --since "5 minutes ago" | grep -i ramp

Expected: no "executing rebalance" lines for RAMP.

Record the pause time in docs/progress/20260515_RAMP_PHASE4.md.


## Step 2: Deploy branch to EC2

OPTION A -- deploy branch directly:

    ssh <ec2-host>
    cd ~/Homeguard
    git fetch origin
    git checkout ramp-phase4-turnover-regime-research
    # Restart the service (adjust command to your deploy mechanism)
    sudo systemctl restart homeguard-multi

OPTION B -- merge to main first, then deploy main:

    # Local:
    git checkout main
    git merge ramp-phase4-turnover-regime-research --no-ff
    git push origin main
    # On EC2:
    ssh <ec2-host>
    cd ~/Homeguard
    git pull origin main
    sudo systemctl restart homeguard-multi

After restart, verify the new code is active:

    journalctl -u homeguard-multi -n 100 | grep -i "target.planner"

Expected line (logged by RAMPLiveAdapter.__init__):
    [RAMP] use_target_planner=True

If this line is absent, the old code is still running. Check that the
service was fully restarted and that Python is picking up the right files.


## Step 3: Re-enable RAMP paper (NOT production-live yet)

After confirming the new code is active (Step 2 verification), re-enable
RAMP in the toggle file ON EC2:

    # On EC2 only -- do NOT change the local file
    # Edit config/trading/strategy_toggle.yaml
    # Set: ramp: enabled: true

The runner uses use_target_planner=True so this now runs the new code path.
The service does not need a restart -- the toggle is read at each rebalance.

Confirm at the next rebalance window (15:55 ET) that the log shows:
    [RAMP] computing plan via target planner
(or equivalent log from _execute_rebalance_target_aware)


## Step 4: Run >=5 consecutive clean paper sessions

After each daily paper rebalance, run locally (or on EC2 with PYTHON set):

    bash scripts/ops/check_ramp_paper_session.sh

This script:
1. Reads data/trading/decisions/_latest/ramp.json (most recent rebalance).
2. Runs scripts/trading/compare_paper_vs_plan.py against it.
3. Prints the comparator status (PASS / FAIL + any divergences).
4. Tracks the consecutive-clean count in docs/progress/.a7_clean_sessions.

To override the JSON path (e.g. to check a specific date):

    LATEST_JSON=data/trading/decisions/ramp_20260516.jsonl bash scripts/ops/check_ramp_paper_session.sh

NOTE: if pointing at a multi-line JSONL file, extract one line first:
    tail -1 data/trading/decisions/ramp_20260516.jsonl > /tmp/ramp_latest.json
    LATEST_JSON=/tmp/ramp_latest.json bash scripts/ops/check_ramp_paper_session.sh

Possible outcomes:

  [CLEAN] -- count increments. After 5, the gate is passed.
  [GATE PASSED] -- printed when count >= 5. Proceed to Step 5.

  [FAILED] -- count resets to 0. Investigate:
    1. Check the divergence lines printed by the comparator.
    2. Look at the decision log for the failing session.
    3. Check adapter logs for exceptions or fallback-mode triggers.
    4. If a bug is found: fix, commit, redeploy, restart the 5-session
       count from 0.


## Step 5: Task 14 -- production resume (after 5 clean sessions)

After docs/progress/.a7_clean_sessions reads >= 5:

1. Market timing guard (CRITICAL):
       TZ='America/New_York' date '+%H:%M'
   If the output is between 15:42 and 16:00, WAIT until after 16:00.
   Never change the toggle during the rebalance window.

2. RAMP is already re-enabled for paper (Step 3). If paper and production
   share the same EC2 toggle file, production is already live. If they
   are separate deployments, enable production-side toggle separately.

3. Run the closeout script to finalize the progress doc:

       bash scripts/ops/ramp_phase4_close_progress_doc.sh

   This script checks that the 5-session gate is met before making any
   changes. If the gate is not met it exits 1 without touching anything.

4. Push the branch (or merge to main) and close out the spec.


## Reference paths

- Decision log root:   data/trading/decisions/
- Latest snapshot:     data/trading/decisions/_latest/ramp.json
- Clean session count: docs/progress/.a7_clean_sessions
- Comparator:         scripts/trading/compare_paper_vs_plan.py
- Session check:      scripts/ops/check_ramp_paper_session.sh
- Closeout script:    scripts/ops/ramp_phase4_close_progress_doc.sh
- Progress doc:       docs/progress/20260515_RAMP_PHASE4.md
- Toggle file (EC2):  config/trading/strategy_toggle.yaml
- Service unit:       homeguard-multi (systemd, EC2)
