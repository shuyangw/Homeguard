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


## Step 4: Set up the EC2-resident A7 check

After the branch is deployed and paper trading is active, install the systemd timer
that runs the per-session check automatically on EC2.

On EC2:

    sudo cp /home/ec2-user/Homeguard/infra/ec2/services/homeguard-ramp-paper-check.service /etc/systemd/system/
    sudo cp /home/ec2-user/Homeguard/infra/ec2/services/homeguard-ramp-paper-check.timer   /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now homeguard-ramp-paper-check.timer

Verify:

    systemctl list-timers homeguard-ramp-paper-check.timer
    cat /var/lib/homeguard/a7_clean_sessions     # initially nonexistent
    sudo journalctl -u homeguard-ramp-paper-check --since "1h ago"

After the next 16:05 ET trigger:

    cat /var/lib/homeguard/a7_clean_sessions     # 1 if clean, 0 if failed
    cat /var/lib/homeguard/a7_last_session_date  # today's UTC date

Grafana dashboard: the gauge `hg_a7_clean_sessions` appears automatically once
VM scrapes node_exporter's textfile output. Optionally add a single-stat panel
to portfolio_overview.json.


## Step 4a: "Clean session" semantics

| Case | Counter behavior |
|---|---|
| Comparator PASS (positions match plan within rounding) | increment by 1 |
| Comparator FAIL (positions diverge from plan) | reset to 0 |
| Comparator setup error (decision log missing) | unchanged; `hg_a7_check_error=1` |
| Regime returned SAFE_MODE / no rebalance fired today | unchanged (no decision log update for today) |
| Market holiday or EC2 was off when timer would have fired | unchanged |
| Multiple triggers in one day (defensive) | only the first counts; subsequent runs see the marker file and skip |


## Step 5: Production resume gate + rollback

Production resumes ONLY when ALL four pre-conditions hold:

1. `cat /var/lib/homeguard/a7_clean_sessions` returns >= 5.
2. Current UTC time is OUTSIDE the [15:42, 16:00] ET market guard window
   (see scripts/ops/ramp_phase4_close_progress_doc.sh for the canonical check).
3. EC2's config/trading/strategy_toggle.yaml currently has `ramp.enabled: false`
   (verifies the pause was in fact active).
4. Runner is using `use_target_planner=True`:

       sudo journalctl -u homeguard-multi -n 200 | grep "use_target_planner=True"

When all four pass, run on EC2:

    cd /home/ec2-user/Homeguard
    bash scripts/ops/ramp_phase4_close_progress_doc.sh

That script:
- toggles `ramp.enabled: true` in the EC2 yaml,
- updates docs/progress/20260515_RAMP_PHASE4.md status to COMPLETE,
- creates the closing commit on the ramp branch.


### Rollback paths

| Failure mode | Action |
|---|---|
| Resume produces zero orders for two consecutive sessions | toggle `ramp.enabled: false`; tail journal `homeguard-multi`; investigate why no signals fire |
| Resume produces orders that diverge from planner output (live comparator FAIL) | toggle `ramp.enabled: false`; manually flatten any opened real positions via IBKR UI; investigate decision logs |
| `use_target_planner` regression (logged as `False` after deploy) | revert to last known-good commit; redeploy; `sudo systemctl restart homeguard-multi` |
| A7 helper writes corrupted counter | `echo 0 | sudo tee /var/lib/homeguard/a7_clean_sessions`; restart the timer; resume validation from session 1 |


## Reference paths

- Decision log root:   data/trading/decisions/
- Latest snapshot:     data/trading/decisions/_latest/ramp.json
- Clean session count: /var/lib/homeguard/a7_clean_sessions (EC2)
- Last session date:   /var/lib/homeguard/a7_last_session_date (EC2)
- Comparator:         scripts/trading/compare_paper_vs_plan.py
- Session check:      scripts/ops/check_ramp_paper_session.sh (invoked by systemd timer on EC2)
- Timer unit:         infra/ec2/services/homeguard-ramp-paper-check.timer
- Service unit:       infra/ec2/services/homeguard-ramp-paper-check.service
- Closeout script:    scripts/ops/ramp_phase4_close_progress_doc.sh
- Progress doc:       docs/progress/20260515_RAMP_PHASE4.md
- Toggle file (EC2):  config/trading/strategy_toggle.yaml
- Service unit:       homeguard-multi (systemd, EC2)
