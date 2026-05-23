# RAMP V11 Production Paper Deploy Prep - 2026-05-23

## Summary

All local-side V11 production paper deploy work is committed and pushed to `origin/ramp-phase4-turnover-regime-research` at `2cb1b7c`. Remaining work: EC2 deploy (pull + restart + counter reset) and post-deploy verification on the first V11 rebalance (Mon 2026-05-25 15:55 ET).

This doc is the runbook for that EC2 deploy plus the post-deploy verification checklist. Convert this to a post-deploy session log once the steps are actually executed.

## What's queued (already on origin)

Six commits past the prior `d86d4c0` baseline, in order:

- `6685cda` feat(ramp): track position_open_dates in live adapter for V11 filters
- `ea46bac` feat(decision_log): write position_state.json alongside decision log
- `968d8cb` feat(config): add variant field to strategy_toggle.yaml for V01/V11 selection
- `6716c3c` feat(ramp): backport V11 rank_buffer + min_hold + delta_threshold into live adapter
- `b5f55b0` feat(comparator): extend compare_paper_vs_plan for V11 filter state
- `20d4d92` ops(a7): wire position-ledger and variant args into comparator invocation
- `2cb1b7c` config(toggle): enable RAMP V11 in production paper

40 new tests across the changeset. 921 tests pass under `tests/trading/`, 569 under `tests/research/ramp_phase4/`. No regressions vs the pre-2A baseline.

## EC2 deploy runbook (Phase 2G)

### 0. Pre-flight

EC2 instance may be auto-stopped outside market hours. Start it first:

```bash
# From Windows host (with infra/ec2/load_env.bat sourced):
infra\ec2\local_start_instance.bat
# Wait ~60s for SSH-ready state.
```

Or via aws CLI if instance ID is known.

### 1. SSH to the bot

```bash
ssh ec2  # or whatever alias is configured in ~/.ssh/config
```

Verify connection: `bot-status` should report current systemd state for `homeguard-multi`, `homeguard-cscm`, `homeguard-ramp-paper-check.timer`.

### 2. Pull and inspect

```bash
cd ~/Homeguard
git fetch origin
git log --oneline origin/ramp-phase4-turnover-regime-research..HEAD  # should be empty
git pull origin ramp-phase4-turnover-regime-research
git log --oneline -8  # confirm 6685cda..2cb1b7c are present
```

### 3. Verify the toggle yaml

```bash
cat config/trading/strategy_toggle.yaml
# Expected:
#   ramp:
#     enabled: true
#     shutdown_requested: false
#     variant: v11
```

If the yaml on disk diverges from origin (someone could have edited via state-manager API), reconcile to the committed version.

### 4. Restart the service

```bash
sudo systemctl restart homeguard-multi
sleep 5
sudo systemctl status homeguard-multi --no-pager | head -30
# Expect: active (running); recent log lines should show the
# new variant being honored: '[RAMP] variant=v11' (or equivalent).
```

### 5. Tail logs to verify clean boot

```bash
journalctl -u homeguard-multi --since "2 minutes ago" -f
# Watch for ImportError, AttributeError, or variant-related ValueError.
# Ctrl-C after a clean startup.
```

The runner is now idle until the next market-hours fire (Mon 15:55 ET).

### 6. Reset the A7 counter

```bash
sudo rm -f /var/lib/homeguard/a7_clean_sessions /var/lib/homeguard/a7_last_session_date
# Counter starts fresh at 0 against V11 decisions.
```

### 7. Verify the A7 timer is enabled

```bash
systemctl list-timers homeguard-ramp-paper-check.timer
# Expect: next firing at 2026-05-25 16:05 ET (Monday). Persistent: yes.
```

If the timer is disabled, enable per the runbook at `docs/progress/20260515_RAMP_PHASE4_DEPLOY_RUNBOOK.md:100-103`:

```bash
sudo systemctl enable --now homeguard-ramp-paper-check.timer
```

## Post-deploy verification (Mon 2026-05-25 after 15:55 ET)

### 1. Confirm the rebalance fired

```bash
journalctl -u homeguard-multi --since "15:50 ET today" | grep -E "rebalance|target|trade"
```

Look for evidence that V11 logic ran:
- `variant=v11` log line at session start
- `_apply_v11_filters` log entry (if the adapter emits one) showing rank_buffer / min_hold composition
- Position adjustments consistent with V11 (small turnover; protected names retained)

### 2. Verify the two snapshot files exist with today's UTC date

```bash
ls -la data/trading/decisions/_latest/ramp*.json
cat data/trading/decisions/_latest/ramp_position_state.json | python -m json.tool | head
```

Expected: both `ramp.json` and `ramp_position_state.json` present with timestamp ~16:00 ET today.

### 3. Manual comparator dry-run

```bash
cd ~/Homeguard
python -m scripts.trading.compare_paper_vs_plan \
    data/trading/decisions/_latest/ramp.json \
    --position-ledger data/trading/decisions/_latest/ramp_position_state.json \
    --variant v11
echo "Exit code: $?"
```

Expected exit 0 (PASS). Day 1 may exit 3 (VACUOUS) if positions weren't actually changed (start-from-cash situation). Day 1 may also exit 1 (FAIL) due to V01-vs-V11 weight discrepancy on the first rebalance after the variant flip; this is benign and the counter will not increment.

### 4. After the 16:05 ET timer fire

```bash
cat /var/lib/homeguard/a7_clean_sessions
# Expected: 1 (if PASS), 0 (if FAIL or VACUOUS-then-FAIL).
cat /var/lib/homeguard/a7_last_session_date
# Expected: 2026-05-25 (today's UTC date)
journalctl -u homeguard-ramp-paper-check --since "16:00 ET today"
# Look for [CLEAN] consecutive clean sessions: N/5 or [FAIL]/[VACUOUS] messages.
```

### 5. Grafana

`hg_a7_clean_sessions` gauge in the portfolio_overview dashboard (or scrape via VictoriaMetrics `:8428/api/v1/query?query=hg_a7_clean_sessions`) should now read 1 (or 0 on Day 1 if FAIL).

## Known risks

- **Day 1 may FAIL**: the position_state ledger starts empty on the new deploy. The first V11 rebalance after the variant flip will see an empty ledger -> V11 filters are no-ops -> V11 plan == V01 plan -> matches the live plan. But: the LIVE plan IS V11 by then (variant=v11 in toggle). Comparator and live runner are both reading the same empty ledger. Should agree. If the broker had pre-existing positions from earlier V01 sessions, those positions exist but lack open_dates -- the live runner will be forgiving (min_hold can't protect names without dates), and the comparator must do the same. This was tested in `test_recompute_plan_v11_with_empty_ledger_equals_v01`.
- **Comparator vs live divergence**: the byte-identical guarantee from Phase 2E (commit `b5f55b0`) covers normal operation but assumes both processes are reading the SAME version of the position_state.json. If the live runner writes the ledger AFTER its trades fill, but BEFORE the systemd timer fires the comparator, the comparator sees the post-trade state. This is the right ordering. Confirm by inspecting timestamps if FAILs accumulate.
- **Timer fires while market is open**: 16:05 ET is after the 15:55 ET rebalance. ~10 min margin. If trades have not filled by 16:00 ET (slow fills, IBKR latency), the position_state.json may be stale. Inspect log timestamps if intermittent FAILs.
- **Variant mismatch panic**: if `strategy_toggle.yaml` on EC2 says `variant: v01` but A7 helper passes `--variant v11`, the comparator will compute V11 plan, live runner ran V01 plan, divergence guaranteed. Mitigation: step 3 above (verify yaml). The Phase 2C / 2D code raises `ValueError` on unknown variants but does NOT cross-check that live runner and helper agree.

## What this prep doc does NOT do

- Actually execute the EC2 commands. That's the deploy itself; needs SSH access from a human-controlled session.
- Reset the A7 counter remotely. Same.
- Trigger a manual rebalance outside market hours. Could be done via the runner's CLI but adds noise; just wait for Monday.

## Next session

After the EC2 deploy completes:
1. Convert this prep doc into a post-deploy session log: rename to `20260523_RAMP_PHASE4_V11_PRODUCTION_PAPER.md` (or the actual deploy date), strip the runbook, add actual results.
2. Append a changelog entry to `docs/strategies/production/RAMP_STRATEGY.md` documenting V11 production paper start date, the PARTIAL significance caveat, the A7 gate (5 clean sessions), and "production live remains gated."
3. Begin Phase 3 monitoring: daily check on the A7 counter and Grafana gauge.
