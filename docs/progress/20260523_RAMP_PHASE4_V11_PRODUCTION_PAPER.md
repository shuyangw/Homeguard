# RAMP V11 Production Paper Deploy - 2026-05-23

## Summary

V11 (Phase 4 Wave 1 combined turnover-lite variant: rank_buffer + min_hold + delta_threshold) is now live in production paper trading on EC2. Deploy completed Sat 2026-05-23 04:30 UTC (Sat 00:30 ET). First V11 rebalance: Mon 2026-05-25 15:55 ET. A7 paper-validation gate begins at the next A7 helper fire (Mon 2026-05-25 20:05 UTC = 16:05 ET).

V11 readiness was PARTIAL: passes PBO (0.126) and one-day-lag robustness (+9.79%), fails strict PSR (0.944) and DSR (0.811) under per-period BLdP application. Paper trading itself is the OOS validation channel; the A7 gate (5 clean sessions) provides the go/no-go signal.

## Changes deployed (commits 6685cda..6f6d0d6)

Eight commits past the pre-deploy EC2 HEAD (`1548574`):

- `6685cda` feat(ramp): track position_open_dates in live adapter for V11 filters (+9 tests)
- `ea46bac` feat(decision_log): write position_state.json alongside decision log (+5 tests)
- `968d8cb` feat(config): add variant field to strategy_toggle.yaml for V01/V11 selection (+6 tests)
- `6716c3c` feat(ramp): backport V11 rank_buffer + min_hold + delta_threshold into live adapter (+10 tests)
- `b5f55b0` feat(comparator): extend compare_paper_vs_plan for V11 filter state (+4 tests)
- `20d4d92` ops(a7): wire position-ledger and variant args into comparator invocation
- `2cb1b7c` config(toggle): enable RAMP V11 in production paper
- `6f6d0d6` docs(progress): V11 production paper deploy prep + EC2 runbook

Total: 34 new tests, 0 regressions, 921 tests pass under `tests/trading/`.

## Deploy execution log

| Time (UTC) | Step | Result |
|---|---|---|
| 04:25 | EC2 was stopped (off-hours) | started via `aws ec2 start-instances` |
| 04:27 | Wait for instance-status-ok | SSH-ready, same elastic IP |
| 04:28 | git fetch + diff against origin | EC2 was at `1548574`; 15 commits behind. Local diff: `ramp.enabled: true` only (now superseded by `2cb1b7c`). |
| 04:28 | `git reset --hard origin/ramp-phase4-turnover-regime-research` | clean to `6f6d0d6` |
| 04:28 | Verify `strategy_toggle.yaml` | `ramp: enabled: true, variant: v11` |
| 04:28 | `sudo systemctl restart homeguard-multi` | active (running), MainPID 2603 |
| 04:29 | Log inspection for V11 activation | `RAMP variant: v11` confirmed in two log lines |
| 04:29 | A7 counter pre-reset state | counter file `0`, marker file missing |
| 04:29 | `sudo rm -f /var/lib/homeguard/a7_*` | clean |
| 04:29 | A7 timer verification | enabled; next fire Mon 2026-05-25 20:05 UTC |

## V11 activation confirmation (from EC2 logs)

```
May 23 04:28:39 ... homeguard-multi[2603]:    RAMP variant: v11
May 23 04:28:39 ... homeguard-multi[2603]:  [RAMP]   Variant: v11
```

Non-blocking IBKR warnings (paper account, expected): `code=10167: Requested market data is not subscribed. Displaying delayed market data.` These are routine for paper data subscriptions; not a deploy issue.

## Post-deploy verification (Mon 2026-05-25 after 15:55 ET)

After the first V11 rebalance:

1. **Both snapshot files exist** at `data/trading/decisions/_latest/`:
   - `ramp.json` (existing decision log)
   - `ramp_position_state.json` (new V11 ledger; the file written by Phase 2B)

2. **Manual comparator dry-run** with exit code 0:
   ```bash
   ssh ec2-user@$EC2_IP 'cd ~/Homeguard && python -m scripts.trading.compare_paper_vs_plan \
       data/trading/decisions/_latest/ramp.json \
       --position-ledger data/trading/decisions/_latest/ramp_position_state.json \
       --variant v11'
   ```

3. **A7 counter increments** at 16:05 ET. Should read `1` after the first clean session:
   ```bash
   cat /var/lib/homeguard/a7_clean_sessions
   ```

4. **Grafana** `hg_a7_clean_sessions` gauge updates within the scrape interval.

## Known Day-1 risks

- The position_state.json ledger starts empty after the deploy. The first V11 rebalance will execute with empty `_position_open_dates`, so `min_hold` will not protect any name (no positions have open_dates yet). `rank_buffer` will be a no-op for the same reason (`state.positions` is empty). V11 plan == V01 plan on Day 1. This is expected.
- If broker reports pre-existing positions from earlier V01 sessions (~10 SP500 names), those will be in `state.positions` but their `position_open_dates` will be empty (no entry). `min_hold` skips names that aren't in `position_open_dates`. The V11 filters will see them as "held with unknown open_date" and treat them as not protected -- they'll be eligible to drop. Same for `rank_buffer`: held names without rank info in the buffer test are also not retained. So Day 1 will likely produce a rebalance that liquidates carry-over positions, then opens V11-selected names with fresh open_dates.
- Comparator divergence is possible Day 1 due to ledger empty / live carrying old positions, but per `test_recompute_plan_v11_with_empty_ledger_equals_v01` (Phase 2E test), the comparator falls back to V01 logic in this case. Both sides should agree. If they don't, investigate which side has the inconsistency.

## Phase 3: monitoring window

Until A7 counter reaches `REQUIRED_CLEAN=5`:

- Daily: `journalctl -u homeguard-ramp-paper-check --since yesterday` for the A7 helper's PASS/FAIL log.
- Daily: read `/var/lib/homeguard/a7_clean_sessions` for the counter.
- Grafana: dashboard panel on `hg_a7_clean_sessions`.

Failure modes to watch:
- Counter resets to 0 -> investigate via `trade-log-analyzer` agent.
- Comparator over-strict -> may need a tolerance bump in Phase 2E.
- Position-state-ledger desync (rare; would mean state_manager and on-disk file diverged).

Estimated calendar to gate: 5-10 trading days assuming no FAILs.

## Branch state

Local at `6f6d0d6`. EC2 at `6f6d0d6` (matched to origin via hard reset). Branch `ramp-phase4-turnover-regime-research`. Not merged to `main` (intentional; stays open through paper validation per the methodology).

## What this deploy does NOT do

- Push V11 to production live (Phase 4 gate after counter clears).
- Brainstorm Wave 2 V12 (BEAR-to-cash on V11 base; deferred).
- Merge to `main` (after Phase 4 decision).
- Stop the EC2 instance. It is currently RUNNING. EventBridge schedule (if configured) will manage the auto stop/start cycle. If the bot should idle over the weekend, manual `infra\ec2\local_stop_instance.bat` works, but the systemd timer would not fire on Monday if the instance is stopped at the wrong moment.

## Validation

Local pytest pre-push: 921 / 921 passing in `tests/trading/`, 569 in `tests/research/ramp_phase4/`, 16 in `tests/backtesting/statistics/`, 8 in `tests/backtesting/validation/`.

EC2 service post-restart: active, V11 variant confirmed in logs, no Python tracebacks, IBKR connected (paper port 4002), WebSocket subscribed to 504 symbols.
