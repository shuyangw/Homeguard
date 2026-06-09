# RAMP Preflight Crash-Loop (IBKR cold-start race) - 2026-06-09

## Summary
RAMP's equity curve went flat again -- but this time because `homeguard-multi`
crash-looped ~1000x starting at the 12:00 UTC daily boot, on a preflight false
zero. NOT a regression of the 6/3 fixes (equity tracked fine 6/3->6/8; sell-only
pruned 33 -> ~21). Root cause: `ib.portfolio()` is empty until IBKR fires
`accountDownloadEnd`; the connection layer used a blind `sleep(1.0)` that races on
a cold start. Positions were never lost (verified: $1.02M / 20 positions).

## Root cause chain (why it surfaced now)
- The 6/3 `adopt_broker_positions` fix made state authoritative -> preflight now
  actually QUERIES the broker (before, empty state short-circuited it).
- `preflight_reconcile` calls `broker.get_stock_positions()` -> `ib.portfolio()`.
- On the 6/9 12:00 UTC cold boot, data farms were still warming (`2157`/`2107`
  inactive); the account download took >1.0s, so `portfolio()` returned empty.
- Preflight saw "state says 11 on ibkr, broker reports 0" -> mismatch -> exit(1).
- systemd restarted every 30s; each cycle reconnected clientId=10 and died in
  ~1s, never long enough to finish the download -> self-perpetuating loop (~1000x).
- Independent clientId=99 saw all 20 positions concurrently -> data present, just
  not delivered to the fast-cycling session. (Matches the long-standing
  "known issue with IBKRBroker.get_positions()".)

## Immediate mitigation (applied)
Reversible systemd drop-in adds `--force-start` to break the loop and restore
metrics. Path on EC2: `/etc/systemd/system/homeguard-multi.service.d/override.conf`.
**MUST be removed after this fix deploys** (`rm` the file, daemon-reload, restart).

## Changes Made
- **`src/trading/brokers/ibkr/connection.py`** (fix #1, root cause): replace the
  fixed `sleep(1.0)` after `reqAccountUpdates` with `reqAccountUpdatesAsync(acct)`
  wrapped in a 10s `wait_for` -- a coroutine that subscribes AND awaits
  `accountDownloadEnd`, so `portfolio()` is populated before callers use it.
  Timeout/error falls back to the low-level request + short settle.
- **`scripts/trading/run_live_paper_trading.py`** (fix #2, defense-in-depth):
  `preflight_reconcile` retries `get_stock_positions()` up to `max_attempts`
  (`retry_delay` apart) when state expects positions on THIS broker but the book
  is empty. No retry for cross-broker (expected 0); still blocks after retries on
  a genuine loss.
- **`tests/trading/test_run_live_paper_trading_preflight.py`**: +3 tests, 1 updated.

## Commits (branch `fix/ramp-equity-position-adoption`)
- `6a8367e` fix(ibkr): await accountDownloadEnd on connect instead of fixed sleep(1.0)
- `e21d3a8` fix(runner): retry broker query in preflight before trusting an empty book

## Validation
- `tests/trading/brokers/ibkr` + `test_run_live_paper_trading_preflight.py`:
  134 passed, 5 skipped + 13 preflight passed. New preflight tests RED-verified first.
- **IBKR paper smoke test REQUIRED on EC2** (connection-layer change) before
  dropping force-start -- pending deploy.
- True end-to-end proof: after deploy, REMOVE the force-start override and restart;
  preflight must pass on its own (clean start, no loop).

## ACTUAL ROOT CAUSE (found after 4 failed timing fixes)
The timing fixes (sleep -> accountDownloadEnd -> portfolio poll -> reqPositions
fallback) ALL failed clean-start validation. A definitive read diagnostic
(clientId=97, after-hours) settled it:
- `portfolio()` AND `positions()` both reliably return the SAME 20 symbols.
- State has 21 ibkr-tagged symbols; 20 match the broker exactly.
- The ONLY mismatch is **`WELL`**: state says open on ibkr (qty>0), broker holds 0.

So it was NOT a read/timing problem -- the read is correct. `WELL` is a genuine
**stale state entry**. Root cause: `_execute_rebalance_target_aware` (the
deployed exit path, use_target_planner=True) sold via `execute_order` but never
called `remove_position` or `log_exit` (the legacy path does). Every full exit
left a phantom, swept only by the NEXT rebalance's `sync_with_broker`. WELL was
sold 6/8; the 6/9 boot crash-looped at preflight before the 6/9 rebalance could
sweep it -> WELL lingered and blocked preflight.

Secondary: the missing `log_exit` means target-aware exits were NOT written to
the trade log, so `compute_lifetime_realized_pnl` (which feeds the equity gauge)
has been understating realized PnL for every target-aware exit.

### Fix (root cause)
- **`src/trading/adapters/ramp_live_adapter.py`**: target-aware exit loop now
  calls `log_exit` + `remove_position` on a filled exit (mirrors legacy). Commit
  `f7ea22f`. TDD: `test_target_aware_exit_removes_position_from_state`.
- **One-time reconcile**: remove the existing stale `WELL` entry from state
  (backed up) during the deploy stop-window.
- The earlier timing improvements (poll `portfolio()` `23abaaa`, reqPositions
  fallback `876bd44`) are harmless and retained as robustness, but were NOT the cure.

## Known Issues / Remaining Work
- Drop the `--force-start` override after deploy + smoke test (see path above).
- **Why was WELL's exit not removed for 6 days?** The target-aware path never
  removed exits from state; relied on next-rebalance sync. Now fixed at source.
  Audit whether other target-aware exits (6/4-6/8 pruning) left trade-log gaps
  that understated realized PnL.
- **Operational gap**: ~1000 silent restarts with no alert. Add a systemd
  StartLimitIntervalSec/Burst cap and/or an alert so a crash-loop pages instead of
  spinning. (Deferred -- flagged to user.)
- Open question: is `max_positions=25` right for V11 (rank_buffer can hold ~30)?
  (carried over from the 6/3 sell-only fix.)
