# RAMP Flat Equity Curve - Root Cause + Fix - 2026-06-03

## Summary
RAMP's Grafana equity curve was frozen at exactly $98,702.94 for 6 days while the
strategy was actually holding 33 IBKR paper positions (~$95k) and rebalancing
correctly. Root cause: the strategy state file's `positions` dict was empty, so
`_compute_strategy_equity` fell back to the constant `initial_capital +
lifetime_realized`. Added `adopt_broker_positions` to the state manager and wired
it into the RAMP rebalance so broker-held-but-untracked positions are backfilled
into state, healing the equity gauge and the ownership/realized-PnL accounting.

## Root Cause (verified in code + on live system)
- Equity gauge `hg_strategy_equity_usd{strategy="ramp"}` = $98,702.94, with
  VictoriaMetrics `distinct_vals = 1` over 6 days (literally one value).
- Decomposes as `100,000 (initial) + (-1,297.06 lifetime_realized) + 0 unrealized`.
- `_compute_strategy_equity` (`scripts/trading/run_live_paper_trading.py:629-631`)
  reads `state_manager.get_positions("ramp")`; it returned `{}` -> `owned_symbols`
  empty -> returns the constant.
- The state file had `strategies.ramp.positions = {}` but `position_open_dates`
  with all 33 symbols. The `positions` dict is only written on a net BUY fill
  (`ramp_live_adapter.py:2019`, guarded by `shares_to_buy > 0`). On boot, RAMP
  rebuilds `current_positions` from the broker for sizing, sees positions already
  at target (STRONG_BULL 100%), `continue`s, and never calls
  `add_or_update_position`. `sync_with_broker` only reconciles symbols already in
  the state dict (`strategy_state_manager.py:761`), so it cannot backfill. Once
  desynced, state stays empty permanently.
- Secondary (not the cause): IBKR paper throws error 10167 (delayed market data)
  every ~60-90s; unrealized PnL is computed from 15-min-delayed quotes. Cosmetic.

## Impact beyond the chart
The empty `positions` dict also drives `symbol_owned_by_other` and the rebalance
ownership checks, so RAMP believed it owned nothing (another strategy could grab
its symbols) and realized-PnL-on-exit used broker avg_entry_price fallbacks
instead of true tagged entries.

## Changes Made
- **`src/trading/state/strategy_state_manager.py`**: new `adopt_broker_positions(
  strategy, broker_name, broker_positions)` -> List[str]. Adds broker-held
  symbols that no strategy tracks into the strategy's `positions` dict, tagged
  with the broker and using broker `avg_entry_price`. Conservative: skips
  already-tracked, owned-by-other, and zero-qty. Ownership check inlined to avoid
  the `_load_state` reload that would orphan the in-memory positions reference.
- **`src/trading/adapters/ramp_live_adapter.py`**: call `adopt_broker_positions`
  in the rebalance precondition stage, immediately before the existing
  `sync_with_broker`, passing full broker position dicts (with avg_entry_price).
- **`tests/trading/test_state_manager_adopt.py`** (new): 6 TDD tests.

## Commits (branch `fix/ramp-equity-position-adoption`, off `e25e5a5`)
- `779bc6d` fix(state): add adopt_broker_positions to heal empty positions-dict desync
- `4f2bb69` fix(ramp): adopt untracked broker positions before sync on each rebalance

## Tests
TDD, RED-verified before implementation. 6 new tests in
`test_state_manager_adopt.py`:
- adopt untracked broker position
- skip symbol already tracked by strategy
- skip symbol owned by other strategy
- skip zero quantity
- adopted position is reconcilable by sync_with_broker
- multi-adopt returns all
First RED failure on a stale-reference bug (calling `symbol_owned_by_other`
reloaded state and orphaned the write); fixed by inlining the ownership check.

## Validation
- `tests/trading/test_state_manager_adopt.py` + `test_state_manager_broker_aware.py`
  + `test_state_manager_migration.py` + `tests/trading/adapters/`: **53 passed**.
- Pre-existing (NOT caused by this change): 2 failures in
  `tests/trading/test_ramp_decision_log.py` (`test_run_once_writes_record_on_exception`,
  `test_run_once_records_blocked_when_health_check_fails`) -- confirmed identical
  failures on stashed/unmodified code; a mocked health-checker path returning None.
- IBKR paper smoke test NOT run (requires EC2/IBKR connection); this change does
  not touch the broker order-submission path, only state bookkeeping.

## Known Issues / Remaining Work
- **Deploy**: branch `fix/ramp-equity-position-adoption` is committed in worktree
  `.claude/worktrees/ramp-equity-fix`, based off deployed archive HEAD `e25e5a5`.
  NOT pushed, NOT merged, NOT deployed to EC2. Needs user review + deploy decision.
- **Live state already desynced**: the fix self-heals on the NEXT rebalance after
  deploy (adoption runs every rebalance). No manual state edit required, but the
  equity gauge stays flat until the first post-deploy rebalance (3:55 PM ET market
  day) writes the adopted positions.
- The same desync class affects OMR/MP adapters if they ever boot holding
  untracked broker positions; only RAMP was wired here. Consider wiring the other
  adapters if the pattern recurs.
- Branch lineage note: `origin/main` (5afd6d5) has diverged from the deployed
  archive branch (e25e5a5 / now 9745b26); the campaign + V11 code lives only on
  the archive branch. This fix is based on the archive branch (deployed code).
