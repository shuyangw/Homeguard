# CSCM Position Sizing Fix - 2026-04-26

## Summary

Diagnosed and fixed the 2026-04-26 00:19 UTC CSCM rebalance failure (`decision_id 80acb1ac`) where 6 of 7 target fills succeeded but the 7th (SUSHI/USD) failed with `Insufficient funds: need $13,790.39, have $13,693.89`. Root cause: each fill consumed ~12.5 bps more cash than planned (slippage + fees), and the rebalance loop sized all 7 positions from a single `total_value` snapshot without re-reading remaining cash. Fix: sequential cash tracking + per-fill safety factor in `cscm_live_adapter.rebalance()`.

This bug was first surfaced by the 2026-04-24 decision-log integration (commit `05d272c`) — which captured the full execution arc and made the per-symbol failure visible in a single `python -m src.trading.decision_log show cscm` lookup. Without that observability work, the silent drop of the lowest-momentum symbol would likely have continued unnoticed.

## Changes Made

- **`src/trading/adapters/cscm_live_adapter.py`** — refactored `rebalance()` from a single-pass close+buy+sell+hold loop into 4 phases:
  1. Close positions not in target (unchanged from prior behavior)
  2. Classify each target into hold/sell-down/buy lists using the initial price snapshot + 5% adjustment threshold
  3. Execute sell-downs first (frees additional cash)
  4. Execute buys with sequential cash tracking — each BUY re-reads `broker.get_account()['cash']` and divides by `remaining_buys` so cumulative drift is absorbed by smaller subsequent targets, never starving the final symbol
  - Added `DEFAULT_BUY_SAFETY_FACTOR = Decimal('0.998')` module constant (20 bps headroom; 12.5 bps DemoBroker drag + ~7 bps margin for randomization variance)
  - Added `buy_safety_factor` init param to `CSCMLiveAdapter.__init__`
  - Tracking decremented in `finally` clause to handle errors and skips uniformly

- **`config/trading/cscm_live.yaml`** — added `buy_safety_factor: 0.998` with comment explaining tradeoffs (0.998 default, 0.995 for higher-volatility brokers)

- **`scripts/trading/run_cscm_live.py`** — config plumbing: read `buy_safety_factor` from YAML, pass to `CSCMLiveAdapter`

- **`tests/trading/test_cscm_position_sizing.py`** (new file, 181 lines) — three regression tests using real `DemoBroker` with deterministic worst-case slippage (`randomize_slippage=False`) so they exercise the actual bug scenario:
  - `test_rebalance_all_targets_fill_under_slippage_and_fees` — 7 equal-weight buys all fill, no `InsufficientFundsError`, ending cash >= 0
  - `test_rebalance_skips_below_threshold_for_existing_positions` — 5% threshold preserved for symbols already at target weight
  - `test_rebalance_uses_safety_factor_to_size_under_cash` — first BUY notional ~= `cash * safety_factor / N`, proving the factor is applied

## Commits

- `bc52252` fix(cscm): sequential cash tracking + safety factor in rebalance buy phase

## Why other adapters (RAMP/MP) don't have this bug

- They trade whole-share equity with `int(target_value / price)` floor rounding → built-in under-sizing buffer
- They size against `ask` (the broker's fill price) → no slippage gap
- Equity commissions are zero on these brokers

CSCM's combination — fractional crypto qty (no floor buffer) + non-zero fees + sized-against-`last`-not-ask + multiple symbols sized from a single snapshot — is unique among current strategies.

## Known Issues / Remaining Work

- **Sat 2026-05-02 ~8 PM ET / Sun 2026-05-03 00:00 UTC** is the next live CSCM rebalance — the real proof point. After it fires, run `python -m src.trading.decision_log show cscm` and verify all N executions show `status: filled` (no `error`, no `skipped`).
- **Cosmetic decision-log issue**: `Positions: 0 CSCM-tagged` shown in `cscm` decision-log render even when fills succeed. The post-state snapshot queries `state_manager.get_positions('cscm')` but CSCM doesn't write to `StrategyStateManager` — it goes through `DemoBroker` directly. Cosmetic only; fix in a separate change.
- **"Current Positions" panel placement**: aggregated cross-strategy panel currently lives in `strategy_breakdown.json` but logically belongs in `portfolio_overview.json`. Flagged during this session; defer to a separate Grafana cleanup change.
- **No generalization to a shared utility yet**: RAMP/MP don't have this bug today, so a CSCM-only fix is right scope. If a future strategy trades fractional shares with non-zero commissions, extract the sizing helper then.
- **Min-notional handling not implemented**: if `available_cash * safety_factor / N` falls below a symbol's min-notional on Coinbase/Alpaca, current behavior records the broker's error rather than pre-filtering. Acceptable for v1.

## Validation

- **Local pytest**: `tests/trading/test_cscm_position_sizing.py` 3/3 pass; `tests/trading/ -k cscm` 21/21 pass; `tests/trading/decision_log/` 38/38 pass
- **EC2 pytest**: `tests/trading/test_cscm_position_sizing.py` 3/3 pass after `git pull --ff-only`
- **EC2 service health**: `homeguard-cscm.service` restarted cleanly after pull, `is-active` returns `active`, journal scan shows all 12 symbols streaming, no errors/exceptions/failures
- **Smoke-equivalent**: the new tests use real `DemoBroker` with `randomize_slippage=False` (forcing the worst-case 5 bps adverse slippage on every fill), so they exercise the same code path that produced the original bug. The fact that they pass under those conditions proves the fix holds under the maximum slippage the simulator can produce.

Not validated in this session (intrinsically time-bound):
- **Live Sat 2026-05-02 rebalance**: the real end-to-end proof. After it fires, the decision-log record should show all N executions filled with no shortfall.
