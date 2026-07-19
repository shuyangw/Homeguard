# #20 London Open Breakout -- Cost-Sensitivity Re-Gate (override_pips=0.5)

Generated 2026-07-19. Integrity re-gate of the already-FAILED (2026-07-19)
London Breakout gate, testing an IBKR-optimistic cost assumption. NOT a
parameter re-tune -- entries, stops, targets, and all other strategy params
are unchanged from `config/backtesting/fx_london_breakout.yaml`. Only the
round-trip spread cost term is overridden.

## Cost assumption tested

`override_pips=0.5` per side (1.0 pip round-trip after the internal x2), an
IBKR-optimistic bound, passed via a new `override_pips` kwarg on
`LondonBreakoutStrategy` that bypasses the tier lookup in
`fx_round_trip_pips()` (`src/backtesting/costs/fx.py`). The BASE gate used
`tier="major"`, `session="london"` -> a 1.2x-session-multiplied round-trip
spread of ~1.2 pips (the tier's 0.5-1.5 pip range midpoint of 1.0, x1.2 london
session, x2 round-trip = 2.4 pips round-trip). The 0.5x-per-side override
(1.0 pip round-trip) is materially tighter than BASE.

## Results

| Metric | BASE (major tier) | OPTIMISTIC (0.5 pip/side) |
|---|---|---|
| OOS Sharpe (net) | -1.5995 | -0.7483 |
| S&P Sharpe (same OOS dates) | 0.6767 | 0.6767 |
| Beats S&P | False | False |
| IS Sharpe (mean per-window) | -1.2177 | -0.4237 |
| PSR (diag) | 0.0000 | ~2.02e-307 (effectively 0) |
| DSR (diag) | 0.0000 | ~2.02e-307 (effectively 0) |
| PBO (diag) | 0.6441 | 0.7195 |
| Correlation to S&P | -0.0277 | -0.0292 |
| Information ratio vs S&P | -0.8738 | -0.7644 |
| n_windows / n_oos_days | 13 / 3064 | 13 / 3064 |
| Active OOS days (>=1 trade) | 2228 | 2228 |
| OOS window | 2014-01-03 .. 2026-04-01 | 2014-01-03 .. 2026-04-01 |

Trial count (DSR): project-wide cumulative count from `n_trials_project_wide()`
(`output/experiments.duckdb`) at the time of this run was 0 (no prior
`backtest-optimizer`-agent rows), so `trial_count = 0 + 1 = 1` was used for
both the BASE reproduction and the OPTIMISTIC leg -- consistent with the
frozen report's `DSR (diag, trials=1)`. Note `n_trials_project_wide()` sums
`combinations_in_run` filtered to `agent_name = 'backtest-optimizer'` only;
this harness's own registry rows (`agent_name='fx-harness-walkforward'`, this
task) do not feed that counter.

## Verdict: FAIL

Halving the round-trip cost assumption to an IBKR-optimistic 0.5 pip/side
improves OOS Sharpe from -1.5995 to -0.7483 -- a large improvement in
magnitude but the strategy remains deeply negative and does not clear the
primary gate (stitched OOS Sharpe > S&P Sharpe over the same OOS dates:
-0.7483 vs 0.6767). PBO is materially worse at the optimistic cost level
(0.7195 vs 0.6441), and both PSR/DSR remain effectively zero. This confirms
the original FAIL is not primarily a cost-model artifact -- the edge does not
exist even under a favorable cost assumption; the underlying signal is
adverse to the market (negative correlation, negative information ratio in
both legs).

## Registry

- Backfill run_id (BASE, `docs/reports/fx/FX_LONDON_BREAKOUT_WALK_FORWARD.md`
  numbers, retroactive): `27396cad-7044-4d76-a6ce-073ef2419aad`
- New trial run_id (OPTIMISTIC leg, `override_pips=0.5`):
  `33e45b88-6ba3-46eb-9d6e-ad066d871d2e`

## Sanity-check regression (Task 3, mandatory pre-check)

Before trusting the optimistic-leg numbers, `run()` was re-executed with the
modified code and `override_pips=None` (default) to confirm the
`override_pips` threading introduced no regression. Result matched the frozen
report to 4-6 decimal places:

| Metric | Frozen report | Reproduced |
|---|---|---|
| OOS Sharpe | -1.5995 | -1.599506 |
| S&P Sharpe | 0.6767 | 0.676695 |
| PBO | 0.6441 | 0.644134 |
| n_windows / n_oos_days / active | 13 / 3064 / 2228 | 13 / 3064 / 2228 |

PASS -- no regression from the code change.

## Limitations

- **Clean isolation confirmed**: read of `LondonBreakoutStrategy.__init__`,
  `_arm_oco`, `_maybe_arm`, `_maybe_open`, and `_book` in
  `src/strategies/advanced/fx_london_breakout.py` confirms `self._rt_pips`
  (fed by `override_pips`) is used EXCLUSIVELY in `_book()` as a subtracted
  R-multiple cost term (`rt_spread_r`). `self.offset` (from the separate
  `offset_pips` parameter) drives entry-trigger placement in `_arm_oco` and is
  untouched by this change; stop/target levels in `_maybe_open` derive only
  from the Asian range and fill price. The override is a pure cost-term
  substitution -- entries, stops, and targets are bit-identical to BASE for
  any given day's price action.
- Inherits all BASE-report limitations: conservative 1m fills (worst-of
  trigger/open, adverse both-in-one-bar), half-spread slippage as a floor,
  approximate tier-1 event dates, daily ATR(14) sourced from `fx_daily` where
  available and otherwise aggregated from 1m bars per FX trading day
  (EURGBP/GBPJPY absent from `fx_daily`).
- This is a single alternate cost point (0.5 pip/side), not a full
  sensitivity curve; it does not by itself establish a cost threshold at
  which the strategy would pass (it remains a hard FAIL at this optimistic
  level).
- No entry/stop/target logic, parameters, or universe were changed --
  this is purely a cost-assumption substitution per the task's integrity
  constraints.
