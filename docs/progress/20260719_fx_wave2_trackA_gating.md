# FX Wave 2 Track A Gating (#33, #39, #42) - 2026-07-19

## Summary

Resumed FX Catalog Wave 2 Track A: fixed two crashed strategy implementations (#39 PCA
Dollar-Factor Residual, #42 RORO Regime Spread) and re-ran the combined statistical
gate for all 3 Track A strategies. All 3 fail/weak: #33 REJECT (confirmed), #39 REJECT,
#42 WEAK (positive 1x Sharpe but fails 1.5x cost sensitivity and DSR). None clears the
combined gate. Committed the 3 strategy implementations, configs, and tests that had
been sitting uncommitted in the working tree.

## Changes Made

- **`src/strategies/advanced/fx_pca_dollar_residual.py`**: fixed `IndexError('index 0
  is out of bounds for axis 0 with size 0')` in the PCA rebalance loop. The prior
  `returns_df.dropna(how="any")` emptied the trailing 250-day window whenever any of
  the 22 pairs (esp. Nordic/exotic crosses) had a gap on a single day within the
  window, feeding `np.linalg.svd` a zero-row matrix. Fixed to drop columns without
  complete history over the window instead of rows, requiring `>= 2*n_legs`
  complete-history columns to proceed (else flatten that rebalance cycle without
  disturbing the `pc1_jump` tracking state).
- **`src/strategies/advanced/fx_roro_regime_spread.py`**: fixed `KeyError` risk when
  the universe grows past the 3 hardcoded output columns (AUDJPY/CHFJPY/XAUUSD).
  `forecast_panel` now defaults every column in `cols` to a 0.0 forecast before
  overriding AUDJPY/CHFJPY from the state machine, so an added conversion-only leg
  (USDJPY) is carried in the panel but never traded.
- **`config/backtesting/fx_roro_regime_spread.yaml`**: added `USDJPY` to `universe` --
  both traded legs (AUDJPY, CHFJPY) are JPY-quoted, so the panel needs a USDJPY leg to
  mark JPY P&L to USD (`build_quote_usd_panel` requires it; XAUUSD needed no
  conversion, already USD-quoted).
- Ran `scripts/backtest_scripts/run_fx_wave2_gate.py` for #39 and #42 (fintech conda
  env, `python -m scripts.backtest_scripts.run_fx_wave2_gate`). Both completed cleanly
  this time; reports written to `docs/reports/fx/fx_pca_dollar_residual_wave2_gate.md`
  and `docs/reports/fx/fx_roro_regime_spread_wave2_gate.md`.
- Verified registry integrity directly against `output/experiments.duckdb`: exactly one
  run row per strategy (no duplicate/garbage rows from the earlier crashed attempts,
  which errored before the registry-append step). Honest project-wide trial count grew
  104 -> 105 -> 106, one increment per gate call.
- Wrote durable results copy: `docs/strategies/research/20260719_fx_wave2_trackA_results.md`.
- Updated `docs/strategies/FX_60_CATALOG_TRACKER.md` rows for #33, #39, #42 (BT/WF/Gate
  columns + notes) and bumped "Last updated" to 2026-07-19.
- Committed the previously-uncommitted Wave 2 Track A implementation set: the 3
  strategies, their configs, their tests, and `src/strategies/registry.py`
  (registrations for `FxTurnOfMonth`, `FxPcaDollarResidual`, `FxRoroRegimeSpread`).

## Verdicts

| # | Strategy | OOS Sharpe (1x/1.5x) | PSR | DSR | PBO | N | S&P corr | Verdict |
|---|---|---|---|---|---|---|---|---|
| 33 | Turn-of-Month USD | -0.28 / -0.36 | 0.00 | 0.00 | 0.84 | 104 | 0.03 | REJECT |
| 39 | PCA Dollar-Factor Residual | -0.12 / -0.22 | 0.00 | 0.00 | 0.38 | 105 | 0.02 | REJECT |
| 42 | RORO Regime Spread | 0.06 / -0.03 | 1.00 | 0.00 | 0.17 | 106 | 0.00 | WEAK |

None clears the Section 2.5 combined gate. #42's positive 1x Sharpe does not survive a
1.5x cost stress and shows DSR=0.0000 once deflated for the honest, growing trial
count -- this is a decisive FAIL, not a "genuinely close" case under the pre-registered
stopping rule (Section 6 of the design doc), which requires a meaningfully positive
DEFLATED Sharpe, not just a naive-PSR pass.

**Stopping rule not yet triggered either way.** The Wave 2 pre-registration's stopping
rule is scoped to all 6 strategies (3 Track A here + 3 Track B: #35, #37, #30, which
depend on the not-yet-built beta-weighted spread-execution engine). Track A alone
failing does not by itself trigger "declare the finding and stop" -- that requires
Track B to also complete and fail. This session did not touch Track B.

## Commits

See `git log` for the commit(s) made in this session (strategy implementations +
configs + tests + registry; gate reports; tracker + results doc + this session log).

## Known Issues / Remaining Work

- **Track B outstanding**: #35 (AUD/NZD beta-weighted spread), #37 (cointegration
  scanner), #30 (XAU/XAG relative-vol pair) all require the beta-weighted 2-leg
  spread-execution engine (per the Wave 2 design doc, Section 3) which has not been
  built. This is the remaining half of Wave 2 and is what determines whether the
  campaign's pre-registered stopping rule resolves to "Wave 3" or "decisive 8-mechanism
  failure, stop the campaign."
- **Minor numerical-hygiene item (non-blocking)**: `fx_pca_dollar_residual.py`'s
  standardization (`(X - mean) / std(ddof=0)`) can produce `inf` when a pair's trailing
  window has (near-)zero return variance (observed as `RuntimeWarning: overflow/invalid
  value in matmul` during the #39 re-gate, confined to early/thin-data windows). Did
  not affect the REJECT verdict (finite, sane monthly output) and the strategy fails
  the gate decisively regardless, so this was documented rather than fixed. If #39 is
  ever revisited (e.g. as a Wave 3 neighbor of a surviving statistical-residual
  mechanism), guard zero-variance columns before standardizing.

## Validation

- Both re-gate runs completed without errors (confirmed via captured stdout/stderr, not
  just exit code) and produced finite, monthly-coherent Sharpe/return series.
- Registry integrity verified by direct DuckDB query against `output/experiments.duckdb`
  (`runs` table): one row per strategy, no duplicates, monotonic trial-count growth.
- `git status`/`git log` confirms all 3 strategy files, their configs, their tests, and
  `registry.py` are committed (see Commits section).
