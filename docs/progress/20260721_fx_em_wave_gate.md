# FX EM Wave -- Verdict Campaign - 2026-07-21

## Summary

Ran the 7 pre-registered EM7 trials (`docs/strategies/research/20260721_fx_em_wave_preregistration.md`)
through walk-forward + the combined statistical gate. All 7 FAIL net of realistic EM costs. Per the
pre-registration's stopping rule, the EM carry/trend/mean-reversion catalog extension is declared
exhausted -- no wave-2 EM, no ML.

## Changes Made

- **`config/backtesting/em/*.yaml`**: 7 new configs, one per trial (EM7 universe, full
  2011-2026-04 data range, EM-specific cost model already wired in
  `src/backtesting/costs/fx.py`).
- **`scripts/backtest_scripts/run_fx_wave2_gate.py`**: fixed a bug where `run_gate()` read
  `rebalance` from the config into `kw` but never threaded it into the PRIMARY gated
  `walk_forward_fx(...)` call (only into the separate, non-gating S&P book-context helper). Latent
  since Wave 2 (2026-07-19, all trials were `weekly` so the bug was a no-op); surfaced by this
  wave's `rebalance: daily` trial (EM-CARRY-daily), whose first attempt silently re-ran the weekly
  gate (byte-identical metrics to EM-CARRY-weekly). One-line fix
  (`rebalance=kw.get("rebalance", "weekly")`); re-ran EM-CARRY-daily correctly afterward. The
  buggy first attempt's registry row (`run_id 3f2e6d66...`) is retained (never shrink N) and
  documented in the results doc as an apparatus artifact, not one of the 7 pre-registered trials.
- **Verified before running**: all 6 strategy classes (`FxCarry`, `FxCarrySeatbelt`, `FxTSMOM`,
  `FxXSectMom`, `FxCarryMom`, `FxMeanRev`) construct and run on the EM7 universe with no errors
  (smoke test on a short 2019-2020 window). EM7 spot cache loads cleanly for all 7 pairs
  2011/2014-2026-04-22; EM FRED rate panel loads with no NaNs.
- **Integrity finding (not a code change, a diagnostic)**: `FxCarrySeatbelt`'s crash-filter
  (`compute_unwind_score`) never generalized to EM7 -- its four terms (JPY/CHF strength delta,
  AUDJPY vol, XAUUSD return) are all absent from the EM7 universe, so the score is identically 0.0
  across the full history (verified: `score.min() == score.max() == 0.0`, 3,993/3,993 days). The
  EM-CARRY-SEATBELT trial therefore ran as a degenerate long-only carry+momentum-gate book with a
  non-functioning veto, not the pre-registered crash-filter mechanism. Documented in the results
  doc and the tracker.
- **Results docs**: `docs/reports/fx/em_wave_gate.md` (working copy, gitignored) +
  `docs/strategies/research/20260721_fx_em_wave_results.md` (durable tracked copy) -- full
  per-trial metrics table, FAIL reasoning per trial, cumulative-N/DSR accounting chain, fills
  artifact verification, stopping-rule outcome.
- **`docs/strategies/FX_60_CATALOG_TRACKER.md`**: added an EM WAVE RESOLUTION callout; updated
  rows #3, #4, #8 (EM-universe variant results), #18 (BT/WF/Gate filled, FAIL), #19 (EM seatbelt
  degenerate-filter finding); moved #18 from DATA to READY in the summary-counts table; updated
  the EM-spot-pairs unblock-roadmap row.

## Commits

(see `git log` for hashes -- committed by explicit path per repo convention, main dir, no push)

## Known Issues / Remaining Work

- None outstanding for this wave -- it is closed per the pre-registered stopping rule.
- `run_fx_wave2_gate.py`'s rebalance-threading bug fix should be spot-checked against any other
  in-flight non-weekly-rebalance gate work that might have used the buggy code path before this
  fix (none identified in this session -- Wave 2 was entirely `weekly`).

## Validation

- Smoke-tested all 6 strategy classes on the EM7 universe (short window, `register=False`) before
  committing to the full 15-year walk-forward runs.
- Verified `output/backtests/<strategy>/runs/<run_id>/manifest.csv` + non-empty
  `trades_oos.csv.gz` exist for all 7 trials (methodology Section 12 fills-level trade-log
  mandate) -- fill counts range 106 (EM-CARRY-SEATBELT, degenerate) to 21,282 (EM-CARRY-daily).
- Verified the rebalance-threading fix by confirming EM-CARRY-daily's re-run produced materially
  different metrics from EM-CARRY-weekly (0.0586 vs 0.0245 OOS Sharpe, ~5x more fills), consistent
  with genuinely different rebalance cadence.
- Confirmed all 8 registry rows (7 valid trials + 1 bug-artifact duplicate) present in
  `output/experiments.duckdb` with correct `asset_class='fx'`, `data_frequency='daily'`.
