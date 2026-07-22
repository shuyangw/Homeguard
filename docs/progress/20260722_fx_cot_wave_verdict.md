# FX COT Positioning Wave -- Verdict - 2026-07-22

## Summary
Ran the 3 pre-registered CFTC-COT positioning trials (COT-CONTRARIAN-TS, COT-MOMENTUM-TS,
COT-CONTRARIAN-XS) through walk-forward + the combined statistical gate. All 3 FAIL --
non-positive OOS Sharpe that worsens at 1.5x cost. Per the pre-registration's stopping rule,
this scoped slice (weekly net%OI z-score signal, D+7 publication lag, daily-spot-taker
execution, COT8 universe) is closed.

## Changes Made
- **Ran 3 walk-forward gates** via `scripts/backtest_scripts/run_fx_wave2_gate.py` (reused
  unchanged), one invocation per pre-registered spec, train=36m/test=12m/step=12m,
  2011-2026 COT/price overlap.
- **Verified fills-level trade logs**: each run's `output/backtests/<strategy>/runs/<run_id>/`
  contains a non-empty `trades_oos.csv.gz` (~272KB uncompressed each) + `manifest.csv` before
  accepting any verdict.
- **Verified cumulative trial-count accounting**: confirmed baseline N=120 before the wave via
  `get_campaign_trial_distribution()`, confirmed N=123 after (each of the 3 trials appended
  exactly one registry row via `walk_forward_fx`).
- **Wrote working gate report**: `docs/reports/fx/cot_wave_gate.md` (gitignored, generated
  output) plus per-trial reports in `docs/reports/fx/cot_wave/`.
- **Wrote durable results doc**: `docs/strategies/research/20260722_fx_cot_wave_results.md`
  (tracked, committed).
- **Updated the FX catalog tracker**: added a COT WAVE RESOLUTION banner to
  `docs/strategies/FX_60_CATALOG_TRACKER.md` and updated the SCOPE banner to note COT is now a
  tested-and-failed corner in this construction (not untested).

## Commits
- `7da7f2c` docs(fx): COT positioning wave verdict -- all 3 trials FAIL, scoped

## Results

| Trial | OOS Sharpe (1x) | OOS Sharpe (1.5x) | PSR | DSR | PBO | S&P corr | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| COT-CONTRARIAN-TS | -0.1292 | -0.1648 | 0.0000 | 0.0000 | 0.4732 | 0.0286 | FAIL |
| COT-MOMENTUM-TS | -0.1040 | -0.1965 | 0.0000 | 0.0000 | 0.4771 | 0.0064 | FAIL |
| COT-CONTRARIAN-XS | -0.1278 | -0.1601 | 0.0000 | 0.0000 | 0.2374 | -0.1293 | FAIL |

All three fail on the primary clause (non-positive OOS Sharpe) alone -- no near-miss, no
post-hoc degree of freedom invoked. Cost sensitivity WORSENS the Sharpe in all three cases
(genuine friction drag, not a marginal edge tipped by extra cost).

## Known Issues / Remaining Work
- None outstanding for this wave -- it is closed per its own stopping rule.
- Per the tracker's SCOPE banner, the untested FX families remain: liquidity-provider/maker
  execution (needs tick/L2 data), microstructure frequency, and other non-price signal families
  (order-flow, options-implied risk-reversals, cross-venue/triangular).

## Validation
- Confirmed `alt_data/cot/cot_fx.parquet` resolves via `get_local_storage_dir()` (not the
  repo-relative path implied by the pre-reg doc's prose) and is populated (2000-2026, 8 pairs).
- Confirmed no lookahead in `src/data/cot.py` (D+7 publication lag, forward-fill from active
  date only) and in `FxCotPositioningStrategy` (rolling z-scores computed entirely on the
  already-lagged weekly series).
- Confirmed all 3 runs appended to `output/experiments.duckdb` automatically by
  `walk_forward_fx` (no manual registry append needed).
- Confirmed fills artifacts non-empty for all 3 trials before accepting verdicts.
- Backtest sentinel (`.claude/.strategy-lead-active`) created at session start, removed at
  session end.
