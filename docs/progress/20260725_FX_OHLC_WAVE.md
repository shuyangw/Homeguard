# FX OHLC / Range-Based Wave -- Verdict Phase - 2026-07-25

## Summary

Ran the VERDICT phase of the LOCKED pre-registered FX OHLC / range-based wave: 4
trials (Keltner reversion #12, bandwidth squeeze #27, vol-spike fade #29, ADX-gated
trend #6) through `run_fx_walkforward.py` on the G10-22 universe. **All four FAIL**
the pre-committed combined gate, confirming the registered prediction. Cumulative
project-wide N went 137 -> 141. The range-based (ATR/ADX/Parkinson) family is closed
for this daily-spot-taker construction; no sweep, no ML variant.

## What was run

Walk-forward train 36m / test 12m / step 12m -> 13 non-overlapping OOS windows,
2014-01-01 .. 2026-04-30, 3,217 stitched OOS days, cost legs 1.0x and 1.5x,
`execution_lag=1`, weekly rebalance, vol target 0.03/instrument, IDM on,
PSR/DSR at `periods_per_year=252`.

| Trial | Sharpe 1x | Sharpe 1.5x | PSR | DSR | PBO | N | Verdict |
|---|---|---|---|---|---|---|---|
| OHLC-KELTNER (#12) | -0.4762 | -0.6621 | 0.0459 | 0.0000 | 0.6806 | 137 | FAIL |
| OHLC-SQUEEZE (#27) | -0.4783 | -0.6573 | 0.0437 | 0.0000 | 0.6352 | 138 | FAIL |
| OHLC-VOLSPIKE (#29) | +0.0873 | +0.0114 | 0.6235 | 0.0000 | 0.2448 | 139 | FAIL |
| OHLC-ADX-TREND (#6) | -0.3089 | -0.4234 | 0.1357 | 0.0000 | 0.2727 | 140 | FAIL |

Deflated bar SR_zero = 1.129 at N=137 (v = 0.42778 from the Tier B wave), matching
the pre-registered ~1.13 prediction to 3 decimals. Best result (+0.0873) is 7.5% of
the bar. S&P Sharpe over the same OOS dates = 0.6842; all four have IR vs S&P of
-0.60 to -0.88, so marginal book contribution is negative for all four.

## Changes Made

- **`docs/strategies/research/20260725_fx_ohlc_wave_results.md`** (new, durable):
  full per-trial metrics, failure reasons, ADX-vs-FxTrend baseline comparison,
  trial-count/deflation accounting, fills verification, apparatus notes, scoped
  conclusion, integrity self-audit.
- **`docs/reports/fx/ohlc_wave_gate.md`** (new, working): condensed gate report.
- **`docs/reports/fx/ohlc/{keltner,squeeze,volspike,adx_trend}_wf.md`** (new):
  per-trial readiness reports emitted by the runner, including per-window OOS Sharpe.
- **`docs/strategies/FX_60_CATALOG_TRACKER.md`**: OHLC WAVE RESOLUTION banner (matching
  the SCOPE banner framing); Gate column filled for rows #6, #12, #27, #29; summary
  counts updated (OHLC 8 -> 4, READY 16 -> 20); note recording WHY the remaining 4
  OHLC entries (#1, #8, #28, #47) were explicitly excluded rather than merely untested.

## Key decisions

- **Baseline CITED, not re-run.** The ADX enhancement's FxTrend baseline (#3: OOS
  -0.02, DSR 0.20, PBO 0.85) was cited from the existing catalog record rather than
  re-executed, per the pre-committed rule that an existing baseline does not consume
  a trial. Caveat recorded: the baseline pre-dates the 2026-07-25 apparatus
  hardening, so the -0.02 -> -0.31 delta is NOT a controlled A/B.
- **S&P benchmark leg computed read-only.** `run_fx_walkforward.py` does not produce
  it, so it was recomputed with `register=False` and no FillSink -- a diagnostic is
  not a trial, and N stayed at 141.
- **VOLSPIKE reported as FAIL, not as a lead.** It is the only positive spec and it
  survives 1.5x cost, but at 594 fills with per-window Sharpe from -1.90 to +1.93 it
  is noise (PSR 0.62). Promoting it would require exactly the degrees of freedom the
  pre-registration exists to prevent.
- **No parameter swept.** Per pre-reg, no alternative ATR multiple, ADX threshold, or
  z-window was tried after the failures.

## Known Issues / Remaining Work

Two apparatus defects FILED during adjudication (not patched mid-run, per the
integrity rule that patching gate code during adjudication is itself a researcher
degree of freedom). Neither can have manufactured a false pass, since nothing passed:

1. **`scripts/backtest_scripts/run_fx_walkforward.py` omits the S&P benchmark leg**
   that `scripts/backtest_scripts/run_fx_wave2_gate.py` computes (`correlation_sp500`,
   `information_ratio_sp500`, `marginal_deflated_contribution_proxy` via
   `src/backtesting/benchmark.py`). The pre-registered gate REQUIRES the benchmark /
   marginal-contribution check, so the walk-forward runner cannot adjudicate the full
   gate alone -- it depends on the adjudicator remembering to add it. Reconcile the
   two runners.
2. **`run_fx_walkforward.py --report` defaults to the shared baseline path**
   `docs/reports/fx/FX_WALK_FORWARD.md`, so consecutive gate runs silently clobber
   each other's report and the FxTrend baseline report. Worked around with explicit
   per-trial paths; the default should be run-scoped.

Minor inconsistency, not changed: the tracker's summary counts INTRADAY as 22 (its
listed strategies do number 22) while the OHLC pre-registration's stopping rule says
"INTRADAY (21)". Left alone since it is outside this wave's scope.

Next: remaining catalog blockers are INTRADAY and ML, both substantial builds. No
further daily-spot FX specs are unblocked by this wave's engine work.

## Validation

- Fills verified on disk BEFORE any verdict was accepted (methodology Section 12):
  non-empty `trades_oos.csv.gz` + 53-artifact `manifest.csv` under each run-scoped
  sink -- 12,171 / 8,723 / 594 / 6,999 OOS fills respectively.
- All 4 runs confirmed appended to `output/experiments.duckdb` (`phase=walk_forward`,
  `asset_class=fx`) with per-trial N of 137/138/139/140; registry ends at 141,
  re-queried via `get_campaign_trial_distribution()`.
- Cross-trial spread v = 0.42778 and SR_zero = 1.1291 at N=137 confirmed
  independently against the pre-registration's stated values.
- Universe confirmed identical (22 pairs) across all four configs before running.
