# Experiment 4 -- V12 Lag-Asymmetry Decomposition

**Date**: 2026-05-24
**Branch**: v12-bear-to-cash
**Builds on**: V12 readiness lag-degradation panel (Sharpe 0.665 lag vs 0.268 near_close at 5 bps)
**Decision target**: V12c readiness cost-sensitivity gate stringency

## Summary

V12 one_day_lag beats V12 near_close by +0.397 Sharpe at 5 bps. Decomposing the daily P&L difference (`lag - near_close`) by regime-transition bucket shows the asymmetry is **DIFFUSE**: only 38.1% of the gap localizes to any-transition days, while 61.9% lives on persistent (non-regime-flip) days. The single largest sub-bucket is BEAR-exit (39.8% share) but BEAR-onset is small (6.9%) and other-transition days actually drag the gap down by 8.6%. Per the decision criterion (< 50% transition share -> DIFFUSE), the lag tax is not a clean "BEAR-transition cost" story and the V12c readiness gate should not be tightened on the transition-localization hypothesis alone -- a deeper mechanism investigation is required before V12c deploys.

## Methodology

- Two V12 backtests at 5 bps with v12.0.0 defaults (BEAR='cash', `delta_rebalance_pct=0.02`, `min_regime_days=0`): one in `near_close` mode and one in `one_day_lag` mode. **Re-ran the harness** -- no per-day NAV/P&L artifacts from the V12 readiness orchestrator were cached under `output/backtests/` or elsewhere, so the script re-invokes `run_variant` twice (~50s each, ~100s total). Sharpes reproduced exactly: near_close=0.2683, one_day_lag=0.6650, gap=+0.3967, matching the readiness report row-for-row.
- Per-day returns extracted from `DailyRecord.daily_return` on each run; aligned on the common date index (2017-01-03 .. 2026-05-16, n=2355 trading days).
- Days classified by regime transition status using `diagnostics/regime/v0/labels.parquet`:
  - `bear_onset`: `regime[t-1] != BEAR and regime[t] == BEAR`
  - `bear_exit`: `regime[t-1] == BEAR and regime[t] != BEAR`
  - `other_transition`: any other regime change
  - `persistent`: `regime[t-1] == regime[t]` (also includes the first day as a degenerate accounting bucket; impact is one row, no diff)
- Sharpe-gap attribution per bucket = `(sum_of_daily_diff_bucket / total_sum_of_daily_diff) * gap`, where `gap = 0.3967`. This is the linear share approximation documented in the script header; see Limitations.

## Decomposition

| Bucket | n_days | sum_daily_diff | mean_daily_diff | share_of_total | implied_sharpe_contribution |
|---|---:|---:|---:|---:|---:|
| bear_onset | 63 | +0.08353 | +0.001326 | 6.9% | +0.027 |
| bear_exit | 63 | +0.48065 | +0.007629 | 39.8% | +0.158 |
| other_transition | 359 | -0.10437 | -0.000291 | -8.6% | -0.034 |
| persistent | 1870 | +0.74823 | +0.000400 | 61.9% | +0.246 |

Total `sum_daily_diff = +1.208`; total implied Sharpe contribution = +0.397 (recovers the observed gap).

## Localization analysis

- Transition-day share of Sharpe gap: **38.1%** (bear_onset + bear_exit + other_transition)
- BEAR-onset specifically: **6.9%** (63 days, mean diff +13 bps/day)
- BEAR-exit specifically: **39.8%** (63 days, mean diff +76 bps/day -- the strongest single bucket per day)
- Other-transition days: **-8.6%** (359 days, mean diff -3 bps/day -- net drag on the gap)
- Persistent days: **61.9%** (1870 days, mean diff +4 bps/day -- diffuse but dominant by mass)

The mean-diff column is informative: BEAR-exit days carry by far the largest **per-day** lag advantage (+76 bps), consistent with the readiness panel's observation that the detector fires ~3.4 trading days AFTER the SPY trough. In near_close mode, V12 re-enters at the (already-recovered) close of the exit day; in one_day_lag mode, V12 re-enters the next day, which on average gives back less of the rebound. But because BEAR-exit days are few (63 of 2355, 2.7%), their total share is capped at 40%. The 1870 persistent days each contribute only +4 bps but their mass pushes their total share to 62%.

## Verdict

**DIFFUSE.**

Per the decision criterion: *"DIFFUSE: < 50% of the asymmetry localizes to transition days. Implication: mechanism is elsewhere; deeper diagnosis needed before any cash-transition variant deploys."*

38.1% < 50%, so this is unambiguously the DIFFUSE bucket. The +0.397 gap is **not** a clean transition-day execution premium that a tighter cost stress would surface; it is partly a transition effect (concentrated in BEAR-exit, +0.158 contribution) and partly a structural same-bar / next-bar P&L difference that affects all 1870 hold-and-rebalance days.

## Implications for V12c readiness (Experiment 6)

Since the verdict is DIFFUSE (not TRANSITION-LOCALIZED), the simple "add a 10 bps stress" prescription **does not apply** on the transition-localization argument alone. Instead, the V12c readiness path must answer the prior question: what mechanism causes near_close to underperform one_day_lag by ~4 bps per persistent day?

Candidate hypotheses to investigate before V12c deploys:

1. **Same-bar pricing penalty on every rebalance.** `near_close` plans the signal at close T from `panel.loc[:T]` (inclusive of T) and trades at close T. If the signal uses any same-day feature that the close price already reflects, the strategy is "buying the close after it moved," and that penalty applies to every rebalance day, not just regime flips. `one_day_lag` defers execution to T+1 close, which gives the signal a full bar of distance from the price it executes at.
2. **Regime-panel artefact on the held side.** Holdings (not just regime-flip transitions) are MTM'd at close T in both modes, but the rebalance toward target uses a different price reference. Need to confirm the engine's MTM formula handles the modes symmetrically.
3. **Intraday momentum reversion.** Daily winners often give back into the close; near_close enters at the (high-water-mark) close, one_day_lag waits a day and benefits from the reversion.

V12c readiness gate stringency should match this diagnosis path: keep the existing 5-7.5 bps cost grid, but **add a same-bar vs next-bar P&L diagnostic** that decomposes by rebalance type (full rebalance vs delta-only) before sizing any new gates. Tightening cost stress to 10 bps without that diagnostic would be addressing the wrong free parameter.

## Limitations

- **Sharpe decomposition uses a linear share approximation.** Actual annualized Sharpe is non-linear in daily returns (depends on `mean / std`, and per-bucket std would have to be combined via the cross-bucket covariance to give a true Sharpe attribution). The linear `share_of_total * gap` formula attributes the gap proportionally to the daily P&L difference sum, which is exact for the CAGR-like component of Sharpe (sum of returns) but ignores how each bucket affects the denominator (volatility). Per the task spec this is the documented assumption.
- **Per-day P&L was acquired by re-running the harness, not from a cached artifact.** No V12 per-day NAV files existed under `output/backtests/ramp_phase4/` or `output/backtests/`. The 100-second double-run reproduced the readiness report's Sharpes to four decimal places, so the input is trustworthy. The wrapper that exposes per-day returns is `_records_to_returns` in the script -- it just pulls `DailyRecord.daily_return` into an indexed Series.
- **No standard error on bucket contributions.** With 1870 persistent days the persistent bucket's mean estimate is statistically stable. The transition buckets are noisier: BEAR-onset and BEAR-exit have 63 days each, and a single outlier day on those small buckets can move the share by several percentage points. The qualitative verdict (DIFFUSE vs LOCALIZED) is robust to noise of this scale -- transition share would need to roughly double to cross the 80% threshold -- but the specific +0.158 BEAR-exit contribution carries an uncertainty band that this experiment does not quantify.

## Artifacts

- `notebooks/research/experiment4_lag_asymmetry.py`
- `diagnostics/regime/lag_asymmetry/decomposition.csv`
- `diagnostics/regime/lag_asymmetry/verdict.txt`
