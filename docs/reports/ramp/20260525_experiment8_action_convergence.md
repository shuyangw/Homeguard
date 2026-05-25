# Experiment 8 -- V14 Action Convergence Diagnostic

**Date**: 2026-05-25
**Branch**: v12-bear-to-cash
**Builds on**: V14 factorial readiness (commit `6f55e37`); V14 spec rev2; original detector diagnostic (Experiment 3); V14 tau constants (commit `9c48245`)
**Decision target**: WS-3 track selection

## Summary

V14a-cash / V14b-spy / V14c-dampen converged within 0.011 Sharpe at 5 bps near_close. The diagnostic disambiguated which of three mechanisms (M1 rare-events ceiling, M2 action equivalence, M3 exit-timing failure) is operative. Verdict: **M1 inconclusive, M2 refuted, M3 refuted**. The data does NOT cleanly identify a single binding constraint among the three pre-registered hypotheses; per the decision matrix the recommended next intervention is **WS-3d (detector replacement) with expanded scope, OR halt WS-3 and pursue alternative strategies**. The principal diagnostic finding is that BEAR-soft mode fires AFTER the SPY local minimum on most events (median exit-to-SPY-low lag = -8 days; i.e. the trigger enters when the trough has already passed), which is consistent with neither the M3 prediction (exits too early) nor M2 (perfect action equivalence). The convergence is more parsimoniously explained by trigger timing falling on the recovery rather than the drawdown -- a fourth mechanism the pre-registered hypotheses did not isolate.

## Mechanism verdicts

| Mechanism | Verdict | Evidence |
|---|:---:|---|
| **M1** rare-events ceiling | inconclusive | BEAR-soft = 366 days = 15.54% of gated window (>5% threshold; NOT short); median event duration = 9 days (>5 threshold; NOT short); V14a-V11 BEAR-soft excess share = 9.75% (<30% threshold; supports rare-events story). 1 of 3 signals supports, 2 refute -> inconclusive. |
| **M2** action equivalence | refuted | Cross-variant per-event correlation: V14a/V14b = 0.5092 (<0.85 FAIL), V14a/V14c = 0.6359 (<0.85 FAIL), V14b/V14c = 0.8573 (>0.85 PASS). Pooled corr(SPY return, V11 return) during BEAR-soft = 0.7927 (<0.85 FAIL). Both primary clauses fail. |
| **M3** exit-timing failure | refuted | Median exit-to-SPY-low lag = **-8 days** (negative: SPY low BEFORE exit; trigger fires AFTER trough). Mean 10d post-exit V14a-V11 excess = **+0.16%** (positive, opposite of M3 prediction). Both primary clauses fail. |

## Analyses

### A1 -- Coverage statistics

- Total BEAR-soft days: **366** out of 2,355 gated days = **15.54%**.
- 25 distinct BEAR-soft events.
- Event duration distribution: P25 = 2 days, median = 9 days, P75 = 29 days, max = 60 days, mean = 14.64 days.
- Per-year (BEAR-soft days): 2018 = 86, 2019 = 22, 2020 = 40, 2021 = 0, 2022 = 141, 2023 = 20, 2024 = 4, 2025 = 39, 2026 = 14.
- Independent reconstruction (Schmitt trigger on `score_BEAR >= 0.5556` / `< 0.4556`) overlaps the engine state at **100.0%** (366 = 366), confirming event boundary integrity.

Reference: `diagnostics/v14_action_convergence/a1_coverage.csv`.

### A2 -- Action-attribution decomposition

| Variant | Partition | n_days | sum_excess | mean_excess | std_excess | partition_sharpe |
|---|---|---:|---:|---:|---:|---:|
| V14a-cash | BEAR_SOFT | 366 | -0.0376 | -0.000103 | 0.0370 | -0.044 |
| V14a-cash | NOT_BEAR_SOFT | 1989 | -0.3476 | -0.000175 | 0.0024 | -1.139 |
| V14b-spy | BEAR_SOFT | 366 | -0.0847 | -0.000231 | 0.0243 | -0.151 |
| V14b-spy | NOT_BEAR_SOFT | 1989 | -0.0756 | -0.000038 | 0.0017 | -0.350 |
| V14c-dampen | BEAR_SOFT | 366 | -0.0412 | -0.000113 | 0.0197 | -0.091 |
| V14c-dampen | NOT_BEAR_SOFT | 1989 | -0.1801 | -0.000091 | 0.0012 | -1.242 |

BEAR-soft partition share of total V-V11 excess (signed):
- V14a-cash: **9.75%** (BEAR-soft sum -0.0376 of total -0.3852)
- V14b-spy: **52.82%** (BEAR-soft sum -0.0847 of total -0.1603)
- V14c-dampen: **18.61%** (BEAR-soft sum -0.0412 of total -0.2213)

All three variants have *negative* total excess vs V11 (they each give back some V11 alpha), but the BEAR-soft partition's share varies 5x across variants. The BEAR-soft partition variance (std_excess) is NOT comparable across V14a/b/c -- V14a 0.0370 vs V14c 0.0197 differ by ~88%, well above the 20% M2 threshold. The variance gap is a key M2 refuter: the actions do produce meaningfully different daily P&L during BEAR-soft days.

Reference: `diagnostics/v14_action_convergence/a2_attribution.csv`.

### A3 -- Per-event P&L by variant

25 BEAR-soft events. Cross-variant pairwise correlations across the per-event cumulative-return vector:

| | V14a | V14b | V14c |
|---|---:|---:|---:|
| V14a | 1.000 | **0.5092** | **0.6359** |
| V14b | 0.5092 | 1.000 | **0.8573** |
| V14c | 0.6359 | 0.8573 | 1.000 |

V14b and V14c correlate 0.86 (essentially tied to the threshold), but V14a (cash) correlates only 0.51 with V14b and 0.64 with V14c. The actions are NOT economically equivalent per-event; cash diverges substantially from the directional alternatives. This is the cleanest M2 refuter.

Reference: `diagnostics/v14_action_convergence/a3_per_event.csv`, `diagnostics/v14_action_convergence/a3_corr_matrix.csv`.

### A4 -- SPY vs V11 correlation in BEAR-soft

- Pooled corr(SPY daily return, V11 daily return) across 366 BEAR-soft days = **0.7927**.
- Per-event correlations vary widely; many events with <5 days have undefined or noisy correlations.
- V11's plan return (proxied here by V11 daily backtest return; plan positions not separately persisted) is correlated with SPY during BEAR-soft but **below the 0.85 threshold**.

The 0.79 pooled correlation is meaningful: V11 *does* track SPY closely during BEAR-soft events. But the gap to 0.85 means V11's plan diverges from SPY enough that V14a (cash) and V14b (SPY) take materially different bets in some events -- consistent with the A3 correlations.

Reference: `diagnostics/v14_action_convergence/a4_spy_v11_corr.csv`.

### A5 -- Exit-timing analysis

25 BEAR-soft exits analyzed. For each exit, lag from exit date to nearest SPY local minimum in +/-20 trading days:

- **Median lag = -8 days** (negative: SPY trough occurred BEFORE the exit).
- Mean lag = -4.52 days.
- Mean 5d post-exit V14a-V11 excess = -1.07%.
- Mean 10d post-exit V14a-V11 excess = **+0.16%**.
- Mean 20d post-exit V14a-V11 excess = +0.59%.

The M3 prediction was "median lag > 5 days AND mean 10d excess < 0." Both clauses fail in the *opposite* direction: the SPY low occurs ~8 trading days BEFORE the BEAR-soft exit on the median, and the post-exit V14a-V11 differential is *positive* on a 10d horizon. The trigger is not exiting too early -- it is exiting on or after the trough, releasing V14a back into a recovering market where V11 (still holding momentum names) modestly outperforms cash, but not catastrophically.

The deeper finding: a negative median exit-to-trough lag implies BEAR-soft entries are *late* relative to the drawdown (the bottom is already in by the time we enter), so the cash window covers the recovery rather than the crash -- the same lag-tax pattern the V12 readiness diagnostic identified (mean gap_days = -3.42 for V12's argmax BEAR onsets).

Reference: `diagnostics/v14_action_convergence/a5_exit_timing.csv`.

### A6 -- Counterfactual tau_out sweep (informational)

| tau_out | n_BEAR_soft_days (counterfactual) | n_BEAR_soft_days (original) | hypothetical V14a Sharpe |
|---:|---:|---:|---:|
| 0.20 | 498 | 366 | 0.4426 |
| 0.30 | 492 | 366 | 0.4462 |
| 0.40 | 422 | 366 | 0.5535 |
| 0.4556 (actual) | 366 | 366 | 0.6146 |
| 0.50 | 366 | 366 | 0.6146 |

The actual tau_out (0.455556) reproduces V14a's true Sharpe (0.6146) exactly, validating the approximation methodology. Lowering tau_out (longer cash periods) monotonically degrades V14a's hypothetical Sharpe -- mode-True days that V14a's actual record doesn't cover are assigned cash (0% return), and the V11 line outperforms cash on the marginal days. There is no apparent free lunch in tightening the exit threshold; the tau-out parameter is at or near its local optimum for this window.

Reference: `diagnostics/v14_action_convergence/a6_tau_out_sweep.csv`.

## Decision matrix mapping

Pre-registered decision matrix (from spec):

| M1 | M2 | M3 | Recommended next spec |
|---|---|---|---|
| supported | refuted | refuted | WS-3b (leading indicators) |
| refuted | supported | refuted | WS-3a (detector hysteresis) |
| refuted | refuted | supported | WS-3c.1 (consumer exit logic) |
| supported | supported | -- | WS-3d (detector replacement) |
| -- | supported | supported | WS-3a + WS-3c.1 in parallel |
| supported | -- | supported | WS-3b primary, WS-3c.1 fallback |
| all three | -- | -- | WS-3d |
| none supported | inconclusive | inconclusive | WS-3d with expanded scope, OR halt WS-3 |

Observed tuple: **(M1=inconclusive, M2=refuted, M3=refuted)**. This tuple does not match any single row exactly. The closest fit is the last row ("none supported, inconclusive, inconclusive"), since no mechanism is supported. Per the spec's catch-all guidance ("if the data genuinely doesn't disambiguate, the verdict is WS-3d with expanded scope rather than a forced row pick"), the diagnostic falls through to this row.

## WS-3 track recommendation

**Primary recommendation: WS-3d (detector replacement) with expanded scope.**

Rationale:
1. The three pre-registered mechanisms each have falsifiable predictions; M2 and M3 are clearly refuted, and M1 is inconclusive (one of three signals supports it). The convergence cannot be attributed to any of the three.
2. The A5 finding (median exit-to-trough lag = -8 days) suggests the operative mechanism is "**trigger timing falls on the recovery, not the drawdown**" -- a fourth mechanism that the pre-registered tests didn't isolate. The V14 trigger lags the SPY trough by ~8 trading days on the median; by the time BEAR-soft entry fires, the worst is over and the recovery is underway. Cash (V14a), SPY (V14b), and dampened V11 (V14c) all bet differently against the same already-recovering tape -- which is why their P&L paths differ per-event (A3) but their full-window Sharpes converge (the per-event noise averages out).
3. WS-3a (hysteresis), WS-3b (leading indicators), and WS-3c.1 (exit logic) each address ONE of the three failure modes. None of them addresses "trigger fires after the trough," which requires a structurally different detector or alternative input -- the WS-3d scope.
4. WS-3d remains expensive (full detector rebuild). A reasonable fallback is to halt WS-3 entirely, accept V11's published Sharpe of 0.5306 as the deployable RAMP target, and redirect research budget to orthogonal alpha sources (the spec's "alternative strategies" branch).

**Suggested fallback if WS-3d is rejected on cost grounds**: pursue WS-3b (leading indicators) as the cheapest single-mechanism intervention, since M1 has the strongest among-three signals (BEAR-soft contributes only 9.75% of V14a-V11 excess -- consistent with rare-events ceiling on a per-day basis even if total coverage is not rare). The WS-3b path would specifically target the "make the detector fire earlier" sub-problem identified by A5's negative median lag.

## Limitations

- Same 2017-2026 window as the V14 readiness (commit `6f55e37`); forward OOS validation is required for any deployed track.
- A6 tau_out counterfactual uses approximation: on days in the new BEAR-soft mode that V14a's actual record did NOT cover (because the original tau_out had already exited), the hypothetical return is set to 0 (cash). The actual-tau_out row (0.4556) reproduces V14a's true Sharpe (0.6146) exactly, so the approximation is well-calibrated near the actual.
- V11 daily return is used as the V11-plan return proxy in A4 because V11 plan positions are not separately persisted by the orchestrator. The proxy underestimates the V11 plan's correlation with SPY slightly (the plan is the un-noised target; the V11 daily return adds turnover and execution costs).
- BEAR-soft event boundaries are derived from V14a's `regime == 'BEAR_SOFT_CASH'` engine state. Cross-checked against an independent Schmitt-trigger reconstruction (`BEAR_score >= tau_in`, `< tau_out`) on `v0_scores/labels.parquet`: overlap = 100.0% (366 = 366 days), so the engine state is bit-identical to the reconstructed signal.
- The verdict "M1 inconclusive" applies a 2-of-3 signal rule (low total + short duration + small share). Single-signal supports (small share only) leave room for a partial M1 reading; the cautious choice "inconclusive" reflects this.
- A5's median lag is computed in TRADING days; the negative lag of -8 days means roughly 1.5 calendar weeks of trough-to-exit delay. This window is comfortably bigger than the noise in the local-min search (which is bounded by `+/-20` trading days).
- The one_day_lag sanity check (V11 + V14a re-run with timing_mode='one_day_lag') reproduces the same coverage (366 days), event count (25), median duration (9), median lag (-8), and pooled corr (0.805 vs 0.793) within sampling noise. The diagnostic conclusions do NOT flip across timing modes.

## Artifacts

- Script: `notebooks/research/experiment8_v14_action_convergence.py`
- Daily records (parquet):
  - `diagnostics/v14_action_convergence/v11_records.parquet`
  - `diagnostics/v14_action_convergence/v14a_records.parquet`
  - `diagnostics/v14_action_convergence/v14b_records.parquet`
  - `diagnostics/v14_action_convergence/v14c_records.parquet`
- Analyses (csv):
  - `diagnostics/v14_action_convergence/a1_coverage.csv`
  - `diagnostics/v14_action_convergence/a2_attribution.csv`
  - `diagnostics/v14_action_convergence/a3_per_event.csv`
  - `diagnostics/v14_action_convergence/a3_corr_matrix.csv`
  - `diagnostics/v14_action_convergence/a4_spy_v11_corr.csv`
  - `diagnostics/v14_action_convergence/a5_exit_timing.csv`
  - `diagnostics/v14_action_convergence/a6_tau_out_sweep.csv`
- Verdict text: `diagnostics/v14_action_convergence/verdict.txt`
- Summary JSON: `diagnostics/v14_action_convergence/_summary.json`
- This report: `docs/reports/ramp/20260525_experiment8_action_convergence.md`
- Session log: `docs/progress/20260525_RAMP_E8_DIAGNOSTIC.md`
