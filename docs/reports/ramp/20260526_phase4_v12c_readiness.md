# V12c Phase D Readiness Report (Experiment 6)

**Code commit**: d0b3464
**Data**: Alpaca SIP daily-aggregated, 2017-01-01 to 2026-05-16
**Universe**: config\universes\sp500-2025.csv
**Cost tier for PSR/DSR/PBO gates**: 5.0 bps per side
**n_trials_project**: 23 (hard-coded 23: V12 used 22 (4 from experiments.duckdb + 18 V12 runs); V12c is trial #23, V12-up-cash sensitivity now formalized as its own gate)

**Variant definition**: V12c = V12 plan_fn (BEAR-to-cash) with `regime_positions[UNPREDICTABLE] = "cash"`. Discovered as the V12-up-cash sensitivity finding in the 2026-05-24 V12 readiness report; formalized here.

**Pre-gate conditional-proceed context (E2 / E4)**: E2 hand-inspection of UNPREDICTABLE's drawdown-avoidance attribution returned AMBIGUOUS (top-3 event share 53.6%, COVID-dominant). E4 lag-asymmetry decomposition returned DIFFUSE (transition-day share 38.1%, below the 50% threshold), so the standard cost grid is used (no 10 bps stress add). Per analyst direction, this report INCLUDES a COVID-excluded subgroup panel (informational only; gates stand on full-window numbers).

## Summary -- 5-gate verdict

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| 1. PSR(V12c @ 5bps near_close, vs SR=0) | PASS | 0.9645 | > 0.95 |
| 2. DSR(V12c, n_trials=23) | FAIL | 0.8337 | > 0.95 |
| 3. PBO across {V01,V04,V05,V06,V11,V12,V12c} | FAIL | 0.7085 | < 0.5 |
| 4. Lag-degradation (5 bps) | PASS | nc=0.586, lag=0.850, nc-lag=-0.263 | <= max(0.2*|nc|, 0.1) = 0.117 |
| 5a. Sharpe(V12c @ 7.5bps lag) > 0.30 | PASS | 0.7762 | > 0.3 |
| 5b. Sharpe(V12c) >= 0.9 * Sharpe(V11) @ 7.5bps lag | PASS | V12c=0.7762, 0.9*V11=0.4776 (V11=0.5306) | >= 0.9 * V11 |

**Overall tier**: TIER 4 -- one or more structural/cost gates failed; do not advance

## PSR / DSR detail

_Units note: PSR and DSR formulas (Bailey-Lopez de Prado / methodology Section 2.2-2.3) require **per-period (daily)** Sharpe with daily `n`. Annualized values are reported for human narrative only._

| Metric | Daily (formula input) | Annualized (narrative) |
|---|---:|---:|
| Observed Sharpe (V12c) | 0.036941 | 0.5864 |
| Expected max under null (n_trials=23) | 0.017115 | 0.2717 |

| Metric | Value |
|---|---:|
| PSR (vs SR=0) | 0.9645 |
| DSR probability (true SR > expected max) | 0.8337 |
| Trial Sharpes (annualized) | V01=0.282, V04=0.313, V05=0.503, V06=0.278, V11=0.528, V12=0.268, V12c=0.586 |
| sqrt(V[trial_sharpes]) (daily) | 0.008726 |
| Sample skewness | 0.4660 |
| Sample Pearson kurtosis | 9.7723 |
| Sample size (days) | 2355 |

## Cost grid (V12c)

| Cost bps | Mode | Sharpe | CAGR |
|---:|:--|---:|---:|
| 1.0 | near_close | 0.7736 | 13.82% |
| 1.0 | one_day_lag | 0.9519 | 22.80% |
| 5.0 | near_close | 0.5863 | 9.83% |
| 5.0 | one_day_lag | 0.8498 | 19.58% |
| 7.5 | near_close | 0.4770 | 7.56% |
| 7.5 | one_day_lag | 0.7762 | 17.44% |
| 10.0 | near_close | 0.3462 | 4.90% |
| 10.0 | one_day_lag | 0.7022 | 15.42% |

## Cross-variants comparison (5 bps near_close, 7 variants)

| Variant | Sharpe | CAGR |
|---|---:|---:|
| V01 | 0.2818 | 3.74% |
| V04 | 0.3133 | 4.89% |
| V05 | 0.5030 | 11.08% |
| V06 | 0.2781 | 3.62% |
| V11 | 0.5279 | 11.93% |
| V12 | 0.2683 | 3.52% |
| V12c | 0.5863 | 9.83% |

## PBO

- Matrix shape: 2355 x 7 (V01, V04, V05, V06, V11, V12, V12c)
- s (CSCV submatrices): 16
- PBO value: 0.7085
- Interpretation: 0.7085 >= 0.5 = elevated overfitting risk

## Detector-onset alignment panel (BEAR or UNPREDICTABLE)

Per-onset breakdown of V12c cash response. V12c goes to cash on BOTH detector-BEAR and detector-UNPREDICTABLE onsets. `gap_days` = trading-day gap between actual onset and the SPY drawdown trough within [-20d, +30d]; positive = onset late vs trough. `avoided_return` = SPY return during the cash window.

| Onset | Regime | Window | Cash window | Cash days | SPY trough | Gap days | Avoided return |
|---|---|---|---|---:|---|---:|---:|
| 2018-02-05 | UNPREDICTABLE | 2018-01-16 .. 2018-03-07 | 2018-02-05 .. 2018-02-14 | 10 | 2018-02-08 | -3 | 2.09% |
| 2018-02-22 | BEAR | 2018-02-02 .. 2018-03-23 | 2018-02-22 .. 2018-02-22 | 1 | 2018-02-08 | +9 | 0.00% |
| 2018-02-27 | BEAR | 2018-02-07 .. 2018-03-29 | 2018-02-27 .. 2018-03-09 | 11 | 2018-02-08 | +12 | 1.62% |
| 2018-04-06 | BEAR | 2018-03-19 .. 2018-05-04 | 2018-04-06 .. 2018-04-25 | 20 | 2018-04-02 | +4 | 1.45% |
| 2018-04-30 | BEAR | 2018-04-10 .. 2018-05-30 | 2018-04-30 .. 2018-05-02 | 3 | 2018-05-03 | -3 | -0.47% |
| 2018-06-25 | BEAR | 2018-06-05 .. 2018-07-25 | 2018-06-25 .. 2018-06-25 | 1 | 2018-06-27 | -2 | 0.00% |
| 2018-06-27 | BEAR | 2018-06-07 .. 2018-07-27 | 2018-06-27 .. 2018-06-29 | 3 | 2018-06-27 | +0 | 0.71% |
| 2018-10-10 | UNPREDICTABLE | 2018-09-20 .. 2018-11-09 | 2018-10-10 .. 2018-10-11 | 2 | 2018-10-29 | -13 | -2.13% |
| 2018-10-17 | BEAR | 2018-09-27 .. 2018-11-16 | 2018-10-17 .. 2018-11-29 | 44 | 2018-10-29 | -8 | -2.30% |
| 2018-12-10 | BEAR | 2018-11-20 .. 2019-01-09 | 2018-12-10 .. 2019-01-16 | 38 | 2018-12-24 | -10 | -1.19% |
| 2019-01-22 | BEAR | 2019-01-02 .. 2019-02-21 | 2019-01-22 .. 2019-01-23 | 2 | 2019-01-03 | +12 | 0.14% |
| 2019-05-13 | BEAR | 2019-04-23 .. 2019-06-12 | 2019-05-13 .. 2019-05-14 | 2 | 2019-06-03 | -14 | 0.84% |
| 2019-05-29 | BEAR | 2019-05-09 .. 2019-06-28 | 2019-05-29 .. 2019-05-29 | 1 | 2019-06-03 | -3 | 0.00% |
| 2019-05-31 | BEAR | 2019-05-13 .. 2019-06-28 | 2019-05-31 .. 2019-06-04 | 5 | 2019-06-03 | -1 | 1.87% |
| 2019-08-05 | UNPREDICTABLE | 2019-07-16 .. 2019-09-04 | 2019-08-05 .. 2019-08-06 | 2 | 2019-08-14 | -7 | 1.31% |
| 2019-08-23 | BEAR | 2019-08-05 .. 2019-09-20 | 2019-08-23 .. 2019-08-28 | 6 | 2019-08-14 | +7 | 1.41% |
| 2019-08-30 | BEAR | 2019-08-12 .. 2019-09-27 | 2019-08-30 .. 2019-09-03 | 5 | 2019-08-14 | +12 | -0.58% |
| 2019-10-02 | BEAR | 2019-09-12 .. 2019-11-01 | 2019-10-02 .. 2019-10-03 | 2 | 2019-10-02 | +0 | 0.81% |
| 2019-10-08 | BEAR | 2019-09-18 .. 2019-11-07 | 2019-10-08 .. 2019-10-08 | 1 | 2019-10-02 | +4 | 0.00% |
| 2020-02-24 | UNPREDICTABLE | 2020-02-04 .. 2020-03-25 | 2020-02-24 .. 2020-04-21 | 58 | 2020-03-23 | -20 | -15.32% |
| 2020-05-13 | BEAR | 2020-04-23 .. 2020-06-12 | 2020-05-13 .. 2020-05-14 | 2 | 2020-04-23 | +14 | 1.20% |
| 2020-06-11 | BEAR | 2020-05-22 .. 2020-07-10 | 2020-06-11 .. 2020-06-11 | 1 | 2020-05-22 | +13 | 0.00% |
| 2020-06-26 | BEAR | 2020-06-08 .. 2020-07-24 | 2020-06-26 .. 2020-06-26 | 1 | 2020-06-26 | +0 | 0.00% |
| 2020-10-26 | BEAR | 2020-10-06 .. 2020-11-25 | 2020-10-26 .. 2020-11-03 | 9 | 2020-10-30 | -4 | -1.01% |
| 2021-01-27 | UNPREDICTABLE | 2021-01-07 .. 2021-02-26 | 2021-01-27 .. 2021-01-27 | 1 | 2021-01-29 | -2 | 0.00% |
| 2021-01-29 | BEAR | 2021-01-11 .. 2021-02-26 | 2021-01-29 .. 2021-01-29 | 1 | 2021-01-29 | +0 | 0.00% |
| 2021-09-20 | BEAR | 2021-08-31 .. 2021-10-20 | 2021-09-20 .. 2021-09-21 | 2 | 2021-10-04 | -10 | -0.09% |
| 2021-10-04 | BEAR | 2021-09-14 .. 2021-11-03 | 2021-10-04 .. 2021-10-04 | 1 | 2021-10-04 | +0 | 0.00% |
| 2021-11-26 | UNPREDICTABLE | 2021-11-08 .. 2021-12-23 | 2021-11-26 .. 2021-11-26 | 1 | 2021-12-01 | -3 | 0.00% |
| 2021-12-01 | UNPREDICTABLE | 2021-11-11 .. 2021-12-31 | 2021-12-01 .. 2021-12-01 | 1 | 2021-12-01 | +0 | 0.00% |
| 2022-01-18 | BEAR | 2021-12-29 .. 2022-02-17 | 2022-01-18 .. 2022-01-20 | 3 | 2022-01-27 | -7 | -2.13% |
| 2022-01-27 | BEAR | 2022-01-07 .. 2022-02-25 | 2022-01-27 .. 2022-02-08 | 13 | 2022-02-23 | -18 | 4.55% |
| 2022-02-10 | BEAR | 2022-01-21 .. 2022-03-11 | 2022-02-10 .. 2022-03-24 | 43 | 2022-03-08 | -17 | 0.23% |
| 2022-04-11 | BEAR | 2022-03-22 .. 2022-05-11 | 2022-04-11 .. 2022-04-18 | 8 | 2022-05-11 | -21 | -0.45% |
| 2022-04-21 | BEAR | 2022-04-01 .. 2022-05-20 | 2022-04-21 .. 2022-04-26 | 6 | 2022-05-19 | -20 | -5.01% |
| 2022-05-02 | BEAR | 2022-04-12 .. 2022-06-01 | 2022-05-02 .. 2022-07-18 | 78 | 2022-05-19 | -13 | -7.84% |
| 2022-09-06 | BEAR | 2022-08-17 .. 2022-10-06 | 2022-09-06 .. 2022-09-06 | 1 | 2022-09-30 | -18 | 0.00% |
| 2022-09-15 | BEAR | 2022-08-26 .. 2022-10-14 | 2022-09-15 .. 2022-10-27 | 43 | 2022-10-12 | -19 | -2.61% |
| 2023-01-04 | BEAR | 2022-12-15 .. 2023-02-03 | 2023-01-04 .. 2023-01-05 | 2 | 2022-12-28 | +4 | -1.13% |
| 2023-03-15 | BEAR | 2023-02-23 .. 2023-04-14 | 2023-03-15 .. 2023-03-15 | 1 | 2023-03-13 | +2 | 0.00% |
| 2023-03-17 | BEAR | 2023-02-27 .. 2023-04-14 | 2023-03-17 .. 2023-03-17 | 1 | 2023-03-13 | +4 | 0.00% |
| 2023-10-19 | BEAR | 2023-09-29 .. 2023-11-17 | 2023-10-19 .. 2023-10-23 | 5 | 2023-10-27 | -6 | -1.40% |
| 2023-10-25 | BEAR | 2023-10-05 .. 2023-11-24 | 2023-10-25 .. 2023-10-30 | 6 | 2023-10-27 | -2 | -0.49% |
| 2024-04-15 | BEAR | 2024-03-26 .. 2024-05-15 | 2024-04-15 .. 2024-04-16 | 2 | 2024-04-19 | -4 | -0.20% |
| 2024-07-24 | BEAR | 2024-07-05 .. 2024-08-23 | 2024-07-24 .. 2024-07-25 | 2 | 2024-08-05 | -8 | -0.53% |
| 2024-07-30 | BEAR | 2024-07-10 .. 2024-08-29 | 2024-07-30 .. 2024-07-30 | 1 | 2024-08-05 | -4 | 0.00% |
| 2024-08-01 | BEAR | 2024-07-12 .. 2024-08-30 | 2024-08-01 .. 2024-08-07 | 7 | 2024-08-05 | -2 | -4.49% |
| 2024-08-09 | BEAR | 2024-07-22 .. 2024-09-06 | 2024-08-09 .. 2024-08-14 | 6 | 2024-08-05 | +4 | 2.04% |
| 2024-08-20 | BEAR | 2024-07-31 .. 2024-09-19 | 2024-08-20 .. 2024-08-21 | 2 | 2024-08-05 | +11 | 0.37% |
| 2024-09-06 | BEAR | 2024-08-19 .. 2024-10-04 | 2024-09-06 .. 2024-09-10 | 5 | 2024-09-06 | +0 | 1.55% |
| 2024-12-18 | UNPREDICTABLE | 2024-11-29 .. 2025-01-17 | 2024-12-18 .. 2024-12-20 | 3 | 2025-01-10 | -14 | 0.78% |
| 2024-12-30 | BEAR | 2024-12-10 .. 2025-01-29 | 2024-12-30 .. 2024-12-30 | 1 | 2025-01-10 | -7 | 0.00% |
| 2025-01-13 | BEAR | 2024-12-24 .. 2025-02-12 | 2025-01-13 .. 2025-01-14 | 2 | 2025-01-10 | +1 | 0.11% |
| 2025-02-24 | BEAR | 2025-02-04 .. 2025-03-26 | 2025-02-24 .. 2025-02-28 | 5 | 2025-03-13 | -13 | -0.52% |
| 2025-03-11 | BEAR | 2025-02-19 .. 2025-04-10 | 2025-03-11 .. 2025-03-21 | 11 | 2025-04-08 | -20 | 1.45% |
| 2025-03-26 | BEAR | 2025-03-06 .. 2025-04-25 | 2025-03-26 .. 2025-05-06 | 42 | 2025-04-08 | -9 | -1.75% |
| 2025-11-17 | BEAR | 2025-10-28 .. 2025-12-17 | 2025-11-17 .. 2025-11-20 | 4 | 2025-11-20 | -3 | -1.98% |
| 2026-01-20 | BEAR | 2025-12-31 .. 2026-02-19 | 2026-01-20 .. 2026-01-20 | 1 | 2026-02-05 | -12 | 0.00% |
| 2026-03-12 | BEAR | 2026-02-20 .. 2026-04-10 | 2026-03-12 .. 2026-04-08 | 28 | 2026-03-30 | -12 | 1.47% |

**Aggregate gap days (mean)**: -4.05
**Aggregate avoided return (mean)**: -0.47%
**Onset count**: 59

## Sensitivity appendix -- COVID-excluded subgroup (E2 robustness)

_Per analyst direction following E2 verdict AMBIGUOUS (53.6% attribution in top-3 events, COVID-dominant). This panel is INFORMATIONAL ONLY and does NOT influence the gate verdict per spec rev4 honesty discipline; the gates stand on the full-window numbers._

COVID exclusion window: 2020-02-24 .. 2020-04-30 (inclusive).

| Metric | Full window | COVID-excluded | Delta |
|---|---:|---:|---:|
| Sharpe (V12c @ 5bps near_close) | 0.5863 | 0.5714 | -0.0149 |
| CAGR | 9.83% | 9.53% | -0.30pp |
| Sample days | 2355 | 2307 | -48 |

**Note**: Sharpe shift under COVID exclusion is small (-0.0149, 2.5% of full-window magnitude). V12c edge is not concentrated in the COVID event.

## Methodology decisions

- n_trials_project = 23 (hard-coded 23: V12 used 22 (4 from experiments.duckdb + 18 V12 runs); V12c is trial #23, V12-up-cash sensitivity now formalized as its own gate)
- PBO s = 16 (methodology Section 2.4 default)
- Cost tier for PSR/DSR/PBO: 5.0 bps per side
- one_day_lag definition: signal computed at close T from `panel.loc[:T]`, trades executed at close T+1, MTM at close T+1.
- All metrics computed on net-of-cost daily returns.
- Variants run at full window (start..end). Phase 4 is single-pass, not walk-forward.
- Gate 4 (rev4, directional): `(nc - lag) <= max(0.2 * |nc|, 0.1)` -- lag > near_close is the safe direction and is not penalized.
- Gate 5 (rev4-followup): both clauses required: Sharpe(V12c @ 7.5bps lag) > 0.30 AND >= 0.9 * Sharpe(V11 @ 7.5bps lag).
- Tier classification: TIER 1 (all 5 pass) / TIER 3 (structural pass, PSR+DSR fail) / TIER 4 (any structural fail).
- COVID exclusion is post-hoc filter of the V12c@5bps-near_close record stream (NOT a fresh backtest, so no PSR/DSR distortion).

## Metadata

- Git SHA: d0b3464
- Run datetime: 2026-05-24T03:10:37
- n_trials_project source: hard-coded 23: V12 used 22 (4 from experiments.duckdb + 18 V12 runs); V12c is trial #23, V12-up-cash sensitivity now formalized as its own gate
- V11 reference (Gate 5): inline re-run @ 7.5bps one_day_lag = 0.5306; V11-readiness-doc value = 0.5306
- Total gate-influencing unique runs: 15 (8 V12c cost grid + 6 cross [V12c reused] + 1 V11 ref)
- Sensitivity panels: 1 (COVID-excluded subgroup, post-hoc filter only)
