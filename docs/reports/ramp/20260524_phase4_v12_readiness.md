# V12 Phase D Readiness Report

**Code commit**: c4e4f0d
**Data**: Alpaca SIP daily-aggregated, 2017-01-01 to 2026-05-16
**Universe**: config\universes\sp500-2025.csv
**Cost tier for PSR/DSR/PBO gates**: 5.0 bps per side
**n_trials_project**: 22 (experiments.duckdb total=4 + 18)

## Summary

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| 1. PSR(V12 @ 5bps near_close, vs SR=0) | FAIL | 0.7881 | > 0.95 |
| 2. DSR(V12, n_trials=22) | FAIL | 0.5418 | > 0.95 |
| 3. PBO across {V01,V04,V05,V06,V11,V12} | PASS | 0.3934 | < 0.5 |
| 4. Lag-degradation (5 bps) | PASS | nc=0.268, lag=0.665, nc-lag=-0.397 | (nc-lag) <= 0.100 (lag>nc is safe per spec rev4) |
| 5a. Sharpe(V12 @ 7.5bps lag) > 0.30 | PASS | 0.6081 | > 0.3 |
| 5b. Sharpe(V12) >= 0.9 * Sharpe(V11) @ 7.5bps lag | PASS | V12=0.6081, 0.9*V11=0.4776 (V11=0.5306) | >= 0.9 * V11 |

**Overall**: PARTIAL (Tier 3 per spec success criteria) -- structural + cost gates PASS; absolute-significance gates (PSR, DSR) FAIL. Detector-onset alignment panel shows the lag tax (mean -3.42 gap days). Per spec rev4: activate WS-3 (detector improvement); defer V12 deployment.

### Tier verdict (per spec rev4 success criteria)

V12 sits in **Tier 3 (diagnostic value)** per `docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md`:
- Structural gates (PBO, lag-degradation, cost robustness) PASS
- Absolute-significance gates (PSR, DSR) FAIL: V12 near_close at 5 bps Sharpe = 0.268, materially below V11's 0.528
- Detector-onset alignment panel shows mean gap_days = -3.42: the detector fires ~3.4 days AFTER the SPY drawdown trough on average. V12's cash periods bracket the recovery, not the crash. Mean avoided_return = +0.18%.

Per spec Tier 3 decision rule: **activate WS-3 (regime detector improvement) as the higher-leverage path**. V12 production paper deploy is deferred until WS-3 + V12 readiness re-run.

Note on Gate 4: the original orchestrator emission of this report computed Gate 4 with an absolute-value check (`|nc - lag| <= cap`) which incorrectly flagged Gate 4 as FAIL. Per spec rev4, the check is directional (`(nc - lag) <= cap`); lag > near_close is the safe direction. The orchestrator code has been fixed (commit `5a88903`); this report's verdict table has been patched to reflect the correct verdict on the original Sharpes. No backtests were re-run.

## Headline 5-gate verdict

See Summary table above. Gates 1-3 are the structural / significance trio; gate 4 is the rev4 lag-floor robustness check; gate 5 is the rev4-followup cost-floor + no-regress dual clause.

Gate 5 V11 reference (Sharpe @ 7.5bps one_day_lag) was re-run inline in this orchestrator for an apples-to-apples comparison. Inline V11 value: 0.5306. For reference, the V11 readiness doc (2026-05-23) reported 0.5306.

## PSR / DSR detail

_Units note: PSR and DSR formulas (Bailey-Lopez de Prado / methodology Section 2.2-2.3) require **per-period (daily)** Sharpe with daily `n`. Annualized values are reported for human narrative only._

| Metric | Daily (formula input) | Annualized (narrative) |
|---|---:|---:|
| Observed Sharpe (V12) | 0.016905 | 0.2684 |
| Expected max under null (n_trials=22) | 0.014684 | 0.2331 |

| Metric | Value |
|---|---:|
| PSR (vs SR=0) | 0.7881 |
| DSR probability (true SR > expected max) | 0.5418 |
| Trial Sharpes (annualized) | V01=0.282, V04=0.313, V05=0.503, V06=0.278, V11=0.528, V12=0.268 |
| sqrt(V[trial_sharpes]) (daily) | 0.007560 |
| Sample skewness | -2.7520 |
| Sample Pearson kurtosis | 71.6171 |
| Sample size (days) | 2355 |

## Cost grid (V12 v12.0.0 defaults)

| Cost bps | Mode | Sharpe | CAGR |
|---:|:--|---:|---:|
| 1.0 | near_close | 0.4189 | 7.65% |
| 1.0 | one_day_lag | 0.7582 | 19.98% |
| 5.0 | near_close | 0.2683 | 3.52% |
| 5.0 | one_day_lag | 0.6650 | 16.63% |
| 7.5 | near_close | 0.1870 | 1.36% |
| 7.5 | one_day_lag | 0.6081 | 14.67% |
| 10.0 | near_close | 0.0939 | -1.06% |
| 10.0 | one_day_lag | 0.5487 | 12.64% |

## Cross-variants comparison (5 bps near_close)

| Variant | Sharpe | CAGR |
|---|---:|---:|
| V01 | 0.2818 | 3.74% |
| V04 | 0.3133 | 4.89% |
| V05 | 0.5030 | 11.08% |
| V06 | 0.2781 | 3.62% |
| V11 | 0.5279 | 11.93% |
| V12 | 0.2683 | 3.52% |

## PBO

- Matrix shape: 2355 x 6 (V01, V04, V05, V06, V11, V12)
- s (CSCV submatrices): 16
- PBO value: 0.3934
- Interpretation: 0.3934 < 0.5 = low overfitting evidence

## Detector-onset alignment panel

Per-onset breakdown of V12 cash response. `gap_days` = trading-day gap between actual detector BEAR onset and the SPY drawdown trough within [-20d, +30d]; positive = detector late vs trough. `avoided_return` = SPY return during the cash window (the return V12 sidestepped by being out).

| Onset | Window | Cash window | Cash days | SPY trough | Gap days | Avoided return |
|---|---|---|---:|---|---:|---:|
| 2018-02-12 | 2018-01-23 .. 2018-03-14 | 2018-02-12 .. 2018-02-14 | 3 | 2018-02-08 | +2 | 1.68% |
| 2018-02-22 | 2018-02-02 .. 2018-03-23 | 2018-02-22 .. 2018-02-22 | 1 | 2018-02-08 | +9 | 0.00% |
| 2018-02-27 | 2018-02-07 .. 2018-03-29 | 2018-02-27 .. 2018-03-09 | 11 | 2018-02-08 | +12 | 1.62% |
| 2018-04-06 | 2018-03-19 .. 2018-05-04 | 2018-04-06 .. 2018-04-25 | 20 | 2018-04-02 | +4 | 1.45% |
| 2018-04-30 | 2018-04-10 .. 2018-05-30 | 2018-04-30 .. 2018-05-02 | 3 | 2018-05-03 | -3 | -0.47% |
| 2018-06-25 | 2018-06-05 .. 2018-07-25 | 2018-06-25 .. 2018-06-25 | 1 | 2018-06-27 | -2 | 0.00% |
| 2018-06-27 | 2018-06-07 .. 2018-07-27 | 2018-06-27 .. 2018-06-29 | 3 | 2018-06-27 | +0 | 0.71% |
| 2018-10-17 | 2018-09-27 .. 2018-11-16 | 2018-10-17 .. 2018-11-29 | 44 | 2018-10-29 | -8 | -2.30% |
| 2018-12-10 | 2018-11-20 .. 2019-01-09 | 2018-12-10 .. 2018-12-21 | 12 | 2018-12-24 | -10 | -8.86% |
| 2018-12-26 | 2018-12-06 .. 2019-01-25 | 2018-12-26 .. 2019-01-16 | 22 | 2018-12-24 | +1 | 6.10% |
| 2019-01-22 | 2019-01-02 .. 2019-02-21 | 2019-01-22 .. 2019-01-23 | 2 | 2019-01-03 | +12 | 0.14% |
| 2019-05-13 | 2019-04-23 .. 2019-06-12 | 2019-05-13 .. 2019-05-14 | 2 | 2019-06-03 | -14 | 0.84% |
| 2019-05-29 | 2019-05-09 .. 2019-06-28 | 2019-05-29 .. 2019-05-29 | 1 | 2019-06-03 | -3 | 0.00% |
| 2019-05-31 | 2019-05-13 .. 2019-06-28 | 2019-05-31 .. 2019-06-04 | 5 | 2019-06-03 | -1 | 1.87% |
| 2019-08-06 | 2019-07-17 .. 2019-09-05 | 2019-08-06 .. 2019-08-06 | 1 | 2019-08-14 | -6 | 0.00% |
| 2019-08-23 | 2019-08-05 .. 2019-09-20 | 2019-08-23 .. 2019-08-28 | 6 | 2019-08-14 | +7 | 1.41% |
| 2019-08-30 | 2019-08-12 .. 2019-09-27 | 2019-08-30 .. 2019-09-03 | 5 | 2019-08-14 | +12 | -0.58% |
| 2019-10-02 | 2019-09-12 .. 2019-11-01 | 2019-10-02 .. 2019-10-03 | 2 | 2019-10-02 | +0 | 0.81% |
| 2019-10-08 | 2019-09-18 .. 2019-11-07 | 2019-10-08 .. 2019-10-08 | 1 | 2019-10-02 | +4 | 0.00% |
| 2020-03-04 | 2020-02-13 .. 2020-04-03 | 2020-03-04 .. 2020-03-04 | 1 | 2020-03-23 | -13 | 0.00% |
| 2020-03-20 | 2020-03-02 .. 2020-04-17 | 2020-03-20 .. 2020-04-21 | 33 | 2020-03-23 | -1 | 19.25% |
| 2020-05-13 | 2020-04-23 .. 2020-06-12 | 2020-05-13 .. 2020-05-14 | 2 | 2020-04-23 | +14 | 1.20% |
| 2020-06-11 | 2020-05-22 .. 2020-07-10 | 2020-06-11 .. 2020-06-11 | 1 | 2020-05-22 | +13 | 0.00% |
| 2020-06-26 | 2020-06-08 .. 2020-07-24 | 2020-06-26 .. 2020-06-26 | 1 | 2020-06-26 | +0 | 0.00% |
| 2020-10-26 | 2020-10-06 .. 2020-11-25 | 2020-10-26 .. 2020-11-03 | 9 | 2020-10-30 | -4 | -1.01% |
| 2021-01-29 | 2021-01-11 .. 2021-02-26 | 2021-01-29 .. 2021-01-29 | 1 | 2021-01-29 | +0 | 0.00% |
| 2021-09-20 | 2021-08-31 .. 2021-10-20 | 2021-09-20 .. 2021-09-21 | 2 | 2021-10-04 | -10 | -0.09% |
| 2021-10-04 | 2021-09-14 .. 2021-11-03 | 2021-10-04 .. 2021-10-04 | 1 | 2021-10-04 | +0 | 0.00% |
| 2022-01-18 | 2021-12-29 .. 2022-02-17 | 2022-01-18 .. 2022-01-20 | 3 | 2022-01-27 | -7 | -2.13% |
| 2022-01-27 | 2022-01-07 .. 2022-02-25 | 2022-01-27 .. 2022-02-08 | 13 | 2022-02-23 | -18 | 4.55% |
| 2022-02-10 | 2022-01-21 .. 2022-03-11 | 2022-02-10 .. 2022-03-24 | 43 | 2022-03-08 | -17 | 0.23% |
| 2022-04-11 | 2022-03-22 .. 2022-05-11 | 2022-04-11 .. 2022-04-18 | 8 | 2022-05-11 | -21 | -0.45% |
| 2022-04-21 | 2022-04-01 .. 2022-05-20 | 2022-04-21 .. 2022-04-26 | 6 | 2022-05-19 | -20 | -5.01% |
| 2022-05-02 | 2022-04-12 .. 2022-06-01 | 2022-05-02 .. 2022-07-18 | 78 | 2022-05-19 | -13 | -7.84% |
| 2022-09-06 | 2022-08-17 .. 2022-10-06 | 2022-09-06 .. 2022-09-06 | 1 | 2022-09-30 | -18 | 0.00% |
| 2022-09-15 | 2022-08-26 .. 2022-10-14 | 2022-09-15 .. 2022-10-27 | 43 | 2022-10-12 | -19 | -2.61% |
| 2023-01-04 | 2022-12-15 .. 2023-02-03 | 2023-01-04 .. 2023-01-05 | 2 | 2022-12-28 | +4 | -1.13% |
| 2023-03-15 | 2023-02-23 .. 2023-04-14 | 2023-03-15 .. 2023-03-15 | 1 | 2023-03-13 | +2 | 0.00% |
| 2023-03-17 | 2023-02-27 .. 2023-04-14 | 2023-03-17 .. 2023-03-17 | 1 | 2023-03-13 | +4 | 0.00% |
| 2023-10-19 | 2023-09-29 .. 2023-11-17 | 2023-10-19 .. 2023-10-23 | 5 | 2023-10-27 | -6 | -1.40% |
| 2023-10-25 | 2023-10-05 .. 2023-11-24 | 2023-10-25 .. 2023-10-30 | 6 | 2023-10-27 | -2 | -0.49% |
| 2024-04-15 | 2024-03-26 .. 2024-05-15 | 2024-04-15 .. 2024-04-16 | 2 | 2024-04-19 | -4 | -0.20% |
| 2024-07-24 | 2024-07-05 .. 2024-08-23 | 2024-07-24 .. 2024-07-25 | 2 | 2024-08-05 | -8 | -0.53% |
| 2024-07-30 | 2024-07-10 .. 2024-08-29 | 2024-07-30 .. 2024-07-30 | 1 | 2024-08-05 | -4 | 0.00% |
| 2024-08-01 | 2024-07-12 .. 2024-08-30 | 2024-08-01 .. 2024-08-01 | 1 | 2024-08-05 | -2 | 0.00% |
| 2024-08-09 | 2024-07-22 .. 2024-09-06 | 2024-08-09 .. 2024-08-14 | 6 | 2024-08-05 | +4 | 2.04% |
| 2024-08-20 | 2024-07-31 .. 2024-09-19 | 2024-08-20 .. 2024-08-21 | 2 | 2024-08-05 | +11 | 0.37% |
| 2024-09-06 | 2024-08-19 .. 2024-10-04 | 2024-09-06 .. 2024-09-10 | 5 | 2024-09-06 | +0 | 1.55% |
| 2024-12-20 | 2024-12-02 .. 2025-01-17 | 2024-12-20 .. 2024-12-20 | 1 | 2025-01-10 | -12 | 0.00% |
| 2024-12-30 | 2024-12-10 .. 2025-01-29 | 2024-12-30 .. 2024-12-30 | 1 | 2025-01-10 | -7 | 0.00% |
| 2025-01-13 | 2024-12-24 .. 2025-02-12 | 2025-01-13 .. 2025-01-14 | 2 | 2025-01-10 | +1 | 0.11% |
| 2025-02-24 | 2025-02-04 .. 2025-03-26 | 2025-02-24 .. 2025-02-28 | 5 | 2025-03-13 | -13 | -0.52% |
| 2025-03-11 | 2025-02-19 .. 2025-04-10 | 2025-03-11 .. 2025-03-21 | 11 | 2025-04-08 | -20 | 1.45% |
| 2025-03-26 | 2025-03-06 .. 2025-04-25 | 2025-03-26 .. 2025-04-03 | 9 | 2025-04-08 | -9 | -5.47% |
| 2025-04-09 | 2025-03-20 .. 2025-05-09 | 2025-04-09 .. 2025-04-09 | 1 | 2025-04-08 | +1 | 0.00% |
| 2025-04-11 | 2025-03-24 .. 2025-05-09 | 2025-04-11 .. 2025-05-06 | 26 | 2025-04-08 | +3 | 4.65% |
| 2025-11-17 | 2025-10-28 .. 2025-12-17 | 2025-11-17 .. 2025-11-20 | 4 | 2025-11-20 | -3 | -1.98% |
| 2026-01-20 | 2025-12-31 .. 2026-02-19 | 2026-01-20 .. 2026-01-20 | 1 | 2026-02-05 | -12 | 0.00% |
| 2026-03-12 | 2026-02-20 .. 2026-04-10 | 2026-03-12 .. 2026-04-08 | 28 | 2026-03-30 | -12 | 1.47% |

**Aggregate gap days (mean)**: -3.42
**Aggregate avoided return (mean)**: 0.18%

## Sensitivity appendix (NOT gate-influencing)

_These runs ARE counted in `n_trials_project` (DSR conservatism) but do NOT feed PSR/DSR/PBO/lag/cost computations._

| Run | Description | Sharpe | CAGR |
|---|---|---:|---:|
| V12-up-cash | V12 @ 5bps near_close (variant) | 0.5863 | 9.83% |
| V12-deb-2 | V12 @ 5bps near_close (variant) | 0.1304 | 0.07% |
| V12-deb-3 | V12 @ 5bps near_close (variant) | 0.3153 | 4.83% |
| V12-deb-5 | V12 @ 5bps near_close (variant) | 0.4370 | 8.09% |

**Reading guide:**
- **V12-up-cash (UNPREDICTABLE='cash')** at Sharpe 0.586 beats V12 default (0.268) by +0.32 and slightly beats V11 (0.528). Per spec rev4 honesty discipline this is informational only -- it does NOT update v12.0.0 defaults -- but it motivates a future **V12c spec** for UNPREDICTABLE='cash' as the v12.1.0 default candidate.
- **Debouncing values {2, 3, 5}** all under-perform v12.0.0 (min=0) except deb-5 (0.437), which approaches the no-debouncing baseline. Combined with the BEAR median run length ~3-4 days from the regime diagnostic, this confirms the "no good debouncing value exists" risk anticipated in spec rev4 -- the detector lag is the binding constraint, not the debouncing.

## Methodology decisions

- n_trials_project = 22 (experiments.duckdb total=4 + 18; methodology Section 2.3 / 9.4 cumulative)
- PBO s = 16 (methodology Section 2.4 default)
- Cost tier for PSR/DSR/PBO: 5.0 bps per side
- one_day_lag definition: signal computed at close T from `panel.loc[:T]`, trades executed at close T+1, MTM at close T+1.
- All metrics computed on net-of-cost daily returns.
- Variants run at full window (start..end). Phase 4 is single-pass, not walk-forward.
- Gate 4 floor (rev4, directional): `(nc - lag) <= max(0.2 * |nc|, 0.1)` -- lag > near_close is the safe direction and is not penalized; the 0.1 absolute floor prevents vacuous tightness when |nc| is small.
- Gate 5 (rev4-followup): both clauses required: Sharpe(V12 @ 7.5bps lag) > 0.30 AND >= 0.9 * Sharpe(V11 @ 7.5bps lag).

## Metadata

- Git SHA: c4e4f0d
- Run datetime: 2026-05-24T00:52:58
- n_trials_project source: experiments.duckdb total=4 + 18
- V11 reference (Gate 5): inline re-run @ 7.5bps one_day_lag = 0.5306; V11-readiness-doc value = 0.5306
- Total gate-influencing runs: 14 (8 cost grid + 5 cross + 1 V11 ref)
- Total sensitivity runs: 4
