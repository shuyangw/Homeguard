# V13-bear-invert Readiness Report (Experiment 1)

**Code commit**: d0b3464
**Data**: Alpaca SIP daily-aggregated, 2017-01-01 to 2026-05-16
**Universe**: config\universes\sp500-2025.csv
**Cost tier for PSR/DSR/PBO gates**: 5.0 bps per side
**n_trials_project**: 23 (hard-coded: 22 (V12 readiness 2026-05-24, 4 from experiments.duckdb + 18 from V12 readiness run) + 1 (V13 introduction))

> **Honesty discipline**: V13-bear-invert was discovered from inspection of V12's onset-alignment panel (mean gap_days = -3.42 across 59 events 2017-2026 -- the detector fires AFTER the SPY drawdown trough). V13 inverts the sign of V12. This was discovered from EXT-OOS data and is **NOT OOS in the strict sense**; the same 2017-2026 window that motivated the hypothesis is the test window. Forward OOS validation is required before any paper deploy regardless of verdict. The DSR n_trials_project counter has been incremented (22 -> 23) to reflect V13's introduction.

## Summary

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| 1. PSR(V13 @ 5bps near_close, vs SR=0) | FAIL | 0.8830 | > 0.95 |
| 2. DSR(V13, n_trials=23) | FAIL | 0.7074 | > 0.95 |
| 3. PBO across 7 variants | FAIL | 0.6290 | < 0.5 |
| 4. Lag-degradation (5 bps, directional) | PASS | nc=0.400, lag=0.381, nc-lag=+0.019 | <= max(0.2*|nc|, 0.1) = 0.100 |
| 5a. Sharpe(V13 @ 7.5bps lag) > 0.30 | PASS | 0.3080 | > 0.3 |
| 5b. Sharpe(V13) >= 0.9 * Sharpe(V11) @ 7.5bps lag | FAIL | V13=0.3080, 0.9*V11=0.4776 (V11=0.5306) | >= 0.9 * V11 |

### V13 vs V11 vs V12 -- direct head-to-head at 5 bps near_close

| Variant | Sharpe @ 5bps near_close | Delta vs V11 |
|---|---:|---:|
| V11 (inline, baseline) | 0.5279 | 0.0000 |
| V12 (BEAR -> cash, inline) | 0.2683 | -0.2596 |
| **V13 (BEAR -> SPY 100%)** | **0.4002** | **-0.1277** |

TIER 1 lift threshold (V13 vs V11 @ 5bps near_close): +0.10 required, observed -0.1278 (vs documentary V11=0.528). Lift gate: FAIL.

**Overall verdict**: TIER 4 -- BEAR-as-buy is spurious -- one or more structural gates FAIL. Close V13; continue WS-3c roadmap.

## PSR / DSR detail

_Units note: PSR and DSR formulas (Bailey-Lopez de Prado / methodology Section 2.2-2.3) require **per-period (daily)** Sharpe with daily `n`. Annualized values are reported for human narrative only._

| Metric | Daily (formula input) | Annualized (narrative) |
|---|---:|---:|
| Observed Sharpe (V13) | 0.025216 | 0.4003 |
| Expected max under null (n_trials=23) | 0.013654 | 0.2168 |

| Metric | Value |
|---|---:|
| PSR (vs SR=0) | 0.8830 |
| DSR probability (true SR > expected max) | 0.7074 |
| Trial Sharpes (annualized) | V01=0.282, V04=0.313, V05=0.503, V06=0.278, V11=0.528, V12=0.268, V13-bear-invert=0.400 |
| sqrt(V[trial_sharpes]) (daily) | 0.006961 |
| Sample skewness | -1.9256 |
| Sample Pearson kurtosis | 50.0106 |
| Sample size (days) | 2355 |

## Cost grid (V13-bear-invert)

| Cost bps | Mode | Sharpe | CAGR |
|---:|:--|---:|---:|
| 1.0 | near_close | 0.5475 | 11.92% |
| 1.0 | one_day_lag | 0.5243 | 10.86% |
| 5.0 | near_close | 0.4002 | 7.41% |
| 5.0 | one_day_lag | 0.3812 | 6.73% |
| 7.5 | near_close | 0.3091 | 4.71% |
| 7.5 | one_day_lag | 0.3080 | 4.68% |
| 10.0 | near_close | 0.2110 | 1.89% |
| 10.0 | one_day_lag | 0.2200 | 2.26% |

## Cross-variants comparison (5 bps near_close)

| Variant | Sharpe | CAGR |
|---|---:|---:|
| V01 | 0.2818 | 3.74% |
| V04 | 0.3133 | 4.89% |
| V05 | 0.5030 | 11.08% |
| V06 | 0.2781 | 3.62% |
| V11 | 0.5279 | 11.93% |
| V12 | 0.2683 | 3.52% |
| V13-bear-invert | 0.4002 | 7.41% |

Documentary references (no re-run): V11 @ 5bps near_close = 0.528, V11 @ 5bps lag = 0.580, V12 @ 5bps near_close = 0.268, V12 @ 5bps lag = 0.665.

## PBO

- Matrix shape: 2355 x 7 (V01, V04, V05, V06, V11, V12, V13-bear-invert)
- s (CSCV submatrices): 16
- PBO value: 0.6290
- Interpretation: 0.6290 >= 0.5 = high overfitting evidence
- **Methodology decision (PBO scope)**: V13 PBO is computed across 7 variants (V01, V04, V05, V06, V11, V12, V13) -- V12 included so V12 vs V13 is a direct PBO neighbor (the sign-inversion test motivating V13 is most informative when V12 is in the matrix). V12 readiness ran across 6 variants (no V13). This is a deliberate expansion, documented here for reproducibility.

## Detector-onset alignment panel (V13 SPY-holding response)

Per-onset breakdown of V13 SPY-holding response. `gap_days` = trading-day gap between actual detector BEAR onset and the SPY drawdown trough within [-20d, +30d]; positive = detector late vs trough. `spy_return` = SPY return during the contiguous BEAR window (V13 holds 100% SPY here).

| Onset | Window | BEAR window | BEAR days | SPY trough | Gap days | SPY return |
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
**Aggregate SPY return during BEAR window (mean)**: 0.18%

Interpretation: if mean gap_days is negative (detector late vs trough) AND mean SPY return during the BEAR window is positive, the BEAR-as-buy hypothesis is empirically supported on this sample. Sign of mean SPY return is the headline observable; magnitude feeds the gate Sharpe via daily returns.

## Limitations and honesty discipline

- **NOT OOS in strict sense.** V13 was generated from inspection of V12's 2017-2026 BEAR onset panel. The same window is now the test window. PSR/DSR partially correct for this via n_trials_project=23, but the correction is not perfect; the only definitive check is forward OOS data.
- **Single-name concentration risk.** V13 collapses to 100% SPY on BEAR days. This is concentrated single-name risk that V11/V12 do not carry. Position sizing risk gates (production strategy framework) would need to relax for V13 deploy.
- **Detector lag is structural, not random.** The gap_days mean is driven by SPY-DD / VIX-percentile / momentum-slope thresholds in the detector. If WS-3 improves the detector (earlier BEAR firing), V13 edge would shrink. V13 is conditional on the current detector spec.
- **No sensitivity appendix.** V13 has no UNPREDICTABLE-cash or debouncing analog (BEAR is the only branch that differs from V11), so the V12 readiness sensitivity slice doesn't translate.

## Methodology decisions

- n_trials_project = 23 (hard-coded: 22 (V12 readiness 2026-05-24, 4 from experiments.duckdb + 18 from V12 readiness run) + 1 (V13 introduction))
- PBO s = 16 (methodology Section 2.4 default)
- PBO matrix variants: V01, V04, V05, V06, V11, V12, V13-bear-invert (7-variant; see PBO section for rationale).
- Cost tier for PSR/DSR/PBO: 5.0 bps per side
- one_day_lag definition: signal computed at close T from `panel.loc[:T]`, trades executed at close T+1, MTM at close T+1.
- All metrics computed on net-of-cost daily returns.
- Variants run at full window (start..end). Phase 4 is single-pass, not walk-forward.
- Gate 4 (rev4 directional): `(nc - lag) <= max(0.2 * |nc|, 0.1)` -- lag > near_close is the safe direction and is not penalized.
- Gate 5 (rev4-followup): both clauses required: Sharpe(V13 @ 7.5bps lag) > 0.30 AND >= 0.9 * Sharpe(V11 @ 7.5bps lag).
- TIER 1 lift gate: Sharpe(V13 @ 5bps nc) > Sharpe(V11 @ 5bps nc) + 0.10. Required in addition to passing all 5 gates for TIER 1 verdict (per V13 spec).

## Metadata

- Git SHA: d0b3464
- Run datetime: 2026-05-24T03:14:40
- n_trials_project source: hard-coded: 22 (V12 readiness 2026-05-24, 4 from experiments.duckdb + 18 from V12 readiness run) + 1 (V13 introduction)
- V11 reference (Gate 5): inline re-run @ 7.5bps one_day_lag = 0.5306; V11-readiness-doc value = 0.5306
- Total gate-influencing runs: 15 (8 V13 cost grid + 6 cross-variants + 1 V11 ref). No sensitivity appendix runs.
