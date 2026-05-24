# V14 factorial Phase D Readiness Report (Experiment 7)

**Code commit**: 6f55e37
**Data**: Alpaca SIP daily-aggregated, 2017-01-01 to 2026-05-16
**Universe**: config\universes\sp500-2025.csv
**Cost tier for PSR/DSR/PBO gates**: 5.0 bps per side
**n_trials_project**: 36
**Tau constants**: tau_in=0.555556, tau_out=0.455556 (loaded from config/research/v14_tau_constants.json)

**Variant definitions**: V14a/V14b/V14c are all V11 + a Schmitt-trigger BEAR_score consumer, differing in the on-soft-bear action:
- V14a-soft-bear-cash: go to cash
- V14b-soft-bear-spy: go to SPY 100%
- V14c-soft-bear-dampen: scale V11 plan by dampen_factor (default 0.5)

## Summary

| Variant | Tier | Sharpe@5bps_nc | PSR | DSR | PBO | nc-lag | Sharpe@7.5bps_lag |
|---|:---:|---:|---:|---:|---:|---:|---:|
| V14a-soft-bear-cash | TIER 4 | 0.6146 | 0.9703 | 0.8175 | 0.9528 | -0.0775 | 0.6339 |
| V14b-soft-bear-spy | TIER 4 | 0.6035 | 0.9674 | 0.8075 | 0.9528 | -0.0409 | 0.5510 |
| V14c-soft-bear-dampen | TIER 4 | 0.6131 | 0.9695 | 0.8153 | 0.9528 | -0.1301 | 0.6503 |

**Selected for Phase D**: NONE (no V14 variant landed in TIER 1)

_Selection rule when multiple TIER 1 candidates (pre-registered): Sharpe@5bps_nc desc -> lower PBO -> lower DSR penalty -> run-off if still tied._

## 5-gate verdict -- V14a-soft-bear-cash

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| 1. PSR(V14a-soft-bear-cash @ 5bps nc, vs SR=0) | PASS | 0.9703 | > 0.95 |
| 2. DSR(V14a-soft-bear-cash, n_trials=36) | FAIL | 0.8175 | > 0.95 |
| 3. PBO across 8 variants ('V01', 'V11', 'V12', 'V12c-cfg', 'V13-bear-invert', 'V14a-soft-bear-cash', 'V14b-soft-bear-spy', 'V14c-soft-bear-dampen') | FAIL | 0.9528 | < 0.5 |
| 4. Lag-degradation (5 bps) | PASS | nc=0.615, lag=0.692, nc-lag=-0.077 | <= max(0.2*|nc|, 0.1) = 0.123 |
| 5a. Sharpe(V14a-soft-bear-cash @ 7.5bps lag) > 0.30 | PASS | 0.6339 | > 0.3 |
| 5b. Sharpe(V14a-soft-bear-cash) >= 0.9 * Sharpe(V11) @ 7.5bps lag | PASS | V14a-soft-bear-cash=0.6339, 0.9*V11=0.4776 (V11=0.5306) | >= 0.9 * V11 |
| TIER 1 lift: Sharpe(V14a-soft-bear-cash @ 5bps nc) > Sharpe(V11 @ 5bps nc) + 0.10 | FAIL | V14a-soft-bear-cash=0.6146, V11=0.5279, delta=+0.0867 | > 0.1 |

**V14a-soft-bear-cash tier**: TIER 4

## 5-gate verdict -- V14b-soft-bear-spy

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| 1. PSR(V14b-soft-bear-spy @ 5bps nc, vs SR=0) | PASS | 0.9674 | > 0.95 |
| 2. DSR(V14b-soft-bear-spy, n_trials=36) | FAIL | 0.8075 | > 0.95 |
| 3. PBO across 8 variants ('V01', 'V11', 'V12', 'V12c-cfg', 'V13-bear-invert', 'V14a-soft-bear-cash', 'V14b-soft-bear-spy', 'V14c-soft-bear-dampen') | FAIL | 0.9528 | < 0.5 |
| 4. Lag-degradation (5 bps) | PASS | nc=0.604, lag=0.644, nc-lag=-0.041 | <= max(0.2*|nc|, 0.1) = 0.121 |
| 5a. Sharpe(V14b-soft-bear-spy @ 7.5bps lag) > 0.30 | PASS | 0.5510 | > 0.3 |
| 5b. Sharpe(V14b-soft-bear-spy) >= 0.9 * Sharpe(V11) @ 7.5bps lag | PASS | V14b-soft-bear-spy=0.5510, 0.9*V11=0.4776 (V11=0.5306) | >= 0.9 * V11 |
| TIER 1 lift: Sharpe(V14b-soft-bear-spy @ 5bps nc) > Sharpe(V11 @ 5bps nc) + 0.10 | FAIL | V14b-soft-bear-spy=0.6035, V11=0.5279, delta=+0.0757 | > 0.1 |

**V14b-soft-bear-spy tier**: TIER 4

## 5-gate verdict -- V14c-soft-bear-dampen

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| 1. PSR(V14c-soft-bear-dampen @ 5bps nc, vs SR=0) | PASS | 0.9695 | > 0.95 |
| 2. DSR(V14c-soft-bear-dampen, n_trials=36) | FAIL | 0.8153 | > 0.95 |
| 3. PBO across 8 variants ('V01', 'V11', 'V12', 'V12c-cfg', 'V13-bear-invert', 'V14a-soft-bear-cash', 'V14b-soft-bear-spy', 'V14c-soft-bear-dampen') | FAIL | 0.9528 | < 0.5 |
| 4. Lag-degradation (5 bps) | PASS | nc=0.613, lag=0.743, nc-lag=-0.130 | <= max(0.2*|nc|, 0.1) = 0.123 |
| 5a. Sharpe(V14c-soft-bear-dampen @ 7.5bps lag) > 0.30 | PASS | 0.6503 | > 0.3 |
| 5b. Sharpe(V14c-soft-bear-dampen) >= 0.9 * Sharpe(V11) @ 7.5bps lag | PASS | V14c-soft-bear-dampen=0.6503, 0.9*V11=0.4776 (V11=0.5306) | >= 0.9 * V11 |
| TIER 1 lift: Sharpe(V14c-soft-bear-dampen @ 5bps nc) > Sharpe(V11 @ 5bps nc) + 0.10 | FAIL | V14c-soft-bear-dampen=0.6131, V11=0.5279, delta=+0.0852 | > 0.1 |

**V14c-soft-bear-dampen tier**: TIER 4

## Cost grid -- V14a-soft-bear-cash

| Cost bps | Mode | Sharpe | CAGR |
|---:|:--|---:|---:|
| 1.0 | near_close | 0.8011 | 14.43% |
| 1.0 | one_day_lag | 0.8181 | 18.94% |
| 5.0 | near_close | 0.6146 | 10.40% |
| 5.0 | one_day_lag | 0.6921 | 15.24% |
| 7.5 | near_close | 0.5220 | 8.48% |
| 7.5 | one_day_lag | 0.6339 | 13.58% |
| 10.0 | near_close | 0.3937 | 5.85% |
| 10.0 | one_day_lag | 0.5631 | 11.58% |

## Cost grid -- V14b-soft-bear-spy

| Cost bps | Mode | Sharpe | CAGR |
|---:|:--|---:|---:|
| 1.0 | near_close | 0.7691 | 16.47% |
| 1.0 | one_day_lag | 0.7789 | 16.47% |
| 5.0 | near_close | 0.6035 | 12.05% |
| 5.0 | one_day_lag | 0.6445 | 12.92% |
| 7.5 | near_close | 0.5081 | 9.57% |
| 7.5 | one_day_lag | 0.5510 | 10.51% |
| 10.0 | near_close | 0.4220 | 7.39% |
| 10.0 | one_day_lag | 0.4950 | 9.12% |

## Cost grid -- V14c-soft-bear-dampen

| Cost bps | Mode | Sharpe | CAGR |
|---:|:--|---:|---:|
| 1.0 | near_close | 0.7795 | 15.85% |
| 1.0 | one_day_lag | 0.8954 | 18.28% |
| 5.0 | near_close | 0.6131 | 11.69% |
| 5.0 | one_day_lag | 0.7432 | 14.49% |
| 7.5 | near_close | 0.5233 | 9.50% |
| 7.5 | one_day_lag | 0.6503 | 12.25% |
| 10.0 | near_close | 0.4163 | 6.95% |
| 10.0 | one_day_lag | 0.5826 | 10.68% |

## Cross-variants comparison (5 bps near_close)

| Variant | Sharpe | CAGR |
|---|---:|---:|
| V01 | 0.2818 | 3.74% |
| V11 | 0.5279 | 11.93% |
| V12 | 0.2683 | 3.52% |
| V12c-cfg | 0.5863 | 9.83% |
| V13-bear-invert | 0.4002 | 7.41% |
| V14a-soft-bear-cash | 0.6146 | 10.40% |
| V14b-soft-bear-spy | 0.6035 | 12.05% |
| V14c-soft-bear-dampen | 0.6131 | 11.69% |

## PBO

### Gate PBO (8 variants -- gate-influencing)

- Matrix shape: 2355 x 8 (V01, V11, V12, V12c-cfg, V13-bear-invert, V14a-soft-bear-cash, V14b-soft-bear-spy, V14c-soft-bear-dampen)
- s (CSCV submatrices): 16
- PBO value: 0.9528
- Interpretation: 0.9528 >= 0.5 = elevated overfitting risk

### Diagnostic PBO (4 orthogonal variants -- NOT gate-influencing)

- Variant set: V01, V11, V12, V14a-soft-bear-cash
- Matrix shape: 2355 x 4
- s (CSCV submatrices): 16
- PBO value: 0.6505
- **[!] PBO DIVERGENCE FLAG**: diagnostic PBO differs from gate PBO by 0.3023 (> 0.20). Investigate variant correlation structure.

## PSR / DSR detail per V14 variant

_Units note: PSR and DSR formulas (Bailey-Lopez de Prado / methodology Section 2.2-2.3) require **per-period (daily)** Sharpe with daily `n`. Annualized values are reported for human narrative only._

### V14a-soft-bear-cash

| Metric | Daily (formula input) | Annualized (narrative) |
|---|---:|---:|
| Observed Sharpe (V14a-soft-bear-cash) | 0.038723 | 0.6147 |
| Expected max under null (n_trials=36) | 0.020115 | 0.3193 |

| Metric | Value |
|---|---:|
| PSR (vs SR=0) | 0.9703 |
| DSR probability (true SR > expected max) | 0.8175 |
| Trial Sharpes (annualized) | V01=0.282, V11=0.528, V12=0.268, V12c-cfg=0.586, V13-bear-invert=0.400, V14a-soft-bear-cash=0.615, V14b-soft-bear-spy=0.604, V14c-soft-bear-dampen=0.613 |
| sqrt(V[trial_sharpes]) (daily) | 0.009367 |
| Sample skewness | 0.2420 |
| Sample Pearson kurtosis | 7.8856 |
| Sample size (days) | 2355 |

### V14b-soft-bear-spy

| Metric | Daily (formula input) | Annualized (narrative) |
|---|---:|---:|
| Observed Sharpe (V14b-soft-bear-spy) | 0.038027 | 0.6037 |
| Expected max under null (n_trials=36) | 0.020115 | 0.3193 |

| Metric | Value |
|---|---:|
| PSR (vs SR=0) | 0.9674 |
| DSR probability (true SR > expected max) | 0.8075 |
| Trial Sharpes (annualized) | V01=0.282, V11=0.528, V12=0.268, V12c-cfg=0.586, V13-bear-invert=0.400, V14a-soft-bear-cash=0.615, V14b-soft-bear-spy=0.604, V14c-soft-bear-dampen=0.613 |
| sqrt(V[trial_sharpes]) (daily) | 0.009367 |
| Sample skewness | 0.0646 |
| Sample Pearson kurtosis | 9.8041 |
| Sample size (days) | 2355 |

### V14c-soft-bear-dampen

| Metric | Daily (formula input) | Annualized (narrative) |
|---|---:|---:|
| Observed Sharpe (V14c-soft-bear-dampen) | 0.038630 | 0.6132 |
| Expected max under null (n_trials=36) | 0.020115 | 0.3193 |

| Metric | Value |
|---|---:|
| PSR (vs SR=0) | 0.9695 |
| DSR probability (true SR > expected max) | 0.8153 |
| Trial Sharpes (annualized) | V01=0.282, V11=0.528, V12=0.268, V12c-cfg=0.586, V13-bear-invert=0.400, V14a-soft-bear-cash=0.615, V14b-soft-bear-spy=0.604, V14c-soft-bear-dampen=0.613 |
| sqrt(V[trial_sharpes]) (daily) | 0.009367 |
| Sample skewness | 0.0495 |
| Sample Pearson kurtosis | 9.6224 |
| Sample size (days) | 2355 |

## Sensitivity panels (INFORMATIONAL, NOT gate-influencing)

_Per spec rev2 honesty discipline, these panels are NOT used in the gate decision. They inform the post-deploy parameter-monitoring plan._

| Panel key | Sharpe | CAGR |
|---|---:|---:|
| V14a|tau_out=0.4056 | 0.5623 | 8.95% |
| V14a|tau_out=0.5056 | 0.6146 | 10.40% |
| V14c|dampen=0.25 | 0.6903 | 12.45% |
| V14c|dampen=0.75 | 0.5952 | 12.73% |

## V11 reference values

| Reference | Sharpe |
|---|---:|
| V11 @ 5 bps near_close (from cross-variants) | 0.5279 |
| V11 @ 7.5 bps one_day_lag (Gate 5 baseline) | 0.5306 |
| V11-readiness-doc Sharpe @ 7.5 bps lag | 0.5306 (legacy reference; gate uses freshly-run V11) |

## Methodology decisions

- n_trials_project = 36 (audited count per spec rev2 honesty discipline: V11+pre-V11 22 + V12+sensitivity 5 + V12c 1 + V13 1 + V14a/b/c 3 + V14a tau sens 2 + V14c dampen sens 2 = 36)
- PBO s = 16 (methodology Section 2.4 default)
- Cost tier for PSR/DSR/PBO: 5.0 bps per side
- one_day_lag definition: signal computed at close T from `panel.loc[:T]`, trades executed at close T+1, MTM at close T+1.
- All metrics computed on net-of-cost daily returns.
- Variants run at full window (start..end). Phase 4 is single-pass, not walk-forward.
- Gate 4 (directional): `(nc - lag) <= max(0.2 * |nc|, 0.1)` -- lag > near_close is the safe direction and is not penalized.
- Gate 5: both clauses required: Sharpe(variant @ 7.5bps lag) > 0.30 AND >= 0.9 * Sharpe(V11 @ 7.5bps lag).
- Tier classification: TIER 1 (all 5 structural+sig gates pass AND Sharpe lift > 0.1 over V11 @ 5bps nc) / TIER 3 (structural pass, PSR+DSR or lift fails) / TIER 4 (any structural gate fails).
- Tau constants loaded ONCE at module import from `config/research/v14_tau_constants.json` (NOT hardcoded).
- V12c-cfg is synthesized via `regime_positions[UNPREDICTABLE] = "cash"` on the V12 REGISTRY plan_fn -- not a separate REGISTRY entry.
- Diagnostic PBO (4 orthogonal variants) is reported alongside the gate PBO but does NOT influence the gate verdict.
- Sensitivity panels (V14a tau-band, V14c dampen) are INFORMATIONAL only.

## Metadata

- Git SHA: 6f55e37
- Run datetime: 2026-05-24T19:43:34
- n_trials_project: 36
- Tau constants source: config/research/v14_tau_constants.json (tau_in=0.555556, tau_out=0.455556)
- Total gate-influencing unique runs: 30 (24 V14 cost grid + 5 cross-variants + 1 V11 ref)
- Sensitivity panels: 4 (informational only; V14a tau-band x2 + V14c dampen x2)
