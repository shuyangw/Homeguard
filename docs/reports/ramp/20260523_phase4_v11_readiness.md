# V11 Phase D Readiness -- 2026-05-23

**Code commit**: a374023
**Data**: Alpaca SIP daily-aggregated, 2017-01-01 to 2026-05-16
**Universe**: config\universes\sp500-2025.csv
**Cost tier for gates**: 5.0 bps per side
**n_trials**: 20 (conservative; covers Phase 4 + Phase 3 variant grid)

## Verdict

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| PSR (vs SR=0) | PASS | 1.0000 | > 0.95 |
| DSR (n_trials=20) | PASS | 1.0000 | > 0.95 |
| PBO across {V01,V04,V05,V06,V11} | PASS | 0.1256 | < 0.5 |
| One-day-lag Sharpe robustness (5 bps) | PASS | nc=0.528, lag=0.580, delta=+9.79% | within 20% |

**Overall**: READY for Phase D paper deploy

## PSR / DSR detail

| Metric | Value |
|---|---:|
| Observed annualized Sharpe | 0.5280 |
| Expected max Sharpe under null (n_trials=20, scaled by V[trial_sharpes]) | 0.2356 |
| DSR probability (true SR > expected max) | 1.0000 |
| PSR (vs SR=0) | 1.0000 |
| Trial Sharpes used for V[trial] term | V01=0.282, V04=0.313, V05=0.503, V06=0.278, V11=0.528 |
| sqrt(V[trial_sharpes]) | 0.1240 |
| Skewness | -0.5963 |
| Pearson kurtosis | 33.3085 |
| Sample size (days) | 2355 |

## PBO

- Matrix shape: 2355 x 5 (V01, V04, V05, V06, V11)
- s (submatrices): 16
- PBO value: 0.1256
- Interpretation: 0.1256 < 0.5 is low overfitting evidence

### Per-variant Sharpe at 5 bps near_close

| Variant | Sharpe | CAGR |
|---|---:|---:|
| V01 | 0.2818 | 3.74% |
| V04 | 0.3133 | 4.89% |
| V05 | 0.5030 | 11.08% |
| V06 | 0.2781 | 3.62% |
| V11 | 0.5279 | 11.93% |

## One-day-lag robustness sweep (V11)

| Cost bps | near_close Sharpe | one_day_lag Sharpe | Delta % |
|---|---:|---:|---:|
| 0.0 | 0.6933 | 0.7397 | +6.71% |
| 2.5 | 0.6053 | 0.6612 | +9.23% |
| 5.0 | 0.5279 | 0.5796 | +9.79% |
| 7.5 | 0.4516 | 0.5306 | +17.51% |

If one_day_lag Sharpes at 5 bps degrade by more than 20% from near_close, V11 has structural lookahead.

## Methodology decisions

- n_trials = 20 (conservative; methodology Section 2.3 cumulative trial count)
- PBO s = 16 (methodology Section 2.4 default)
- Cost tier for PSR/DSR/PBO: 5.0 bps per side
- one_day_lag definition: signal computed at close T from `panel.loc[:T]`, trades executed at close T+1, MTM at close T+1
- All metrics computed on net-of-cost daily returns
- Variants run at full window (start..end). No purge/embargo applied within-variant (Phase 4 is single-pass, not walk-forward).

## What happens next

READY: extend the A7 paper-validation comparator (`scripts/trading/compare_paper_vs_plan.py`) to model V11's rank_buffer + min_hold + delta_threshold filter state. Then enable V11 in production paper trading on EC2 per the A7 discipline (4-6 weeks paper validation).
