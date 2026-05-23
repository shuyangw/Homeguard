# V11 Phase D Readiness -- 2026-05-23

**Code commit**: bea4bd9
**Data**: Alpaca SIP daily-aggregated, 2017-01-01 to 2026-05-16
**Universe**: config\universes\sp500-2025.csv
**Cost tier for gates**: 5.0 bps per side
**n_trials**: 20 (conservative; covers Phase 4 + Phase 3 variant grid)

## Verdict

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| PSR (vs SR=0) | FAIL | 0.9442 | > 0.95 |
| DSR (n_trials=20) | FAIL | 0.8108 | > 0.95 |
| PBO across {V01,V04,V05,V06,V11} | PASS | 0.1256 | < 0.5 |
| One-day-lag Sharpe robustness (5 bps) | PASS | nc=0.528, lag=0.580, delta=+9.79% | within 20% |

**Overall**: PARTIAL -- passes structural gates (PBO, one-day-lag robustness); fails absolute-significance gates (PSR, DSR). V11 is structurally sound (no overfitting evidence, no lookahead) but its Sharpe magnitude is not large enough to clear strict Bailey-Lopez de Prado significance hurdles after multi-trial correction. Decision to advance is a judgment call, not a clean PASS.

## PSR / DSR detail

_Units note: PSR and DSR formulas (Bailey-Lopez de Prado / methodology Section 2.2-2.3) require **per-period (daily)** Sharpe with daily `n`. We display annualized for the human narrative and report the daily values used as formula inputs._

| Metric | Daily (formula input) | Annualized (narrative) |
|---|---:|---:|
| Observed Sharpe | 0.033260 | 0.5280 |
| Expected max under null (n_trials=20) | 0.014842 | 0.2356 |

| Metric | Value |
|---|---:|
| PSR (vs SR=0) | 0.9442 |
| DSR probability (true SR > expected max) | 0.8108 |
| Trial Sharpes used (annualized) | V01=0.282, V04=0.313, V05=0.503, V06=0.278, V11=0.528 |
| sqrt(V[trial_sharpes]) (daily) | 0.007809 |
| Sample skewness | -0.5963 |
| Sample Pearson kurtosis | 33.3085 |
| Sample size (days) | 2355 |

### DSR sensitivity to n_trials

| n_trials | Expected max Sharpe (annual) | DSR | Pass (> 0.95) |
|---:|---:|---:|:---:|
| 2 | 0.0644 | 0.9188 | FAIL |
| 3 | 0.1057 | 0.8984 | FAIL |
| 6 | 0.1612 | 0.8655 | FAIL |
| 20 | 0.2356 | 0.8108 | FAIL |

_Reading this table: lower n_trials shrinks the multi-trial selection-bias adjustment, which would help V11 if the limit were the correction; if DSR fails across all rows, the limit is V11's Sharpe magnitude, not n_trials choice._

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

PARTIAL READINESS. V11 passes the structural gates (no overfitting per PBO, no lookahead per one-day-lag) but fails the absolute-significance gates (PSR, DSR). Three options:

1. **Advance V11 to paper anyway.** Paper trading is itself an OOS validation channel; the significance gates measure backtest noise, not live performance. Requires extending the A7 comparator (`scripts/trading/compare_paper_vs_plan.py`) for V11's filter state and accepting the significance caveat in the deployment record.
2. **Fall back to V05.** V05 has similar Sharpe (~0.50) and the same significance situation (also fails strict PSR/DSR). Simpler filter chain. Comparator extension is simpler since only min_hold is needed.
3. **Pause Phase D until Wave 2.** V12 (BEAR-to-cash on V11 base) is the natural next variant that could raise Sharpe enough to clear DSR. Separate brainstorm.
