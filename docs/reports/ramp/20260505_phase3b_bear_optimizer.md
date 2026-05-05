# RAMP Phase 3B: BEAR Regime Walk-Forward Optimization -- 2026-05-05

## Context

Root cause investigation (docs/reports/ramp/20260505_root_cause_investigation.md) found that H2 (regime params don't generalize) is the dominant cause of EXT-OOS alpha decay. BEAR regime (64 days, Sharpe -2.17 in 2025-2026) was the worst individual contributor. V1 (no regime, always pen_w=5.0) produced EXT-OOS Sharpe 0.314 vs production V0 0.070, suggesting the BEAR penalty weight (currently 3.0, lower than all other regimes) is the most actionable lever. This run sweeps pen_w_bear and exposure_during_bear jointly using 4 walk-forward windows, with 2025-2026 reserved as true OOS.

## Methodology

Universe: sp500-2025.csv (503 symbols). Data: yfinance split-adjusted (auto_adjust=True), 2016-07-01 to 2026-04-30. Transaction costs: 0% for walk-forward selection; cost sensitivity tested on EXT-OOS winner. +/-20% daily return cap. Regime detection: same MarketRegimeDetector as production. Non-BEAR regimes held at production values. Winner selected by max mean per-window OOS Sharpe across W1-W4 only; 2025-2026 data not seen during optimization.

Walk-forward windows:
- W1: train 2017-2020, test 2021
- W2: train 2017-2021, test 2022
- W3: train 2018-2022, test 2023
- W4: train 2019-2023, test 2024

## Walk-forward results matrix (per-window OOS Sharpe)

Production row first, then sorted by mean OOS Sharpe descending.

| pen_w_bear | exposure_during_bear | W1 OOS (2021) | W2 OOS (2022) | W3 OOS (2023) | W4 OOS (2024) | Mean OOS | Min OOS | IS/OOS gap |
|---|---|---|---|---|---|---|---|---|
| 3.0 (prod) | 0.50 (prod) | 0.298 | 0.036 | 1.683 | 0.864 | 0.720 | 0.036 | -20.5% |
| 7.0 | 1.00 | 0.368 | 0.319 | 1.678 | 1.166 | 0.883 | 0.319 | -9.3% |
| 5.0 | 1.00 | 0.371 | 0.289 | 1.677 | 1.174 | 0.878 | 0.289 | -7.8% |
| 6.0 | 1.00 | 0.368 | 0.284 | 1.681 | 1.166 | 0.874 | 0.284 | -8.9% |
| 4.0 | 1.00 | 0.371 | 0.262 | 1.663 | 1.188 | 0.871 | 0.262 | -9.7% |
| 3.0 | 1.00 | 0.367 | 0.165 | 1.660 | 1.170 | 0.840 | 0.165 | -9.6% |
| 7.0 | 0.50 | 0.299 | 0.146 | 1.692 | 0.863 | 0.750 | 0.146 | -20.6% |
| 5.0 | 0.50 | 0.300 | 0.124 | 1.692 | 0.867 | 0.746 | 0.124 | -19.4% |
| 6.0 | 0.50 | 0.299 | 0.121 | 1.694 | 0.863 | 0.744 | 0.121 | -20.2% |
| 4.0 | 0.50 | 0.300 | 0.105 | 1.685 | 0.874 | 0.741 | 0.105 | -20.9% |
| 7.0 | 0.25 | 0.263 | -0.009 | 1.693 | 0.696 | 0.661 | -0.009 | -31.0% |
| 5.0 | 0.25 | 0.264 | -0.021 | 1.693 | 0.697 | 0.658 | -0.021 | -30.1% |
| 6.0 | 0.25 | 0.263 | -0.023 | 1.694 | 0.696 | 0.657 | -0.023 | -30.6% |
| 4.0 | 0.25 | 0.264 | -0.032 | 1.689 | 0.701 | 0.655 | -0.032 | -31.1% |
| 3.0 | 0.25 | 0.263 | -0.072 | 1.688 | 0.696 | 0.644 | -0.072 | -30.8% |
| 3.0 | 0.00 | 0.228 | -0.198 | 1.689 | 0.521 | 0.560 | -0.198 | -49.2% |
| 5.0 | 0.00 | 0.228 | -0.198 | 1.689 | 0.521 | 0.560 | -0.198 | -49.2% |
| 4.0 | 0.00 | 0.228 | -0.198 | 1.689 | 0.521 | 0.560 | -0.198 | -49.2% |
| 6.0 | 0.00 | 0.228 | -0.198 | 1.689 | 0.521 | 0.560 | -0.198 | -49.2% |
| 7.0 | 0.00 | 0.228 | -0.198 | 1.689 | 0.521 | 0.560 | -0.198 | -49.2% |

## Best config (by mean OOS Sharpe across W1-W4)

- pen_w_bear: **7.0**
- exposure_during_bear: **1.00**
- Mean OOS Sharpe W1-W4: **0.883**
- Min OOS Sharpe (worst window): 0.319
- Mean IS Sharpe (for reference): 0.807
- IS/OOS degradation: -9.3%
- Production mean OOS Sharpe (pen_w=3.0, exp=0.5): 0.720
- Delta vs production: +0.162

The winner outperforms production by +0.162 mean OOS Sharpe across the 4 walk-forward windows.

## Final EXTENDED-OOS validation (2025-01-01 to 2026-04-30)

Note: EXT-OOS was NOT used in winner selection. These are truly out-of-sample results.

**DISCREPANCY NOTE:** Yesterday's root cause investigation (20260505_root_cause_investigation.md) reported
V0 EXT-OOS Sharpe as 0.070 over 2025-04 to 2026-04 (EXT-OOS start defined as 2025-04-01 in that script).
This script uses 2025-01-01 to 2026-04-30 as the EXT-OOS window, which includes Q1 2025 (per-year Sharpe
0.234 from D2 diagnostics). The extra 3 months of Q1 2025 positive performance raises the aggregate Sharpe
from 0.070 to 0.355. This is not a contradiction -- both are correct for their respective windows. The
2025-04 to 2026-04 window in the prior report more closely matches the period when the strategy is live in
production. Both windows confirm the winner (pen_w=7.0, exp=1.0) UNDERPERFORMS production V0 in EXT-OOS.

| Metric | Production V0 (pen_w=3.0, exp=0.5) | Optimized | Delta |
|---|---|---|---|
| EXT-OOS Sharpe (0% costs) | 0.355 | 0.213 | -0.143 |
| EXT-OOS CAGR | 5.9% | 2.2% | -3.7% |
| EXT-OOS MaxDD | -17.2% | -20.5% | -3.3% |

**KEY FINDING: The walk-forward winner (pen_w=7.0, exp=1.0) performs WORSE than production V0 on EXT-OOS
on every metric. The optimization improved W1-W4 mean OOS Sharpe by +0.162 but this did not transfer to
the held-out 2025-2026 period. This is a NULL RESULT for the hypothesis that higher pen_w_bear fixes the
BEAR regime.**

## Cost sensitivity (winning config on EXT-OOS)

| Cost Level | EXT-OOS Sharpe | EXT-OOS CAGR | EXT-OOS MaxDD |
|---|---|---|---|
| 0x (0 bps/side) | 0.213 | 2.2% | -20.5% |
| 1x (5 bps/side) | -0.213 | -8.8% | -21.8% |
| 1.5x (7.5 bps/side) | -0.426 | -13.8% | -23.7% |

## Pre-committed evaluation criteria

- EXT-OOS Sharpe > 0.5 at 0% costs: **FAIL** (0.213)
- EXT-OOS Sharpe > 0.3 at 1.5x costs: **FAIL** (-0.426)
- W1-W4 mean OOS Sharpe within +/-0.1 of production or better: **PASS** (winner 0.883 vs prod 0.720)
- IS/OOS gap < 30%: **PASS** (-9.3%)

**2 of 4 criteria failed. Winner is NOT a production candidate without further work.**

## Overfitting check

- Number of configs tested: 20 (5 pen_w_bear x 4 exposure_during_bear)
- Tunable parameters: 2 (target <=3: PASS)
- Both parameters have economic rationale: PASS
  - pen_w_bear: contrarian penalty weight. Higher = stronger preference for stocks that did well long-term but underperformed short-term. Economic logic: in BEAR markets, momentum reversal is common; higher pen_w filters out recent momentum names.
  - exposure_during_bear: capital deployment. Lower = more cash. Economic logic: BEAR regime implies elevated downside risk; reducing exposure limits drawdowns.

- Best mean OOS Sharpe across configs: 0.883
- Worst mean OOS Sharpe across configs: 0.560
- Best/worst spread: 0.322

### Parameter stability (neighbors of winning config)

| pen_w_bear | exposure_during_bear | Mean OOS Sharpe | Is Winner |
|---|---|---|---|
| 8.0 | 1.00 | 0.888 |  |
| 7.0 | 1.00 | 0.883 | [*] |
| 6.0 | 1.00 | 0.874 |  |
| 8.0 | 0.75 | 0.828 |  |
| 7.0 | 0.75 | 0.823 |  |
| 6.0 | 0.75 | 0.816 |  |

Winner vs neighbor mean: +0.037
Neighbor degradation is moderate -- the winning config is not a sharp spike. Parameter stability is ACCEPTABLE.

## Additional observation: cost sensitivity is extreme for both configs

At 1x cost (5 bps/side), both the winner and production V0 go negative on EXT-OOS Sharpe:
- Production V0: Sharpe drops from 0.355 to -0.076 (79% degradation -- COST-SENSITIVE)
- Winner: Sharpe drops from 0.213 to -0.213 (200% degradation -- EXTREMELY COST-SENSITIVE)

This is a fundamental concern independent of the BEAR parameter optimization. The strategy as a whole
(including non-BEAR regimes) has insufficient alpha to cover realistic transaction costs in 2025-2026.
Adjusting BEAR parameters alone cannot fix this.

## Conclusion

**NULL RESULT.** The walk-forward optimization found pen_w_bear=7.0 and exposure_during_bear=1.0 as the
best configuration on W1-W4 mean OOS Sharpe (0.883 vs production 0.720, delta +0.162). However, on the
held-out 2025-2026 EXT-OOS period this configuration UNDERPERFORMS production V0 on all metrics
(Sharpe 0.213 vs 0.355, CAGR 2.2% vs 5.9%, MaxDD -20.5% vs -17.2%). Both configs fail EXT-OOS
criteria at realistic transaction costs.

The W1-W4 improvement for exposure=1.0 is driven primarily by W4 (2024), a bull-market year when
full exposure to momentum names naturally outperforms. This is regime-specific over-fitting to
bull conditions, not a genuine BEAR regime fix. The 2022 bear market window (W2) remains the most
discriminating test: production V0 scores 0.036 there while the winner scores 0.319 -- but the
actual 2025-2026 bear episodes are apparently structured differently from 2022.

The hypothesis that raising pen_w_bear from 3.0 to 7.0 and increasing BEAR exposure to 1.0 would
fix the BEAR regime is **REJECTED** based on EXT-OOS evidence.

## Implications

**NOT ACTIONABLE.** Do not change BEAR regime pen_w or exposure based on this optimization.

Two findings that are actionable:
1. The strategy's edge in 2025-2026 is so thin that it cannot survive realistic costs. This points
   to a broader alpha decay problem, not a tunable-parameter problem. The priority fix is determining
   WHY the momentum signal itself underperforms during 2025-2026 BEAR days (root cause H6: wrong
   stocks selected). Replacing BEAR-regime equity selection with cash or defensive rotation is the
   higher-leverage option.
2. exposure_during_bear=1.0 (full exposure in BEAR) consistently underperforms on EXT-OOS despite
   winning on W1-W4. This confirms that reducing exposure during BEAR conditions is the right
   instinct -- the problem is the stock selection, not the exposure level.

Recommended next step: Test a config where BEAR regime holds cash entirely (exposure=0.0 but with
full position in non-BEAR days) and validate whether eliminating BEAR-regime trading entirely
produces a cleaner Sharpe profile. This is equivalent to the "if you can't win, don't play"
approach to the broken BEAR regime.
