# RAMP Re-Evaluation 2026-05-04

**Purpose:** Validate the 0.846 OOS baseline + test alpha decay on truly-OOS 2025-2026 data.
**Methodology:** Identical to `ramp_walk_forward_validation.py` (yfinance split-adjusted; 0% transaction costs; 1/N equal weight; production REGIME_PARAMS).
**Universe:** `config/universes/sp500-2025.csv` (503 symbols).
**Reference baseline (2025-12-12):** OOS Sharpe 0.846 on 2022-2024.
**Script:** `scripts/backtest_scripts/ramp_re_eval_20260504.py`

## Results

| Period | Range | Sharpe | CAGR | Max DD | Win Rate | Trading Days | Notes |
|---|---|---|---|---|---|---|---|
| IS | 2017-2021 | 0.743 | 15.1% | -47.0% | 53.1% | 1258 | Training |
| OOS | 2022-2024 | 0.823 | 15.7% | -15.5% | 52.8% | 752 | **Compare to 0.846 baseline** |
| EXTENDED-OOS | 2025-01 to 2026-04 | 0.074 | -1.5% | -21.7% | 50.0% | 330 | **Truly out-of-sample** |

## Comparison to 0.846 baseline

The 2022-2024 OOS Sharpe of **0.823** differs from the 2025-12-12 reference of **0.846** by **-0.023**. This is within the +/-0.1 tolerance and the baseline is confirmed. The small delta is consistent with expected run-to-run variation from yfinance data (minor corporate action differences, index composition timing). The methodology is identical: same universe file, same REGIME_PARAMS, same 0% costs, same +/-20% return cap.

**Assessment: MATCHES BASELINE**

## Alpha decay assessment (2025-2026)

The EXTENDED-OOS period (2025-01-01 to 2026-04-30, 330 trading days) produced a Sharpe of **0.074** versus the 0.846 baseline -- a decay of **0.772 Sharpe units**.

Decay thresholds:
- Within +/-0.1: HEALTHY
- 0.1-0.3 below: MILD DEGRADATION
- >0.3 below: MATERIAL DECAY
- Negative: SEVERE DECAY

The EXTENDED-OOS Sharpe of 0.074 is 0.772 below baseline -- well past the MATERIAL DECAY threshold. The strategy produced a CAGR of -1.5% and a total return of -2.0% on the 2025-2026 period, effectively flat-to-negative. This is a sharp deterioration from the 15.7% CAGR seen OOS (2022-2024).

**Assessment: MATERIAL DECAY -- warrants investigation**

Note: the IS period also showed Sharpe 0.743, below the OOS Sharpe of 0.823. This atypical IS < OOS pattern is not a concern for the re-evaluation itself (parameters were chosen from IS data), but confirms the strategy's edge was genuinely better in 2022-2024 than in the training window.

## Regime breakdown (EXTENDED-OOS)

For 2025-2026 only:

| Regime | Days | % Time | Cum Contrib | Avg Daily | Sharpe |
|---|---|---|---|---|---|
| WEAK_BULL | 144 | 43.6% | -0.091 | -0.0006 | -0.778 |
| SIDEWAYS | 77 | 23.3% | +0.128 | +0.0017 | 1.157 |
| BEAR | 64 | 19.4% | -0.153 | -0.0024 | -2.174 |
| STRONG_BULL | 41 | 12.4% | +0.075 | +0.0018 | 3.175 |
| UNPREDICTABLE | 4 | 1.2% | +0.068 | +0.0169 | 6.691 |

Key observations:
1. **BEAR regime (19.4% of time, Sharpe -2.17)** is the largest drag. The strategy's BEAR parameters (long_p=21, short_p=5, top_n=10) appear to be selecting momentum stocks that continue losing during drawdowns -- not a mean-reversion overlay. The 50% exposure reduction via VIX/SPY-DD trigger is insufficient to offset the selection loss.
2. **WEAK_BULL (43.6% of time, Sharpe -0.78)** -- the modal regime -- is also slightly negative. Over half the 2025-2026 period the strategy is generating losses even in non-bear conditions.
3. **SIDEWAYS and STRONG_BULL** are profitable (Sharpe 1.16 and 3.18 respectively) but represent only 35.8% of trading days.
4. The regime distribution in 2025-2026 is heavily weighted toward BEAR and WEAK_BULL (63% combined), whereas 2022-2024 had more favorable regime mix. This is partly a market structure change (2025-2026 saw increased volatility from tariff shocks and rate uncertainty), but the severity of the BEAR drawdown suggests the strategy itself may be directionally exposed to market selloffs even when it is supposed to be protected.

## Caveats and methodology notes

- 0% transaction costs (research-grade methodology, matches 0.846 baseline)
- 100% daily turnover assumption (no actual portfolio simulation; signals are recomputed fresh each day)
- +/-20% daily return cap on individual stocks (data quality filter)
- Survivorship bias: uses `sp500-2025.csv` (current S&P 500 list) for all historical periods; results for IS and OOS periods are optimistic relative to what was knowable at the time
- yfinance split-adjusted data (auto_adjust=True); BRK.B, etc. converted to yfinance ticker format
- Production REGIME_PARAMS used verbatim -- no optimization in this script
- IS period Max Drawdown of -47% is large; this includes the COVID crash (Feb-Mar 2020) and is expected given 100% exposure and no short-selling
- UNPREDICTABLE regime (4 days, Sharpe 6.69) is a statistical artifact of very few observations -- not meaningful
- The 2025-2026 window is approximately 16 months; Sharpe estimates over this window carry wide confidence intervals (standard error ~0.17 for n=330 days), so the point estimate of 0.074 should be interpreted with caution; however, -2.0% total return and -21.7% max drawdown are concrete facts

## Conclusion

The 0.846 OOS baseline is confirmed -- the re-run on identical methodology produces 0.823, well within tolerance. The strategy was validated through 2024.

However, the truly-out-of-sample 2025-2026 period (16 months the strategy was actually live) shows **material decay**: Sharpe 0.074, CAGR -1.5%, Max DD -21.7%. The root cause appears to be a regime composition shift -- 2025-2026 has been heavily BEAR and WEAK_BULL, and the BEAR regime parameters in particular are generating significant losses (-2.17 Sharpe, 64 days). This is a structural concern: if the strategy's BEAR-regime behavior is wrong, no amount of momentum tweaking in other regimes will fix it.

**Recommendation: DO NOT PROCEED TO PHASE B (z-score infrastructure swap) without first investigating BEAR regime performance.** Phase B would swap the signal normalisation method, which is unlikely to address the regime-classification or BEAR-parameter issue revealed here. Suggested prior investigation:

1. Confirm the regime detector is classifying 2025 correctly (tariff-driven selloffs may look different from 2022 rate-shock drawdowns)
2. Examine whether the BEAR-mode VIX/drawdown trigger fired as expected during 2025-Q1 selloffs
3. Consider whether the BEAR parameters should hold cash rather than hold 50% momentum exposure

Surface findings to human review before committing further engineering cycles to Phase B.

## Next steps

MATERIAL DECAY detected -- **pause; surface to human review before Phase B.**

If after review the team decides to proceed anyway, Phase B should be treated as exploratory and the 2025-2026 window used as the primary validation target, not the 2022-2024 window.
