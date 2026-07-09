# Futures SP-A Trials -- Pre-Registration Ledger (2026-07-07)

Every SP-A strategy is a parameter-free, pre-registered trial (DSR trial count
+1 each). Construction and expected sign are fixed before any OOS result is
seen; no post-hoc sign flips. Results are appended after each walk-forward run.

| # | Strategy (registry name) | Template | Universe | Expected sign | Config |
|---|---|---|---|---|---|
| 3 | FuturesXSMomentum | CrossSectionalRank | commodity | long_short | xs_commodity_momentum.yaml |
| 10 | FuturesCarryXS (commodity) | CrossSectionalRank | commodity | long_short | curve_slope_xs.yaml |
| 15 | FuturesSameMonthSeasonality | CrossSectionalRank | commodity | long_short | same_month_seasonality.yaml |
| 16 | FuturesTurnOfMonth | CalendarMask | index | long | turn_of_month.yaml |
| 23 | FuturesReversal | continuous | index | long_short | index_reversal.yaml |
| 13 | FuturesCarryTrend | ConditioningOverlay | broad | long_short | carry_trend_gate.yaml |

Deferred (needs a multi-horizon carry cache, tracked with SP-E data work):
- #9 multi-horizon carry blend.

## Results -- walk-forward gate (2026-07-09)

Walk-forward 36m/12m/12m, 2010-06-07..2026-02-20, IDM on, 13 windows.
Gate = PSR>=0.95 AND DSR>=0.95 AND PBO<0.25 AND 1.5x cost. Benchmark carry_idm = 0.765 / PBO 0.189 / PASS.

| # | Strategy | OOS 1x | OOS 1.5x | PSR | DSR | PBO | skew | verdict |
|---|---|---|---|---|---|---|---|---|
| 3 | XS commodity momentum | 0.209 | 0.181 | 1.00 | 1.00 | 0.579 | -0.08 | WEAK (PBO) |
| 10 | curve-slope XS | 0.846 | 0.833 | 1.00 | 1.00 | 0.690 | -0.80 | WEAK (PBO -- highest raw Sharpe but unstable, same failure mode as rejected XS-carry) |
| 15 | same-month seasonality | 0.180 | 0.166 | 1.00 | 1.00 | 0.281 | 0.38 | WEAK (PBO just over 0.25; contested [C], expected) |
| 16 | turn-of-month | -0.274 | -0.279 | 0.00 | 0.00 | 0.217 | -17.84 | REJECT* (MIS-SAMPLED: daily signal on a weekly-rebalance runner -- verdict unreliable, needs a daily-rebalance walk-forward) |
| 23 | short-horizon reversal | 0.297 | 0.288 | 1.00 | 1.00 | 0.805 | 7.04 | WEAK (PBO) |
| 13 | carry-trend gate | 0.357 | 0.336 | 1.00 | 1.00 | 0.189 | -0.88 | PASS (clears gate, but 0.357 << carry 0.765; a carry+trend re-expression -- needs the marginal-Sharpe-vs-the-pair check before it earns a sleeve) |

Bottom line: NONE beats the incumbent carry_idm (0.765). Only #13 clears the gate and it is
weaker than carry and likely a re-expression. #10 has the highest raw Sharpe (0.846 > carry) but
PBO 0.690 = badly overfit/unstable (the XS-carry failure mode). All others WEAK/REJECT. No survivorship:
every outcome recorded. CAVEAT: DSR uses the per-run parameter-free count; a best-of-N deflation across
this sweep (and prior campaign trials) would further deflate even #13's pass.
