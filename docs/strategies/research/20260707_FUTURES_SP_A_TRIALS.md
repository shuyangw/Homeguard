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

Results (append after each run: OOS Sharpe 1x/1.5x, PBO, PSR, DSR, verdict).
