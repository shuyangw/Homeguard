# Experiment 2 -- UNPREDICTABLE Hand-Inspection

**Date**: 2026-05-24
**Branch**: v12-bear-to-cash
**Builds on**: V12 readiness sensitivity (V12-up-cash); regime detector v0 labels (2017-2026)
**Decision target**: gate Experiment 6 (V12c formal readiness)

## Summary

V12-up-cash (UNPREDICTABLE='cash') produced Sharpe 0.586 vs V11 0.528 at 5 bps near_close -- a +0.058 edge from being in cash on the 41 UNPREDICTABLE days (14 maximal-contiguous events, 2017-2026). The top-3 events explain **53.6%** of total absolute avoided-loss attribution, which lands in the 50-75% AMBIGUOUS band of the pre-registered decision criterion. The COVID crash dominates (event id=7, 2020-03-05..2020-03-19, 23.1% SPY drawdown during the window, ~30.6% of total attribution); without the two COVID events the proxy Sharpe contribution from the remaining 12 firings would be roughly +0.033 vs V11. **Verdict: AMBIGUOUS -- defer to analyst.** Recommendation in the Implications section.

## Methodology

- UNPREDICTABLE events identified from `diagnostics/regime/v0/labels.parquet` as maximal contiguous runs of `regime == 'UNPREDICTABLE'`. 14 events, 41 total days.
- Per-event SPY return computed from `diagnostics/data/spy_vix_2016_2026.parquet` as simple close-to-close: `close[end] / close[day_before_start] - 1`. Forward 5/10/20 trading-day returns measured from `close[end]`.
- **Avoided-loss attribution under V12-up-cash assumption**: under V12-up-cash, the portfolio is 0% SPY during UNPREDICTABLE, so it forgoes the SPY return over the event window. Avoided loss = `-spy_return_during`. Negative SPY return (sell-off) -> positive avoided loss (cash sidestepped the drop). Positive SPY return (relief rally) -> negative avoided loss (cash missed the gain).
- **Sharpe contribution proxy**: linear scaling of the +0.058 V12-up-cash vs V11 Sharpe delta by each event's share of total absolute attribution. This is a **first-order approximation** -- it assumes the per-event volatility contribution scales with the per-event return contribution. Documented in Limitations.
- All 14 events had **negative** SPY return during -- v0 UNPREDICTABLE fires exclusively on volatile sell-offs in this window, so attribution magnitude and signed attribution coincide.

## Events table

Sorted by absolute avoided-loss attribution descending. All returns are simple close-to-close. `forward_Nd` measures from event end_date close.

| rank | event_id | start_date | end_date | n_days | spy_return_during | forward_5d | forward_10d | forward_20d | avoided_loss | cum_share |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 2020-03-05 | 2020-03-19 | 11 | -23.13% | +8.60% | +4.71% | +19.18% | +23.13% | 30.6% |
| 2 | 6 | 2020-02-24 | 2020-03-03 | 7 | -9.97% | -3.94% | -15.80% | -14.15% | +9.97% | 43.7% |
| 3 | 13 | 2025-04-04 | 2025-04-08 | 3 | -7.49% | +8.28% | +7.84% | +13.03% | +7.49% | 53.6% |
| 4 | 3 | 2018-10-10 | 2018-10-11 | 2 | -5.30% | +1.55% | -0.77% | +3.06% | +5.30% | 60.6% |
| 5 | 2 | 2018-02-05 | 2018-02-09 | 5 | -5.06% | +4.44% | +6.27% | +6.51% | +5.06% | 67.3% |
| 6 | 11 | 2024-08-02 | 2024-08-07 | 4 | -4.48% | +4.84% | +8.09% | +5.97% | +4.48% | 73.3% |
| 7 | 14 | 2025-04-10 | 2025-04-10 | 1 | -4.38% | +0.35% | +4.97% | +7.58% | +4.38% | 79.0% |
| 8 | 12 | 2024-12-18 | 2024-12-19 | 2 | -3.01% | +1.52% | +1.58% | +3.47% | +3.01% | 83.0% |
| 9 | 5 | 2019-08-05 | 2019-08-05 | 1 | -3.01% | +1.50% | +3.00% | +2.44% | +3.01% | 87.0% |
| 10 | 4 | 2018-12-24 | 2018-12-24 | 1 | -2.64% | +6.76% | +10.08% | +12.46% | +2.64% | 90.5% |
| 11 | 8 | 2021-01-27 | 2021-01-27 | 1 | -2.44% | +1.99% | +4.19% | +2.12% | +2.44% | 93.7% |
| 12 | 9 | 2021-11-26 | 2021-11-26 | 1 | -2.23% | -1.21% | +2.56% | +3.99% | +2.23% | 96.7% |
| 13 | 1 | 2017-08-10 | 2017-08-10 | 1 | -1.41% | -0.27% | +0.09% | +1.16% | +1.41% | 98.5% |
| 14 | 10 | 2021-12-01 | 2021-12-01 | 1 | -1.11% | +4.22% | +4.46% | +5.70% | +1.11% | 100.0% |

Run length distribution: mean 2.93 days, median 1.5 days, max 11 days (event 7 = COVID crash leg). Eight of fourteen events are single-day firings.

## Concentration analysis

- **Total events**: 14
- **Total absolute attribution**: +0.7567 (sum of |spy_return_during| across all events)
- **Top-3 events' attribution share**: 53.6%
- **Top-3 events' implied Sharpe contribution**: ~+0.0311 out of the +0.058 V12-up-cash vs V11 Sharpe delta (53.6% x 0.058)
- **Top-5 share**: 67.3% (still below 75% fragility threshold but trending close)
- **Cumulative ranks**: 8 events needed to reach 80% of total attribution
- **Top-2 (both COVID-related)**: 43.7% of total attribution alone -- one macro regime change (Feb-Mar 2020) drives nearly half of the V12-up-cash edge
- **Bottom-7 events**: each contributes less than 4.5% of total attribution; the smallest 4 are single-day firings of < 3% SPY moves and contribute 1.1-2.6% each

The top-2 events are both inside the COVID crash (event 6 = pre-crash slide, event 7 = the crash itself). Treating them as a single macro event would shift the concentration analysis materially -- effectively one regime episode (Feb-Mar 2020) drives 43.7% of total attribution. The 2025-04 tariff event (event 13, +event 14 single-day adjacent firing) adds another ~12% combined. Three macro episodes (2020 COVID, 2018 Volmageddon/year-end, 2025 tariff) account for the majority.

## Verdict

**VERDICT: AMBIGUOUS**

The pre-registered decision criterion:
- FRAGILE if top-3 attribution share > 75% -> do NOT run E6
- ROBUST if top-3 attribution share < 50% -> proceed to E6
- AMBIGUOUS if 50%-75% -> analyst decides

Top-3 share = 53.6% sits just above the ROBUST boundary and well below the FRAGILE threshold. The headline number does not concentrate dangerously in 1-3 events, but the COVID episode (events 6+7 combined = 43.7% of attribution) is a single regime change and represents a meaningful overhang. Outside COVID and the 2025-04 tariff event, the remaining 11 firings each contribute < 6% of total attribution -- the distribution is reasonably even at the tail.

## Implications for Experiment 6

Recommendation: **conditional proceed to E6 with a robustness clause**. The verdict is AMBIGUOUS, not FRAGILE, so the formal V12c readiness run is not blocked. However:

1. E6 should compute V12c performance with the COVID period (Feb 2020 - Apr 2020) explicitly excluded as a robustness check. If V12c still beats V11 OOS without COVID, ROBUST is confirmed empirically. If V12c collapses to <= V11 without COVID, FRAGILE is confirmed and V12c deployment should be deferred.
2. The Sharpe contribution figure (+0.031 from top-3 out of +0.058 total) is a linearity proxy. A definitive answer requires per-day P&L from a V12-up-cash backtest replay -- worth running before E6 finalizes.
3. The 2025-04 tariff event is recent and OOS-adjacent; it may be partly responsible for V12c's apparent edge in the 2024-2026 segment. E6 should report the V12c segment-by-segment Sharpe to flag this.

## Limitations

- **No per-day V12-up-cash P&L was used**. Sharpe contribution is a proxy under a linearity assumption: each event's Sharpe-delta share equals its share of total absolute avoided-loss attribution. A rigorous version would re-run V12-up-cash with per-day P&L logging and compute the actual per-event Sharpe difference vs V11 (numerator = daily excess return, denominator = daily volatility of excess returns) -- the proxy collapses if some events have outsized volatility per unit return (e.g. event 7 spans 11 days of extreme intraday whipsaw and likely contributes disproportionately to volatility, not just to mean).
- **14-event sample is small**. Top-3 share is sensitive to single-event reclassification; if event 14 (2025-04-10 single day) is treated as a continuation of event 13 (2025-04-04..2025-04-08), the combined entity would rank top-3 and shift concentration figures slightly. The detector's debouncing parameter (currently 0) interacts directly with how runs are bucketed.
- **2017-2026 window overlaps the V11/V12 in-sample period**. Forward OOS validation is needed before any V12c deployment regardless of this verdict. The 2025-04 events sit at the OOS boundary and could be capturing detector overfit.
- **Avoided-loss sign convention**: positive avoided_loss means SPY went DOWN during the event (cash avoided the loss). All 14 events have positive avoided_loss in this dataset -- v0 UNPREDICTABLE fires only on volatile sell-offs in this window, which is itself a feature of the detector design and may not generalize OOS.
- **V11 reference is fully invested in equity during UNPREDICTABLE**. This is approximately true (UNPREDICTABLE is not a V11 cash trigger) but the actual V11 NAV-fraction-in-SPY on any given UNPREDICTABLE day depends on V11's other regime branches and position sizing. The proxy approximation here is to use SPY return directly as the V12-up-cash forgone return; per-day P&L would resolve this.

## Artifacts

- Analysis script: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\notebooks\research\experiment2_unpredictable_inspection.py`
- Per-event CSV: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\diagnostics\regime\unpredictable_events\per_event.csv`
- Verdict text: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\diagnostics\regime\unpredictable_events\verdict.txt`
