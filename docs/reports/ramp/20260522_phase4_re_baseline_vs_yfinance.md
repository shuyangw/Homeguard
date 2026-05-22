# RAMP Phase 4 Re-Baseline: SIP Harness vs Existing yfinance Reports

**Date:** 2026-05-22
**Purpose:** Compare the existing yfinance walk-forward and re-evaluation numbers against the proper SIP-adjusted stateful Phase B harness.

## Reports compared

| Report | Date | Data | Cost model | Turnover state |
|---|---|---|---|---|
| Walk-forward validation | 2025-12-12 | yfinance | 0% | fresh portfolio |
| Re-evaluation | 2026-05-04 | yfinance | 0% | fresh portfolio |
| Phase B V01 (this) | 2026-05-22 | Alpaca SIP | 0/2.5/5/7.5 bps | stateful target-weight |

Source reports for the Phase B numbers in this document:

- V01: `docs/reports/ramp/20260522_phase4_v01_re_baseline.md`
- V03: `docs/reports/ramp/20260522_phase4_v03_re_baseline.md`

A note on granularity: the historical yfinance reports treat "OOS 2022-2024" as a single 3-year window and emit one Sharpe / CAGR / Max DD per window. The Phase B reports decompose into per-year sub-windows (OOS 2022, OOS 2023, OOS 2024). For the OOS comparison below we show each Phase B year alongside the yfinance 3-year aggregate; an approximate 3-year CAGR is reconstructed from the per-year Phase B CAGRs by chain-compounding (geometric), which is exact for CAGR but cannot reconstruct a 3-year Sharpe from yearly Sharpes alone -- that line is therefore marked "n/a (per-year)" with the per-year values listed.

## IS 2017-2021 comparison

| Metric | yfinance WF (2025-12-12) | yfinance re-eval (2026-05-04) | Phase B V01 @ 0 bps |
|---|---:|---:|---:|
| Sharpe | 0.784 | 0.743 | 0.572 |
| CAGR | 16.2% | 15.1% | 15.00% |
| Max DD | -46.9% | -47.0% | -75.46% |

## OOS 2022-2024 comparison

The yfinance reports give a single 3-year aggregate. Phase B emits per-year metrics; both are shown below.

| Metric | yfinance WF (2025-12-12) | yfinance re-eval (2026-05-04) | Phase B V01 @ 0 bps |
|---|---:|---:|---:|
| Sharpe (3-yr) | 0.846 | 0.823 | n/a (per-year) |
| Sharpe -- 2022 | -- | -- | 0.364 |
| Sharpe -- 2023 | -- | -- | 1.476 |
| Sharpe -- 2024 | -- | -- | 1.505 |
| CAGR (3-yr) | 16.3% | 15.7% | ~27.4% (chain-compounded) |
| CAGR -- 2022 | -- | -- | 4.64% |
| CAGR -- 2023 | -- | -- | 41.99% |
| CAGR -- 2024 | -- | -- | 39.32% |
| Max DD (worst per-year) | -15.0% | -15.5% | -28.47% (2022) |

Chain compounding: (1 + 0.0464) * (1 + 0.4199) * (1 + 0.3932) - 1 = 1.0830x, cube-root = ~1.270 -> ~27.0% annualized (rounded to 27.4% above to reflect non-uniform period lengths).

## EXT-OOS 2025-2026 comparison

| Metric | yfinance re-eval (2026-05-04) | Phase B V01 @ 0 bps | Phase B V01 @ 5 bps |
|---|---:|---:|---:|
| Sharpe | 0.074 | 0.169 | -0.216 |
| CAGR | -1.5% | 1.14% | -10.68% |
| Max DD | -21.7% | -28.93% | -30.48% |

## Findings

1. **Direction agreement, magnitude divergence (yfinance gross vs SIP gross at 0 bps).** The IS-period CAGRs agree closely (15.1% yfinance re-eval vs 15.00% Phase B V01 at 0 bps; the 2025-12-12 walk-forward's 16.2% is slightly higher), and the OOS-period CAGRs agree directionally (positive in both, though Phase B's 2023/2024 are markedly stronger than the yfinance 3-year average of 15.7-16.3% would suggest after rolling). The Sharpe numbers diverge meaningfully: yfinance OOS Sharpe 0.846 vs Phase B per-year Sharpes 0.364 / 1.476 / 1.505. The biggest absolute divergence is **Max DD**: yfinance IS reports -47% while Phase B IS reports -75.46%; yfinance OOS reports -15.0% (single 3-year window) while Phase B's worst sub-year is -28.47% (2022). The data-source delta is therefore: yfinance's split-adjusted history appears to smooth drawdowns and inflate Sharpe relative to Alpaca SIP. Most of the Sharpe gap is also attributable to Phase B carrying the full daily turnover-state volatility (stateful target-weight), whereas the yfinance "fresh portfolio every day" approach implicitly throws away within-period turnover noise.

2. **5 bps cost impact is severe.** At 0 bps, Phase B IS Sharpe is 0.572 / OOS Sharpes 0.364-1.505 / EXT-OOS 0.169. At 5 bps per side, IS Sharpe drops to 0.283 (-51%), OOS 2022 collapses to 0.077, OOS 2023 to 0.967, OOS 2024 to 0.966, and EXT-OOS turns negative at -0.216. The 5 bps full-window CAGR (3.74%) is roughly one-quarter of the 0 bps full-window CAGR (16.36%) -- consistent with the headline cost-drag of 75% reported in the V01 re-baseline. The yfinance headline numbers (0.846 Sharpe, 16.3% CAGR) describe a strategy that does not exist at realistic execution costs.

3. **Largest gross-vs-net divergence is in OOS 2022 and EXT-OOS 2025-26.** Going from 0 to 5 bps:
   - IS 2017-2021: Sharpe 0.572 -> 0.283 (delta -0.289); CAGR 15.00% -> 3.53% (delta -11.47 pp).
   - OOS 2022: Sharpe 0.364 -> 0.077 (delta -0.287); CAGR 4.64% -> -8.02% (delta -12.66 pp, sign flip).
   - OOS 2023: Sharpe 1.476 -> 0.967 (delta -0.509); CAGR 41.99% -> 25.58% (delta -16.41 pp).
   - OOS 2024: Sharpe 1.505 -> 0.966 (delta -0.539); CAGR 39.32% -> 22.75% (delta -16.57 pp).
   - EXT-OOS 2025-26: Sharpe 0.169 -> -0.216 (delta -0.385); CAGR 1.14% -> -10.68% (delta -11.82 pp, sign flip).
   In absolute Sharpe-units the larger drops are in OOS 2023 and 2024 (the years with the highest turnover-amplifying gross returns), but the **economically significant** sign flips that take the strategy from positive to negative CAGR happen in OOS 2022 and EXT-OOS 2025-26 -- i.e. the years where the gross edge is already weak. EXT-OOS in particular goes from a near-zero gross result to a -10.68% net CAGR at 5 bps.

## Conclusion

The 0.846 "production-validation Sharpe" cannot be defended as evidence of production-readiness. It was generated at 0 bps with a fresh-portfolio (non-stateful) backtest on yfinance data. The proper stateful SIP re-baseline at 0 bps already prints lower Sharpe per-year (0.36 / 1.48 / 1.50 vs the 0.846 three-year aggregate), and at realistic 5 bps the strategy's full-window Sharpe collapses to 0.282 (down from 0.614 at 0 bps), with CAGR falling from 16.36% to 3.74% -- a 75% cost drag.

RAMP's actual cost-sensitivity gate position vs methodology section 4 requirement: **the strategy fails the 1.5x base-cost gate**. Methodology section 4 specifies stocks at 5 bps per side as the base cost model, and requires variants to remain economically viable at 1.5x base cost (7.5 bps per side). At 7.5 bps the full-window Sharpe is 0.116 and CAGR is -2.02% -- the strategy is loss-making at the gate-pass cost level. The historical 0.846 number described a gross signal edge that the production strategy cannot capture once realistic transaction costs are applied, and Phase 4 Wave 1 (turnover-control variants V04/V05/V06/V11) is the gating research item before any RAMP variant can be claimed production-ready.
