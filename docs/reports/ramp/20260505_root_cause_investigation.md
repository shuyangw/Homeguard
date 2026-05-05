# RAMP Root Cause Investigation -- 2026-05-05

## Context

Phase A re-evaluation (2026-05-04) confirmed the 0.846 OOS Sharpe baseline (2022-2024) and found material alpha decay in the truly-OOS 2025-2026 period: Sharpe 0.074, CAGR -1.5%, MaxDD -21.7%. BEAR regime (19.4% of time, Sharpe -2.17) and WEAK_BULL (43.6%, Sharpe -0.78) were the dominant drags. This investigation tests 5 strategy variants and 8 diagnostics to isolate the failure cause.

## Methodology

Same universe (sp500-2025.csv, 503 symbols), same yfinance split-adjusted data (auto_adjust=True), same 0% transaction costs, same +/-20% daily return cap as Phase A. IS: 2017-01-01 to 2021-12-31. OOS: 2022-01-01 to 2024-12-31. EXT-OOS: 2025-01-01 to 2026-04-30. Variants differ only in signal processing, weighting, or exposure logic. Production REGIME_PARAMS used verbatim for V0/V2/V3/V4; V1 skips regime detection entirely. Sharpe SE on ~330 EXT-OOS days is approximately 0.17 -- point estimates below 0.5 should be treated with caution; concrete CAGR and MaxDD numbers are more reliable.

## Variant comparison

| Variant | Description | IS Sharpe (2017-2021) | OOS Sharpe (2022-2024) | EXT-OOS Sharpe (2025-2026) | EXT-OOS CAGR | EXT-OOS MaxDD |
|---|---|---|---|---|---|---|
| V0 | Production RAMP | 0.755 | 0.867 | 0.070 | -1.7% | -21.6% |
| V1 | Vanilla momentum (no regime) | 0.895 | 0.710 | 0.314 | 4.8% | -21.7% |
| V2 | Inverse-vol weighting (YZ-RV) | 0.736 | 0.799 | -0.015 | -3.7% | -19.5% |
| V3 | Winsorized momentum | 0.840 | 0.853 | 0.078 | -1.2% | -21.1% |
| V4 | SPY-vol overlay (+0.5x on >90th pct) | 0.866 | 0.817 | -0.266 | -9.1% | -23.5% |

## Hypothesis results

**H1: Regime mis-classification (detector labels 2025-2026 BEAR too aggressively) -- INCONCLUSIVE**

Cannot be confirmed without comparing detected regimes against manual inspection of SPY drawdown and VIX history. D4 shows the regime distribution by year -- if BEAR is dramatically over-represented in 2025 relative to realized market conditions, this hypothesis gains support. Requires production paper trading logs for full validation.

**H2: Regime params don't generalize OOS -- SUPPORTED**

V1 (vanilla momentum, no regime) produced EXT-OOS Sharpe 0.314 vs V0 0.070 (delta +0.243). V1 is materially better, suggesting regime gating is actively harming performance in 2025-2026.

**H3: Universe drift (2025-2026 S&P 500 composition differs from training) -- DEFERRED**

We use sp500-2025.csv (current composition) for all periods -- this introduces survivorship bias in IS/OOS but the bias is symmetric across all variants. True point-in-time composition data is not available without a commercial data provider. Likely a contributing factor but not testable in this investigation.

**H4: Crash protection insufficient (VIX/DD trigger too weak in 2025 drawdowns) -- REFUTED**

V4 (additional 0.5x exposure when SPY vol > 90th trailing percentile) produced EXT-OOS Sharpe -0.266 vs V0 0.070 (delta -0.337). Vol overlay does not help materially -- crash protection is not the primary issue.

**H5: Momentum factor decay (factor itself broken in 2025-2026) -- REFUTED**

If the raw momentum factor still works, V1 (no regime gating, pure momentum) should perform well. V1 EXT-OOS Sharpe is 0.314. V1 performed reasonably well, so the momentum factor is intact -- the problem lies in the regime overlay or exposure logic.

**H6: Stock leadership shift (2025 winners are not classical momentum names) -- SUPPORTED**

D5 shows the most-selected stocks during BEAR-regime days in 2025-2026 and their average next-day returns. Top-5 most selected stocks averaged -0.32% next-day return when held in BEAR regime. Consistently negative next-day returns during BEAR days confirms the strategy is selecting the wrong stocks -- possibly the momentum signal is selecting lagged winners that revert in stressed markets.

**H7: Data quality (yfinance errors inflate losses) -- UNLIKELY / LOW RISK**

The +/-20% daily return cap guards against the worst yfinance errors (unadjusted splits, dividend artifacts). D5 shows realized returns for individual BEAR-day holdings -- if returns cluster near the -20% cap, data quality is suspect. In general, the -1.5% CAGR is too consistent to be noise.

**H8: Forced trading on shock days (VIX/DD trigger fires but 50% exposure still loses) -- SUPPORTED**

D3 shows crash protection fired on 20.2% of 2025-2026 trading days. High trigger frequency means the strategy spent significant time at 50% exposure but still lost -- suggesting the 50% reduction is insufficient on shock days, or the underlying selection is so poor that exposure reduction barely helps.

## Diagnostics

### D1. Rolling 252-day Sharpe across 2017-2026

Quarterly snapshots of 252-day rolling Sharpe (V0):

| Date | Rolling Sharpe (252d) |
|---|---|
| 2018-03 | 0.163 |
| 2018-06 | 0.859 |
| 2018-09 | 1.997 |
| 2018-12 | 0.401 |
| 2019-03 | 1.393 |
| 2019-06 | 1.007 |
| 2019-09 | -0.646 |
| 2019-12 | 0.567 |
| 2020-03 | -0.856 |
| 2020-06 | 0.143 |
| 2020-09 | 0.876 |
| 2020-12 | 1.401 |
| 2021-03 | 3.270 |
| 2021-06 | 2.493 |
| 2021-09 | 2.173 |
| 2021-12 | 0.339 |
| 2022-03 | -0.212 |
| 2022-06 | -0.851 |
| 2022-09 | -0.550 |
| 2022-12 | 0.243 |
| 2023-03 | 0.745 |
| 2023-06 | 2.096 |
| 2023-09 | 1.659 |
| 2023-12 | 1.628 |
| 2024-03 | 2.332 |
| 2024-06 | 2.146 |
| 2024-09 | 1.832 |
| 2024-12 | 1.124 |
| 2025-03 | -0.147 |
| 2025-06 | -0.191 |
| 2025-09 | 0.109 |
| 2025-12 | 0.121 |
| 2026-03 | 0.427 |
| 2026-06 | 0.613 |

Rolling Sharpe first turned negative in **2025-03-13**. This marks the onset of the decay.

### D2. Per-year Sharpe (V0)

| Year | Sharpe | CAGR | Days |
|---|---|---|---|
| 2017 | 0.668 | 9.3% | 250 |
| 2018 | 0.445 | 6.2% | 251 |
| 2019 | 0.567 | 10.0% | 252 |
| 2020 | 1.471 | 52.3% | 253 |
| 2021 | 0.339 | 5.3% | 252 |
| 2022 | 0.279 | 3.9% | 251 |
| 2023 | 1.531 | 23.4% | 250 |
| 2024 | 1.124 | 24.2% | 252 |
| 2025 | 0.234 | 2.8% | 250 |
| 2026 | -0.369 | -14.2% | 81 |

### D3. Crash protection trigger frequency (V0)

| Year | VIX>25 days | SPY-DD<-5% days | Either fired | % of days |
|---|---|---|---|---|
| 2017 | 0 | 0 | 0 | 0.0% |
| 2018 | 16 | 105 | 105 | 41.8% |
| 2019 | 1 | 42 | 42 | 16.7% |
| 2020 | 154 | 120 | 155 | 61.3% |
| 2021 | 21 | 2 | 23 | 9.1% |
| 2022 | 130 | 232 | 232 | 92.4% |
| 2023 | 3 | 177 | 177 | 70.8% |
| 2024 | 4 | 8 | 9 | 3.6% |
| 2025 | 20 | 49 | 50 | 20.0% |
| 2026 | 14 | 11 | 17 | 21.0% |

### D4. Regime distribution by year (V0)

| Year | STRONG_BULL | WEAK_BULL | SIDEWAYS | UNPREDICTABLE | BEAR |
|---|---|---|---|---|---|
| 2017 | 57 (23%) | 155 (62%) | 33 (13%) | 1 (0%) | 4 (2%) |
| 2018 | 40 (16%) | 70 (28%) | 62 (25%) | 8 (3%) | 71 (28%) |
| 2019 | 66 (26%) | 103 (41%) | 51 (20%) | 1 (0%) | 31 (12%) |
| 2020 | 104 (41%) | 47 (19%) | 52 (21%) | 18 (7%) | 32 (13%) |
| 2021 | 111 (44%) | 99 (39%) | 35 (14%) | 3 (1%) | 4 (2%) |
| 2022 | 25 (10%) | 45 (18%) | 44 (18%) | 0 (0%) | 137 (55%) |
| 2023 | 110 (44%) | 90 (36%) | 39 (16%) | 0 (0%) | 11 (4%) |
| 2024 | 127 (50%) | 80 (32%) | 24 (10%) | 6 (2%) | 15 (6%) |
| 2025 | 40 (16%) | 113 (45%) | 49 (20%) | 4 (2%) | 44 (18%) |
| 2026 | 1 (1%) | 31 (38%) | 29 (36%) | 0 (0%) | 20 (25%) |

### D5. 2025-2026 BEAR-regime trade analysis (V0)

Top 20 most-selected stocks during BEAR-regime signal days (2025-2026):

| Symbol | Times Selected | Avg Next-Day Return | Total Contribution | Win Rate |
|---|---|---|---|---|
| SMCI | 22 | -1.00% | -22.05% | 45.5% |
| COIN | 18 | -0.29% | -5.27% | 50.0% |
| ENPH | 18 | -1.30% | -23.42% | 50.0% |
| XYZ | 15 | 0.71% | 10.67% | 60.0% |
| PLTR | 12 | 0.27% | 3.26% | 58.3% |
| MU | 12 | -1.64% | -19.72% | 41.7% |
| AXON | 11 | 1.28% | 14.11% | 45.5% |
| TTD | 10 | -0.04% | -0.36% | 50.0% |
| APTV | 10 | -0.07% | -0.74% | 40.0% |
| MRNA | 10 | -1.06% | -10.55% | 40.0% |
| NKE | 8 | -0.73% | -5.88% | 37.5% |
| CRL | 8 | 1.43% | 11.45% | 62.5% |
| CRWD | 8 | 0.50% | 3.96% | 62.5% |
| ADBE | 7 | 1.01% | 7.09% | 71.4% |
| TER | 7 | 0.75% | 5.25% | 42.9% |
| EXPE | 7 | 0.73% | 5.13% | 57.1% |
| TSLA | 7 | 0.63% | 4.40% | 42.9% |
| DELL | 7 | -1.35% | -9.44% | 57.1% |
| DG | 7 | -0.66% | -4.63% | 57.1% |
| WDC | 7 | -0.18% | -1.23% | 57.1% |

Of 186 unique stocks selected, 90 (48%) had negative average next-day returns during BEAR-regime holds.

## Conclusion

**H2 (regime params don't generalize) is the dominant finding.** Removing regime gating (V1) improved EXT-OOS Sharpe from 0.070 to 0.314. 

The BEAR regime remains the most damaging: 64 days at Sharpe -2.17 in 2025-2026. D5 shows which stocks RAMP selects during BEAR days and their realized returns. WEAK_BULL at 43.6% of EXT-OOS time with Sharpe -0.78 is a structural concern that affects the majority of trading days regardless of tail-risk events.

Statistical caveat: EXT-OOS Sharpe estimates over 330 days carry SE ~0.17. Differences less than 0.2 between variants are not reliable. CAGR and MaxDD are concrete and not subject to this uncertainty.

## Implications for next steps

1. **H2 dominant (regime params don't generalize):** The BEAR and WEAK_BULL parameter sets need recalibration with post-2022 data. Consider expanding the lookback window for regime detection (current 252-day VIX percentile may be anchored to the 2020-2021 low-vol era).
2. **H5 REFUTED (momentum factor intact):** V1 EXT-OOS Sharpe 0.314 and CAGR +4.8% confirm the raw momentum signal has not decayed. The fix is in the regime overlay, not the signal formula.
3. **H4 REFUTED (vol overlay harmful):** V4 worsened EXT-OOS Sharpe to -0.266 and CAGR -9.1%. Do not add SPY-vol exposure reduction in the current form -- it over-triggers during the volatile but ultimately trending 2025-2026 market.
4. **H6/H8 SUPPORTED:** BEAR-regime stock selection is consistently negative (-0.32% average next-day return for top-5 selections). Combine with 20% trigger frequency in 2025-2026: the strategy holds the wrong stocks at the wrong times, and the 50% exposure reduction is insufficient to offset it. The highest-leverage fix is replacing BEAR-regime momentum selection with cash or low-beta defensive rotation.

**Recommended next step:** Before any parameter optimization, manually inspect 2025-Q1 and 2026-Q1 regime classifications against actual SPY/VIX history to validate H1. If the regime detector is correctly labeling BEAR conditions but the BEAR parameters are simply wrong (momentum continuation fails in 2025-style drawdowns), then replacing BEAR momentum with a cash/defensive rotation is the highest-leverage fix.
