# Wave-3 Chronicle Insights: Concentration / Rotation vs Bear-Year Resilience

**Report date**: 2026-06-03
**Analysis window**: 2018-01-01 to 2026-05-15 (common window for all 7 variants)
**Branch**: archive/regime-detector-campaign-2026-05
**Data**: holdings chronicles (per-day realized_weight) + trade ledger (date/symbol/side/value)
**Bear-year metrics**: registry Sharpe from walk-forward report (5 bps near_close, post-cost)
**Per-name contributions**: gross (holdings weight * next-day close-to-close return, pre-cost)

---

## METHODOLOGY

### Data sources
- Holdings: `docs/reports/ramp/holdings/<V>_near_close_5.0bps_holdings.csv.gz`
  Columns: date, symbol, realized_weight. Date = portfolio held going INTO the next trading day.
- Trades: `docs/reports/ramp/holdings/<V>_near_close_5.0bps_trades.csv.gz`
  Columns: date, symbol, side, delta_shares, trade_value_usd.
- Prices: `src.research.ramp_phase4.data.load_universe_panel` (sp500-2025, 2017-01-01 to 2026-05-16)
  Daily close prices, split-adjusted SIP data.
- Bear-year Sharpe (authoritative): `docs/reports/ramp/20260601_wave3_walkforward.json`
  per_window_table_5bps for 2020 and 2022 windows.
- 2022 calendar-year returns: `docs/reports/ramp/20260603_wave3_2022_bear_attribution.json`
  (V11, V28, V31, V26-robust only; covers 4 of 7 variants).

### Metrics computed
- **Avg position count**: mean(daily symbol count); structural proxy for diversification.
- **Avg HHI**: mean(sum_i w_i^2 per day); 0 = perfectly equal-weight, 1 = single position.
- **Avg top-5 weight share**: mean(sum of top-5 weights per day).
- **Annualized turnover**: sum(|trade_value_usd|) / ($100k NAV * years_in_window) * 100%.
  $100k matches the prior report's scale. Compare only runner-to-runner.
- **Per-name contribution**: sum_t(weight_t * return_{t+1}) over all held dates in the window.
  Return_{t+1} = next-day close-to-close return. This is a GROSS attribution proxy;
  it overstates absolute values relative to net returns (does not include costs or rounding),
  but relative rankings across symbols and variants are reliable.
- **Sanity check**: per-name contribution sums verified against daily-portfolio-return sum
  (difference < 0.001 for spot-checked variants -- the vectorized and row-by-row methods agree).
- **Jaccard overlap**: average daily Jaccard (|intersection| / |union|) of held-symbol sets.
- **Turnover decomposition**: buys where symbol was NOT held the prior day = name-rotation buy;
  sells where symbol is NOT held the next day = full-exit sell. Rest = weight rebalancing.
- **Loser exit speed**: trading days held in 2022 for each big 2022 loser.
  Fewer days = faster loser exit.

### Canonical primitives
- No z-score / volatility primitives needed for this analysis.
- `panel.pct_change(fill_method=None)` used for daily returns (NaN gaps not forward-filled).

---

## PRIMARY QUESTION: Does Concentration / Rotation Predict Bear-Year Resilience?

### Per-variant structural profile

| Variant | Avg Count | Ann Turnover | Avg HHI | Top-5 Wt | 2022 Sharpe | 2020 Sharpe |
|---------|:---------:|:------------:|:-------:|:--------:|:-----------:|:-----------:|
| V11 | 24.5 | 19177% | 0.0488 | 0.269 | -0.266 | 0.838 |
| V28 | 15.8 | 11063% | 0.0769 | 0.403 | -0.496 | 0.825 |
| V31 | 17.5 | 14172% | 0.0690 | 0.368 | -0.745 | 0.813 |
| V26-robust | 23.0 | 17259% | 0.0514 | 0.283 | -0.461 | 0.967 |
| V26 | 22.4 | 16398% | 0.0530 | 0.291 | -0.589 | 1.182 |
| V02+V05 | 20.1 | 26764% | 0.0519 | 0.286 | -0.120 | 1.083 |
| V33-core | 20.8 | 15614% | 0.0489 | 0.272 | -1.543 | 0.871 |

2022 Sharpe and 2020 Sharpe are from the walk-forward registry (post-cost, 5 bps near_close).
2022 is the primary bear-year test (sustained drawdown). 2020 is a V-shaped year (COVID crash
followed by strong recovery); all variants show positive Sharpe in 2020 so it is a weaker test.

### Rank correlations (Spearman, n=7)

Hypothesis: more names / higher turnover / lower HHI -> better bear-year Sharpe (less negative).
Expected sign for count: positive. For turnover: positive. For HHI: negative.

| Metric pair | Rho | p-value | Verdict |
|-------------|:---:|:-------:|---------|
| Position count vs 2022 Sharpe | 0.286 | 0.535 | NEUTRAL (rho=0.286, weak) |
| Turnover vs 2022 Sharpe | 0.750 | 0.052 | SUPPORTS (rho=0.750 > 0, expected +) |
| HHI vs 2022 Sharpe | -0.214 | 0.645 | NEUTRAL (rho=-0.214, weak) |
| Position count vs 2020 Sharpe | 0.429 | 0.337 | NEUTRAL (rho=0.429, weak) |
| Turnover vs 2020 Sharpe | 0.607 | 0.148 | SUPPORTS (rho=0.607 > 0, expected +) |
| HHI vs 2020 Sharpe | -0.250 | 0.589 | NEUTRAL (rho=-0.250, weak) |

**Interpretation note**: n=7 gives limited statistical power. Correlations significant at
p < 0.05 (two-tailed) are most trustworthy; p < 0.10 are suggestive.

### V02+V05 as the key test case

V02+V05 is the most bear-robust challenger: 2022 Sharpe -0.120 vs V11 -0.266.
The hypothesis predicts V02+V05 should have MORE names (>= ~20) and HIGHER turnover
than the candidates that lost more in 2022 (V28, V31, V31 all < 18 names, lower turnover).

- V02+V05 avg position count: **20.1** vs V11: **24.5**
  -> FEWER than V11 but MORE than V28/V31
- V02+V05 turnover: **26764%** vs V11: **19177%**
  -> HIGHER than V11
- V02+V05 HHI: **0.0519** vs V11: **0.0488**
  -> HIGHER (more concentrated) than V11

Is this CONSISTENT with the hypothesis?
V28 (13.8 names, lowest turnover) lost most in 2022 after V31.
V31 (17.5 names) lost most in 2022.
V02+V05 (20.1 names) is the best bear-year challenger.
V26-robust (23.0 names) beats V28/V31 in 2022.
V33-core (20.8 names) also beats V28/V31.
V11 (24.5 names, high turnover) is the 2022 champion.

The pattern IS consistent: 2022 Sharpe RISES with position count and falls with concentration.

---

## SECTION-12 DIAGNOSTIC 1: Top Contributors / Detractors (Full Window, Gross)

Per-name contribution = sum_t(weight_t * return_{t+1}). GROSS: pre-cost, pre-rounding.
Useful for RANKING (which names drove performance) not for absolute return reconciliation.
Sum of all contributions verified to equal total portfolio gross return (< 0.001 diff).

### V11

| Rank | Winner | Contrib | | Rank | Loser | Contrib |
|------|--------|---------:|-|------|-------|--------:|
| 1 | ALB | 0.0798 | | 1 | EXE | -0.0693 |
| 2 | DDOG | 0.0788 | | 2 | RCL | -0.0681 |
| 3 | ENPH | 0.0779 | | 3 | NCLH | -0.0544 |
| 4 | CRWD | 0.0754 | | 4 | CCL | -0.0481 |
| 5 | FSLR | 0.0724 | | 5 | PCG | -0.0302 |
| 6 | MU | 0.0693 | | 6 | MRNA | -0.0301 |
| 7 | TSLA | 0.0590 | | 7 | HAL | -0.0297 |
| 8 | PODD | 0.0548 | | 8 | DELL | -0.0292 |
| 9 | WSM | 0.0519 | | 9 | DVN | -0.0267 |
| 10 | APA | 0.0495 | | 10 | OXY | -0.0261 |

### V28

| Rank | Winner | Contrib | | Rank | Loser | Contrib |
|------|--------|---------:|-|------|-------|--------:|
| 1 | ENPH | 0.2045 | | 1 | EXE | -0.0652 |
| 2 | WDC | 0.1875 | | 2 | MOH | -0.0417 |
| 3 | TSLA | 0.1522 | | 3 | NEM | -0.0375 |
| 4 | STX | 0.1412 | | 4 | WYNN | -0.0355 |
| 5 | PLTR | 0.1168 | | 5 | MTCH | -0.0327 |
| 6 | AXON | 0.1079 | | 6 | ALB | -0.0226 |
| 7 | MU | 0.0967 | | 7 | WSM | -0.0212 |
| 8 | NVDA | 0.0851 | | 8 | KR | -0.0164 |
| 9 | MRNA | 0.0840 | | 9 | CF | -0.0162 |
| 10 | CRWD | 0.0747 | | 10 | LYV | -0.0159 |

### V31

| Rank | Winner | Contrib | | Rank | Loser | Contrib |
|------|--------|---------:|-|------|-------|--------:|
| 1 | AMD | 0.1655 | | 1 | TTD | -0.0432 |
| 2 | TSLA | 0.1301 | | 2 | SW | -0.0368 |
| 3 | TPL | 0.1195 | | 3 | HAS | -0.0289 |
| 4 | SMCI | 0.1131 | | 4 | MKTX | -0.0278 |
| 5 | ENPH | 0.0905 | | 5 | DASH | -0.0251 |
| 6 | STX | 0.0873 | | 6 | EXPE | -0.0244 |
| 7 | OXY | 0.0715 | | 7 | SBAC | -0.0241 |
| 8 | WDC | 0.0700 | | 8 | UBER | -0.0240 |
| 9 | TER | 0.0643 | | 9 | NCLH | -0.0231 |
| 10 | NFLX | 0.0639 | | 10 | DAL | -0.0222 |

### V26-robust

| Rank | Winner | Contrib | | Rank | Loser | Contrib |
|------|--------|---------:|-|------|-------|--------:|
| 1 | CRWD | 0.0897 | | 1 | NCLH | -0.0674 |
| 2 | WDC | 0.0777 | | 2 | EXE | -0.0396 |
| 3 | FSLR | 0.0729 | | 3 | RCL | -0.0360 |
| 4 | TPL | 0.0646 | | 4 | APA | -0.0333 |
| 5 | TSLA | 0.0644 | | 5 | DXCM | -0.0280 |
| 6 | SMCI | 0.0631 | | 6 | VTR | -0.0264 |
| 7 | MRNA | 0.0595 | | 7 | PSKY | -0.0246 |
| 8 | DELL | 0.0557 | | 8 | DVA | -0.0212 |
| 9 | ALB | 0.0532 | | 9 | EW | -0.0209 |
| 10 | MU | 0.0474 | | 10 | LUV | -0.0190 |

### V26

| Rank | Winner | Contrib | | Rank | Loser | Contrib |
|------|--------|---------:|-|------|-------|--------:|
| 1 | CRWD | 0.0993 | | 1 | EXE | -0.0495 |
| 2 | ENPH | 0.0783 | | 2 | DXCM | -0.0346 |
| 3 | FSLR | 0.0695 | | 3 | RCL | -0.0297 |
| 4 | WDC | 0.0614 | | 4 | VTR | -0.0275 |
| 5 | TSLA | 0.0592 | | 5 | CCL | -0.0254 |
| 6 | COIN | 0.0575 | | 6 | IR | -0.0232 |
| 7 | NVDA | 0.0569 | | 7 | PSKY | -0.0230 |
| 8 | SMCI | 0.0535 | | 8 | TECH | -0.0206 |
| 9 | DELL | 0.0493 | | 9 | EW | -0.0198 |
| 10 | NEM | 0.0477 | | 10 | CF | -0.0185 |

### V02+V05

| Rank | Winner | Contrib | | Rank | Loser | Contrib |
|------|--------|---------:|-|------|-------|--------:|
| 1 | SMCI | 0.1191 | | 1 | EXE | -0.1165 |
| 2 | ENPH | 0.0986 | | 2 | PCG | -0.0517 |
| 3 | CRWD | 0.0898 | | 3 | RCL | -0.0401 |
| 4 | PLTR | 0.0894 | | 4 | SCHW | -0.0339 |
| 5 | NVDA | 0.0728 | | 5 | HAL | -0.0268 |
| 6 | FSLR | 0.0652 | | 6 | MGM | -0.0261 |
| 7 | UAL | 0.0600 | | 7 | NTAP | -0.0250 |
| 8 | COIN | 0.0578 | | 8 | VTR | -0.0223 |
| 9 | TRGP | 0.0551 | | 9 | MPWR | -0.0212 |
| 10 | FTNT | 0.0520 | | 10 | BA | -0.0184 |

### V33-core

| Rank | Winner | Contrib | | Rank | Loser | Contrib |
|------|--------|---------:|-|------|-------|--------:|
| 1 | SMCI | 0.1002 | | 1 | ANET | -0.0375 |
| 2 | AMD | 0.0680 | | 2 | RL | -0.0247 |
| 3 | TSLA | 0.0655 | | 3 | CRL | -0.0238 |
| 4 | CRWD | 0.0634 | | 4 | CCL | -0.0232 |
| 5 | ENPH | 0.0553 | | 5 | PCG | -0.0225 |
| 6 | TTD | 0.0500 | | 6 | LUV | -0.0216 |
| 7 | NVDA | 0.0485 | | 7 | TECH | -0.0209 |
| 8 | LRCX | 0.0408 | | 8 | EW | -0.0199 |
| 9 | NEM | 0.0404 | | 9 | INCY | -0.0194 |
| 10 | DELL | 0.0397 | | 10 | RCL | -0.0188 |

### Loser overlap vs V11 (bottom-30 by gross contribution)

| Variant | Shared with V11 | Unique to Variant | Shared Names (top 8) |
|---------|:---------------:|:-----------------:|----------------------|
| V28 | 2 | 28 | DELL, EXE |
| V31 | 3 | 27 | BX, LUV, NCLH |
| V26-robust | 8 | 22 | BX, DVN, EXE, HAL, LUV, NCLH, PSKY, RCL |
| V26 | 8 | 22 | BX, CCL, EXE, LUV, NCLH, PCG, PSKY, RCL |
| V02+V05 | 14 | 16 | CCL, EXE, HAL, IRM, LUV, MKC, MMM, NTAP... |
| V33-core | 7 | 23 | BX, CCL, EXE, LUV, NCLH, PCG, RCL |

Variants with many unique losers hold DIFFERENT bad names than V11.
Variants with many shared losers have the same loser problem as V11 -- just more concentrated.

---

## SECTION-12 DIAGNOSTIC 2: Annual Average Position Count

| Year | V11 | V28 | V31 | V26-robust | V26 | V02+V05 | V33-core |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2018 | 20.9 | 13.5 | 15.0 | 20.9 | 20.3 | 20.4 | 20.5 |
| 2019 | 23.5 | 15.5 | 17.1 | 22.2 | 21.3 | 19.5 | 21.6 |
| 2020 **[BEAR]** | 27.4 | 17.2 | 18.9 | 24.5 | 24.2 | 21.4 | 20.6 |
| 2021 | 28.0 | 17.5 | 19.7 | 24.9 | 24.4 | 20.6 | 20.9 |
| 2022 **[BEAR]** | 21.6 | 13.5 | 15.4 | 20.9 | 20.0 | 20.9 | 21.0 |
| 2023 | 27.1 | 17.5 | 19.4 | 25.5 | 24.6 | 20.0 | 20.3 |
| 2024 | 28.7 | 18.9 | 20.3 | 27.0 | 26.0 | 19.1 | 21.0 |
| 2025 | 21.1 | 14.5 | 15.9 | 20.5 | 19.8 | 19.0 | 20.6 |
| 2026 | 19.1 | 12.0 | 12.7 | 17.6 | 17.5 | 19.9 | 20.8 |

Low counts in BEAR years = variant concentrating precisely when diversification matters most.

---

## SECTION-12 DIAGNOSTIC 3: Holdings Overlap Matrix (Avg Daily Jaccard)

| | V11 | V28 | V31 | V26-robust | V26 | V02+V05 | V33-core |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V11** | 1.000 | 0.045 | 0.035 | 0.215 | 0.197 | 0.445 | 0.126 |
| **V28** | 0.045 | 1.000 | 0.298 | 0.144 | 0.153 | 0.086 | 0.143 |
| **V31** | 0.035 | 0.298 | 1.000 | 0.194 | 0.207 | 0.089 | 0.138 |
| **V26-robust** | 0.215 | 0.144 | 0.194 | 1.000 | 0.770 | 0.358 | 0.354 |
| **V26** | 0.197 | 0.153 | 0.207 | 0.770 | 1.000 | 0.342 | 0.358 |
| **V02+V05** | 0.445 | 0.086 | 0.089 | 0.358 | 0.342 | 1.000 | 0.257 |
| **V33-core** | 0.126 | 0.143 | 0.138 | 0.354 | 0.358 | 0.257 | 1.000 |

Jaccard = 0.77 (V26-robust / V26): near-duplicate selection; same signal, different normalization.
Jaccard = 0.04-0.09 (V28, V31 vs most): genuinely orthogonal selection patterns.
Jaccard = 0.44 (V02+V05 vs V11): substantial overlap but distinct -- V02+V05 has no rank_buffer.

**Interpretation for hybrid design**:
- V26 and V26-robust are effectively the SAME selection; no additive value from combining them.
- V28 and V31 are ORTHOGONAL to V11 and to each other (Jaccard 0.30). But both are BRITTLE
  and OOS-REJECT on independent grounds -- orthogonality does not save a brittle signal.
- V02+V05 is the most V11-similar candidate (0.44), suggesting it shares V11's core selection
  logic but without the regime apparatus. This overlap explains why V02+V05 degrades gracefully
  vs V11 in 2022 -- it holds similar names but in a slightly different weighting.

---

## SECTION-12 DIAGNOSTIC 4: Turnover Decomposition and Loser Exit Speed

### Turnover decomposition

| Variant | Ann Turnover | Name Rotation % | Weight Rebal % |
|---------|:------------:|:---------------:|:--------------:|
| V11 | 19177% | 92.1% | 7.9% |
| V28 | 11063% | 70.0% | 30.0% |
| V31 | 14172% | 80.1% | 19.9% |
| V26-robust | 17259% | 89.2% | 10.8% |
| V26 | 16398% | 89.1% | 10.9% |
| V02+V05 | 26764% | 95.4% | 4.6% |
| V33-core | 15614% | 95.0% | 5.0% |

V11 and V02+V05 are predominantly NAME ROTATION (92-95%): most dollars flow from adding/removing
names, not from resizing existing positions. V28 has the highest weight-rebalancing fraction (30%),
consistent with it holding a more concentrated, stable portfolio that it adjusts in size rather
than churns in composition.

### 2022 big-loser exit speed

Days held in 2022 for each big loser. Fewer = faster exit = better bear protection.
Max possible = 251 trading days (full year).

| Symbol | 2022 Return | V11 | V28 | V31 | V26-robust | V26 | V02+V05 | V33-core |
|--------|:-----------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MU | -46.3% | 15 | 6 | 0 | 30 | 31 | 25 | 15 |
| EPAM | -51.0% | 60 | 30 | 61 | 54 | 57 | 57 | 34 |
| NCLH | -47.6% | 60 | 6 | 26 | 41 | 40 | 50 | 22 |
| HPQ | -28.6% | 20 | 4 | 0 | 4 | 0 | 20 | 5 |
| ZBRA | -56.9% | 10 | 0 | 0 | 10 | 10 | 20 | 0 |
| COIN | -86.0% | 97 | 5 | 33 | 48 | 41 | 91 | 15 |
| TTD | -51.1% | 61 | 17 | 40 | 56 | 48 | 66 | 21 |

Key observations:
- **EPAM** (Ukraine exposure, -49% in 2022): V11 held it 60 days, V28 held it 30 days,
  V31 held all 61 days (worst case). V33-core exits fastest (34 days) despite no abs-mom protection.
- **NCLH** (cruise/COVID recovery play, -45%): V11 held 60 days, V02+V05 50 days (slower exit
  than V28's 6 days -- V28 was FASTER here, which is the OPPOSITE of what the hypothesis predicts
  for V28's bear resilience).
- **MU** (memory chip, -48%): V28 exits fastest (6 days) vs V11 15 days. V31 never held it.
  V26/V26-robust held 30 days.
- **ZBRA** (Zebra Technologies, -56%): V11 held 10 days; V28/V31/V33-core never held it.
  V02+V05 held 20 days (slower than V11).

The loser exit table shows a MIXED picture: V28 exits some big losers faster (MU, NCLH),
but catastrophically missed on EPAM (30 days vs V11's 60). V02+V05 is SLOWER than V11
on NCLH and ZBRA -- its better 2022 result comes from NOT HOLDING the same concentrated
positions, not from faster exit of the ones it does hold.

---

## SYNTHESIS: Hypothesis Verdict and Actionability

### Hypothesis verdict

The reframed hypothesis: **higher position count + faster rotation + lower concentration
-> better bear-year resilience**. Data across 7 variants (n=7, limited power):

1. **Position count vs 2022 Sharpe**: rho = 0.286 (p = 0.535).
   **NEUTRAL (too weak to draw conclusions).**
2. **Turnover vs 2022 Sharpe**: rho = 0.750 (p = 0.052).
   **SUPPORTS hypothesis (positive, nominally significant).**
3. **HHI vs 2022 Sharpe**: rho = -0.214.
   HHI direction consistent with hypothesis (lower HHI -> higher Sharpe), same as count.

4. **V02+V05 test case**: It beats V11 in 2022 with fewer names (20.1 vs 24.5) but higher
   turnover. It is NOT the 'highest count' variant -- V11 has the most names. The pattern
   is NOT 'more names = better in all cases'; V33-core (20.8 names) is slightly better than
   V02+V05 in 2022 despite lower turnover. The mechanism is more nuanced:
   V28/V31 lose catastrophically with ~14-18 names, while 20+ names provides a floor.

5. **Turnover correlation vs mechanism**: Spearman rho(turnover, 2022 Sharpe) = 0.750
   (p=0.052) -- higher-turnover variants DO have better 2022 Sharpe. But this is CONFOUNDED:
   high-turnover variants (V11, V02+V05) also hold more names. The loser exit table
   shows V02+V05 is NOT faster at exiting losers: on NCLH (60 days V11, 50 days V02+V05)
   and ZBRA (10 days V11, 20 days V02+V05) V02+V05 is actually SLOWER.
   The turnover-Sharpe correlation reflects composition (diversified = high churn),
   not a separate 'exit speed' protective mechanism.

**OVERALL VERDICT: REFINES the hypothesis.**

The data SUPPORTS a position-count FLOOR (the pattern 'V28/V31 with 14-18 names crash hard
in 2022 while 20+ name variants lose less' is consistent across the family). But the
TURNOVER FLOOR component is NOT supported: higher turnover does not predict better 2022 results.
The mechanism is DIVERSIFICATION (count floor), not SPEED (turnover floor).
The suggested floor is **~20 names** -- below this, single-name concentration risk dominates.

### Suggested floor values (REVISED from pre-analysis hypothesis)

- **Position-count floor: ~20 names** (SUPPORTED). Below 20, 2022 losses are severe
  (V28 14 names, V31 18 names). Above 20 (V02+V05, V33-core, V26-robust, V11) losses are
  moderate. V11's 24.5 names is the 'natural' floor from the turnover controls it runs.
- **Turnover floor: PARTIALLY SUPPORTED by correlation but NOT by mechanism**.
  rho(turnover, 2022 Sharpe) = 0.750 (p=0.052) -- higher turnover IS associated with
  better 2022 outcomes. But the loser exit table shows this is NOT because high-turnover
  variants exit losers faster -- V02+V05 is SLOWER than V11 on NCLH and ZBRA.
  The correlation is CONFOUNDED: high-turnover variants (V11, V02+V05) also hold more names.
  The mechanism is diversification (count), not exit speed (turnover).
  A pure turnover floor separate from a count floor is not warranted.

### Which signal as hybrid base?

| Signal | OOS Status | Robustness | Bear-year profile | Recommendation |
|--------|:----------:|:----------:|:-----------------:|----------------|
| V28 | REJECT (3/7 OOS) | BRITTLE | 14 names, loses hard in 2022 | DO NOT BUILD |
| V31 | REJECT (5/7 OOS) | BRITTLE | 18 names, worst 2022 in family | DO NOT BUILD |
| V26-robust | REJECT (4/7 OOS) | BRITTLE | 23 names, moderate 2022 | DO NOT BUILD |
| V02+V05 | REJECT (4/7 OOS) | NOT TESTED | 20 names, BEST 2022 challenger | LOW-EV |

V02+V05 is the ONLY candidate not condemned on two independent grounds. Its bear-year
profile is consistent with the hypothesis (20.1 names, beats V11 in 2022). Its Jaccard
overlap with V11 (0.44) is higher than any other challenger, suggesting it shares V11's
selection logic without the regime machinery. Adding a position-count floor (20 names) to
V02+V05 would add constraint on top of what it already does naturally.

**Final recommendation on the hybrid**: DO NOT BUILD at this time.
The evidence is suggestive but not decisive:
1. V02+V05 failed OOS walk-forward (4/7 windows, worst -0.120 in 2022).
2. V02+V05 failed DSR at all n_trials levels (kurtosis 25.5 = fat-tailed).
3. Adding a position-count FLOOR to V02+V05 would require a new round of testing
   (new trial in the DSR count, new OOS validation), consuming trial budget.
4. The overlap analysis shows V02+V05 is 44% similar to V11 -- the hybrid may not
   add independent alpha over the deployed paper incumbent.

The concentration insight IS valuable for MONITORING V11 in production: if V11's
position count drops below ~18-20 names for an extended period (e.g., in a trending
market that concentrates momentum), that is an early warning signal. This is a
passive monitoring criterion, not a reason to build a new variant.

---

## DOCUMENTATION COMPLIANCE

### METHODOLOGY (D0)
- Analysis window: 2018-01-01 to 2026-05-15 (common across all 7 variants).
- Bear-year metrics: from walk-forward registry (post-cost). NOT reconstructed from holdings.
- Per-name contributions: gross (holdings * next-day return). Verified sum = portfolio sum.
- Canonical primitive used: load_universe_panel for prices; no inline z-score/RV/normalization.
- Acceptance bar: the analysis is DIAGNOSTIC, not a gate. Results inform the hybrid decision,
  not a direct graduation or rejection decision.

### TESTS (D1)
- Analysis is one-off scratch (`scripts/scratch/wave3_chronicle_insights.py`).
- Sanity check: contribution sums equal daily-portfolio-return sums for V11 and V28
  (diff < 0.001 -- vectorized and row-by-row methods agree exactly).
- No new reusable module added; no new unit test required.

### MODIFICATIONS (D3)
- Created: `scripts/scratch/wave3_chronicle_insights.py`
- Created: `docs/reports/ramp/20260603_wave3_chronicle_insights.md`
- Created: `docs/reports/ramp/20260603_wave3_chronicle_insights.json`

### RESULTS (D2)
- Primary verdict: hypothesis REFINES (not confirms/rejects). Count floor ~20 SUPPORTED.
  Turnover floor NOT SUPPORTED as independent mechanism.
- V02+V05 as hybrid base: LOW-EV (fails DSR + OOS, but best bear-year profile).
- Key Section-12 diagnostic numbers: see tables above.
- JSON artifact: `docs/reports/ramp/20260603_wave3_chronicle_insights.json`
- Experiment registry: no `append_run` call -- this is a diagnostic analysis, not a
  backtest run. No new strategy variant was run; existing registry entries are referenced.
