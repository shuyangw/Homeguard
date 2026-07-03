# Futures Carry -- Deep-Dive Expansion (concentration diagnosis + de-concentration plan)

**Date:** 2026-07-03 - **Author:** Strategy Lead - **Type:** analysis / write-up only (no code)
**Under review:** absolute (Carver-style) carry, 33 roots, $10M, 2010-06..2026-02.
**Source of record:** `docs/reports/futures/CARRY_BROAD_READINESS.md` (registry run_id `2b9a02bc-...`),
design `docs/strategies/research/20260702_FUTURES_CARRY_STRATEGY_DESIGN.md`,
impl `src/strategies/advanced/futures_carry_strategy.py`, calc `src/data/carry_calculator.py`.

## 0. Executive summary

- Carry is the strongest signal on the board: **OOS Sharpe 0.88 (1x) / 0.87 (1.5x)** -- essentially
  cost-insensitive, PSR/DSR = 1.00 at the honest trial=1 -- and it is **positive in 10 of 11 scored
  windows**. It did **not** gate-fail for lack of edge. It gate-failed on **concentration**:
  **PBO 0.632** (windows-as-columns CSCV) with **skew +1.85, kurtosis 33.5**.
- The failure is temporal/directional, not tail-cosmetic. Window Sharpes range **-0.57 (W9, 2021-22)
  to +2.42 (W7, 2019-20)**; a handful of windows carry the aggregate, so the best window's OOS rank
  flips below median under resampling -> high PBO. This is "performance concentrated in a few
  regimes," which is exactly what PBO is built to catch.
- **Critical integrity finding that reframes the whole result:** in *this* 33-root universe the six
  rate roots (ZT/ZF/ZN/TN/ZB/UB) all hit `CarryCalculator`'s **bond SOFR v1 fallback -> carry = 0**.
  So the popular "bond carry rode the 2010-2021 falling-rate regime" thesis **does not apply here** --
  the rates sleeve is muted to zero. The realized concentration is therefore in **FX + energy +
  equity-index carry**, and 6/33 roots are dead weight, *understating* diversification.
- Skew/kurt is most plausibly a **vol-estimation-lag x correlated-shock** artifact centered on the
  **COVID windows (W7 +2.42, W8 +1.79)** amplified by **lumpy energy carry** (deep contango/backwardation
  swings, the April-2020 negative-WTI regime). The 25-day vol window undersizes going into the shock.
- **Ranked de-concentration levers:** (1) cross-sectional demeaning, (2) carry+trend combine,
  (3) IDM / instrument-risk weighting, (4) per-cluster risk caps, (5) slower/blended vol-scaling.
  #1 and #2 are the PBO levers; #3-#5 are mostly kurtosis levers. A gate-passing carry keeps
  Sharpe meaningfully >0 (target >= 0.4 demeaned-only, >= 0.7 combined) while pulling PBO < 0.25.
- **Discipline constraints that bind the whole plan:** stay parameter-free (prefer demean/combine/IDM,
  which are doctrine, over caps/vol-windows, which introduce tunables); pre-commit to ONE variant so
  DSR trial count stays at 1 (sweeping levers and picking best-PBO is selection -> DSR must deflate);
  resolve the two **nan windows (W11/W12, 2023-25)** before trusting the aggregate.

## 1. The per-window picture (what the readiness table is telling us)

Windows are train=36m / test=12m / step=12m, test start = 2013-06. Approx test years and OOS Sharpe:

| W | Test year (approx) | OOS Sharpe | Read |
|---|---|---|---|
| 1 | 2013-14 | 0.728 | normal |
| 2 | 2014-15 | 1.522 | oil crash begins (energy contango) |
| 3 | 2015-16 | 0.441 | weakest scored non-negative |
| 4 | 2016-17 | 1.365 | strong |
| 5 | 2017-18 | 0.837 | normal |
| 6 | 2018-19 | 1.278 | strong |
| 7 | 2019-20 | **2.418** | **COVID -- the dominant positive outlier** |
| 8 | 2020-21 | 1.786 | post-COVID reflation |
| 9 | 2021-22 | **-0.570** | **the only negative -- inflation/rate-hike shock** |
| 10 | 2022-23 | 1.248 | carry recovers under high-but-stable rates |
| 11 | 2023-24 | **nan** | **no scored Sharpe -- diagnostic gap** |
| 12 | 2024-25 | **nan** | **no scored Sharpe -- diagnostic gap** |
| 13 | 2025-26 (partial) | 0.871 | normal |

Two structural facts jump out:

1. **W7 (COVID) alone is a 2.42-Sharpe window** and W8 backs it at 1.79. Two adjacent windows around
   one macro event contribute a hugely disproportionate share of the aggregate and of the +1.85 skew
   (a right tail of large positive days, consistent with risk-off flight-to-safety + energy dislocation
   paying a net-long-carry basket, and with vol-target undersizing into the vol spike).
2. **W9 is the lone loser (-0.57)** and it sits exactly on the 2021-22 inflation/hiking regime break.
   A directional carry basket that made its best money in the risk-off COVID rally gives some of it
   back when the macro regime flips. That co-movement of window outcomes with one macro axis is the
   mechanical driver of PBO 0.63: the "good windows" and "bad window" are selected by regime, so their
   rank does not generalize under CSCV resampling.

**nan windows W11/W12 (2023-24, 2024-25):** these are two of the three most recent, most
decision-relevant windows (the rate-normalization regime), and they produce no Sharpe. Before any
aggregate number is trusted, this must be explained -- insufficient OOS days in the stitched segment,
a data phase-in gap, or a stitching artifact. A carry result that is blind to 2023-25 is not a
finished result.

## 2. Concentration diagnosis -- which clusters, which periods, and why the tails

### 2.1 Asset-class clusters actually carrying risk

The 33-root universe splits: equity-index {ES,NQ,YM}, FX {6A,6B,6C,6E,6J,6M,6N,6S}, rates
{ZT,ZF,ZN,TN,ZB,UB}, commodity {CL,BZ,NG,HO,RB, GC,SI,HG,PL, ZC,ZW,ZS,ZL,ZM, LE,HE}.

- **Rates (6 roots): contributing ~0.** All six are price-traded bond futures that fall through
  `CarryCalculator.compute` to the **v1 `return 0.0` fallback** (yield needs a CTD conversion factor
  that v1 does not compute; the SOFR path only fires for MICRO_YIELD_ROOTS 2YY/5YY/10Y/30Y, none of
  which are in this universe). Forecast ~= 0 -> ~0 contracts. **The rates sleeve is inert.** This
  simultaneously (a) refutes the "bond carry rode falling rates" story for this specific run,
  (b) removes carry's single most reliable historical diversifier (rates carry is classically the
  smoothest sleeve), and (c) means effective breadth is ~27, not 33 -- concentration is worse than the
  headline root count suggests.
- **FX (8 roots): the largest *coherent* cluster.** Absolute carry on 8 G10 pairs is close to the
  canonical FX carry trade; the pairs share a common risk-on/risk-off factor, so in a shock they move
  together and the "8 names" behave more like ~2-3 independent bets. Primary contributor to directional
  beta.
- **Energy (5 roots CL/BZ/NG/HO/RB): the lumpiness engine.** Energy term structure swings between deep
  contango (2014-15, 2020) and backwardation violently; carry magnitudes are large and episodic. The
  `days_to_second = abs(months)*30` denominator and volume-ranked front/second (Section 5) make energy
  carry jump around rolls. This is the most plausible single source of the **kurtosis 33.5** (a few
  enormous days).
- **Equity-index (3 roots): structurally net-long carry** (dividend yield vs financing is usually
  positive -> persistent long). Adds to the directional/risk-on beta that pays in W7/W8 and hurts in W9.

### 2.2 The unifying mechanism: an all-long-carry directional bet, undersized into shocks

Absolute (time-series) carry does not net long-vs-short across the basket -- each instrument is sized
by its own signed carry. In practice the basket runs **net long the risk-on carry factor** (long equity
carry, long high-yield FX, long-backwardated commodities). That common component:

- **pays spectacularly in the COVID flight-to-safety + commodity dislocation (W7/W8)** -> right-skew,
- **reverses in the 2021-22 regime break (W9)**,
- and is amplified at both ends by **vol-target lag**: `close_to_close_rv(rets, 25) * sqrt(252)` is a
  25-day trailing vol. Going into a fast shock it is stale-low -> positions are oversized -> a cluster
  of very large days -> **kurtosis**. This is a sizing artifact layered on a real signal, not a data bug
  (the report already confirms equity stays non-negative post bankruptcy-floor fix; tails are finite).

**Net:** skew +1.85 and kurt 33.5 = (COVID-window directional payoff) x (energy carry lumpiness) x
(25-day vol lag). PBO 0.63 = window outcomes are selected by one macro regime axis, so their ranking
does not generalize.

## 3. De-concentration levers -- mechanism and predicted effect

Predictions are directional priors, to be measured, not asserted. Each must re-run FULL integrity
(Section 6) and be judged on the combined gate.

| # | Lever | Mechanism | PBO | Kurtosis | Sharpe | Parameter-free? |
|---|---|---|---|---|---|---|
| 1 | **Cross-sectional demean** (relative carry) | subtract the cross-sectional mean carry each day -> removes the common all-long-carry directional beta, leaving long-high / short-low | **strong DOWN** | **strong DOWN** | DOWN (0.88 -> ~0.5-0.7) | YES (pure demean, no constant) |
| 2 | **Carry + trend combine** | forecast-average carry with Carver trend (doctrine 50/50 or fdm-weighted); trend pays in the 2022 selloff where carry lost (W9) | **DOWN** | DOWN | UP or flat | YES (doctrine weights) |
| 3 | **IDM / instrument-risk weighting** | Carver IDM scales the diversified portfolio; correlation-based weighting down-weights the 8-name FX and 5-name energy clusters | modest DOWN | **DOWN** | flat / slight UP | YES (computed, Carver doctrine) |
| 4 | **Per-cluster risk caps** | hard cap gross risk per asset class (energy <= x, FX <= y) | modest DOWN | **DOWN** | flat | **NO -- introduces a tunable cap** |
| 5 | **Slower / blended vol-scaling** | replace/blend the 25-day vol with a longer or two-speed estimate to kill undersizing-into-shocks | ~none | **DOWN** | flat | borderline (a window length is a parameter) |

### Detail on the two PBO levers

**Lever 1 -- cross-sectional demeaning (the single highest-leverage move for the gate).**
The gate failure is PBO/concentration and its root is the directional beta. Demeaning removes exactly
that beta: the portfolio becomes long-the-high-carry, short-the-low-carry instruments, dollar/risk
roughly neutral to the common factor. Consequences:
- Window outcomes stop co-moving with the risk-on/risk-off regime -> W7/W8 shrink, W9 stops being a
  deep loser -> **window Sharpe dispersion collapses -> PBO drops hardest here.**
- The correlated-shock right tail is largely netted out -> **skew toward 0, kurtosis into single/low-double
  digits.**
- **Cost:** part of the 0.88 is genuine directional carry premium; expect a Sharpe haircut to ~0.5-0.7.
  Also **turnover roughly increases** (relative ranks flip more than signed levels) -> the 1.5x cost
  gate must be *re-checked*, do not assume the current cost-insensitivity survives.
- Caveat: with rates inert (Section 2.1), the cross-section is FX/energy/metals/grains/equity only; a
  demean over a basket missing its smoothest sleeve is noisier than a full-breadth demean. Turning on
  bond carry (v2) would materially improve this lever.

**Lever 2 -- carry + trend combine.**
Trend is the least-correlated diversifier to carry and, crucially, pays in the exact window carry loses
(W9 2021-22 trend-following was a banner year). Combining fills the bad carry windows with good trend
windows -> window Sharpes broadly positive and low-dispersion -> PBO down, aggregate Sharpe up. Two
cautions: (a) it changes the strategy's identity into a combined system, and (b) the standalone Carver
momentum leg was WEAK, so the combine must be justified as diversification, not as leaning on a weak
leg; report the carry-only and trend-only legs alongside the combine so the gate sees all three.

### Detail on the kurtosis levers (3-5)

IDM/instrument-risk weighting and per-cluster caps both attack the *cross-sectional* concentration (FX
and energy clusters dominating gross risk) -> they cut kurtosis and give modest PBO help, but they do
not remove the *directional* beta, so on their own they will not take PBO from 0.63 to <0.25.
Vol-scaling attacks only the sizing-lag tail; near-zero PBO benefit. Treat 3-5 as **stackers on top of
1 or 2**, not as standalone fixes. Per-cluster caps and a chosen vol window introduce parameters --
prefer IDM (doctrine) and avoid caps unless a demean+combine variant still shows a specific cluster tail.

## 4. What a GATE-PASSING carry looks like

The combined gate (methodology 2.5) requires ALL of: PSR>0.95, DSR>0.95 (project-wide trials), PBO<0.25,
trades>=30 OOS, OOS/IS>=0.7. For carry specifically:

- **PBO < 0.25 (the binding constraint).** In the windows-as-columns construction this requires the
  window Sharpes to be **broadly positive and low-dispersion** -- no single window carrying the result
  (kill the W7 dominance) and no deep-negative window (kill the W9 reversal). Operationally that means
  removing the directional beta (Lever 1) and/or diversifying the regime exposure (Lever 2). Target
  window-Sharpe spread compressed from [-0.57, +2.42] toward roughly [0.2, 1.2].
- **Sharpe meaningfully > 0.** Accept a haircut from 0.88. Realistic targets: **>= 0.4-0.5 demean-only**
  (relative carry on futures is historically ~0.4-0.6 Sharpe), **>= 0.7 for carry+trend combine**.
  A demeaned Sharpe that collapses toward 0 means the 0.88 was mostly directional beta, not carry alpha
  -- an honest and important negative result if it happens.
- **PSR/DSR stay at trial=1.** Only achievable if we **pre-commit to ONE doctrine-justified variant**
  (recommended: cross-sectional demean) rather than sweeping all five levers and reporting the best PBO.
  See Section 5 on trials.
- **Tails:** skew toward 0, **kurtosis down to single / low-double digits** (a sanity co-check
  alongside PBO, per the readiness report's own guidance).
- **1.5x cost re-cleared.** Demeaning raises turnover; re-run the 1.5x gate rather than inheriting the
  current cost-insensitivity.
- **W11/W12 nan resolved** so the aggregate covers 2023-25.

A carry that clears this is a genuine momentum-uncorrelated diversifier and a deploy candidate; the
demeaned + trend-combined form is the natural building block for a combined futures book.

## 5. Discipline: parameter-freedom, DSR trial count, PBO/concentration first-class, full-data

- **Parameter-free doctrine (design Section "Parameter-free discipline").** `carry_scalar~30`,
  `ewma_span~10`, `cap=20` are fixed Carver constants and must never be optimized. Of the levers,
  **demean, combine (doctrine weight), and IDM are parameter-free**; **per-cluster caps and a chosen
  vol-window are NOT** -- each cap level / window length is a fit knob that both breaks the doctrine and
  adds DSR trials. This is why caps rank below demean/combine even though they help kurtosis.
- **DSR trial count is the trap in this whole exercise.** Today the run is honestly trial=1 (single
  non-selected config -> `expected_max_sharpe=0` -> DSR=PSR). The moment we **evaluate several
  de-concentration variants and select the best-PBO one, that is selection over trials** -- the
  project-wide cumulative trial count (methodology 2.3 / 9.4, `output/experiments.duckdb`) must
  increment to the number of variants tried, and DSR must deflate accordingly. **Mitigation: pre-commit,
  in this document, to cross-sectional demeaning as THE carry v2** on economic grounds (removes a known
  directional beta), test it once, and keep trial=1. If instead we explore the menu, we must register
  every variant and report the deflated DSR -- do not cherry-pick a PBO winner and quote trial=1.
- **PBO / concentration is first-class, not a footnote.** It is the gate that failed; skew/kurt are
  corroborating. Every variant reports PBO, per-window Sharpe dispersion, and per-cluster risk share.
- **Full data, no window-shopping.** Keep the full 2010-06..2026-02 range and the per-window phase-in
  (`load_daily_panel` graceful exclusion). Do not "fix" PBO by dropping W9 or the COVID windows -- that
  is the cherry-picking the full-data rule exists to prevent. The nan windows must be *filled*, not
  *dropped*.

## 6. Carry-specific integrity / point-in-time risks

1. **Bond SOFR v1 fallback = the biggest hidden risk (`carry_calculator.py` L130-133).** ZT/ZF/ZN/TN/ZB/UB
   -> `return 0.0`. In this universe that silently zeroes 6/33 roots. Consequences: the "falling-rate
   carry" narrative is void here; effective breadth ~27; and **a future v2 that turns bond carry on will
   materially change every number** -> the entire walk-forward must be re-run and re-gated, not patched.
   Document the inert sleeve explicitly in the readiness report.
2. **Front/second identification near rolls (`_find_front_second_close`, L53-84).** Front/second are chosen
   by *daily volume rank* from `per_contract_1min`. Around a roll, volume migrates front->next intraday,
   so the volume-ranked pair can flip day-to-day, making `(second-front)/front` jump -- a spurious carry
   spike, and a lumpy-carry / near-lookahead source if the volume-ranked pair disagrees with the roll
   calendar that the continuous *price* series uses. **Reconcile the carry pair against the roll calendar**
   so carry and price refer to the same contract pair; a mismatch is a silent bias.
3. **`days_to_second = abs(months)*30` (L113).** Crude annualizer; when the identified spacing flips
   between 1- and 2-month gaps near rolls, the annualization jumps ~2x -> another lumpy-carry / kurtosis
   contributor. A calendar-day denominator from actual expiries would be cleaner.
4. **No-lookahead at the sizing boundary (impl L35-41, harness).** Carry uses same-day front/second close
   and same-day 25-day price vol; the forecast is indexed to `close.index`. **Verify the harness lags the
   forecast (shift(1)) before it multiplies day-d returns** -- if day-d carry sizes the day-d close return,
   that is lookahead. The Carver harness convention must be confirmed for the carry path, not assumed.
5. **`carry.reindex(close.index).ffill()` (impl L36).** Forward-filling carry across holiday/data gaps
   carries a stale carry level forward -- a mild point-in-time smear (stale signal, not lookahead). Bound
   the ffill horizon so a long gap does not propagate a weeks-old carry.
6. **Carry cache is a point-in-time snapshot** (`compute_history` -> `carry_dir()/{root}.parquet`,
   design Caveats). Reproducibility depends on the cache SHA; log the cache snapshot in the Section 8
   identity fields and rebuild-and-re-run whenever the per-contract store changes.
7. **`derive_sofr(d)` as-of-date** (only fires for micro-yield roots, none in this universe today) --
   must be point-in-time / as-published if bond-carry v2 brings those roots in.

## 7. Recommended next step (single, pre-committed)

Ship **carry v2 = cross-sectional demeaned (relative) carry**, parameter-free, tested ONCE on the full
2010-2026 walk-forward, trial count held at 1. It targets the actual gate failure (directional
concentration -> PBO) at its root and is doctrine-clean. Report carry-abs vs carry-demeaned side by side;
if demeaned Sharpe holds >= ~0.5 with PBO < 0.25 and kurtosis in the low double digits, it graduates to a
deploy candidate and to the carry leg of a future carry+trend combine. In parallel (not as a gate-shopping
menu): resolve the W11/W12 nan windows and add the inert-bond-sleeve note to the readiness report. Defer
IDM/caps/vol-scaling as stackers to evaluate only if demeaning alone leaves a specific residual cluster tail.
