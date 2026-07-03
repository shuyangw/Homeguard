# Momentum Deep-Dive: Why Carver TSMOM Is WEAK, and Its Only Justified Role

**Date:** 2026-07-03 - **Author:** Strategy Lead (analysis + write-up; no code)
**Subject:** Deep-dive expansion of the already-tested Carver multi-speed TSMOM result
**Primary source:** `docs/reports/futures/CARVER_TSMOM_BROAD_READINESS.md` (run_id `2973f465-ce64-4135-8211-35ec06ffe67a`)
**Comparators:** `docs/reports/futures/CARRY_BROAD_READINESS.md` (run_id `2b9a02bc...`), `docs/strategies/research/20260703_FUTURES_STRATEGY_BACKLOG.md`
**Code under review:** `src/strategies/advanced/carver_momentum_strategy.py`, `src/strategies/advanced/carver_indicators.py`

---

## Executive summary

1. Carver multi-speed TSMOM on the 33-root basket (2013-06..2026-02 stitched OOS, 3964 days) returns **OOS Sharpe 0.08 (0.06 at 1.5x cost), PBO 0.35, per-window Sharpe swinging -1.15..+1.50**. This is a near-coin-flip: WEAK, no standalone edge. That is a *valid, clean finding*, not a bug -- the equity-feedback/bankruptcy-floor fix made the PSR/DSR/tail stats reliable.
2. **The weak result is an ERA/dead-signal problem, not a construction or breadth problem.** Momentum is negative in 7 of 13 windows, and its two large positive windows (W8 2020-21 +1.45, W9 2021-22 +1.50) sit squarely in the COVID-vol / 2021 commodity supertrend. The 2013-2019 stretch (W1-W6) is the textbook managed-futures "trend drought," and 2022-2025 (W10-W12) is whipsaw. The full-window average washes to ~0.
3. **Construction audit finds one real defect -- a missing Forecast Diversification Multiplier -- but it is Sharpe-neutral** under a portfolio vol-target simulator (it changes achieved vol, not risk-adjusted return). It cannot explain 0.08 and fixing it will not lift the Sharpe. Fixed doctrine speeds (4/16, 16/64, 64/256) are correct; re-tuning them is the classic trend overfit and is forbidden.
4. **Diversifying 3 -> 33 markets did not help**, which rules out breadth as the fault. The trend premium was absent over most of this window; more markets cannot manufacture a premium that is not being paid.
5. **No parameter-free momentum variant plausibly clears the gate standalone. Honest prior: low (~10-15%).** The null is robust across 15.7y and doctrine speeds; XS momentum is the only variant with a non-trivial chance, and even that is more likely WEAK.
6. **Momentum's only justified role is as a low-correlation ENSEMBLE diversifier to carry.** At the window level, momentum and carry Sharpes correlate **~ -0.29** (11 common windows), and momentum's best window (W9, +1.50) is carry's worst (W9, -0.57). The value of the combine (W2) is **lower PBO/kurtosis on carry, at a small Sharpe cost -- not a higher Sharpe.**

**Verdict: momentum is dead-standalone-but-useful-as-a-diversifier.** Retire it as a standalone candidate; keep it alive ONLY as an equal-forecast ensemble member around the carry book, where its acceptance metric is PBO/tail reduction, not Sharpe.

---

## 1. The result under review (exact numbers)

| Metric | Momentum (TSMOM) | Carry (comparator) |
|---|---|---|
| OOS Sharpe (1x cost) | **0.0835** | 0.8818 |
| OOS Sharpe (1.5x cost) | 0.0645 | 0.8719 |
| PBO (windows-as-columns CSCV) | **0.3493** | 0.6319 |
| PSR / DSR (n_trials=1) | 1.00 / 1.00 | 1.00 / 1.00 |
| skew / Pearson kurtosis | **-0.565 / 10.5** | +1.851 / 33.5 |
| n_windows / n_oos_days | 13 / 3964 | 13 / 3340 |
| Verdict | **WEAK** (no edge) | WEAK (strong-but-concentrated) |

Per-window OOS Sharpe, mapped to the OOS test year (train=36m warmup / test=12m / step=12m):

| Window | OOS period | Momentum | Carry | Regime note |
|---|---|---|---|---|
| 1 | 2013-06..2014-06 | -0.48 | 0.73 | trend drought |
| 2 | 2014-06..2015-06 | **+1.21** | 1.52 | USD/energy trend (2014 oil crash) |
| 3 | 2015-06..2016-06 | -0.40 | 0.44 | drought/chop |
| 4 | 2016-06..2017-06 | -0.95 | 1.37 | drought (post-election chop) |
| 5 | 2017-06..2018-06 | +0.07 | 0.84 | low-vol grind |
| 6 | 2018-06..2019-06 | -0.06 | 1.28 | Q4-18 whipsaw |
| 7 | 2019-06..2020-06 | -0.80 | 2.42 | COVID crash edge (carry shines) |
| 8 | 2020-06..2021-06 | **+1.45** | 1.79 | reflation supertrend |
| 9 | 2021-06..2022-06 | **+1.50** | **-0.57** | commodity supertrend (carry's worst) |
| 10 | 2022-06..2023-06 | **-1.15** | 1.25 | late-22 reversal + Mar-23 bank crisis whipsaw |
| 11 | 2023-06..2024-06 | +0.09 | nan | -- |
| 12 | 2024-06..2025-06 | -0.51 | nan | -- |
| 13 | 2025-06..2026-02 | +0.66 | 0.87 | -- |

The distribution is the story: momentum pays in vol-expansion/supertrend windows (W2, W8, W9) and bleeds in drought/whipsaw windows (W1, W4, W7, W10, W12). Average ~0.

---

## 2. Attribution -- why is momentum WEAK?

Three candidate causes were on the table. The evidence assigns them very different weights.

### 2a. PRIMARY cause: dead signal in this era (trend drought). Weight: ~70%.

The window pattern is the well-documented managed-futures narrative, reproduced on our basket:

- **2013-2019 trend drought (W1-W6):** the SG Trend Index was roughly flat across this stretch; central-bank vol suppression killed sustained trends. Our momentum is negative or flat in 5 of these 6 windows. This is a *market-state* fact, not a code fact.
- **2020-2021 supertrend (W8-W9, +1.45/+1.50):** COVID vol expansion plus the 2021 reflation/commodity trend is exactly the regime trend is built for. Momentum earned its convexity here.
- **2022-2025 whipsaw (W10-W12):** the late-2022 reversal and March-2023 banking crisis produced sharp mean-reverting reversals that chopped a trend book (W10 = -1.15).

Corroborating tail evidence: momentum's aggregate **skew is -0.57**, i.e. it did NOT deliver the positive "crisis-alpha convexity" that is the entire portfolio rationale for owning trend. A vol-targeted trend book that shows negative skew over the sample is telling you the trend premium was absent-to-inverted for most of the window. The premium simply was not paid here; this is "dead-in-this-era," not "dead-forever" and not "miscoded."

### 2b. SECONDARY: one real construction defect, but it is Sharpe-neutral. Weight: ~5%.

Audit of `carver_indicators.combined_forecast`:

```
combined = sum(forecasts) / len(forecasts)   # simple mean of 3 EWMAC speeds
return combined.clip(-cap, cap)
```

- **Missing Forecast Diversification Multiplier (FDM).** Averaging three positively-correlated EWMAC speeds yields a combined forecast whose standard deviation is *below* the target scaling (Carver targets average-absolute-forecast ~= 10). Carver restores it with FDM = 1/sqrt(w' rho w), ~1.1-1.25 for a 3-speed trend blend. It is omitted here, so the combined forecast is under-scaled by ~10-20%.
  - **But this is Sharpe-neutral.** FDM is a single global constant applied uniformly to every instrument's combined forecast, and the book is then re-levered to the 0.20 vol target by the equity-feedback simulator. Scaling every position by a constant that the vol-targeter then normalizes away changes *achieved vol*, not *risk-adjusted return*. It cannot move a 0.08 Sharpe. Fix it for correctness (and it matters for W2's cross-signal weighting), but do NOT expect it to rescue the result.
- **Fixed doctrine speeds (4/16, 16/64, 64/256) and cap 20 are correct and must stay fixed.** They are the reason trial_count=1 and DSR is honest. Searching for a speed set that "would have worked" on this basket is precisely the trend-overfitting trap the methodology (Sec 5, parameter sensitivity) forbids. The robust null holds *at doctrine*, which is the only trustworthy place to evaluate it.
- **Vol estimator (`close_to_close_rv`, 25-day) and price-vol normalization are standard and not implicated.**

Net: the construction is doctrinally slightly under-scaled but structurally sound. No construction fix changes the WEAK verdict.

### 2c. RULED OUT: diversification / breadth. Weight: ~0% as the fault.

Going from 3 roots to 33 roots (equity/rates/FX/energy/metals/grains/meats) did NOT rescue momentum. Broadening the basket is the standard institutional fix for trend, and it failed here. That is dispositive: the fault is not too-few-markets. You cannot diversify your way into a premium that is not being paid across the whole complex simultaneously. (Contrast carry, whose problem *is* a cross-sectional weighting/concentration issue -- opposite diagnosis.)

**Attribution conclusion:** ~70% era/dead-signal, ~5% construction (Sharpe-neutral), ~0% breadth, remainder irreducible noise. Momentum is a robust null on our data/era, not a fixable-by-tuning result.

---

## 3. Can ANY momentum variant clear the gate standalone? Honest prior: LOW.

| Variant | Standalone-pass prior | Why |
|---|---|---|
| Add FDM to TSMOM | **~2%** | Sharpe-neutral (Sec 2b). Correctness only. |
| Re-tuned EWMAC speeds | forbidden | Overfit trap; inflates DSR; the null holds at doctrine. |
| XS momentum (dollar-neutral rank) | **~15%** | Removes the basket-directional bet; can behave differently in tails. Best of a weak set, still likely WEAK. |
| Acceleration / trend-of-trend | **~5%** | Same premium, higher-frequency; low prior given the TSMOM null. |
| Donchian / MAC re-parameterization | **~5%** | Re-parameterizing a null; each channel length is a trial (DSR-expensive). |
| Short-term reversal | n/a (anti-momentum) | Cost-fragile; a *reversal*, not momentum -- separate thesis. |

Blended honest prior that *some* parameter-free momentum variant clears PSR/DSR/PBO + 1.5x cost **standalone**: **~10-15%.** The base rate is low because (a) the null is robust across 15.7y and the full complex, (b) doctrine discipline forbids the one lever (speed search) that could data-mine a pass, and (c) even the strongest variant (XS momentum) is a de-concentration play whose upside is "better-behaved WEAK," not "clears the gate."

**We should not spend the campaign trying to make momentum pass standalone.** The expected value of that path is low and the DSR/PBO cost of the attempts is real.

---

## 4. The justified role: low-correlation ensemble diversifier to carry (W2)

This is where momentum earns its keep. The argument, quantified.

### 4a. Momentum <-> carry correlation

Using the 11 windows where both have a scored Sharpe (carry is NaN in W11/W12), the **window-Sharpe correlation is approximately -0.29** (means: momentum +0.10, carry +1.08). The single most important data point: **momentum's best window (W9 2021-22, +1.50) is carry's worst (W9, -0.57)**, and carry's best (W7 2019-20, +2.42) is one of momentum's worst (W7, -0.80). That is a textbook diversification pattern -- the two premia get paid in different regimes (trend in vol-expansion supertrends; carry in calm term-structure-premium harvesting).

**Honest caveat on the number.** This is an 11-point *window-Sharpe* correlation, which is noisy and overstates the diversification benefit relative to the *daily-return* correlation that actually drives the ensemble. The managed-futures literature puts the daily TSMOM-vs-carry return correlation at a small *positive* value (~+0.1 to +0.25), not negative. The true ensemble math should be evaluated at both the observed window signal (-0.29) and a conservative literature prior (+0.15). The point survives either way: the correlation is well below 1, so the combine reduces portfolio variance.

### 4b. What the combine actually does to Sharpe (it does NOT raise it)

Two-sleeve equal-risk combine, carry Sharpe 0.88, momentum Sharpe 0.08:

- At correlation **-0.29**: combined Sharpe ~= (0.5*0.88 + 0.5*0.08) / sqrt(0.5*(1 + (-0.29))) ~= 0.48 / 0.596 ~= **0.81**.
- At correlation **+0.15** (literature): ~= 0.48 / sqrt(0.5*1.15) ~= 0.48 / 0.758 ~= **0.63**.
- Standalone carry: **0.88**.

So a naive 50/50 combine *lowers* Sharpe (you are diluting a 0.88 sleeve with a 0.08 sleeve). **The Sharpe cost is the point, and it is small.** The value of W2 is NOT a higher Sharpe -- it is:

- **Lower PBO.** Carry fails the gate on PBO 0.63 (concentration), not on Sharpe. Blending a decorrelated, zero-mean sleeve that pays in carry's weak windows (W9) mechanically stabilizes the window-Sharpe ranking that CSCV PBO measures.
- **Lower kurtosis / tamer tails.** Carry's +1.85 skew / 33.5 kurtosis is driven by a few energy-backwardation blowout days; a decorrelated sleeve dilutes those days' portfolio share.

W2's acceptance metric is therefore **PBO and kurtosis, evaluated against standalone carry** -- not Sharpe. If the combine takes carry from PBO 0.63 toward < 0.5 while holding blended Sharpe above ~0.7, W2 is a win even though carry's *standalone* Sharpe is higher. This is exactly the backlog's W2 framing (item 3), now quantitatively supported.

### 4c. Weighting nuance

Carver would give carry and trend equal *forecast* weight (then FDM-rescale). A risk-parity / IDM view would down-weight the near-zero-Sharpe momentum sleeve. Because momentum's mean is ~0 and its correlation to carry is negative-to-small, a *small* momentum allocation behaves like a cheap hedge: it can modestly reduce carry's PBO/kurtosis at minimal Sharpe cost, and -- if the negative correlation partially holds OOS -- possibly a marginal Sharpe gain at a small weight. The equal-forecast-weight version is the doctrine default and the honest trial=1 test; a weight search would inflate DSR and is deferred.

---

## 5. Ranked, testable momentum sub-hypotheses (by expected value)

EV here = P(improves the deployable book) x magnitude, net of DSR/PBO cost. "The book" = the carry-centered ensemble, since standalone momentum is a near-dead path.

| Rank | Sub-hypothesis | Type | Standalone prior | Ensemble EV | Build | Notes |
|---|---|---|---|---|---|---|
| 1 | **W2: carry + trend equal-forecast combine** (backlog #3) | Ensemble | n/a | **High** | S | Cheapest test of momentum's only real value. Acceptance = PBO/kurtosis vs standalone carry, NOT Sharpe. Add FDM here so the two signals are correctly co-scaled. Pre-commit the 50/50 weight (trial=1). |
| 2 | **XS momentum (dollar-neutral rank)** (backlog #8) | Trend XS | ~15% | Medium | S | Demean the existing combined EWMAC forecast cross-sectionally each day; rescale/cap. Removes the basket-directional bet; distinct tail behavior; genuine ensemble member. Low standalone prior but near-zero marginal cost. |
| 3 | **Add FDM to combined forecast** (construction fix) | Correctness | ~2% | Low-but-required | XS | Sharpe-neutral standalone; but REQUIRED for W2 so carry and trend are scaled to the same forecast target before averaging. Do it as part of #1, not as a standalone edge test. |
| 4 | **Short-term XS reversal (weekly)** (backlog #9) | Reversal | uncertain | Medium | S | Not momentum -- its opposite -- but the natural decorrelated partner in a trend/reversal ensemble. Cost-fragile: the 1.5x gate is the real test. Separate hypothesis; list here for completeness. |
| 5 | **Acceleration / trend-of-trend** (backlog #15) | Trend | ~5% | Low | S | Same premium, noisier. Only as an ensemble member; low prior given the null. |
| 6 | **Donchian / MAC re-parameterization** (backlog #16) | Trend | ~5% | Low | S | Re-parameterizes a demonstrated null; each channel length is a DSR trial. Lowest EV; do not pursue unless an ensemble specifically needs a breakout-flavored member. |

**Recommended action:** run only **#1 (W2)** now, with **#3 (FDM)** folded into it as a correctness prerequisite. Hold **#2 (XS momentum)** as the next cheap member if W2 validates the ensemble mechanism. Treat #4-#6 as contingent, low-priority ensemble fillers. Do NOT open a momentum parameter search.

---

## 6. Integrity requirements (baked in, non-negotiable)

1. **Parameter-free / DSR trial count.** Doctrine constants (speeds 4/16/64/256, cap 20, forecast scalars, FDM formula, equal combine weight) are FIXED, never optimized -> each item stays a single non-selected configuration -> project-wide DSR trial count (`output/experiments.duckdb`, methodology Sec 2.3) stays honest. **Re-tuning EWMAC speeds is prohibited** -- it is the trend overfit trap and would silently inflate the trial count that gates every other futures result. If W2 tries multiple weightings, each counts as a trial; pre-commit the 50/50 weight.
2. **PBO / kurtosis / skew are first-class acceptance metrics.** Momentum's WEAK verdict already rests on PBO 0.35 as much as on Sharpe 0.08. For W2 the *primary* acceptance test is PBO/kurtosis improvement over standalone carry -- a combine that raises Sharpe but not PBO is not the win we want; a combine that holds Sharpe and cuts PBO below the gate is.
3. **Full data range, no sub-window.** 2010-06-07 .. 2026-02-20 with per-window data-availability filtering (instruments phase in). Never cherry-pick the 2020-2021 supertrend windows where momentum looked good -- that is the exact ramp-long-calls window-artifact failure mode.
4. **No lookahead.** Any correlation/FDM estimate used for weighting comes from the TRAIN segment only; forecasts are as-of daily close on shift-safe features. Same discipline the shipped carry class already honors.
5. **1.5x cost gate.** Re-check at 1.5x for every variant. Momentum degrades only mildly under cost (0.08 -> 0.06), so cost is not its problem -- but the short-term reversal member (#4) lives or dies on it.
6. **WEAK is a valid, publishable outcome.** The value of this deep-dive is the *decision* -- retire momentum as a standalone, keep it as an ensemble diversifier -- not a rescue.

---

## 7. Verdict on momentum's future role

**Momentum (Carver TSMOM) is a robust null on our basket and era -- dead as a standalone candidate, and no parameter-free variant is likely (~10-15%) to change that.** The WEAK result is an era/trend-drought fact (2013-2019 flat, 2022-2025 whipsaw, punctuated only by the 2020-2021 supertrend), not a construction or breadth defect; the one real construction issue (missing FDM) is Sharpe-neutral.

**Its only justified role is as a low-correlation ensemble diversifier to carry.** Momentum and carry get paid in different regimes (window-Sharpe correlation ~ -0.29; momentum's best window is carry's worst), so blending them (W2) should reduce carry's PBO and kurtosis -- carry's actual failure mode -- at a small, quantified Sharpe cost. That is the single highest-value piece of momentum-related work on the board, and it is cheap.

**Recommendation:**
- **Retire** standalone momentum and any momentum parameter search.
- **Run W2** (carry + trend equal-forecast combine, FDM folded in) as the next test, judged on PBO/kurtosis vs standalone carry.
- **Hold XS momentum** as the next-cheapest ensemble member if W2 validates the mechanism.
- Keep the campaign's center of gravity on **de-concentrating carry** (backlog W1/XS-carry); momentum contributes to that book as a tail-smoother, never as the engine.
