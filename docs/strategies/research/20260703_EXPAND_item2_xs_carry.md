# EXPAND item 2 -- Cross-Sectional Carry (XS demeaned carry)

**Date:** 2026-07-03  -  **Author:** Strategy Lead (design-precursor / research brief)
**Status:** research / write-up only -- NOT an implementation. Feeds a later design + build.
**Backlog ref:** `docs/strategies/research/20260703_FUTURES_STRATEGY_BACKLOG.md`, item 2.
**Methodology:** `docs/methodology/backtesting.md` Sections 2 (PSR/DSR/PBO + combined gate),
3 (walk-forward purge/embargo), 4 (futures costs + 1.5x gate).
**Harness contract:** a new `forecast_panel(close_panel) -> DataFrame` strategy class (per-root
forecast in +/-20 Carver units) + a `src/strategies/registry.py` entry + a YAML config. NO runner
changes. Forecasts flow unchanged into `FuturesPortfolioSimulator.run_sized`
(`src/backtesting/engine/futures_portfolio_simulator.py`) via
`size_from_forecast` (`src/backtesting/utils/position_sizer_futures.py`), with the futures cost
model and 1.5x cost sensitivity, gated by walk-forward PSR/DSR/PBO.

---

## Executive summary

1. **Construction:** reuse the shipped risk-adjusted carry (`EWMA(carry)/ann_vol`, same as
   `FuturesCarryStrategy`), then each day **subtract the cross-sectional mean WITHIN asset class**,
   normalize by the train-window cross-sectional dispersion, and scale/cap to +/-20. Delta over the
   shipped carry class is ~5 lines.
2. **My call -- DEMEAN, not rank; WITHIN-class, not global.** Demeaning is the minimal delta,
   preserves the carry *magnitude* that is the actual edge, and the existing `/vol` normalization +
   the +/-20 cap already provide outlier control. Rank-normalization is the pre-committed
   *escalation* only if demean's OOS kurtosis stays high -- and it costs a second trial. Within-class
   because risk-adjusted carry levels are NOT commensurable across rates vs commodities; global
   demeaning manufactures a persistent structural cross-class bet (always-short-contango-commodities
   / always-long-positive-carry-rates) that RE-INTRODUCES a common component -- the exact thing item
   2 exists to remove.
3. **Sizing crux (it works, approximately):** a mean-zero demeaned forecast fed through the existing
   per-instrument vol-target sizer produces some longs, some shorts, EACH scaled to the same
   per-instrument risk budget. Because `contracts_i ~ (forecast_i/10) * capital * vol_target /
   (risk_i)`, the signed sum of *risk-dollar* exposures is proportional to `sum_i forecast_i`, which
   is zero by construction. So the book is **vol-/risk-neutral by construction** even though our
   simulator is NOT a dollar-neutral long-short book. Residuals to handle: integer-contract rounding,
   `max_contracts` caps, and -- the important one -- vol-neutral is NOT beta/factor-neutral (the longs
   can all be one correlated cluster).
4. **Predicted effect:** removing the common directional carry bet should cut skew (from +1.85
   toward ~0), collapse kurtosis (from 33.5), and drop PBO (from the failing 0.63). **Expected Sharpe
   cost is real:** the common component absolute carry captured is itself a genuine premium, so raw
   Sharpe likely falls from 0.88 to ~0.4-0.7 -- the trade is Sharpe for a distribution that clears
   PBO. This mirrors the equities-campaign lesson: structural de-concentration beats signal cleverness.
5. **Composition with item 1 (IDM):** XS demean de-concentrates on the SIGNAL side (removes common
   mode before sizing); IDM/instrument-risk-weighting de-concentrates on the PORTFOLIO side (caps any
   single cluster's risk share). Orthogonal, composable, and genuinely complementary -- IDM is exactly
   what cleans up the residual net-factor exposure XS leaves behind (crux point 3). Build both;
   compose as "XS carry forecast -> IDM-weighted sizer."
6. **Integrity:** demeaning is parameter-free IF the within-class grouping (fixed asset-class
   taxonomy) and the cross-sectional scalar (fixed doctrine, train-window dispersion) are pre-committed
   -> this is ONE non-selected config -> it contributes **+1** to the project-wide DSR trial count.
   The demean-vs-rank and within-vs-global choices are the trial-inflation traps: pre-commit them
   here (done above); any escalation logs an additional trial to `output/experiments.duckdb`.

---

## 1. Precise construction

### 1.1 Base signal (unchanged from the shipped carry class)

Per root, reuse `FuturesCarryStrategy._forecast_root`
(`src/strategies/advanced/futures_carry_strategy.py`) up to but NOT including the final cap:

```
raw_i(t) = EWMA_span10( carry_i(t) ) / ann_vol_i(t)
```

where `carry_i` is the cached annualized `CarryCalculator` output per root and `ann_vol_i` is the
25-day close-to-close realized vol annualized by sqrt(252). This is the risk-adjusted carry -- the
same quantity absolute carry maps straight to a forecast. `raw_i` is dimensionless (carry per unit
of price vol), which is what makes it comparable across instruments *within a class*.

### 1.2 Cross-sectional demeaning (the delta)

Each day `t`, partition the available roots by asset class `g` in {equity_index, bond/rates, fx,
energy, metals, grains, meats} (the same taxonomy `CarryCalculator.compute` already switches on).
For each class `g` with `>= 2` roots reporting on `t`:

```
demeaned_i(t) = raw_i(t) - mean_{j in g}( raw_j(t) )        for i in g
```

Singleton classes on a given day (only one root with data) demean to zero and contribute no
position that day -- correct: a single instrument cannot express relative value against its class.
The demean uses ONLY that day's cross-section -> strictly point-in-time, no lookahead
(methodology Section 1.1 / 3).

### 1.3 Scale and cap to +/-20

Map `demeaned_i` to Carver units so the average absolute forecast is ~10 (the shipped convention),
using a FIXED doctrine scalar derived from the *train-window* cross-sectional dispersion (not a
fitted constant, and not a full-sample statistic -- same train-only discipline item 1's correlations
use):

```
xs_scalar = 10 / mean_over_train( cross_sectional_stdev_g( demeaned(t) ) )
forecast_i(t) = clip( demeaned_i(t) * xs_scalar, -20, +20 )
```

`xs_scalar` is one number fixed on the train segment of each walk-forward window; it is not searched.
(Equivalent doctrine alternative: normalize each day by that day's expanding cross-sectional stdev
and multiply by 10 -- also lookahead-free and constant-free. Pre-commit one; do not try both and
keep the better, that is a trial.)

### 1.4 Demean vs rank -- the decision and why

| | Demean (subtract class mean) | Rank-normalize (XS rank -> [-1,1]) |
|---|---|---|
| Outlier robustness | Partial -- extreme carry still -> extreme forecast (mitigated by `/vol` + cap) | Full -- one blowout cannot dominate |
| Magnitude info | Preserved (the edge partly lives here) | Discarded -- a small and a large carry gap look similar if adjacent in rank |
| Turnover | Lower -- moves only when carry moves | Higher -- rank flips on noise near ties (worse under the 1.5x cost gate) |
| Neutrality math | Clean: mean-zero -> exact risk-dollar neutrality (Sec 2) | Approx: symmetric rank also ~mean-zero |
| Delta over shipped class | ~5 lines | More; new rank transform |

**Call: DEMEAN as the pre-committed primary.** It is the minimal delta, preserves the risk-adjusted
carry magnitude that is the genuine premium, keeps the neutrality property exact, and the existing
`/ann_vol` normalization plus the +/-20 cap already truncate the worst tails. **Rank is the
pre-committed fallback** ONLY if demean's stitched-OOS Pearson kurtosis does not fall materially
below absolute carry's 33.5 (e.g., stays > ~10) while PBO still fails -- and running it counts as a
second trial, logged to the registry.

### 1.5 Within-class vs global -- the decision and why

**Call: WITHIN asset class.** Risk-adjusted carry is comparable *inside* a class but not across
classes: commodity carry is `(second-front)/front` annualized; rates carry is
`duration*(yield-funding)`; equity-index carry is a dividend/funding basis. Even after dividing by
price vol, classes carry systematically different means (commodities persistently negative in
contango regimes, rates persistently positive on a normal curve). Demeaning GLOBALLY subtracts a
mean of incommensurable quantities and leaves a persistent static tilt -- always short the low-carry
class, always long the high-carry class -- which is itself a common directional bet across classes.
That defeats item 2's entire purpose (remove the common component) and would re-import concentration
along the class axis. Within-class isolates the *relative* term-structure premium among truly
comparable instruments (e.g., which energy contract is steepest in backwardation relative to the
energy complex), which is the clean, de-concentrated bet.

Caveat (a design consideration, not a free parameter): with 33 roots across 7 classes, some classes
are thin (2-3 members). A 2-member class collapses to a pure equal-and-opposite pair; a 1-member
class contributes nothing that day. This is acceptable and honest -- it just means the XS book is
built from several small within-class relative-value bets, which is precisely the diversified,
low-concentration structure we want.

---

## 2. Sizing-compatibility crux (does a demeaned forecast express the XS bet through OUR sizer?)

**Short answer: yes, approximately -- and cleanly.** XS strategies are classically built as
dollar- or beta-neutral long-short books. Our simulator does NOT construct such a book: each
instrument is sized independently to a per-instrument vol target (Carver style). The question is
whether a mean-zero demeaned forecast, fed through that independent sizer, still expresses "long the
high-carry names, short the low-carry names, risk-balanced."

The sizer (`size_from_forecast`) is signed and linear in the forecast:

```
contracts_i = round( (forecast_i / 10) * capital * vol_target * div_mult
                     / (multiplier_i * price_i * ann_vol_i) )
```

A positive demeaned forecast -> long, a negative one -> short, and each leg is scaled to
`(|forecast_i|/10) * (capital * vol_target)` DOLLARS OF ANNUALIZED RISK. So each name carries
forecast-proportional, vol-equalized risk regardless of its price or contract size -- exactly the
risk-balancing an XS book wants.

**The neutrality property.** Ignoring rounding and caps, the signed *risk-dollar* exposure of leg
`i` is `(forecast_i/10) * capital * vol_target` (the price/vol/multiplier terms cancel into "one
vol-target unit of risk"). Summed across the book:

```
sum_i risk_dollars_i = (capital * vol_target / 10) * sum_i forecast_i = 0
```

because the within-class demean makes `sum_i forecast_i = 0`. So a demeaned forecast through the
existing per-instrument vol-target sizer yields a book whose **net risk-dollar exposure is zero by
construction** -- vol-/risk-neutral, i.e. the XS bet is expressed correctly at the risk level, even
though the book is not dollar-neutral and not beta-neutral. This is the key result: no sizer change
is needed.

**Residual net exposures to flag and handle:**

1. **Integer-contract rounding.** `contracts_i` is rounded to an integer. In a small basket (and
   thin within-class groups), rounding perturbs the exact zero-sum. Report the realized net
   risk-dollar exposure per day as a diagnostic; it should be small and mean-zero, not a persistent
   tilt.
2. **`max_contracts` caps.** If one leg is capped (`+/- spec.max_contracts`), its symmetry partner
   is not, breaking neutrality. Given demeaned forecasts are bounded at +/-20 and vol-scaled, caps
   should rarely bind, but verify none bind persistently for any root.
3. **Vol-neutral is NOT beta/factor-neutral (the important one).** Risk-dollars summing to zero does
   NOT mean the book has no net directional exposure: if on a given day the high-carry (long) names
   are all one correlated cluster (e.g., all energy) and the shorts are all rates, the book carries a
   large net *factor* bet whose realized-return risk does not cancel. Within-class demeaning already
   limits this (each class is internally balanced), but cross-class the residual factor exposure
   remains. **This residual is exactly what item 1 (IDM / cluster-risk weighting) neutralizes** ->
   see Section 4. Report the realized correlation of the long book vs the short book as a diagnostic.

**Net:** the demeaned forecast is a correct, no-plumbing-change expression of the XS carry bet at the
risk level; the one residual that matters (net factor/cluster exposure) is explicitly the seam where
item 1 composes.

---

## 3. Predicted de-concentration effect and expected Sharpe cost

**Why absolute carry failed:** OOS Sharpe 0.88 but PBO 0.63, skew +1.85, kurtosis 33.5, positive in
10/11 windows. The edge is real but its *distribution* is dominated by a few instruments/days -- the
returns concentrate because absolute carry is largely a COMMON directional bet (most instruments
carry the same sign in a regime, so the book is really "long the whole carry factor").

**Predicted effect of XS demeaning:**

- **Skew -> toward 0.** Removing the common long-everything component removes the asymmetric
  "everything rallies / everything blows out together" days that drive +1.85 skew.
- **Kurtosis -> far below 33.5.** The fat tail is the common-mode blowout; a within-class
  relative-value book has offsetting legs, so single-day extremes shrink. (If demean does not achieve
  this, escalate to rank per 1.4.)
- **PBO -> down, target < 0.25.** A less concentrated, more symmetric edge whose window ranking is
  more stable under CSCV. This is the primary acceptance metric here.

**Expected Sharpe cost (state honestly):** the common component that absolute carry rode is itself a
*genuine* risk premium (the aggregate carry factor -- long the basket when it is in backwardation).
Demeaning DELETES that premium and keeps only the relative-value premium, which is empirically
smaller. So expect raw OOS Sharpe to fall from 0.88 to roughly **0.4-0.7**. XS carry is the more
common academic form precisely because it is better-behaved in the tails, not because it is
higher-Sharpe. The bet item 2 makes is explicit: **trade raw Sharpe for a distribution that clears
PBO and the combined gate.** A WEAK/REJECT is a valid, publishable outcome; a lower-Sharpe result
that CLEARS PBO where 0.88 absolute carry did NOT is the target win.

---

## 4. Composition with item 1 (IDM / instrument-risk weighting)

Items 1 and 2 are **two independent de-concentration levers acting on different stages** of the same
pipeline:

| | Item 2 -- XS demean | Item 1 -- IDM / cluster-risk weighting |
|---|---|---|
| Acts on | the SIGNAL (before sizing) | the RISK ALLOCATION (portfolio construction) |
| Mechanism | removes the common directional carry component | down-weights correlated clusters, caps single-instrument risk share, scales book by IDM=1/sqrt(w'*rho*w) |
| Concentration source attacked | signal-level common mode (skew/kurt from "everything same sign") | portfolio-level cluster domination (one cluster eating the risk budget) |
| Parameter status | doctrine grouping + scalar | doctrine correlation->weights->multiplier formula |

They **compose directly**: feed the XS-demeaned carry forecast INTO the IDM-weighted sizer
("XS carry -> IDM sizer"). XS removes the common-mode return; IDM then ensures the residual
relative-value book is not dominated by any single cluster -- and the residual it cleans up is
exactly the "vol-neutral but not beta-neutral / longs all one cluster" exposure flagged in Section 2
point 3. So the composition is genuinely complementary, not redundant: XS closes the signal-level gap
and IDM closes the portfolio-level gap that XS provably leaves open.

Caveat on stacking: both partly address the energy-cluster concentration, so the marginal benefit is
sub-additive; do not assume the improvements simply add. Test the three configs -- XS carry alone,
IDM carry alone, and XS+IDM -- as **pre-committed** members (each is one trial; see Section 5), and
let the combined gate decide. Sequence: prove XS and IDM independently clear (or fail) PBO first,
then run the composition.

---

## 5. Integrity and trial-count accounting

- **Parameter-free? Yes, conditionally.** Demeaning adds NO fitted parameter *provided* the two
  structural choices are fixed doctrine and pre-committed:
  (a) grouping = the fixed asset-class taxonomy (equity/rates/fx/energy/metals/grains/meats) already
  encoded in `CarryCalculator`; (b) the cross-sectional scalar = a doctrine constant set from the
  train-window cross-sectional dispersion (Section 1.3), never searched. The EWMA span (10) and cap
  (20) are inherited doctrine from the shipped carry class. No lookback is fitted.
- **No lookahead.** The demean uses only that day's cross-section; the scalar and any dispersion
  normalization come from the TRAIN segment only (same discipline as item 1's correlations and the
  shipped carry class). Carry itself is as-of daily close, point-in-time. (Methodology Section 1.1,
  3.2-3.3.)
- **Full data range.** Run 2010-06-07 .. 2026-02 with per-window data-availability filtering
  (instruments phase in). NEVER a sub-window. (Backlog integrity req 2.)
- **Trial count (the real risk).** This single pre-committed XS-demean-within-class config is ONE
  non-selected configuration -> it adds **+1** to the PROJECT-WIDE cumulative trial count fed to DSR
  (methodology Section 2.3, queried from `output/experiments.duckdb` per Section 9.4). The
  campaign's DSR benchmark keeps rising as configs are logged (TSMOM, absolute carry, IDM carry, XS
  carry, XS+IDM ...). The trial-inflation traps are the design choices themselves:
  - demean vs rank -> pre-committed to demean (rank = fallback, +1 trial if run);
  - within-class vs global -> pre-committed to within-class (+1 trial if global is ever run);
  - the scalar convention -> pre-commit one (fixed vs expanding), do not keep the better.
  Every escalation or alternative that is actually RUN must be appended to the registry and counted;
  DSR uses the cumulative count, not 1. Honest accounting here is what keeps this de-concentration
  campaign from silently manufacturing a false edge.

---

## 6. Acceptance criteria (what makes item 2 a PASS)

Run through `run_carver_walkforward.py` (parameter-free path: no parameter search; rolls
non-overlapping OOS windows, stitches OOS returns, evaluates the gate on the stitched series), with
the futures cost model and the 1.5x cost leg. Acceptance = methodology Section 2.5 combined gate
plus the concentration diagnostics that are FIRST-CLASS here:

- PSR(0) > 0.95 and DSR > 0.95 (DSR using the project-wide cumulative trial count).
- **PBO < 0.25** -- the metric absolute carry failed at 0.63; this is the headline test.
- 1.5x-cost OOS Sharpe >= 0.5 and PSR(0) at 1.5x > 0.90 (methodology Section 4.6).
- Report skew and Pearson kurtosis alongside -- the win condition is "kurtosis collapsed from 33.5
  and PBO cleared," even at a materially lower Sharpe than 0.88.
- Diagnostics from Section 2: realized net risk-dollar exposure per day (rounding/cap residual) and
  long-book vs short-book realized correlation (residual factor exposure).

A lower-Sharpe XS carry that CLEARS PBO is a better outcome than the higher-Sharpe absolute carry
that did not. If XS alone is WEAK on PBO, the next move is the item 1 composition (Section 4), not
parameter tinkering.

---

## 7. Build sketch (for the later design doc -- NOT built here)

- New class `FuturesXSCarryStrategy` (registry name e.g. `FuturesXSCarry` / display "XS Carry"),
  `src/strategies/advanced/futures_xs_carry_strategy.py`.
- Reuse the shipped carry base up to `raw_i = EWMA(carry)/ann_vol`; add: per-day within-class
  demean, train-window scalar, +/-20 clip. Emits the standard `forecast_panel(close_panel) ->
  DataFrame`.
- Registry entry in `src/strategies/registry.py`; a YAML config mirroring the carry config; no runner
  or sizer changes (the crux in Section 2 is why none are needed).
- Then compose with item 1 by routing this forecast through the IDM-weighted sizer.
