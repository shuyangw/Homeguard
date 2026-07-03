# EXPAND item 3 -- W2: Carry + Trend Combined Forecast (Research Brief)

**Date**: 2026-07-03
**Author**: Strategy Lead (design-precursor; analysis + write-up only, NO implementation)
**Backlog item**: 3 (W2) -- combine the two existing futures sub-forecasts into one
**Methodology**: `docs/methodology/backtesting.md` Sections 2 (statistical framework),
3 (walk-forward), 4 (costs). This brief references those sections; it does not restate them.

---

## 0. TL;DR

Build a `CombinedForecast` strategy that calls BOTH existing `forecast_panel`s
(`CarverMomentumStrategy` trend + `FuturesCarryStrategy` carry), takes a **pre-committed
weighted average** of the capped sub-forecasts, multiplies by a **Forecast Diversification
Multiplier (FDM ~= 1.4)** derived from the sub-forecast correlation, and re-caps to +/-20.
The thesis is NOT that trend adds standalone edge (it does not: OOS Sharpe 0.08). The thesis is
that trend is a **near-uncorrelated tail-smoother** for carry's concentrated, fat-tailed return
stream (carry OOS Sharpe 0.88 but PBO 0.63, skew 1.85, Pearson kurt 33.5). We expect the blend
to **give up ~0.20 of Sharpe to buy a large reduction in kurtosis and skew**, and a moderate
reduction in PBO. The single largest failure mode is NOT quantitative -- it is the
**ensemble degrees-of-freedom trap**: weights/FDM/signal-selection are researcher choices that
inflate the project-wide DSR trial count. They MUST be pre-committed doctrine and logged to
`output/experiments.duckdb`, or the ensemble manufactures a false edge -- the exact failure that
sank the RAMP equity campaign.

---

## 1. Construction -- the `CombinedForecast` strategy

### 1.1 What already exists (do not rebuild)

Both sub-strategies already emit a `forecast_panel(close_panel) -> DataFrame` in +/-20
Carver forecast units, resolved by config `name` via `src/strategies/registry.py`:

| Sleeve | File | Forecast |
|---|---|---|
| Trend | `src/strategies/advanced/carver_momentum_strategy.py` | multi-speed EWMAC blend, capped +/-20 |
| Carry | `src/strategies/advanced/futures_carry_strategy.py` | `EWMA(carry)/ann_vol * scalar`, capped +/-20 |

The existing `src/strategies/advanced/carver_indicators.py::combined_forecast` already
demonstrates the *intra-signal* Carver pattern (average the three EWMAC speeds, then re-cap).
The combine reuses that pattern one level up -- across signals rather than across speeds -- but
adds the FDM step, which the intra-EWMAC helper omits.

### 1.2 The combination math (Carver, *Systematic Trading* / *Advanced Futures*)

For instrument `i` at time `t`, with capped sub-forecasts `f_trend` and `f_carry` (each already
in +/-20 units):

```
raw_combined(i,t) = FDM * ( w_trend * f_trend(i,t) + w_carry * f_carry(i,t) )
forecast(i,t)     = clip(raw_combined(i,t), -20, +20)
```

Two properties matter:

1. **Weighted average, not sum.** `w_trend + w_carry = 1`. Averaging two capped forecasts of the
   same scale keeps the result in the same unit space -- but averaging *shrinks* the typical
   absolute forecast (two imperfectly-aligned signals partially cancel), so the combined stream
   would under-use the risk budget without a correction. That correction is the FDM.

2. **FDM counters averaging-down.** The Forecast Diversification Multiplier restores the combined
   forecast to the target scale. For a weight vector `w` and sub-forecast correlation matrix `C`:

   ```
   FDM = 1 / sqrt( w' C w )      (Carver caps FDM at ~2.5)
   ```

   For two sleeves at equal weight `w = [0.5, 0.5]` and correlation `rho`:

   ```
   w' C w = 0.5 * (1 + rho)   ->   FDM = sqrt( 2 / (1 + rho) )
   ```

   | rho (trend,carry) | FDM |
   |---|---|
   | -0.10 | 1.49 |
   | 0.00  | 1.41 |
   | +0.25 | 1.26 |
   | +0.50 | 1.15 |

   Trend and carry are near-uncorrelated (est. rho ~= 0), so **FDM ~= 1.4** -- squarely inside the
   1.1-1.5 range in the backlog item. The FDM is computed ONCE from the long-run sub-forecast
   correlation on a fixed reference window (or fixed at the doctrine value 1.4) and then frozen.
   It is NOT re-estimated per-window in a way that peeks at OOS, and it is NOT a tunable.

### 1.3 Weight choice -- a DOCTRINE decision, never a fit

This is the crux of the integrity story (Section 4). The weights are chosen by doctrine BEFORE
seeing combined results, from a short menu of defensible priors:

- **Equal weight (50/50)** -- Carver's default when you lack a strong, out-of-sample-justified
  prior on relative signal quality. RECOMMENDED as the pre-committed baseline.
- **Risk-weighted (equal risk contribution)** -- since both sub-forecasts are already vol-scaled
  into the same +/-20 unit space, equal-risk collapses to ~equal-weight here; it is not a
  meaningfully different doctrine for these two sleeves.
- **Sharpe-weighted / inverse-variance** -- FORBIDDEN as a baseline. Tilting toward carry
  *because we observed* carry's 0.88 vs trend's 0.08 is using OOS performance to set weights.
  That is in-sample optimization wearing a doctrine costume; it re-introduces exactly the
  overfitting the combine is meant to reduce, and it would make the trend sleeve vestigial
  (defeating the tail-smoothing purpose).

**Decision: pre-commit 50/50 equal weight, FDM = 1.4 (frozen), cap +/-20.** One configuration.
If the team insists on evaluating alternative weight schemes, every scheme evaluated counts as a
trial (Section 4).

### 1.4 Implementation shape (for the eventual Phase-3 spec, not this brief)

A thin `CombinedForecast` that instantiates the two existing strategies, calls each
`forecast_panel(close_panel)`, aligns the two panels on the shared (date x root) index, applies
the weight/FDM/cap formula element-wise, and returns the +/-20 panel. It plugs into the SAME
`run_carver_walkforward.py` harness unchanged (the harness is signal-agnostic -- it resolves the
strategy by config `name` and sizes with the equity-feedback vol-target simulator). No new
sizing, cost, or gate code is required.

---

## 2. THE KEY QUESTION -- will combining reduce PBO / kurtosis vs standalone carry?

The whole item lives or dies on this. Reason it through from the moment algebra of blending two
**independent** streams (trend/carry correlation est. ~= 0), assuming each sleeve is vol-matched
by the simulator (equal risk, equal weight, so the blend return is `0.5*r_carry + 0.5*r_trend`).

### 2.1 Kurtosis -- strong compression expected

For independent X (carry) and Y (trend) at equal variance and equal weight, excess kurtosis of
the sum is:

```
exkurt(0.5 X + 0.5 Y) = 0.25 * ( exkurt_X + exkurt_Y )
```

Carry excess kurt = 33.5 - 3 = 30.5; trend is far flatter (excess ~= 2). So:

```
exkurt_blend ~= 0.25 * (30.5 + 2) = 8.1   ->   Pearson kurt ~= 11
```

**Kurtosis 33.5 -> ~11** -- a ~73% cut in EXCESS kurtosis. This is the core mechanism: on carry's
worst tail days, the independent trend sleeve is on average flat (sometimes offsetting), so it
dilutes the single-day concentration that drives carry's fat tail. The blend still has fat tails
(kurt 11 > 3) but is far from the pathological 33.5.

### 2.2 Skew -- reduced but still positive

For independent equal-variance, equal-weight streams:

```
skew(0.5 X + 0.5 Y) = ( skew_X + skew_Y ) / sqrt(8)   [equal-var case]
                    ~= (1.85 + 0) / 2.83 ~= 0.65
```

**Skew 1.85 -> ~0.65.** We keep favorable positive skew but shed the extreme right-tail
concentration. Lower kurtosis and lower (still positive) skew both *raise* PSR for a given Sharpe
(Section 2.2: the PSR denominator `sqrt(1 - skew*SR + (kurt-1)/4 * SR^2)` shrinks as kurt falls),
so the combine can pass PSR at a lower raw Sharpe than carry needs.

### 2.3 Sharpe -- the give-up

Independent sleeves, Sharpe 0.88 (carry) and 0.08 (trend), equal risk / equal weight:

```
Sharpe_blend = (S_carry + S_trend) / sqrt(2)   [uncorrelated, equal-vol sleeves]
             = (0.88 + 0.08) / 1.414 ~= 0.68
```

**Sharpe 0.88 -> ~0.68.** We give up **~0.20 Sharpe (~23%)**. Note this is scale-invariant: FDM
changes leverage/vol, not the Sharpe, so 0.68 holds regardless of the FDM value. The give-up is
real and expected -- adding a near-null uncorrelated sleeve to a strong one *lowers* combined
Sharpe (the Sharpe-maximizing weight on trend would be tiny). We accept it deliberately, buying
tail insurance (2.1-2.2) with Sharpe.

### 2.4 PBO -- moderate improvement, the uncertain part

Carry's PBO 0.63 (>0.50 = strong overfitting, Section 2.4) reflects that carry's best-ranked
windows do NOT generalize -- its edge is concentrated in a few instruments/episodes. Adding an
uncorrelated sleeve should stabilize the cross-window Sharpe ranking somewhat, because a chunk of
each window's return no longer comes from carry's concentrated tail. BUT trend's standalone
Sharpe is ~null (0.08), so it contributes little *stabilizing signal* of its own; most of the PBO
benefit comes indirectly, via the tail compression in 2.1.

**Predicted PBO: 0.63 -> ~0.40.** Materially better, but I do NOT expect the combine ALONE to
clear the <0.25 gate. Carry's PBO is driven mostly by its *internal* concentration (one or two
roots carrying the sleeve), which the external trend blend does not directly fix. That is why
this item is complementary to -- not a substitute for -- items 1-2 (Section 3).

### 2.5 Summary of the predicted trade-off

| Metric | Carry standalone | Combined (predicted) | Direction |
|---|---|---|---|
| OOS Sharpe (1x) | 0.88 | ~0.65-0.70 | worse (bought insurance) |
| Pearson kurtosis | 33.5 | ~10-12 | MUCH better |
| Skew | 1.85 | ~0.65 | better (still +) |
| PBO | 0.63 | ~0.40 | better, likely still > 0.25 |
| PSR(0) | (borderline) | higher for given Sharpe | better (tail terms) |

---

## 3. Relationship to items 1-2 (IDM / cross-sectional) -- complementary, stackable

The three items attack carry's weakness at two DIFFERENT levels, and they compose:

- **Items 1-2 (IDM / cross-sectional demeaning) fix carry's INTERNAL concentration.** An
  Instrument Diversification Multiplier and/or cross-sectional forecast demeaning spread carry's
  risk budget across the 33 roots and strip the common (basket-directional) component, so no
  single root or macro tilt dominates. This attacks the *source* of carry's PBO 0.63 and its
  tail -- the concentration -- directly.
- **Item 3 (combine) adds an EXTERNAL uncorrelated sleeve.** Trend diversifies carry at the
  portfolio level regardless of how carry's internal risk is spread.

These are orthogonal fixes: one makes the carry sleeve internally healthier, the other adds a
second sleeve. **They should stack.** The most promising end-state is NOT "combine instead of
IDM" but **IDM/XS-cleaned carry THEN blended 50/50 with trend**: internal diversification pulls
carry's own PBO down from 0.63 toward ~0.30-0.40, and the external trend blend pulls it further
and compresses the residual tail. Sequencing note: items 1-2 should land first, because the FDM
and the combined moment profile depend on the carry sub-forecast's post-fix distribution. Re-run
the combine on the *fixed* carry, not the current concentrated one.

**Guard against double-counting the win:** if items 1-2 already move carry inside the gate on
their own, the combine's marginal PBO contribution shrinks. Evaluate the combine's *marginal*
effect on top of the item-1-2 carry (delta-PBO, delta-kurtosis, delta-Sharpe), not against raw
carry, so we do not credit the combine for a fix that IDM already delivered.

---

## 4. The multi-signal ENSEMBLE integrity trap (the item's real risk)

Everything above is secondary to this. **Choosing which signals to combine, the weights, and the
FDM is researcher degrees-of-freedom.** Each distinct choice you evaluate and could have selected
on is a trial, and DSR (methodology Section 2.3) deflates the reported Sharpe by the
**project-wide cumulative trial count** -- summed across the whole project, not this run:

```sql
-- Section 9.4: the N fed to DSR
SELECT SUM(combinations_in_run) FROM runs WHERE agent_name = 'backtest-optimizer';
```

If we quietly try {50/50, 60/40, 70/30, risk-weighted, Sharpe-weighted} x {FDM 1.2, 1.4, 1.6} x
{trend+carry, trend-only, carry-only} and report the best, that is ~45 implicit trials. Reporting
the winner as if N=1 (the way the parameter-free Carver TSMOM run legitimately logs
`trial_count = 1`) would be a lie: the DSR would be massively overstated and the ensemble would
"pass" on noise. **This is precisely how the RAMP equity campaign manufactured a false edge** --
signal/regime/weight selection was not counted against the trial budget, so the deflation never
happened and the OOS edge evaporated live.

Non-negotiable controls for this item:

1. **Pre-commit the ensemble as ONE configuration BEFORE running:** 50/50 equal weight,
   FDM = 1.4 (frozen from long-run sub-forecast correlation), cap +/-20, signal set = {trend,
   carry}. Written into the eventual strategy spec, not chosen after seeing results.
2. **If ANY alternative is evaluated, log every evaluated configuration to
   `output/experiments.duckdb`** with an honest `combinations_in_run` (Section 9.1 field), under
   `agent_name = 'backtest-optimizer'` so Section 9.4's `SUM` sweeps it into the project-wide N.
   The pre-committed combine, run through the parameter-free walk-forward harness, logs as a
   single trial only if it truly was the only configuration evaluated.
3. **Weights and FDM are doctrine, not fit** -- never set from observed combined Sharpe. Sharpe-
   or inverse-variance-weighting the sleeves using OOS-observed performance is the forbidden move.
4. **Report DSR against the project-wide N.** A combine that clears PSR but not DSR-at-project-N
   has not passed -- it has only shown it *would* pass if we had never searched. The RAMP lesson
   is that this distinction is the whole game.

---

## 5. Pass / fail target

The combine's job is to do what standalone carry could NOT: clear PBO while keeping a live-worthy
Sharpe. Evaluate on the stitched OOS series from `run_carver_walkforward.py` (train warm-up /
non-overlapping OOS windows, weekly rebalance, 1x and 1.5x cost legs).

**PASS (graduate the combine to the next stage) requires ALL of:**

- **PBO < 0.25** (Section 2.4/2.5) -- the primary objective; AND materially below carry's 0.63.
  Realistically this is achieved only *stacked on the item-1-2 carry* (Section 3); the combine on
  raw carry is expected to land ~0.40 and MISS this gate.
- **OOS Sharpe >= 0.5 at 1x cost** AND **>= 0.5 at 1.5x cost with PSR(0) at 1.5x > 0.90**
  (Section 4.6 cost-sensitivity gate). If the trend dilution drags Sharpe below 0.5, the 50/50
  doctrine was too costly -- REJECT the combine; do NOT then go weight-searching (that is the trap).
- **PSR(0) > 0.95** and **DSR > 0.95 using the project-wide trial count** (Section 2.5). DSR must
  be computed against the honest N (Section 4), not N=1, unless the pre-committed config was
  genuinely the only one evaluated.
- **OOS/IS Sharpe >= 0.7** and **OOS trade/observation count >= 30** (Section 2.5).

**Supporting diagnostics (evidence the thesis is real, not hard gates):**

- **Pearson kurtosis reduced to < ~15** (predicted ~11) vs carry's 33.5 -- confirms the tail-
  smoothing mechanism actually fired. If kurtosis does NOT fall, trend was not the uncorrelated
  hedge we assumed (re-check the realized trend/carry correlation) and the item's premise is wrong.
- **Skew still positive** (predicted ~0.65) -- we want to shed the extreme, not flip the sign.
- **Realized trend/carry sub-forecast correlation ~= 0** -- validates the FDM = 1.4 doctrine choice.

**FAIL / REJECT conditions:**

- Sharpe < 0.5 after the blend -> the give-up exceeded the tail benefit; combine not worth it.
- Kurtosis not materially reduced -> the sleeves were not independent; thesis falsified.
- PBO not improved vs carry -> external diversification did not help; lean entirely on items 1-2.
- Any temptation to weight-search after a marginal result -> STOP and log all trials; a searched
  pass is not a pass.

**Most likely outcome (my prediction):** the combine on *raw* carry lands ~Sharpe 0.68,
kurtosis ~11, skew ~0.65, PBO ~0.40 -- healthier tails but still WEAK on the PBO gate. The real
win is the **stack**: items 1-2 (IDM/XS) applied first to fix carry's internal concentration,
THEN the 50/50 trend blend -- together plausibly PBO < 0.25 at Sharpe ~0.6-0.7. Recommend
sequencing items 1-2 before item 3, and judging item 3 on its *marginal* contribution to the
fixed carry.
