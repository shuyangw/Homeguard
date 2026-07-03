# Research Brief -- Backlog Item 1: Carry Attribution + IDM / Instrument-Risk Weighting

**Date:** 2026-07-03 - **Author:** Strategy Lead (design-precursor; analysis + write-up only, NOT an implementation)
**Backlog source:** `docs/strategies/research/20260703_FUTURES_STRATEGY_BACKLOG.md` item 1
**Depends on (all shipped):** `FuturesCarryStrategy` (`src/strategies/advanced/futures_carry_strategy.py`), the carry cache, the equity-feedback vol-target sizer (`src/backtesting/engine/futures_portfolio_simulator.py::run_sized`), the futures cost model, `run_carver_walkforward.py`.
**Methodology governing this work:** `docs/methodology/backtesting.md` Sections 2 (statistical gate), 3 (walk-forward purge/embargo), 4 (cost). PBO < 0.25 is the acceptance metric.

---

## 0. Problem restatement (why this item is top of the board)

Absolute carry is our strongest signal by mean return -- OOS Sharpe **0.88**, near cost-insensitive (1.5x -> 0.87), positive in **10/11** scored windows -- but it **fails the gate on PBO 0.63** with **skew +1.85, kurtosis 33.5**. The edge is real; its *distribution* is dominated by a few instruments/clusters/days. This is a CONCENTRATION failure, not a signal failure. Item 1 attacks exactly that failure mode by adding the cross-instrument risk-allocation layer the stack currently lacks.

**The structural gap (verified in code):** every instrument is sized *independently* to the full vol target. In `size_from_forecast` (`src/backtesting/utils/position_sizer_futures.py:66`):

```
contracts = (forecast/10) * capital * vol_target * div_mult / (multiplier * price * ann_vol)
```

`div_mult` defaults to `1.0` and is **never passed** -- `run_futures_backtest` calls `sim.run_sized(close, forecasts, daily_vol, vol_target)` at `src/backtesting/engine/futures_backtest.py:89` with no `div_mult`. So there is **no correlation term anywhere**: 33 instruments each target the full vol budget as if they were the only bet. When a whole cluster (e.g. CL/BZ/HO/RB/NG) co-moves, the book takes ~5x the intended risk in that one theme, and that theme's tail days drive the kurtosis. IDM + instrument-risk weights are the doctrine fix for precisely this.

---

## 1. ATTRIBUTION FIRST (prerequisite diagnostic -- run BEFORE touching sizing)

Attribution is a go/no-go router, not a formality. It answers the one question that determines whether IDM is the right tool: **is the concentration cross-sectional (a few instruments/clusters) or is it common-mode (one factor / a handful of dates hitting everything at once)?** IDM fixes the former. It does *nothing* for the latter -- if the +1.85 skew is a single 2020-03 / 2022-energy day common to all instruments, the fix is XS-demean (item 2) or a tail overlay (item 11), and we route there instead. Do not build the weighting layer until attribution confirms the diagnosis.

### 1.1 What to reconstruct

The simulator emits `equity_curve` and `trades` but not per-instrument P&L. Reconstruct the **per-instrument daily P&L contribution** from positions and price diffs (the same MTM identity the simulator uses at `futures_portfolio_simulator.py:93`):

```
pnl_i(t) = position_i(t-1) * multiplier_i * (close_i(t) - close_i(t-1))
r_i(t)   = pnl_i(t) / equity(t-1)          # instrument contribution to book return
R(t)     = sum_i r_i(t)                     # stitched OOS book return (identity check vs equity curve)
```

Positions per day are recoverable by replaying `run_sized`'s provider, or by cumulating the `trades` frame. Verify `sum_i r_i(t)` reconstitutes the reported OOS return series (identity check) before trusting any decomposition.

### 1.2 The three decompositions (per-root AND per-cluster: equity/rates/FX/energy/metals/grains/meats)

Clusters use the EXISTING doctrine `asset_class` map already in `FuturesCarryStrategy` (extended to the 7 economic complexes; energy split out from `commodity` because energy is the prime suspect).

1. **Variance share** -- `Var_share_i = Cov(r_i, R) / Var(R)`; these sum to 1 exactly. Aggregate to clusters. This is the primary concentration read.
2. **Skew / kurtosis attribution** -- two complementary views:
   - *Co-moment contribution:* co-skew `E[r_i * R^2] / E[R^2]^{3/2}`-style term and co-kurtosis term per root/cluster -- how much each instrument *adds to the book's* skew/kurt (not its standalone skew).
   - *Tail-day attribution (more interpretable):* sort `R(t)`; for the top-k and bottom-k |R| days, attribute the day to the root/cluster with the largest `|r_i(t)|`. This directly localizes the +1.85 skew and kurt-33.5 days in time and in instrument. Expect a small date set (COVID-2020, energy-2022) and a small root set.
3. **Return share** -- `Return_share_i = mean(r_i) / mean(R)`, per root and cluster. This is the calibration counterweight to variance share (see 1.4).

### 1.3 Concentration scalars to report

- **Effective number of bets (ENB)** = `1 / sum_i (Var_share_i^2)`. With 33 roots, a healthy book is ENB ~ 10-20; the hypothesis is carry's ENB is ~3-5. ENB is the single number that quantifies the disease and against which the IDM fix is measured.
- **Diversification ratio** = `(sum_i w_i * sigma_i) / sigma_R`.
- **Per-cluster variance share** table (7 rows) + the top-5 single-day contributions with dates and dominant roots.

### 1.4 The routing decision attribution produces

| Attribution finding | Diagnosis | Route |
|---|---|---|
| One/two clusters carry >> their share of variance, ENB low, tail days spread across dates | Cross-sectional cluster concentration | **Proceed to IDM (this item)** |
| Variance share ~ evenly spread but a few DATES dominate all instruments | Common-mode / factor tail | Route to item 2 (XS carry) or 11 (tail overlay); IDM won't help |
| A cluster's **return share >> variance share** | That cluster is the *edge*, not just the risk | IDM will cap it -> Sharpe falls; document the Sharpe/PBO trade explicitly |
| A cluster's **variance share >> return share** | Pure noise concentration | IDM is free lunch (PBO down, Sharpe flat/up) |

The last two rows are THE calibration: IDM's Sharpe cost depends entirely on whether the dominant cluster is also the return driver. Attribution tells us this *before* we run the sizing experiment, so the result is interpretable rather than a surprise.

---

## 2. THE IDM MATH (Carver Instrument Diversification Multiplier + risk weights)

Carver's framework separates two things our stack currently conflates:

**(a) Instrument risk weights `w_i`** -- the fraction of the risk budget allocated to each instrument, `sum_i w_i = 1`. Doctrine allocation is *handcrafted top-down*: equal risk across the 7 clusters, then equal risk within each cluster. So energy (5 roots) collectively gets 1/7 of the budget and each energy root gets `(1/7)/5`; rates (many roots) also collectively capped at 1/7. This is the **cluster cap** -- no single complex can exceed its cluster's budget share no matter how many correlated roots it contains or how large their forecasts are. An optional per-instrument cap (e.g. no single root > 0.15 of total risk) is a doctrine constant on top.

**(b) Instrument Diversification Multiplier `IDM`** -- because the `w_i`-weighted book of imperfectly-correlated instruments has lower realized vol than the sum of its parts, Carver scales the *whole book* back up to the vol target:

```
IDM = 1 / sqrt( w' C w )          # C = correlation matrix of instrument (subsystem) returns
IDM_applied = min(IDM, 2.5)        # doctrine cap
```

For an all-identical-correlation book of N instruments with average correlation rho, `w'Cw = rho + (1-rho)/N`, so IDM ranges from 1 (rho=1, no diversification) toward `1/sqrt(rho)` as N grows. The 2.5 cap is Carver doctrine and prevents a low-correlation estimate from levering the book absurdly.

**How the two compose into sizing:** the effective per-instrument multiplier that replaces the current global `div_mult=1.0` is

```
div_mult_i = w_i * IDM * N_scale
```

where `N_scale` renormalizes so that `div_mult_i` reproduces "each instrument targets its *share* of one vol-target book" rather than "each instrument targets a full vol-target book" (the current, over-risked behavior). Net effect per instrument: today's implicit weight (full budget each) is replaced by `w_i * IDM` (a small fraction each, correlation-aware). Concentrated clusters get their per-root allocation cut hardest; that is the mechanism that compresses skew/kurt.

**Subsystem returns for C:** `C` is estimated from per-instrument *subsystem* returns -- the return of trading instrument i alone at unit risk (forecast x vol-scaled price return), which is exactly `r_i(t)` from the attribution step at unit weight. This reuses the attribution machinery; no new return series definition.

---

## 3. EXACTLY WHERE IT PLUGS INTO OUR STACK

**Decision: it plugs into the SIZING / AGGREGATION step, NOT `forecast_panel`.** This is both the Carver-correct location and the minimal-code location.

- **Why NOT `forecast_panel`:** forecasts are capped +/-20 and are *per-instrument* signal-strength units with no cross-instrument meaning. Multiplying forecasts by `w_i * IDM` would (i) push through the +/-20 cap (IDM > 1), corrupting the forecast convention, and (ii) mislabel a risk-allocation decision as a signal decision. Cross-instrument risk allocation is definitionally a portfolio-construction step, not a signal step. Keep `FuturesCarryStrategy` untouched -- carry stays comparable to and combinable with every other signal.
- **Why the sizing step -- and the slot already exists:** `size_from_forecast` and `run_sized` already thread a `div_mult` argument (`position_sizer_futures.py:53,66`; `futures_portfolio_simulator.py:142,156`). It is currently a global scalar hardwired to `1.0`. The item-1 change is to **promote `div_mult` from a scalar to a per-root vector** `div_mult_i = w_i * IDM * N_scale` and thread it from config through `run_futures_backtest` (`futures_backtest.py:89`, currently omits the argument) into `run_sized`. That is the entire wiring surface: one new weighting module that computes `w_i` (doctrine handcraft) and `IDM` (from `C`), one signature widening (scalar -> `Series`), one call-site change. No signal code, no runner-structure change, harness contract preserved.

**Placement of the correlation estimate in walk-forward:** `C` is computed inside `_run_window` on the **train segment's** subsystem returns only, frozen, then applied to the test segment. It must never see test data (Section 3.2/3.3 -- see integrity below).

---

## 4. THE INTEGRITY CRUX -- IDM is ESTIMATED, so is it parameter-free?

This is the crux the whole item lives or dies on. IDM is derived from a correlation matrix `C`, and any estimate carries three researcher-degrees-of-freedom hazards:

1. **Lookahead:** a full-sample `C` "knows" the future co-movement -- including the crisis co-movements that dominate the tails. Using it is a subtle, severe leak.
2. **Silent trial inflation:** correlation lookback length, shrinkage intensity, and the IDM cap are all knobs. Tuning any of them to maximize OOS Sharpe turns "1 doctrine config" into an unlogged multi-trial sweep and invalidates DSR (Section 2.3: N is project-wide and must count *every* configuration tried).
3. **Is the lookback a fitted parameter for DSR?** Only if it is *selected against results*. An estimate that is fixed by doctrine and never chosen is not a trial (same status as the 25-day vol window in `close_to_close_rv` -- estimated, never counted).

### 4.1 Parameter-free-compatible formulation (pre-committed BEFORE any run)

Two admissible constructions, both `trial_count = 1`; **pre-commit to ONE in the config before running.** Recommend (A) as primary.

- **(A) Handcrafted fixed-correlation IDM (fully data-free -- strongest).** Carver's canonical handcrafted method: cluster grouping is doctrine (the existing `asset_class` map), within/across-cluster correlations are *assumed doctrine constants* (e.g. 0.5 intra-cluster, 0.0 inter-cluster), and IDM comes from a fixed lookup of (N instruments, assumed avg correlation). **Zero estimation -> zero lookahead -> unambiguously not a fitted parameter.** `w_i` is the top-down equal-risk handcraft. This is the cleanest possible DSR posture.
- **(B) Strictly-causal expanding-window empirical `C` (adaptive -- documented alternative).** `C` estimated from all subsystem returns with date <= decision date (expanding, never rolling-with-a-chosen-length, never full-sample). Group structure still doctrine. No lookback to select (expanding = the whole available past), so still non-selected. Carries mild estimation noise early in the sample and a defensible-but-real "why expanding not 3y?" question -- hence the backup, not the primary.

**Trial-count implication (state plainly in `experiments.duckdb`):** committing to exactly ONE of {A, B} with doctrine `w_i`, doctrine cluster map, doctrine IDM cap 2.5, and NO lookback/shrinkage knob keeps this item at **trial_count = 1** -- it does not increment the project-wide DSR `N`. The moment we run both A and B and pick the better OOS, that is **2 trials** and both must be logged; the moment we grid the cap or the lookback, every grid point is a trial. The discipline is: pick A, pre-register it, run it once, log one trial. Only if A fails and we have a pre-committed reason to try B does B get logged as a second trial and DSR re-deflated for N=2.

### 4.2 The rest of the integrity checklist (unchanged from backlog)

- **Full data range** 2010-06-07 .. 2026-02-20, per-window availability filtering; NEVER a sub-window (methodology Sec 1 / backlog integrity item 2).
- **1.5x cost re-check** via `src/backtesting/costs/futures.py` `cost_mult` (Sec 4). IDM raises gross book leverage (multiplier up to 2.5x), so turnover-driven cost must be re-verified -- a bigger book pays more absolute cost; confirm the 1.5x gate still holds.
- **PBO is the primary acceptance metric**; skew/kurt/ENB reported as first-class concentration diagnostics alongside PSR/DSR (Sec 2.5).
- **Purge/embargo** on the walk-forward per Sec 3.2/3.3 (label horizon purge; embargo = 2% of T, NOT the correlation lookback).

---

## 5. PREDICTED EFFECT AND PASS/FAIL TARGET

**Mechanism -> metrics:**

- **Kurtosis (33.5 -> target < ~8):** kurtosis is driven by a few instruments'/clusters' outsized daily contributions. Capping each cluster at 1/7 of risk and each root within it mechanically truncates those contributions' share of `R`, shrinking the tail of the book-return distribution. Strongest, most reliable predicted effect.
- **Skew (+1.85 -> target toward ~0.5):** if the +1.85 is one cluster's asymmetric carry blowups (energy backwardation), cluster-capping compresses it. If attribution (Section 1.4) shows the skew is common-mode across dates, IDM will *not* fix it -- which is exactly why attribution gates this item.
- **ENB (~3-5 -> target ~10-15):** direct target of the weighting; the cleanest before/after read.
- **PBO (0.63 -> target < 0.25 = GATE):** carry is parameter-free, so its PBO does not come from parameter selection -- it comes from the CSCV rank of the return series being *unstable across time blocks* because a few instruments/days dominate different blocks differently. De-concentrating makes the OOS return series more homogeneous across blocks -> the strategy's cross-block rank stabilizes -> `lambda_c < 0` fraction falls -> PBO drops. This is the core bet of item 1.
- **Sharpe (0.88 -> hold >= ~0.6 to stay a deploy candidate):** the outcome is set by the Section 1.4 calibration. If the dominant cluster's variance share >> its return share, Sharpe holds or *rises* (we shed uncompensated risk). If the dominant cluster IS the return driver, Sharpe falls proportionally to the cap -- and a PBO pass bought by cutting Sharpe below ~0.6 is a Pyrrhic pass we would document as MARGINAL, not deploy.

**Pass/fail:**

| Metric | Carry baseline | Item-1 target | Role |
|---|---|---|---|
| PBO | 0.63 | **< 0.25** | GATE (primary acceptance) |
| Kurtosis | 33.5 | < ~8 | concentration diagnostic |
| Skew | +1.85 | toward ~0.5 | concentration diagnostic |
| ENB | ~3-5 (measure) | ~10-15 | weighting effect |
| OOS Sharpe | 0.88 | >= ~0.6 | must-not-break |
| PSR / DSR | (baseline) | > 0.95 each, N=1 | gate |
| 1.5x-cost Sharpe | 0.87 | > 0.5, ~flat | cost gate |

**Verdict logic:** PBO < 0.25 **and** Sharpe >= ~0.6 **and** DSR/PSR pass -> deploy candidate. PBO < 0.25 but Sharpe < 0.6 -> MARGINAL, compose with item 2 (XS carry) before deploying. PBO still >= 0.25 -> IDM was not the fix; the concentration is common-mode, route to item 2/11. A WEAK/REJECT is a valid, publishable outcome.

---

## 6. Handoff / sequencing

Item 1 composes with item 2 (XS carry) -- they are independent de-concentration levers and can stack (XS-demeaned carry sized with IDM). Run item 1 first (attribution + IDM on absolute carry); if attribution shows common-mode concentration, jump to item 2. Whichever clears PBO becomes the ensemble base for items 3/4. This brief is a design-precursor; the implementation plan (weighting module, `div_mult` vectorization, `_run_window` train-only `C`, config schema) is the next step, not part of this document.
