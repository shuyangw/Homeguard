# Futures Strategy Backlog (Prioritized)

**Date:** 2026-07-03 - **Author:** Strategy Lead (orchestrator planning doc)
**Status:** research / planning - NOT implementations. Each item is a test to run through the pluggable futures harness (`run_futures_backtest` + `run_carver_walkforward.py`).

**Harness contract (every item obeys this):** a new signal = a `forecast_panel(close_panel) -> DataFrame` strategy class (per-root forecast in +/-20 Carver units) + a `src/strategies/registry.py` entry + a YAML config. NO runner changes. Forecasts flow unchanged into the equity-feedback vol-target simulator (`FuturesPortfolioSimulator.run_sized`) with the futures cost model (`src/backtesting/costs/futures.py`) and are gated by walk-forward PSR/DSR/PBO + 1.5x cost per `docs/methodology/backtesting.md` Sections 2-4.

---

## Thesis (what the two results so far imply)

Two clean, gate-checked results now bracket the futures alpha space on our 33-root basket:

- **Price-only time-series trend is a robust null here.** Carver multi-speed TSMOM returned OOS Sharpe **0.08**, PBO **0.35** (near coin-flip), and diversifying 3->33 markets did not rescue it. This is not a tuning problem -- it is the absence of an edge in the pure-price-momentum family over this basket/window.
- **Term-structure / risk-premium (carry) is where the edge lives, but it arrives CONCENTRATED.** Absolute carry returned OOS Sharpe **0.88**, near cost-insensitive (1.5x -> 0.87), positive in **10/11** scored windows -- yet gate-failed on **PBO 0.63** with **skew +1.85, kurtosis 33.5**. The edge is real; the *distribution* of that edge is dominated by a few instruments/days.

The implication mirrors the RAMP equities campaign, where momentum-signal tinkering repeatedly died on concentration/BEAR fragility and the durable fixes were **structural diversification**, not signal cleverness. So the highest-EV work here is NOT "find a new signal" -- it is **de-concentrate the carry edge we already have** (instrument/cluster risk weighting, cross-sectional neutralization, tail control) so it clears PBO, and then **add economically-decorrelated term-structure signals and ensemble them** (Carver's "combine many weak, low-correlation signals" thesis) to average the concentration down. Pure-price trend variants sit low on the backlog because the null is robust; they earn their keep only as low-correlation *ensemble members* that smooth carry's tails, never as standalone candidates. Our one unique data asset -- per-contract **open interest** and the built roll calendar -- points at a differentiated cluster (basis-momentum, OI signals) that most retail futures research cannot run.

---

## Ranked summary table

| # | Item | Tier | Family | Feasibility (our data) | Build | EV rationale (vs momentum=WEAK, carry=strong-but-concentrated) |
|---|------|------|--------|------------------------|-------|-----------------------------------------------------------------|
| 1 | **W1: Carry attribution + IDM / instrument-risk weighting** | Do-Next | Carry structural | Feasible now | M | Directly attacks the ONLY thing carry failed on (PBO/concentration). Highest EV on the board. |
| 2 | **Cross-sectional carry (XS demeaned carry)** | Do-Next | Carry structural | Feasible now | S | Ranking removes the common directional bet -> mechanically lower concentration; strongest independent de-concentration lever. |
| 3 | **W2: Carry + trend combined forecast** | High | Ensemble | Feasible now | S | Carver combine; trend is null alone but low-correlation, smooths carry tails. Cheap, high option value. |
| 4 | **Multi-signal ensemble skeleton (carry + trend + basis)** | High | Ensemble | Feasible now | M | The meta-thesis: averaging weak decorrelated signals lowers PBO/kurtosis. Framework unlocks items 5-9. |
| 5 | **Basis-momentum (Boons-Moskowitz)** | High | Term-structure | Feasible now (OI) | M | Uses our unique OI front/back data; economically distinct from carry level -> decorrelated ensemble member. |
| 6 | **Curve/term-structure carry (multi-point slope)** | Medium | Carry variant | Needs curve extension | M | Deeper-than-front/second carry; richer signal, but CarryCalculator is front/second only in v1. |
| 7 | **Open-interest momentum / flow signal** | Medium | Positioning | Feasible now (OI) | M | Unique-data lever; positioning proxy in absence of COT. Unproven edge -> medium. |
| 8 | **Cross-sectional momentum (XS TSMOM)** | Medium | Trend XS | Feasible now | S | XS ranking sometimes survives where TS trend dies; cheap given TS null already mapped. |
| 9 | **Short-term reversal (weekly XS reversal)** | Medium | Reversal | Feasible now | S | Classic commodity reversal; decorrelated to carry/trend; ensemble candidate. |
| 10 | **Skew / positioning proxy from OI+price** | Medium | Positioning | Feasible now (OI) | M | Proxy for the COT/skew premium we cannot source directly. Speculative. |
| 11 | **Trend-strength / vol-regime scaling of carry** | Medium | Carry overlay | Feasible now | S | Scales carry exposure by trend agreement; a tail-control overlay, not new edge. |
| 12 | **Realized-vol term-structure (carry-of-vol proxy)** | Exploratory | Vol | Feasible now (proxy) | M | No vol surface; proxy from realized-vol ratios. Weak prior. |
| 13 | **Seasonality (energy/grains/meats calendar)** | Exploratory | Seasonal | Feasible now | M | Real in ags/energy but inherently parameter-selecting -> DSR-expensive; handle with care. |
| 14 | **XS value / long-horizon reversal (5y)** | Exploratory | Value | Feasible-but-thin | M | Asness value premium; 15.7y gives ~1 non-overlapping estimate -> low statistical power. |
| 15 | **Acceleration / trend-of-trend** | Exploratory | Trend | Feasible now | S | Momentum family -> low prior given the robust TSMOM null. |
| 16 | **Donchian breakout / MAC variants** | Exploratory | Trend | Feasible now | S | Trend re-parameterization; robust null makes this low EV. |
| 17 | **Intraday/overnight session signal (1-min)** | Exploratory | Microstructure | Feasible now (1-min) | L | We have 1-min; session-of-day effects possible but high build + capacity questions. |
| 18 | **COT hedger/speculator positioning** | Blocked-on-data | Positioning | Needs COT data | M | Canonical futures premium; we have NO COT positioning data. |
| 19 | **Options-implied skew / put-call positioning** | Blocked-on-data | Options | Needs options data | M | No futures options / vol-surface data. |
| 20 | **Variance risk premium / implied-vol carry** | Blocked-on-data | Vol | Needs options data | M | Requires implied vol; not sourced. |

---

## Do-Next tier

### 1. W1 - Carry attribution + IDM / instrument-risk weighting
**Hypothesis:** Carry's edge is real (10/11 positive windows) but fails PBO because a few instruments/clusters (likely energy carry blowouts) and a few days dominate; down-weighting correlated clusters and capping any single instrument's risk share will collapse skew/kurtosis and drive PBO below the gate without killing Sharpe.

**Signal construction sketch:** Two-part.
- *Attribution first (analysis, prerequisite):* run the existing `FuturesCarry` walk-forward and decompose the stitched OOS return series by root and by cluster (equity/rates/FX/energy/metals/ags). Identify which roots/clusters produce the +1.85 skew and the largest single-day contributions. This is a diagnostic, not a new `forecast_panel`.
- *IDM / instrument weighting (the strategy):* wrap `FuturesCarryStrategy.forecast_panel` output with Carver instrument weights + Instrument Diversification Multiplier. Compute pairwise correlations of per-instrument *subsystem returns* (forecast x vol-scaled price return), form handcrafted/equal-risk cluster weights so a whole correlated cluster (e.g. all of CL/BZ/HO/RB/NG) cannot dominate, apply per-instrument risk-share cap, then scale the book by IDM = 1/sqrt(w' rho w) capped at ~2.5. Forecasts stay in +/-20 units; only the cross-sectional risk allocation changes.

**Data feasibility:** Feasible now. Uses existing carry cache + continuous close; correlations from in-sample (train segment) subsystem returns only (point-in-time).

**Difficulty:** M (new weighting layer; the correlation estimation must be strictly train-window to avoid lookahead).

**Priority / EV:** Do-Next, top of board. It attacks the single failure mode of our strongest result. Calibration: carry is already Sharpe 0.88; if IDM cuts kurtosis and PBO without cutting Sharpe below ~0.6, this is a deploy candidate -- the best expected value of anything on the list.

**Integrity:** IDM/instrument weights are a DOCTRINE formula (correlation -> weights -> multiplier, cap 2.5), NOT tuned -> trial_count stays 1. Correlations estimated on the TRAIN segment only (no lookahead). Full data range. 1.5x cost re-checked. PBO is the primary acceptance metric here; kurtosis/skew reported alongside as first-class concentration diagnostics.

### 2. Cross-sectional carry (XS demeaned carry)
**Hypothesis:** Ranking instruments by carry and going long high-carry / short low-carry removes the common (basket-directional) component that concentrates absolute carry's returns, yielding a more symmetric, lower-PBO edge.

**Signal construction sketch:** In `forecast_panel`, compute the same risk-adjusted carry as `FuturesCarryStrategy` (`EWMA(carry)/ann_vol`), then **cross-sectionally demean within each asset class** (or within the whole panel) each day and rescale to the +/-20 cap via a fixed cross-sectional scalar. Output = per-root demeaned carry forecast. This is a small delta over the shipped carry class (subtract the daily cross-sectional mean before scaling/capping).

**Data feasibility:** Feasible now. Reuses the carry cache; no new data.

**Difficulty:** S (a demean + rescale on top of existing carry).

**Priority / EV:** Do-Next. Demeaning is the most direct mechanical de-concentration lever after IDM and is independent of it -- if either clears PBO we have a viable carry. Pairs with W1 (they can compose: XS carry + IDM). Prior from equities/futures literature: XS carry is the more common academic form and tends to be better-behaved in the tails than absolute carry.

**Integrity:** Cross-sectional scalar is doctrine (fixed), parameter-free -> trial_count 1. Demean uses only that day's cross-section (no lookahead). Full data range; report PBO + kurtosis as acceptance metrics.

---

## High tier

### 3. W2 - Carry + trend combined forecast
**Hypothesis:** Trend is a null on its own here, but it is low-correlation to carry; a Carver-style equal-weight combined forecast will not add trend's (absent) mean return but will diversify carry's tail days, lowering kurtosis/PBO for a small Sharpe cost.

**Signal construction sketch:** A `CombinedForecastStrategy` whose `forecast_panel` computes both the `CarverMomentumStrategy` combined EWMAC forecast and the `FuturesCarryStrategy` carry forecast per root, then averages them with FIXED weights (Carver forecast-diversification-multiplier applied) and re-caps at +/-20. No new signal math -- pure composition of two shipped signals.

**Data feasibility:** Feasible now. Both component signals already exist.

**Difficulty:** S (composition + FDM re-cap).

**Priority / EV:** High. Cheapest test with real option value: the whole reason to keep the null-Sharpe trend signal is exactly this diversification role. If the combine lowers PBO vs standalone carry it validates the ensemble thesis directly and cheaply.

**Integrity:** Combine weights + FDM are doctrine (fixed) -> trial_count 1. Both components are already parameter-free. Full data; 1.5x cost; PBO/kurtosis reported.

### 4. Multi-signal ensemble skeleton (carry + trend + basis)
**Hypothesis:** Averaging N weak, low-mutual-correlation forecasts reduces the variance of the *portfolio* edge faster than it reduces the mean -> lower PBO and kurtosis than any single signal (Carver's central thesis; especially relevant since carry alone is strong-but-concentrated).

**Signal construction sketch:** Generalize item 3 into an `EnsembleForecastStrategy` that takes a list of component strategy names, calls each component's `forecast_panel`, combines with a doctrine forecast-diversification-multiplier derived from the components' forecast correlation, and re-caps at +/-20. Seed it with {carry, trend, basis-momentum (item 5)}. This is the framework that later items 5/7/8/9 plug into as members.

**Data feasibility:** Feasible now for carry+trend; basis-momentum needs item 5 built first.

**Difficulty:** M (generic composition + correlation-based FDM; the plumbing that unlocks all later ensembles).

**Priority / EV:** High. Even if no single new signal clears the gate alone, the ensemble is the mechanism by which the campaign produces a deployable book. It is the structural analogue of RAMP's "diversification floor" fix.

**Integrity:** FDM is doctrine; **the risk here is silent trial inflation** -- every "which components" choice is a selection. Pre-commit the component set BEFORE seeing results; if we try multiple ensemble compositions, each counts toward the project-wide DSR trial count (`output/experiments.duckdb`). Full data; 1.5x cost.

### 5. Basis-momentum (Boons-Moskowitz)
**Hypothesis:** The momentum of the *basis* (the change in the front-back price relationship), not the price level, predicts returns -- an economically distinct signal from carry *level* and from price trend, hence a strong decorrelated ensemble member.

**Signal construction sketch:** Precompute (cache, mirroring `build_carry_cache.py`) a per-root series of the front and second-contract cumulative returns using the OI-ranked front/back from the roll calendar (`RollCalendar.get_nth_by_oi`, n in {0,1}). Basis-momentum = trailing return of front minus trailing return of second, over a Carver-doctrine lookback. In `forecast_panel`, EWMA-smooth, normalize by price vol, scale + cap to +/-20.

**Data feasibility:** Feasible now -- this is the item that leans into our UNIQUE data (per-contract OI + roll calendar give a clean, point-in-time front/back pair). Requires a new cached series (name it: `basis_dir()/{root}.parquet`).

**Difficulty:** M (new cache builder + strategy; must reuse OI roll logic to avoid a second, inconsistent front/back definition).

**Priority / EV:** High within the "new signal" set because it is (a) data-differentiated and (b) economically decorrelated from carry level. Calibrated expectation: standalone likely WEAK, but a high-value ensemble member for item 4.

**Integrity:** Lookback span uses Carver doctrine (fixed) -> parameter-free. Front/back from OI-ranked point-in-time roll calendar (no lookahead -- same discipline as carry). Full data; 1.5x cost. If a span must be *chosen*, it carries a purged/embargoed walk-forward + honest trial count.

---

## Medium tier

### 6. Curve / term-structure carry (multi-point slope)
**Hypothesis:** Carry measured across more of the curve (front / second / third) than the front-second pair is a smoother, less roll-artifact-prone estimate of the term-structure premium.

**Signal construction sketch:** Extend `CarryCalculator` (currently front/second only, v1) to rank the top-3 OI contracts and fit a slope of log-price vs time-to-expiry; use the fitted slope as annualized carry. `forecast_panel` otherwise identical to carry.

**Data feasibility:** Needs a CarryCalculator extension to a 3rd contract point (data exists -- per-contract 1-min has all outrights -- but the calculator does not expose the 3rd point yet). Name: `CarryCalculator.get_nth` / curve-fit method.

**Difficulty:** M (calculator extension + new cache).

**Priority / EV:** Medium. A refinement of the strongest edge; upside is a cleaner (lower-kurtosis) carry, but it does not fundamentally change the concentration story the way IDM/XS do.

**Integrity:** Curve-fit is deterministic (fixed 3 points), parameter-free. Point-in-time OI ranking. Full data; 1.5x cost.

### 7. Open-interest momentum / flow signal
**Hypothesis:** Rising open interest alongside a price move signals conviction/flow that predicts continuation; falling OI signals exhaustion -- a positioning proxy we can build because we uniquely hold per-contract OI.

**Signal construction sketch:** From per-contract 1-min OI aggregated to daily (front + back), build a root-level OI series (or total-across-outrights). Signal = sign(price trend) x normalized OI change, EWMA-smoothed, vol-normalized, scaled/capped. Cache the daily OI-total series per root.

**Data feasibility:** Feasible now (unique OI data). New cache: `oi_dir()/{root}.parquet`.

**Difficulty:** M (OI aggregation cache + strategy).

**Priority / EV:** Medium. Unique-data lever and a partial substitute for the COT positioning premium we cannot source -- but the edge is unproven and OI dynamics are noisy. Best value as an ensemble member.

**Integrity:** Doctrine spans (fixed). OI is point-in-time (as-of daily close). Full data; 1.5x cost. Any span selection -> purged/embargoed WF + honest trial count.

### 8. Cross-sectional momentum (XS TSMOM)
**Hypothesis:** Ranking instruments by trailing return and going long winners / short losers may survive where absolute time-series trend died, because XS ranking is dollar/basket-neutral and isolates relative strength.

**Signal construction sketch:** In `forecast_panel`, compute each root's trailing risk-adjusted return (reuse the Carver EWMAC combined forecast as the raw score), then cross-sectionally demean each day and rescale/cap -- the trend analogue of item 2.

**Data feasibility:** Feasible now (close only).

**Difficulty:** S (demean over the existing momentum forecast).

**Priority / EV:** Medium. Cheap given TS trend is already mapped; XS momentum has an independent literature and could behave differently in the tails. Low prior on a standalone pass given the TS null, but useful as an ensemble member and near-zero marginal cost.

**Integrity:** Cross-sectional scalar doctrine (fixed) -> parameter-free. Demean same-day only. Full data; 1.5x cost.

### 9. Short-term reversal (weekly XS reversal)
**Hypothesis:** Over ~1 week, futures that outperformed the cross-section revert; a short-horizon reversal signal is decorrelated to both carry (level) and trend (long-horizon), a clean ensemble diversifier.

**Signal construction sketch:** `forecast_panel` = negative of the cross-sectionally demeaned trailing 5-day return, vol-normalized, scaled/capped to +/-20.

**Data feasibility:** Feasible now (close only).

**Difficulty:** S.

**Priority / EV:** Medium. Distinct horizon and sign from everything else on the list -> genuine diversification for the ensemble. Standalone edge uncertain; capacity is a concern (short horizon = high turnover) -- the 1.5x cost gate is the real test here.

**Integrity:** 5-day horizon is doctrine (fixed weekly). Watch turnover/cost sensitivity closely (this item is the most cost-fragile). Full data; 1.5x cost mandatory as the deciding gate.

### 10. Skew / positioning proxy from OI + price
**Hypothesis:** A proxy for the skewness/positioning risk premium (normally from COT or options) can be constructed from realized return skew and OI concentration; instruments with crowded positioning / negative skew earn a premium.

**Signal construction sketch:** Per root, rolling realized skew of daily returns (doctrine window) combined with OI concentration (front OI share of total OI). `forecast_panel` = cross-sectionally ranked composite, scaled/capped.

**Data feasibility:** Feasible now via OI + close (a proxy -- NOT true COT/options positioning).

**Difficulty:** M.

**Priority / EV:** Medium-low. Speculative proxy for a real premium; value is mostly as a decorrelated ensemble member. Explicitly a substitute for the blocked-on-data COT/options items.

**Integrity:** Windows doctrine (fixed). Point-in-time OI/returns. Full data; 1.5x cost. Composite weighting must be pre-committed, not tuned.

### 11. Trend-strength / vol-regime scaling of carry
**Hypothesis:** Carry pays best when the trend does not disagree; scaling carry exposure down when trend and carry conflict (or when vol regime is elevated) tames the tail days that drive carry's kurtosis.

**Signal construction sketch:** Overlay: carry forecast x an agreement/vol-regime multiplier in [0,1] derived from the sign concordance of the carry and trend forecasts (and/or a realized-vol regime flag). Re-cap +/-20. No new signal -- a multiplicative overlay on carry.

**Data feasibility:** Feasible now.

**Difficulty:** S.

**Priority / EV:** Medium. A targeted tail-control overlay aimed squarely at carry's kurtosis; overlaps with the intent of item 3 (combine) but is multiplicative rather than additive. Test only if item 3 underwhelms.

**Integrity:** Multiplier form is doctrine/rule-based (fixed thresholds); any threshold choice inflates trial count -> prefer a continuous doctrine transform. Full data; 1.5x cost.

---

## Exploratory tier

### 12. Realized-vol term-structure ("carry of vol" proxy)
**Hypothesis:** The ratio of short-horizon to long-horizon realized vol proxies the vol term-structure premium; instruments in backwardated vol earn a premium. **Feasibility:** feasible now as a realized-vol proxy (we have NO implied-vol surface). **Build:** M. **EV:** Exploratory -- weak prior, proxy quality unknown. **Integrity:** doctrine vol spans; parameter-free; full data; 1.5x cost.

### 13. Seasonality (energy / grains / meats calendar)
**Hypothesis:** Ags/energy/meats have genuine calendar seasonality (planting/harvest, driving/heating season, herd cycles). **Signal:** per-root month-of-year expected-return estimated on the TRAIN window only, projected forward as a capped forecast. **Feasibility:** feasible now. **Build:** M. **EV:** Exploratory -- real economic basis but **inherently parameter-selecting** (which months, which roots). **Integrity flag:** this item CANNOT stay trial_count=1 -- seasonal estimation is a fit; it MUST carry a purged/embargoed walk-forward and an honest project-wide trial count, and seasonal means must be estimated strictly in-sample (severe lookahead risk otherwise). Lowest-trust item that still has a real prior.

### 14. Cross-sectional value / long-horizon reversal (5y)
**Hypothesis:** The Asness-style value premium -- 5-year reversal (cheap = strongly negative 5y return) predicts outperformance. **Signal:** XS-demeaned negative 5y return, scaled/capped. **Feasibility:** feasible-but-thin -- 15.7y of data yields only ~1 non-overlapping 5y estimate, so statistical power for a 5y signal is very low. **Build:** M. **EV:** Exploratory (data-limited). **Integrity:** doctrine horizon; the thin sample means DSR/PBO will be unforgiving -- report power honestly.

### 15. Acceleration / trend-of-trend
Second-difference of the trend forecast. Feasible now, S build. Exploratory -- momentum family, so the robust TSMOM null gives it a low prior. Parameter-free (doctrine spans).

### 16. Donchian breakout / MAC variants
N-day channel breakout or moving-average-crossover re-parameterizations of trend. Feasible now, S build. Exploratory -- re-parameterizing a signal family we already showed is null; only justified as ensemble members, and each channel length is a trial (guard DSR).

### 17. Intraday / overnight session signal (1-min)
**Hypothesis:** Session-of-day effects (overnight vs RTH drift) exist in index/rates futures. **Feasibility:** feasible now -- we have per-contract 1-min (front/second). **Build:** L (the harness is daily-close; an intraday signal needs a daily *summary* feature -- e.g., overnight-return sign -- fed through `forecast_panel`, or a harness extension). **EV:** Exploratory -- high build, capacity/cost questions at intraday turnover. Keep as a daily-summarized feature to stay inside the daily harness.

---

## Blocked-on-data tier

### 18. COT hedger / speculator positioning
Canonical futures premium (commercials vs non-commercials net positioning predicts returns). **Blocked:** we have NO COT positioning data. Needs: CFTC Commitments-of-Traders weekly series ingested and cached point-in-time (release-lagged to avoid lookahead). Item 10 (OI/price skew proxy) is the feasible-now stand-in.

### 19. Options-implied skew / put-call positioning
Risk-reversal skew and put-call ratios as a positioning/tail-premium signal. **Blocked:** no futures options data. Needs: futures options chains / vol surface.

### 20. Variance risk premium / implied-vol carry
Long realized-vol-cheap / short implied-vol-rich, or IV-RV carry. **Blocked:** requires implied vol (options). Item 12 (realized-vol term-structure) is the feasible-now proxy.

---

## Integrity requirements baked into every item

1. **Parameter-free discipline (DSR trial count).** Doctrine constants (forecast scalars, EWMA spans 4/16/64/256, cap 20, carry_scalar ~30, IDM cap ~2.5, cross-sectional scalars) are FIXED and NEVER optimized -> each such item is a single non-selected configuration -> project-wide DSR trial count stays low. Items that INHERENTLY require selection are flagged: **#13 seasonality** (seasonal-month fit), **#16 Donchian** (channel length), and any item where a lookback must be *chosen* rather than taken from doctrine. Those MUST carry a purged/embargoed walk-forward and an honest project-wide trial count logged to `output/experiments.duckdb`. **#4/#16 ensemble-composition and channel choices silently inflate trials -- pre-commit them.**
2. **Full available data range.** Every item runs 2010-06-07 .. 2026-02-20 with per-window data-availability filtering (instruments phase in). NEVER a sub-window. Ratio-adjusted continuous close is the return/vol basis; per-contract data (OI, front/back, curve points) is for SIGNAL construction only.
3. **No lookahead.** Carry, OI, roll calendar, basis, and any correlation/seasonal estimate are point-in-time: correlations and seasonal means come from the TRAIN segment only; OI/carry are as-of daily close; front/back come from the OI-ranked roll calendar. This is the same discipline the shipped carry class already honors.
4. **Combined statistical gate + cost sensitivity.** Acceptance = PSR/DSR/PBO on the stitched OOS series + 1.5x cost (`src/backtesting/costs/futures.py`) per methodology Sections 2-4. **PBO/kurtosis/skew are first-class acceptance metrics, not afterthoughts** -- concentration is the demonstrated failure mode (carry), so an item with high Sharpe but PBO > gate is a REJECT. A WEAK/REJECT is a VALID, publishable outcome.
5. **Capacity / turnover honesty.** High-turnover items (#9 short-term reversal, #17 intraday) live or die on the 1.5x cost gate; report realized turnover and per-instrument capacity for any live-bound candidate.

---

## Recommended sequence

Do items **1, 2, 3** first (all reuse the shipped carry/trend classes, all directly test the de-concentration thesis, S/M build). Build **4** (ensemble skeleton) as the framework, then **5** (basis-momentum) as its first data-differentiated member. Everything below #5 is contingent on whether the carry-de-concentration work (#1/#2) produces a gate-passing base to ensemble around.
