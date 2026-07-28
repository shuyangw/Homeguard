# Options Slate Feasibility Screen — 50 Candidates vs the 1-Minute OHLCV Constraint (v1)

**Date:** 2026-07-24
**Companion to:** `2026-07-24_options_strategy_slate_v1.md` (candidate definitions, pre-registered parameters, priors)
**Inputs:** slate v1 + deep research pass (vendor reality, backtest methodology, 2024–2026 cost stack, anomaly-decay evidence, auxiliary-data availability, liquidity screens)
**Pipeline stages executed here:** stage 5 (data screen), stage 6 (cost-viability screen), stage 7 (diversity economization), stage 8 (adversarial critique), plus verdicts and a validation-wave protocol
**Status:** Still blind. No candidate has touched a backtest return. Every number below is either a vendor/market fact from the research pass or pre-registered analytic arithmetic.

---

## TL;DR

- **No options strategy is researchable from underlying 1-minute Open-High-Low-Close-Volume (OHLCV) bars alone — but that was never the real question.** The real question is the minimum options-side purchase, and the answer is decisive: **ORATS near-end-of-day (EOD) history (~$399 one-time, full chains with National Best Bid and Offer (NBBO)-derived quotes + smoothed greeks/implied volatility (IV), 2007–present)** unlocks 21 GO candidates immediately. A second-stage purchase (ThetaData Value ~$40/mo quote bars, or Databento OPRA CBBO-1m) unlocks the intraday-dependent handful — and should be gated on Stage-1 results, not bought speculatively.
- **Verdict tally: 21 GO · 18 CONDITIONAL (named, mostly cheap conditions) · 7 DEFER (near-duplicates folded into parents) · 3 DROP (dead priors, not data) · 1 NO-GO (OPT-023 PEAD — point-in-time consensus unobtainable at retail AND the large-cap anomaly is evidenced dead).** The slate survives the feasibility screen; **no reiteration of the generation step is required.** Reiteration criteria are defined at the end for the *validation* stage, where zero survivors remains a legitimate outcome.
- **The 1-minute underlying data earns its keep in exactly four flagship candidates** — OPT-019, 036, 048, 049 — where it powers Yang-Zhang / Heterogeneous Autoregressive Realized Volatility (HAR-RV) estimation that EOD-bar competitors cannot match, plus free signal-side pre-tests for the intraday candidates. It is a **volatility-estimator and pre-test asset, not an options-pricing asset.** Counterintuitively, the most "1-minute-flavored" candidates (0DTE — zero days to expiration — and intraday structures) are the *least* worth exploring: they need purchased quote data, face the worst cost arithmetic, and sit on the most-decayed premia.
- **Hard rule inherited from the research pass:** trade-print-derived option OHLC bars are prohibited as marks for out-of-the-money (OTM) or low-volume strikes. Intraday options research uses NBBO **quote** bars or does not happen. This single rule is what separates a legitimate harness from a silently corrupted one.

---

## 1. Method

**Verdict taxonomy (pre-registered):**
- **GO** — researchable after the Stage-1 purchase; passes the analytic cost screen; prior not evidenced dead. Assigned a validation wave.
- **CONDITIONAL** — a *named* blocker must clear first: a data purchase, a free/cheap diagnostic outcome, a spec amendment (registered here, before any data contact), a parent candidate's result, or an external gate (e.g., the Homeguard PR6 momentum re-baseline).
- **DEFER** — flagged near-duplicate in slate v1; folded into its parent(s). Tested only if the parent clears validation, under the sequential-testing rule in §7. Economizes trials.
- **DROP** — removed from the queue on prior + research evidence (a dead trial is not worth a draw from the finite historical sample). Not data-blocked.
- **NO-GO** — data unobtainable at retail scale and/or mechanism evidenced dead. Removed.

**Screens applied per candidate:** (a) data screen against the purchase ladder in §2; (b) analytic cost screen per the model in §3 — verdicts PASS (plausible gross ≥ 2× modeled cost drag), MARGINAL (1–2×), FAIL (<1×); (c) research-informed prior update (this is research-based prior movement, permitted by the generator's invariant — it is not backtest-based); (d) adversarial critique: the most likely reason the mechanism is spurious *now*, and the sharpest falsifier.

**Diagnostics vs trials.** Several candidates get "descriptive diagnostics" — measurements of data properties (financing ratios, episode counts, IV pre-discounting) that involve **no strategy P&L and no selection on returns**. These are logged but do not increment the Deflated Sharpe Ratio (DSR) trial count. Strategy backtests do. The distinction is load-bearing and recorded per candidate.

---

## 2. Data ground truth — the purchase ladder

| Rung | Product | What it is | History | Cost | Unlocks |
|---|---|---|---|---|---|
| 0 | Already owned | Alpaca SIP underlying 1m OHLCV; CBOE VIX term structure (free); Homeguard classifier history (internal) | 2016+/2010+ | $0 | All Wave-0 diagnostics and pre-tests |
| 1 | **ORATS near-EOD historical** | Full chains, NBBO-based bid/ask, smoothed (SMV) greeks + IV, snapshot ~14 min before close | **2007–present (~19.5 y)** | **~$399 one-time** | 21 GO + most CONDITIONAL diagnostics |
| 2a | ThetaData Value | 1-minute **quote (NBBO)** + OHLC + open interest (OI) endpoints; no greeks at this tier | 2020-01+ (~6.5 y) | ~$40/mo | OPT-014/041/042 marks; OPT-043 OI; open-print marks for 021/040 |
| 2b | Databento OPRA CBBO-1m | Consolidated minute NBBO, cleanest raw feed, no greeks | 2013+ (~13 y) | usage-based (cheap for 2 roots) | Same as 2a for SPY/QQQ, longer history, DIY IV |
| 3 | Earnings-calendar API (FMP / Massive-Benzinga tier) | Approximate historical announce dates + BMO/AMC | varies | low $ | 020/021/024/026/045 exclusion windows (with QA caveat) |
| — | True point-in-time (PIT) consensus (Bloomberg/Zacks/Estimize) | Non-restated estimates/surprise | — | institutional | **Not acquired** — OPT-023 blocked |

**Prohibitions and rules inherited from the research pass (binding on the harness):**
1. **No trade-print OHLC bars as option marks** for OTM/low-volume strikes — quote bars only. Stale/empty trade bars on illiquid strikes silently bias P&L.
2. **No raw "close" prints as EOD marks** — use the ORATS near-close snapshot convention.
3. **The T+1-open mark is structurally unreliable** (widest spreads, sparsest prints). Event candidates are amended in this document to near-close exits, or wait for Rung-2 quote data.
4. **ThetaData greeks caveat:** their Black-Scholes IV has documented rising `iv_error` deep ITM/OTM and ignores dividends by default; ORATS' smoothed low-delta IVs are the preferred deep-OTM source (matters for OPT-030/047).
5. **Universe floor:** below roughly the top ~100–150 names by option volume, multi-leg retail execution degrades badly; universe re-pulled quarterly from OCC/volume rankings, ≥1,000 contracts/day average.

**History depth is a statistical asset.** σ(annualized Sharpe estimate) ≈ 1/√years under the null: 19.5 y (ORATS) → ~0.23; 13 y (Databento) → ~0.28; 6.5 y (ThetaData Value) → ~0.39. A 6.5-year intraday trial must clear roughly the same best-of-null bar as ~30 EOD trials on 19.5 years (§7). Short history is expensive in significance terms — a second, independent reason intraday candidates rank last.

---

## 3. Pre-registered cost model v1 (analytic stand-in for `cost_viability.py`)

Fill convention (restated vs mid, from the ORATS practitioner convention): single-leg orders cost ~**25% of quoted width beyond mid** per transit; 3–4-leg native combos ~**3–6% of width per leg** beyond mid (complex-order-book netting); **event/stressed windows: treat combos as single-leg-grade (15–25%)**. Fees: ~$0.42–0.72/contract one-way all-in at retail (commission $0.35–0.65 tiered/Lite + ~$0.05–0.07 regulatory: Options Clearing Corporation (OCC) $0.025, Options Regulatory Fee ~$0.023, Consolidated Audit Trail $0.0003, FINRA Trading Activity Fee $0.00329 on sells; SEC Section 31 returns at $20.60/$1M of sale proceeds from 2026-04-04 — negligible at these premia). Width assumptions (normal conditions): SPY/QQQ ATM 30–45 days-to-expiration (DTE) $0.03–0.12; SPY 0DTE ATM $0.01–0.02; SPY 0.05–0.10Δ $0.02–0.06 on $0.30–1.50 value; tier-1 single-name ATM $0.05–0.15; rank-50–100 ATM $0.10–0.30; event windows 1.5–3× normal; LEAPS $0.30–1.00. **Sensitivity ±50% mandatory; replace with own IBKR fill data once live.**

Two structural conclusions fall out before any per-candidate work: **(i)** monthly-cadence index structures pass by an order of magnitude under normal widths (round-trip drag ~0.2–1% of premium) — their binding cost risk is *stressed-regime* widths and hedge slippage, not steady-state friction; **(ii)** the cost screen bites, in descending severity, on: daily-cadence multi-leg (0DTE), event-window single-name multi-leg, rank-50–100 names, deep-OTM legs, and LEAPS rolls.

---

## 4. What the 1-minute underlying OHLCV asset is actually for

Direct answer to the framing question ("worth exploring with 1m OHLCV data?"):

1. **Volatility-estimator edge (the four flagships):** OPT-019 (HAR-RV throttle), OPT-036 (Yang-Zhang gate), OPT-048 (compression percentile), OPT-049 (hedge-band triggers + hedge ledger). Here 1m bars are a genuine informational input EOD-bar competitors lack — this is where the asset creates prior elevation, and all four are GO or cheap-CONDITIONAL.
2. **Free signal-side pre-tests (Wave 0):** gap-continuation and first-hour-trend effects (for OPT-014/042), weekend realized-variance decomposition (OPT-040), compression-episode census (OPT-048), classifier-transition realized-vol timing (OPT-050) — all measurable on already-owned data at zero cost and zero options-data spend, killing or promoting the dependent candidates before any purchase.
3. **Hedge-execution ledgers:** daily/band-triggered share hedges for OPT-015/019/028/044/049 are priced off own underlying data — option marks are only needed at entry/exit/terminal, which is precisely what keeps these candidates EOD-researchable.
4. **What it is NOT:** a source of option marks, ever; and 1-minute *option trade* bars — the superficially adjacent product — are affirmatively dangerous (§2 rule 1).

---

# 5. Per-Candidate Screens

Format: verdict + wave, then Data / Cost / Prior update / Adversarial / Condition-or-next-action.

## F7 — Overwrite / Equity-Carry

### OPT-001 — Delta-Targeted Covered Call Overwrite — **GO (Wave 1)**
**Data:** D2 fully satisfied by ORATS near-EOD (chains + deltas for the 0.25Δ strike pick, 2007+). Regime gate uses live classifier — point-in-time by construction.
**Cost:** 1 leg, ~12 round trips/yr/name, ATM-adjacent index/tier-1 widths → drag <1% of premium collected annually. Fees trivial. **PASS** with wide margin. Build item: dividend-driven early-assignment module for short in-the-money calls (mandatory for every short-call structure below; flagged once here).
**Prior update:** BXM-family evidence intact structurally, but gross premium thinner in the compressed-VRP regime — expect income, not alpha, and judge against forfeited upside per the registered falsifier.
**Adversarial:** most likely spurious "edge" = mistaking equity beta + premium income for skill; the regime gate could also lag STRONG_BULL entries, systematically forfeiting the best months. Falsifier unchanged.
**Next:** queue Wave 1 on ORATS purchase. Portfolio note: overlaps RAMP's long-equity capital — a book-construction question, not a research blocker.

### OPT-002 — Cash-Secured Put Wheel — **GO (Wave 1)**
**Data:** D2 via ORATS. **PASS.**
**Cost:** as OPT-001; assignment transitions add share transactions (cheap). **PASS.**
**Prior update:** PUT-index long-run evidence intact (~10.4% annualized since 1986, lower vol than S&P). Critical economics note sharpened by the rate environment: the secured cash competes against the risk-free rate — the researchable object is *premium net of T-bill yield on the secured cash*, and the harness must account P&L that way or the result flatters.
**Adversarial:** clustering of assignments at regime breaks is the whole risk; the BEAR-suspension gate is doing all the work — test with and without is NOT permitted post hoc (one registered spec only).
**Next:** Wave 1.

### OPT-003 — Financed Collar Carry — **CONDITIONAL (diagnostic; then Wave 2)**
**Data:** D2+D5 via ORATS (two expiries, marks fine). 
**Cost:** ~2 legs/mo + quarterly put roll → PASS.
**Prior update:** research reinforced the drafting-time skepticism — financing ratios frequently fail.
**Adversarial:** the structure can silently become "expensive insurance plus capped upside," the worst of both; falsifier is the financing-ratio distribution itself.
**Condition (descriptive diagnostic, no trial consumed):** measure the historical distribution of (1M 0.20Δ call premium × 3) ÷ (3M 0.15Δ put cost) on ORATS 2007+. Pre-registered gate: proceed to a backtest only if financing ratio ≥ 0.7 in ≥ 60% of months; otherwise DROP without spending a trial.

### OPT-004 — Covered Strangle — **DEFER (parents: OPT-001 ∧ OPT-002)**
Flagged ½-trial duplicate in slate v1. Data/cost identical to parents (PASS). Tested only if both parents clear validation; its marginal question — does the *joint* structure with its regime kill-rules beat the sum of parents — is only worth a draw from the sample after the parents earn one. Sequential rule registered in §7.

### OPT-005 — Laddered CSP Pullback Grid — **GO (Wave 2)**
**Data:** D2 + D1 trigger (own bars). **PASS.**
**Cost:** low-moderate turnover; index widths → **PASS.**
**Prior update:** unchanged (B); the conditional-richening thesis remains plausible and under-documented — genuinely informative either way.
**Adversarial:** the 2% pullback trigger is a regime-transition detector wearing a mean-reversion costume; in 2008/2020/2022-style tapes the ladder deploys into the knife. The registered falsifier (post-trigger put VRP ≤ unconditional) is measurable as a *diagnostic* on ORATS before the P&L test — run it first; if it fails, DROP pre-trial.
**Next:** diagnostic then Wave 2.

### OPT-006 — Poor Man's Covered Call (LEAPS Diagonal) — **GO (Wave 2, cost-sensitivity mandatory)**
**Data:** D2+D5; ORATS smoothed marks mitigate the wide-LEAPS-quote problem; ThetaData greeks explicitly disfavored here (dividend handling + deep-ITM iv_error).
**Cost:** LEAPS round trip ~1.5–3% of option value at $0.30–1.00 widths (single-leg convention — combos across expiries fill worse) + monthly short legs. Annualized drag ~3–5% of deployed capital worst case. **MARGINAL-PASS**; the ±50% sensitivity run is the actual test.
**Prior update:** unchanged (B).
**Adversarial:** capital efficiency is the sales pitch; vega is the bill — a vol crush marks the long leg down exactly when the overwrite is working, and backtests that mark LEAPS at smoothed mids will understate exit friction. Falsifier: net carry < OPT-001 on matched notional.
**Next:** Wave 2, only alongside OPT-001 (its benchmark).

## F5 — Directional Regime Translation

### OPT-007 — Bull Call Debit Spread on Momentum — **CONDITIONAL (external gate: PR6 re-baseline)**
**Data:** D2+D5 via ORATS; signal side from own bars. Feasible.
**Cost:** 2-leg tier-1 single-name, monthly-ish → PASS.
**Prior update:** demoted. This candidate inherits its edge entirely from underlying momentum — and Homeguard's own live evidence (RAMP alpha decay; cost floor vs gross alpha) plus the parked PR6 weekly-cadence re-baseline mean the *underlying* edge is currently unproven at the relevant holding period. The options wrapper adds spread drag to whatever remains.
**Adversarial:** the most likely spurious outcome is a backtest that re-discovers 2009–2021 momentum beta. Falsifier unchanged (spread P&L < delta-equivalent stock P&L on matched signals).
**Condition:** blocked until the PR6 plain-momentum re-baseline (already the registered go/no-go for RAMP Phase 4C) returns positive at matched cadence. If PR6 fails, OPT-007/008/009 DROP together without consuming trials — correct cross-workstream sequencing, zero wasted budget.

### OPT-008 — ZEBRA Synthetic Momentum — **CONDITIONAL (same PR6 gate)**
Data/cost: fine (3 legs, index, low frequency → PASS). Prior/adversarial as OPT-007, plus the specific falsifier that the tail-floor's value must beat a plain index-put budget. Rides or dies with the PR6 gate.

### OPT-009 — Bullish Risk Reversal on Momentum Leaders — **CONDITIONAL (same PR6 gate)**
Data/cost: fine (2 legs, tier-1 names → PASS; margin-intensive noted). Additional adversarial sharpening from research: the short-put leg is a concentrated bet that 5 correlated leaders avoid a regime break simultaneously — the undefined-risk member of the gated trio, and the first to cut if only some survive the gate.

### OPT-010 — Call Ratio Backspread on Breakout — **DROP**
Slate prior C; the research pass surfaced no evidence that breakout right-tails are fat enough to pay the valley-of-death, and the structure's edge claim was payoff-shape cleverness rather than mechanism. A dead trial is not worth a draw from the finite sample. Removed; ledger status = emitted-not-tested.

### OPT-011 — Bear Put Debit Spread on Breakdown — **GO (Wave 3)**
**Data:** D2+D5+D1. Feasible; ORATS 2007+ contains a usable census of BEAR regimes (2008–09, 2011, 2015–16, 2018, 2020, 2022).
**Cost:** as OPT-007 → PASS.
**Prior update:** held at B. Distinct from the PR6-gated trio: short-side momentum has a different mechanism (borrow constraints, which options bypass) and RAMP's BEAR-regime failure was long-side *selection*, not evidence against short-side drift. Explicit design note: the signal layer must NOT reuse RAMP's selection machinery.
**Adversarial:** put IV is skew-rich exactly when the signal fires — the strategy buys expensive direction; falsifier (entry IV richness consumes drift) is measurable as a diagnostic (entry-day IV percentile census) before the P&L run.
**Next:** diagnostic then Wave 3.

### OPT-012 — Bearish Risk Reversal — **DROP**
½-trial duplicate of OPT-011 with an undefined rally-tail; the research pass reinforced that bear-market rallies are the most violent, which is precisely this structure's short leg. OPT-011's defined-risk version dominates ex ante. Removed.

### OPT-013 — Put Ratio Backspread on Deterioration — **GO (Wave 2)**
**Data:** D2+D5 + free CBOE VIX term structure. Fully unlocked at Rung 1.
**Cost:** 3 legs, episodic → PASS.
**Prior update:** held at B; the conditional-entry version remains the defensible form of crash convexity.
**Adversarial:** the 2022 falsifier is live — slow-grind declines settle in the valley between strikes. Sharpened pre-test: census of historical drawdown *shapes* (gap vs grind) conditional on the entry trigger, from own underlying data + VIX term — free, Wave 0.
**Next:** Wave 2.

### OPT-014 — Opening-Range Gap Same-Day Verticals — **CONDITIONAL (Wave-0 pre-test → Rung-2 purchase)**
**Data:** D3/D4 REQUIRED — the designated intraday-marks probe. Legitimate only on NBBO quote bars (ThetaData Value or Databento CBBO-1m); trade-OHLC bars prohibited.
**Cost:** ~2 legs × 2 transits × ~daily-when-triggered at 0–1 DTE: ~$0.05–0.07/trade modeled → demands per-trade gross ≥ ~$0.12–0.15. **MARGINAL-to-FAIL** unless the effect size is large.
**Prior update:** demoted — intraday index momentum is documented but heavily exploited post-0DTE-boom; post-publication decay likely.
**Adversarial:** the effect may live entirely in the underlying's first minutes and be gone by a 10:00 options fill.
**Condition (free, Wave 0):** measure gap-continuation effect size on own SPY/QQQ 1m bars, 2016+. Pre-registered gate: proceed to Rung-2 purchase for this candidate only if the underlying effect ≥ 15 bps/trade net of a 2-bp underlying-slippage haircut post-2023. Otherwise DROP with zero spend.

## F1 — Volatility Risk Premium (VRP)

### OPT-015 — Index Delta-Hedged Short Straddle — **GO (Wave 1, family anchor)**
**Data:** D2+D5 via ORATS; daily 15:50 hedge marks from own underlying data align with the ORATS ~15:46 snapshot convention — internally consistent mark times, a nontrivial correctness detail now locked in.
**Cost:** 2-leg index combo monthly + daily share hedges (~1–2 bps each): total drag ~1–2% of premium annually under normal widths → **PASS** by an order of magnitude. Binding cost risk is stressed-width exits, which the harness must model with the event/stressed convention.
**Prior update:** the material update of the whole screen — index VRP has compressed post-2022, with 12-month rolling VRP episodes turning negative in 2023–24; long-run structure (~2–4 vol points, ~73% hit rate) intact but thinner. The anchor's research question is explicitly *sizing the current premium*, not confirming existence.
**Adversarial:** most likely spurious result = a 2007–2026 backtest whose Sharpe is manufactured by the 2012–2019 short-vol golden age; mandate regime-sliced reporting (per classifier state) in the harness output so the aggregate cannot hide it. Falsifier unchanged.
**Next:** Wave 1, first in queue. Everything in F1/F3/F6 is interpreted relative to this anchor.

### OPT-016 — Iron Condor, Delta-Managed — **GO (Wave 1)**
**Data:** D2 only — the cheapest data footprint on the slate. **Cost:** 4-leg index combo monthly → drag ~4–8% of collected premium; **PASS.**
**Prior update:** unchanged; wings buy the tail the naked sibling sells — the registered comparison vs OPT-015 at matched vega is the actual content.
**Adversarial:** the touch-exit rule converts gap days into realized max-loss clusters; a backtest at EOD granularity will *understate* touch frequency (intraday touches invisible) — recorded as a known, directionally-conservative-for-losses bias that must be stated with results, and a legitimate future use of Rung-2 data if the candidate survives.
**Next:** Wave 1.

### OPT-017 — Iron Butterfly — **DEFER (parent: OPT-016)**
Near-duplicate (½ trial, registered in slate). Same data/cost profile (PASS). Runs only if OPT-016 clears; its marginal question is profit-zone width vs touch frequency, answerable then.

### OPT-018 — IV-Rank-Gated Short Strangle — **GO (Wave 2, ladder step 2)**
**Data:** D2+D5; IV-rank history computable from ORATS 2007+. **Cost:** PASS (2 legs; undefined risk raises margin, not friction).
**Prior update:** unchanged (B+): the IVR>50 gate is practitioner lore with thin rigorous support — high information value either way.
**Adversarial:** conditioning trap fully live: IVR>50 states cluster where vol keeps rising; the *diagnostic* (state-transition matrix of IVR regimes, no P&L) runs at Wave 0 on ORATS and frames the ladder result.
**Next:** Wave 2, strictly after OPT-015 (the unconditional baseline the gate must beat — pre-registered ablation).

### OPT-019 — Forecast-Throttled Short Vol (HAR-RV vs IV) — **GO (Wave 1, flagship 1m candidate)**
**Data:** D2+D5 (ORATS IV) + D1-intensive: HAR-RV(1,5,22) on Yang-Zhang estimates from own 1m bars, 2016+ (own-bar history is the binding window for the throttle input; ORATS extends option marks further back than the signal — harness must align windows honestly rather than backfilling the estimator with daily-bar approximations).
**Cost:** as OPT-015 → PASS.
**Prior update:** promoted in salience: with VRP compressed, an estimator that sizes exposure to the *current* premium is exactly the right research object; this is the candidate where the 1m asset's edge is largest.
**Adversarial:** frozen HAR spec is a feature (no tuning) and a risk (spec drift); the falsifier (throttled ≤ unthrottled) is clean because OPT-015 is in the same wave. Second spurious path: the throttle could just be a VIX-level proxy — mandate reporting of throttle-vs-VIX correlation with results.
**Next:** Wave 1, jointly with OPT-015 (ladder steps 1 and 3 bracket step 2 in Wave 2).

### OPT-020 — Single-Name VRP Basket — **CONDITIONAL (parent OPT-015 + earnings-date QA; Wave 3)**
**Data:** D2+D5 via ORATS + D6 approximate earnings dates (Rung 3). The QA condition: exclusion windows must be validated against a second source on a sample; a single mis-dated print inside an "excluded" window contaminates the ambient-premium claim.
**Cost:** 10 names × wider single-name spreads: drag ~8–15% of premium → **MARGINAL**; the registered comparison (basket VRP net of spreads vs index VRP) is the kill switch.
**Prior update:** unchanged (B); single-name VRP > index VRP historically, but the spread differential eats much of it at retail.
**Adversarial:** earnings-date noise is a *leakage* mechanism, not just noise — dates sourced today may be corrected relative to what was known then. Treat exclusion dating as an integrity item, mirroring the ALFRED point-in-time discipline from the macro stack.
**Next:** Wave 3, only if OPT-015 shows a live premium worth extending.

## F2 — Event-Vol Lifecycle

### OPT-021 — Earnings Vol-Crush Short Iron Fly — **CONDITIONAL (registered spec amendment v1.1; Wave 2)**
**Data:** D2 via ORATS + D6 approximate dates (Rung 3). **Registered amendment, made now before any data contact:** exit moves from T+1 *open* to T+1 *near-close snapshot* — the research pass established the open mark as structurally unreliable at EOD-data rungs. Consequence honestly stated: v1.1 carries day-1 post-earnings drift exposure the v1.0 spec did not; it is a different (researchable) strategy, not a proxy for the original. The T+1-open version waits for Rung-2 quote data if ever.
**Cost:** 4 legs × 2 transits, single-name, event-window stressed convention → 10–25% of collected premium. **MARGINAL** — survivable because the premium is large, but the sensitivity run is mandatory.
**Prior update:** premium persists but is name-dependent and crowded (beat-the-implied-move rates ranging ~25–63% across names). **Integrity rule:** the universe filter must be *ex-ante* (liquidity, market cap) — selecting names by historical beat-rate imports the outcome into the universe and is prohibited.
**Adversarial:** the published beat-rate statistics are themselves in-sample folklore; the mechanism's persistence argument (unhedgeable single-name tail) is sound, but the compensation may now be fully priced. Falsifier unchanged.
**Next:** Wave 2 under v1.1.

### OPT-022 — Pre-Earnings IV Run-Up (long) — **DROP**
Slate prior C ("deliberate dead-trial to bound the family"); research confirmed: run-up exists, capture net of theta and single-name spreads unreliable. Bounding the family is not worth a draw from the finite sample. Removed; ledger status = emitted-not-tested.

### OPT-023 — PEAD Verticals — **NO-GO**
The screen's one clean data kill. Requires true point-in-time consensus/surprise — Bloomberg/Zacks/Estimize territory, effectively unobtainable at retail — AND the research pass found simple large-cap post-earnings-announcement drift essentially dead post-2006 (revivals require ML/text machinery outside this slate's scope). Both the data screen and the prior fail independently. Removed.

### OPT-024 — Post-Crush Overshoot Short Strangle — **CONDITIONAL (parents OPT-020 ∧ OPT-021; Wave 4)**
Feasible on the same data as its parents; cost MARGINAL (single-name, but non-event-window widths). Its registered falsifier — post-print VRP ≤ ambient single-name VRP — makes it strictly interpretable only after OPT-020 exists as the baseline. Shelved to Wave 4.

### OPT-025 — Macro-Event Index Straddle (CPI/FOMC) — **CONDITIONAL (registered amendment; Wave 4)**
**Amendment:** exit T+0 near-close snapshot (CPI 08:30 and FOMC 14:00 both resolve intraday; near-close exit is well-defined at Rung 1). Macro calendar free. Cost: index 2-leg → PASS.
**Prior update:** demoted toward null-anchor — research reiterates the standard finding (implied event moves ≥ realized on average). The only interesting content is the regime-conditional cut, and slicing results by classifier state post hoc is reporting, not a new trial.
**Next:** Wave 4, low priority; expected to confirm the null.

### OPT-026 — Sector-Sympathy IV Fade — **CONDITIONAL (Wave 4, research-to-disprove)**
Feasible (ORATS + Rung-3 dates; cost MARGINAL on single-name spreads). Prior demoted: the falsifier is *currently live* — 2023–26 AI-capex tapes show peer moves on leader prints are frequently real, systematic information transfer, not sympathy overpricing. Retained at Wave 4 explicitly as a research-to-disprove candidate; a confirming result would be surprising and therefore checked hardest.

## F3 — Skew / Tail Premium

### OPT-027 — Steep-Skew Put-Spread Harvest — **GO (Wave 1)**
**Data:** D2+D5; 25Δ skew history computable from ORATS 2007+ (smoothed surface is exactly the right input for a percentile signal). **Cost:** 2-leg index monthly → PASS.
**Prior update:** held at A−; the structural-hedging-demand persistence argument survives the research pass, and skew premium is among the less-decayed premia.
**Adversarial:** identical conditioning trap to OPT-018 — steep skew is a regime label, not automatically a mispricing; the Wave-0 diagnostic (state persistence of skew percentiles; realized crash frequency conditional on steepness) frames the result before P&L exists. Falsifier unchanged (conditional ≤ unconditional premium).
**Next:** Wave 1.

### OPT-028 — Skew-Extreme Risk-Reversal Relative Value — **GO (Wave 3)**
**Data:** D2+D5; daily delta hedge from own bars (hedge-ledger accounting per §4.3). **Cost:** 2 legs + daily hedges → PASS.
**Prior update:** held at B. The delta-hedged construction is what makes this a *skew* trade rather than a directional one — the harness must verify realized residual delta stays small or the result is mislabeled.
**Adversarial:** skew extremes carry information (steepness sometimes correctly prices imminent stress); the falsifier — hedged reversion P&L ≤ 0 net of hedge slippage — is honest because slippage is modeled from own-bar hedge fills, not assumed.
**Next:** Wave 3 (needs the hedge-ledger harness module built for OPT-015/019 first — sequencing, not doubt).

### OPT-029 — Jade Lizard — **DEFER (parents OPT-002 ∧ OPT-027; starvation diagnostic first)**
Registered ½-trial duplicate. Data/cost fine. The free diagnostic — frequency with which the credit>width entry constraint binds on ORATS history — runs at Wave 0; if the structure starves (<6 qualifying entries/yr average), it drops without a trial. Otherwise it queues only behind both parents clearing.

### OPT-030 — Crash-Zone Broken-Wing Put Butterfly — **GO (Wave 3, deep-OTM sensitivity mandatory)**
**Data:** D2+D5 — and this is the candidate where the ORATS-over-ThetaData greeks decision (§2 rule 4) matters most: the signal *is* skew curvature at low deltas, where ThetaData's own docs concede rising IV error. ORATS smoothed low-delta IVs are the registered source.
**Cost:** 4 OTM legs → 10–25% of structure value round trip; **MARGINAL.** The zero-cost entry constraint partially self-insures (no debit at risk), but exit friction in stressed tapes is the real bill.
**Prior update:** held at B; the 2022 grind-bear falsifier is live and testable via the same drawdown-shape census as OPT-013 (shared Wave-0 diagnostic — one measurement serves both).
**Adversarial:** curvature "richness" may be a vendor-smoothing artifact at exactly the strikes in question — cross-check curvature signals against raw NBBO quotes on a sample before trusting the surface. Integrity item, not a trial.
**Next:** Wave 3.

### OPT-031 — Inverted-Call-Skew Fade — **CONDITIONAL (Wave 4 shelf)**
**Data:** feasible (ORATS surface flags inversions). **Cost:** the problem — squeeze-name spreads are the widest on the slate; modeled drag frequently exceeds the 0.15Δ spread's premium: **FAIL-leaning MARGINAL.**
**Prior update:** lottery-overpricing mechanism is documented (OTM single-name calls have dreadful average returns), implementation hostile — exactly the drafting-time read.
**Adversarial:** the entry regime *selects for* squeeze candidates; defined risk caps but does not remove the adverse selection.
**Condition:** shelved until live IBKR fill data (from other strategies) establishes achievable spreads in this name class; the cost model, not the mechanism, is the blocker.

## F4 — Term-Structure Carry

### OPT-032 — Contango Calendar Carry — **GO (Wave 1)**
**Data:** D2+D5 across two expiries — ORATS multi-expiry chains 2007+ are precisely built for this. **Cost:** 4 contract-transits/cycle, index widths → PASS.
**Prior update:** held at B+; term-premium-as-VRP-analog survives the research pass.
**Adversarial:** long back-month vega means a vol crush marks the "carry" position down while the carry accrues — the falsifier (slope-conditional carry ≤ 0 after vega mark-to-market) requires the harness to decompose P&L into theta-differential vs vega components, a build item worth doing once for all of F4.
**Next:** Wave 1.

### OPT-033 — Backwardation Reverse Calendar — **CONDITIONAL (episode-count diagnostic; Wave 3)**
**Data:** feasible as OPT-032. **Cost:** PASS mechanically; margin-hostile (short back-month) noted for live, irrelevant for research.
**Prior update:** held at B−.
**Adversarial:** the registered sample-starvation falsifier is the whole question — inversion episodes are few.
**Condition (Wave-0 diagnostic on ORATS):** count non-overlapping M2/M1 < 0.97 episodes 2007–2026. Pre-registered gate: ≥ 12 independent episodes → Wave 3 backtest; < 12 → route to **forward paper validation** per the generator's saturation rule (blind candidates are look-ahead-free forward; the scarce historical sample is not spent on an unestimable trial).

### OPT-034 — Term-Slope Sign-Switched Program — **DEFER (parents OPT-032 ∧ OPT-033)**
The registered ablation parent: only meaningful once both episodic legs have verdicts. If OPT-033 routes to forward paper, this defers with it.

### OPT-035 — Double-Calendar Range Harvest — **DEFER (parent OPT-032)**
½-trial duplicate; 8 contract-transits/cycle makes it strictly cost-worse than its parent, so it must earn its test by the parent clearing first. The registered falsifier (≤ parent at matched vega) is then a one-comparison question.

## F6 — Range / Mean-Reversion Theta

### OPT-036 — Range-Gated Iron Condor (Yang-Zhang gate) — **GO (Wave 2, ablation vs OPT-016)**
**Data:** D2+D5 + D1-intensive — flagship-class use of the 1m asset (10-day Yang-Zhang from own bars vs ORATS 30-day ATM IV). Same window-alignment honesty rule as OPT-019: the gate exists only where own 1m history exists (2016+).
**Cost:** as OPT-016 → PASS.
**Prior update:** promoted slightly within the family — of all the F1/F6 conditioning schemes, a measured RV/IV ratio is the least folklore-like.
**Adversarial:** low RV/IV can mean IV correctly prices a known upcoming event, not seller overpricing; consider (registered now as part of the spec, not a later tweak): the gate reads the ratio only on days ≥ 5 sessions from scheduled CPI/FOMC — calendar is free, and this closes the most obvious spurious channel.
**Next:** Wave 2, interpreted strictly against OPT-016's Wave-1 baseline.

### OPT-037 — Directional-Tilt Broken-Wing Butterfly — **DEFER (parent OPT-016/017 + starvation diagnostic)**
Feasible/PASS on data and cost; the no-cost-entry constraint gets the same Wave-0 starvation census as OPT-029. Queues only behind a surviving symmetric-fly baseline — the tilt is a one-bit modification whose test is cheap then and uninterpretable before.

### OPT-038 — Double Diagonal Income — **GO (Wave 3)**
**Data:** D2+D5, multi-expiry (ORATS). **Cost:** 4 legs, staggered maintenance → PASS but the ops-fiddliest F6 entry; research adds nothing hostile.
**Prior update:** held at B−; the registered falsifier (≤ OPT-016 at matched risk) plus the F4 vega-decomposition build make this a cheap incremental test *after* Waves 1–2 establish the condor and calendar baselines it interpolates between.
**Adversarial:** back-leg vega crush, as OPT-032; nothing new.
**Next:** Wave 3.

### OPT-039 — Band-Recentered Short Straddle — **GO (Wave 3)**
**Data:** D2 + daily-close banding from own bars (deliberately not intraday — stays at Rung 1). **Cost:** recenters add transits; still PASS at index widths.
**Prior update:** held at B−. The capped single recenter is the discipline separating this from martingale adjustment folklore; the research pass's practitioner-lore warnings reinforce keeping it capped.
**Adversarial:** in trending tapes recentering realizes losses mechanically; the registered falsifier (≤ OPT-015 within SIDEWAYS-classified periods) is regime-sliced by construction — the harness's regime-attribution output (built for OPT-015) serves it directly.
**Next:** Wave 3.

## F9 — Flow & Calendar Seasonality

### OPT-040 — Weekend Theta Capture — **CONDITIONAL (Wave-0 diagnostic; expected DROP)**
**Data:** the Friday 15:45 entry is Rung-1-compatible (near-close snapshot); the Monday 09:45 exit is not — open-adjacent marks need Rung-2 quote data, or a registered amendment to Monday near-close (which changes the exposure to include Monday's session and dilutes the thesis).
**Cost:** weekly 2-leg round trips → drag material relative to a small weekend premium; **MARGINAL.**
**Prior update:** demoted — research indicates Friday IV generally pre-discounts the weekend (calendar-day vs trading-day theta conventions), i.e., the "free" decay is likely already priced.
**Adversarial:** the trade may be structurally equivalent to selling weekend gap risk at fair value — negative selection on geopolitical Sundays.
**Condition (free/cheap, Wave 0):** (a) weekend realized-variance share from own 1m bars (already free); (b) Friday-vs-Thursday term-adjusted ATM IV census on ORATS. Pre-registered gate: proceed only if measured Friday IV discount < 50% of the calendar-day theta differential. Expected outcome per research: fails → DROP with ~zero spend.

### OPT-041 — 0DTE Post-Opening-Range Iron Condor — **CONDITIONAL (Rung-2 purchase + strict cost bar; Wave 3)**
**Data:** D3/D4 REQUIRED. SPY 0DTE quotes are continuous, so quote bars are *valid* here — the data objection is cost/history (ThetaData Value 2020+ ≈ 6.5 y; Databento 2013+ alternative), not integrity.
**Cost:** the worst profile on the slate: 8 contract-transactions/day → fees ~5–7% of a one-lot credit *before* spread; total modeled drag ~10–15% of daily credit, every day. **MARGINAL-to-FAIL.**
**Prior update:** demoted — 0DTE is now ~24% of all US options volume (2025) and ~47% of SPX volume; systematic short-premium studies show thin-to-negative net edge with severe left-tail events (Aug 2024). This is a **research-to-disprove** candidate: the interesting result is a clean measurement of *how dead* intraday VRP is after costs, on 6.5 years that must also clear a fat short-history significance penalty (§7).
**Adversarial:** any positive result on 2020–2026 is suspect of being three good years and no 2018-style vol event in-window.
**Condition:** Rung-2 purchase happens only if Wave-1/2 EOD results justify continued short-vol research at all; this candidate never justifies the purchase alone.

### OPT-042 — 0DTE First-Hour-Trend Debit Spread — **CONDITIONAL (Wave-0 pre-test; Wave 4)**
Same data/cost regime as OPT-041 (2 legs, so half the fee stack; still hostile). **Condition:** the free underlying pre-test (first-hour → rest-of-day continuation effect size on own 1m bars, 2016+, with post-2023 sub-sample reported separately) gates it exactly as OPT-014; the two share one diagnostic. Expected per research: post-publication decay shows up in the post-2023 slice → DROP.

### OPT-043 — Expiration-Week Pin/Charm Short Straddle — **CONDITIONAL (OI acquisition + pre-test; Wave 3)**
**Data:** D2 + D6 strike-level OI history — obtainable (ThetaData Value OI endpoint or ORATS), so the drafting-time acquisition worry resolves to a small purchase. **Point-in-time discipline:** OI is reported next-morning; the strike selection must use T−1 OI, never same-day.
**Cost:** monthly, index/tier-1 → PASS.
**Prior update:** narrowed, not killed — pinning remains detectable at *monthly* OpEx in high-OI names; weekly-expiry pinning diluted. The registered spec (monthly OpEx only) already matches the surviving effect.
**Adversarial:** the OI→dealer-positioning inference is the weak link (OI is signed-ambiguous); the registered pre-test — max-OI-strike distance behavior expiry week vs control weeks, a pure distance measurement with no P&L — runs first and is the cheapest mechanism check on the slate.
**Next:** pre-test at Wave 0 (post-ORATS/OI pull) → Wave 3 if distance behavior confirms.

## F8 — Relative Value / Dispersion

### OPT-044 — Dispersion-Lite (SPY vs top-6 components) — **GO (Wave 2, cost-sensitivity mandatory)**
**Data:** D2+D5 across 7 chains — ORATS covers all in one dataset; weekly hedges from own bars.
**Cost:** 14 legs/cycle: index legs cheap, 6 single-name tier-1 legs at $0.05–0.15 widths → modeled drag ~10–20% of the premium capture; **MARGINAL-PASS.**
**Prior update:** promoted within tier — the implied-vs-realized correlation gap (~6.9 points long-run; similar recent estimates) is among the *least*-decayed premia found, and 2024–25 dispersion was strongly profitable. Counterweighted by the tail: correlation → 1 on crash days (Apr 2025: 494/500 names down... up — a +9.5% index day with near-total co-movement — cost dispersion books dearly).
**Adversarial:** "dispersion" with 6 names is substantially a bet on 6 idiosyncratic vol processes plus index concentration itself; report the decomposition (index-leg vs component-leg P&L) so the label cannot launder the exposure.
**Next:** Wave 2.

### OPT-045 — Same-Sector IV-Percentile Pairs — **CONDITIONAL (Wave 4)**
Feasible (ORATS + Rung-3 earnings exclusion, same QA condition as OPT-020); cost MARGINAL (two names' spreads + double hedging). Prior held at C+: divergences are frequently justified idiosyncratic information; heavy operational load for a modest premium. Shelved behind OPT-044 — if the clean index-vs-component version of the correlation premium fails, the noisier pairwise version is not the rescue.

### OPT-046 — Index-vs-Single-Name VRP Allocation Switch — **DEFER (parents OPT-015 ∧ OPT-020)**
Registered ½-trial meta-strategy; mechanically dies or lives with its parents. The hysteresis-banded switch is a portfolio-construction question that only exists once both VRP estimates exist. No independent screen content.

## F10 — Crisis Convexity / Long Vol

### OPT-047 — Rolling Far-OTM Put Ladder (tail program) — **GO (Wave 1, infrastructure criterion)**
**Data:** D2+D5 with the deep-OTM sourcing rule (§2 rule 4) binding: 0.05Δ marks come from ORATS smoothed low-delta IVs; sample cross-checks against raw NBBO before trusting the surface (integrity item shared with OPT-030).
**Cost:** deep-OTM widths are proportionally wide, but the modeled friction (~2.5–4% of premium per rung-roll) inflates the *deliberate* 40 bps/month carry budget by only a few percent relative → **PASS** (cost is small relative to a budgeted cost).
**Prior update:** unchanged — and the compressed-VRP finding cuts *for* this candidate: cheaper insurance regime.
**Adversarial:** the registered evaluation criterion (portfolio Conditional Value-at-Risk improvement per bp of drag, jointly with the book — never standalone Sharpe) is the only honest frame; a standalone backtest of a tail hedge is a machine for concluding "insurance loses money."
**Next:** Wave 1 — but its harness output is a *joint* simulation with the Wave-1 short-vol anchors, which is precisely the book-level object that matters.

### OPT-048 — Vol-Compression Breakout Long Straddle — **GO (Wave 2)**
**Data:** D2+D5 + D1-intensive (compression percentile from own 1m Yang-Zhang — flagship use #3). **Cost:** 2 legs, episodic → PASS.
**Prior update:** held at B; the sharpened, *measurable* pre-condition from slate v1 stands: if IV in compressed states already sits at its own percentile floor, sellers are not extrapolating and the thesis dies before P&L. That measurement is a Wave-0 diagnostic on ORATS × own-bar percentiles.
**Adversarial:** 2017-style persistence bleeds the position; the exit rules cap it, but a long calm regime is the falsifier's natural habitat — regime-sliced reporting again mandatory.
**Next:** diagnostic then Wave 2.

### OPT-049 — Gamma Scalping when Forecast RV > IV — **CONDITIONAL (episode-count diagnostic; Wave 2)**
**Data:** the single best structural fit between the owned asset and a candidate: hedge P&L accrues on own 1m bars via the hedge ledger; option marks needed only at entry/exit/terminal (Rung 1 suffices). Slate v1's subtle feasibility point survives contact with the vendor evidence.
**Cost:** share hedges cheap; hedge frequency spikes exactly when underlying spreads widen — the hedge-ledger must model a stressed underlying-slippage tier (bps schedule registered in the harness, from own Alpaca/IBKR execution data).
**Prior update:** held at B.
**Adversarial:** negative-VRP states are rare and crash-clustered; the diagnostic — census of HAR-RV-forecast > ORATS-IV episodes over the aligned window — gates it exactly as OPT-033: **≥ 15 non-overlapping episodes → Wave 2; fewer → forward-paper route** per the saturation rule.
**Next:** diagnostic (shares OPT-019's estimator machinery — one build, two candidates).

### OPT-050 — Regime-Transition Long Vol — **GO (Wave 1, cheapest build on the slate)**
**Data:** D2+D5 + internal classifier history. The free pre-test from slate v1 upgrades with Rung 1: entry-day IV percentile at historical classifier transitions (ORATS IV × classifier log) — if transitions already land at IV percentile > 80 on average, the classifier lags the vol and the candidate dies pre-trial.
**Cost:** episodic 2-leg index → PASS.
**Prior update:** unchanged (B) — and it reuses live Homeguard machinery, so marginal build cost is near zero.
**Adversarial:** the candidate is really a test of the classifier's *lead time*; a failure is diagnostic information about RAMP's regime layer even if no strategy survives — unusual in that both outcomes pay.
**Next:** pre-test at Wave 0 → Wave 1.

---

# 6. Verdict Summary

**Tally: 21 GO · 18 CONDITIONAL · 7 DEFER · 3 DROP · 1 NO-GO = 50.**

| Verdict | Candidates |
|---|---|
| **GO (21)** | 001, 002, 005, 006, 011, 013, 015, 016, 018, 019, 027, 028, 030, 032, 036, 038, 039, 044, 047, 048, 050 |
| **CONDITIONAL (18)** | 003 (financing diag) · 007/008/009 (PR6 gate) · 014/042 (underlying pre-test → Rung-2) · 020 (parent 015 + earnings QA) · 021 (v1.1 amendment) · 024 (parents 020∧021) · 025 (amendment; null-anchor) · 026 (research-to-disprove) · 031 (live-fill cost evidence) · 033 (episode count ≥ 12) · 040 (Friday-IV diag; expected DROP) · 041 (Rung-2 + strict cost bar) · 043 (OI pull + distance pre-test) · 045 (behind 044) · 049 (episode count ≥ 15) |
| **DEFER (7)** | 004 (001∧002) · 017 (016) · 029 (002∧027 + starvation diag) · 034 (032∧033) · 035 (032) · 037 (016/017 + starvation diag) · 046 (015∧020) |
| **DROP (3)** | 010 (no tail-distribution support) · 012 (dominated duplicate of 011) · 022 (theta > run-up; confirmed weak) |
| **NO-GO (1)** | 023 (PIT consensus unobtainable at retail ∧ large-cap PEAD evidenced dead) |

**By family:** F7 4 GO / 1 COND / 1 DEFER · F5 3 COND-gated + 2 GO + 2 DROP + 1 COND · F1 4 GO / 1 COND / 1 DEFER · F2 1 COND(amended) / 3 COND-shelf / 1 DROP / 1 NO-GO · F3 3 GO / 1 COND / 1 DEFER · F4 1 GO / 1 COND / 2 DEFER · F6 3 GO / 1 DEFER · F9 4 COND · F8 1 GO / 1 COND / 1 DEFER · F10 3 GO / 1 COND. The two families the screen genuinely gutted are F2 (event vol — marks + point-in-time integrity) and F9 (flow/seasonality — intraday data economics + decay), which is exactly where the research evidence was most hostile.

## 6.1 Validation queue (waves)

- **Wave 0 — free/near-free, zero trials consumed:** underlying-only pre-tests (gap continuation 014/042; weekend variance 040a; drawdown-shape census 013/030; classifier-transition RV timing 050a) run **today** on owned data. Post-ORATS diagnostics: financing ratio (003), post-pullback put-VRP (005), entry-IV census (011), IVR state matrix (018), skew-state census (027), starvation censuses (029/037), inversion episodes (033), Friday-IV discount (040b), pin-distance behavior (043, after OI pull), compressed-state IV floor (048), negative-VRP episodes (049), transition IV percentile (050b).
- **Wave 1 (9 trials, post-$399):** 015, 019, 016, 001, 002, 027, 032, 047, 050. The anchors: index VRP ladder ends, overwrite carry, skew harvest, term carry, tail infrastructure, classifier-native long vol.
- **Wave 2 (≤10 trials, conditional on Wave-1 signal + diagnostics):** 018, 036 (ablations vs anchors), 005, 006, 013, 021-v1.1, 044, 048, 049†, 003† († = diagnostic-gated).
- **Wave 3 (≤12):** 011, 020, 028, 030, 033†, 038, 039, 041 (only if Rung-2 purchased on independent grounds), 043†, plus 007/008/009 iff the PR6 gate opens.
- **Wave 4 / shelf (7):** 014, 024, 025, 026, 031, 042, 045 — each with a named unlock; none justifies spend on its own.

**Data purchases:** Rung 1 (ORATS ~$399) — buy now; it gates everything. Rung 2 (quote bars) — buy only if Waves 0–2 leave live intraday questions; never for a single shelf candidate. Rung 3 (earnings API) — small spend when 020/021 reach the front. PIT consensus — never at retail.

## 6.2 Harness build items (consolidated, for CC)

Dividend-driven early-assignment + pin module (all short-call/put structures) · near-close mark convention (§2 rule 2) aligned to the ORATS ~15:46 snapshot · two-tier cost model (normal combo vs stressed single-leg-grade) with ±50% sensitivity switch · theta/vega P&L decomposition (F4, reused by 038) · hedge-ledger accounting with stressed underlying-slippage schedule (015/019/028/044/049) · regime-sliced attribution in standard output (015/039/048 falsifiers require it) · ledger integration: `emitted / tested / deferred / dropped` status per candidate, n_trials counted on test execution.

# 7. Multiple-Testing Ledger & DSR Implications

**Emitted vs tested.** Slate v1 appended +50 *emitted* candidates. The Deflated Sharpe Ratio's N counts trials **evaluated against the overlapping data window** — generated-but-never-tested candidates are not draws from the sample. Wave selection here is prior/feasibility-based and registered *before* any test, so it is not results-conditioned selection. DEFER-on-parent rules are sequential testing under pre-registered decision rules: they economize the budget legitimately, provided (a) the rules are fixed now (they are, above) and (b) every deferred candidate that *does* run increments N normally.

**Null best-of-N arithmetic** (Bailey–López de Prado expected-maximum; σ_SR ≈ 1/√years): on ORATS' 19.5-year window, σ_SR ≈ 0.23, so the expected best in-sample annualized Sharpe under zero edge is ≈ **0.34 after Wave 1 (N=9)**, ≈ **0.42 after Wave 2 (N≈19)**, ≈ **0.47–0.49 through Waves 3–4 (N≈31–38)** — before adding the pre-existing lifetime count. On the 6.5-year intraday window σ_SR ≈ 0.39: a mere **N=5 intraday trials already put the null max at ≈ 0.47** — one small intraday wave costs as much significance as ~30 EOD trials. This is the quantitative form of "intraday candidates rank last."

**Lifetime reconciliation (CC task):** the true N adds the Homeguard ledger's prior history (RAMP Phase-3/4 variants et al.) against overlapping windows — the equities-daily trials overlap the 2015+ portion of the options window, and the honest DSR treatment counts shared-window trials, not calendar-disjoint ones. Illustrative: lifetime N=150 on the long window pushes the null max to ≈ 0.60. The exact count comes from the repo ledger, not from memory. This document's delta: **+50 emitted; ≤ 9 tested at Wave 1.**

# 8. The Reiterate Decision

**No regeneration of the slate is required.** The feasibility screen was the requested kill test, and the slate survives it: only one candidate is data-blocked (023), three die on priors, and the modal outcome is "researchable after a $399 purchase." The generation step did its job — the constraint was never the 1-minute underlying data, and the screen has converted the vague "worth exploring with 1m OHLCV?" question into a priced purchase ladder plus a wave protocol.

**Registered triggers for future reiteration** (all mean *broadening the hypothesis space*, never re-tuning against results):
1. **Wave-1 anchor failure:** if OPT-015/019 establish that index VRP is fully dead net of costs in the current regime, the F1/F3/F6 short-vol block loses its economic foundation — reiterate into families this slate under-weights (e.g., financing/rates-adjacent structures, cross-asset vol) rather than generating more short-vol variants.
2. **PR6 gate failure:** F5's long trio drops silently (already wired above) — no reiteration needed, the budget was never spent.
3. **Zero survivors after CPCV/PBO/DSR validation:** a valid outcome, stated in advance. Reiteration then means new mechanism families or new data (forward paper for the saturation-routed candidates), and STOP remains Shuyang's call, not the analysis's.

# 9. Honesty Block & Caveats

- Every verdict above is a **feasibility-and-prior** judgment. No P&L exists; nothing here predicts performance; "GO" means *worth one draw from the finite sample*, nothing more.
- The **cost model is provisional** (vendor-cited conventions + width assumptions, ±50% sensitivity mandatory) and must be replaced with own IBKR fill statistics once any strategy trades live. OPT-031's shelf status is explicitly a cost-model-confidence artifact.
- **Vendor facts decay:** ORATS/ThetaData tiering and pricing were verified against vendor documentation in the research pass but partly cross-referenced through secondary comparisons — re-verify at checkout before treating $399/$40/$80 as fixed.
- **Decay evidence quality is mixed:** correlation-premium, BXM/PUT, and 0DTE-share figures trace to named studies/index providers; VRP-compression and weekend-effect readings lean on practitioner sources. Where evidence was thin (040, 022, 026) the verdicts already discount it in the conservative direction.
- **Amended specs (021-v1.1, 025) are different strategies** than their v1.0 forms — registered as amendments *before any data contact*, logged as such in the ledger; the v1.0 forms are retired, not silently overwritten.
- The +50 ledger delta, wave rules, diagnostic gates, and amendment log in this document are the pre-registration of record for this slate. Deviations require a logged amendment before the affected test runs.

*End of feasibility screen v1. Next artifact in the chain: Wave-0 diagnostic spec (runnable today on owned data) and the ORATS ingestion spec for CC.*
