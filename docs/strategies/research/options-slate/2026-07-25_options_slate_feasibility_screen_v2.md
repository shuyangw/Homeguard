# Options Slate Feasibility Screen — 50 Candidates vs the 1-Minute OHLCV Constraint (v2)

**Date:** 2026-07-24 · **Amended:** 2026-07-25 per **Amendment A1** (data-inventory correction). v1 is retired; where v1 and v2 differ, v2 governs.
**Companion to:** `2026-07-25_options_strategy_slate_v1_1.md` (candidate definitions, pre-registered parameters, priors) · `2026-07-25_options_docchain_amendment_A1.md` (correction record) · `2026-07-25_options_slate_cc_handoff_spec_v2.md` (buildable spec)
**Inputs:** slate + deep research pass (methodology, 2024–2026 cost stack, anomaly-decay evidence, auxiliary-data availability, liquidity screens) + **CC direct-disk inventory of `options_combined/` (2026-07-25)** — the corrected data ground truth
**Pipeline stages executed here:** stage 5 (data screen), stage 6 (cost-viability screen), stage 7 (diversity economization), stage 8 (adversarial critique), plus verdicts and a validation-wave protocol
**Status:** Still blind. No candidate has touched a backtest return. Every number below is either a vendor/market/disk fact or pre-registered analytic arithmetic. Verdict changes vs v1 are driven exclusively by data-availability facts, which is prior-free information — not results-conditioning.

---

## TL;DR

- **The v1 framing question ("what is the minimum options-side purchase?") is dissolved: the options-side data is already owned.** `options_combined/` holds ~233 GB of 1-minute options data — 31 underlyings, 2012-06 → 2026-02, per-minute OHLCV + bid/ask close + implied volatility (IV) + delta/theta/vega, plus end-of-day (EOD)-joined gamma and open interest (OI). All 21 GO candidates are unlocked at **$0** via an EOD snapshot derived from the minute data. **ORATS near-EOD (~$399, 2007+) is demoted from gate to recommended extension**: it is the only cure for the missing-2008 caveat (the owned window contains no GFC-class regime), supplies dividend-aware smoothed surfaces, adds breadth beyond 31 roots, and provides independent cross-validation of the owned quotes/IV. Nothing blocks on it.
- **Verdict tally unchanged: 21 GO · 18 CONDITIONAL · 7 DEFER · 3 DROP (dead priors, not data) · 1 NO-GO (OPT-023 PEAD — point-in-time consensus unobtainable at retail AND the large-cap anomaly is evidenced dead).** The slate survives the screen; no reiteration of the generation step is required. What changed inside the CONDITIONAL set: purchase-conditions became **on-disk verification conditions** (the V-battery of Amendment A1 §5), and two candidates (021, plus the universe-dependent group) now carry explicit decisions rather than purchases.
- **The 1-minute *underlying* data still earns its keep in the four flagship candidates** — OPT-019, 036, 048, 049 (Yang-Zhang / Heterogeneous Autoregressive Realized Volatility (HAR-RV) estimation) — plus free signal-side pre-tests. The owned 1-minute *options* data adds three things v1 could not assume: empirical spread measurement (cost model v2), intraday touch quantification, and native support for the intraday candidates. The 0DTE/intraday candidates remain the **least** attractive — but now purely on economics (hostile cost arithmetic, most-decayed premia), no longer on data access.
- **The hard rule stands as a principle:** trade-print-derived option OHLC bars are prohibited as marks for out-of-the-money (OTM) or low-volume strikes. The owned set appears to be quote-bearing minute records (rows independent of trades) — **V1–V3 of the verification battery decide**; if OTM quote coverage fails the gate, the prohibition re-binds exactly as written and Databento CBBO-1m is the fallback.

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

## 2. Data ground truth — sources and rules (v2, per Amendment A1)

The v1 purchase ladder is superseded. The primary source is owned; purchases are extensions, not gates.

| Source | What it is | Window | Cost | Role |
|---|---|---|---|---|
| **Owned — `options_combined/`** | 1-min per-contract records: OHLCV, trade_count, vwap, **bid/ask close**, IV, delta/theta/vega, underlying_px + EOD-joined gamma/OI. 31 roots (SPX, VIX, SPY, QQQ, IWM, DIA, 6 sector/theme ETFs, 5 commodity/bond/intl ETFs, ~13 singles incl. AAPL/NVDA/TSLA/AMZN/MSFT/META/AMD/GOOGL/AVGO/MSTR/PLTR/COIN/IBIT, FB legacy) | 2012-06 → 2026-02 (~13.7 y; late listings shorter) | $0 | **Primary, pending the V-battery** (Amendment A1 §5): quote population/semantics, IV/greeks coverage, OI-join leakage, corporate actions, provenance, spread census, refresh |
| Owned — Alpaca SIP underlying 1m · CBOE VIX term structure · classifier history | — | 2016+/2010+/internal | $0 | Signals, hedges, gates, Wave-0 pre-tests |
| **ORATS near-EOD historical** | Full chains, NBBO-derived quotes, dividend-aware smoothed (SMV) greeks/IV, ~5,000 symbols | 2007+ | ~$399 one-time | **Recommended, non-blocking:** (i) 2007–2012 GFC extension; (ii) smoothed surfaces for 006/027/030/047 without building M7; (iii) breadth beyond 31 roots; (iv) cross-vendor validation of owned quotes/IV |
| Databento OPRA CBBO-1m | Consolidated minute NBBO | 2013+ | usage-based | Dormant fallback if V1–V3 fail on quote quality |
| Earnings-calendar API | Approximate dates + BMO/AMC | varies | low | Deferred until 020/021 front the queue |
| True point-in-time (PIT) consensus | — | — | institutional | Never at retail — OPT-023 stays NO-GO |

**Binding rules (v2):**
1. **Trade-bar prohibition retained verbatim** (marks from trade-print bars on OTM/low-volume strikes are forbidden). The owned set appears quote-bearing; **V1–V3 decide** whether it satisfies the rule.
2. **Marks = the derived EOD snapshot** at the registered minute **15:45:00 ET** (mid of bid/ask close; crossed/zero-bid rows excluded per V3; low-delta marks cross-checked against the smoothed surface). Raw session "close" prints prohibited.
3. **Snapshot symmetry** replaces v1's ORATS-basis truncation: signals from 1m bars truncate at the same registered minute, and daily hedges execute at it (P6 `daily_1545` — hedge time amended 15:50→15:45 per A1, a data-alignment amendment logged before any test). One clock for signals, marks, and hedges; the 14-minute lookahead trap of v1 no longer exists because we control the snapshot.
4. **Greeks caveat now applies to owned data:** the shipped IV/greeks are ThetaData-computed (Black-Scholes, dividends ignored by default, rising error deep ITM/OTM). |Δ| < 0.10 work reads the **smoothed surface** — ORATS SMV if purchased, else the M7 fitted surface — never shipped per-contract greeks. V5 gates trust by year/root.
5. **OI/gamma are EOD-joined fields:** V6 must establish which date's value lands on row-date *t* before any OI/gamma-conditioned logic runs; the T−1-known rule stands regardless.
6. **Universe floor unchanged** (≥ 1,000 contracts/day, roughly top ~100–150 names). Availability = 31 on-disk roots; breadth-dependent candidates (021, 026, 031, 045; 001/002 beyond the on-disk megacaps) carry an explicit **universe decision**: narrow (logged amendment) vs top-up download (subscription status = Phase 0.10) vs ORATS breadth.

**History depth as a statistical asset (revised):** primary window ≈ **13.7 y** → σ(annualized Sharpe) ≈ 0.27 for EOD **and** intraday candidates alike — v1's intraday-specific short-history penalty is retired. Two caveats replace it: **missing-2008** (no GFC-class regime in-window; curable at EOD granularity by the ORATS extension, permanent for intraday-native candidates) and **per-root windows** (PLTR ≈ 5.7 y, COIN ≈ 4.7 y — each candidate's DSR uses *its* window). Live edge is 2026-02; refresh to present pending Phase 0.10/V9.

---

## 3. Pre-registered cost model v1 (analytic stand-in for `cost_viability.py`)

Fill convention (restated vs mid, from the ORATS practitioner convention): single-leg orders cost ~**25% of quoted width beyond mid** per transit; 3–4-leg native combos ~**3–6% of width per leg** beyond mid (complex-order-book netting); **event/stressed windows: treat combos as single-leg-grade (15–25%)**. Fees: ~$0.42–0.72/contract one-way all-in at retail (commission $0.35–0.65 tiered/Lite + ~$0.05–0.07 regulatory: Options Clearing Corporation (OCC) $0.025, Options Regulatory Fee ~$0.023, Consolidated Audit Trail $0.0003, FINRA Trading Activity Fee $0.00329 on sells; SEC Section 31 returns at $20.60/$1M of sale proceeds from 2026-04-04 — negligible at these premia). Width assumptions (normal conditions): SPY/QQQ ATM 30–45 days-to-expiration (DTE) $0.03–0.12; SPY 0DTE ATM $0.01–0.02; SPY 0.05–0.10Δ $0.02–0.06 on $0.30–1.50 value; tier-1 single-name ATM $0.05–0.15; rank-50–100 ATM $0.10–0.30; event windows 1.5–3× normal; LEAPS $0.30–1.00. **Sensitivity ±50% mandatory.** The width table above is now **provisional**: V11 (empirical spread census on the owned quote data, by root × moneyness × DTE × regime × era) re-parameterizes it as `cost_model_v2`. Fill-fraction conventions remain assumptions until replaced by own IBKR fill statistics once live.

Two structural conclusions fall out before any per-candidate work: **(i)** monthly-cadence index structures pass by an order of magnitude under normal widths (round-trip drag ~0.2–1% of premium) — their binding cost risk is *stressed-regime* widths and hedge slippage, not steady-state friction; **(ii)** the cost screen bites, in descending severity, on: daily-cadence multi-leg (0DTE), event-window single-name multi-leg, rank-50–100 names, deep-OTM legs, and LEAPS rolls.

---

## 4. What the owned 1-minute data is actually for (v2)

Direct answer to the framing question, updated for the corrected inventory:

1. **Volatility-estimator edge (the four flagships, unchanged):** OPT-019 (HAR-RV throttle), OPT-036 (Yang-Zhang gate), OPT-048 (compression percentile), OPT-049 (hedge-band triggers + hedge ledger). Underlying 1m bars remain a genuine informational input EOD-bar competitors lack.
2. **Free signal-side pre-tests (Wave 0, unchanged):** gap-continuation and first-hour-trend effects (OPT-014/042), weekend realized-variance decomposition (OPT-040), compression-episode census (OPT-048), classifier-transition realized-vol timing (OPT-050) — all measurable today at zero cost, killing or promoting dependents before any strategy trial.
3. **Hedge-execution ledgers (unchanged):** share hedges for OPT-015/019/028/044/049 priced off own underlying data; option marks needed only at entry/exit/terminal — the derived EOD snapshot supplies them.
4. **New — what the owned 1-minute *options* data adds:** (a) the derived EOD chain snapshot itself (the D2 backbone, at $0); (b) empirical spread distributions → `cost_model_v2` (V11), materially strengthening the chain's weakest input; (c) intraday touch quantification for OPT-016's known bias; (d) native D3/D4 support for 014/041/042, pending V1–V3.
5. **What it is still NOT:** trade-print bars are not marks (rule 1); shipped deep-OTM greeks are not a surface (rule 4); and more data is not more edge — no prior was upgraded by this correction.

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
**Data:** D2 from the owned derived EOD snapshot. **PASS.**
**Cost:** as OPT-001; assignment transitions add share transactions (cheap). **PASS.**
**Prior update:** PUT-index long-run evidence intact (~10.4% annualized since 1986, lower vol than S&P). Critical economics note sharpened by the rate environment: the secured cash competes against the risk-free rate — the researchable object is *premium net of T-bill yield on the secured cash*, and the harness must account P&L that way or the result flatters.
**Adversarial:** clustering of assignments at regime breaks is the whole risk; the BEAR-suspension gate is doing all the work — test with and without is NOT permitted post hoc (one registered spec only).
**Next:** Wave 1.

### OPT-003 — Financed Collar Carry — **CONDITIONAL (diagnostic; then Wave 2)**
**Data:** D2+D5 from the owned derived EOD snapshot (two expiries, marks fine). 
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
**Data:** D2+D5 from the owned derived EOD snapshot; signal side from own bars. Feasible.
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

### OPT-014 — Opening-Range Gap Same-Day Verticals — **CONDITIONAL (Wave-0 pre-test → V1–V3)**
**Data:** D3/D4 — **satisfied by the owned minute set pending V1–V3** (quote-bearing rows on the relevant near-the-money strikes; expected to pass on SPY/QQQ). Trade-only bars remain prohibited if the checks fail. History 13.7 y; segment by SPY expiry-availability era (Friday weeklies full-history; M/W ~2016+; daily ~2022+). Intraday fills follow the registered decide-on-bar-*t*, fill-at-bar-*t+1* convention.
**Cost:** unchanged — ~2 legs × 2 transits × ~daily-when-triggered at 0–1 DTE: ~$0.05–0.07/trade modeled → demands per-trade gross ≥ ~$0.12–0.15. **MARGINAL-to-FAIL** unless the effect size is large. The economics objection never depended on the data question.
**Prior update:** demoted — intraday index momentum is documented but heavily exploited post-0DTE-boom; post-publication decay likely.
**Adversarial:** the effect may live entirely in the underlying's first minutes and be gone by a 10:00 options fill.
**Condition (free, Wave 0):** measure gap-continuation effect size on own SPY/QQQ 1m bars, 2016+, **post-2023 sub-sample reported separately**. Pre-registered gate: proceed to the options backtest only if the underlying effect ≥ 15 bps/trade net of a 2-bp underlying-slippage haircut in the post-2023 slice. Otherwise DROP with zero spend.

## F1 — Volatility Risk Premium (VRP)

### OPT-015 — Index Delta-Hedged Short Straddle — **GO (Wave 1, family anchor)**
**Data:** D2+D5 from the owned derived EOD snapshot. Signals, option marks, and the daily hedge all sit on the registered 15:45 snapshot minute (hedge time amended 15:50→15:45 per A1) — one clock, zero basis, the correctness detail v1 had to engineer around a vendor's snapshot time.
**Cost:** 2-leg index combo monthly + daily share hedges (~1–2 bps each): total drag ~1–2% of premium annually under normal widths → **PASS** by an order of magnitude. Binding cost risk is stressed-width exits, which the harness must model with the event/stressed convention.
**Prior update:** the material update of the whole screen — index VRP has compressed post-2022, with 12-month rolling VRP episodes turning negative in 2023–24; long-run structure (~2–4 vol points, ~73% hit rate) intact but thinner. The anchor's research question is explicitly *sizing the current premium*, not confirming existence.
**Adversarial:** most likely spurious result = a 2007–2026 backtest whose Sharpe is manufactured by the 2012–2019 short-vol golden age; mandate regime-sliced reporting (per classifier state) in the harness output so the aggregate cannot hide it. Falsifier unchanged.
**Next:** Wave 1, first in queue. Everything in F1/F3/F6 is interpreted relative to this anchor.

### OPT-016 — Iron Condor, Delta-Managed — **GO (Wave 1)**
**Data:** D2 only — the cheapest data footprint on the slate. **Cost:** 4-leg index combo monthly → drag ~4–8% of collected premium; **PASS.**
**Prior update:** unchanged; wings buy the tail the naked sibling sells — the registered comparison vs OPT-015 at matched vega is the actual content.
**Adversarial:** the touch-exit rule converts gap days into realized max-loss clusters; a backtest at daily-snapshot granularity will *understate* touch frequency — and this bias is now **quantifiable immediately** from the owned minute data. Report the measured touch-understatement alongside results; the registered spec still evaluates exits at the daily snapshot (the measurement is reporting, not a spec change).
**Next:** Wave 1.

### OPT-017 — Iron Butterfly — **DEFER (parent: OPT-016)**
Near-duplicate (½ trial, registered in slate). Same data/cost profile (PASS). Runs only if OPT-016 clears; its marginal question is profit-zone width vs touch frequency, answerable then.

### OPT-018 — IV-Rank-Gated Short Strangle — **GO (Wave 2, ladder step 2)**
**Data:** D2+D5; IV-rank history computable from the owned chains, 2012+ (2007+ with the ORATS extension). **Cost:** PASS (2 legs; undefined risk raises margin, not friction).
**Prior update:** unchanged (B+): the IVR>50 gate is practitioner lore with thin rigorous support — high information value either way.
**Adversarial:** conditioning trap fully live: IVR>50 states cluster where vol keeps rising; the *diagnostic* (state-transition matrix of IVR regimes, no P&L) runs at Wave 0 on ORATS and frames the ladder result.
**Next:** Wave 2, strictly after OPT-015 (the unconditional baseline the gate must beat — pre-registered ablation).

### OPT-019 — Forecast-Throttled Short Vol (HAR-RV vs IV) — **GO (Wave 1, flagship 1m candidate)**
**Data:** D2+D5 (owned IV at the snapshot, V5-gated; smoothed surface as the ATM cross-check) + D1-intensive: HAR-RV(1,5,22) on Yang-Zhang estimates from own 1m bars, 2016+ (own-bar history is the binding window for the throttle input; the 2012+ option history extends further back than the signal — harness must align windows honestly rather than backfilling the estimator with daily-bar approximations).
**Cost:** as OPT-015 → PASS.
**Prior update:** promoted in salience: with VRP compressed, an estimator that sizes exposure to the *current* premium is exactly the right research object; this is the candidate where the 1m asset's edge is largest.
**Adversarial:** frozen HAR spec is a feature (no tuning) and a risk (spec drift); the falsifier (throttled ≤ unthrottled) is clean because OPT-015 is in the same wave. Second spurious path: the throttle could just be a VIX-level proxy — mandate reporting of throttle-vs-VIX correlation with results.
**Next:** Wave 1, jointly with OPT-015 (ladder steps 1 and 3 bracket step 2 in Wave 2).

### OPT-020 — Single-Name VRP Basket — **CONDITIONAL (parent OPT-015 + earnings-date QA; Wave 3)**
**Data:** D2+D5 from the owned chains (`U_MEGA10` ≈ the on-disk singles — confirm with a PIT option-volume ranking) + D6 approximate earnings dates (Rung 3). The QA condition: exclusion windows must be validated against a second source on a sample; a single mis-dated print inside an "excluded" window contaminates the ambient-premium claim.
**Cost:** 10 names × wider single-name spreads: drag ~8–15% of premium → **MARGINAL**; the registered comparison (basket VRP net of spreads vs index VRP) is the kill switch.
**Prior update:** unchanged (B); single-name VRP > index VRP historically, but the spread differential eats much of it at retail.
**Adversarial:** earnings-date noise is a *leakage* mechanism, not just noise — dates sourced today may be corrected relative to what was known then. Treat exclusion dating as an integrity item, mirroring the ALFRED point-in-time discipline from the macro stack.
**Next:** Wave 3, only if OPT-015 shows a live premium worth extending.

## F2 — Event-Vol Lifecycle

### OPT-021 — Earnings Vol-Crush Short Iron Fly — **CONDITIONAL (exit-version decision + universe decision; Wave 2)**
**Exit-version decision (required before any earnings data is touched):** v1.1 (T+1 near-close exit) was registered solely because the T+1 *open* was unmeasurable at the assumed data rung. Owned minute quotes restore v1.0 (T+1 open — the original pure-crush hypothesis, minimal drift exposure) to feasibility, with stressed-tier open costs. **Exactly one version enters the queue. Recommendation: restore v1.0.** The unchosen version is retired in the ledger, not deleted; running both would be two trials on one mechanism.
**Universe decision:** the registered universe ("liquid names with weeklies, priced expected move ≥ 4%") exceeds the ~13 on-disk singles. Narrow to on-disk roots (logged amendment; materially fewer events per year) or top-up download (Phase 0.10 determines feasibility). The filter stays **ex-ante** (liquidity/size/expected-move threshold) — selecting names by historical beat-the-implied-move rate imports the outcome into the universe and is prohibited.
**Data:** owned chains + Rung-3 approximate earnings dates (leakage-integrity item unchanged). Requires M1.
**Cost:** 4 legs × 2 transits, single-name, stressed tier mandatory → 10–25% of collected premium. **MARGINAL**; sensitivity run required.
**Prior update:** premium persists but is name-dependent and crowded (beat rates ~25–63% across names); published beat-rate statistics are themselves in-sample folklore.
**Next:** Wave 2 once both decisions are logged.

### OPT-022 — Pre-Earnings IV Run-Up (long) — **DROP**
Slate prior C ("deliberate dead-trial to bound the family"); research confirmed: run-up exists, capture net of theta and single-name spreads unreliable. Bounding the family is not worth a draw from the finite sample. Removed; ledger status = emitted-not-tested.

### OPT-023 — PEAD Verticals — **NO-GO**
The screen's one clean data kill. Requires true point-in-time consensus/surprise — Bloomberg/Zacks/Estimize territory, effectively unobtainable at retail — AND the research pass found simple large-cap post-earnings-announcement drift essentially dead post-2006 (revivals require ML/text machinery outside this slate's scope). Both the data screen and the prior fail independently. Removed.

### OPT-024 — Post-Crush Overshoot Short Strangle — **CONDITIONAL (parents OPT-020 ∧ OPT-021; Wave 4)**
Feasible on the same data as its parents; cost MARGINAL (single-name, but non-event-window widths). Its registered falsifier — post-print VRP ≤ ambient single-name VRP — makes it strictly interpretable only after OPT-020 exists as the baseline. Shelved to Wave 4.

### OPT-025 — Macro-Event Index Straddle (CPI/FOMC) — **CONDITIONAL (amendment stands; Wave 4)**
The registered T+0 near-close exit **stands** — it matches the original 15:50-style design, and the exit mark is now directly verifiable from owned minute quotes rather than approximated. Macro calendar free; index 2-leg cost → PASS. Prior unchanged: demoted toward null-anchor (implied event moves ≥ realized on average is the standard finding); regime-conditional slicing of the result is reporting, not a new trial. Wave 4, low priority.

### OPT-026 — Sector-Sympathy IV Fade — **CONDITIONAL (Wave 4, research-to-disprove; universe decision)**
Feasible on owned chains + Rung-3 dates; cost MARGINAL on single-name spreads. **Universe note (new):** peer sets around a mega-cap reporter are thin at 31 roots (sector ETFs are present, single-name peers are not) — the universe decision applies before this leaves the shelf. Prior demoted, unchanged from v1: the falsifier is *currently live* — 2023–26 AI-capex tapes show peer moves on leader prints are frequently real information transfer, not sympathy overpricing. A confirming result would be surprising and therefore checked hardest.

## F3 — Skew / Tail Premium

### OPT-027 — Steep-Skew Put-Spread Harvest — **GO (Wave 1)**
**Data:** D2+D5; 25Δ skew history computable from the owned chains, 2012+ (2007+ with the ORATS extension). The percentile signal reads the **smoothed surface** (ORATS SMV if purchased, else M7); owned shipped IV is acceptable near-the-money pending V5. **Cost:** 2-leg index monthly → PASS.
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
**Data:** D2+D5 — the candidate where §2 rule 4 binds hardest: the signal *is* skew curvature at low deltas, exactly where shipped greeks are least trustworthy. Registered source: the smoothed surface (ORATS SMV if purchased, else M7). The D-047/030 integrity check runs against owned raw deep-OTM quotes — and upgrades to a true cross-vendor check if ORATS is bought.
**Cost:** 4 OTM legs → 10–25% of structure value round trip; **MARGINAL.** The zero-cost entry constraint partially self-insures; exit friction in stressed tapes is the real bill (now measurable in-era via V11).
**Prior update:** held at B; the 2022 grind-bear falsifier is live and testable via the shared drawdown-shape census (D-013/030).
**Adversarial:** curvature "richness" may be a smoothing artifact at exactly the strikes in question — the raw-quote cross-check is the control.
**Next:** Wave 3.

### OPT-031 — Inverted-Call-Skew Fade — **CONDITIONAL (Wave 4 shelf; universe decision)**
**Data:** mechanically feasible (owned surface flags inversions) — but squeeze names are mostly **off-disk**, so the universe decision applies on top of the original blocker. **Cost:** unchanged, the problem — squeeze-name spreads are the widest on the slate; modeled drag frequently exceeds the 0.15Δ spread's premium: **FAIL-leaning MARGINAL.**
**Prior update:** unchanged — lottery-overpricing mechanism documented, implementation hostile.
**Adversarial:** the entry regime *selects for* squeeze candidates; defined risk caps but does not remove the adverse selection.
**Condition:** shelved until live IBKR fill data establishes achievable spreads in this name class **and** the universe decision lands; the cost model, not the mechanism, remains the blocker.

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
**Condition (Wave-0 diagnostic on the owned chain history):** count non-overlapping M2/M1 < 0.97 episodes 2012–2026 (2007–2026 if the ORATS extension is purchased). Pre-registered gate: ≥ 12 independent episodes → Wave 3 backtest; < 12 → route to **forward paper validation** per the generator's saturation rule (blind candidates are look-ahead-free forward; the scarce historical sample is not spent on an unestimable trial).

### OPT-034 — Term-Slope Sign-Switched Program — **DEFER (parents OPT-032 ∧ OPT-033)**
The registered ablation parent: only meaningful once both episodic legs have verdicts. If OPT-033 routes to forward paper, this defers with it.

### OPT-035 — Double-Calendar Range Harvest — **DEFER (parent OPT-032)**
½-trial duplicate; 8 contract-transits/cycle makes it strictly cost-worse than its parent, so it must earn its test by the parent clearing first. The registered falsifier (≤ parent at matched vega) is then a one-comparison question.

## F6 — Range / Mean-Reversion Theta

### OPT-036 — Range-Gated Iron Condor (Yang-Zhang gate) — **GO (Wave 2, ablation vs OPT-016)**
**Data:** D2+D5 + D1-intensive — flagship-class use of the 1m asset (10-day Yang-Zhang from own bars vs the derived 30-day ATM IV). Same window-alignment honesty rule as OPT-019: the gate exists only where own 1m history exists (2016+).
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
**Data:** D2 + daily-close banding from own bars (deliberately not intraday — a registered design choice, retained now that it is no longer a data constraint). **Cost:** recenters add transits; still PASS at index widths.
**Prior update:** held at B−. The capped single recenter is the discipline separating this from martingale adjustment folklore; the research pass's practitioner-lore warnings reinforce keeping it capped.
**Adversarial:** in trending tapes recentering realizes losses mechanically; the registered falsifier (≤ OPT-015 within SIDEWAYS-classified periods) is regime-sliced by construction — the harness's regime-attribution output (built for OPT-015) serves it directly.
**Next:** Wave 3.

## F9 — Flow & Calendar Seasonality

### OPT-040 — Weekend Theta Capture — **CONDITIONAL (Wave-0 diagnostic; expected DROP)**
**Data:** both legs of the registered clock — Friday 15:45 entry and **Monday 09:45 exit** — are now feasible from owned minute quotes as written; v1's forced-amendment discussion (Monday near-close) is retired. Intraday exit follows the *t*/*t+1* fill convention.
**Cost:** weekly 2-leg round trips against a small weekend premium; **MARGINAL** — unchanged.
**Prior update:** demoted, unchanged — Friday IV generally pre-discounts the weekend (calendar-day vs trading-day theta conventions).
**Adversarial:** structurally equivalent to selling weekend gap risk at fair value; negative selection on geopolitical Sundays.
**Condition (free/cheap, Wave 0):** (a) weekend realized-variance share from own 1m bars; (b) Friday-vs-Thursday term-adjusted ATM IV census on the owned chains. Pre-registered gate unchanged: proceed only if the measured Friday IV discount < 50% of the calendar-day theta differential. Expected outcome per research: fails → DROP with ~zero spend.

### OPT-041 — 0DTE Post-Opening-Range Iron Condor — **CONDITIONAL (V1–V3 + strict cost bar; Wave 3)**
**Data:** D3/D4 — **on disk, 2012+ (13.7 y), pending V1–V3**; SPY quote continuity makes a pass expected here. Segment and report by expiry era (Friday weeklies full-history → M/W ~2016 → daily ~2022): the daily-expiry regime is only ~4 years even inside a 13.7-year file.
**Cost:** unchanged and still the worst profile on the slate: 8 contract-transactions/day → fees ~5–7% of a one-lot credit before spread; total modeled drag ~10–15% of daily credit, every day. **MARGINAL-to-FAIL.**
**Prior update:** unchanged — 0DTE ≈ 24% of US options volume (2025), ~47% of SPX; systematic short-premium evidence thin-to-negative net of costs with severe left tails (Aug 2024). **Research-to-disprove:** the valuable output is a clean measurement of how dead intraday VRP is after costs. The v1 purchase objection is moot; the economics objection stands verbatim.
**Adversarial (sharpened by the longer window):** the Friday-weekly history now *includes* Aug 2015, Feb 2018, and Mar 2020 — a positive aggregate that evaporates in those eras is the expected signature of a dead edge, and the era-sliced report is where it will show.
**Condition:** V1–V3 pass; runs only if Waves 0–2 leave short-vol research alive at all.

### OPT-042 — 0DTE First-Hour-Trend Debit Spread — **CONDITIONAL (Wave-0 pre-test; Wave 4)**
Same data regime as OPT-041 (on disk pending V1–V3; 2 legs, so half the fee stack; still hostile). **Condition unchanged:** the free underlying pre-test (first-hour → rest-of-day continuation on own 1m bars, 2016+, post-2023 slice separate) gates it exactly as OPT-014 — the two share one diagnostic. Expected per research: post-publication decay shows in the post-2023 slice → DROP.

### OPT-043 — Expiration-Week Pin/Charm Short Straddle — **CONDITIONAL (V6 leakage gate + pre-test; Wave 3)**
**Data:** D2 + D6 — `open_interest_eod` is **on disk**; v1's acquisition condition is retired and replaced by **V6**: establish which date's OI lands on row-date *t* before any OI-conditioned logic runs. The strategy selects strikes on **T−1-known** OI; a same-day join is a hard leak.
**Prior-art reconciliation (new):** the repo's shelved OpEx pinning strategy (2025-12) is adjacent prior art — reusable GEX/calendar machinery with 152 passing tests, and its 2025 backtest (18 trades, 11.1% win rate, −$415, on *estimated* gamma/OI) is a **prior tested trial for the lifetime ledger**. OPT-043's hypothesis (max-OI-strike short straddle, monthly OpEx) is distinct from that GEX-directional design — related family, different mechanism; both count trials.
**Cost:** monthly, index/tier-1 → PASS.
**Prior update:** narrowed, not killed — pinning remains detectable at *monthly* OpEx in high-OI names; weekly diluted. The registered monthly-only spec already matches.
**Adversarial:** the OI → dealer-positioning inference is signed-ambiguous; the D-043 distance pre-test (pure measurement, no P&L) runs first and is the cheapest mechanism check on the slate.
**Next:** V6 → D-043 → Wave 3.

## F8 — Relative Value / Dispersion

### OPT-044 — Dispersion-Lite (SPY vs top-6 components) — **GO (Wave 2, cost-sensitivity mandatory)**
**Data:** D2+D5 across 7 chains — ORATS covers all in one dataset; weekly hedges from own bars.
**Cost:** 14 legs/cycle: index legs cheap, 6 single-name tier-1 legs at $0.05–0.15 widths → modeled drag ~10–20% of the premium capture; **MARGINAL-PASS.**
**Prior update:** promoted within tier — the implied-vs-realized correlation gap (~6.9 points long-run; similar recent estimates) is among the *least*-decayed premia found, and 2024–25 dispersion was strongly profitable. Counterweighted by the tail: correlation → 1 on crash days (Apr 2025: 494/500 names down... up — a +9.5% index day with near-total co-movement — cost dispersion books dearly).
**Adversarial:** "dispersion" with 6 names is substantially a bet on 6 idiosyncratic vol processes plus index concentration itself; report the decomposition (index-leg vs component-leg P&L) so the label cannot launder the exposure.
**Next:** Wave 2.

### OPT-045 — Same-Sector IV-Percentile Pairs — **CONDITIONAL (Wave 4; universe decision)**
Feasible mechanics (owned chains + Rung-3 earnings exclusion, same QA condition as OPT-020); cost MARGINAL (two names' spreads + double hedging). **Universe note (new):** same-industry liquid pairs barely exist at 31 roots — the universe decision applies. Prior held at C+: divergences are frequently justified idiosyncratic information; heavy operational load for a modest premium. Shelved behind OPT-044 — if the clean index-vs-component version of the correlation premium fails, the noisier pairwise version is not the rescue.

### OPT-046 — Index-vs-Single-Name VRP Allocation Switch — **DEFER (parents OPT-015 ∧ OPT-020)**
Registered ½-trial meta-strategy; mechanically dies or lives with its parents. The hysteresis-banded switch is a portfolio-construction question that only exists once both VRP estimates exist. No independent screen content.

## F10 — Crisis Convexity / Long Vol

### OPT-047 — Rolling Far-OTM Put Ladder (tail program) — **GO (Wave 1, infrastructure criterion)**
**Data:** D2+D5 with §2 rule 4 binding: 0.05Δ marks come from the **smoothed surface** (ORATS SMV if purchased, else M7); shipped deep-OTM greeks are not trusted. The integrity check (shared with OPT-030) runs against owned raw deep-OTM quotes, and upgrades to cross-vendor if ORATS is bought.
**Cost:** deep-OTM widths proportionally wide, but modeled friction (~2.5–4% of premium per rung-roll) inflates the *deliberate* 40 bps/month carry budget by only a few percent → **PASS** (a small cost on a budgeted cost) — now verifiable against measured deep-OTM widths (V11).
**Prior update:** unchanged — and the compressed-VRP finding still cuts *for* this candidate: cheaper insurance regime.
**Adversarial:** unchanged — the registered evaluation criterion (portfolio Conditional Value-at-Risk improvement per bp of drag, jointly with the book, never standalone Sharpe) is the only honest frame.
**Next:** Wave 1 — joint simulation with the Wave-1 short-vol anchors.

### OPT-048 — Vol-Compression Breakout Long Straddle — **GO (Wave 2)**
**Data:** D2+D5 + D1-intensive (compression percentile from own 1m Yang-Zhang — flagship use #3). **Cost:** 2 legs, episodic → PASS.
**Prior update:** held at B; the sharpened, *measurable* pre-condition from slate v1 stands: if IV in compressed states already sits at its own percentile floor, sellers are not extrapolating and the thesis dies before P&L. That measurement is a Wave-0 diagnostic on ORATS × own-bar percentiles.
**Adversarial:** 2017-style persistence bleeds the position; the exit rules cap it, but a long calm regime is the falsifier's natural habitat — regime-sliced reporting again mandatory.
**Next:** diagnostic then Wave 2.

### OPT-049 — Gamma Scalping when Forecast RV > IV — **CONDITIONAL (episode-count diagnostic; Wave 2)**
**Data:** still the best structural fit between owned assets and a candidate: hedge P&L accrues on own 1m bars via the hedge ledger; option marks needed only at entry/exit/terminal — the **owned derived EOD snapshot suffices**. The entry condition (HAR-RV forecast > IV_30) now reads owned IV (V5-gated).
**Cost:** share hedges cheap, **but** hedge frequency spikes exactly when underlying spreads widen — the tiered slippage schedule in P6 is what keeps this honest; a flat bps assumption flatters it materially.
**Prior update:** held at B.
**Adversarial:** negative-VRP states are rare and crash-clustered; the D-049 census (non-overlapping forecast-above-IV episodes over the aligned window) gates it: **≥ 15 episodes → Wave 2; fewer → forward-paper route**, do not spend the historical sample.
**Next:** diagnostic (shares OPT-019's estimator machinery — one build, two candidates).

### OPT-050 — Regime-Transition Long Vol — **GO (Wave 1, cheapest build on the slate)**
**Data:** D2+D5 + internal classifier history. The free pre-test from slate v1 upgrades with Rung 1: entry-day IV percentile at historical classifier transitions (ORATS IV × classifier log) — if transitions already land at IV percentile > 80 on average, the classifier lags the vol and the candidate dies pre-trial.
**Cost:** episodic 2-leg index → PASS.
**Prior update:** unchanged (B) — and it reuses live Homeguard machinery, so marginal build cost is near zero.
**Adversarial:** the candidate is really a test of the classifier's *lead time*; a failure is diagnostic information about RAMP's regime layer even if no strategy survives — unusual in that both outcomes pay.
**Next:** pre-test at Wave 0 → Wave 1.

---

# 6. Verdict Summary

**Tally unchanged by A1: 21 GO · 18 CONDITIONAL · 7 DEFER · 3 DROP · 1 NO-GO = 50.** Conditions inside the CONDITIONAL set are restated below in their v2 (post-A1) form.

| Verdict | Candidates |
|---|---|
| **GO (21)** | 001, 002, 005, 006, 011, 013, 015, 016, 018, 019, 027, 028, 030, 032, 036, 038, 039, 044, 047, 048, 050 |
| **CONDITIONAL (18)** | 003 (financing diag) · 007/008/009 (PR6 gate) · 014/042 (underlying pre-test → V1–V3) · 020 (parent 015 + earnings QA) · 021 (exit-version + universe decisions) · 024 (parents 020∧021) · 025 (amendment stands; null-anchor) · 026 (research-to-disprove; universe) · 031 (live-fill cost evidence; universe) · 033 (episode count ≥ 12) · 040 (Friday-IV diag; expected DROP) · 041 (V1–V3 + strict cost bar) · 043 (V6 leakage gate + distance pre-test) · 045 (behind 044; universe) · 049 (episode count ≥ 15) |
| **DEFER (7)** | 004 (001∧002) · 017 (016) · 029 (002∧027 + starvation diag) · 034 (032∧033) · 035 (032) · 037 (016/017 + starvation diag) · 046 (015∧020) |
| **DROP (3)** | 010 (no tail-distribution support) · 012 (dominated duplicate of 011) · 022 (theta > run-up; confirmed weak) |
| **NO-GO (1)** | 023 (PIT consensus unobtainable at retail ∧ large-cap PEAD evidenced dead) |

**By family:** F7 4 GO / 1 COND / 1 DEFER · F5 3 COND-gated + 2 GO + 2 DROP + 1 COND · F1 4 GO / 1 COND / 1 DEFER · F2 1 COND(decisions) / 3 COND-shelf / 1 DROP / 1 NO-GO · F3 3 GO / 1 COND / 1 DEFER · F4 1 GO / 1 COND / 2 DEFER · F6 3 GO / 1 DEFER · F9 4 COND · F8 1 GO / 1 COND / 1 DEFER · F10 3 GO / 1 COND. The two families the screen genuinely gutted remain F2 (event vol — point-in-time earnings integrity, event-window cost realism) and F9 (flow/seasonality — cost arithmetic and decayed premia); F9's v1 *data* objections are resolved by the owned set, its economics objections are not.

## 6.1 Validation queue (waves)

- **Wave 0 — free/near-free, zero trials consumed.** **Group C first — the V-battery (V1–V12, Amendment A1 §5):** quote population/semantics, schema/dtype reconciliation, IV/greeks coverage by year, OI-join leakage, corporate actions, root continuity, partition/refresh integrity, provenance mix, empirical spread census, legacy-store disposition. Then **Group A** underlying-only pre-tests, runnable today (D-014/042 gap + first-hour; D-040a weekend variance; D-013/030 drawdown shapes; D-050a transition RV timing). Then **Group B** post-canonicalization diagnostics — the v1 list unchanged, precondition moved from "post-ORATS" to "post-Phase-1" at $0 (D-003, D-005, D-011, D-018, D-027, D-029, D-033, D-037, D-040b, D-043 after V6, D-047/030, D-048, D-049, D-050b).
- **Wave 1 (9 trials, post-canonicalization):** 015, 019, 016, 001, 002, 027, 032, 047, 050 — unchanged.
- **Wave 2 (≤10, conditional on Wave-1 signal + diagnostics):** unchanged membership (018, 036, 005, 006, 013, 021, 044, 048, 049†, 003†); 021 additionally awaits its two logged decisions.
- **Wave 3 (≤12):** unchanged membership; 041's condition is now V1–V3 + the cost bar; 043's is V6 + D-043; 007/008/009 still ride the PR6 gate.
- **Wave 4 / shelf (7):** unchanged membership; unlocks restated — 014/042 (pre-test then V1–V3), 040 (D-040b), 024 (parents), 025 (as amended), 026/031/045 (universe decision + original blockers).

**Data purchases: none required.** The owned set is primary pending the V-battery. **ORATS (~$399) is recommended, non-blocking** — GFC extension for the short-vol core, dividend-aware smoothed surfaces (006/027/030/047), breadth beyond 31 roots, cross-vendor validation. Databento CBBO-1m: dormant fallback if V1–V3 fail. Rung-3 earnings API: deferred until 020/021 front the queue. PIT consensus: never at retail.

## 6.2 Harness build items (consolidated, for CC)

Dividend-driven early-assignment + pin module (all short-call/put structures) · near-close mark convention (§2 rule 2) at the registered **15:45** snapshot minute on owned data (per A1 snapshot symmetry) · two-tier cost model (normal combo vs stressed single-leg-grade) with ±50% sensitivity switch · theta/vega P&L decomposition (F4, reused by 038) · hedge-ledger accounting with stressed underlying-slippage schedule (015/019/028/044/049) · regime-sliced attribution in standard output (015/039/048 falsifiers require it) · ledger integration: `emitted / tested / deferred / dropped` status per candidate, n_trials counted on test execution.

# 7. Multiple-Testing Ledger & DSR Implications (v2)

**Emitted vs tested — unchanged.** Slate appended +50 *emitted*; DSR's N counts trials **evaluated against the overlapping data window**. Wave selection and DEFER rules are pre-registered sequential testing; every deferred candidate that runs increments N normally.

**Null best-of-N arithmetic, revised for the owned window** (σ_SR ≈ 1/√years; 13.7 y → ≈ 0.27):

| Stage | N (this slate) | E[max SR | null], 13.7 y | with ORATS extension (19.5 y) |
|---|---|---|---|
| Wave 1 | 9 | ≈ **0.41** | ≈ 0.34 |
| + Wave 2 | ≈ 19 | ≈ **0.51** | ≈ 0.42 |
| + Waves 3–4 | ≈ 31–38 | ≈ **0.56–0.59** | ≈ 0.47–0.49 |

v1's intraday-specific penalty ("N = 5 on 6.5 y ≈ 0.47") is **retired** — intraday and EOD candidates share the 13.7 y window. Its replacements: **(1) missing-2008** — no GFC-class regime in-window; for the short-vol core this right-censors the loss distribution; curable at EOD granularity by the ORATS extension, permanent for intraday-native candidates; **(2) per-root windows** — PLTR ≈ 5.7 y (σ ≈ 0.42), COIN ≈ 4.7 y (σ ≈ 0.46); each candidate's DSR uses *its* window, and mixed-window baskets (020) report the binding shortest root.

**Lifetime reconciliation (CC task, Phase 4):** true N adds the repo ledger's prior history against overlapping windows — the RAMP equities-daily trials overlap the 2015+ span, and **the shelved OpEx pinning strategy's 2025 backtest is a prior tested trial** to be recorded. The real count comes from the repo, not from this chain. This slate's delta: +50 emitted; ≤ 9 tested at Wave 1.

# 8. The Reiterate Decision

**No regeneration of the slate is required.** The feasibility screen was the requested kill test, and the slate survives it: only one candidate is data-blocked (023), three die on priors, and the modal outcome is "researchable after a $399 purchase." The generation step did its job — the constraint was never the 1-minute underlying data, and the screen has converted the vague "worth exploring with 1m OHLCV?" question into a priced purchase ladder plus a wave protocol.

**Registered triggers for future reiteration** (all mean *broadening the hypothesis space*, never re-tuning against results):
1. **Wave-1 anchor failure:** if OPT-015/019 establish that index VRP is fully dead net of costs in the current regime, the F1/F3/F6 short-vol block loses its economic foundation — reiterate into families this slate under-weights (e.g., financing/rates-adjacent structures, cross-asset vol) rather than generating more short-vol variants.
2. **PR6 gate failure:** F5's long trio drops silently (already wired above) — no reiteration needed, the budget was never spent.
3. **Zero survivors after CPCV/PBO/DSR validation:** a valid outcome, stated in advance. Reiteration then means new mechanism families or new data (forward paper for the saturation-routed candidates), and STOP remains Shuyang's call, not the analysis's.

# 9. Honesty Block & Caveats (v2)

- Every verdict is a **feasibility-and-prior** judgment; no P&L exists; "GO" means *worth one draw from the finite sample*, nothing more. **No prior was upgraded by the data correction** — better feasibility is not better edge.
- **The cost model remains the weakest quantitative input, with a repair path:** V11's empirical spread census re-parameterizes widths from owned quotes (`cost_model_v2`); fill-fraction conventions stay assumptions until live IBKR fills replace them, at which point prior results are re-run.
- **Data-quality trust is conditional on the V-battery.** Shipped IV/greeks carry the registered caveats (dividends off, deep-ITM/OTM error, gamma absent at minute level); quote coverage on OTM strikes, OI-join semantics, provenance mix (ThetaData vs possible IBKR rows), and corporate-action handling are all unverified until Group C reports. Any result that predates a relevant V-item is provisional.
- **ThetaData subscription status/tier is open** (Phase 0.10) — it determines the marginal cost of universe top-ups and the 2026-03→present refresh. ORATS pricing (~$399) re-verified at checkout if purchased.
- **Decay evidence quality is mixed**, unchanged: correlation-premium, index-performance, and 0DTE-share figures trace to named studies/providers; VRP-compression and weekend-effect readings lean on practitioner sources; thin-evidence verdicts (026, 040, 022) already discount conservatively.
- **Amendment state:** 021 carries a pending exit-version decision (one version only; recommendation v1.0); 025's amendment stands; the P6 hedge-time alignment (15:50→15:45) is logged per A1. All registered before any data contact by the affected tests.
- The +50 ledger delta, wave rules, diagnostic gates, and this document's conditions are the pre-registration of record together with slate v1.1 and Amendment A1. Deviations require a logged amendment before the affected test runs. **Zero survivors after validation remains a legitimate outcome.**

*End of feasibility screen v2 (v1 retired per Amendment A1). Buildable form: `2026-07-25_options_slate_cc_handoff_spec_v2.md`.*
