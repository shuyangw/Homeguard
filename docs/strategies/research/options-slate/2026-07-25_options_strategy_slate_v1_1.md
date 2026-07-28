# Equity Options Strategy Slate — 50 Pre-Registered Candidates (v1.1)

**Date:** 2026-07-24
**Generator:** `/strat-gen` (executed inline from SKILL.md design draft — `references/`, `scripts/`, `assets/` not yet built; deterministic screens performed analytically and flagged for CC scripting)
**Asset class:** US-listed equity options (single names + ETF options: SPY, QQQ, IWM, sector ETFs)
**Destination:** Homeguard research harness (historical first), forward-paper queue for anything the historical data screen blocks
**Status:** BLIND slate. No candidate has touched any backtest return. Ranking is ex-ante prior only.

---

## What this document does NOT do (read first)

- It does **not** report or imply any backtest performance. Zero out-of-sample (OOS) returns were seen.
- It does **not** rank by expected Sharpe. It ranks by **ex-ante prior strength** (mechanism quality, persistence argument, feasibility headroom).
- It does **not** reset the multiple-testing budget. These 50 candidates are a **+50 increment** to the lifetime ledger (see Honesty Block).
- It does **not** issue GO decisions. The companion feasibility document (data + cost screens, informed by the research pass) issues GO / CONDITIONAL / NO-GO per candidate; the harness (Combinatorial Purged Cross-Validation (CPCV) / Probability of Backtest Overfitting (PBO) / Deflated Sharpe Ratio (DSR)) validates survivors; STOP remains Shuyang's call.

---

## Scope intake — recorded assumptions (flag if wrong; corrections propagate to the companion doc)

1. **Universe:** Optionable US equities/ETFs with liquid chains. Working default: SPY/QQQ/IWM + top ~100 single names by option volume. Per-candidate universes noted.
2. **Data inventory.** Underlying 1-minute Open-High-Low-Close-Volume (OHLCV) via Alpaca Securities Information Processor (SIP) feed is assumed solid. **Options-side data is owned** (corrected 2026-07-25 per Amendment A1; the original v1.0 text asserted that options-side data was an open decision variable with nothing above end-of-day chains assumed — that assertion was false at drafting time and is retired): `options_combined/` holds ~233 GB of 1-minute options data, 31 roots, 2012-06 → 2026-02, with per-minute OHLCV, bid/ask close, implied volatility, and delta/theta/vega, plus end-of-day (EOD)-joined gamma and open interest. Every candidate remains tagged with a **data class** (below); those tags are **requirements**, not availability claims, and they now resolve against disk rather than against a vendor decision — subject to the verification battery in Amendment A1 §5.
3. **Cost model placeholder:** $0.65/contract commission + regulatory fees; spread cost = fraction of quoted half-spread, moneyness/DTE-dependent (research pass to pin realistic numbers). Solo retail on Interactive Brokers (IBKR); multi-leg via native combo orders.
4. **Capital/capacity frame:** low-six-figure strategy allocations; capacity flags are relative to that, not institutional scale.
5. **Slate size:** 50, per request. Note the skill's own warning: more candidates raises the √(ln N) hurdle while diluting average prior. Fifty is accepted here because the user explicitly wants a broad regime-covering inventory that the feasibility screen will then cut — the *expectation is that a large fraction dies in screening*, which is a valid and intended outcome.

## Data-class taxonomy (used by every candidate; the pivot for the companion doc)

| Class | Requires | Notes |
|---|---|---|
| **D1** | Underlying 1m OHLCV only | Signal side only — no options strategy is P&L-computable from D1 alone |
| **D2** | D1 + **EOD option chains** (quotes or settle marks) | Supports daily-granularity entries/exits, hold-to-expiry and roll-on-close logic |
| **D3** | D1 + **intraday (1m) option OHLCV bars** | Trade-print bars; sparse/stale for OTM strikes — usability itself is a research question |
| **D4** | D1 + **option NBBO quote data** (EOD or intraday) | Needed wherever spread realism or intraday option marks materially drive the result |
| **D5** | Implied-vol (IV) surface / greeks | Derivable from D2/D3/D4 + pricing model, or vendor-supplied |
| **D6** | Auxiliary: point-in-time earnings calendar, open interest (OI), VIX term structure, corporate actions | Candidate-specific |

## Mechanism-family taxonomy and slate budget

Constructed for equity options (the skill's `references/taxonomy.md` does not exist yet; this table is its de facto v0 for this asset class):

| Family | Mechanism | Budget |
|---|---|---|
| F1 | Volatility risk premium (VRP) harvest — IV systematically exceeds subsequently realized vol | 6 |
| F2 | Event-vol lifecycle — predictable IV build/crush around scheduled announcements | 6 |
| F3 | Skew / tail premium — OTM puts (and occasionally calls) structurally rich | 5 |
| F4 | Term-structure carry — front/back IV slope mean-reversion and roll-down | 4 |
| F5 | Directional regime translation — convex option expression of momentum/trend signals | 8 |
| F6 | Range/mean-reversion theta — short premium conditioned on range-regime confirmation | 4 |
| F7 | Overwrite / equity-carry hybrids — equity risk premium + VRP blends (covered calls, CSPs, collars) | 6 |
| F8 | Relative value / dispersion — index-vs-component and cross-name IV spreads | 3 |
| F9 | Flow & calendar seasonality — 0DTE intraday patterns, weekend theta, expiration pinning | 4 |
| F10 | Crisis convexity / long-vol — regime hedges and vol-expansion capture | 4 |

Diversity note (stage-7 discipline, performed manually): parameter variants of one structure were collapsed into single candidates with fixed pre-registered parameters. 50 entries ≈ 50 mechanism-bearing hypotheses, not 50 knobs on 5 ideas. Residual intra-family correlation is expected and recorded per candidate.

## Prior tiers (ranking rule)

- **A** — Documented, economically persistent premium with a credible reason it survives at retail scale; feasibility headroom likely.
- **B** — Plausible mechanism, weaker or more contested evidence, or feasibility/cost headroom uncertain.
- **C** — Speculative, likely decayed/crowded, or carries a structural data/cost objection at drafting time. Retained for completeness; expected to die in screening.

## Honesty block — multiple-testing ledger delta

This slate appends **+50 trials** to the lifetime ledger (each candidate's parameters are FIXED; any later bounded-range sweep multiplies its `n_trials_contribution` by the sweep cardinality and must be re-appended). Under the null of zero edge, with N = 50 independent trials evaluated on ~3 years of daily-granularity results, the Bailey–López de Prado expected-maximum in-sample annualized Sharpe is ≈ **1.3** (σ_SR ≈ 1/√3 ≈ 0.58; E[max₅₀] ≈ 2.28·σ_SR), with a right tail well above 2. Any survivor must clear a DSR hurdle computed from the **lifetime** ledger — which already includes the RAMP Phase-3/4 variant history — not from this batch of 50. Zero survivors after validation is a valid outcome. The ledger append itself (`scripts/ledger_update.py`) does not exist yet; CC task: reconcile this delta into the repo ledger location once built.

## Iteration protocol (what "reiterate" means here)

If the companion feasibility screen kills a large fraction: iteration = **broadening the hypothesis space and deepening the research pass within surviving families** (or acquiring the data that flips CONDITIONALs), never re-generating against backtest results and never tuning survivors toward a Sharpe target. That path is the overfit.

---

# The Slate

Field key per candidate: **Family / Regime fit** (mapped to the Homeguard classifier: STRONG_BULL, WEAK_BULL, SIDEWAYS, BEAR, UNPREDICTABLE; CROSS = regime-agnostic or event-driven) · **Universe** · **DTE/hold** · **Structure & rules** · **Pre-registered parameters (fixed)** · **Thesis & persistence** · **Data class** · **Kill / falsifier** · **Cost & capacity note** · **Correlation expectation** (vs slate siblings and vs RAMP/OMR/CSCM) · **Prior tier**.

## F7 — Overwrite / Equity-Carry Hybrids

### OPT-001 — Delta-Targeted Covered Call Overwrite
**Family/Regime:** F7 / WEAK_BULL, SIDEWAYS · **Universe:** SPY, QQQ + 20 liquid megacaps · **DTE/hold:** sell 30–45 DTE, hold to 21 DTE or 50% max profit
**Structure & rules:** Long 100 shares per unit; sell 1 call at 0.25 delta. Roll at 21 DTE or 50% premium capture, whichever first. Skip overwrite when regime = STRONG_BULL (cap forfeiture) or UNPREDICTABLE (gap risk through short strike).
**Params (fixed):** Δ=0.25, entry DTE window 30–45, exit at 21 DTE / 50% profit.
**Thesis & persistence:** Collects VRP conditional on holding the equity anyway; persists because the marginal seller is compensated for capping convex upside — a risk transfer, not an anomaly that arbitrage removes. CBOE BXM-style evidence is decades long.
**Data class:** D2 (EOD chains suffice; underlying 1m only for regime gate).
**Kill/falsifier:** Net-of-cost premium capture < forfeited upside over full regime cycle; falsified if realized vol persistently ≥ implied at 0.25Δ strikes.
**Cost/capacity:** Low turnover (~1 round trip/month/name); spread cost small at near-ATM liquid strikes. Capacity effectively unconstrained at retail.
**Correlation:** High vs OPT-002/004/005; long-equity component overlaps RAMP beta.
**Prior:** **A** — persistent risk-transfer premium, trivially feasible.

### OPT-002 — Cash-Secured Put Wheel
**Family/Regime:** F7 / WEAK_BULL · **Universe:** 20 liquid megacaps + SPY/QQQ · **DTE/hold:** sell 30–45 DTE puts, 0.30Δ; if assigned, overwrite per OPT-001 until called away.
**Structure & rules:** Sell cash-secured put; take assignment; overwrite; repeat. No averaging down; one unit per name; regime gate: suspend new put sales in BEAR.
**Params (fixed):** put Δ=0.30, call Δ=0.25, DTE 30–45, exit 21 DTE/50%.
**Thesis & persistence:** Same VRP-plus-equity-premium blend as PUT index; persists as compensation for writing insurance against drawdowns buyers systematically overpay for (loss aversion of hedgers).
**Data class:** D2.
**Kill/falsifier:** Assignment clusters in regime transitions erase premium; falsified if put-side VRP at 0.30Δ ≤ 0 net of costs across a full cycle.
**Cost/capacity:** Low turnover; capital-inefficient (full cash securing) — return-on-capital hurdle noted. Capacity unconstrained.
**Correlation:** Very high vs OPT-001; equity beta overlaps RAMP.
**Prior:** **A** — PUT-index-class evidence; the question is return on tied-up cash, not existence of premium.

### OPT-003 — Financed Collar Carry
**Family/Regime:** F7 / CROSS with BEAR resilience · **Universe:** SPY · **DTE/hold:** quarterly structure, monthly call resets.
**Structure & rules:** Long SPY; buy 3M 0.15Δ put; finance with monthly 0.20Δ calls. Static roll calendar; no discretionary adjustment.
**Params (fixed):** put Δ=0.15 3M; call Δ=0.20 1M; roll on fixed calendar.
**Thesis & persistence:** Tests whether short-dated call richness can pay for longer-dated tail insurance (term + skew interaction). Persistence rests on the same overwrite premium; the put leg is a cost center by design.
**Data class:** D2 + D5 (needs consistent marks across two expiries).
**Kill/falsifier:** Financing ratio < ~70% on average → structure is just expensive insurance; falsified if collar drags > unhedged drawdown improvement across a bear regime.
**Cost/capacity:** Two legs/month; modest. Unconstrained capacity.
**Correlation:** Moderate vs OPT-001/047; diversifying vs RAMP in BEAR.
**Prior:** **B** — mechanism sound, but financing arithmetic frequently fails; kept for its regime-completing role.

### OPT-004 — Covered Strangle
**Family/Regime:** F7 / WEAK_BULL · **Universe:** 10 highest-conviction liquid megacaps · **DTE/hold:** 30–45 DTE both legs.
**Structure & rules:** Long shares + short 0.25Δ call + short 0.20Δ cash-secured put. Roll per OPT-001 rules; suspend put leg in BEAR/UNPREDICTABLE.
**Params (fixed):** as stated.
**Thesis & persistence:** Doubles VRP collection where willingness to add shares lower is genuine; persistence identical to OPT-001/002.
**Data class:** D2.
**Kill/falsifier:** Down-gap through put strike in vol expansion; falsified if incremental put premium ≤ incremental drawdown vs OPT-001 alone.
**Cost/capacity:** Low; unconstrained.
**Correlation:** Near-duplicate risk vs OPT-001+002 acknowledged — retained because the *joint* short-strangle-around-shares position has distinct regime kill rules; flagged to diversity screen as ½ effective trial.
**Prior:** **B**.

### OPT-005 — Laddered Cash-Secured-Put Pullback Grid
**Family/Regime:** F7 / WEAK_BULL, SIDEWAYS · **Universe:** SPY, QQQ · **DTE/hold:** 21–35 DTE, laddered strikes.
**Structure & rules:** Maintain 3-rung put ladder at 0.30/0.20/0.10Δ; sell next rung only after ≥2% underlying pullback (1m-derived daily close basis); unwind ladder in BEAR.
**Params (fixed):** rung deltas, 2% trigger, DTE 21–35.
**Thesis & persistence:** Conditions put-writing on short-term oversold states where put IV richens (demand spike from hedgers) — sells insurance when it is most overpriced.
**Data class:** D2 + D1 trigger.
**Kill/falsifier:** Pullback trigger is exactly the regime-transition filter that catches falling knives; falsified if post-trigger put VRP ≤ unconditional put VRP.
**Cost/capacity:** Low-moderate turnover; unconstrained.
**Correlation:** High vs OPT-002.
**Prior:** **B** — conditional-richening thesis is plausible but under-documented vs unconditional versions.

### OPT-006 — Poor Man's Covered Call (LEAPS Diagonal)
**Family/Regime:** F7 / STRONG_BULL, WEAK_BULL · **Universe:** SPY, QQQ, 10 megacaps · **DTE/hold:** long 12–18M 0.80Δ call; short 30–45 DTE 0.25Δ call.
**Structure & rules:** Capital-efficient synthetic of OPT-001. Roll short leg per OPT-001; roll long leg at 6M remaining. Regime gate identical to OPT-001.
**Params (fixed):** long Δ=0.80 12–18M, short Δ=0.25 30–45 DTE.
**Thesis & persistence:** Same overwrite premium at ~⅓ the capital; adds long-vega and term-structure exposure as the price of leverage.
**Data class:** D2 + D5 (LEAPS marks are wide; mark quality matters).
**Kill/falsifier:** LEAPS spread cost + vega drawdown in vol crush exceeds capital-efficiency gain; falsified if net carry < OPT-001 on matched notional.
**Cost/capacity:** LEAPS spreads are the cost center; capacity fine.
**Correlation:** Very high vs OPT-001 — flagged ½ effective trial.
**Prior:** **B**.

## F5 — Directional Regime Translation (convex expression of momentum/trend)

### OPT-007 — Bull Call Debit Spread on Momentum Confirmation
**Family/Regime:** F5 / STRONG_BULL · **Universe:** top-decile 12-1 momentum names within S&P 500, capped at 10 positions · **DTE/hold:** 45–60 DTE, exit at 21 DTE or +100%/−50% of debit.
**Structure & rules:** On regime = STRONG_BULL and name in momentum top decile with 20-day high breakout: buy ATM call, sell +1σ OTM call. Equal-debit sizing.
**Params (fixed):** DTE 45–60, wing at 1σ (30-day ATM IV), exits as stated.
**Thesis & persistence:** Momentum premium (documented, persistent via slow-moving capital and career-risk limits) expressed with defined risk and reduced vol-crush exposure vs naked calls. Options add convexity to an edge that already exists; they do not need to *be* the edge.
**Data class:** D2 + D1 signals + D5 (σ for wing placement).
**Kill/falsifier:** RAMP Phase-4 lesson applies — if underlying momentum edge is dead net of costs, the options wrapper only adds spread cost. Falsified if spread P&L < delta-equivalent stock P&L minus financing on matched signals.
**Cost/capacity:** 2 legs, monthly-ish turnover; single-name spreads wider than ETF. Capacity fine.
**Correlation:** High vs OPT-008/009 and vs RAMP itself (same underlying signal family) — this family deliberately re-expresses RAMP-style signals; diversity discount recorded.
**Prior:** **B** — mechanism inherited from momentum literature; wrapper cost is the open question.

### OPT-008 — ZEBRA (Zero Extrinsic Back-Ratio) Synthetic Momentum
**Family/Regime:** F5 / STRONG_BULL · **Universe:** SPY, QQQ · **DTE/hold:** 60–90 DTE, exit on regime downgrade or 21 DTE.
**Structure & rules:** Buy 2× ~0.70Δ calls, sell 1× ATM call → ~1.0 net delta, ~zero net extrinsic. Entered on STRONG_BULL confirmation; exit on regime exit.
**Params (fixed):** 2:1 at 0.70Δ/ATM, DTE 60–90.
**Thesis & persistence:** Stock-replacement with hard floor: max loss = net debit ≪ stock drawdown tail. Persistence rides the equity/momentum premium; the structure buys crash protection nearly free of theta.
**Data class:** D2 + D5.
**Kill/falsifier:** Falsified if roll costs + residual theta exceed the value of the tail floor across a cycle (i.e., you paid for insurance you could have replicated cheaper with an index put budget).
**Cost/capacity:** 3 legs per entry, low frequency. Fine.
**Correlation:** Very high vs OPT-007; high vs RAMP beta.
**Prior:** **B**.

### OPT-009 — Bullish Risk Reversal on Cross-Sectional Momentum Leaders
**Family/Regime:** F5 / STRONG_BULL · **Universe:** top-20 momentum liquid single names · **DTE/hold:** 45 DTE, exit 21 DTE.
**Structure & rules:** Sell 0.25Δ put, buy 0.25Δ call, ~zero cost. Max 5 concurrent names, equal notional-at-risk (put strike basis).
**Params (fixed):** 0.25Δ both legs, 45 DTE.
**Thesis & persistence:** Monetizes put-over-call skew richness *and* momentum simultaneously: in leaders, skew often stays elevated while drift is up. Two documented premia stacked in one structure.
**Data class:** D2 + D5 (skew measurement).
**Kill/falsifier:** Undefined downside — a single regime break across 5 correlated leaders is the risk concentrator. Falsified if skew capture ≤ tail losses in transition months.
**Cost/capacity:** Cheap to enter (near-zero premium); margin-intensive. Capacity fine.
**Correlation:** High vs OPT-007/008, RAMP; also touches F3 (skew) — cross-family link recorded.
**Prior:** **B**.

### OPT-010 — Call Ratio Backspread on Breakout
**Family/Regime:** F5 / STRONG_BULL, UNPREDICTABLE · **Universe:** QQQ + 10 high-beta megacaps · **DTE/hold:** 45–60 DTE.
**Structure & rules:** Sell 1 ATM call, buy 2 OTM (+1σ) calls for ~zero cost on 50-day-high breakout with rising 1m-derived realized vol. Convex to continuation; loses in slow drift to short strike.
**Params (fixed):** 1:2 ATM/+1σ, DTE 45–60, exit 21 DTE.
**Thesis & persistence:** Breakouts with expanding realized vol have fat right tails (momentum + vol clustering); the structure is long that specific tail while short the "grind" path.
**Data class:** D2 + D1 (RV trigger) + D5.
**Kill/falsifier:** The valley-of-death payoff at the long strikes at expiry; falsified if breakout follow-through distribution is not fat-tailed enough to pay for the valley.
**Cost/capacity:** 3 legs; fine.
**Correlation:** Moderate vs OPT-007/049.
**Prior:** **C** — payoff-shape cleverness exceeding evidence; expected to die in screening unless research finds tail-distribution support.

### OPT-011 — Bear Put Debit Spread on Breakdown Momentum
**Family/Regime:** F5 / BEAR · **Universe:** bottom-decile momentum S&P 500 names, max 10 · **DTE/hold:** 45–60 DTE, exits as OPT-007.
**Structure & rules:** Mirror of OPT-007: on regime ∈ {BEAR} and 20-day-low breakdown, buy ATM put / sell −1σ put.
**Params (fixed):** mirror OPT-007.
**Thesis & persistence:** Short-side momentum exists but is weaker and costlier than long-side (borrow constraints create it; options bypass borrow). Defined risk caps squeeze losses — the classic short-side killer.
**Data class:** D2 + D1 + D5.
**Kill/falsifier:** Put IV is already rich in BEAR (skew) — paying up for direction. Falsified if entry IV richness consumes the drift edge.
**Cost/capacity:** As OPT-007.
**Correlation:** Negative vs F7 block and RAMP — genuine diversifier; recorded as such.
**Prior:** **B** — diversification value props up a middling standalone prior.

### OPT-012 — Bearish Risk Reversal
**Family/Regime:** F5 / BEAR · **Universe:** SPY, QQQ · **DTE/hold:** 45 DTE.
**Structure & rules:** Sell 0.25Δ call, buy 0.25Δ put on regime = BEAR confirmation. Index-only (no single-name squeeze exposure on the short call).
**Params (fixed):** 0.25Δ both, 45 DTE, exit on regime exit or 21 DTE.
**Thesis & persistence:** In confirmed BEAR, call selling collects elevated IV while the put leg is the directional payload; structure is short the rally-tail, which is precisely the falsifier.
**Data class:** D2 + D5.
**Kill/falsifier:** Bear-market rallies are the most violent; falsified if short-call losses in rallies exceed put gains in continuations (this is where OPT-011's defined risk likely dominates).
**Cost/capacity:** Fine.
**Correlation:** Very high vs OPT-011 — flagged ½ effective trial.
**Prior:** **C**.

### OPT-013 — Put Ratio Backspread on Deterioration
**Family/Regime:** F5→F10 boundary / BEAR, UNPREDICTABLE · **Universe:** SPY · **DTE/hold:** 60 DTE.
**Structure & rules:** Sell 1 ATM put, buy 2 −1σ puts, ~zero cost, entered on regime downgrade (WEAK_BULL→SIDEWAYS→BEAR transitions) with VIX term flattening.
**Params (fixed):** 1:2, −1σ wing, 60 DTE.
**Thesis & persistence:** Crash convexity financed by near-the-money premium exactly when transition probability is elevated; conditional entry is the prior-elevation vs a permanent backspread (which bleeds).
**Data class:** D2 + D5 + D6 (VIX term).
**Kill/falsifier:** Valley of death on moderate declines to the long strikes; falsified if declines are predominantly slow grinds (the 2022 pattern) rather than gaps.
**Cost/capacity:** Fine.
**Correlation:** Moderate vs OPT-047/050; diversifying vs book.
**Prior:** **B**.

### OPT-014 — Opening-Range Gap Directional Spreads (intraday)
**Family/Regime:** F5 / CROSS (intraday) · **Universe:** SPY, QQQ · **DTE/hold:** same-day or next-expiry verticals, entered ~10:00, closed by 15:45.
**Structure & rules:** If open gaps > ±0.5% vs prior close and first-30-min range confirms direction (close of 10:00 bar beyond opening range), enter a 1-wide debit vertical in gap direction; hard time stop 15:45.
**Params (fixed):** 0.5% gap, 30-min confirmation, fixed strikes ATM/±1 strike.
**Thesis & persistence:** Gap continuation is a documented intraday pattern (underreaction + intraday momentum); options cap the adverse tail of fade days.
**Data class:** **D3/D4** — requires intraday option marks; this candidate is a deliberate probe of the 1m-options-bar question. D1 alone cannot price it.
**Kill/falsifier:** Spread crossing twice a day at 0-1 DTE is the cost killer; falsified if per-trade gross edge < round-trip spread (the skill's 70–90%-turnover DOA warning applies in spirit).
**Cost/capacity:** High turnover — weak cost prior recorded at drafting time.
**Correlation:** Low vs rest of slate; overlaps OMR's overnight/intraday decomposition conceptually.
**Prior:** **C** pending cost evidence.

## F1 — Volatility Risk Premium (VRP) Harvest

### OPT-015 — Index Delta-Hedged Short Straddle (core VRP)
**Family/Regime:** F1 / CROSS, best in SIDEWAYS · **Universe:** SPY · **DTE/hold:** sell 30 DTE ATM straddle; hold to 7 DTE; delta-hedge with shares at fixed daily rebalance on the registered snapshot minute, **15:45** (amended from 15:50 on 2026-07-25 per Amendment A1 — snapshot symmetry; logged before any test).
**Structure & rules:** One straddle unit sized to fixed vega budget; daily EOD delta hedge (not intraday — deliberate, to stay within D2); suspend entries when regime = UNPREDICTABLE.
**Params (fixed):** 30→7 DTE, daily **15:45** hedge (amended from 15:50 per A1), vega-budget sizing.
**Thesis & persistence:** The variance/VRP premium is the most-documented options premium: index IV exceeds subsequent RV on average because hedgers pay for insurance and dealers demand inventory compensation. Persists because the loss profile (short crash) is exactly what most capital cannot warehouse — retail with strict sizing can.
**Data class:** D2 + D5 (+D1 for hedge marks). The **daily-hedge** variant is the researchable one; intraday hedging escalates to D3/D4 (see OPT-049).
**Kill/falsifier:** Falsified if post-2022 index VRP has compressed below hedge-slippage + costs (research pass to establish current level); killed by two consecutive vol-regime losses exceeding pre-registered drawdown.
**Cost/capacity:** Low leg count, daily share hedges cheap. Capacity unconstrained.
**Correlation:** The family anchor — OPT-016/017/018/020 are its defined-risk/conditioned siblings. Short-vol P&L correlates with RAMP drawdowns (both lose in crashes) — book-level note.
**Prior:** **A** — strongest single prior on the slate.

### OPT-016 — Iron Condor, Delta-Managed
**Family/Regime:** F1 / SIDEWAYS · **Universe:** SPY, QQQ, IWM · **DTE/hold:** 30–45 DTE; exit 50% profit or 21 DTE; defend nothing (mechanical close at short-strike touch).
**Structure & rules:** Short 0.20Δ strangle + 0.05Δ wings. No adjustment — touch = close (adjustment rules are researcher-degrees-of-freedom generators).
**Params (fixed):** 0.20Δ/0.05Δ, DTE 30–45, exits as stated.
**Thesis & persistence:** OPT-015's premium with bought tail insurance; pays away part of VRP for a defined worst case and margin efficiency.
**Data class:** D2.
**Kill/falsifier:** Wings cost > tail benefit at retail sizing (wings are the most-overpriced strikes — you're buying the skew you elsewhere sell); falsified if long-run P&L < OPT-015 at matched vega with worse Sharpe.
**Cost/capacity:** 4 legs — commission/spread drag ×2 vs straddle. Fine.
**Correlation:** Very high vs OPT-015/017/018 — family concentration acknowledged; retained because defined-risk feasibility may survive screens that kill undefined-risk siblings.
**Prior:** **A−**.

### OPT-017 — Iron Butterfly
**Family/Regime:** F1 / SIDEWAYS · **Universe:** SPY · **DTE/hold:** 30 DTE → 50%/21 DTE exits.
**Structure & rules:** Short ATM straddle + 1σ wings; mechanical exits as OPT-016.
**Params (fixed):** as stated.
**Thesis & persistence:** Max-theta point of the defined-risk family; harvests the richest (ATM) IV with capped tails.
**Data class:** D2.
**Kill/falsifier:** Narrow profit zone → path sensitivity; falsified if realized drift/vol makes touch-exits dominate (same falsifier family as OPT-016; near-duplicate flagged, ½ effective trial).
**Cost/capacity:** 4 legs; fine.
**Correlation:** Near-duplicate of OPT-016.
**Prior:** **B+**.

### OPT-018 — IV-Rank-Gated Short Strangle
**Family/Regime:** F1 / SIDEWAYS · **Universe:** SPY, QQQ + 10 megacaps · **DTE/hold:** 45 DTE → 50%/21 DTE.
**Structure & rules:** Sell 0.16Δ strangle only when 1-year IV rank > 50 (Implied Volatility Rank (IVR): current IV percentile vs trailing year). Undefined risk, half-size vs OPT-015 vega budget.
**Params (fixed):** 0.16Δ, IVR>50 gate, 45 DTE.
**Thesis & persistence:** Conditions VRP collection on IV richness — sell insurance when it's expensive. The gate is the prior-elevation vs unconditional short vol.
**Data class:** D2 + D5 (IV history for rank).
**Kill/falsifier:** IVR>50 states cluster in exactly the regimes where vol keeps rising (conditioning trap); falsified if gated VRP ≤ ungated VRP — a clean pre-registered comparison vs OPT-015.
**Cost/capacity:** Fine.
**Correlation:** High vs OPT-015.
**Prior:** **B+** — the gate is popular practitioner lore with surprisingly thin rigorous support; worth the clean test.

### OPT-019 — Forecast-Throttled Short Vol (RV-model vs IV)
**Family/Regime:** F1 / CROSS · **Universe:** SPY · **DTE/hold:** as OPT-015.
**Structure & rules:** OPT-015's structure, but position size ∝ (IV − RV_forecast)⁺ where RV_forecast is a Heterogeneous Autoregressive (HAR-RV) model on 1m-bar Yang-Zhang realized vol. Zero size when spread ≤ 0.
**Params (fixed):** HAR-RV(1,5,22) spec frozen; linear throttle; cap at 2× base vega.
**Thesis & persistence:** Replaces the binary IVR gate with a continuous, model-based VRP estimate; the 1m underlying data is the genuine input edge here (better RV estimation than daily bars).
**Data class:** D2 + D5 + **D1-intensive** (this is where the 1m OHLCV asset actually earns its keep).
**Kill/falsifier:** HAR forecast adds nothing over trailing RV (falsifier: throttled ≤ unthrottled net Sharpe); model-risk: frozen spec drifts.
**Cost/capacity:** As OPT-015.
**Correlation:** High vs OPT-015/018 — this trio is a pre-registered *ablation ladder* (unconditional → gated → forecast-throttled), deliberately structured per the cheap-first/ablation principle.
**Prior:** **A−**.

### OPT-020 — Single-Name VRP Basket
**Family/Regime:** F1 / SIDEWAYS · **Universe:** 10 megacaps (options ADV screen) · **DTE/hold:** 30 DTE delta-hedged straddles, daily hedge.
**Structure & rules:** OPT-015 replicated per-name, equal vega, earnings-window excluded (no position spanning a print — that's F2's job).
**Params (fixed):** as OPT-015 + earnings exclusion.
**Thesis & persistence:** Single-name VRP historically exceeds index VRP (idiosyncratic vol overpricing + retail call demand distortions both directions); earnings exclusion isolates the ambient premium.
**Data class:** D2 + D5 + D6 (point-in-time earnings calendar — a leakage trap flagged: calendar must be as-known-then).
**Kill/falsifier:** Single-name gap risk between hedges; falsified if basket VRP net of wider single-name spreads ≤ index VRP (then OPT-015 dominates and this dies).
**Cost/capacity:** Wider spreads, 10× positions; the cost-screen's main F1 casualty candidate.
**Correlation:** High vs OPT-015; adds idiosyncratic-vol exposure the index lacks.
**Prior:** **B**.

## F2 — Event-Vol Lifecycle

### OPT-021 — Earnings Vol-Crush Short Iron Fly
**Family/Regime:** F2 / CROSS · **Universe:** liquid single names with weekly options, |expected move| priced ≥ 4% · **DTE/hold:** enter T−1 close before print, nearest expiry, exit T+1 open/first liquid mark.
**Structure & rules:** Short ATM straddle + wings at the priced expected move. Defined risk always (undefined-risk earnings shorts are an uninsurable career-ender at retail).
**Params (fixed):** entry T−1 close, wings at priced move, exit T+1 open.
**Thesis & persistence:** Event IV systematically overprices realized earnings moves on average (documented for decades); persists because the left tail (single-name ±20% prints) is unhedgeable, so sellers demand — and mostly earn — a premium.
**Data class:** **D2 minimum + D6 (point-in-time earnings calendar); D4 preferred** — T+1-open option marks from trade-prints are unreliable; quote data materially changes measured P&L. Flagged as a primary research-pass question.
**Kill/falsifier:** Falsified if post-2020 crowding (retail flow, 0DTE-era vol selling) has compressed the crush below wing cost + spread; killed by regime where realized > implied moves persistently (e.g., macro-dominated tapes).
**Cost/capacity:** 4 legs × 2 transits near an event with wide spreads — cost realism is THE screen here. Capacity fine.
**Correlation:** Low vs F1 (event-window vs ambient premium — deliberately disjoint by OPT-020's exclusion rule).
**Prior:** **A−** — strong documented premium, cost-fragile.

### OPT-022 — Pre-Earnings IV Run-Up (long)
**Family/Regime:** F2 / CROSS · **Universe:** same as OPT-021 · **DTE/hold:** buy T−6, exit T−1 close (never hold through print).
**Structure & rules:** Long ATM straddle in the expiry spanning the print; exit before the event. Pure vega-appreciation trade.
**Params (fixed):** T−6 entry, T−1 exit.
**Thesis & persistence:** IV mechanically builds into scheduled events; the trade is long that build while theta bleeds against it. Persistence claim is weaker: the build is known, so entry IV should embed it — edge only if the build is systematically underpriced at T−6.
**Data class:** D2 + D5 + D6.
**Kill/falsifier:** Theta > vega gain (most likely outcome); falsified directly by that inequality on average. Explicit dead-trial candidate retained to bound the family.
**Cost/capacity:** 2 legs, moderate spreads.
**Correlation:** Negative vs OPT-021 by construction (long vs short the same event vol at different phases).
**Prior:** **C**.

### OPT-023 — Post-Earnings-Announcement Drift (PEAD) Verticals
**Family/Regime:** F2/F5 hybrid / CROSS · **Universe:** S&P 500 reporters with |surprise| in top quintile (point-in-time consensus) · **DTE/hold:** enter T+1, 30–45 DTE debit vertical in surprise direction, exit 21 DTE.
**Params (fixed):** top-quintile surprise gate, ATM/1σ vertical.
**Thesis & persistence:** PEAD is among the most durable anomalies (underreaction, limits to arbitrage); post-print IV crush makes long premium *cheap* precisely at entry — a rare alignment where the options wrapper improves the underlying anomaly's economics.
**Data class:** D2 + D6 (**point-in-time consensus/surprise data — a real acquisition question**; flagged).
**Kill/falsifier:** PEAD decay post-2010s in large caps is documented; falsified if top-quintile drift < spread cost at 30–45 DTE horizon.
**Cost/capacity:** Fine.
**Correlation:** Moderate vs OPT-007; low vs F1.
**Prior:** **B+** — anomaly durable, data acquisition is the constraint.

### OPT-024 — Post-Crush Overshoot Short Strangle (T+1)
**Family/Regime:** F2 / CROSS · **Universe:** as OPT-021 · **DTE/hold:** enter T+1 close, 21–30 DTE 0.16Δ strangle, standard exits.
**Params (fixed):** as stated.
**Thesis & persistence:** After prints, IV sometimes stays elevated relative to the now-resolved information set (sticky-IV overshoot); sells the residual.
**Data class:** D2 + D5 + D6.
**Kill/falsifier:** Post-event IV may be elevated *because* continuation risk is real (guidance digestion, analyst-day follow-ons); falsified if post-print VRP ≤ ambient single-name VRP (then OPT-020 dominates).
**Cost/capacity:** Fine.
**Correlation:** High vs OPT-020/021.
**Prior:** **C+**.

### OPT-025 — Macro-Event Index Straddle (CPI/FOMC)
**Family/Regime:** F2 / UNPREDICTABLE-adjacent · **Universe:** SPY · **DTE/hold:** buy nearest-expiry ATM straddle at T−1 close before CPI/FOMC, exit T+0 at the registered **15:45** snapshot minute (amended from 15:50 per A1 — snapshot symmetry).
**Params (fixed):** event set = {CPI, FOMC}, entry/exit as stated.
**Thesis & persistence:** Tests whether index options *underprice* scheduled macro vol (the mirror of OPT-021's claim for single names — index event premia are more contested). Post-2022 macro regime made these the dominant vol events.
**Data class:** D2 (+D3/D4 for the intraday exit mark — same probe as OPT-014) + D6 (event calendar).
**Kill/falsifier:** Falsified if implied event moves ≥ realized on average (the standard finding); this is a deliberately-included null-anchor with a regime-conditional twist (does it flip in UNPREDICTABLE?).
**Cost/capacity:** SPY spreads tight; cost-tolerant.
**Correlation:** Low vs slate; long-vol diversifier.
**Prior:** **C+**.

### OPT-026 — Sector-Sympathy IV Fade
**Family/Regime:** F2/F8 boundary / CROSS · **Universe:** sector peers of a mega-cap reporter (e.g., peers on NVDA week) · **DTE/hold:** sell peer 0.20Δ strangles T−1 before the *leader's* print, exit T+1.
**Params (fixed):** leader set = top-5 index-weight reporters; peer set = same-GICS-industry liquid names.
**Thesis & persistence:** Peers' IV inflates in sympathy with a leader's event but their realized response is dampened (partial information transfer); sells the sympathy premium without direct print exposure.
**Data class:** D2 + D5 + D6.
**Kill/falsifier:** Sympathy moves are real when the leader's news is sector-systematic (the 2023–24 AI-capex pattern); falsified if peer realized moves ≥ peer implied on leader-print days.
**Cost/capacity:** Single-name spreads; moderate.
**Correlation:** Moderate vs OPT-021, OPT-045.
**Prior:** **B−** — original enough to be interesting, thin literature; adversarial critique notes the falsifier is currently *live* in AI-linked sectors.

## F3 — Skew / Tail Premium

### OPT-027 — Steep-Skew Put-Spread Harvest
**Family/Regime:** F3 / WEAK_BULL, SIDEWAYS · **Universe:** SPY · **DTE/hold:** 45 DTE; sell 0.25Δ/buy 0.10Δ put spread when 25Δ put-call IV skew > trailing-2y 70th percentile; exits 50%/21 DTE.
**Params (fixed):** skew percentile 70, strikes as stated.
**Thesis & persistence:** Index put skew embeds a persistent crash-insurance premium (structural hedging demand from institutions that must hedge regardless of price); selling *spreads* keeps the tail bought. Conditioning on skew steepness sells insurance at its richest.
**Data class:** D2 + D5 (skew history).
**Kill/falsifier:** Steep skew predicts steep skew (it's a regime, not a mispricing) — falsified if conditional premium ≤ unconditional; also the OPT-018 conditioning-trap critique applies verbatim.
**Cost/capacity:** 2 legs; fine.
**Correlation:** High vs F1 block (short-vol kin) and vs OPT-002.
**Prior:** **A−**.

### OPT-028 — Skew-Extreme Risk-Reversal Relative Value
**Family/Regime:** F3 / CROSS · **Universe:** SPY, QQQ · **DTE/hold:** 60 DTE; when 25Δ skew > 90th pct: sell put/buy call (collect rich side); when < 10th pct: reverse. Delta-hedged daily.
**Params (fixed):** 90/10 percentiles, 0.25Δ legs, daily hedge.
**Thesis & persistence:** Trades skew as a mean-reverting priced factor rather than holding it as carry — the delta hedge isolates the skew/vega P&L from direction.
**Data class:** D2 + D5, hedge via D1.
**Kill/falsifier:** Skew extremes coincide with regime information (steep = crash-fear that's sometimes right); falsified if hedged skew-reversion P&L ≤ 0 net after the hedge slippage.
**Cost/capacity:** Daily hedges add drag; moderate.
**Correlation:** Moderate vs OPT-027; the hedged version is the diversifier, the unhedged one would be OPT-009's cousin.
**Prior:** **B**.

### OPT-029 — Jade Lizard
**Family/Regime:** F3/F7 hybrid / WEAK_BULL · **Universe:** 10 megacaps + SPY · **DTE/hold:** 45 DTE; short 0.25Δ put + short call spread (0.25Δ/0.15Δ) with total credit > call-spread width → zero upside risk.
**Params (fixed):** as stated; credit>width constraint is a hard entry filter, not a target.
**Thesis & persistence:** Structurally monetizes put-skew richness (fat put premium funds the no-upside-risk constraint). Persistence inherits from F3/F7.
**Data class:** D2.
**Kill/falsifier:** Downside is a naked put — falsified under the same conditions as OPT-002, and if the credit>width filter binds so rarely the strategy starves (a feasibility falsifier, checkable cheaply).
**Cost/capacity:** 3 legs; fine.
**Correlation:** Very high vs OPT-002/027 — flagged ½ effective trial.
**Prior:** **B**.

### OPT-030 — Crash-Zone Broken-Wing Put Butterfly
**Family/Regime:** F3 / BEAR-hedge with positive expected carry · **Universe:** SPY · **DTE/hold:** 60–90 DTE; long 1× −1σ put, short 2× −1.5σ puts, long 1× −2.5σ put (asymmetric wings) for ~zero cost; rolled monthly.
**Params (fixed):** wing geometry as stated.
**Thesis & persistence:** Exploits skew *curvature*: mid-crash strikes are the richest per unit of tail probability; the BWB sells that hump to own the far tail nearly free.
**Data class:** D2 + D5 (curvature measurement).
**Kill/falsifier:** The short hump is exactly where slow-grind bears settle; falsified if terminal distributions concentrate at −1.5σ (2022-style) rather than bimodal (calm/crash).
**Cost/capacity:** 4 legs at OTM strikes — spread-cost sensitive.
**Correlation:** Complements OPT-047; moderate vs OPT-013.
**Prior:** **B**.

### OPT-031 — Inverted-Call-Skew Fade (squeeze names)
**Family/Regime:** F3 / UNPREDICTABLE · **Universe:** names where 25Δ call IV > 25Δ put IV (inverted skew — squeeze/meme signature), liquid chains only · **DTE/hold:** 21–30 DTE; short 0.15Δ call *spread* (defined risk, never naked), half-size.
**Params (fixed):** inversion trigger, 0.15Δ/0.05Δ spread.
**Thesis & persistence:** Call-skew inversion marks retail lottery demand; lottery-preference overpricing of OTM calls is documented (OTM single-name calls have historically dreadful average returns).
**Data class:** D2 + D5.
**Kill/falsifier:** Squeezes are the fat tail being sold; defined risk caps it but the entry regime *selects for* squeeze candidates. Falsified if inversion-conditional call returns aren't negative enough to beat spread cost at 0.15Δ.
**Cost/capacity:** Wide spreads in exactly these names — cost screen will bite.
**Correlation:** Low vs slate — genuine tail-flavor diversifier.
**Prior:** **C+** — mechanism documented, implementation hostile.

## F4 — Term-Structure Carry

### OPT-032 — Contango Calendar Carry
**Family/Regime:** F4 / SIDEWAYS · **Universe:** SPY, QQQ · **DTE/hold:** short 30 DTE ATM straddle / long 60 DTE ATM straddle when M2/M1 ATM IV ratio > 1.05; exit at front expiry −7d.
**Params (fixed):** 1.05 slope trigger, 30/60 structure.
**Thesis & persistence:** Upward-sloping IV term structure pays the calendar holder theta differential + roll-down; persists as the term-premium analog of VRP (longer-dated vol embeds uncertainty premium).
**Data class:** D2 + D5 across two expiries.
**Kill/falsifier:** Long vega — vol crush hits the back month; falsified if slope-conditional carry ≤ 0 after the vega mark-to-market, i.e., if slope is compensation not mispricing.
**Cost/capacity:** 2 expiries × 2 legs; fine.
**Correlation:** Moderate vs F1 (net short gamma, long vega — different vol axis).
**Prior:** **B+**.

### OPT-033 — Backwardation Reverse Calendar
**Family/Regime:** F4 / UNPREDICTABLE · **Universe:** SPY · **DTE/hold:** long 30 DTE / short 60 DTE ATM straddles when M2/M1 < 0.97 (inverted term); exit on slope re-normalization or front −7d.
**Params (fixed):** 0.97 trigger.
**Thesis & persistence:** Inversion marks stress; front vol both realizes hardest and mean-reverts fastest — owns the gamma where it pays, short the vega that normalizes.
**Data class:** D2 + D5.
**Kill/falsifier:** Margin treatment of short back-month is hostile; falsified if inversion episodes are too few/short for the edge to be estimable at all (sample-starvation falsifier — checkable before any backtest).
**Cost/capacity:** Fine mechanically, margin-intensive.
**Correlation:** Negative vs OPT-032 by construction; long-vol diversifier.
**Prior:** **B−**.

### OPT-034 — Term-Slope Sign-Switched Calendar Program
**Family/Regime:** F4 / CROSS · **Universe:** SPY · **DTE/hold:** unified rule: position = sign(slope − 1) × calendar, sized by |slope−1| percentile, always-on.
**Params (fixed):** the OPT-032/033 triggers unified into one continuous rule.
**Thesis & persistence:** The always-on version tests whether term-slope is a *tradable factor* rather than two episodic trades; sample efficiency is the draw.
**Data class:** D2 + D5.
**Kill/falsifier:** If OPT-032/033 have asymmetric mechanisms (carry vs stress-gamma), unifying them destroys both; falsified if the program ≤ the better episodic leg. Recorded as the *ablation parent* of 032/033 — the trio counts as an ablation ladder, like the F1 trio.
**Cost/capacity:** As above.
**Correlation:** Very high vs 032/033 (by design).
**Prior:** **B**.

### OPT-035 — Double-Calendar Range Harvest
**Family/Regime:** F4/F6 hybrid / SIDEWAYS · **Universe:** SPY · **DTE/hold:** calendars at ±0.5σ strikes (put below, call above), 30/60 DTE, exit front −7d.
**Params (fixed):** ±0.5σ placement.
**Thesis & persistence:** Widens the calendar's profit zone across a range; harvests term carry while tolerating drift — the range-regime-native F4 expression.
**Data class:** D2 + D5.
**Kill/falsifier:** Falsified if it underperforms OPT-032 at matched vega (extra legs must buy real path tolerance, not just cost).
**Cost/capacity:** 4 legs, 2 expiries — cost-heavier.
**Correlation:** Very high vs OPT-032 — flagged ½ effective trial.
**Prior:** **B−**.

## F6 — Range / Mean-Reversion Theta

### OPT-036 — Range-Gated Iron Condor (1m realized-range gate)
**Family/Regime:** F6 / SIDEWAYS · **Universe:** SPY, QQQ · **DTE/hold:** as OPT-016.
**Structure & rules:** OPT-016's structure, entered only when (a) regime = SIDEWAYS and (b) 10-day Yang-Zhang realized vol (from 1m bars) < 30-day ATM IV × 0.8.
**Params (fixed):** 0.8 ratio, YZ window 10d.
**Thesis & persistence:** Conditions the condor on *measured* range behavior rather than IV alone; the 1m-bar YZ estimator is the informational edge over daily-bar sellers.
**Data class:** D2 + D5 + D1-intensive.
**Kill/falsifier:** Low RV/IV ratio may just mean IV correctly anticipates event risk ahead; falsified if gated ≤ ungated OPT-016 (pre-registered ablation vs OPT-016, mirroring the F1 ladder).
**Cost/capacity:** As OPT-016.
**Correlation:** Very high vs OPT-016 — ablation sibling, ½ effective trial.
**Prior:** **B+**.

### OPT-037 — Directional-Tilt Broken-Wing Butterfly
**Family/Regime:** F6 / SIDEWAYS with drift tolerance · **Universe:** SPY · **DTE/hold:** 30–45 DTE; put BWB with wings 1σ/2.5σ skewed opposite the 20-day drift sign, no-cost or small-credit entries only.
**Params (fixed):** wing geometry, drift-sign tilt rule.
**Thesis & persistence:** Theta harvest that pre-positions for the *continuation* of modest drift instead of pure pinning — accepts range-regime persistence as the mechanism.
**Data class:** D2 + D5.
**Kill/falsifier:** Falsified if tilt adds nothing vs symmetric fly (ablation vs OPT-017); the no-cost entry filter may starve it (feasibility falsifier, cheap to check).
**Cost/capacity:** 3–4 legs; fine.
**Correlation:** High vs OPT-016/017/030.
**Prior:** **B−**.

### OPT-038 — Double Diagonal Income
**Family/Regime:** F6 / SIDEWAYS · **Universe:** SPY · **DTE/hold:** short 30 DTE 0.25Δ strangle / long 60 DTE 0.15Δ strangle; roll short legs monthly, re-strike long legs quarterly.
**Params (fixed):** deltas/DTEs as stated.
**Thesis & persistence:** Condor with term-carry replacing bought same-expiry wings — the insurance is financed by roll-down instead of fully paid. Hybrid of F6 theta and F4 carry.
**Data class:** D2 + D5.
**Kill/falsifier:** Vega exposure of long back legs — vol crush hurts what the condor's wings wouldn't; falsified if it can't beat OPT-016 at matched risk (ablation).
**Cost/capacity:** 4 legs, staggered maintenance — the most operationally fiddly F6 entry.
**Correlation:** High vs OPT-016/032.
**Prior:** **B−**.

### OPT-039 — Band-Recentered Short Straddle
**Family/Regime:** F6 / SIDEWAYS · **Universe:** SPY · **DTE/hold:** 30 DTE short ATM straddle; recenter (roll to new ATM) when underlying exits a ±0.75σ band; hard stop on second recenter; else exits per OPT-015.
**Params (fixed):** 0.75σ band, max 1 recenter.
**Thesis & persistence:** Mean-reversion-native management of the F1 anchor: in true range regimes, recentering converts drift-through-strike losses into re-collected premium. The capped recenter count is the discipline separating this from martingale adjustment lore.
**Data class:** D2 + D1 (band monitoring at daily close keeps it D2-clean; intraday banding would escalate to D3).
**Kill/falsifier:** Trending regimes make recentering a loss-realization machine; falsified if recentered ≤ non-recentered OPT-015 in SIDEWAYS-classified periods specifically.
**Cost/capacity:** Extra transits on recenters; moderate.
**Correlation:** Very high vs OPT-015.
**Prior:** **B−**.

## F9 — Flow & Calendar Seasonality

### OPT-040 — Weekend Theta Capture
**Family/Regime:** F9 / SIDEWAYS-favored · **Universe:** SPY · **DTE/hold:** sell 7–10 DTE 0.16Δ strangle Friday 15:45, close Monday 09:45.
**Params (fixed):** entry/exit clock times, 0.16Δ.
**Thesis & persistence:** Options decay three calendar days over a weekend while realized underlying variance over the same span is typically below ⅗ of three trading days' worth — *if* Friday IV doesn't already discount it. The persistence question (does Friday pricing pre-crush weekend theta?) is exactly what the research pass must answer; practitioner claims conflict.
**Data class:** D2 with precise Friday-close/Monday-open marks; **D4 preferred** (open prints unreliable) — a data-quality probe like OPT-021.
**Kill/falsifier:** Weekend gap risk (geopolitics prints on Sundays); falsified if Friday IV already embeds the weekend (measurable: Friday ATM IV vs Thursday, adjusted).
**Cost/capacity:** Weekly round trips — cost-sensitive.
**Correlation:** High vs F1 family.
**Prior:** **C+**.

### OPT-041 — 0DTE Post-Opening-Range Iron Condor
**Family/Regime:** F9 / intraday, SIDEWAYS-day-favored · **Universe:** SPY (0DTE = zero days to expiration — same-day expiry) · **DTE/hold:** enter 10:30 after opening range establishes, wings beyond ±1× opening range, hold to 15:55 or short-strike touch.
**Params (fixed):** 10:30 entry, range-multiple wing placement, touch exit.
**Thesis & persistence:** Intraday VRP: 0DTE IV must price the day's remaining move; after the opening range resolves auction uncertainty, remaining implied moves have historically exceeded remaining realized moves on non-event days. Persistence contested — the 2023–26 0DTE boom crowded exactly this.
**Data class:** **D3/D4 required** — the flagship probe of the intraday-options-data question. Not researchable from D2.
**Kill/falsifier:** Falsified if post-2023 crowding compressed intraday VRP below the (large) round-trip cost of 4 legs × daily frequency; the skill's turnover-DOA warning applies at full force.
**Cost/capacity:** Daily 4-leg round trips — the worst cost profile on the slate. Capacity fine.
**Correlation:** Moderate vs F1 (intraday vs multiday premium — partially distinct risk).
**Prior:** **C+** — big literature interest, hostile economics; kept as the designated 0DTE representative rather than proliferating variants.

### OPT-042 — 0DTE First-Hour Trend Debit Spread
**Family/Regime:** F9/F5 hybrid / intraday · **Universe:** SPY · **DTE/hold:** if first-hour return ≥ |0.35%|, enter same-day 1-wide debit vertical in trend direction at 10:30, exit 15:45.
**Params (fixed):** 0.35% trigger, strikes ATM/±1.
**Thesis & persistence:** Intraday momentum (first-hour → rest-of-day continuation) is documented in index futures; 0DTE verticals cap the reversal tail and lever the move. Same crowding caveat as OPT-041.
**Data class:** **D3/D4 required.**
**Kill/falsifier:** As OPT-014/041 — spread cost at 0DTE vs per-trade edge; falsified if continuation effect post-2023 ≤ costs.
**Cost/capacity:** Daily-ish; hostile.
**Correlation:** High vs OPT-014.
**Prior:** **C**.

### OPT-043 — Expiration-Week Pin/Charm Short Straddle
**Family/Regime:** F9 / CROSS · **Universe:** SPY + 5 megacaps · **DTE/hold:** monthly OpEx week; short straddle at the max-OI strike Wednesday, exit Friday 15:00.
**Params (fixed):** max-OI strike selection, Wed→Fri window.
**Thesis & persistence:** Dealer hedging of large open interest (charm/gamma decay into expiry) historically damps movement near high-OI strikes ("pinning"). Documented in older literature; dealer-positioning inference from OI alone is the weak link.
**Data class:** D2 + **D6 (OI history — an acquisition item)**; strike-level OI must be point-in-time.
**Kill/falsifier:** Post-2020 flow structure (0DTE dominance, weekly expiries diluting monthly OpEx) may have erased it; falsified if max-OI-strike distance behaves no differently expiry week vs control weeks — a clean, cheap pre-test before any P&L work.
**Cost/capacity:** Monthly; fine.
**Correlation:** Moderate vs F1.
**Prior:** **C+**.

## F8 — Relative Value / Dispersion

### OPT-044 — Dispersion-Lite (index vs top components)
**Family/Regime:** F8 / CROSS · **Universe:** SPY vs its top-6 weight names · **DTE/hold:** 30–45 DTE; short SPY ATM straddle vs long weight-proportional component ATM straddles, vega-balanced; hold to 21 DTE; delta-hedge weekly.
**Params (fixed):** top-6 basket, vega-neutral ratio, weekly hedge.
**Thesis & persistence:** Implied correlation (index IV vs component IVs) trades persistently above subsequently realized correlation — the correlation risk premium; index insurance demand vs single-name lottery demand pushes the two IV markets apart structurally.
**Data class:** D2 + D5 across 7 chains; the heaviest data footprint on the slate.
**Kill/falsifier:** Correlation spikes to 1 in crashes — the premium is crash compensation; falsified if post-2022 implied-realized correlation gap ≤ costs of a 14-leg structure. Concentration risk: top-6 ≈ 30%+ of index makes "dispersion" partly a bet on 6 names' idiosyncratic vol.
**Cost/capacity:** Leg-count heavy; the cost screen's flagship F8 test.
**Correlation:** Low vs rest of slate — the genuinely distinct premium here; correlates with OPT-020-minus-OPT-015 by construction.
**Prior:** **B+** — real premium, retail implementability is the question.

### OPT-045 — Same-Sector IV-Percentile Pairs
**Family/Regime:** F8 / CROSS · **Universe:** liquid same-GICS-industry pairs (e.g., within semis, money-center banks) · **DTE/hold:** 30–45 DTE; short 30-day ATM IV of the pair member at its own 1-y IV percentile > 80 vs long the member < 50, delta-hedged, vega-balanced; exit on percentile convergence or 21 DTE.
**Params (fixed):** 80/50 percentile gates.
**Thesis & persistence:** Cross-name relative IV mean-reverts within a sector when the divergence isn't news-driven; the earnings-exclusion rule (no leg within 10 days of either print) removes the dominant legitimate divergence cause.
**Data class:** D2 + D5 + D6 (earnings calendar).
**Kill/falsifier:** Divergences are frequently *right* (idiosyncratic developments); falsified if convergence P&L ≤ 0 after hedging both names' deltas — a heavy operational load for a modest premium.
**Cost/capacity:** 2 names × hedging; moderate-heavy.
**Correlation:** Low vs slate.
**Prior:** **C+**.

### OPT-046 — Index-vs-Single-Name VRP Allocation Switch
**Family/Regime:** F8 / CROSS · **Universe:** SPY vs OPT-020's basket · **DTE/hold:** monthly; allocate the F1 vega budget to whichever VRP estimate (IV − HAR-RV forecast, per OPT-019's machinery) is wider: index or single-name-basket, hysteresis band to limit switching.
**Params (fixed):** shared OPT-019 estimator, 20% hysteresis.
**Thesis & persistence:** The index/single-name VRP gap varies with the correlation premium (F8's own object); allocating toward the wider premium is the portfolio-construction expression of dispersion without the 14-leg structure.
**Data class:** D2 + D5 + D1.
**Kill/falsifier:** Falsified if the switch ≤ static 50/50 (ablation); it's a meta-strategy over OPT-015/020 and dies automatically if either parent dies.
**Cost/capacity:** Inherits parents'.
**Correlation:** By construction ≈ convex combination of OPT-015/020 — counted as ½ effective trial, retained for its portfolio-layer information.
**Prior:** **B−**.

## F10 — Crisis Convexity / Long Vol

### OPT-047 — Rolling Far-OTM Put Ladder (tail program)
**Family/Regime:** F10 / BEAR, UNPREDICTABLE hedge · **Universe:** SPY · **DTE/hold:** always-on: 3-month 0.05Δ puts laddered monthly (3 rungs), fixed 40 bps/month premium budget; monetization rule: sell a rung on VIX > 40 or rung delta > 0.30, immediately re-strike.
**Params (fixed):** 0.05Δ, 40 bps/month budget, monetization triggers.
**Thesis & persistence:** Not an alpha claim — a negative-carry convexity *budget* whose job is book-level: it funds re-risking at the bottom (its P&L must be judged jointly with what it enables, per the pre-registered criterion below). Persistent "cost" because tail insurance is structurally rich — which is exactly why every F1/F3 short-vol candidate exists.
**Data class:** D2 + D5.
**Kill/falsifier:** Pre-registered criterion: judged on portfolio CVaR improvement per bp of drag vs the un-hedged book, NOT standalone Sharpe. Killed if drag exceeds budget or monetization rules never trigger across a full cycle containing a crash.
**Cost/capacity:** Deep-OTM spreads are proportionally wide; budgeted.
**Correlation:** Strongly negative vs F1/F7 blocks and RAMP — the designated book-level anticorrelator.
**Prior:** **B+** as portfolio infrastructure (would be C as standalone alpha; scored on its actual pre-registered job).

### OPT-048 — Vol-Compression Breakout Long Straddle
**Family/Regime:** F10 / UNPREDICTABLE-entry · **Universe:** SPY, QQQ · **DTE/hold:** 30–45 DTE long ATM straddle when 10-day YZ realized vol (1m bars) < 20th trailing-2y percentile AND 20-day price range < 15th percentile; exit +75%/−40% of debit or 10 DTE.
**Params (fixed):** 20/15 percentile gates, exits.
**Thesis & persistence:** Vol clustering implies compression precedes expansion; the claim is that option IV under-adjusts to extreme compression (sellers extrapolate calm). The 1m-based YZ estimator sharpens the compression measurement — same data edge as OPT-019/036.
**Data class:** D2 + D5 + D1-intensive.
**Kill/falsifier:** Long-vol carry bleed if compression persists (2017-style); falsified if IV in compressed states is already at its own percentile floor (i.e., sellers are not extrapolating — measurable pre-test).
**Cost/capacity:** 2 legs, episodic; fine.
**Correlation:** Negative vs F1; sibling of OPT-033/050.
**Prior:** **B**.

### OPT-049 — Gamma Scalping when Forecast RV > IV
**Family/Regime:** F10 / UNPREDICTABLE · **Universe:** SPY · **DTE/hold:** long 30 DTE ATM straddle when OPT-019's estimator flips sign (HAR-RV forecast > IV); delta-hedge on ±0.25σ underlying moves using 1m bars; exit on sign flip or 10 DTE.
**Params (fixed):** shared estimator, 0.25σ hedge bands.
**Thesis & persistence:** The exact mirror of OPT-015/019 — harvests *negative* VRP states. Hedging realizes the gamma; 1m bars make the band-triggered hedge implementable.
**Data class:** D2 + D5 + **D1-intensive for hedging; option marks between hedges can stay D2 (P&L accrues via hedge ledger + terminal value), which is what keeps this researchable — flagged as a subtle but critical feasibility point for the companion doc.**
**Kill/falsifier:** Negative-VRP states are rare and cluster in crashes where hedge slippage explodes; falsified if hedge-ledger P&L ≤ theta paid across the episodic entries; sample starvation is the second falsifier.
**Cost/capacity:** Hedge-frequency-dependent; share hedges cheap, but frequency spikes exactly when spreads widen.
**Correlation:** Strongly negative vs F1 — the designated regime-complement to OPT-015/019.
**Prior:** **B**.

### OPT-050 — Regime-Transition Long Vol
**Family/Regime:** F10 / meta (classifier-native) · **Universe:** SPY · **DTE/hold:** 21–30 DTE long ATM straddle for 5 sessions following any Homeguard classifier transition *into* UNPREDICTABLE or *out of* STRONG_BULL; fixed small vega budget.
**Params (fixed):** transition set, 5-session hold, budget.
**Thesis & persistence:** Trades the classifier itself: transitions concentrate realized vol (regime-change turbulence) while IV adjusts with a lag. Directly reuses existing Homeguard infrastructure — the marginal build cost is the lowest on the slate.
**Data class:** D2 + D5 + D6 (classifier state — internal).
**Kill/falsifier:** If the classifier lags (transitions detected after the vol), the trade buys IV *post*-spike — falsified if entry IV percentile at transitions is already > 80 on average (cheap pre-test on existing classifier history, zero options data needed).
**Cost/capacity:** Episodic; fine.
**Correlation:** Negative vs F1/F7; overlaps OPT-048/033.
**Prior:** **B** — with the noted pre-test as a near-free prior sharpener.

---

# Slate summary — ex-ante prior ranking

**Tier A / A− (highest prior):** OPT-015 (index VRP anchor), OPT-001, OPT-002, OPT-016, OPT-019, OPT-021, OPT-027.
**Tier B+ (strong):** OPT-017, OPT-018, OPT-023, OPT-032, OPT-036, OPT-044, OPT-047.
**Tier B / B− (plausible):** OPT-003–006, OPT-007–009, OPT-011, OPT-013, OPT-020, OPT-026, OPT-028–030, OPT-033–035, OPT-037–039, OPT-046, OPT-048–050.
**Tier C+/C (speculative / expected screen casualties):** OPT-010, OPT-012, OPT-014, OPT-022, OPT-024, OPT-025, OPT-031, OPT-040–043, OPT-045.

Ablation ladders pre-registered inside the slate (structure that buys sample-efficiency, not extra trials): F1 ladder {015→018→019}, F4 ladder {032/033→034}, F6 ladder {016→036}, F8 meta {015/020→046}.

**Effective-independent-trial estimate (diversity screen, manual):** flagged near-duplicates (004, 006, 012, 017, 029, 035, 036, 046 at ½ weight) ⇒ ~46 effective trials against overlapping data. The ledger records the full **+50** regardless — the conservative direction.

**Data-class census (input to the companion feasibility doc):** D2-sufficient: 30 candidates. D2+D6 acquisitions needed (earnings calendar/consensus/OI): 8. D1-intensive (1m underlying is a genuine input edge): 6 (019, 036, 039, 048, 049, plus gates in 005/010). **D3/D4-dependent (intraday options data required): 6 (014, 021*, 025*, 040*, 041, 042; * = marks-quality-sensitive rather than strictly intraday). The vendor question is closed — these resolve on owned disk data, conditional on the quote-population and row-semantics gates (Amendment A1 §5, V1–V3).**

*End of slate v1. No backtests were run. Validation order, GO/NO-GO, and data-acquisition decisions belong to the companion feasibility document.*


---

## Changelog — v1.1 (2026-07-25)

Amended per **Amendment A1** (`2026-07-25_options_docchain_amendment_A1.md`), a data-inventory correction. All changes are driven by data-availability facts discovered after drafting; none is driven by any observed strategy return. No candidate definition, fixed parameter, family assignment, prior tier, falsifier, or ablation ladder changed.

| § | Change |
|---|---|
| Scope intake, assumption 2 | Retired the false premise that options-side data was unresolved and that nothing above EOD chains could be assumed. Replaced with the owned-data record. Data-class tags reaffirmed as requirements, not availability claims |
| OPT-015 | Daily hedge time 15:50 → **15:45**, aligning hedge, marks, and signal truncation to one registered snapshot minute (snapshot symmetry). Structure, DTE band, sizing, falsifier unchanged |
| OPT-025 | T+0 exit 15:50 → **15:45**, same rule. The separately registered T+0 near-close exit amendment stands |
| Data-class census | Vendor question closed; D3/D4 candidates resolve on owned data conditional on verification gates V1–V3 |

**Unchanged and still binding:** all 50 candidate definitions and fixed parameters · the +50 trial-ledger delta · priors A/B/C · kill criteria and falsifiers · the generation-side blindness rule (no candidate here has been backtested) · the note that a large fraction is expected to die in screening, which is an intended outcome.

**Pending decision (registered, not resolved):** OPT-021 carries two exit versions — v1.0 (T+1 open) and v1.1 (T+1 near-close, adopted when the open mark was thought unmeasurable). The open mark is now measurable from owned minute quotes. **Exactly one version may be tested.** Recommendation: restore v1.0, since the amendment away from it was forced by a data limitation that no longer exists and the pure-crush mechanism is cleaner. Decision is Shuyang's and must be logged before any earnings data is touched.

*Companion documents: `2026-07-25_options_slate_feasibility_screen_v2.md` (verdicts) · `2026-07-25_options_slate_cc_handoff_spec_v2.md` (buildable spec) · `2026-07-25_options_docchain_amendment_A1.md` (correction record).*
