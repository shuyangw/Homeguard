# RAMP Options Strategy Catalog

**31 Options Strategies Derived from RAMP Signals — Feasibility Assessment on Alpaca**
**Status**: Research / Planning
**Date**: 2026-03-25

---

## How to Read This Document

Every strategy in this catalog uses one or more outputs from RAMP (the Regime-Aware Momentum Protection strategy) — either the momentum ranking, the regime classification, or the crash protection signals — to inform an options position. Each entry covers the strategy logic, which RAMP signal drives it, the Alpaca options level required, realistic execution mechanics, and known constraints.

Strategies are organized into families by what they are fundamentally trying to do, then ranked within each family by feasibility. Two strategies (#11 short strangles, #18 short straddles) are excluded entirely because they require naked short options, which Alpaca does not allow.

### Alpaca Options Level Reference

| Level | Strategies Allowed |
|-------|--------------------|
| **Level 1** | Covered calls, cash-secured puts |
| **Level 2** | Level 1 + long calls, long puts |
| **Level 3** | Level 1 + 2 + spreads, straddles, iron condors, butterflies, multi-leg |

Paper trading accounts have Level 3 enabled by default. Live accounts require approval per level.

### Key Alpaca Constraints

- **No naked short options**: Uncovered calls and uncovered puts are not permitted. All short options must be covered by shares (covered call), secured by cash (cash-secured put), or paired with a long option in the same multi-leg order.
- **No equity legs in multi-leg orders**: You cannot combine stock and options in a single MLeg order. Strategies like collars must be executed as separate orders.
- **Multi-leg self-coverage rule**: All legs in an MLeg order must be covered within that order. An MLeg with an uncovered short leg is rejected.
- **Rolling constraint**: Rolling a short contract within a multi-leg order creates a temporarily uncovered leg and may be rejected. Close the full spread and reopen instead.
- **Expiration handling**: Alpaca auto-liquidates positions starting at 3:30 PM EST on expiration day if there is insufficient buying power for exercise. ITM positions are auto-exercised if ITM by at least $0.01.
- **No assignment websocket events**: Assignment notifications (OPASN) must be polled via REST API. No real-time push notification.
- **Index options not yet supported**: VIX, SPX, and other index options are not available. Must proxy through ETF options (SPY, UVXY, VXX).

---

## Family 1: Premium Selling

These strategies monetize the volatility risk premium — the persistent tendency for implied volatility to exceed realized volatility. RAMP's regime detector determines *when* the VRP is most favorable (STRONG_BULL) and the momentum ranking determines *which names* to sell premium on.

### #8 — Cash-Secured Puts on Top Momentum Names

**Feasibility Rank: 1 of 31 | Level 1 | Feasible**

**Logic**: Sell OTM puts on stocks ranked in RAMP's top_n during STRONG_BULL regime. Collect premium. If assigned, you acquire a momentum stock at a discount — which RAMP would have bought with equity anyway.

**RAMP Signal Used**: Momentum ranking (top_n list) + regime detection (STRONG_BULL gate) + crash protection (emergency exit trigger).

**Contract Selection**: Puts at delta -0.25 to -0.35, 21-35 DTE, open interest ≥ 100, bid-ask spread ≤ 15% of mid price. Select by maximizing premium collected among qualifying contracts.

**Position Sizing**: Allocate 30% of portfolio to CSP strategy. Maximum 5 concurrent positions at 6% each. Cash required per contract = strike × 100.

**Entry**: Sell put at bid price (conservative fill assumption). Only when regime = STRONG_BULL and crash protection is not active.

**Exit Rules**:
- Buy to close at 50% of premium collected (profit target)
- Buy to close if unrealized loss exceeds 200% of premium (loss limit)
- Buy to close when DTE ≤ 5 (avoid expiration mechanics)
- Immediate close on regime change to BEAR or UNPREDICTABLE
- Immediate close if crash protection triggers (VIX > 25 or SPY drawdown > 5%)
- Close if underlying drops out of RAMP's top_n ranking

**Alpaca Execution**: Single-leg sell-to-open order. Cash-secured put requires full collateral (strike × 100 minus premium) held in cash. Alpaca validates buying power at order time.

**Capital Requirement**: Medium. With a $100,000 portfolio, $30,000 allocated. Each position secures one contract on a $40-60 stock. Stocks above ~$300/share require more than one position allocation per contract.

**Complexity**: Low. One order to open, one to close. No Greeks management beyond monitoring delta drift. Daily check against exit rules at 3:55 PM.

**Edge**: Dual edge — VRP (selling overpriced insurance) + momentum (positive expected forward return on the underlying). Both edges documented academically and both working in your favor simultaneously.

**Key Risk**: Correlated drawdown. All CSP positions are on momentum stocks that may sell off together. Regime gate and crash protection are the primary defenses. Max portfolio allocation of 30% bounds worst-case exposure.

**Full Implementation Plan**: See `RAMP_CSP_IMPLEMENTATION_PLAN.md`.

---

### #9 — Covered Calls on Existing RAMP Equity Positions

**Feasibility Rank: 2 of 31 | Level 1 | Feasible**

**Logic**: Sell OTM calls against stocks already held in the RAMP equity portfolio. Collect premium as income on top of the equity position. The regime determines how aggressive the strike selection is.

**RAMP Signal Used**: Existing equity positions (shares already held) + regime detection (strike distance selection).

**Contract Selection**: Calls at delta 0.20-0.35, 21-35 DTE, matching the underlying shares held by RAMP. Regime adjusts delta target:
- STRONG_BULL: delta 0.20 (further OTM, less likely to cap upside in a strong trend)
- WEAK_BULL: delta 0.30 (closer to ATM, higher premium as upside is fading)
- SIDEWAYS: delta 0.35 (even closer, maximize premium in range-bound market)
- BEAR / UNPREDICTABLE: do not sell covered calls (risk of being locked into a position that's declining)

**Position Sizing**: One contract per 100 shares held. RAMP typically holds positions sized at 5-20% of portfolio depending on regime. A $5,000 RAMP equity position in a $50 stock = 100 shares = 1 covered call contract.

**Entry**: Sell call at bid price. Only when the underlying is in RAMP's current portfolio and regime is STRONG_BULL, WEAK_BULL, or SIDEWAYS.

**Exit Rules**:
- Buy to close at 50% of premium collected
- Buy to close if RAMP generates a sell signal for the underlying (you need the shares free to sell)
- Buy to close when DTE ≤ 5
- Immediate close on regime change to BEAR or UNPREDICTABLE
- If assigned (stock called away): the equity position is closed, record the combined equity + premium P&L

**Alpaca Execution**: Single-leg sell-to-open order. Requires 100 shares of the underlying per contract. Alpaca validates share ownership at order time. If you try to sell a covered call without sufficient shares, the order is rejected.

**Capital Requirement**: Low incremental capital. The equity is already deployed via RAMP. The only cost is if you need to buy back the call at a loss.

**Complexity**: Low. Main operational concern is coordination with RAMP's daily rebalance — if RAMP wants to sell the underlying, you must close the covered call first, which adds a step to the execution flow.

**Edge**: Theta decay on sold calls generates income. In STRONG_BULL, the premium is smaller per trade but assignment risk is lower. In SIDEWAYS, premium is larger and the stock is unlikely to run past the strike.

**Key Risk**: Capping upside in a strong momentum move. If AAPL is your top momentum pick and it surges 10% in a week, your covered call caps your gain at the strike price. The regime-based delta selection mitigates this by using further OTM strikes in STRONG_BULL. Assignment risk: if assigned, you lose the equity position and RAMP would need to re-enter, incurring transaction costs.

---

### #31 — Systematic Covered Call Writing

**Feasibility Rank: 3 of 31 | Level 1 | Feasible**

**Logic**: Extension of #9 applied systematically across all RAMP equity positions rather than selectively. Every stock held by RAMP gets a covered call. This is a "buy-write" overlay strategy.

**RAMP Signal Used**: All current equity positions + regime detection for strike selection.

**Difference from #9**: Strategy #9 is selective (choose which positions to write calls on). #31 writes calls on every position, creating a full portfolio overlay. This generates more premium income but caps upside on the entire equity portfolio.

**Contract Selection**: Same regime-adaptive delta targeting as #9, applied uniformly. In STRONG_BULL (20 positions), this means 20 covered calls.

**Position Sizing**: One contract per 100-share lot held. Fractional lots (less than 100 shares) cannot have covered calls written against them — a limitation at smaller portfolio sizes.

**Alpaca Execution**: Same as #9, but 5-20 separate sell-to-open orders, one per position. Must be sequenced carefully — if any position has fewer than 100 shares, skip it.

**Capital Requirement**: Low incremental. Same as #9.

**Complexity**: Low per trade, but higher operationally due to managing 5-20 concurrent covered calls. Need robust tracking of which calls are open against which equity positions. If RAMP rebalances and wants to sell a stock, the covered call must be closed first.

**Edge**: Same as #9 but amplified across the full portfolio. Backtesting research (BXM index) shows covered call writing on broad portfolios reduces volatility and improves risk-adjusted returns in sideways-to-mildly-bullish environments, at the cost of underperforming in strong bull runs.

**Key Risk**: In STRONG_BULL, the collective call premium may not compensate for the collective upside cap. This is why using delta 0.20 (far OTM) in STRONG_BULL is important — gives the stocks room to run.

---

### #12 — Put Credit Spreads in STRONG_BULL

**Feasibility Rank: 16 of 31 | Level 3 | Feasible**

**Logic**: Sell an OTM put and simultaneously buy a further OTM put on top momentum names during STRONG_BULL. The long put defines your max loss, so this is a defined-risk version of the cash-secured put (#8).

**RAMP Signal Used**: Momentum ranking (top_n) + regime detection (STRONG_BULL gate).

**Contract Selection**: Short leg at delta -0.25 to -0.35, long leg 5-10 points below the short strike. Same expiry, 21-35 DTE. Width of the spread determines max loss.

**Position Sizing**: Max loss per spread = (spread width × 100) - net credit received. Size positions so that max loss per trade ≤ 2% of portfolio.

**Alpaca Execution**: Multi-leg order with `order_class: "mleg"`. Two legs: sell put (higher strike) + buy put (lower strike). Both legs are covered within the order, so MLeg validation passes. Execute as a limit order on the net credit.

**Capital Requirement**: Lower than CSP (#8). Margin requirement is the spread width, not the full strike price. A 5-point wide spread requires ~$500 in margin vs. $5,000+ for a CSP on the same stock.

**Complexity**: Medium. Multi-leg order construction required. Must monitor both legs. Closing requires buying to close the short put and selling to close the long put — ideally as an MLeg close order.

**Edge**: Same VRP + momentum edge as #8, but with defined maximum loss. Less premium collected per trade, but more capital-efficient, allowing more concurrent positions.

**Key Risk**: Defined max loss means you know your worst case. The risk is that a regime change causes many spreads to hit max loss simultaneously. Regime gate and crash protection still apply.

---

### #10 — Iron Condors in SIDEWAYS Regime

**Feasibility Rank: 19 of 31 | Level 3 | Feasible**

**Logic**: Sell a put spread and a call spread on the same underlying when the regime is SIDEWAYS. Profits when the stock stays within a range — which is exactly what SIDEWAYS regime implies.

**RAMP Signal Used**: Regime detection (SIDEWAYS gate) + momentum ranking (select the 5 names RAMP holds in SIDEWAYS, which are the most stable). Alternatively, run on SPY alone for maximum liquidity.

**Contract Selection**: 4 legs:
1. Sell OTM put (delta -0.20 to -0.30)
2. Buy further OTM put (protection leg, 5-10 strikes below)
3. Sell OTM call (delta 0.20 to 0.30)
4. Buy further OTM call (protection leg, 5-10 strikes above)

All same expiry, 21-35 DTE. The sold strikes define the "profit zone" — the range the stock must stay within.

**Position Sizing**: Max loss = wider spread width × 100 - net credit. Size so max loss ≤ 2% of portfolio per iron condor.

**Alpaca Execution**: 4-leg MLeg order. All legs are self-covering (long wings protect short strikes). Alpaca explicitly supports iron condors. Execute as limit order on net credit.

**Capital Requirement**: Medium. Margin is the width of the wider spread. Capital-efficient relative to the premium collected.

**Complexity**: High. Four legs to manage, track, and close. Greeks monitoring is more involved — delta, gamma, and vega all matter. Rolling is complicated by Alpaca's MLeg self-coverage rule.

**Edge**: Theta decay on both sides works in your favor. In SIDEWAYS, realized vol is typically low, so the options you sold lose value faster than the options you bought. Double premium collection (put side + call side).

**Key Risk**: A breakout in either direction. If the regime transitions from SIDEWAYS to STRONG_BULL or BEAR while the condor is open, one side gets tested. The regime gate limits new entries, and existing positions should be closed if regime changes.

---

### #33 — Put Ladder on Momentum Names

**Feasibility Rank: 20 of 31 | Level 1 | Feasible**

**Logic**: Sell puts at multiple strike prices below the current price on the same momentum stock. Creates a "ladder" of entry points. If the stock dips to the first strike, you're assigned at a small discount. If it crashes through all strikes, you accumulate shares at progressively lower prices.

**RAMP Signal Used**: Momentum ranking (highest-conviction names) + regime detection (STRONG_BULL only).

**Contract Selection**: 2-3 puts on the same underlying at different strikes:
- Rung 1: delta -0.25 (modest OTM)
- Rung 2: delta -0.15 (further OTM)
- Rung 3: delta -0.10 (deep OTM, very cheap premium)

Same expiry for all rungs, 21-35 DTE.

**Position Sizing**: Each rung is a separate cash-secured put. Total cash secured across all rungs for one name = sum of (strike × 100) per contract. Very capital-intensive.

**Alpaca Execution**: Three separate single-leg sell-to-open orders. Each is independently cash-secured. Alpaca validates buying power for each.

**Capital Requirement**: High. Three CSPs on one $50 stock requires ~$15,000 in secured cash. Limits this to high-conviction positions with smaller account sizes.

**Complexity**: Medium. Three positions to track per name, but each is a standard CSP with the same exit logic.

**Edge**: Enhanced premium collection. The aggregate premium from three rungs exceeds one CSP. If the stock stays flat, you keep all three premiums. Provides multiple "entry points" if assigned.

**Key Risk**: If the stock crashes through all strikes, you're assigned on all three rungs — accumulating a concentrated position in a falling stock. The capital lockup is substantial.

---

### #13 — Calendar Spreads

**Feasibility Rank: 22 of 31 | Level 3 | Feasible**

**Logic**: Sell a near-term option and buy a longer-term option at the same strike on a momentum stock. Profits from the near-term option decaying faster than the long-term one (differential theta).

**RAMP Signal Used**: Momentum ranking (select the underlying) + regime detection (STRONG_BULL or SIDEWAYS).

**Contract Selection**: Same strike (ATM or slightly OTM), short leg at 21-30 DTE, long leg at 50-65 DTE. Can use puts or calls depending on directional bias.

**Alpaca Execution**: Multi-leg order. Alpaca supports calendar spreads. However, rolling the short leg to a new expiration when it nears expiry creates a temporarily uncovered short — Alpaca may reject this as an MLeg. Workaround: close the entire calendar spread and reopen with new dates.

**Capital Requirement**: Medium. Net debit strategy (you pay the difference between long and short premiums). Max loss is the net debit paid.

**Complexity**: High. Requires monitoring time decay differential, implied volatility across expirations (term structure), and managing the roll at short leg expiration. More Greeks-intensive than single-leg strategies.

**Edge**: Exploits the fact that near-term options decay faster than longer-term options. In a stable market (SIDEWAYS), this decay differential is maximized.

**Key Risk**: A large move in either direction collapses the spread value. Also sensitive to changes in implied volatility term structure — if the market suddenly prices near-term vol higher than long-term vol (inversion), the spread loses value.

---

### #30 — Regime-Timed 0DTE Selling

**Feasibility Rank: 25 of 31 | Level 1 | Feasible with Workaround**

**Logic**: Sell 0DTE (zero days to expiration) cash-secured puts or covered calls on momentum names at 3:55 PM, collecting the last bits of theta decay before market close.

**RAMP Signal Used**: Momentum ranking + regime detection (STRONG_BULL only, where VRP is widest).

**The Workaround**: Alpaca begins auto-evaluating expiring positions at 3:30 PM EST and may liquidate them before RAMP's 3:55 PM execution window. This creates a timing conflict. To work around this, shift execution to 3:25 PM (before Alpaca's evaluation) and accept that positions expire 35 minutes later. Alternatively, use 1DTE options (expiring the next trading day) to avoid the same-day liquidation entirely.

**Alpaca Execution**: Single-leg orders (CSP or covered call). Technically Level 1. The challenge is purely operational — the timing conflict with auto-liquidation.

**Capital Requirement**: Medium. Cash-secured puts still require full collateral even for 0DTE.

**Complexity**: High. Extremely time-sensitive execution. Any delay means the option expires worthless before you can sell it (if buying) or exposes you to last-minute assignment (if selling).

**Edge**: 0DTE options have the highest theta decay rate of any expiration. Premium collected relative to time at risk is maximized.

**Key Risk**: Gamma is at its maximum on expiration day. Small moves in the underlying cause large changes in option value. A sudden adverse move in the last 30 minutes can cause outsized losses. Additionally, the theoretical tail risk of 0DTE selling has been documented as severe — losses when the regime detector is slightly late can violate the 10% max drawdown constraint.

---

## Family 2: Directional Strategies

These replace equity positions with options that express the same directional view from RAMP's momentum signal, providing leverage, defined risk, or capital efficiency.

### #1 — Long Calls on Top Momentum Names

**Feasibility Rank: 6 of 31 | Level 2 | Feasible**

**Logic**: Instead of buying stock, buy call options on RAMP's top_n names. Leveraged upside with a defined maximum loss (the premium paid).

**RAMP Signal Used**: Momentum ranking (top_n stocks) + regime detection (entry timing).

**Contract Selection**: Calls at delta 0.50-0.70 (ATM to slightly ITM), 30-60 DTE. Higher delta = more equity-like behavior. Lower delta = more leverage but higher breakeven.

**Position Sizing**: Premium paid per position ≤ 3-5% of portfolio. Unlike equity, you can only lose the premium — so a 5% allocation has a defined 5% max loss. This is more capital-efficient than holding $5,000 in stock.

**Entry**: Buy call at ask price. Enter when momentum signal is strong and regime supports directional trades (STRONG_BULL, WEAK_BULL).

**Exit Rules**:
- Sell when RAMP would sell the equity position (momentum rank drops)
- Sell when DTE ≤ 7 (avoid accelerating theta decay)
- Sell on regime change to BEAR or UNPREDICTABLE
- Take profit at 100% of premium paid (double your money)

**Alpaca Execution**: Single-leg buy-to-open. Simplest possible options order.

**Capital Requirement**: Low. Premium is the only capital at risk. Typically 3-8% of the stock's price for a 30-60 DTE ATM call.

**Complexity**: Low. One order in, one order out. Monitor position value and exit triggers.

**Edge**: Leveraged exposure to RAMP's momentum signal. If RAMP's momentum ranking generates alpha (which the 0.846 OOS Sharpe suggests it does), capturing that alpha with 3-8% capital at risk per position is highly efficient.

**Key Risk**: Time decay. Unlike equity, options lose value every day even if the stock is flat. If the momentum move is slow, theta eats into your position. This makes DTE selection critical — too short and theta kills you, too long and premiums are expensive.

---

### #3 — Long Puts on Bottom Momentum Names

**Feasibility Rank: 7 of 31 | Level 2 | Feasible**

**Logic**: Buy puts on the *worst* ranked stocks in RAMP's momentum scoring. These are stocks with weak or negative momentum — short candidates.

**RAMP Signal Used**: Momentum ranking (bottom_n stocks — the mirror of top_n) + regime detection (BEAR or WEAK_BULL for entries).

**Contract Selection**: Puts at delta -0.50 to -0.70 (ATM to slightly ITM), 30-60 DTE.

**Position Sizing**: Premium paid ≤ 3-5% of portfolio per position. Maximum 3-5 concurrent positions.

**Entry**: Buy put at ask price. Enter when regime is BEAR (strongest case for shorts) or WEAK_BULL (momentum is fading).

**Exit Rules**:
- Sell when the stock's momentum rank improves (no longer bottom_n)
- Sell when DTE ≤ 7
- Sell on regime change to STRONG_BULL
- Take profit at 100% of premium paid

**Alpaca Execution**: Single-leg buy-to-open.

**Capital Requirement**: Low. Premium only.

**Complexity**: Low operationally. Conceptually requires implementing bottom_n ranking alongside RAMP's existing top_n — the scoring is already computed, just select the lowest-ranked instead of highest.

**Edge**: Cross-sectional momentum is well-documented as a long/short phenomenon — weak momentum stocks underperform just as strong momentum stocks outperform. Buying puts on weak names captures the short side without needing to borrow shares.

**Key Risk**: Short squeezes. The weakest momentum stocks occasionally experience violent reversals. Defined risk (premium only) is the main protection. Also, implied volatility on weak stocks tends to be elevated, making puts expensive — the breakeven move is larger.

---

### #6 — Deep ITM Call Replacement

**Feasibility Rank: 8 of 31 | Level 2 | Feasible**

**Logic**: Instead of holding stock, hold deep in-the-money calls (delta ~0.80) on RAMP's top momentum names. This closely replicates equity exposure but uses less capital, freeing cash for other strategies.

**RAMP Signal Used**: Momentum ranking (top_n) + regime detection (same rules as equity RAMP).

**Contract Selection**: Calls at delta 0.75-0.85, 60-90 DTE. Deep ITM means the call behaves almost like stock (high delta) but has some downside protection because your max loss is the premium, not the full stock price.

**Position Sizing**: The premium for a deep ITM call is typically 70-85% of the stock price. This saves 15-30% capital compared to holding stock outright. On a $100,000 RAMP portfolio, this frees roughly $15,000-30,000 for other strategies (like #8 CSPs).

**Entry**: Buy deep ITM call at ask price. This replaces an equity buy in RAMP's rebalance flow.

**Exit Rules**: Same as RAMP equity — sell when the stock drops out of the top_n ranking or regime changes.

**Alpaca Execution**: Single-leg buy-to-open. The call is deep enough ITM that it has minimal bid-ask spread issues on liquid S&P 500 names.

**Capital Requirement**: Medium. 70-85% of what equity would require, but frees the remainder for other strategies.

**Complexity**: Low. Behaves almost identically to equity. Main difference is managing expiration — roll to a new expiry before DTE gets short.

**Edge**: Capital efficiency. Same momentum exposure with less capital deployed. The freed capital earns additional return if deployed in #8 or #9.

**Key Risk**: Time decay, even on deep ITM calls. If the stock is flat for 60 days, you lose the time value component of the premium (roughly 2-5% of position value). Also, early assignment risk on deep ITM American-style calls, especially near ex-dividend dates.

---

### #2 — Bull Call Spreads on Top Momentum Names

**Feasibility Rank: 17 of 31 | Level 3 | Feasible**

**Logic**: Buy a call and sell a higher-strike call on top_n momentum names. Defined risk, defined reward. Lower cost than a naked long call (#1) but caps your upside.

**RAMP Signal Used**: Momentum ranking (top_n) + regime detection (STRONG_BULL, WEAK_BULL).

**Contract Selection**: Long call at delta 0.50-0.60 (ATM), short call 5-10 strikes higher. Same expiry, 30-45 DTE.

**Position Sizing**: Max loss = net debit paid. Max profit = spread width - net debit. Size so max loss ≤ 2% of portfolio.

**Alpaca Execution**: 2-leg MLeg order. Buy call + sell call. Self-covering.

**Capital Requirement**: Low. Cheaper than naked calls since the short call partially offsets the long call cost.

**Complexity**: Medium. Multi-leg order construction and close management.

**Edge**: Better risk/reward ratio than naked calls when you have a modest price target. If RAMP's momentum signal implies "stock will continue rising but not explosively," a bull call spread captures that view more efficiently.

**Key Risk**: Upside is capped. If the stock surges far past the short call strike, you miss the excess gain. In STRONG_BULL, this cap can be painful.

---

### #4 — Bear Put Spreads on Bottom Momentum Names

**Feasibility Rank: 18 of 31 | Level 3 | Feasible**

**Logic**: Buy a put and sell a lower-strike put on bottom_n momentum names. Defined-risk short expression.

**RAMP Signal Used**: Momentum ranking (bottom_n) + regime detection (BEAR, WEAK_BULL).

**Contract Selection**: Long put at delta -0.50 to -0.60, short put 5-10 strikes lower. Same expiry, 30-45 DTE.

**Position Sizing**: Max loss = net debit paid. Size so max loss ≤ 2% of portfolio.

**Alpaca Execution**: 2-leg MLeg order. Buy put + sell put. Self-covering.

**Capital Requirement**: Low. Cheaper than naked puts.

**Complexity**: Medium. Same as #2 but in the opposite direction.

**Edge**: Defined-risk short expression. Cheaper than buying puts outright, with a defined max loss.

**Key Risk**: If the stock doesn't fall enough, the net debit is lost. Also, bear put spreads have a capped profit, so a true crash isn't fully captured.

---

### #5 — Long/Short Options Spread (Market Neutral)

**Feasibility Rank: 14 of 31 | Level 2 | Feasible**

**Logic**: Buy calls on top_n momentum names and simultaneously buy puts on bottom_n names. This creates a market-neutral momentum portfolio expressed through options.

**RAMP Signal Used**: Momentum ranking (both top_n and bottom_n) + regime detection.

**Contract Selection**: Long calls (delta 0.50) on top_n. Long puts (delta -0.50) on bottom_n. 30-60 DTE. All are buy-to-open — no short options at all.

**Position Sizing**: Equal capital on each side. Premium on the call side roughly matches premium on the put side for market neutrality.

**Alpaca Execution**: Separate single-leg orders. No multi-leg needed since all positions are long.

**Capital Requirement**: Medium. Double the premium outlay of a long-only approach since you're buying both calls and puts.

**Complexity**: Medium. Two sets of positions to manage (long calls and long puts). Must rebalance when RAMP's rankings change.

**Edge**: Market-neutral momentum capture. If the market goes up, long calls gain more than long puts lose (because top momentum outperforms bottom momentum). If the market goes down, long puts gain more than long calls lose. The edge is in the spread between top and bottom momentum, not market direction.

**Key Risk**: Double theta decay — both calls and puts lose value daily. The momentum spread must be wide enough to overcome the combined time decay. This is the fundamental tension of options-based market-neutral strategies.

---

### #7 — Synthetic Long (Long Call + Short Put)

**Feasibility Rank: 26 of 31 | Level 1+2 | Feasible with Workaround**

**Logic**: Buy a call and sell a cash-secured put at the same strike to replicate stock ownership. Used when you want leverage or can't buy fractional shares.

**RAMP Signal Used**: Momentum ranking (top_n).

**Contract Selection**: Both at the same strike (ATM), same expiry. The premium dynamics roughly cancel — you receive premium on the put and pay premium on the call.

**The Workaround**: Alpaca doesn't allow combining equity and options in one order, and the short put must be cash-secured. This means you need the full strike × 100 in cash to secure the put, plus additional cash to buy the call. The capital efficiency advantage largely disappears compared to just buying the stock. The strategy is technically possible but defeats its own purpose at typical portfolio sizes.

**Alpaca Execution**: Two separate orders — buy-to-open call (Level 2), sell-to-open cash-secured put (Level 1).

**Capital Requirement**: Very high. Strike × 100 in cash for the put, plus call premium. More capital than owning the stock in most cases.

**Complexity**: Medium. Two positions to manage for one directional view.

**Edge**: In theory, synthetic longs have lower cost-of-carry than equity in some rate environments. In practice, the cash-securing requirement on Alpaca eliminates this advantage.

**Key Risk**: Full directional exposure (same as stock) plus the operational complexity of two positions.

---

## Family 3: Volatility Trading

These strategies trade volatility itself rather than direction. RAMP's regime transitions are the primary signal — regime changes indicate shifts in the volatility risk premium.

### #15 — Long Straddles in UNPREDICTABLE Regime

**Feasibility Rank: 10 of 31 | Level 2 or Level 3 | Feasible**

**Logic**: Buy an ATM call and an ATM put on the same underlying when RAMP classifies the market as UNPREDICTABLE. Profits from a large move in either direction. The thesis is that UNPREDICTABLE regimes precede resolution events where realized vol exceeds implied vol.

**RAMP Signal Used**: Regime detection (UNPREDICTABLE trigger) + optionally momentum ranking (select the most volatile names within the UNPREDICTABLE regime).

**Contract Selection**: ATM call + ATM put, same strike, same expiry. 21-45 DTE to give the move time to develop.

**Position Sizing**: Total premium paid ≤ 3-5% of portfolio. Straddles are expensive — ATM call + ATM put on a $100 stock at 30 DTE might cost $8-12 per share ($800-1200 per straddle). Scale accordingly.

**Alpaca Execution**: Either as a 2-leg MLeg order (buy call + buy put, Level 3) or as two separate single-leg buy-to-open orders (Level 2). Both approaches work since all positions are long.

**Capital Requirement**: Medium. The combined premium is substantial, but the payoff is convex — gains accelerate the larger the move.

**Complexity**: Medium. Must manage two positions and decide when to close. Can sell the profitable side and hold the other, or close both together. Requires a view on whether the move is complete or ongoing.

**Edge**: In UNPREDICTABLE regimes, the market is confused about direction but tends to underprice magnitude. Realized vol during regime transitions historically exceeds implied vol by a wide margin. This is a direct bet on that excess.

**Key Risk**: Time decay is fierce on straddles. If the expected resolution doesn't happen within the DTE window, both sides decay. UNPREDICTABLE regimes are also relatively rare, so this generates few trades per year.

---

### #16 — Long Strangles Entering UNPREDICTABLE

**Feasibility Rank: 11 of 31 | Level 2 or Level 3 | Feasible**

**Logic**: Same thesis as #15 but with OTM strikes instead of ATM. Cheaper entry, wider breakevens, but higher percentage gains if a large move occurs.

**RAMP Signal Used**: Regime detection (UNPREDICTABLE trigger).

**Contract Selection**: OTM call (delta 0.25-0.30) + OTM put (delta -0.25 to -0.30), same expiry, 21-45 DTE.

**Position Sizing**: Total premium ≤ 2-3% of portfolio. Strangles are cheaper than straddles, allowing slightly more sizing.

**Alpaca Execution**: Same as #15 — either MLeg (Level 3) or two single-leg buys (Level 2).

**Capital Requirement**: Low. OTM options are cheaper than ATM.

**Complexity**: Medium. Same management as #15.

**Edge**: Cheaper entry means lower breakeven in percentage terms. If the expected vol expansion is large, strangles have higher percentage returns than straddles.

**Key Risk**: Wider breakevens mean a moderate move still results in a loss. The underlying must move significantly past either OTM strike to profit. If the UNPREDICTABLE regime resolves with a moderate drift rather than a sharp move, both sides expire worthless.

---

### #19 — Volatility Regime Switching (SPY)

**Feasibility Rank: 28 of 31 | Level 3 | Feasible with Workaround**

**Logic**: Systematically go long vol in UNPREDICTABLE/BEAR (buy straddles/strangles) and short vol in STRONG_BULL/SIDEWAYS (sell premium). The regime detector is the sole signal. Trade SPY only for maximum liquidity.

**RAMP Signal Used**: Regime detection (primary and only signal). No momentum ranking needed.

**The Workaround**: The short-vol side of this strategy is constrained by Alpaca's no-naked-short rule. You cannot sell strangles or straddles directly. Instead, you must use iron condors (which have defined risk from the long wings) for the short-vol expression. This reduces premium collected and adds two extra legs, but is functionally similar. The long-vol side (buy straddles/strangles) is unaffected.

**Regime-to-Action Mapping**:
- STRONG_BULL: Sell iron condors on SPY (short vol, defined risk)
- SIDEWAYS: Sell iron condors on SPY (short vol, narrower wings)
- UNPREDICTABLE: Buy straddles on SPY (long vol)
- BEAR: Buy puts on SPY (directional + long vol)
- WEAK_BULL: Flat / small iron condors

**Alpaca Execution**: Iron condors via 4-leg MLeg orders. Straddles via 2-leg MLeg or two single-leg orders. All feasible.

**Capital Requirement**: Medium. Iron condor margin is spread width. Straddle cost is premium paid.

**Complexity**: High. Multiple strategy expressions depending on regime. Transition logic is the challenge — when the regime changes, you may need to close one type of position and open a completely different one.

**Edge**: Pure regime timing of the VRP. Academic research shows the VRP is earned almost entirely in calm periods and entirely lost in crises. RAMP's regime detector is essentially a filter for when to collect vs. when to buy.

**Key Risk**: Regime misclassification. If the detector says STRONG_BULL but we're actually entering a crash, the iron condor gets steamrolled. The crash protection triggers (VIX > 25, SPY DD > 5%) provide a secondary defense.

---

### #17 — VIX Call Spreads When Crash Protection Triggers

**Feasibility Rank: 30 of 31 | Level 3 | Feasible with Workaround**

**Logic**: When RAMP's crash protection triggers (VIX > 25 or SPY drawdown > 5%), buy call spreads on VIX to directly profit from further volatility expansion.

**RAMP Signal Used**: Crash protection signal (VIX threshold + SPY drawdown).

**The Workaround**: Alpaca does not yet support index options (VIX, SPX). You must proxy VIX exposure through ETF options on UVXY, VXX, or similar volatility ETFs. These are imperfect proxies — they suffer from contango drag, and their options have wider bid-ask spreads than VIX options. The trade's P&L profile is similar but not identical to actual VIX options.

**Contract Selection**: Bull call spread on UVXY — buy ATM call, sell OTM call. 14-30 DTE. Tight spread (3-5 points) to limit cost.

**Alpaca Execution**: 2-leg MLeg order on UVXY options.

**Capital Requirement**: Medium. Net debit paid is the max loss.

**Complexity**: High. Requires understanding VIX ETF dynamics, term structure contango/backwardation effects, and the decay characteristics of leveraged volatility products.

**Edge**: Directly monetizes the crash protection signal. If RAMP's VIX > 25 trigger is accurate, VIX typically continues rising (vol clustering). The call spread profits from this continuation.

**Key Risk**: The VIX ETF proxy may diverge from VIX itself. Also, if VIX spikes and immediately mean-reverts (a false alarm), the call spread expires worthless.

---

### #29 — Gamma Scalping with Momentum Bias

**Feasibility Rank: 27 of 31 | Level 2 | Feasible with Workaround**

**Logic**: Buy options (calls or puts) on momentum names, then delta-hedge continuously with equity. Profits when realized vol exceeds implied vol (the option is underpriced).

**RAMP Signal Used**: Momentum ranking (select underlyings) + regime detection (UNPREDICTABLE for long gamma).

**The Workaround**: Gamma scalping requires intraday delta rebalancing — typically every 30-60 minutes. RAMP's execution cadence is once daily at 3:55 PM, which is fundamentally incompatible. To make this work, you'd need to build a separate intraday execution loop with real-time price monitoring. This is a significant infrastructure addition beyond what RAMP currently supports.

**Alpaca Execution**: Buy-to-open options (Level 2) + equity hedging via standard stock orders. Technically feasible as separate orders, but the frequency and automation requirements are demanding.

**Capital Requirement**: High. Need both the option premium and capital for the equity hedge.

**Complexity**: Very high. This is a professional market-maker strategy requiring real-time Greeks calculation, continuous position monitoring, and high-frequency execution.

**Edge**: When the regime suggests vol is underpriced (UNPREDICTABLE), buying options and scalping gamma captures the excess realized vol.

**Key Risk**: If realized vol is less than implied vol, you lose on every hedge adjustment (buy high, sell low). Transaction costs from frequent rebalancing can eat the edge. Requires a very accurate vol forecast.

---

### #20 — Dispersion Trading

**Feasibility Rank: 31 of 31 | Level 3 | Feasible with Workaround**

**Logic**: Sell options on SPY (the index) and buy options on individual S&P 500 stocks. Profits when the correlation between stocks is lower than what the index options price implies — stocks move independently rather than in lockstep.

**RAMP Signal Used**: Regime detection (STRONG_BULL has lower correlations, favoring dispersion) + momentum ranking (select which single-stock options to buy).

**The Workaround**: The short SPY leg must be a defined-risk structure (iron condor or credit spread) since naked shorts are not allowed. The long single-stock legs are straightforward buy-to-open. However, properly sizing the trade requires portfolio-level Greeks management (aggregate delta, vega, gamma across 10-20 positions), which is a significant analytical overhead. Additionally, capital requirements for running both sides across many names may exceed typical portfolio sizes.

**Alpaca Execution**: Iron condor or credit spread on SPY (MLeg, Level 3) + long straddles/strangles on individual names (Level 2). All technically feasible as separate orders.

**Capital Requirement**: Very high. Multiple positions across many names plus the SPY short-vol structure.

**Complexity**: Very high. This is an institutional strategy. Requires real-time correlation monitoring, precise vega matching between index and single-stock positions, and sophisticated position management.

**Edge**: Dispersion is a well-documented institutional premium. Index vol is systematically overpriced relative to single-stock vol because of portfolio hedging demand.

**Key Risk**: Correlation spikes in stress events. If a crisis causes all stocks to move together (correlation → 1.0), the short index vol position doesn't lose enough to offset the collapse of the long single-stock positions.

---

## Family 4: Portfolio Protection

These strategies replace or enhance RAMP's existing crash protection mechanism (which currently just reduces equity exposure to 50%) with more nuanced hedging.

### #21 — Portfolio Puts When VIX > 25

**Feasibility Rank: 5 of 31 | Level 2 | Feasible**

**Logic**: When crash protection triggers, buy SPY puts instead of selling half the equity portfolio. This maintains the momentum equity positions while capping downside.

**RAMP Signal Used**: Crash protection signal (VIX > 25 or SPY drawdown > 5%).

**Contract Selection**: SPY puts at delta -0.30 to -0.50, 14-30 DTE. Size to cover approximately 50% of portfolio delta (matching the current 50% exposure reduction).

**Position Sizing**: Hedge notional = 50% of portfolio value. Number of contracts = hedge notional / (SPY price × 100). Premium paid = cost of the hedge.

**Entry**: Buy put at ask price. Triggered by crash protection signal, not a daily rebalance decision.

**Exit Rules**:
- Sell put when crash protection signal clears (VIX drops below 25 AND SPY drawdown recovers)
- Sell when DTE ≤ 5 and roll to new expiration if protection still needed
- Take profit if SPY drops significantly (the puts appreciate in value)

**Alpaca Execution**: Single-leg buy-to-open on SPY puts.

**Capital Requirement**: Low. Premium is typically 2-4% of portfolio value for 30-day ATM SPY puts. This is the "cost of insurance."

**Complexity**: Low. One position to manage. The main decision is sizing — how many contracts to buy to match the desired hedge ratio.

**Edge**: Preserves RAMP equity positions during turbulence. The current approach (sell half) forces you to buy back later at potentially higher prices. Portfolio puts let you stay invested while capping downside. If the crash doesn't materialize, you lose only the premium.

**Key Risk**: The premium cost is a drag on returns if crash protection triggers frequently. In choppy markets where VIX oscillates around 25, you may buy puts repeatedly and lose premium each time.

---

### #22 — Collars on Equity Positions

**Feasibility Rank: 12 of 31 | Level 1+2 | Feasible**

**Logic**: When crash protection triggers, collar each RAMP equity position by buying a put (downside protection) and selling a covered call (partially funds the put). This is a "costless" or "low-cost" hedge.

**RAMP Signal Used**: Crash protection signal + existing equity positions.

**Contract Selection**: For each equity position held by RAMP:
- Buy put at delta -0.30 to -0.40, 21-30 DTE (downside protection)
- Sell covered call at delta 0.20 to 0.30, same expiry (fund the put)

The call premium partially or fully offsets the put cost. When VIX is elevated (which it is when crash protection triggers), call premiums are also elevated — making the collar cheaper or free.

**Position Sizing**: One put and one call per 100-share lot held.

**Alpaca Execution**: Two separate single-leg orders per position — buy-to-open put (Level 2) and sell-to-open covered call (Level 1). Cannot be combined in an MLeg order because Alpaca doesn't support equity legs in MLeg. Execution risk: the stock could move between placing the two orders. Mitigated by placing both in rapid succession at 3:55 PM.

**Capital Requirement**: Low. The call premium offsets the put cost. Net cost may be near zero ("zero-cost collar") or a small net debit.

**Complexity**: Medium. Two orders per position. If RAMP holds 10-20 positions, this means 20-40 orders to collar the portfolio. Managing these positions adds operational overhead — each collar must be closed before RAMP can sell the underlying.

**Edge**: Maintains exposure to momentum stocks while defining max downside. Superior to selling half the portfolio because you keep the positions that may recover first.

**Key Risk**: Upside is capped by the short call. If the "crash" was a false alarm and stocks rally, the collar caps your recovery. Also, managing 10-20 collars operationally is significant.

---

### #25 — Dynamic Collar Width by Regime

**Feasibility Rank: 13 of 31 | Level 1+2 | Feasible**

**Logic**: Same as #22 but the collar strike selection adapts based on regime and severity of the protection trigger.

**RAMP Signal Used**: Crash protection signal + regime detection + equity positions.

**Regime-Adaptive Parameters**:
- WEAK_BULL (early warning): Wide collar — put at delta -0.20, call at delta 0.15. Cheap insurance with minimal upside cap.
- BEAR (confirmed downturn): Tight collar — put at delta -0.40, call at delta 0.35. Maximum protection, most upside cap.
- UNPREDICTABLE (uncertain): Medium collar — put at delta -0.30, call at delta 0.25. Balanced.

**Alpaca Execution**: Same as #22 — separate single-leg orders.

**Capital Requirement**: Low. Same as #22.

**Complexity**: Medium. Same as #22 plus the regime-dependent parameter selection.

**Edge**: More nuanced than a one-size-fits-all collar. In early-warning regimes, provides cheap protection. In confirmed downturns, provides maximum protection.

**Key Risk**: Same as #22. The added parameter selection risk is that the wrong regime classification leads to the wrong collar width.

---

### #24 — Tail Risk Hedging

**Feasibility Rank: 9 of 31 | Level 2 | Feasible**

**Logic**: Continuously hold small positions in far OTM puts on SPY as insurance against catastrophic market events. Size the hedge based on regime — larger in WEAK_BULL (early warning), smaller in STRONG_BULL (low immediate risk).

**RAMP Signal Used**: Regime detection (sizing) + crash protection (not a trigger — the hedge is always on, just sized differently).

**Contract Selection**: SPY puts at delta -0.10 to -0.15 (deep OTM, very cheap), 30-60 DTE. These only pay off in a severe crash (SPY down 10%+). Roll monthly.

**Regime-Adaptive Sizing**:
- STRONG_BULL: 0.25% of portfolio per month on puts (minimal, insurance level)
- WEAK_BULL: 0.50% of portfolio per month (elevated risk)
- SIDEWAYS: 0.25% (low risk of crash, low cost)
- UNPREDICTABLE: 0.75% (elevated risk, higher allocation)
- BEAR: 1.0% (maximum tail hedge)

**Entry**: Buy put at ask price. Monthly roll — close current puts and open new ones with fresh DTE.

**Alpaca Execution**: Single-leg buy-to-open. Roll on a fixed schedule (e.g., third Friday of each month).

**Capital Requirement**: Low. 0.25-1.0% of portfolio per month, or 3-12% annualized. This is the "insurance premium."

**Complexity**: Low. One position to manage, monthly roll cycle.

**Edge**: Convex payoff. In a normal year, you lose 3-12% of the portfolio to put premiums. In a crash year (2008, 2020), the puts can return 500-2000% of their cost, dramatically reducing portfolio drawdown. This is portfolio insurance, not an alpha source.

**Key Risk**: The cost is a persistent drag in non-crash years, which reduces portfolio Sharpe ratio. The key question is whether the tail protection is worth the premium cost. Backtesting across multiple cycles is essential.

---

### #23 — Put Ratio Backspread

**Feasibility Rank: 21 of 31 | Level 3 | Feasible**

**Logic**: Sell 1 ATM put and buy 2 OTM puts on SPY. This creates a position that loses a small amount if SPY dips modestly but profits enormously if SPY crashes. The sold ATM put partially funds the two purchased OTM puts.

**RAMP Signal Used**: Crash protection signal (trigger) or continuous tail hedge.

**Contract Selection**: Sell 1 SPY put at delta -0.50 (ATM), buy 2 SPY puts at delta -0.25 to -0.30 (OTM). Same expiry, 21-45 DTE.

**Position Sizing**: Net debit should be ≤ 0.5% of portfolio. The credit from the sold put significantly reduces cost.

**Alpaca Execution**: 3-leg MLeg order. The sold ATM put is covered by the two long OTM puts (net long). Alpaca's self-coverage rule should be satisfied.

**Capital Requirement**: Low. Small net debit or possibly net zero.

**Complexity**: High. Three legs. The P&L profile is nonlinear — slightly negative for small drops, then sharply positive for large drops. This requires understanding the payoff diagram and managing the position accordingly.

**Edge**: Cheap crash protection with convex payoff. In a moderate decline, you lose a small amount. In a severe decline, the two long puts dominate and generate large gains. The cost is that small dips are mildly negative.

**Key Risk**: The "valley of death" — a moderate decline (e.g., SPY down 5-8%) hits the sweet spot where the short put loses but the long puts haven't gained enough. Also, early assignment on the short ATM put is a real possibility.

---

## Family 5: Hybrid / Multi-Signal

These strategies combine multiple RAMP outputs — momentum ranking, regime detection, and crash protection — or chain multiple options structures into a lifecycle.

### #27 — Regime-Adaptive Wheel Strategy

**Feasibility Rank: 4 of 31 | Level 1 | Feasible**

**Logic**: A continuous cycle that rotates between cash-secured puts and covered calls based on regime:

1. **STRONG_BULL**: Sell cash-secured puts on top momentum names (collect premium while waiting to buy). If assigned → hold shares.
2. **WEAK_BULL / SIDEWAYS**: Sell covered calls on assigned shares (collect premium on shares you hold). If called away → back to cash.
3. **BEAR / UNPREDICTABLE**: No new positions. Close existing if stops are hit. Wait for regime to improve.

**RAMP Signal Used**: Momentum ranking (name selection) + regime detection (phase selection) + crash protection (emergency exit).

**The Lifecycle**:
```
Cash → Sell CSP → [Assigned?] → Hold Stock → Sell CC → [Called Away?] → Cash → ...
         │ (No)                       │ (No)
         └→ Keep premium,             └→ Keep premium,
            sell new CSP                 sell new CC
```

**Position Sizing**: Same as #8 — 30% of portfolio, 5 concurrent wheels maximum.

**Alpaca Execution**: All Level 1. CSPs are single-leg sell-to-open. Covered calls are single-leg sell-to-open (when holding assigned shares). Assignment handling via NTA polling.

**Capital Requirement**: Medium. Cash-secured puts require strike × 100 in cash. Once assigned, that cash becomes equity.

**Complexity**: Medium. The strategy itself is simple (two alternating single-leg positions), but the lifecycle management adds logic — tracking whether you're in the "selling puts" or "selling calls" phase for each name, detecting assignment, and handling the transition.

**Edge**: Compounds the premium from both sides of the wheel. In STRONG_BULL, you collect put premium and often keep it (stocks go up, puts expire worthless). If assigned, you transition to collecting call premium on a momentum stock. The regime gate ensures you're always doing the regime-appropriate action.

**Key Risk**: Being assigned in a declining market (the regime changed between when you sold the put and when it expired). The regime gate mitigates this, but assignment can happen overnight before the next day's regime check.

---

### #26 — Momentum-Weighted Risk Reversal

**Feasibility Rank: 24 of 31 | Level 1+2 | Feasible**

**Logic**: Buy calls on top momentum names and sell cash-secured puts on the same or different names. The call side captures upside momentum. The put side collects premium that partially funds the calls. Regime determines the weighting.

**RAMP Signal Used**: Momentum ranking (top_n for calls, next-best for puts) + regime detection.

**Regime Weighting**:
- STRONG_BULL: Heavier call allocation (more upside capture), lighter put allocation
- WEAK_BULL: Even allocation
- SIDEWAYS: Heavier put allocation (more premium collection), lighter call allocation

**Position Sizing**: Total allocation ≤ 30% of portfolio split between calls and puts.

**Alpaca Execution**: Separate single-leg orders — buy-to-open calls (Level 2) and sell-to-open cash-secured puts (Level 1).

**Capital Requirement**: High. Call premium plus full cash collateral for puts.

**Complexity**: High. Two sets of positions with different management rules. Calls are momentum-driven (exit when rank drops). Puts follow CSP exit rules (profit target, loss limit, regime gate).

**Edge**: Leveraged upside (calls) + income generation (puts). If both RAMP's momentum signal and regime signal are accurate, both sides generate positive returns.

**Key Risk**: In a sharp downturn, calls lose value and puts may be assigned. Both sides lose simultaneously. The regime gate is critical to prevent this.

---

### #28 — Pairs Options (Sector Neutral Momentum)

**Feasibility Rank: 15 of 31 | Level 2 | Feasible**

**Logic**: Within each sector, buy calls on the highest-momentum stock and buy puts on the lowest-momentum stock. This creates sector-neutral momentum exposure through options.

**RAMP Signal Used**: Momentum ranking (within sectors — requires sector mapping of S&P 500 constituents) + regime detection.

**Example**: In the Technology sector, if NVDA has the highest momentum and INTC has the lowest:
- Buy NVDA calls (delta 0.50, 30-45 DTE)
- Buy INTC puts (delta -0.50, 30-45 DTE)

**Position Sizing**: Equal premium on each side per sector pair. Maximum 5-10 sector pairs active.

**Alpaca Execution**: All single-leg buy-to-open orders. No multi-leg needed. No short options.

**Capital Requirement**: Medium. All positions are long options — premium is the only capital at risk.

**Complexity**: Medium. Requires sector classification of the S&P 500 (readily available data). Managing 10-20 positions (5-10 pairs × 2 legs).

**Edge**: Isolates the momentum factor by removing sector and market beta exposure. If RAMP's momentum ranking is effective within sectors, this captures the intra-sector spread without market direction risk.

**Key Risk**: Within a sector, the top and bottom names can both move in the same direction. If the whole sector rallies, NVDA and INTC both go up — the put on INTC loses value. Sector-level momentum can overwhelm within-sector momentum.

---

### #32 — Diagonal Spreads

**Feasibility Rank: 23 of 31 | Level 3 | Feasible**

**Logic**: Sell a short-dated call at a higher strike and buy a longer-dated call at a lower strike. Combines the benefits of a calendar spread (theta differential) with a directional bias (bull spread). Used on momentum names where you expect gradual appreciation.

**RAMP Signal Used**: Momentum ranking (underlying selection) + regime detection (SIDEWAYS or WEAK_BULL preferred).

**Contract Selection**: Long call at delta 0.50-0.60, 50-65 DTE. Short call at delta 0.25-0.30, 21-30 DTE, at a higher strike. Different expirations and different strikes.

**Alpaca Execution**: 2-leg MLeg order. Same rolling caveat as #13 — rolling the short leg may be rejected. Close and reopen instead.

**Capital Requirement**: Medium. Net debit — the long call costs more than the short call premium.

**Complexity**: High. Two different expirations to manage. The short call expires before the long call, requiring a decision: roll the short call to a new expiration, or close the whole position.

**Edge**: Collects time decay differential (short-dated decays faster) while maintaining directional exposure through the long call.

**Key Risk**: If the stock surges past the short call strike before it expires, the spread collapses. Also, the rolling mechanics are operationally complex on Alpaca.

---

### #14 — Jade Lizards

**Feasibility Rank: 29 of 31 | Level 1+3 | Feasible with Workaround**

**Logic**: Sell a cash-secured put and simultaneously sell a call credit spread (sell call, buy higher call) on the same underlying. The combined credit eliminates upside risk if structured correctly — the total premium received exceeds the width of the call spread.

**RAMP Signal Used**: Momentum ranking (underlying selection) + regime detection (STRONG_BULL or SIDEWAYS).

**The Workaround**: The CSP is Level 1 (single-leg). The call credit spread is Level 3 (MLeg). These cannot be combined in one Alpaca MLeg order because the CSP is a separate single-leg position. You must execute them as two separate orders — the CSP and the call spread — which introduces execution risk (the stock could move between orders). Additionally, Alpaca won't recognize the combined position as a jade lizard for margin purposes, so you'll need full cash collateral for the CSP plus spread margin for the call spread.

**Alpaca Execution**: Two orders — sell-to-open CSP (Level 1), then 2-leg MLeg call credit spread (Level 3).

**Capital Requirement**: High. Cash for CSP + margin for spread.

**Complexity**: Very high. Three positions to manage, two different order types, coordination required.

**Edge**: Combined premium from both sides. If structured so total credit > call spread width, there is zero upside risk (only downside risk on the put side).

**Key Risk**: Execution risk between the two orders. Also, the complexity may not justify the incremental premium over a simple CSP (#8).

---

## Summary: Strategy Implementation Sequencing

Based on the feasibility analysis, the recommended build order is:

**Immediate (Level 1, minimal infrastructure)**:
1. #8 — Cash-secured puts (implementation plan complete)
2. #9 — Covered calls on RAMP equity
3. #27 — Wheel (combines #8 and #9 lifecycle)
4. #31 — Systematic covered call writing

**Near-term (Level 2, adds buy-side options)**:
5. #21 — Portfolio puts for crash protection
6. #24 — Tail risk hedging (continuous)
7. #1 — Long calls on top momentum
8. #6 — Deep ITM call replacement

**Medium-term (Level 3, multi-leg required)**:
9. #15 — Long straddles in UNPREDICTABLE
10. #12 — Put credit spreads in STRONG_BULL
11. #10 — Iron condors in SIDEWAYS
12. #2 / #4 — Bull call / bear put spreads

**Research priority (feasible with workarounds)**:
13. #19 — Vol regime switching (needs iron condors as proxy for strangles)
14. #17 — VIX call spreads (needs UVXY proxy)
15. #29 — Gamma scalping (needs intraday infrastructure)

**Defer (capital-intensive or marginally beneficial)**:
16. #20 — Dispersion trading
17. #7 — Synthetic longs
18. #14 — Jade lizards
19. #30 — 0DTE selling

---

## Appendix: Blocked Strategies

### #11 — Short Strangles in STRONG_BULL
**Blocked**: Requires naked short calls. Alpaca does not allow uncovered calls.
**Workaround**: Use iron condors (#10) instead — achieves similar premium collection with defined risk via long wings.

### #18 — Short Straddles in SIDEWAYS
**Blocked**: Requires naked short calls. Same constraint as #11.
**Workaround**: Use iron condors (#10) or iron butterflies instead.

---

## Appendix: Alpaca-Specific Implementation Notes

### Options Contract Discovery
```
GET /v2/options/contracts?underlying_symbols={SYMBOL}
    &type={call|put}
    &expiration_date_gte={YYYY-MM-DD}
    &expiration_date_lte={YYYY-MM-DD}
    &strike_price_gte={PRICE}
    &strike_price_lte={PRICE}
```

Use the `options_enabled` attribute on the assets endpoint to confirm an underlying supports options.

### Order Placement (Single-Leg)
```json
{
  "symbol": "AAPL240315P00170000",
  "qty": "1",
  "side": "sell",
  "type": "limit",
  "time_in_force": "day",
  "limit_price": "3.50",
  "position_intent": "sell_to_open"
}
```

### Order Placement (Multi-Leg)
```json
{
  "order_class": "mleg",
  "qty": "1",
  "type": "limit",
  "limit_price": "2.50",
  "time_in_force": "day",
  "legs": [
    {
      "symbol": "SPY240315P00500000",
      "ratio_qty": "1",
      "side": "sell",
      "position_intent": "sell_to_open"
    },
    {
      "symbol": "SPY240315P00490000",
      "ratio_qty": "1",
      "side": "buy",
      "position_intent": "buy_to_open"
    }
  ]
}
```

### Assignment Detection (Polling)
```
GET /v2/account/activities?activity_types=OPASN,OPEXP,OPEXC
    &after={YYYY-MM-DDTHH:MM:SSZ}
```

Poll this endpoint after market close and before next day's execution to detect overnight assignments.
