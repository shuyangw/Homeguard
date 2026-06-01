# Strategy Pipeline TODO -- RAMP Options Overlays [ARCHIVED 2026-04-02]

> **ARCHIVED / SHELVED -- not the active pipeline.** This tracked the RAMP
> options-overlay campaign (31 candidates, 16 tested across 3 batches). Closed
> 2026-04-02: no options strategy is deployment-ready; the overlay destroys the
> underlying momentum edge (see Portfolio summary). Preserved as a reference for
> structural findings and the per-strategy phase/integrity template.
>
> The active pipeline is now the root `TODO.md` (RAMP equity-momentum, Wave-3
> signal construction). The one still-live item here -- the **Section 11.5
> stop-slippage blocker** -- has been carried forward to the active TODO.

> Status: `[ ]` pending - `[~]` in progress - `[x]` done - `[!]` failed - `[-]` skipped
>
> Run: `claude --agent strategy-lead`
> Resume: `claude --agent strategy-lead --continue`
>
> Orchestrator: read this file FIRST on every session start.
> Mark `[~]` BEFORE starting a phase. Mark `[x]` AFTER verifying output exists.

---

## Active blockers (read before promoting any strategy to Phase 9 / live)

**Section 11.5 stop-slippage multiplier wiring is in flight (as of 2026-05-13).**
`portfolio_simulator.py` and its numba kernel apply uniform slippage to all fills regardless
of exit reason. `CostsSettings.stop_slippage_multiplier` is defined but inert. Until the
wiring PR lands, strategies with any of these exit types cannot graduate to Phase 9 live
promotion -- they would graduate on optimistic metrics that don't reflect the 1.5x-3.0x
stop-slippage reality:

- `fixed_pct_stop`
- `vol_scaled_stop`
- `trailing_stop`
- `time_stop_with_pct_stop`
- `scale_out`

Affected strategies in or queued for the pipeline:
- Darwinex FX MR (mean reversion with FX stops -- gated)
- ORB variants -- `hv_orb_baseline.yaml`, `orb_baseline.yaml` (intraday stops -- gated)
- `hurst_mr_baseline.yaml` (likely has stops -- check spec; gated if so)
- `ml_crypto_mr_baseline.yaml` (likely has stops -- check spec; gated if so)
- RAMP-CSP options strategies in `_archived` (Greek/premium stops -- review when revived)

The gate is enforced by `strategy-lead` Phase 9 validation. Lift this block when the
multiplier PR lands (target: separate PR, ~half day including numba kernel + tests).
Tracked at methodology Section 11.5 ("WIRING IN FLIGHT" callout).

---

## Pipeline setup (one-time, strategy #1)
- [x] **Understand infra** -> `code-explorer` -> `docs/architecture/infra_patterns.md`
  - Output: `docs/agent-learnings/ramp-csp/01_understanding.md`, `docs/architecture/infra_patterns.md`
  - Finding: ramp-csp already substantially implemented in `src/strategies/options/csp/`
- [x] **Design blueprint** -> `code-architect` -> `docs/agent-learnings/ramp-csp/02_architecture.md`
  - Output: `docs/agent-learnings/ramp-csp/02_architecture.md`
  - Finding: implementation complete, gaps in reporting/optimization wrappers only
- [x] **Create implementation skill** -> orchestrator -> `.claude/skills/implement-strategy/`
  - Output: `.claude/skills/implement-strategy/SKILL.md`

---

## Shared context: RAMP Options Strategies

All strategies below are options overlays on the existing RAMP equity strategy.
They use RAMP's three signal outputs as inputs:
- **Momentum ranking**: top_n / bottom_n stock selection from cross-sectional momentum
- **Regime detection**: STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR
- **Crash protection**: VIX > 25 or SPY drawdown > 5%

**Catalog**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md`
**RAMP equity spec**: `docs/strategies/production/RAMP_STRATEGY.md`
**RAMP equity code**: `src/strategies/advanced/` (existing, deployed)
**Alpaca options levels**: L1 = covered calls + CSP | L2 = L1 + long calls/puts | L3 = L2 + spreads, multi-leg

---

## Strategy: ramp-csp (#8 Cash-Secured Puts)

**Catalog ref**: #8 -- Feasibility Rank 1/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #8)
**Implementation**: `src/strategies/options/csp/` (engine, position, contract_selector, mark_to_market, metrics, ramp_integration)
**Config**: `config/strategies/ramp_csp.yaml`
**Tests**: `tests/strategies/options/csp/` (6 test files)
**Backtest script**: `scripts/backtest_scripts/ramp_csp_backtest.py`
**Reports**: `docs/reports/ramp-csp/`
**Agent learnings**: `docs/agent-learnings/ramp-csp/`
**Optimization output**: `output/optimization/ramp-csp/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 1
**Cost tier**: 15 bps (options bid-ask spread on liquid S&P 500 names)
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv`

**Strategy logic summary**:
- Sell OTM puts (delta -0.25 to -0.35, 21-35 DTE) on RAMP top_n momentum stocks
- Gate: STRONG_BULL regime only, crash protection not active
- Profit target: buy to close at 50% premium collected
- Loss limit: buy to close at 200% premium loss
- Exit: DTE <= 5, regime change to BEAR/UNPREDICTABLE, crash protection triggers, stock drops from top_n
- Position sizing: 30% portfolio to CSP, max 5 concurrent positions at 6% each
- Cash required per contract: strike x 100

**Default parameters**:
- put_delta: -0.30
- min_dte: 21, max_dte: 35
- profit_target_pct: 0.50
- loss_limit_pct: 2.00
- max_positions: 5
- portfolio_alloc: 0.30
- min_open_interest: 100
- max_bid_ask_spread_pct: 0.15

**Edge**: Dual -- VRP (selling overpriced insurance) + momentum (positive expected forward return on underlying). Both edges documented academically.
**Key risk**: Correlated drawdown across momentum names. Regime gate + crash protection are primary defenses.

### Phases
- [x] **3. Implement** -> already complete (pre-existing)
  - Files: `src/strategies/options/csp/{engine,position,contract_selector,mark_to_market,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_csp.yaml`
  - Data: `src/strategies/options/data_loader.py`, `src/data/options/`
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass -> `tests/strategies/options/csp/` (47/47 passed)
  - [x] 4b. Code review pass -> `docs/agent-learnings/ramp-csp/04_review.md`
    - No critical/high issues. Moderate: survivorship bias (current SP500 list). Low: redundant data updates.
    - Lookahead: PASS. Costs: PASS. Slippage: PASS. Regime: PASS. Temporal split: PASS.
- [x] **5. Initial backtest** -> `docs/reports/ramp-csp/20260330_initial_backtest.md`
- [x] **6. Validate** -> orchestrator -> notes below (PROCEED with conservative expectations)
- [x] **7. Optimize** -> `output/optimization/ramp-csp/optimization_results.json`
  - 75 combos, 3 WF windows. Best OOS Sharpe = 0.218 (tp=0.50, ll=1.0, delta=deeper OTM)
  - All OOS Sharpes < 0.5. Edge not statistically significant at portfolio level.
- [-] **8. Final validation** -> SKIPPED (OOS Sharpe 0.218 < 0.5 threshold)

### Integrity checklist (orchestrator verifies after each backtest)
- [x] shift(1) confirmed on all signals (callback architecture equivalent - all data filtered <= d)
- [x] Transaction costs included (1% slippage + $0.02/contract fee)
- [x] Slippage model active (sells at bid*(1-slip), buys at mid*(1+slip))
- [x] No lookahead bias detected (95% confidence)
- [x] Temporal train/test split (not random) - IS: 2022-01 to 2023-06, OOS: 2023-07 to 2024-12
- [x] Regime analysis included (5-regime detector + crash protection)
- [x] Data frequency matches strategy logic (daily/EOD)
- [!] Universe: uses current SP500 list (mild survivorship bias, documented)

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | spec defaults | d=-0.30, DTE=21-35, tp=50%, ll=200% | 0.109 | - | 0.44% | 2.99% | 130d | 0.15 | 28.6% | 1.23 | 63 | 3.7d | 100% SB | - | IS:-0.15/OOS:0.26 | 2022-2024 | daily | IS negative, OOS positive, 9 IS trades |
| 2   | optimizer best (WF) | d=(-0.3,-0.2), tp=50%, ll=100% | - | - | 0.78% | - | - | - | 65.0% | - | 60 | - | 100% SB | - | IS:-0.03/OOS:0.218 | 3 WF windows | daily | Best of 75 combos. OOS < 0.5 |

### Optimization summary
- Combinations tested: 75
- Parameters optimized: 3 (profit_target_pct, loss_limit_multiple, delta_range)
- Walk-forward windows: 3 (W1: train 2022-01/2023-03, test 2023-04/09 | W2: train 2022-07/2023-09, test 2023-10/2024-03 | W3: train 2023-01/2024-03, test 2024-04/09)
- Best IS Sharpe: -0.030 | Best OOS Sharpe: 0.218 | Gap: N/A (IS negative, OOS positive - regime difference)
- DSR-adjusted Sharpe: not computed (OOS Sharpe already < 0.5)
- Parameter sensitivity: STABLE (ll=1.0 combos range 0.132-0.218 OOS, no cliff edges)
- Cost sensitivity (1.5x): deferred (edge too thin to matter)
- Magic numbers flagged: NONE (all params have economic rationale)
- Estimated runtime: ~2.5h | Actual runtime: ~2h
- **CRITICAL**: Best OOS Sharpe = 0.218 < 0.5 threshold. Edge not statistically significant.
- **KEY FINDING**: Loss limit = 1.0 clearly dominates (tighter stops help). All other params are roughly equivalent.

### Validation notes (Phase 6)

**Checklist:**
- [x] Sharpe < 3.0: YES (0.109) -- no overfitting concern
- [x] CAGR < 20%: YES (0.44%) -- modest, not suspicious
- [x] Max DD > 5%: NO (2.99%) -- acceptable for 30%-allocated premium strategy with 70% idle cash
- [x] Trades > 30: YES (63 full, 54 OOS) -- IS only 9 trades (insufficient)
- [!] Regime robust: 100% from STRONG_BULL -- BY DESIGN (strategy only enters in SB)
- [x] Transaction costs included
- [x] Slippage modeled
- [x] No lookahead confirmed
- [x] Data frequency matches

**Key observations:**
1. IS/OOS gap metric is INVALID: IS had only 9 trades (2022 bear market = no STRONG_BULL entries)
2. OOS is the meaningful period: Sharpe 0.265, CAGR 1.68%, 54 trades, 66.7% win rate
3. Avg ROC per trade = 0.26% OOS with 3.6 day hold. Annualized on deployed capital = reasonable
4. Strategy is regime-dependent BY SPECIFICATION (not a flaw)
5. Low returns partly due to small options universe (11 symbols from 503 SP500)
6. Left_top_n exit reason accounts for 52/63 trades -- stocks rotate out of RAMP's top_n quickly

**Decision:** PROCEED to Phase 7 with conservative expectations. Optimize profit target and loss limit.
The edge is real but thin. Small universe limits trade opportunities.

### Final validation notes (Phase 8)
SKIPPED -- OOS Sharpe below 0.5 threshold. No final validation warranted.

### Verdict
- [x] **Final**: MARGINAL (not viable for standalone deployment, but edge is directionally real)
- **Reason**: Best walk-forward OOS Sharpe = 0.218 (below 0.5 threshold). The per-trade edge is real (63.5% win rate, positive P&L) but portfolio-level Sharpe is suppressed by: (1) only 11 of 503 SP500 symbols have options data, (2) strategy only enters in STRONG_BULL regime (~30% of time), (3) avg holding time 3.7 days with left_top_n exits accounting for 82% of trades. Capital utilization is ~6%.
- **Overfitting risk**: LOW (parameters are stable, no cliff edges, no magic numbers)
- **Regime classification**: FRAGILE (100% returns from STRONG_BULL, by design)
- **Edge survives 1.5x costs?**: Not tested (edge too thin to matter)
- **Recommendation**: DO NOT deploy as standalone strategy. REVISIT when: (1) options data universe expands to 50+ SP500 symbols, (2) RAMP top_n turnover is reduced (stocks rotate out too quickly for CSP holding periods). The strategy logic and infrastructure are solid -- the constraint is data availability and signal alignment. Consider using as a small allocation within a combined portfolio rather than standalone.

---

## Strategy: ramp-portfolio-puts (#21 Portfolio Puts)

**Catalog ref**: #21 -- Feasibility Rank 5/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #21)
**Implementation**: `src/strategies/options/portfolio_puts/` (engine, position, contract_selector, mark_to_market, metrics, ramp_integration)
**Config**: `config/strategies/ramp_portfolio_puts.yaml`
**Tests**: `tests/strategies/options/portfolio_puts/`
**Reports**: `docs/reports/ramp-portfolio-puts/`
**Optimization output**: `output/optimization/ramp-portfolio-puts/`

**Asset class**: Large-cap index (SPY options)
**Options level**: Level 2
**Cost tier**: 10 bps (SPY options are highly liquid)
**Data frequency**: daily
**Universe**: SPY only

**Strategy logic summary**:
- When crash protection triggers (VIX > 25 or SPY DD > 5%), buy SPY puts instead of selling half equity
- SPY puts at delta -0.30 to -0.50, 14-30 DTE
- Hedge notional = 50% of portfolio value
- Exit: crash signal clears, DTE <= 5 (roll if still needed), SPY drops significantly (take profit)
- Replaces RAMP's current 50% cash-raise mechanism with options hedge

**Default parameters**:
- put_delta: -0.40
- min_dte: 14, max_dte: 30
- hedge_ratio: 0.50

**Edge**: Preserves RAMP equity positions during turbulence. Avoids selling at bottom and re-buying higher.
**Key risk**: Premium drag if crash protection triggers frequently (VIX oscillates around 25).

### Phases
- [x] **3. Implement** -> orchestrator direct (following CSP pattern)
  - Files: `src/strategies/options/portfolio_puts/{__init__,position,contract_selector,mark_to_market,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_portfolio_puts.yaml`
  - Data: reuses `src/strategies/options/data_loader.py` and `src/strategies/options/csp/mark_to_market.py` (B-S)
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass -> `tests/strategies/options/portfolio_puts/` (55/55 passed)
  - [x] 4b. Code review pass -> `docs/agent-learnings/ramp-portfolio-puts/04_review.md`
    - No issues found. Lookahead: PASS. Costs: PASS. Slippage: PASS. Regime: PASS. Temporal split: PASS.
- [x] **5. Initial backtest** -> `docs/reports/ramp-portfolio-puts/20260331_initial_backtest.md`
- [x] **6. Validate** -> orchestrator -> notes below (NOT VIABLE)
- [-] **7. Optimize** -> SKIPPED (negative Sharpe, no edge to optimize)
- [-] **8. Final validation** -> SKIPPED (no optimization to validate)

### Integrity checklist
- [x] shift(1) confirmed (callback architecture, <= d filtering)
- [x] Costs 10 bps (0.5% slippage + $0.02/contract)
- [x] Slippage (buy at ask*(1+slip), sell at mid*(1-slip))
- [x] No lookahead
- [x] Temporal split (IS: 2018-2021, OOS: 2022-2024)
- [x] Regime (5-regime detector + crash protection)
- [x] Freq match (daily)
- [x] Universe ok (SPY only, no survivorship bias)

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | spec defaults | d=[-0.50,-0.30], hedge=50%, tp=100% | -0.597 | - | -1.88% | 14.98% | 1202d | -0.13 | 23.9% | - | 92 | 10.1d | all regimes | - | IS:-0.798/OOS:-0.137 | 2018-2024 | daily | Negative Sharpe. Premium drag > hedge benefit. 699 crash-active days out of 1761 |

### Validation notes (Phase 6)

**Checklist:**
- [x] Sharpe < 3.0: YES (-0.597) -- no overfitting concern (strategy loses money)
- [x] CAGR < 20%: YES (-1.88%) -- negative return
- [x] Max DD > 5%: YES (14.98%) -- significant drawdown from premium payments
- [x] Trades > 30: YES (92 full, 44 OOS) -- sufficient for statistical conclusions
- [x] Regime robust: crash protection fires across all regimes (699 crash-active days / 1761 total = 40%)
- [x] Transaction costs included
- [x] Slippage modeled
- [x] No lookahead confirmed
- [x] Data frequency matches

**Key observations:**
1. **Strategy loses money in both IS and OOS.** IS Sharpe = -0.798, OOS Sharpe = -0.137. No edge.
2. **Crash protection triggers too frequently.** 699 crash-active days out of 1761 (40% of the time). The VIX > 25 or SPY DD > 5% signal is far too liberal for a hedge strategy -- you're buying expensive insurance 40% of the time.
3. **Win rate of 23.9% is too low.** Only 22 of 92 trades hit profit target. Most positions expire worthless or close at a loss when crash clears (41 trades) or DTE expires (32 trades).
4. **Premium drag dominates.** Total premium paid: $57,335 over 7 years. Total PnL: -$12,432. Hedge effectiveness: -0.28% per trade on average.
5. **Comparison to RAMP cash-raise:** RAMP's current approach (sell 50% equity to cash) has zero cost. This strategy costs ~$1,776/year in premium drag for a $100K portfolio. The hedge only pays off if SPY drops sharply WHILE you hold the puts AND the drop is large enough to offset premium. In practice, crash protection triggers and then clears within 10 days on average, before puts appreciate enough.
6. **OOS is less bad than IS** (-0.137 vs -0.798) because 2022 bear market provided some profitable trades ($12,322 from profit targets), but not enough to offset the 31 losing trades.

**Root cause:** The strategy is economically flawed as specified. Buying puts only when VIX > 25 means you're buying puts when IV is ALREADY elevated (expensive). The puts are costly, and by the time crash protection clears, the puts have lost value to theta decay. This is the opposite of insurance timing -- you want to buy insurance when it's cheap (low VIX), not when everyone is already panicking.

**Decision:** NOT VIABLE. No optimization warranted -- the fundamental signal timing is wrong. Skipping Phases 7 and 8.

### Verdict
- [x] **Final**: NOT VIABLE
- **Reason**: Negative Sharpe in both IS (-0.798) and OOS (-0.137). Crash protection triggers 40% of the time (too frequent). Buying puts at elevated VIX = buying expensive insurance. Premium drag ($1.8K/year on $100K) exceeds hedge benefit. 23.9% win rate. Compare to RAMP cash-raise which costs zero.
- **Overfitting risk**: N/A (strategy loses money, not overfit)
- **Regime classification**: N/A (not profitable in any regime)
- **Edge survives 1.5x costs?**: No (edge is negative at 1x costs)
- **Recommendation**: ARCHIVE. The strategy's premise is economically backwards -- buying puts when VIX is already high. If revisiting, consider: (1) buying puts in LOW VIX regimes as cheap insurance before crashes happen, (2) using far OTM puts (delta -0.10) for cheaper tail hedge (see #24 Tail Risk Hedging), (3) only buying puts on extreme crash signals (VIX > 35, not 25). The implementation and infrastructure are solid -- the signal is wrong.

---

## Strategy: ramp-tail-hedge (#24 Tail Risk Hedging)

**Catalog ref**: #24 -- Feasibility Rank 9/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #24)
**Implementation**: `src/strategies/options/tail_hedge/` (engine, position, contract_selector, mark_to_market, metrics, ramp_integration)
**Config**: `config/strategies/ramp_tail_hedge.yaml`
**Tests**: `tests/strategies/options/tail_hedge/`
**Reports**: `docs/reports/ramp-tail-hedge/`
**Optimization output**: `output/optimization/ramp-tail-hedge/`

**Asset class**: Large-cap index (SPY options)
**Options level**: Level 2
**Cost tier**: 10 bps
**Data frequency**: daily
**Universe**: SPY only

**Strategy logic summary**:
- Continuously hold far OTM SPY puts (delta -0.10 to -0.15, 30-60 DTE) as portfolio insurance
- Regime-adaptive sizing: STRONG_BULL=0.25%, WEAK_BULL=0.50%, SIDEWAYS=0.25%, UNPREDICTABLE=0.75%, BEAR=1.0% of portfolio per month
- Monthly roll cycle (close and reopen with fresh DTE)
- Always on -- not triggered by crash protection, sized by regime

**Default parameters**:
- put_delta: -0.12
- min_dte: 30, max_dte: 60
- alloc_strong_bull: 0.0025
- alloc_weak_bull: 0.0050
- alloc_sideways: 0.0025
- alloc_unpredictable: 0.0075
- alloc_bear: 0.0100

**Edge**: Convex payoff. 3-12% annual cost, but 500-2000% return in crash years. Portfolio insurance, not alpha source.
**Key risk**: Persistent drag in non-crash years reduces overall Sharpe.

### Phases
- [x] **3. Implement** -> orchestrator direct (following portfolio-puts pattern)
  - Files: `src/strategies/options/tail_hedge/{__init__,position,contract_selector,mark_to_market,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_tail_hedge.yaml`
  - Tests: `tests/strategies/options/tail_hedge/` (56 tests)
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass -> `tests/strategies/options/tail_hedge/` (56/56 passed)
  - [x] 4b. Code review pass -> `docs/agent-learnings/ramp-tail-hedge/04_review.md`
    - No critical/high issues. Lookahead: PASS. Costs: PASS. Slippage: PASS. Regime: PASS. Temporal split: PASS.
- [x] **5. Initial backtest** -> `docs/reports/ramp-tail-hedge/20260331_initial_backtest.md`
- [x] **6. Validate** -> orchestrator -> PROCEED
- [x] **7. Optimize** -> `output/optimization/ramp-tail-hedge/optimization_results.json`
  - 48 combos, 3 WF windows. Best composite: d=[-0.20,-0.10], alloc_bear=1.5%, roll_dte=10
  - DD reduction 10.4% avg, premium 7.05%, Sharpe change -0.079 avg
  - Delta [-0.20,-0.10] dominates all other ranges. roll_dte=10 is optimal.
- [x] **8. Final validation** -> `docs/reports/ramp-tail-hedge/20260331_final_validation.md`

### Integrity checklist
- [x] shift(1) confirmed (callback architecture, all data filtered <= d)
- [x] Costs 10 bps (0.5% slippage + $0.02/contract)
- [x] Slippage (buy at ask*(1+slip), sell at mid*(1-slip))
- [x] No lookahead
- [x] Temporal split (IS: 2018-2021, OOS: 2022-2024)
- [x] Regime (5-regime detector, regime-adaptive sizing)
- [x] Freq match (daily)
- [x] Universe ok (SPY only, no survivorship bias)

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | spec defaults | d=[-0.15,-0.10], regime-sized | -0.366 | - | -1.14% | 10.57% | 1196d | -0.107 | 16.0% | - | 81 | 26.8d | all regimes | - | IS:-0.310/OOS:-0.430 | 2018-2024 | daily | 81 trades, 5 profit target hits. Annual premium 4.93%. UNPREDICTABLE regime +$9.2K (3 trades). |
| 2   | optimizer best (WF) | d=[-0.20,-0.10], ab=1.5%, rdte=10 | -0.791 | - | - | - | - | - | - | - | 36 OOS | - | all | - | IS:var/OOS:-0.791 avg | 3 WF windows | daily | Best of 48 combos. Avg OOS DD reduction 10.4%. Premium 7.05%. |
| 3   | final validation | d=[-0.20,-0.10], ab=1.5%, rdte=10 | -0.159 | - | -0.82% | 10.14% | - | -0.081 | 19.8% | - | 81 | - | all | -0.186 | IS:-0.129/OOS:-0.194 | 2018-2024 | daily | Combined SPY+TH: Sharpe +0.032, DD -24.3%, Calmar +0.095. OOS: Sharpe +0.016, DD -6.4%. 1.5x Sharpe=-0.186. |

### Optimization summary
- Combinations tested: 48
- Parameters optimized: 3 (delta_range, alloc_bear, roll_dte_target)
- Walk-forward windows: 3 (W1: IS 2018-2020, OOS 2021 | W2: IS 2019-2021, OOS 2022 | W3: IS 2020-2022, OOS 2023)
- Best combo: d=[-0.20,-0.10], alloc_bear=0.015, roll_dte=10
- Avg OOS Sharpe: -0.791 (best standalone) | Avg premium: 7.05%
- Avg combined DD reduction: 10.4% | Avg Sharpe change: -0.079
- Per-window OOS results:
  - W1 (2021 bull): DD reduction 4.9%, Sharpe change -0.136, premium 5.57%
  - W2 (2022 bear): DD reduction 6.2%, Sharpe change -0.168, premium 11.20%
  - W3 (2023 recovery): DD reduction 20.0%, Sharpe change +0.068, premium 4.39%
- Parameter sensitivity: STABLE (d=[-0.20,-0.10] dominates across all alloc/roll combos)
- Cost sensitivity (1.5x): deferred to Phase 8
- Magic numbers flagged: NONE
- **KEY FINDING**: The wider delta range [-0.20,-0.10] allows more contract selection and significantly outperforms the narrow spec default [-0.15,-0.10]. roll_dte=10 is robustly optimal.

### Validation notes (Phase 6)

**Checklist (modified for insurance strategy):**
- [x] Sharpe < 3.0: YES (-0.366) -- no overfitting concern
- [x] CAGR < 20%: YES (-1.14%) -- negative, expected for insurance
- [x] Max DD > 5%: YES (10.57%) -- premium drag accumulation
- [x] Trades > 30: YES (81 full, 36 OOS) -- sufficient
- [x] Regime breakdown: active in all 5 regimes (always-on by design)
- [x] Transaction costs included
- [x] Slippage modeled
- [x] No lookahead confirmed
- [x] Data frequency matches

**Key observations:**
1. **Annual premium cost: 4.93%** -- within 3-12% target range from spec. This is the insurance premium.
2. **Profit target hits: 5 out of 81 trades (6.2%).** Tail events are rare. When they hit: +$13,276. Monthly roll losses: -$20,921. Net: -$7,683.
3. **UNPREDICTABLE regime is the winner:** +$9,216 from just 3 trades. This is the tail hedge working correctly -- UNPREDICTABLE regimes precede crashes, puts pay off.
4. **IS vs OOS:** IS Sharpe -0.310, OOS Sharpe -0.430. OOS worse because 2022-2024 had fewer extreme crash events than 2018-2021 (COVID March 2020). Gap is only 0.12 -- not concerning for binary payoff.
5. **Win rate 16%:** Expected for far OTM puts. Most roll as losers.
6. **Comparison to portfolio-puts (#21):** Tail hedge Sharpe -0.366 vs portfolio-puts -0.597. Tail hedge is BETTER because it buys cheaply (far OTM, including in low-VIX periods) vs portfolio-puts buying at elevated VIX.
7. **Critical missing analysis:** Combined RAMP+tail portfolio metrics not yet computed. This is the KEY evaluation for Phase 7/8.

**Decision:** PROCEED to Phase 7. Optimize put_delta, alloc_bear, roll_dte_target. Must compute combined portfolio metrics as primary objective.

### Final validation notes (Phase 8)

**1. Standard costs (0.5% slippage):**
- Full period: Sharpe -0.159, CAGR -0.82%, MaxDD 10.14%, 81 trades, 19.8% win rate
- IS: Sharpe -0.129, CAGR -0.64%, MaxDD 7.64%, 45 trades
- OOS: Sharpe -0.194, CAGR -1.08%, MaxDD 9.93%, 36 trades
- Annual premium: 7.16% of portfolio

**2. 1.5x costs (0.75% slippage):**
- Sharpe -0.186, CAGR -0.95%, MaxDD 10.33%
- Minimal degradation from standard costs (premium is the dominant cost, not slippage)

**3. Parameter stability (+/-10%):**
- alloc_bear +/-10%: Sharpe range [-0.317, -0.241] -- stable, no cliff edges
- roll_dte +/-1: Sharpe range [-0.177, -0.208] -- stable

**4. Combined SPY + Tail Hedge portfolio (THE KEY METRIC):**
- **Full period (2018-2024):**
  - SPY standalone: Sharpe 0.679, CAGR 11.80%, MaxDD 33.40%, Calmar 0.353
  - Combined SPY+TH: Sharpe 0.711, CAGR 11.32%, MaxDD 25.27%, Calmar 0.448
  - DD reduction: 24.3% | Sharpe change: +0.032 | Calmar change: +0.095
- **OOS (2022-2024):**
  - SPY standalone: Sharpe 0.484, MaxDD 25.21%
  - Combined: Sharpe 0.499, MaxDD 23.60%
  - DD reduction: 6.4% | Sharpe change: +0.016

**Assessment:**
The tail hedge achieves its stated purpose: it reduces portfolio drawdown more than it reduces returns. The full-period Calmar ratio improves 27% (0.353 -> 0.448). The CAGR loss is only 0.48% for an 8.13% reduction in max drawdown. The combined Sharpe actually improves, which is the strongest possible endorsement of a negative-Sharpe insurance strategy.

The OOS improvement is smaller (6.4% DD reduction) because the 2022 bear was a slow grind rather than a sharp crash. Tail puts are most effective during violent crashes (COVID March 2020 style) where far OTM puts explode in value. The strategy had 5 profit-target trades across the full period, with the largest gains coming from UNPREDICTABLE regime entries.

The premium cost of ~7% annually is within the 3-12% target range. The cost is worth it if you value the convex downside protection and improved risk-adjusted returns.

### Verdict
- [x] **Final**: MARGINAL (viable as portfolio hedge, not standalone)
- **Reason**: Standalone Sharpe is negative (-0.159) as expected for insurance. Combined with SPY/RAMP equity: DD reduced 24.3%, Sharpe improved +0.032, Calmar improved +0.095. Annual premium 7.16%. The hedge achieves its stated purpose -- reducing drawdown more than it reduces returns. OOS improvement is smaller (6.4%) because 2022 lacked sharp crashes.
- **Overfitting risk**: LOW (parameters stable +/-10%, delta range dominance clear, no magic numbers)
- **Regime classification**: ROBUST (active in all regimes, sized by regime)
- **Edge survives 1.5x costs?**: YES (Sharpe -0.186 at 1.5x, minimal degradation -- premium is dominant cost, not slippage)
- **Recommendation**: DEPLOY as portfolio overlay with RAMP equity. Not a standalone strategy. Allocate 7% annual budget to put premiums. Use optimized params: delta [-0.20,-0.10], alloc_bear 1.5%, roll_dte 10. The tail hedge is the "insurance premium" for the portfolio -- it costs money every month but dramatically reduces max drawdown during crashes.

---

## Strategy: ramp-cc (#9 Covered Calls on RAMP Equity)

**Catalog ref**: #9 -- Feasibility Rank 2/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #9)
**Implementation**: `src/strategies/advanced/ramp_cc.py`
**Config**: `config/strategies/ramp_cc.yaml`
**Tests**: `tests/strategies/test_ramp_cc.py`
**Reports**: `docs/reports/ramp-cc/`
**Agent learnings**: `docs/agent-learnings/ramp-cc/`
**Optimization output**: `output/optimization/ramp-cc/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 1
**Cost tier**: 15 bps
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv`

**Strategy logic summary**:
- Sell OTM calls against shares already held in RAMP equity portfolio
- Regime-adaptive delta: STRONG_BULL=0.20, WEAK_BULL=0.30, SIDEWAYS=0.35, BEAR/UNPREDICTABLE=no sell
- 21-35 DTE, one contract per 100 shares held
- Exit: 50% profit target, DTE <= 5, RAMP sell signal on underlying, regime to BEAR/UNPREDICTABLE
- Requires coordination with RAMP rebalance (close CC before selling underlying)

**Default parameters**:
- delta_strong_bull: 0.20
- delta_weak_bull: 0.30
- delta_sideways: 0.35
- min_dte: 21, max_dte: 35
- profit_target_pct: 0.50

**Edge**: Theta decay income on existing equity positions. Regime-adaptive strike selection balances premium vs upside cap.
**Key risk**: Capping upside in strong momentum moves. Assignment forces equity position close + re-entry costs.

### Phases
- [x] **3. Implement** -> general-purpose + skill
  - Files: `src/strategies/options/equity_simulator.py` (shared RAMPEquitySimulator)
  - Files: `src/strategies/options/cc/{__init__,position,contract_selector,mark_to_market,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_cc.yaml`
  - All imports verified
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass -> `tests/strategies/options/cc/` + `tests/strategies/options/test_equity_simulator.py` (96/96 passed)
  - [x] 4b. Code review pass -> `docs/agent-learnings/ramp-cc/04_review.md`
    - 2 HIGH fixed (MTM ordering, dead param), 1 MEDIUM fixed (B-S zero guard). Lookahead: PASS. Costs: PASS. Slippage: PASS. Regime: PASS.
- [x] **5. Initial backtest** -> `docs/reports/ramp-cc/20260331_initial_backtest.md`
- [x] **6. Validate** -> orchestrator -> notes below (PROCEED to optimize)
- [x] **7. Optimize** -> `output/optimization/ramp-cc/optimization_results.json`
  - 64 combos, 3 WF windows (192 runs). Best avg OOS Sharpe = 0.097.
  - ALL combos avg OOS < 0.3. W2 (late 2023) destroys all combos. W3 (mid-2024) all positive.
  - dwb=0.25 dominates. tp=0.65 slightly better. dsb insensitive.
- [-] **8. Final validation** -> SKIPPED (avg OOS Sharpe 0.097 < 0.5 threshold)

### Integrity checklist
- [x] shift(1) confirmed (callback architecture, all data filtered <= d)
- [x] Transaction costs included (1% slippage + $0.02/contract fee)
- [x] Slippage model active (sells at bid*(1-slip), buys at mid*(1+slip))
- [x] No lookahead bias detected (95% confidence)
- [x] Temporal train/test split (not random) - IS: 2022-01 to 2023-06, OOS: 2023-07 to 2024-12
- [x] Regime analysis included (5-regime detector, regime-adaptive deltas)
- [x] Data frequency matches strategy logic (daily/EOD)
- [!] Universe: uses current SP500 list (mild survivorship bias, documented). Only 11 symbols have options data.

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | spec defaults | d_SB=0.20, d_WB=0.30, d_SW=0.35, tp=50%, ll=100% | 0.247 | - | 0.19% | 1.27% | - | 0.15 | 48.6% | 1.41 | 37 | 2.7d | SW=$498,SB=$5,WB=$64 | - | IS:0.707/OOS:0.117 | 2022-2024 | daily | IS only 4 trades (insufficient). 22/37 exit underlying_sold. |
| 2   | optimizer best (WF) | d_SB=0.30, d_WB=0.25, tp=65%, ll=100% | - | - | - | - | - | - | - | - | 29 OOS | - | - | - | avg OOS:0.097 | 3 WF windows | daily | Best of 64 combos. W1:0.300 W2:-0.569 W3:0.560. All avg OOS < 0.3. |

### Optimization summary
- Combinations tested: 64 (4 x 4 x 4 grid), 192 total runs (3 WF windows)
- Parameters optimized: 3 (delta_strong_bull, delta_weak_bull, profit_target_pct)
- Walk-forward windows: 3 (W1: IS 2022-01/2023-03, OOS 2023-04/09 | W2: IS 2022-07/2023-09, OOS 2023-10/2024-03 | W3: IS 2023-01/2024-03, OOS 2024-04/09)
- Best avg OOS Sharpe: 0.097 (C52: dsb=0.30, dwb=0.25, tp=0.65)
- DSR-adjusted Sharpe: not computed (OOS Sharpe already < 0.5)
- Parameter sensitivity: dsb=INSENSITIVE, dwb=MODERATE (0.25 best), tp=MODERATE (0.65 best)
- Cost sensitivity (1.5x): deferred (edge too thin to matter)
- Magic numbers flagged: NONE
- Estimated runtime: ~1h | Actual runtime: ~1.1h
- **CRITICAL**: Best avg OOS Sharpe = 0.097 < 0.5 threshold. Edge not statistically significant.
- **KEY FINDING 1**: W2 (late 2023 to early 2024) destroys ALL combos (OOS -0.49 to -0.66). Strong momentum rally = sold calls get blown through.
- **KEY FINDING 2**: W3 (mid-2024 SIDEWAYS) is universally positive (0.2-0.6). CC works when market range-bound.
- **KEY FINDING 3**: underlying_sold still dominant exit (60-70%). RAMP turnover too fast for CC holding periods.
- **FUNDAMENTAL PROBLEM**: Selling calls on momentum stocks caps upside exactly when momentum is strongest. This contradicts RAMP's core edge.

### Validation notes (Phase 6)

**Checklist:**
- [x] Sharpe < 3.0: YES (0.247) -- no overfitting concern
- [x] CAGR < 20%: YES (0.19%) -- modest, not suspicious
- [x] Max DD > 5%: NO (1.27%) -- acceptable for premium-only overlay with no capital at risk
- [x] Trades > 30: YES (37 full) but IS only has 4 trades (insufficient for IS analysis)
- [!] Regime robust: 88% of P&L from SIDEWAYS ($498 of $567 total) -- regime-dependent
- [x] Transaction costs included
- [x] Slippage modeled
- [x] No lookahead confirmed
- [x] Data frequency matches

**Key observations:**
1. **IS/OOS gap metric is INVALID**: IS had only 4 trades (2022 was mostly BEAR/UNPREDICTABLE = no CC entries). IS Sharpe 0.707 is meaningless with 4 trades.
2. **OOS is the meaningful period**: Sharpe 0.117, 28 trades, 42.9% win rate, $158.55 total P&L.
3. **Dominant exit reason: underlying_sold (22/37 = 59.5%)** with P&L -$824.08. RAMP rotates stocks out of top_n before CCs can be profitably closed. This is the same problem as CSP (left_top_n exits).
4. **Avg holding time 2.7 days is very short.** CCs are opened 21-35 DTE but closed after ~3 days because RAMP sells the underlying. Premium capture is minimal in 3 days of theta.
5. **SIDEWAYS regime is the winner ($498 of $567).** This makes economic sense -- SIDEWAYS has highest delta target (0.35 = higher premium) and stocks are less likely to surge past strike.
6. **STRONG_BULL contributes almost nothing ($4.63).** The 0.20 delta in SB means very OTM calls with tiny premiums. Low premium + short holding = near-zero P&L.
7. **Only 5 profit_target exits (13.5%).** Most positions exit early due to underlying rotation, not premium decay.
8. **Comparison to CSP**: CC total P&L $567 vs CSP total P&L $7,683 (net of IS+OOS). CC is worse because it starts in different regimes (SB/WB/SW vs SB-only) but suffers same underlying rotation problem.

**Root cause analysis:**
The covered call strategy is constrained by two reinforcing problems:
1. **RAMP's high turnover**: top_n symbols rotate frequently, forcing CC early exits before theta decays enough to profit
2. **11-symbol options universe**: Only 11 of 503 SP500 symbols have options data, severely limiting opportunities

The strategy logic is sound -- selling calls in SIDEWAYS/WEAK_BULL generates real premium income. But RAMP's daily rebalance is incompatible with 21-35 DTE options that need time to work.

**Decision:** PROCEED to Phase 7. Optimize delta_strong_bull, delta_weak_bull, profit_target_pct. The SIDEWAYS regime edge is real. Focus optimization on maximizing SIDEWAYS P&L and testing whether tighter profit targets (25-40%) capture more value before underlying_sold exits.

### Final validation notes (Phase 8)
SKIPPED -- avg OOS Sharpe 0.097 below 0.5 threshold. No final validation warranted.

### Verdict
- [x] **Final**: NOT VIABLE (edge too thin and contradicts RAMP momentum edge)
- **Reason**: Best walk-forward avg OOS Sharpe = 0.097 (below 0.5 threshold). Strategy works in SIDEWAYS markets (W3 OOS ~0.5) but fails in strong momentum (W2 OOS ~-0.6). Net across regimes is near zero. Fundamental conflict: selling calls on momentum stocks caps the upside that makes momentum profitable. RAMP high turnover (daily rebalance) incompatible with 21-35 DTE options -- 60-70% trades exit early due to underlying_sold with avg hold 2.7 days.
- **Overfitting risk**: LOW (parameters stable, no cliff edges, no magic numbers)
- **Regime classification**: FRAGILE (profitable only in SIDEWAYS, loses in momentum rallies)
- **Edge survives 1.5x costs?**: Not tested (edge near-zero at 1x costs)
- **Recommendation**: DO NOT deploy. Fundamental design flaw: selling upside on momentum stocks contradicts RAMP core edge. If revisiting: (1) Only sell CC in SIDEWAYS regime, (2) Use systematic-cc (#31) as uniform buy-write overlay, (3) Decouple CC from RAMP daily rebalance. Infrastructure (equity simulator, CC engine) is solid and reusable by #31 and #27.

---

## Strategy: ramp-wheel (#27 Regime-Adaptive Wheel)

**Catalog ref**: #27 -- Feasibility Rank 4/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #27)
**Implementation**: `src/strategies/advanced/ramp_wheel.py`
**Config**: `config/strategies/ramp_wheel.yaml`
**Tests**: `tests/strategies/test_ramp_wheel.py`
**Reports**: `docs/reports/ramp-wheel/`
**Agent learnings**: `docs/agent-learnings/ramp-wheel/`
**Optimization output**: `output/optimization/ramp-wheel/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 1
**Cost tier**: 15 bps
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv`

**Strategy logic summary**:
- Lifecycle: Cash -> Sell CSP -> [Assigned?] -> Hold Stock -> Sell CC -> [Called Away?] -> Cash -> ...
- STRONG_BULL: sell CSPs on top momentum names (collect premium while waiting to buy)
- WEAK_BULL/SIDEWAYS: sell CCs on assigned shares (collect premium on held stock)
- BEAR/UNPREDICTABLE: no new positions, close existing if stops hit
- Combines #8 and #9 into a continuous premium-collecting cycle
- Position sizing: 30% portfolio, max 5 concurrent wheels

**Default parameters**:
- Same as #8 for CSP leg, same as #9 for CC leg
- Lifecycle state machine: CSP_PHASE / EQUITY_PHASE / CC_PHASE

**Edge**: Compounds premium from both sides of the wheel. Regime gate ensures regime-appropriate action.
**Key risk**: Assignment during regime transition. Stock assigned in declining market before next regime check.
**Dependency**: Builds on #8 (CSP) and #9 (CC) implementations.

### Phases
- [x] **3. Implement** -> general-purpose + skill
  - Files: `src/strategies/options/wheel/{__init__,position,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_wheel.yaml`
  - Tests: `tests/strategies/options/wheel/{test_position,test_engine,test_metrics}.py` (45 tests)
  - State machine: CASH -> CSP_PHASE -> [assigned?] -> EQUITY_PHASE -> CC_PHASE -> [called away?] -> CASH
  - Reuses: CSPContractSelector, CCContractSelector, CSPMarkToMarket, CCMarkToMarket
  - Does NOT use RAMPEquitySimulator -- manages own shares via CSP assignment
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass (45/45 passed)
  - [x] 4b. Code review pass -> `docs/agent-learnings/ramp-wheel/04_review.md`
    - CRITICAL: Double-counting option premium in equity calc -> FIXED
    - HIGH: Exit fees on expiration not deducted from cash -> FIXED
    - MEDIUM: Log mutation order -> FIXED
    - MEDIUM: Survivorship bias (sp500-2025.csv) -> documented, known limitation
    - Lookahead: PASS. Costs: PASS. Slippage: PASS. Regime: PASS. State machine: PASS.
- [x] **5. Initial backtest** -> `docs/reports/ramp-wheel/20260331_initial_backtest.md`
- [x] **6. Validate** -> orchestrator
  - Sharpe: 0.231 (full), -0.875 (IS), 0.628 (OOS) -- marginal
  - CRITICAL: Wheel lifecycle never activated. Zero assignments, zero CC trades.
  - Root cause: `left_top_n` exit closes CSPs after avg 3.7 days (before any can reach expiry)
  - This is a DESIGN FIX opportunity, not a reject: wheel owns its shares, so RAMP ranking
    should NOT force CSP exit. Removing left_top_n will let CSPs reach expiry -> assignment -> CC.
  - Decision: PROCEED with design fix (remove left_top_n for CSP, keep for initial stock selection)
  - Re-run backtest after fix, then optimize if promising
  - Run 2: removed left_top_n. Still no assignments. Profit_target/dte_exit exit early.
  - Run 3: v2 config (hold-to-expiry). WHEEL WORKS: 2 assignments, 119 CC trades.
  - Full Sharpe=0.353, OOS Sharpe=0.981. Proceed to optimization.
- [x] **7. Optimize** -> `output/optimization/ramp-wheel/optimization_results.json`
  - 27 combos, 3 WF windows. Parameters: cc_profit_target, cc_loss_limit, cc_delta_strong_bull
  - Best OOS Sharpe: 0.617 (pt=0.70, ll=2.0, delta=0.15) but gap=70% -> REJECT
  - Most stable: pt=0.30, ll=2.0, delta=0.20 -> IS=0.359, OOS=0.359, gap=0.0%
  - All OOS Sharpes < 0.5 for stable combos. Best stable OOS = 0.371.
  - WF1 (bear): OOS=-0.358. WF2 (transition): OOS=0.15-0.22. WF3 (bull): OOS=1.2-1.3.
  - Strongly regime-dependent: positive only in bull markets.
  - Edge is stock direction, not options premium capture.
- [-] **8. Final validation** -> SKIPPED (OOS Sharpe 0.371 < 0.5 threshold; regime fragile)

### Integrity checklist
- [x] shift(1) confirmed on all signals (callback architecture: all data <= d)
- [x] Transaction costs included (1% slippage + $0.02 fee)
- [x] Slippage model active (bid*(1-slip) on entries, current*(1+slip) on exits)
- [x] No lookahead bias detected (code review passed)
- [x] Temporal train/test split (3 WF windows, rolling)
- [x] Regime analysis included (returns by regime in all reports)
- [x] Data frequency matches strategy logic (daily)
- [ ] Universe free of survivorship bias (uses sp500-2025.csv -- known limitation)

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | spec defaults | CSP+CC lifecycle | 0.231 | - | 0.44% | 1.94% | 162d | 0.226 | 25.71% | 1.226 | 60 | 3.8d | STRONG_BULL only | - | IS:-0.875/OOS:0.628 | 2022-2024 | daily | Zero wheel cycles; pure CSP; left_top_n exits 82% |
| 2   | v1 fix: no left_top_n | CSP holds longer | -0.069 | - | -0.20% | 3.91% | 311d | -0.052 | 28.57% | 0.914 | 36 | 10.4d | STRONG_BULL only | - | IS:-0.310/OOS:-0.075 | 2022-2024 | daily | Still zero CC; profit_target/dte_exit close before expiry |
| 3   | v2: hold-to-expiry | CSP/CC expire naturally | 0.353 | - | 2.58% | 11.31% | 206d | 0.228 | 28.57% | 1.048 | 127 | 7.9d | SB/WB/SW | - | IS:-0.206/OOS:0.981 | 2022-2024 | daily | WHEEL WORKS: 2 assign, 119 CC. But returns=stock direction |

### Optimization summary
- Combinations tested: 27 (3x3x3 grid)
- Parameters optimized: cc_profit_target (0.30/0.50/0.70), cc_loss_limit (0.5/1.0/2.0), cc_delta_strong_bull (0.15/0.20/0.30)
- Walk-forward windows: 3 (12mo train, 6mo test, rolling)
- Best stable IS Sharpe: 0.359 | OOS Sharpe: 0.359 | Gap: 0.0% (pt=0.30, ll=2.0, delta=0.20)
- DSR-adjusted Sharpe: not computed (OOS < 0.5, not statistically significant)

### Validation notes (Phase 6)
**Run 1 (spec defaults)**: Wheel lifecycle never activated. Pure CSP due to left_top_n exits.
**Run 2 (no left_top_n)**: CSPs hold longer but profit_target/dte_exit still prevent expiry.
**Run 3 (hold-to-expiry, v2 config)**: WHEEL WORKS. 2 CSP assignments, 119 CC trades.
  - Full: Sharpe 0.353, CAGR 2.58%, Max DD 11.31%
  - OOS: Sharpe 0.981, CAGR 11.84% -- but driven by stock appreciation in bull market
  - OOS CC P&L: -$15,766 (CCs capped upside in rallying market, loss_limit exits expensive)
  - OOS equity gains: $18,243 = stock appreciation ($33k) - CC drag ($15k)
  - IS: Sharpe -0.206, only CSPs (bear market, few STRONG_BULL days)
  - IS/OOS gap: large and regime-dependent
  **Key concern**: Returns are stock-direction dependent, not premium-driven.
  In bull markets: stock appreciation > CC drag -> positive. In bear: CSP losses.
  The "options edge" (premium capture) is thin relative to equity direction risk.
  **Decision**: PROCEED to optimization. Test: cc_profit_target, cc_delta, csp_loss_limit.
  The v2 config (hold-to-expiry) is the right design for the wheel.

### Final validation notes (Phase 8)
Skipped. OOS Sharpe 0.371 < 0.5 threshold. Strategy is regime-fragile (profitable only in bull markets).

### Verdict
- [x] **Final**: NOT VIABLE
- **Reason**: The wheel strategy's returns are driven by stock direction, not options premium.
  In bull markets, stock appreciation exceeds CC drag -> positive returns.
  In bear markets, CSP losses dominate -> negative returns.
  The "options premium edge" is thin: CC P&L was -$15,766 on 119 trades in the OOS bull period.
  Best stable OOS Sharpe = 0.371 (below 0.5 threshold). WF bear window Sharpe = -0.358.
  The wheel does work mechanically (assignments, CC cycling), but the economic edge
  is fundamentally stock-directional, not premium-driven. This makes it inferior to
  simply holding the underlying equity.

  WHY THE "MOMENTUM CONTRADICTION" WAS NOT SOLVED:
  The original hypothesis was that the wheel avoids the CC problem (#9) because it
  manages its own shares (not depending on RAMP daily rebalance). This is true mechanically.
  But the deeper problem remains: in a strong bull market, CCs cap upside and get stopped out
  (loss_limit on 40 trades = -$31k). The wheel pays a high price for the premium it collects.

  LESSON: Premium-selling strategies (CSP, CC, wheel) on momentum stocks face a fundamental
  tension. Momentum stocks are chosen BECAUSE they're expected to move strongly -- which is
  exactly when short options lose money (puts in crashes, calls in rallies).
- **Overfitting risk**: LOW / MEDIUM / HIGH
- **Regime classification**: ROBUST / DEPENDENT / FRAGILE
- **Edge survives 1.5x costs?**: yes / no
- **Recommendation**: ___

---

## Strategy: ramp-systematic-cc (#31 Systematic Covered Call Writing)

**Catalog ref**: #31 -- Feasibility Rank 3/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #31)
**Implementation**: `src/strategies/advanced/ramp_systematic_cc.py`
**Config**: `config/strategies/ramp_systematic_cc.yaml`
**Tests**: `tests/strategies/test_ramp_systematic_cc.py`
**Reports**: `docs/reports/ramp-systematic-cc/`
**Optimization output**: `output/optimization/ramp-systematic-cc/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 1
**Cost tier**: 15 bps
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv`

**Strategy logic summary**:
- Extension of #9: write covered calls on ALL RAMP equity positions (full buy-write overlay)
- Same regime-adaptive delta targeting as #9, applied uniformly across 5-20 positions
- One contract per 100-share lot (fractional lots skipped)
- Must close CC before RAMP can sell underlying on rebalance

**Default parameters**: Same as #9 but applied to all positions.

**Edge**: Amplified premium collection across full portfolio. BXM index research supports reduced vol and improved risk-adjusted returns.
**Key risk**: Collective upside cap in STRONG_BULL. Higher operational complexity managing 5-20 concurrent CCs.
**Dependency**: Builds on #9 (CC) implementation.

### Phases
- [-] **3. Implement** -> SKIPPED: No new code needed (see analysis below)
- [-] **4. Test & review** -> SKIPPED: Nothing to test
- [-] **5. Initial backtest** -> SKIPPED: Identical to ramp-cc results
- [-] **6. Validate** -> orchestrator -> see analysis below
- [-] **7. Optimize** -> SKIPPED: Same engine, same results
- [-] **8. Final validation** -> SKIPPED: NOT VIABLE (same as ramp-cc)

### Critical Finding: selection_mode has NO behavioral effect

**Analysis date:** 2026-03-31

The CC engine (`src/strategies/options/cc/engine.py`) accepts a `selection_mode` parameter
("selective" or "all") but **never uses it in any conditional logic**. The parameter is stored
at line 67 (`self.selection_mode = selection_mode`) but never referenced again.

The `_scan_entries()` method (lines 257-327) already iterates over ALL held positions:
```
for symbol, shares in held_positions.items():
    if symbol not in options_set: continue
    if symbol in held_cc_symbols: continue
    if shares < 100: continue
    # ... writes a call on every eligible position
```

This means ramp-cc with `selection_mode="selective"` and ramp-systematic-cc with
`selection_mode="all"` produce **identical backtest results**. There is no filtering
logic that would differentiate "selective" from "all" -- the engine already writes
calls on every held position that has options data and >= 100 shares.

**Conclusion:** Systematic-CC is architecturally identical to CC (#9). The ramp-cc
backtest results (37 trades, OOS Sharpe 0.097, $566.66 total P&L over 3 years)
ARE the systematic-cc results. No separate implementation or backtest is needed.

Even if we added true selective logic (e.g., only write CCs on top-3 momentum names),
the fundamental contradiction identified in ramp-cc still applies:
1. Selling calls on momentum stocks caps the upside RAMP is designed to capture
2. RAMP turnover is too high (60-70% of trades exit within 3 days as "underlying_sold")
3. Premium income ($566 over 3 years on $100K) is negligible

### Integrity checklist
- [x] shift(1) confirmed (inherited from CC engine -- callback architecture filters data <= d)
- [x] Costs 15 bps (1% slippage + $0.02/contract in CC engine)
- [x] Slippage (sells at bid*(1-0.01), buys at mid*(1+0.01))
- [x] No lookahead (inherited from CC engine)
- [x] Temporal split (IS 2022-01 to 2023-06, OOS 2023-07 to 2024-12)
- [x] Regime (regime-adaptive delta targeting in CC engine)
- [x] Freq match (daily)
- [x] Universe ok (11 S&P 500 stocks with ThetaData options)

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | ramp-cc results (identical engine) | selection_mode=all | 0.247 | n/a | +0.19% | 1.27% | n/a | 0.15 | 48.6% | n/a | 37 | 2.7d | entry regimes only | n/a | IS 0.707 vs OOS 0.117 (83% gap) | 2022-2024 | daily | Engine has no selective/all differentiation -- results identical to ramp-cc |

### Verdict
- [x] **Final**: NOT VIABLE
  - **Reason**: selection_mode parameter is a no-op in the CC engine. Results are identical to ramp-cc (#9) which was NOT VIABLE (OOS Sharpe 0.097, fundamental momentum contradiction, high RAMP turnover). No separate implementation needed.
  - **Reference**: `docs/reports/ramp-cc/20260331_initial_backtest.md` (same results apply)

---

## Strategy: ramp-long-calls (#1 Long Calls on Top Momentum)

**Catalog ref**: #1 -- Feasibility Rank 6/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #1)
**Implementation**: `src/strategies/advanced/ramp_long_calls.py`
**Config**: `config/strategies/ramp_long_calls.yaml`
**Tests**: `tests/strategies/test_ramp_long_calls.py`
**Reports**: `docs/reports/ramp-long-calls/`
**Optimization output**: `output/optimization/ramp-long-calls/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 2
**Cost tier**: 15 bps
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv`

**Strategy logic summary**:
- Buy calls (delta 0.50-0.70, 30-60 DTE) on RAMP top_n momentum stocks instead of buying equity
- Entry: strong momentum signal + STRONG_BULL or WEAK_BULL regime
- Exit: RAMP sell signal, DTE <= 7, regime to BEAR/UNPREDICTABLE, 100% profit target
- Premium per position <= 3-5% of portfolio, defined max loss

**Default parameters**:
- call_delta: 0.60
- min_dte: 30, max_dte: 60
- position_alloc_pct: 0.04
- profit_target_pct: 1.00

**Edge**: Leveraged momentum capture at 3-8% capital risk per position. Capital-efficient.
**Key risk**: Theta decay kills the position if momentum move is slow.

### Phases
- [x] **3. Implement** -> orchestrator direct (following CSP/CC pattern)
  - Files: `src/strategies/options/long_calls/{__init__,position,contract_selector,mark_to_market,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_long_calls.yaml`
  - All imports verified, smoke tests passed (position P&L, engine with mock data)
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass -> `tests/strategies/options/long_calls/` (76/76 passed)
  - [x] 4b. Code review pass -> `docs/agent-learnings/ramp-long-calls/04_review.md`
    - No critical/high issues. Lookahead: PASS. Costs: PASS. Slippage: PASS. Regime: PASS. P&L direction: PASS.
- [x] **5. Initial backtest** -> `docs/reports/ramp-long-calls/20260331_initial_backtest.md`
- [x] **6. Validate** -> orchestrator -> notes below (PROCEED to optimize with structural changes)
- [x] **7. Optimize** -> `output/optimization/ramp-long-calls/optimization_results.json`
  - 18 combos, 3 WF windows (54 runs), 26.4 minutes
  - **KEY FINDING**: exit_on_left_top_n is THE decisive parameter
    - exit_lt=True avg OOS Sharpe: -0.762 (ALL 9 combos negative)
    - exit_lt=False avg OOS Sharpe: +0.525 (ALL 9 combos positive)
  - Best: C13 (exit_lt=False, d=[0.40,0.60], tp=0.75) avg OOS Sharpe=0.872
  - Per-window: W1=1.276, W2=2.104, W3=-0.764 (W3 concerning)
  - [!] W2 Sharpe > 2.0: not overfitting, but concentrated in strong rally period
  - [!] W3 universally negative: strategy fails in choppy/rotating markets
- [x] **8. Final validation** -> `docs/reports/ramp-long-calls/20260331_final_validation.md`

### Integrity checklist
- [x] shift(1) confirmed (callback architecture, all data filtered <= d)
- [x] Costs 15 bps (1% slippage + $0.02/contract fee)
- [x] Slippage (buys at ask*(1+slip), sells at mid*(1-slip))
- [x] No lookahead bias detected (95% confidence)
- [x] Temporal split (IS: 2022-01 to 2023-06, OOS: 2023-07 to 2024-12)
- [x] Regime analysis included (entry in SB/WB, exit on BEAR/UNPREDICTABLE)
- [x] Data frequency matches (daily/EOD)
- [!] Universe: current SP500 list (mild survivorship bias). Only 11 symbols have options data.

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | spec defaults | d=0.50-0.70, DTE=30-60, tp=100%, alloc=4%, exit_lt=True | -0.326 | - | -2.83% | 13.85% | 553d | -0.204 | 41.5% | 0.84 | 135 | 2.9d | SB:-$8158/WB:-$67 | - | IS:-1.124/OOS:-0.155 | 2022-2024 | daily | 116/135 left_top_n exits (86%). Only 3 profit targets. Avg hold 2.9d kills theta. |
| 2   | optimizer best (WF) | d=0.40-0.60, tp=75%, exit_lt=False | - | - | - | - | - | - | - | - | 70 OOS | 15.6d | - | - | W1:1.28/W2:2.10/W3:-0.76 avg=0.872 | 3 WF windows | daily | Best of 18 combos. exit_lt=False transforms strategy. W3 negative (choppy market). |
| 3   | final validation | d=0.40-0.60, tp=75%, exit_lt=False | 0.698 | - | 13.64% | 24.18% | 301d | 0.564 | 47.7% | 1.33 | 88 | 16.3d | SB/WB entries | 0.687 | IS:0.591/OOS:0.799 | 2022-2024 | daily | 39 profit targets (44%). 1.5x Sharpe=0.687. TP stable. Delta sensitive. |
| 4   | stat validation (IS) | d=0.40-0.60, tp=75%, exit_lt=False | -0.767 | DSR p=1.0 | n/a | 20.5% | 877d | n/a | 37.0% | 0.532 | 73 | n/a | ALL negative | n/a | n/a | 2018-07 to 2024-12 | daily | EXTENDED IS: strategy loses money. Bootstrap P(Sharpe>0)=0.93%. Kurtosis=71.0. |
| 5   | stat validation (OOS 2025) | d=0.40-0.60, tp=75%, exit_lt=False | -1.313 | n/a | -7.2% | 9.86% | n/a | n/a | 35.3% | 0.479 | 34 | n/a | n/a | n/a | n/a | 2025-01 to 2025-12 | daily | TRUE OOS: strategy hemorrhages money. PF=0.479 (loses $2.09 per $1 gained). |

### Validation notes (Phase 6)

**Checklist:**
- [x] Sharpe < 3.0: YES (-0.326) -- no overfitting concern (strategy loses money)
- [x] CAGR < 20%: YES (-2.83%) -- negative return
- [x] Max DD > 5%: YES (13.85%) -- significant for an options-only allocation
- [x] Trades > 30: YES (135 full, 117 OOS) -- sufficient for statistical conclusions
- [!] Regime robust: 100% entries in SB/WB (by design). SB accounts for -$8,158, WB for -$67
- [x] Transaction costs included
- [x] Slippage modeled
- [x] No lookahead confirmed
- [x] Data frequency matches

**Key observations:**
1. **THE CRITICAL FINDING: left_top_n exit causes 86% of all exits (116/135).** Avg hold = 2.9 days. This confirms the hypothesis concern: RAMP top_n rotates stocks too quickly for 30-60 DTE calls. Theta and bid-ask spread eat the position alive in 2.9 days.
2. **Only 3 profit targets hit (2.2%).** 100% gain on premium in 2.9 days requires a massive move -- essentially impossible for large-cap stocks.
3. **Regime change exits are the worst: -$13,171 across 16 trades.** When regime shifts to BEAR/UNPREDICTABLE, calls lose value rapidly (underlying drops + IV may crush delta gains).
4. **OOS closer to breakeven than IS** (PF 0.93 vs 0.39). IS was dominated by 2022 bear market with few but devastating entries.
5. **Total premium deployed: $412K over 3 years on a $100K portfolio.** Very high turnover of premium capital due to rapid exits and re-entries.
6. **Comparison to CSP/CC:** All three strategies suffer the same structural problem -- RAMP top_n turnover is too fast for options holding periods. CSP had 82% left_top_n exits, CC had 60-70%, long calls have 86%. The difference is that CSP/CC collect time value (theta works for them), so rapid exits still capture some premium. Long calls PAY time value, so rapid exits guarantee loss.

**Structural diagnosis:**
The strategy hypothesis was: "buying calls ALIGNS with momentum signal -- you profit when the stock rallies." This is TRUE but INSUFFICIENT. The alignment is correct (directional bet matches signal), BUT the holding period is too short for calls to profit. Delta gains over 2.9 days are smaller than theta decay + bid-ask spread for 30-60 DTE calls.

**Optimization opportunities:**
1. **Remove left_top_n exit** -- hold calls until DTE exit, profit target, or regime change. This lets the momentum play out over the full 30-60 day window.
2. **Lower delta (0.30-0.50)** -- cheaper premium, more leverage per dollar.
3. **Lower profit target (50% instead of 100%)** -- more achievable within holding period.

**Decision:** PROCEED to Phase 7 with modified exit logic. Test removing left_top_n as exit condition.

### Optimization summary
- Combinations tested: 18 (2 x 3 x 3 grid)
- Total runs: 54 (18 combos x 3 WF windows)
- Parameters optimized: 3 (exit_on_left_top_n, delta_range, profit_target_pct)
- Walk-forward windows: 3 (W1: IS 2022-01/2023-03, OOS 2023-04/09 | W2: IS 2022-07/2023-09, OOS 2023-10/2024-03 | W3: IS 2023-01/2024-03, OOS 2024-04/09)
- Best combo: C13 (exit_lt=False, d=[0.40,0.60], tp=0.75)
- Avg OOS Sharpe: 0.872 | Per-window: W1=1.276, W2=2.104, W3=-0.764
- **CRITICAL FINDING**: exit_on_left_top_n is THE decisive parameter
  - exit_lt=True: avg OOS Sharpe = -0.762 (ALL 9 combos negative, avg hold 2.8d)
  - exit_lt=False: avg OOS Sharpe = +0.525 (ALL 9 combos positive, avg hold 13-18d)
- Parameter sensitivity: delta=MODERATE (0.267-0.698 at +/-10%), tp=STABLE (0.622-0.717 at +/-10%)
- Cost sensitivity (1.5x): Sharpe 0.687 (minimal degradation from 0.698)
- Magic numbers flagged: NONE
- Estimated runtime: ~30 min | Actual runtime: 26.4 min

### Final validation notes (Phase 8)

**1. Standard costs (1% slippage, optimized params):**
- Full period: Sharpe 0.698, CAGR 13.64%, MaxDD 24.18%, 88 trades, 47.7% win rate, PF 1.33
- IS: Sharpe 0.591, 14 trades | OOS: Sharpe 0.799, 74 trades
- Exit breakdown: profit_target=39 ($185K), regime_change=32 (-$71K), dte_exit=17 (-$68K)
- Avg hold: 16.3 days (vs 2.9 days with exit_lt=True)

**2. 1.5x costs (1.5% slippage):**
- Sharpe 0.687 (only 1.6% degradation from standard)
- Edge survives higher costs

**3. Parameter stability:**
- Delta -10% [0.36, 0.54]: Sharpe 0.451
- Delta +10% [0.44, 0.66]: Sharpe 0.267
- TP -10% (67.5%): Sharpe 0.622
- TP +10% (82.5%): Sharpe 0.717
- **Delta is SENSITIVE** -- works best at [0.40, 0.60], degrades at neighbors
- **Profit target is STABLE** -- 0.622 to 0.717 across +/-10%

### Verdict (Phase 8)
- [x] **Phase 8 Final**: VIABLE (with caveats) -- on 2022-2024 window only
- **Reason**: Full-period Sharpe 0.698, OOS Sharpe 0.799, CAGR 13.64%, PF 1.33. exit_on_left_top_n=False transforms the strategy from -0.326 to +0.698 Sharpe. 1.5x costs: 0.687. Walk-forward validated avg OOS Sharpe 0.872 across 3 windows.
- **Overfitting risk**: MODERATE. Delta sensitive. W3 window negative. But exit_lt is structural (not magic number) and profit_target is stable.
- **Regime classification**: REGIME-DEPENDENT. Profitable in trending markets, loses in choppy. 39 profit targets vs 32 regime_change exits.
- **Edge survives 1.5x costs?**: YES (0.687 Sharpe, minimal degradation)

### Verdict (Phase 9 -- SUPERSEDES Phase 8)
- [x] **Final**: NOT VIABLE
- **Reason**: Statistical validation FAILED all 3 HARD gates tested. Extended timeline (2018-2024) reveals IS Sharpe of -0.767 (strategy loses money). True OOS 2025 Sharpe = -1.313. Bootstrap gives 0.93% probability of positive Sharpe. Strategy loses money in ALL 5 market regimes over the full period. The 2022-2024 Sharpe of 0.698 was a window-specific artifact, not a persistent edge.
- **Key insight**: Options theta decay and bid-ask spread consume any directional edge from momentum. The alternative signal test (Sharpe 1.129 on equities) confirms the underlying momentum signal works -- but the options overlay destroys it.
- **Recommendation**: DO NOT DEPLOY. The ramp-long-calls strategy does not have a statistically significant edge. The RAMP momentum signal has value, but options are the wrong instrument to express it. Consider equity-only implementation if pursuing this further.

### Phase 9: Statistical Validation Suite
**Goal**: Rigorous statistical testing before paper trading deployment.
**Runner**: `scripts/backtest_scripts/ramp_long_calls_validation.py`
**Modules**: `src/backtesting/validation/` (deflated_sharpe.py, bootstrap.py, cpcv.py, permutation.py)
**Report**: `docs/reports/ramp-long-calls/`

| Test | Name | Gate | Criterion | Status | Result |
|------|------|------|-----------|--------|--------|
| 1 | Deflated Sharpe Ratio | HARD | p < 0.05 | [!] FAIL | p=1.000, observed Sharpe=-0.767, DSR stat=-33.92 |
| 2 | True OOS 2025 | HARD | Sharpe > 0.3 | [!] FAIL | Sharpe=-1.313, CAGR=-7.2%, 34 trades, WR=35%, PF=0.479 |
| 3 | Bootstrap CI | HARD | CI lower > 0 | [!] FAIL | CI=[-3.566, 0.000], P(positive)=0.93%, PF CI=[0.257, 0.989] |
| 4 | CPCV | SOFT | >70% positive | [-] SKIPPED | Skipped -- all 3 HARD gates already failed |
| 5 | Alternative Signal | INFO | Informational | [x] | Alt Sharpe=1.129, 464 trades, WR=44.2% (alt signal beats RAMP) |
| 6 | Permutation | HARD | p < 0.05 | [-] SKIPPED | Skipped -- all 3 HARD gates already failed |
| 7 | Regime Stress | INFO | Informational | [x] | MaxDD=20.5%, 877d duration. NEGATIVE returns in ALL 5 regimes |

**Go/No-Go**: **DO NOT PROCEED**. 0/3 HARD gates passed. Tests 4 and 6 skipped (2-5 hours compute saved).

**Extended timeline analysis (2018-07 to 2024-12):**
The previous validation used 2022-2024 only (3 years). When extended to 6.5 years (2018-07 to 2024-12):
- IS Sharpe drops from +0.591 to **-0.767** (the edge was window-specific)
- Strategy loses money in ALL regimes (SB: -0.030%/day, WB: -0.037%/day, SIDEWAYS: -0.071%/day, BEAR: -0.037%/day, UNPREDICTABLE: -0.364%/day)
- Bootstrap: only 0.93% chance the Sharpe is positive
- 2025 True OOS: Sharpe -1.313, confirming the strategy has no real edge

**Root cause**: The 2022-2024 Sharpe of 0.698 was driven by a specific market regime window (strong tech momentum 2023-2024). Extended data reveals this was not a persistent edge. The options structure (theta decay, bid-ask spread) makes the strategy a net loser over the full cycle.

**Alt signal finding**: The alternative signal (likely SMA-based) achieved Sharpe 1.129 over the same extended period with 464 trades. This suggests the options implementation destroys whatever edge exists in the underlying momentum signal -- the cost of options (theta, spread) exceeds the directional gain.

**Report**: `docs/reports/ramp-long-calls/20260402_statistical_validation.md` and `.json`

---

## Strategy: ramp-long-puts (#3 Long Puts on Bottom Momentum)

**Catalog ref**: #3 -- Feasibility Rank 7/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #3)
**Implementation**: `src/strategies/advanced/ramp_long_puts.py`
**Config**: `config/strategies/ramp_long_puts.yaml`
**Tests**: `tests/strategies/test_ramp_long_puts.py`
**Reports**: `docs/reports/ramp-long-puts/`
**Optimization output**: `output/optimization/ramp-long-puts/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 2
**Cost tier**: 15 bps
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv`

**Strategy logic summary**:
- Buy puts (delta -0.50 to -0.70, 30-60 DTE) on RAMP bottom_n momentum stocks (weakest momentum)
- Entry: BEAR or WEAK_BULL regime
- Exit: momentum rank improves, DTE <= 7, regime to STRONG_BULL, 100% profit target
- Captures short side of cross-sectional momentum without borrowing shares

**Default parameters**:
- put_delta: -0.60
- min_dte: 30, max_dte: 60
- position_alloc_pct: 0.04
- profit_target_pct: 1.00
- max_positions: 5

**Edge**: Cross-sectional momentum is long/short -- weak names underperform. Defined risk (premium only).
**Key risk**: Short squeezes on weak stocks. Elevated IV on weak names makes puts expensive.
**Note**: Requires implementing bottom_n ranking (scoring already computed, just select lowest).

### Phases
- [x] **3. Implement** -> orchestrator direct (following Long Calls pattern)
  - Files: `src/strategies/options/long_puts/{__init__,position,contract_selector,mark_to_market,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_long_puts.yaml`
  - Key design choices:
    - exit_on_left_bottom_n=False by default (lesson from long-calls: holding to DTE/profit/regime is better)
    - Entry regimes: BEAR, WEAK_BULL (not STRONG_BULL/WEAK_BULL like long-calls)
    - Emergency exit on STRONG_BULL (not BEAR/UNPREDICTABLE like long-calls)
    - get_bottom_n_symbols uses momentum_scores.nsmallest(n) instead of generate_signals top_n
    - bottom_n_size=10 configurable in YAML
  - All imports verified, smoke tests passed (position P&L, B-S put pricing)
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass -> `tests/strategies/options/long_puts/` (79/79 passed)
  - [x] 4b. Code review pass -> `docs/agent-learnings/ramp-long-puts/04_review.md`
    - No critical/high issues. Lookahead: PASS. Costs: PASS. Slippage: PASS. Regime: PASS. Bottom-N: PASS.
- [x] **5. Backtest** -> `docs/reports/ramp-long-puts/20260331_initial_backtest.md`
- [x] **6. Validate** -> orchestrator -> notes below (NOT VIABLE)
- [-] **7. Optimize** -> SKIPPED (negative Sharpe, regime_change exits dominate, no edge to optimize)
- [-] **8. Final validation** -> SKIPPED (no optimization to validate)

### Integrity checklist
- [x] shift(1) confirmed (callback architecture, all data filtered <= d)
- [x] Costs 15 bps (1% slippage + $0.02/contract fee)
- [x] Slippage (buys at ask*(1+slip), sells at mid*(1-slip))
- [x] No lookahead bias detected (code review passed)
- [x] Temporal split (IS: 2022-01 to 2023-06, OOS: 2023-07 to 2024-12)
- [x] Regime analysis (entry BEAR/WEAK_BULL, exit STRONG_BULL)
- [x] Freq match (daily/EOD)
- [!] Universe: sp500-2025.csv (mild survivorship bias). Only 11 symbols have options data.

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | spec defaults (exit_lb=False) | d=[-0.70,-0.50], DTE=30-60, tp=100% | -0.803 | - | -9.45% | 27.79% | 412d | -0.340 | 35.0% | 0.45 | 40 | 15.3d | BEAR:-$2.3K/WB:-$23.4K | - | IS:-1.426/OOS:-0.656 | 2022-2024 | daily | 31/40 regime_change exits (-$35K). Only 4 profit targets (+$13K). |
| 2   | exit_lb=True comparison | d=[-0.70,-0.50], DTE=30-60, tp=100% | -1.395 | - | -7.68% | 21.76% | 412d | -0.353 | 39.7% | 0.34 | 58 | 2.8d OOS | BEAR:-$3.5K/WB:-$17.7K | - | IS:-0.764/OOS:-1.559 | 2022-2024 | daily | exit_lb=True WORSE. Confirms long-calls lesson: holding is better. |

### Validation notes (Phase 6)

**Checklist:**
- [x] Sharpe < 3.0: YES (-0.803) -- no overfitting concern (strategy loses money)
- [x] CAGR < 20%: YES (-9.45%) -- negative return
- [x] Max DD > 5%: YES (27.79%) -- significant
- [x] Trades > 30: YES (40 full, 33 OOS) -- sufficient for statistical conclusions
- [x] Regime breakdown: WEAK_BULL -$23.4K, BEAR -$2.3K -- loses in both entry regimes
- [x] Transaction costs included
- [x] Slippage modeled
- [x] No lookahead confirmed
- [x] Data frequency matches

**Key observations:**
1. **regime_change exits dominate: 31/40 trades (77.5%) with -$35,148 total P&L.** When regime shifts from BEAR/WEAK_BULL to STRONG_BULL, puts lose value rapidly (underlying rallies + IV crush).
2. **Only 4 profit target hits (+$13,341).** Puts rarely double because: (a) weak stocks don't always decline fast enough, (b) IV expansion already priced into the put at entry (weak stocks have elevated IV), (c) regime shifts to bullish before the decline materializes.
3. **WEAK_BULL is the worst regime: -$23,409.** The strategy enters WEAK_BULL expecting decline, but WEAK_BULL often transitions to STRONG_BULL rather than BEAR. The put thesis requires continued weakness or acceleration to BEAR.
4. **BEAR regime contributed only -$2,319** because: (a) very few BEAR regime days in the backtest period, (b) BEAR periods often have elevated IV = expensive puts, (c) by the time BEAR is detected, much of the decline has already happened.
5. **Avg holding time 15.3d (exit_lb=False) vs 2.8d (exit_lb=True).** Confirms the long-calls lesson: holding is better. But even 15d is not enough -- puts need a sustained decline that outpaces theta decay.
6. **A/B test confirms long-calls lesson:** exit_on_left_bottom_n=True (Sharpe -1.395) is WORSE than False (Sharpe -0.803). Holding to DTE/profit/regime is structurally better.

**Root cause analysis:**
The strategy has a FUNDAMENTAL SIGNAL TIMING PROBLEM:
1. **Regime detection is lagging:** By the time the regime detector classifies BEAR or WEAK_BULL, the decline has already happened (regime uses trailing indicators). Buying puts after the decline is buying expensive insurance after the house burned down.
2. **IV on weak stocks is elevated at entry:** Bottom_n stocks (weakest momentum) already have elevated implied volatility. The puts are EXPENSIVE. For the put to profit (100% gain), the underlying needs to decline significantly beyond what's already priced in.
3. **Regime transitions to STRONG_BULL are frequent and devastating:** 31 out of 40 trades exit via regime_change. The market cycles between WEAK_BULL and STRONG_BULL frequently. Each transition wipes out the put position.
4. **Cross-sectional momentum's short side is weaker than long side:** Academic research shows momentum's alpha comes primarily from the long side (winners continue winning). The short side (losers continue losing) is noisier and has higher volatility, mean reversion tendencies, and short squeeze risk.

**Comparison to long-calls (#1):**
| Metric | Long Calls (exit_lt=False) | Long Puts (exit_lb=False) |
|--------|---------------------------|--------------------------|
| Sharpe | 0.698 | -0.803 |
| Win rate | 47.7% | 35.0% |
| Profit target hits | 39/88 (44%) | 4/40 (10%) |
| Regime change exits | 32/88 (36%) | 31/40 (77.5%) |
| Avg hold | 16.3d | 15.3d |

Long calls work because: (a) momentum's long side is stronger, (b) STRONG_BULL/WEAK_BULL regimes are persistent (trending markets last), (c) top_n stocks continue rallying.
Long puts fail because: (a) momentum's short side is noisier, (b) BEAR/WEAK_BULL regimes are SHORT-LIVED (markets recover quickly), (c) IV on weak stocks is already elevated.

**Decision:** NOT VIABLE. No optimization warranted -- the fundamental edge is absent. Skipping Phases 7 and 8.

### Verdict
- [x] **Final**: NOT VIABLE
- **Reason**: Sharpe -0.803 (full), IS -1.426, OOS -0.656. Loses money in both BEAR and WEAK_BULL entry regimes. 77.5% of trades exit via regime_change (STRONG_BULL) with -$35K total loss. Only 4/40 trades hit profit target. The strategy's premise -- buying puts on weak-momentum stocks in bearish regimes -- fails because: (1) regime detection is lagging (decline already priced in), (2) IV on weak stocks is elevated (puts are expensive), (3) BEAR/WEAK_BULL regimes are short-lived (frequent transitions to STRONG_BULL devastate put positions), (4) cross-sectional momentum's short side is academically weaker than long side. A/B test confirms exit_on_left_bottom_n=True is worse (Sharpe -1.395).
- **Overfitting risk**: N/A (strategy loses money, not overfit)
- **Regime classification**: N/A (not profitable in any regime)
- **Edge survives 1.5x costs?**: No (edge is negative at 1x costs)
- **Recommendation**: ARCHIVE. The short side of cross-sectional momentum does not translate to profitable options trading in this framework. If revisiting: (1) consider using put SPREADS instead of naked long puts to reduce IV exposure (see #4 Bear Put Spreads), (2) implement a VIX-relative IV filter to avoid buying puts when IV is already elevated, (3) use momentum's short side for equity short sales rather than put buying. The implementation and infrastructure are solid -- the signal is not actionable as specified.

---

## Strategy: ramp-deep-itm (#6 Deep ITM Call Replacement)

**Catalog ref**: #6 -- Feasibility Rank 8/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #6)
**Implementation**: `src/strategies/options/deep_itm/` (engine, position, contract_selector, mark_to_market, metrics, ramp_integration)
**Config**: `config/strategies/ramp_deep_itm.yaml`
**Tests**: `tests/strategies/options/deep_itm/` (6 test files, 80 tests)
**Reports**: `docs/reports/ramp-deep-itm/`
**Optimization output**: `output/optimization/ramp-deep-itm/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 2
**Cost tier**: 15 bps
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv`

**Strategy logic summary**:
- Hold deep ITM calls (delta 0.75-0.85, 60-90 DTE) instead of stock on RAMP top_n names
- Replicates equity exposure using 70-85% of capital, freeing 15-30% for other strategies
- Same entry/exit as RAMP equity, but roll before DTE gets short
- Freed capital can be deployed in CSP (#8) or other strategies

**Default parameters**:
- call_delta: 0.80
- min_dte: 60, max_dte: 90
- roll_at_dte: 21

**Edge**: Capital efficiency -- same momentum exposure, less capital deployed. Freed capital earns additional return.
**Key risk**: Time value decay (2-5% over 60 days if flat). Early assignment risk near ex-dividend dates.

### Phases
- [x] **3. Implement** -> orchestrator direct (following long_calls pattern)
  - Files: `src/strategies/options/deep_itm/{__init__,position,contract_selector,mark_to_market,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_deep_itm.yaml`
  - Key differences from long_calls: higher delta (0.75-0.85), longer DTE (60-90), roll mechanics at DTE<=21, no exit_on_left_top_n, SIDEWAYS entry regime, equity_replacement_pct sizing
  - All imports verified, 80 unit tests pass
- [~] **4. Test & review**
  - [x] 4a. Unit tests pass -> `tests/strategies/options/deep_itm/` (80/80 passed)
  - [x] 4b. Code review pass -> `docs/agent-learnings/ramp-deep-itm/04_review.md`
    - No critical/high issues. Lookahead: PASS. Costs: PASS. Slippage: PASS. Regime: PASS. left_top_n: PASS (correctly uses ramp_held not top_n). Roll mechanics: PASS.
- [x] **5. Backtest** -> `docs/reports/ramp-deep-itm/20260401_initial_backtest.md`
  - Run 1: ramp_sell exit active (avg hold 1.7d, Sharpe -1.264). Same left_top_n problem.
  - Run 2: ramp_sell exit REMOVED (avg hold 13.8d, Sharpe 0.018). Still losing due to theta + regime exits.
- [x] **6. Validate** -> orchestrator -> notes below (NOT VIABLE)
- [-] **7. Optimize** -> SKIPPED (fundamental economic flaw)
- [-] **8. Final validation** -> SKIPPED

### Integrity checklist
- [x] shift(1) confirmed (callback architecture, all data filtered <= d)
- [x] Costs 15 bps (1% slippage + $0.02/contract fee, 2x on rolls)
- [x] Slippage (buys at ask+slip, sells at mid-slip)
- [x] No lookahead bias detected (95% confidence)
- [x] Temporal split (IS: 2022-01 to 2023-06, OOS: 2023-07 to 2024-12)
- [x] Regime analysis included (entry: SB/WB/SW, exit: BEAR/UNPREDICTABLE)
- [x] Data frequency matches (daily/EOD)
- [!] Universe: current SP500 list (mild survivorship bias). Only 11 symbols have options data.

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | spec defaults (ramp_sell exit active) | d=0.75-0.85, DTE=60-90, tp=20%, roll=21 | -1.264 | - | -21.9% | 52.99% | 692d | -0.413 | 38.2% | 0.58 | 207 | 1.7d | all negative | - | IS:-1.262/OOS:-1.598 | 2022-2024 | daily | 190/207 ramp_sell exits. Same left_top_n problem. 0 rolls. |
| 2   | ramp_sell exit removed | d=0.75-0.85, DTE=60-90, tp=20%, roll=21 | 0.018 | - | -7.86% | 52.81% | 301d | -0.149 | 58.3% | 0.90 | 96 | 13.8d | SB:-$10.9K/WB:-$7.6K/SW:-$3.6K | - | IS:-0.133/OOS:-0.018 | 2022-2024 | daily | 55 profit targets (+$208K), 32 regime exits (-$188K), 5 rolls (-$42K). Roll cost = $42K. vs Equity: Sharpe diff -0.482, DD diff +41%. |

### Validation notes (Phase 6)

**Checklist:**
- [x] Sharpe < 3.0: YES (0.018) -- no overfitting concern (near-zero Sharpe)
- [x] CAGR < 20%: YES (-7.86%) -- negative return
- [x] Max DD > 5%: YES (52.81%) -- catastrophic for equity replacement
- [x] Trades > 30: YES (96 full, 86 OOS) -- sufficient for statistical conclusions
- [!] Regime robust: ALL regimes negative (SB:-$10.9K, WB:-$7.6K, SW:-$3.6K)
- [x] Transaction costs included
- [x] Slippage modeled
- [x] No lookahead confirmed
- [x] Data frequency matches

**Key observations:**
1. **Avg hold 13.8 days after removing ramp_sell exit.** Massive improvement over Run 1 (1.7d), but still short for 60-90 DTE calls. Only 5 positions lasted to DTE roll.
2. **Profit target hits: 55 trades (+$208K).** The strategy finds profitable trades when momentum continues and regime stays favorable.
3. **Regime change exits: 32 trades (-$188K).** Catastrophic. Deep ITM calls (delta 0.80) lose 80% of the stock's decline. When regime shifts to BEAR/UNPREDICTABLE, these positions are hit with full equity-like losses PLUS the theta they've paid.
4. **Roll cost: $42K on only 5 rolls.** Deep ITM calls have wide bid-ask spreads in absolute dollar terms (premium is $30-40). Each roll costs ~$8K in slippage + fees.
5. **Max DD 52.81% vs equity 11.56%.** Deep ITM performs MUCH worse than equity during drawdowns. This is because: (a) options amplify timing risk -- a 10% stock decline can mean 15-20% option loss due to gamma/theta, (b) regime change forces liquidation at the worst time, (c) equity simulator doesn't have forced-sell timing.
6. **Capital saved: 0%.** The equity comparison shows 0% capital savings because both start with $100K. In theory, deep ITM uses ~80% of capital, but the freed 20% would need to earn ~40% annually to offset the -7.86% CAGR drag.
7. **Fundamental economic problem:** Deep ITM calls as equity replacement only work if: (a) theta drag is minimal (~2-5% over 60 days at most), AND (b) you hold through expiry or to profit target most of the time. Here, regime changes force liquidation at losses 33% of the time, destroying the strategy.
8. **Comparison to long_calls (#1):** Long calls had Sharpe 0.698 with lower delta (0.40-0.60) -- cheaper premium, more leverage per dollar, lower absolute bid-ask cost. Deep ITM's higher delta means MORE capital deployed (defeating the purpose) and higher absolute transaction costs.

**Root cause analysis:**
The strategy premise is flawed for this regime-based framework:
- Deep ITM calls are designed for BUY AND HOLD equity replacement (LEAPS)
- RAMP's regime switching creates entry/exit timing that destroys the hold-to-expiry economics
- The theta + slippage costs of repeatedly entering/exiting deep ITM calls exceed the capital savings benefit
- Compare: buying stock has zero theta decay and much tighter bid-ask spreads

**Decision:** NOT VIABLE. No optimization warranted. The fundamental economic problem (theta + slippage > capital savings benefit, amplified by regime switching) cannot be fixed with parameter tuning.

### Verdict
- [x] **Final**: NOT VIABLE
- **Reason**: Sharpe 0.018 (full period), CAGR -7.86%, MaxDD 52.81%. Deep ITM calls as RAMP equity replacement lose money because: (1) theta drag (~$42K in 5 rolls alone) exceeds capital savings benefit, (2) regime change exits at BEAR/UNPREDICTABLE create -$188K in losses (delta 0.80 = 80% of equity downside), (3) bid-ask spreads on deep ITM calls are wide in absolute terms ($1-3 per contract), (4) the strategy uses MORE capital than equity for the same exposure (premium = 70-85% of stock price), while adding theta decay.
- **Overfitting risk**: N/A (strategy loses money, not overfit)
- **Regime classification**: FRAGILE (all regimes negative)
- **Edge survives 1.5x costs?**: No (edge is negative at 1x costs)
- **Recommendation**: ARCHIVE. Deep ITM call replacement only works for passive buy-and-hold (LEAPS held 6-12 months minimum). RAMP's regime switching creates too much entry/exit turnover for the economics to work. The freed capital (15-30%) cannot compensate for the theta + slippage drag. If revisiting: consider 12+ month LEAPS with no regime-based exits (pure equity replacement for tax/capital efficiency, not trading).

---

## Strategy: ramp-straddle (#15 Long Straddles in UNPREDICTABLE)

**Catalog ref**: #15 -- Feasibility Rank 10/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #15)
**Implementation**: `src/strategies/options/straddle/`
**Config**: `config/strategies/ramp_straddle.yaml`
**Tests**: `tests/strategies/options/straddle/`
**Reports**: `docs/reports/ramp-straddle/`
**Optimization output**: `output/optimization/ramp-straddle/`

**Asset class**: Large-cap index / equities (SPY or top momentum names)
**Options level**: Level 2 (two separate buy-to-open orders) or Level 3 (MLeg)
**Cost tier**: 10-15 bps
**Data frequency**: daily
**Universe**: SPY (primary) or top momentum names with options data

**Strategy logic summary**:
- Buy ATM call + ATM put on same underlying when regime = UNPREDICTABLE
- 21-45 DTE to give the resolution event time to develop
- Exit: regime resolves (leaves UNPREDICTABLE), DTE <= 7, 100% profit target on total position
- Profits from large moves in either direction during regime uncertainty
- Total premium paid <= 4% of portfolio per straddle

**Default parameters**:
- call_delta: 0.50
- put_delta: -0.50
- min_dte: 21, max_dte: 45
- position_alloc_pct: 0.04
- profit_target_pct: 1.00

**Edge**: UNPREDICTABLE regimes historically see realized vol exceed implied vol. Regime transitions underprice magnitude.
**Key risk**: Fierce theta decay on straddles. If UNPREDICTABLE resolves slowly, both sides decay. Rare regime = few trades.

### Phases
- [x] **3. Implement** -> general-purpose + skill
  - Files: `src/strategies/options/straddle/{__init__,position,contract_selector,mark_to_market,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_straddle.yaml`
  - Script: `scripts/backtest_scripts/ramp_straddle_backtest.py`
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass -> `tests/strategies/options/straddle/` (49/49 passed)
  - [x] 4b. Code review pass -> `docs/agent-learnings/ramp-straddle/04_review.md`
    - No critical/high issues. Lookahead: PASS. Costs: PASS. Slippage: PASS. Regime: PASS. Temporal split: PASS.
- [x] **5. Initial backtest** -> `docs/reports/ramp-straddle/20260401_initial_backtest.md`
  - 2 trades in 3 years. UNPREDICTABLE regime = 6 days (0.8%) of 753 trading days.
- [x] **6. Validate** -> orchestrator -> NOT VIABLE (insufficient trades)
  - HARD STOP: 2 trades << 30 minimum. Regime frequency is binding constraint.
  - Sharpe: -0.357, CAGR: -0.22%, MaxDD: -1.27%, Win: 0/2, AvgHold: 4d
  - Both trades lost money (regime resolved too quickly, theta+slippage > directional gain)
  - Optimization cannot fix a fundamental frequency problem.
- [-] **7. Optimize** -> SKIPPED (only 2 trades, no optimization possible)
- [-] **8. Final validation** -> SKIPPED (strategy not viable)

### Integrity checklist
- [x] shift(1) confirmed | [x] Costs 10 bps | [x] Slippage | [x] No lookahead | [x] Temporal split | [x] Regime | [x] Freq match | [x] Universe ok

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | spec defaults | d=0.50, DTE=21-45, alloc=4% | -0.357 | N/A | -0.22% | -1.27% | 103d | -0.171 | 0.00% | 0.00 | 2 | 4.0d | UNPREDICTABLE=0.8% | N/A | N/A | 2022-2024 | daily | INSUFFICIENT: 2 trades in 3y |

### Verdict
- [x] **Final**: NOT VIABLE
  - Reason: UNPREDICTABLE regime occurs on 0.8% of trading days (6 out of 753).
  - Only 2 trades generated in 3 years, far below 30-trade statistical minimum.
  - Both trades were losses. Regime resolves in 2-6 days -- too fast for straddles to overcome theta+costs.
  - No parameter adjustment can fix the fundamental regime frequency problem.
  - Recommendation: If pursuing volatility strategies, consider (a) VIX-based triggers instead of regime, (b) shorter-DTE straddles on weekly options, or (c) a different definition of "uncertain" that triggers more frequently.

---

## Strategy: ramp-long-short (#5 Long/Short Options Market Neutral)

**Catalog ref**: #5 -- Feasibility Rank 14/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #5)
**Implementation**: `src/strategies/options/long_short/`
**Config**: `config/strategies/ramp_long_short.yaml`
**Tests**: `tests/strategies/options/long_short/`
**Reports**: `docs/reports/ramp-long-short/`
**Optimization output**: `output/optimization/ramp-long-short/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 2
**Cost tier**: 15 bps
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv`

**Strategy logic summary**:
- Buy calls on top_n momentum stocks + buy puts on bottom_n momentum stocks simultaneously
- Market-neutral momentum capture expressed through options
- Entry: calls in STRONG_BULL/WEAK_BULL, puts in BEAR/WEAK_BULL (both sides may not always be active)
- Exit: same as #1 (calls) and #3 (puts) individually
- Equal capital allocation on each side for market neutrality

**Default parameters**:
- Same as #1 for call side, same as #3 for put side
- Equal capital weighting between long and short sides

**Edge**: Market-neutral momentum capture. Long/short spread should work regardless of market direction.
**Key risk**: Double theta decay -- both calls and puts lose value daily. Momentum spread must be wide enough to overcome.
**Dependency**: Builds on #1 (long calls) and #3 (long puts). SKIP if both are NOT VIABLE.

### Phases
- [x] **3. Implement** -> thin wrapper combining long_calls + long_puts engines
  - Files: `src/strategies/options/long_short/{__init__,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_long_short.yaml`
  - Backtest script: `scripts/backtest_scripts/ramp_long_short_backtest.py`
  - Design: shared data loading, split capital 50/50, separate RAMPSignals instances per side
  - Call side uses optimized params from #1 (d=[0.40,0.60], tp=75%, exit_lt=False)
  - Put side uses defaults from #3 (d=[-0.70,-0.50], tp=100%, exit_lb=False)
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass -> `tests/strategies/options/long_short/` (11/11 passed)
  - [x] 4b. Code review pass (orchestrator direct -- thin wrapper, no new trading logic)
    - Lookahead: PASS (callback architecture, all data filtered <= d, inherited from sub-engines)
    - Costs: PASS (1% slippage + $0.02/contract, both sides)
    - Slippage: PASS (buys at ask*(1+slip), sells at mid*(1-slip))
    - Capital isolation: PASS (50/50 split, separate RAMPSignals instances)
    - Exit logic: PASS (exit_lt=False, exit_lb=False correctly wired)
    - No new trading logic: PASS (all decisions delegated to sub-engines)
- [x] **5. Initial backtest** -> `docs/reports/ramp-long-short/20260331_initial_backtest.md`
- [x] **6. Validate** -> orchestrator -> notes below (NOT VIABLE)
- [-] **7. Optimize** -> SKIPPED (put side is pure drag, combined Sharpe 0.308 < long-calls alone 0.698)
- [-] **8. Final validation** -> SKIPPED (no optimization to validate)

### Integrity checklist
- [x] shift(1) confirmed (callback architecture, all data filtered <= d, inherited from sub-engines)
- [x] Costs 15 bps (1% slippage + $0.02/contract fee, both sides)
- [x] Slippage (buys at ask*(1+slip), sells at mid*(1-slip))
- [x] No lookahead bias (callback architecture, code review passed)
- [x] Temporal split (IS: 2022-01 to 2023-06, OOS: 2023-07 to 2024-12)
- [x] Regime analysis (calls: SB/WB entry, puts: BEAR/WB entry)
- [x] Freq match (daily/EOD)
- [!] Universe: sp500-2025.csv (mild survivorship bias). Only 11 symbols have options data.

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1   | optimized calls + default puts | calls d=[0.40,0.60] tp=75% + puts d=[-0.70,-0.50] tp=100%, 50/50 split | 0.308 | - | 2.86% | 13.94% | 311d | 0.205 | 43.5% | 1.12 | 108 (81c+27p) | 16.5d | SB:$3.5K/WB:$6.3K/BEAR:-$1.1K | - | IS:0.080/OOS:0.448 | 2022-2024 | daily | Put side is pure drag: call P&L +$16.2K, put P&L -$7.4K. Combined Sharpe 0.308 vs long-calls alone 0.698. |

### Validation notes (Phase 6)

**Checklist:**
- [x] Sharpe < 3.0: YES (0.308) -- no overfitting concern
- [x] CAGR < 20%: YES (2.86%) -- low but positive
- [x] Max DD > 5%: YES (13.94%) -- reasonable
- [x] Trades > 30: YES (108) -- sufficient
- [x] Regime analysis: SB +$3.5K, WB +$6.3K, BEAR -$1.1K (both bullish regimes profitable)
- [x] Transaction costs included
- [x] Slippage modeled
- [x] No lookahead confirmed
- [x] Data frequency matches

**Key observations:**
1. **Combined Sharpe 0.308 is WORSE than long-calls alone (0.698).** The put side is pure drag. Adding puts to the portfolio DESTROYS value rather than adding market-neutral protection.
2. **Put P&L contribution: -83.8%.** Calls earned +$16,184, puts lost -$7,377. The put side consumed 50% of capital but generated a large net loss.
3. **Call-side standalone performance degraded from $100K base to $50K base** -- Call Sharpe within long/short (0.572) is lower than standalone (0.698) because it has less capital to compound with.
4. **OOS is better than IS** (Sharpe 0.448 vs 0.080), consistent with long-calls pattern. But even OOS combined (0.448) < long-calls alone OOS (0.799).
5. **Market neutrality thesis FAILED.** The strategy was supposed to capture long/short momentum spread regardless of market direction. Instead: (a) the long side works in trending markets, (b) the short side loses in ALL market environments, (c) combining them just dilutes the call-side alpha.
6. **Root cause is the same as long-puts (#3):** regime_change exits dominate put-side trades. Regime detection is lagging, so puts enter AFTER the decline and exit when regime flips bullish (losing money). The put side has no structural edge to contribute.

**Comparison matrix:**
| Metric | Long Calls Alone | Long/Short Combined | Delta |
|--------|-----------------|---------------------|-------|
| Sharpe | 0.698 | 0.308 | -56% WORSE |
| CAGR | 13.64% | 2.86% | -79% WORSE |
| Max DD | 24.18% | 13.94% | -42% better (only upside) |
| PF | 1.33 | 1.12 | -16% WORSE |

The ONLY metric where long/short wins is Max DD (13.94% vs 24.18%). This is because 50% of capital sits in the put side which loses slowly rather than in the call side which has larger drawdowns. But this is just capital dilution, not real hedging. You could achieve the same DD reduction by running long-calls at 50% capital allocation.

**Decision:** NOT VIABLE. The put side adds no alpha and destroys the call-side edge. Skip optimization (Phase 7) and final validation (Phase 8). The market-neutral approach through long options does not work because:
1. Both sides face theta decay (double bleed)
2. The put side has no structural edge (regime detection too slow)
3. Capital allocated to puts would generate better returns on the call side

### Verdict
- [x] **Final**: NOT VIABLE
- **Reason**: Combined Sharpe 0.308 < long-calls alone 0.698. Put side is pure drag (-$7.4K P&L, -83.8% contribution). Market neutrality thesis failed -- no diversification benefit, just alpha dilution. Long calls (#1) alone is strictly superior.
- **Max DD improvement is illusory**: 13.94% vs 24.18% is just capital dilution (50% in puts), not hedging. Running calls at 50% allocation achieves the same.
- **Recommendation**: Do NOT deploy. Use long-calls (#1) as standalone. The short-side momentum capture through options is structurally broken due to lagging regime detection and theta decay.

---

## Medium-term strategies (Level 3, multi-leg)

These require Level 3 options approval and multi-leg order infrastructure. Start after Level 1+2 strategies are validated.

### ramp-collar (#22 Collars on Equity Positions) -- BATCH 3

**Catalog ref**: #22 -- Feasibility Rank 12/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #22)
**Implementation**: `src/strategies/options/collar/` (engine, position, contract_selector, mark_to_market, metrics, ramp_integration)
**Config**: `config/strategies/ramp_collar.yaml`
**Tests**: `tests/strategies/options/collar/`
**Reports**: `docs/reports/ramp-collar/`
**Optimization output**: `output/optimization/ramp-collar/`

**Asset class**: Large-cap equities (S&P 500 options on RAMP equity holdings)
**Options level**: Level 1+2 (buy put + sell covered call)
**Cost tier**: 15 bps per leg (30 bps total for 2 legs)
**Data frequency**: daily
**Universe**: RAMP equity holdings (subset of 11 options-available symbols)

**Strategy logic summary**:
- When crash protection triggers (VIX > 25 or SPY DD > 5%), wrap RAMP equity positions with collars
- Buy protective put (delta -0.30 to -0.40) + sell covered call (delta 0.20-0.30), same expiry 21-30 DTE
- 1 collar per equity position (100 shares per contract)
- Exit: crash protection clears, DTE <= 5, equity position sold by RAMP rebalance
- Goal: reduce max drawdown at low cost (call premium offsets put cost)
- NOT triggered by momentum ranking -- triggered by crash protection signal

**Default parameters**:
- put_delta_min: -0.40, put_delta_max: -0.30
- call_delta_min: 0.20, call_delta_max: 0.30
- min_dte: 21, max_dte: 30
- vix_threshold: 25.0
- spy_dd_threshold: -0.05

**Edge**: Maintains equity positions during turbulence instead of selling at bottom. Call premium funds put cost (zero-cost collar).
**Key risk**: False alarms cap recovery upside. Orphaned collars if RAMP sells equity while collar active.
**Evaluation**: NOT Sharpe-based. Compare RAMP equity max DD WITH vs WITHOUT collars. Success = meaningful DD reduction > collar cost.
**Code reuse**: CC MTM (bs_call_price), long-puts logic, equity_simulator for holdings
**New callback**: get_equity_positions(date) -> Dict[str, int] (requires equity_simulator to track holdings)

### Phases
- [x] **3. Implement** -> orchestrator (following CC + long-puts patterns)
  - Files: `src/strategies/options/collar/` (6 modules), `config/strategies/ramp_collar.yaml`
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass (65/65 passed)
  - [x] 4b. Code review pass (callback arch = no lookahead, 30bps costs, slippage on both legs, orphaned collar detection)
- [x] **5. Initial backtest**
  - Report: `docs/reports/ramp-collar/20260401_initial_backtest.md`
- [x] **6. Validate** -> NOT VIABLE (fundamental design conflict)
- [-] **7. Optimize** (skipped -- strategy not viable)
- [-] **8. Final validation** (skipped -- strategy not viable)

### Integrity checklist
- [x] shift(1) confirmed (callback architecture)
- [x] Transaction costs included (30 bps for 2 legs)
- [x] Slippage model active
- [x] No lookahead bias
- [x] Temporal train/test split (2022-2023 IS)
- [x] Crash protection signal uses only available data
- [x] Data frequency matches (daily)

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1 | IS | default (put_d=-0.35, call_d=0.25, vix=25) | N/A | N/A | -2.15% drag | -94.27% collar | N/A | 0.11 | 50% | N/A | 28 | 4.0d | N/A | N/A | N/A | 2022-2023 | daily | 100% orphaned exits, 0.41x protection ratio, 10.63%/yr cost |

### Collar-specific metrics
| Metric | Value |
|--------|-------|
| DD reduction | 4.40% |
| Annualized cost | 10.63%/yr |
| Protection ratio | 0.41x (target >3x) |
| CAGR drag | 2.15% |
| Exit: orphaned | 28 (100%) |
| Exit: crash_cleared | 0 (0%) |
| Exit: dte_exit | 0 (0%) |

### Verdict
- [x] **Final**: NOT VIABLE -- fundamental design conflict. RAMP's crash protection (sell equity) directly conflicts with collar overlay (protect equity). When crash triggers, RAMP sells positions within 1-4 days, orphaning 100% of collars. Protection ratio 0.41x (need >3x). Cost 10.63%/yr for only 4.4% DD reduction. Cannot be fixed without fundamentally redesigning either RAMP or the collar trigger.

### ramp-dynamic-collars (#25 Dynamic Collar Width by Regime)
**Catalog ref**: #25 -- Rank 13/31 | Level 1+2 | Same as #22 but regime-adaptive: WEAK_BULL=wide collar, BEAR=tight collar | More nuanced protection | Dependency: builds on #22
- [-] **Phases 3-8** -- SKIPPED: depends on #22 (collar) which is NOT VIABLE due to fundamental design conflict (RAMP sells equity during crash, orphaning collars)

### ramp-pairs (#28 Pairs Options Sector Neutral)
**Catalog ref**: #28 -- Rank 15/31 | Level 2 | Within-sector: buy calls on highest momentum, buy puts on lowest | Sector-neutral momentum | Requires sector classification
- [ ] **Phases 3-8**

### ramp-put-credit-spread (#12 Put Credit Spreads in STRONG_BULL) -- BATCH 3

**Catalog ref**: #12 -- Feasibility Rank 16/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #12)
**Implementation**: `src/strategies/options/put_credit_spread/` (engine, position, contract_selector, mark_to_market, metrics, ramp_integration)
**Config**: `config/strategies/ramp_put_credit_spread.yaml`
**Tests**: `tests/strategies/options/put_credit_spread/`
**Reports**: `docs/reports/ramp-put-credit-spread/`
**Optimization output**: `output/optimization/ramp-put-credit-spread/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 3 (multi-leg)
**Cost tier**: 15 bps per leg (30 bps total for 2 legs)
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv` (11 symbols with options data)

**Strategy logic summary**:
- Sell put (delta -0.25 to -0.35) + buy protective put 5-10 points below on RAMP top_n stocks
- Same expiry, 21-35 DTE
- Gate: STRONG_BULL regime only, crash protection not active
- Profit target: close at 50-80% of max credit received
- Loss limit: close at 150% of max credit loss
- Exit: DTE <= 5, regime change, crash protection triggers
- Position sizing: max loss per spread <= 2% of portfolio. Collateral = max_loss (NOT full strike like CSP)
- exit_on_left_top_n: false

**Default parameters**:
- short_delta_min: -0.35, short_delta_max: -0.25
- spread_width_min: 5.0, spread_width_max: 10.0
- min_dte: 21, max_dte: 35
- profit_target_pct: 0.60 (of max credit)
- loss_limit_pct: 1.50
- max_csp_allocation: 0.30
- max_positions: 5

**Edge**: Capital-efficient CSP. Collateral = max_loss instead of full strike. Tests whether lower capital requirement rescues CSP's marginal 0.218 Sharpe.
**Key risk**: Reduced premium (net credit) may not justify the thinner edge. 30 bps costs on thin credit.
**Code reuse**: CSP engine (short put logic), straddle pattern (2 legs), bs_put_price() for both legs

### Phases
- [x] **3. Implement** -> orchestrator (following CSP + bull-call-spread patterns)
  - Created: `src/strategies/options/put_credit_spread/` (7 files: __init__, position, contract_selector, mark_to_market, engine, metrics, ramp_integration)
  - Config: `config/strategies/ramp_put_credit_spread.yaml`
  - Tests: `tests/strategies/options/put_credit_spread/` (6 test files, 64 tests)
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass (64/64)
  - [x] 4b. Code review pass (shift(1) via callbacks, 30bps costs, slippage OK, collateral=max_loss OK)
- [x] **5. Initial backtest** -- report: `docs/reports/ramp-put-credit-spread/20260401_initial_backtest.md`
- [x] **6. Validate** -- NOT VIABLE, no optimization warranted
- [-] **7. Optimize** -- skipped (full Sharpe -0.276, OOS -0.478, not salvageable)
- [-] **8. Final validation** -- skipped

### Integrity checklist
- [x] shift(1) confirmed (callback architecture)
- [x] Transaction costs included (30 bps for 2 legs)
- [x] Slippage model active
- [x] No lookahead bias
- [x] Temporal train/test split
- [x] Regime analysis included
- [x] Data frequency matches (daily)

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1 | initial | delta -0.25/-0.35, width 5-10, profit 60%, loss 150% | -0.276 | N/A | -7.34% | -34.49% | 199d | -0.21 | 51.5% | 0.68 | 68 | 11.3d | 50% max conc | N/A | 216.4% | 2022-2024 | daily | NEGATIVE Sharpe, massive IS/OOS gap |
| 1-IS | initial | same | 0.410 | N/A | 3.19% | -7.07% | 46d | 0.45 | 30.0% | 0.50 | 10 | 4.6d | -- | N/A | -- | 2022-H1 2023 | daily | Marginal IS, only 10 trades |
| 1-OOS | initial | same | -0.478 | N/A | -15.28% | -34.25% | 199d | -0.45 | 50.0% | 0.61 | 58 | 11.6d | -- | N/A | -- | H2 2023-2024 | daily | STRONGLY NEGATIVE OOS |

### Verdict
- [x] **Final**: NOT VIABLE

**Thesis falsified**: Capital-efficient CSP (collateral = max_loss instead of full strike) does NOT rescue the marginal CSP edge. While CSP had OOS Sharpe 0.218, the put credit spread has OOS Sharpe -0.478.

**Root cause**: Put credit spreads have LOWER net credit than CSPs (avg $2.33/contract credit vs CSP's full premium). When regime transitions cause the short put to move ITM, the losses are proportionally larger relative to the thin credit collected. The long protective put limits max_loss but doesn't limit the FREQUENCY of losses. The strategy still loses on 49% of trades, and the average loss exceeds the average win by a wide margin (profit factor 0.68).

**Key finding**: Premium selling on RAMP momentum stocks is fundamentally flawed. Momentum stocks have high realized vol, which means the short put moves against you faster than theta decays. Both CSP (marginal at 0.218) and put credit spread (-0.478) confirm this. The capital efficiency thesis is irrelevant when the underlying strategy has negative expectancy.

**Pipeline learning**: All premium-selling strategies on high-momentum stocks are likely NOT VIABLE. Remaining put-selling variants (iron condor, risk reversal's CSP leg) should be approached with extreme skepticism.

### ramp-bull-call-spread (#2 Bull Call Spreads) -- BATCH 3

**Catalog ref**: #2 -- Feasibility Rank 17/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #2)
**Implementation**: `src/strategies/options/bull_call_spread/` (engine, position, contract_selector, mark_to_market, metrics, ramp_integration)
**Config**: `config/strategies/ramp_bull_call_spread.yaml`
**Tests**: `tests/strategies/options/bull_call_spread/`
**Reports**: `docs/reports/ramp-bull-call-spread/`
**Optimization output**: `output/optimization/ramp-bull-call-spread/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 3 (multi-leg)
**Cost tier**: 15 bps per leg (30 bps total for 2 legs)
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv` (11 symbols with options data)

**Strategy logic summary**:
- Buy calls (delta 0.50-0.60) + sell calls (delta 0.20-0.30) on RAMP top_n momentum stocks
- Same expiry, 30-45 DTE, spread width 5-15 points
- Gate: STRONG_BULL or WEAK_BULL regime, crash protection not active
- Profit target: close at 50-80% of max profit (not % of premium)
- Exit: DTE <= 7, regime to BEAR/UNPREDICTABLE, crash protection triggers
- Position sizing: net debit per spread <= 2-4% of portfolio, max 5 positions
- exit_on_left_top_n: false (Batch 1 finding)

**Default parameters**:
- long_delta_min: 0.50, long_delta_max: 0.60
- short_delta_min: 0.20, short_delta_max: 0.30
- min_dte: 30, max_dte: 45
- spread_width_min: 5.0, spread_width_max: 15.0
- profit_target_pct: 0.70 (of max profit)
- position_alloc_pct: 0.04
- max_positions: 5
- exit_on_left_top_n: false

**Edge**: Defined-risk version of long-calls (our only VIABLE strategy). Lower cost per trade, capped max loss.
**Key risk**: Capping upside may kill the tail momentum that drives long-calls' Sharpe 0.698. If RAMP's edge is in unbounded upside, spreads destroy it.
**Code reuse**: Straddle position (2-leg pattern), long-calls engine flow, bs_call_price() for both legs

### Phases
- [x] **3. Implement** -> orchestrator (following straddle + long-calls patterns)
  - Files: `src/strategies/options/bull_call_spread/{__init__,position,contract_selector,mark_to_market,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_bull_call_spread.yaml`
  - Tests: `tests/strategies/options/bull_call_spread/` (6 test files)
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass (65/65 passed)
  - [x] 4b. Code review pass (orchestrator self-review)
    - Lookahead: PASS. All callbacks filter with <= d.
    - Costs: PASS. 30 bps (2 legs x slippage on both entry/exit).
    - Slippage: PASS. Correct direction on all 4 trade actions.
    - Fees: PASS. 2 x contract_fee x num_contracts at entry and exit.
    - Profit target: PASS. Uses pnl_pct_of_max_profit (not premium).
    - exit_on_left_top_n: PASS. Default false in config and constructor.
- [x] **5. Initial backtest** -> `docs/reports/ramp-bull-call-spread/20260401_initial_backtest.md`
- [x] **6. Validate** -> NOT VIABLE, skip Phase 7-8
  - Sharpe: -0.098 (NEGATIVE -- strategy is a net loser)
  - Profit Factor: 0.884 (< 1.0)
  - Total P&L: -$5,619 on $100k capital
  - Only WEAK_BULL regime profitable (+16.8%). STRONG_BULL loses (-8.0%).
  - Capping upside via spread destroys the momentum tail that drives long-calls.
  - Thesis falsified: RAMP momentum edge requires unbounded upside.
- [-] **7. Optimize** -> SKIPPED (negative Sharpe, optimization cannot fix broken thesis)
- [-] **8. Final validation** -> SKIPPED

### Integrity checklist
- [x] shift(1) confirmed (callback architecture, all data filtered <= d)
- [x] Transaction costs included (30 bps for 2 legs)
- [x] Slippage model active (worst-fill on all 4 trade actions)
- [x] No lookahead bias
- [x] Temporal train/test split (2022-2024 full period run)
- [x] Regime analysis included
- [x] Data frequency matches (daily)

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1 | Phase 5 | ld=0.50-0.60 sd=0.20-0.30 pt=70% | -0.098 | -- | -1.91% | -21.6% | 310d | -0.089 | 36.4% | 0.884 | 33 | 18.7d | WEAK_BULL only profitable | -- | -- | 2022-2024 | daily | Negative Sharpe, losing strategy |

### Verdict
- [x] **Final**: NOT VIABLE -- Sharpe -0.098, PF 0.884, thesis falsified (capping upside destroys momentum tail)

### ramp-bear-put-spread (#4 Bear Put Spreads)
**Catalog ref**: #4 -- Rank 18/31 | Level 3 | Buy put + sell lower put on bottom_n | Defined-risk short expression | MLeg order | Cheaper than naked puts
- [ ] **Phases 3-8**

### ramp-iron-condor (#10 Iron Condors in SIDEWAYS) -- BATCH 3

**Catalog ref**: #10 -- Feasibility Rank 19/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #10)
**Implementation**: `src/strategies/options/iron_condor/` (engine, position, contract_selector, mark_to_market, metrics, ramp_integration)
**Config**: `config/strategies/ramp_iron_condor.yaml`
**Tests**: `tests/strategies/options/iron_condor/`
**Reports**: `docs/reports/ramp-iron-condor/`
**Optimization output**: `output/optimization/ramp-iron-condor/`

**Asset class**: Large-cap index (SPY options)
**Options level**: Level 3 (4-leg multi-leg)
**Cost tier**: 15 bps per leg (60 bps total for 4 legs)
**Data frequency**: daily
**Universe**: SPY only (V1 -- simplest, always liquid)

**Strategy logic summary**:
- In SIDEWAYS regime: sell OTM put (delta -0.20 to -0.30) + buy further OTM put (5-10pt below) + sell OTM call (delta 0.20-0.30) + buy further OTM call (5-10pt above)
- All same expiry, 21-35 DTE
- Gate: SIDEWAYS regime only (first strategy to use this regime)
- Profit target: close at 50% of max credit
- Exit: DTE <= 5, regime leaves SIDEWAYS, any leg approaches breakeven
- Position sizing: max loss per condor <= 2% of portfolio
- Collateral = wider wing width - net credit

**Default parameters**:
- short_put_delta_min: -0.30, short_put_delta_max: -0.20
- short_call_delta_min: 0.20, short_call_delta_max: 0.30
- wing_width: 10.0 (points between short and long strikes)
- min_dte: 21, max_dte: 35
- profit_target_pct: 0.50 (of max credit)
- max_positions: 1
- position_alloc_pct: 0.04

**Edge**: Premium selling in SIDEWAYS regime is NOT the same contradiction as selling on momentum stocks. SIDEWAYS = low directional movement = when short premium SHOULD work.
**Key risk**: SIDEWAYS periods may be too short for 21-35 DTE condors. 60 bps costs eat into thin credit. First 4-leg strategy.
**Code reuse**: Straddle pattern extended to 4 legs, bs_call_price + bs_put_price for all legs

### Phases
- [x] **3. Implement** -> orchestrator (extending straddle to 4 legs)
  - Created: `src/strategies/options/iron_condor/` (6 files: __init__, position, contract_selector, mark_to_market, engine, metrics, ramp_integration)
  - Created: `config/strategies/ramp_iron_condor.yaml`
  - Created: `tests/strategies/options/iron_condor/` (5 test files: test_position, test_contract_selector, test_mark_to_market, test_engine, test_metrics)
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass (57/57 passed)
  - [x] 4b. Code review pass -- shift(1) via callbacks confirmed, 60bps costs (4 legs), slippage on all legs, collateral accounting correct, no look-ahead
- [x] **5. Initial backtest** -- SPY 2022-2024, 44 trades, Sharpe -3.790, CAGR -28.74%, MaxDD -63.73%
- [x] **6. Validate** -- NOT VIABLE (see verdict below)
- [-] **7. Optimize** -- SKIPPED (strategy fundamentally broken, not a parameter problem)
- [-] **8. Final validation** -- SKIPPED

### Integrity checklist
- [x] shift(1) confirmed (callback architecture)
- [x] Transaction costs included (60 bps for 4 legs)
- [x] Slippage model active
- [x] No lookahead bias
- [x] Temporal train/test split
- [x] Regime analysis included (SIDEWAYS gate)
- [x] Data frequency matches (daily)

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1 | Phase 5 | default (delta 0.20-0.30, wing 10, profit_tgt 0.50) | -3.790 | n/a | -28.74% | -63.73% | 747d | -0.451 | 15.9% | 0.288 | 44 | 3.7d | SIDEWAYS 14.9% of days | n/a | n/a | 2022-2024 | daily | CATASTROPHIC. Avg 2.5d hold. 43/44 regime_departure exits. |

### Verdict
- [x] **Final**: NOT VIABLE -- Sharpe -3.790, CAGR -28.74%, MaxDD -63.73%. Fundamental structural mismatch: SIDEWAYS regime averages 2.5 days per period, but iron condors need 3-4 weeks for theta decay. 43/44 trades closed by regime_departure after 1-5 days. When SIDEWAYS ends, directional moves breach short strikes. 60 bps costs per round trip compound the losses. This is NOT a parameter problem -- no optimization can fix a 2.5-day average hold on a 21-35 DTE strategy. The SIDEWAYS regime as detected by MarketRegimeDetector is too unstable/short-lived for premium-selling strategies that need range-bound conditions to persist.

### ramp-put-ladder (#33 Put Ladder)
**Catalog ref**: #33 -- Rank 20/31 | Level 1 | Sell puts at 2-3 strikes on same name | Enhanced premium but very capital-intensive | High-conviction only
- [ ] **Phases 3-8**

### ramp-put-ratio-backspread (#23 Put Ratio Backspread)
**Catalog ref**: #23 -- Rank 21/31 | Level 3 | Sell 1 ATM put + buy 2 OTM puts on SPY | Cheap crash protection with convex payoff | 3-leg MLeg | "Valley of death" risk on moderate declines
- [ ] **Phases 3-8**

### ramp-calendar-spread (#13 Calendar Spreads)
**Catalog ref**: #13 -- Rank 22/31 | Level 3 | Sell near-term + buy longer-term at same strike | Theta differential | Roll complexity on Alpaca | High Greeks sensitivity
- [ ] **Phases 3-8**

### ramp-diagonal-spread (#32 Diagonal Spreads)
**Catalog ref**: #32 -- Rank 23/31 | Level 3 | Sell short-dated higher call + buy longer-dated lower call | Combines calendar + directional | Roll complexity
- [ ] **Phases 3-8**

---

## Research-priority strategies (feasible with workarounds)

### ramp-risk-reversal (#26 Momentum-Weighted Risk Reversal) -- BATCH 3

**Catalog ref**: #26 -- Feasibility Rank 24/31
**Spec**: `docs/strategies/production/RAMP_OPTIONS_STRATEGY_CATALOG.md` (section: #26)
**Implementation**: `src/strategies/options/risk_reversal/` (engine, metrics, ramp_integration -- THIN WRAPPER, no position/MTM/selector)
**Config**: `config/strategies/ramp_risk_reversal.yaml`
**Tests**: `tests/strategies/options/risk_reversal/`
**Reports**: `docs/reports/ramp-risk-reversal/`
**Optimization output**: `output/optimization/ramp-risk-reversal/`

**Asset class**: Large-cap equities (S&P 500 options)
**Options level**: Level 1+2 (long calls + CSPs)
**Cost tier**: 15 bps per leg
**Data frequency**: daily
**Universe**: `config/universes/sp500-2025.csv` (11 symbols with options data)

**Strategy logic summary**:
- THIN WRAPPER combining LongCallBacktestEngine + CSPBacktestEngine
- Split capital: 60% to long calls, 40% to CSPs (fixed V1 allocation)
- Regime-adaptive: STRONG_BULL heavier calls, WEAK_BULL heavier CSPs, SIDEWAYS CSPs only
- Each side runs independently with own position management
- Combined equity curve merges both engines' daily P&L
- No new position, MTM, or contract selector classes needed

**Default parameters**:
- call_allocation: 0.60
- csp_allocation: 0.40
- Long calls: max_positions=3, position_alloc_pct=0.04, profit_target_pct=1.00, exit_on_left_top_n=false
- CSP: max_positions=3, max_csp_allocation=0.40, profit_target_pct=0.50, loss_limit_multiple=1.0

**Edge**: Combines our two best strategies (long-calls Sharpe 0.698 + CSP Sharpe 0.218). CSP premium partially funds call purchases. Self-funding directional strategy.
**Key risk**: Both sides losing in sharp downturn. Capital split may dilute long-calls' edge.
**Code reuse**: Directly reuses LongCallBacktestEngine + CSPBacktestEngine

### Phases
- [x] **3. Implement** -> orchestrator (coordinator engine reusing existing engines)
  - Files: `src/strategies/options/risk_reversal/{__init__,engine,metrics,ramp_integration}.py`
  - Config: `config/strategies/ramp_risk_reversal.yaml`
  - Tests: `tests/strategies/options/risk_reversal/{__init__,test_engine,test_metrics}.py`
- [x] **4. Test & review**
  - [x] 4a. Unit tests pass (20/20)
  - [x] 4b. Code review pass -- thin wrapper, integrity inherited from sub-engines
- [x] **5. Initial backtest** -- full period 2022-01 to 2024-12
- [x] **6. Validate** -- NOT VIABLE, skip optimization
  - Sharpe 0.447 < 0.5 threshold
  - CSP side Sharpe -0.003 is pure drag on long-calls edge
  - Combined is mathematically worse than long-calls standalone (0.698 Sharpe)
  - No allocation optimization can fix a negative-Sharpe component
  - Monthly win rate 28.57% is very low
  - Regime-dependent: all positive returns from STRONG_BULL + WEAK_BULL
- [-] **7. Optimize** -- SKIPPED: CSP side has negative Sharpe, optimization futile
- [-] **8. Final validation** -- SKIPPED

### Integrity checklist
- [x] shift(1) confirmed (callback architecture, inherited from sub-engines)
- [x] Transaction costs included (inherited from sub-engines)
- [x] Slippage model active (inherited)
- [x] No lookahead bias
- [x] Temporal train/test split
- [x] Regime analysis included
- [x] Data frequency matches (daily)

### Backtest iterations
| Run | Source | Key params | Sharpe | DSR adj | CAGR | MaxDD | MaxDD dur | Calmar | WinRate | ProfitFact | Trades | AvgHold | Regime | Cost 1.5x | IS/OOS gap | Window | Freq | Notes |
|-----|--------|-----------|--------|---------|------|-------|-----------|--------|---------|------------|--------|---------|--------|-----------|------------|--------|------|-------|
| 1 | Initial | 60/40, LC:3pos/4%/100%PT, CSP:3pos/40%/50%PT/1.0x | 0.447 | -- | 4.27% | 15.09% | 302d | 0.283 | 28.57% | 1.139 | 112 | -- | BULL-dep | -- | -- | 2022-01 to 2024-12 | daily | LC=0.480 CSP=-0.003. CSP drag. |

### Verdict
- [x] **Final**: NOT VIABLE -- Sharpe 0.447 < 0.5 threshold. CSP side (Sharpe -0.003) is deadweight that dilutes long-calls edge. Combined strategy is mathematically inferior to long-calls standalone (0.698 Sharpe). The "self-funding" thesis fails because CSP premium income (-0.15% CAGR) does not compensate for the 40% capital diverted from calls. Recommendation: use long-calls standalone instead of this combination.

### ramp-0dte (#30 Regime-Timed 0DTE Selling)
**Catalog ref**: #30 -- Rank 25/31 | Level 1 | Sell 0DTE CSPs/CCs at 3:25 PM | Timing conflict with Alpaca auto-liquidation at 3:30 PM | Maximum gamma risk | Workaround: use 1DTE instead
- [ ] **Phases 3-8**

### ramp-synthetic-long (#7 Synthetic Long)
**Catalog ref**: #7 -- Rank 26/31 | Level 1+2 | Buy call + sell CSP at same strike | Capital efficiency advantage negated by Alpaca cash-securing | Very high capital requirement
- [ ] **Phases 3-8**

### ramp-gamma-scalp (#29 Gamma Scalping with Momentum Bias)
**Catalog ref**: #29 -- Rank 27/31 | Level 2 | Buy options + delta-hedge with equity intraday | Requires intraday execution loop (not in current RAMP infra) | Professional market-maker strategy
- [ ] **Phases 3-8**

### ramp-vol-regime-switch (#19 Volatility Regime Switching SPY)
**Catalog ref**: #19 -- Rank 28/31 | Level 3 | Long vol in UNPREDICTABLE/BEAR, short vol in STRONG_BULL/SIDEWAYS | Must use iron condors for short-vol (no naked shorts) | Pure regime timing of VRP
- [ ] **Phases 3-8**

### ramp-jade-lizard (#14 Jade Lizards)
**Catalog ref**: #14 -- Rank 29/31 | Level 1+3 | Sell CSP + sell call credit spread | Two separate orders, no combined margin | Very high complexity
- [ ] **Phases 3-8**

### ramp-vix-call-spread (#17 VIX Call Spreads)
**Catalog ref**: #17 -- Rank 30/31 | Level 3 | Buy call spread on UVXY when crash protection triggers | VIX proxy via UVXY (no index options on Alpaca) | Contango drag on proxy
- [ ] **Phases 3-8**

### ramp-dispersion (#20 Dispersion Trading)
**Catalog ref**: #20 -- Rank 31/31 | Level 3 | Short SPY vol + long single-stock vol | Institutional strategy | Very high capital + complexity | Portfolio-level Greeks management
- [ ] **Phases 3-8**

---

## Blocked strategies (Alpaca constraints)
- **#11 Short Strangles** -- Requires naked short calls. Use iron condors (#10) instead.
- **#18 Short Straddles** -- Requires naked short calls. Use iron condors (#10) instead.

---

## Portfolio summary (Batches 1-3 complete, 2026-04-01)

### All strategies tested (16 of 31)
| # | Strategy | Family | Level | Universe | Sharpe | CAGR | MaxDD | Trades | Verdict |
|---|----------|--------|-------|----------|--------|------|-------|--------|---------|
| 8 | ramp-csp | Premium Selling | L1 | 11 equities | 0.218 OOS | 0.78% | 2.99% | 63 | MARGINAL |
| 21 | ramp-portfolio-puts | Protection | L2 | SPY | -0.597 | -1.88% | 14.98% | 92 | NOT VIABLE |
| 24 | ramp-tail-hedge | Protection | L2 | SPY | n/a (hedge) | -0.48%/yr cost | reduces DD 24.3% | monthly | MARGINAL |
| 9 | ramp-cc | Premium Selling | L1 | 11 equities | 0.097 OOS | 0.19% | 1.27% | 37 | NOT VIABLE |
| 31 | ramp-systematic-cc | Premium Selling | L1 | 11 equities | 0.117 OOS | 0.19% | 1.27% | 37 | NOT VIABLE |
| 27 | ramp-wheel | Hybrid | L1 | 11 equities | 0.371 OOS | 2.58% | 11.31% | 127 | NOT VIABLE |
| 1 | ramp-long-calls | Directional | L2 | 11 equities | -0.767 (full) | -7.2% (2025 OOS) | 24.18% | 88 (2022-24) / 34 (2025) | NOT VIABLE (stat validation failed) |
| 3 | ramp-long-puts | Directional | L2 | 11 equities | -0.803 | -9.45% | 19.71% | 40 | NOT VIABLE |
| 6 | ramp-deep-itm | Directional | L2 | 11 equities | 0.018 | -7.86% | 52.81% | 96 | NOT VIABLE |
| 15 | ramp-straddle | Volatility | L2 | SPY | -0.357 | n/a | n/a | 2 | NOT VIABLE |
| 5 | ramp-long-short | Hybrid | L2 | 11 equities | 0.308 | 2.86% | 13.94% | 128 | NOT VIABLE |
| 2 | ramp-bull-call-spread | Spread | L3 | 11 equities | -0.098 | -1.91% | 21.6% | 33 | NOT VIABLE |
| 12 | ramp-put-credit-spread | Spread | L3 | 11 equities | -0.276 | -7.34% | 34.49% | 68 | NOT VIABLE |
| 26 | ramp-risk-reversal | Hybrid | L1+2 | 11 equities | 0.447 | 4.27% | 15.09% | 112 | NOT VIABLE |
| 22 | ramp-collar | Protection | L1+2 | 11 equities | n/a (hedge) | n/a | 0.41x protection ratio | 28 | NOT VIABLE |
| 10 | ramp-iron-condor | Volatility | L3 | SPY | -3.790 | -28.74% | 63.73% | 44 | NOT VIABLE |

Also skipped: #25 (dynamic collars) -- depends on #22 which failed.

### Viable strategies (updated 2026-04-02 after statistical validation)
| Strategy | Sharpe | DSR adj | CAGR | MaxDD | Calmar | Trades | Regime | Overfit risk | Verdict |
|----------|--------|---------|------|-------|--------|--------|--------|-------------|---------|
| ramp-long-calls | -0.767 (IS 2018-2024) | p=1.0 (FAIL) | -7.2% (2025 OOS) | 20.5% | n/a | 73 (IS) / 34 (OOS) | ALL negative | HIGH | **NOT VIABLE** |
| ramp-tail-hedge | negative (by design) | n/a | -0.48%/yr | reduces DD 24.3% | n/a | monthly | all | LOW | MARGINAL (hedge) |
| ramp-csp | 0.218 OOS | n/a | 0.78% | 2.99% | 0.26 | 63 | SB only | LOW | MARGINAL |

**NOTE**: ramp-long-calls was previously marked VIABLE (Sharpe 0.698) based on 2022-2024 window only.
Statistical validation with extended data (2018-2024 IS, 2025 OOS) revealed the edge was window-specific.
All 3 HARD statistical gates FAILED. **No options strategies from the 16 tested are deployment-ready.**

### Structural findings from 16 strategies

**1. Momentum alignment is necessary but NOT sufficient**
- Buying calls on momentum stocks (long-calls) ALIGNS with the signal -> appeared VIABLE on 2022-2024 (Sharpe 0.698), but FAILED statistical validation on extended data (Sharpe -0.767 on 2018-2024, Sharpe -1.313 on 2025 OOS)
- Selling calls on momentum stocks (CC, wheel) CONTRADICTS the signal -> NOT VIABLE
- Selling puts on momentum stocks (CSP) partially aligns -> MARGINAL
- **Conclusion**: Options theta decay and bid-ask spread consume the directional edge. The underlying equity momentum signal works (alt signal Sharpe 1.129), but the options overlay destroys it.

**2. The left_top_n exit must be disabled for options**
- RAMP's daily equity rebalance rotates stocks out of top_n every ~3.7 days avg
- Options need 10-18 days to capture theta/delta value
- Removing left_top_n exit transformed long-calls from Sharpe -0.326 to +0.698
- This finding applies to ALL options overlays on RAMP

**3. Short-side momentum doesn't translate to options**
- Bottom_n ranking produces lagging signals (decline already priced into IV)
- IV on weak stocks is elevated at entry (expensive puts)
- BEAR/WEAK_BULL regimes are short-lived (77.5% exit via regime_change)
- Long-puts Sharpe -0.803 confirms this is structurally broken

**4. UNPREDICTABLE regime is too rare for trading**
- 6 days out of 753 (0.8%) in 2022-2024
- Only 2 straddle trades possible -- insufficient for any strategy

**5. Data constraint limits but doesn't break strategies**
- 11 of 503 S&P 500 stocks have ThetaData options data
- Long-calls still produced Sharpe 0.698 with 11 symbols
- Expanding to 50+ symbols would increase trade count and likely improve results

**6. RAMP's momentum edge requires UNBOUNDED upside (Batch 3)**
- Bull call spreads (Sharpe -0.098): capping upside via short call destroys the tail momentum profits
- Put credit spreads (Sharpe -0.276): defined-risk premium selling has negative expectancy on momentum
- Conclusion: spread structures are fundamentally incompatible with RAMP's momentum signal

**7. SIDEWAYS regime is too transient for medium-term options (Batch 3)**
- SIDEWAYS periods average only 2.5 days before transitioning
- Iron condors need 3-4 weeks for theta decay
- 43/44 condor trades closed by regime departure after 1-5 days
- This finding applies to ANY SIDEWAYS-gated strategy with 21+ DTE positions

**8. Collars are structurally incompatible with RAMP's crash protection (Batch 3)**
- When crash protection triggers, RAMP sells equity positions within 1-4 days
- Collar engine tries to protect those same positions with options
- Result: 100% of collars are orphaned before providing meaningful protection
- Protection ratio 0.41x (target >3x), 10.63%/year cost for 4.4% DD reduction

**9. Combining viable + non-viable strategies dilutes the edge (Batch 3)**
- Risk reversal (60% calls + 40% CSPs): combined Sharpe 0.447 < long-calls alone 0.698
- Long-short (calls + puts): combined Sharpe 0.308 < long-calls alone 0.698
- Optimal allocation always converges to 100% long-calls / 0% other

### Redundancy check
- ramp-csp and ramp-long-calls are NOT redundant (different option types, different P&L profiles)
- ramp-tail-hedge is uncorrelated with both (hedge, always-on, SPY-only)
- All three could run simultaneously without conflict

### Capital allocation (updated 2026-04-02 after stat validation)
| Strategy | Allocation | Rationale |
|----------|-----------|-----------|
| RAMP equity (existing) | 99-100% | Core momentum strategy. Sharpe ~0.846 (walk-forward OOS). Only proven edge. |
| ramp-tail-hedge | 0.25-1.0% per month | Optional insurance. Cost: 0.48%/yr for 24.3% max DD reduction. |

### Combined portfolio estimate (updated)
- RAMP equity alone: Sharpe ~0.846 (walk-forward OOS) -- this IS the portfolio
- Adding tail-hedge: reduces max DD ~24% at 0.48%/yr cost -> improves Calmar ratio
- No options alpha overlay -- all 16 tested strategies lack statistically significant edge
- **Estimated combined Sharpe: 0.846** (RAMP equity only)

### Final portfolio recommendation (updated 2026-04-02)

**DO NOT deploy any options strategy as alpha source.**
- ramp-long-calls FAILED statistical validation (0/3 HARD gates passed). Edge was window-specific (2022-2024).
- No other options strategy from 16 tested is viable.

**Consider (hedging only):**
1. **ramp-tail-hedge** as portfolio insurance (Level 2 required). Always-on, regime-sized monthly rolls. Cost: 0.48%/yr CAGR for 24.3% max DD reduction. This is a hedge, not alpha.

**Do not deploy:**
- ramp-long-calls: statistical validation FAILED (DSR p=1.0, OOS 2025 Sharpe=-1.313, Bootstrap CI includes zero)
- All 15 other options strategies: failed in earlier phases

**Do not deploy (13 strategies):**
- ramp-cc, ramp-systematic-cc, ramp-wheel: momentum contradiction (selling calls on rally stocks)
- ramp-portfolio-puts: buying expensive insurance (VIX already elevated at trigger)
- ramp-long-puts: short-side momentum doesn't work via options
- ramp-deep-itm: equity replacement fails with regime switching
- ramp-straddle: UNPREDICTABLE regime too rare
- ramp-long-short: put side is pure drag on viable call side
- ramp-bull-call-spread: capping upside destroys momentum tail
- ramp-put-credit-spread: premium selling negative expectancy on momentum
- ramp-risk-reversal: CSP side dilutes long-calls edge
- ramp-collar: RAMP sells equity before collars protect
- ramp-iron-condor: SIDEWAYS regime too transient (2.5 day avg)
- ramp-dynamic-collars: depends on #22 which failed (skipped)

**Remaining 15 strategies (not yet tested):**
- Most remaining strategies face known structural barriers identified in Batches 1-3
- Bear put spreads (#4): short-side momentum broken (finding #3)
- Calendar/diagonal spreads (#13, #32): spread structures incompatible with momentum (finding #6)
- Put ladder (#33): premium selling on momentum fails (findings #1, #6)
- Vol regime switching (#19): SIDEWAYS too transient (finding #7), UNPREDICTABLE too rare (finding #4)
- Gamma scalping (#29): requires intraday execution (RAMP is daily)
- 0DTE (#30): timing conflict with Alpaca auto-liquidation
- Jade lizard (#14), dispersion (#20), VIX spreads (#17): high complexity, low feasibility
- Synthetic long (#7): capital-intensive, marginal benefit vs equity
- Pairs options (#28): short-side broken (finding #3)
- Put ratio backspread (#23): 3-leg complexity, "valley of death" risk
- **Recommendation**: STOP further strategy development. 16 of 31 tested. The remaining 15 all face structural barriers already identified. ramp-long-calls FAILED statistical validation (2026-04-02). No options strategies are deployment-ready. RAMP equity (Sharpe 0.846) remains the only proven edge.

---

<!--
Build order:
BATCH 1 (COMPLETE - 2026-03-31):
1. ramp-csp (MARGINAL, OOS 0.218)
2. ramp-portfolio-puts (NOT VIABLE, -0.597)
3. ramp-tail-hedge (MARGINAL, viable as hedge)
4. ramp-cc (NOT VIABLE, momentum contradiction)
5. ramp-systematic-cc (NOT VIABLE, identical to #4)
6. ramp-wheel (NOT VIABLE, regime-fragile)

BATCH 2 (COMPLETE - 2026-04-01):
7. ramp-long-calls (FAILED stat validation 2026-04-02: DSR p=1.0, OOS 2025 Sharpe=-1.313, Bootstrap P(+)=0.93%)
8. ramp-long-puts (NOT VIABLE, -0.803)
9. ramp-deep-itm (NOT VIABLE, 0.018)
10. ramp-straddle (NOT VIABLE, 2 trades in 3 years)
11. ramp-long-short (NOT VIABLE, 0.308 -- put side is pure drag)

BATCH 3 (COMPLETE - 2026-04-01):
12. ramp-bull-call-spread (#2) -- NOT VIABLE, -0.098 (capping upside kills momentum)
13. ramp-put-credit-spread (#12) -- NOT VIABLE, -0.276 (premium selling negative expectancy)
14. ramp-risk-reversal (#26) -- NOT VIABLE, 0.447 (CSP dilutes long-calls)
15. ramp-collar (#22) -- NOT VIABLE (RAMP sells equity before collars protect)
16. ramp-iron-condor (#10) -- NOT VIABLE, -3.790 (SIDEWAYS regime too transient)
    Also skipped: #25 dynamic-collars (depends on #22)

REMAINING (Level 3): bear-put-spread, calendar, diagonal, etc.
RESEARCH: 0dte, gamma-scalp, vol-regime-switch, etc.

Orchestrator processes top-to-bottom, one strategy at a time.
After all complete, update the portfolio summary section.
-->
