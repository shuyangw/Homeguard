# FX Beta-Weighted Spread Engine + 3 Relative-Value Strategies (Wave 2 Track B) Design Spec

**Date:** 2026-07-19
**Status:** Approved (brainstorm), pending implementation plan
**Context:** Track B of Wave 2 (`docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md`). Builds the beta-weighted spread-execution engine needed to gate the 3 market-neutral relative-value strategies (#35 AUD/NZD pairs, #37 cointegration scanner, #30 vol-ratio) that complete the 6-strategy Wave 2. Track A (#33/#39/#42, READY) already gated -- all FAIL. These 3 are the last Wave 2 strategies before the pre-registered stopping rule resolves. Gated via `strategy-lead`.

## 1. Purpose

Test whether the market-neutral, cost-advantaged relative-value / cointegration mechanism has edge in G10 FX net of realistic costs -- the structurally-different family the failed directional factors (trend/carry/breakout) do not represent. Doing this fairly requires BETA-WEIGHTED 2-leg spreads (the position tracks the stationary cointegration spread), not the equal-vol spreads the existing per-instrument engine would produce; and it requires charging the round-trip spread on BOTH legs. A FAIL under faithful weighting + honest both-leg costs is a real verdict on the RV mechanism; an equal-vol approximation would leave a FAIL ambiguous (the naive-carry lesson).

## 2. Constraints and inputs

- Reuse: the cointegration artifact (`src/data/artifacts/cointegration.py`: `test_pair(a,b) -> {adf_pvalue, hedge_ratio, half_life}`, `ou_half_life`), the 22-pair G10 daily cache, the FX cost model (`src/backtesting/costs/fx.py`), the walk-forward + combined statistical gate (`src/backtesting/walkforward_common.py`), the S&P benchmark (`src/backtesting/benchmark.py`), and `cb_decisions.yaml` (RBA/RBNZ blackout for #35).
- The existing `FxSpotPortfolioSimulator` sizes each instrument independently to `vol_target` (equal-vol), so it CANNOT express a beta-weighted spread; a purpose-built spread simulator is required. Reuse its MTM / leverage-cap / bankruptcy-floor machinery.
- fintech conda env; ASCII-only, no em dashes, no emojis, no `print()` (use `src.utils.logger`).
- Gated via `strategy-lead` under the Wave 2 pre-registration: combined statistical gate (Section 2.5) as the binding bar, honest every-spec trial count, both cost legs, S&P/corr/IR book-level context only.

## 3. Architecture (components)

1. `src/backtesting/engine/fx_spread_simulator.py` -- `FxSpreadPortfolioSimulator`: holds a book of active beta-weighted 2-leg spreads, sizes each so the spread has `vol_target` annualized vol, applies an IDM-style diversification multiplier across spreads and a portfolio leverage cap, MTM both legs daily, charges round-trip cost on both legs at rebalance, bankruptcy floor. Reuses the spot simulator's MTM/leverage/bankruptcy logic.
2. A spread-forecast interface: strategies expose `spread_panel(close_panel) -> list of per-date active-spread records`, each record `(date, leg_a, leg_b, hedge_ratio, signed_strength)` where `signed_strength` on the Carver scale drives sizing/direction (positive = long the spread ln(A) - beta*ln(B), negative = short). This is the spread analogue of `forecast_panel`.
3. `src/strategies/advanced/fx_audnzd_pairs.py` (#35), `fx_coint_scanner.py` (#37), `fx_vol_ratio_pair.py` (#30).
4. `config/backtesting/fx_{audnzd_pairs,coint_scanner,vol_ratio_pair}.yaml`.
5. `scripts/backtest_scripts/run_fx_spread_walkforward.py` -- aggregates the spread-book daily P&L to a return series and runs the combined statistical gate + S&P context (mirrors the existing FX walk-forward runners).

## 4. Beta-weighted spread sizing

For a spread `s = ln(A) - beta*ln(B)` with per-day returns `r_s = r_a - beta*r_b`:
- Hold notional_A = w (signed by direction) and notional_B = -beta*w, so the position's return tracks `r_s`.
- Choose `w` so the spread position's annualized vol equals `vol_target`: `w = vol_target * equity * strength_scale / (sigma_s * sqrt(252))`, where `sigma_s` is the trailing daily std of `r_s` (e.g. 60d) and `strength_scale = signed_strength / 10` (Carver: 10 = full 1x). Cap per-spread and portfolio leverage.
- Multiple simultaneous spreads: each vol-targeted; apply a diversification multiplier (IDM-style, capped) and a portfolio-gross leverage cap so a book of correlated spreads does not over-lever (the same failure mode as the correlated spot FX book at vol_target 0.20).

## 5. Costs (both legs)

Changing a spread position trades both legs. At each rebalance, charge the round-trip spread cost on EACH leg via `fx_round_trip_pips(tier_of_leg, session)` x `_pip_size(leg)` x abs(units_traded_leg) x quote_to_usd_leg -- i.e. two independent cost charges per spread adjustment. A 2-leg spread therefore pays ~2x the per-trade spread of a single position; this is the honest headwind market-neutral RV must overcome and is central to a fair test.

## 6. The 3 strategies

### #35 AUD/NZD pairs
Rolling 120-day OLS of ln(AUDUSD) on ln(NZDUSD) -> hedge_ratio beta and residual; residual z-score over the same window. Enter the spread when |z| > 2 (short the rich leg / long the cheap leg, beta-weighted, signed by -sign(z)); exit z < 0.5 (target), |z| > 3.25 (stop), or 20 days (time). Blackout: skip entries within 7 days of an RBA or RBNZ decision (`cb_decisions.yaml` keys RBA/RBNZ). One spread (AUDUSD, NZDUSD).

### #37 Rolling cointegration scanner
Monthly scan: for all candidate pairs-of-pairs sharing <= 1 common currency (avoid mechanical triangles), run `cointegration.test_pair` on trailing 250-day ln-prices. Tradeable set: adf_pvalue < 0.05, OU half_life in [5, 25] days, and spread vol sufficient to clear 2x round-trip cost at a 1.5-sigma move. Rank by edge/cost; trade the top 5. Enter |z| > 2; exit z < 0.5, |z| > 3.5 (stop), 2 x half_life (time), or a STRUCTURAL stop: rolling ADF p-value degrades > 0.2 for 10 consecutive days (relationship dying -> exit even at a loss). Up to 5 concurrent spreads.

### #30 Vol-ratio pair (simplified symmetric)
Coupled sets {EURNOK, EURSEK}, {AUDUSD, NZDUSD}, {XAUUSD, XAGUSD}. Weekly: z-score of ln(RV_10d(A) / RV_10d(B)) vs its trailing 2-year distribution. When |z| > 2, bet the ratio reverts: long the low-vol leg / short the high-vol leg, beta-weighted (beta from the rolling price regression of the pair, or 1.0 if the coupled set has no stable price cointegration -- the position expresses vol-ratio convergence, sized market-neutral). Exit when |z| < 1. The asymmetric expansion/fade bracket construction from the research is deferred; this tests the core vol-ratio-reversion mechanism symmetrically. Up to 3 concurrent spreads.

## 7. The gate (via strategy-lead, Wave 2 pre-registration)

Each strategy's spread-book daily P&L aggregates to a daily return series fed to the combined statistical gate (Section 2.5): positive deflated OOS Sharpe clearing PSR/DSR/PBO, net of both cost legs, on the walk-forward (36m/12m/12m, purge+embargo). Honest every-spec trial count (continuing the growing project-wide N). S&P correlation, IR, and marginal deflated contribution reported as book-level context (non-gating) -- especially relevant here since these are market-neutral and expected to be equity-uncorrelated. Gated by `strategy-lead` (sentinel set, registry appends, commit-only/no-push). These 3 complete Wave 2; the pre-registered stopping rule then resolves (any clear -> defines Wave 3; all 6 fail -> declare the catalog exhausted and stop).

## 8. Testing plan

Simulator (heaviest coverage, the reusable core):
1. Beta-weighted sizing: a spread with beta=1.5 holds notional_B = 1.5 x notional_A, opposite signs.
2. Spread vol-targeting: a spread with a known sigma_s sizes to vol_target annualized vol.
3. Both-leg cost: adjusting a spread charges cost on BOTH legs (two trade rows), each per its leg's tier.
4. Multi-spread book: two concurrent spreads each vol-targeted; diversification multiplier and portfolio leverage cap applied.
5. Exits: z-target, hard stop, time stop, and the structural ADF-degradation exit each flatten the right spread.
6. Market-neutrality: a spread's net exposure to the common factor is ~zero by construction (beta-weighted), verified on synthetic co-moving legs.
7. MTM/bankruptcy: reuses and re-verifies the spot simulator's floor.

Strategies (synthetic panels): #35 residual-z entry/exit + RBA/RBNZ blackout; #37 scanner tradeable-set filter (ADF/half-life/vol) + structural exit; #30 vol-ratio z entry/exit. Runner: spread P&L aggregates to a daily series of the right length; report renders with the gate + book-level context.

## 9. Out of scope / deferred

- #30's asymmetric expansion/fade bracket construction (tested as symmetric vol-ratio reversion here).
- #36 Scandi triangle (needs Brent oil data) and #40 correlation-breakdown: not in the 6.
- A live-trading adapter for spread execution.
- Intrabar/tick spread fills (daily close-to-close only).
