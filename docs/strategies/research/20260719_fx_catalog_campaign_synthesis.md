# FX 60-Strategy Catalog Campaign -- Synthesis (Wave 1) - 2026-07-19

## Status
SUPERSEDED (2026-07-26) by `20260726_fx_campaign_synthesis_final.md`, which
carries the corrected bar, the apparatus sweep, and wave 3. The wave-1/2
verdicts below stand; the framing does not, because several figures quoted here
predate defect fixes that moved them.

RESOLVED (2026-07-19): the campaign is CLOSED. This document was written as an
interim synthesis after Wave 1; Wave 2 then ran and ALL 6 Wave 2 strategies also
failed. Across two waves, 12 gated strategies spanning 8+ distinct mechanisms all
fail the combined statistical gate net of realistic costs, robust to an
IBKR-optimistic cost bound. Per the pre-registered stopping rule, the retail G10
FX catalog is declared exhausted. STOP: no Wave 3, no ML-harness build. Final
resolution: `20260719_fx_wave2_resolution.md`; Track B results:
`20260719_fx_wave2_trackB_results.md`. The "Go-forward (Wave 2)" section below is
retained as the historical record of how Wave 2 was scoped; it has now completed
and failed.

## The finding so far
Six strategies gated, all FAIL, and the failure is **robust to the cost
assumption** (re-verified under proper strategy-lead orchestration at both the
base cost model and an IBKR-optimistic 0.5 pip/side bound).

| # | Strategy | Family | Base OOS Sharpe | 0.5x-cost OOS | Binding gate | Verdict |
|---|---|---|---:|---:|---|---|
| 3 | FxTSMOM | trend | -0.016 | +0.075 | combined stat (PSR 0.21, DSR ~0, PBO 0.85) | REJECT |
| 4 | FxXSectMom | trend (x-sect) | -0.051 | +0.058 | combined stat (PSR 0.006, DSR ~0, PBO 0.66) | REJECT |
| 15 | FxCarry | carry | -0.327 | -0.295 | combined stat | REJECT |
| 43 | FxGoldSilver | metals RV | -0.313 | -0.299 | combined stat | REJECT |
| 16/19 | FxCarrySeatbelt (daily) | carry+filter | -0.750 | -0.491 | OOS Sharpe > S&P 0.684 | FAIL |
| 16/19 | FxCarrySeatbelt (weekly) | carry+filter | -0.112 | +0.020 | OOS Sharpe > S&P 0.684 | FAIL |
| 20 | LondonBreakout | session breakout | -1.600 | -0.748 | OOS Sharpe > S&P 0.677 | FAIL |

Three near-misses (#3, #4, weekly seatbelt) cross zero at optimistic cost but
none clear the binding gate: the statistical gate (PSR/DSR/PBO) does not move
with the cost leg, and the seatbelt's +0.02 is ~34x below the S&P bar. Cost
conservatism is real (largest swing: breakout +0.85 Sharpe) but is not the cause
of any FAIL. Details: `20260719_fx_cost_sensitivity_regate.md`.

## What this does and does NOT establish
**Establishes:** the crowded, most-published FX factors -- time-series and
cross-sectional trend, plain and filtered carry, a metals ratio-reversion, and
the single most-traded retail session breakout -- do not survive realistic
retail costs in the 2011-2026 G10 sample, net, out-of-sample. Several lack gross
edge even in-sample (breakout IS -0.99). This is a genuine negative result about
the CROWDED end of the catalog.

**Does NOT establish** that the catalog is exhausted. Only 6 of 60 (10%) are
gated, and the tested set was deliberately front-loaded with the simplest,
most-arbitraged mechanisms (trend/carry/breakout are the most-published FX
strategies). Their failure is weak evidence about the structurally-different,
less-crowded plays that remain untested: market-neutral cointegration/spread
relative-value, ML meta-labeling, and specific calendar/microstructure effects.
The research's own survivorship model expected only 5-10 of 60 to earn capital,
so 6 early fails is on-model, not anomalous.

## Reusable assets built (valuable regardless of any single verdict)
- Spot-FX daily backtesting vertical (asset_class=fx): `forecast_panel` engine +
  `FxSpotPortfolioSimulator` (MTM + calendar-day carry + leverage cap).
- 22-pair G10 daily cache, gap-free, cross-vendor-validated; 1-minute cache
  (Dukascopy) for the intraday pairs.
- **Intraday order engine** (`src/backtesting/engine/intraday_order_engine.py`):
  general minute-bar order book (stop/limit/OCO/bracket, partial fills, trailing
  stops, time controls, no-lookahead, conservative gap-through fills). Reusable
  for all remaining INTRADAY strategies.
- **FX session/DST clock** (`src/backtesting/sessions/fx_clock.py`).
- **Tier-1 EUR/GBP event calendar** (`src/data/macro_calendar_tier1.py`).
- **S&P benchmark harness** (`src/backtesting/benchmark.py`) + walk-forward +
  combined DSR/PBO gate. **carry_unwind** composite risk-off score.
- 8 computed artifacts: spread_model, vol_surface, currency_strength, pca_dollar,
  cointegration, regime, event_registries.

## Known debt (being cleaned up before the next wave)
1. FxCarrySeatbelt / LondonBreakout use a broken DIAGNOSTIC trial-count
   (near-zero vs the honest growing ~95). Non-verdict-affecting (DSR is
   diagnostic-only for those; the S&P bar binds) but must be fixed before any
   future DSR-GATING decision.
2. `fx_clock.fx_trading_day` `DateOffset(hours=7)` raises on 1m data crossing the
   spring-forward gap; the London Breakout runner works around it. Harden
   fx_clock itself before more intraday strategies rely on it.
3. Two unauthorized subagent-to-main git pushes occurred this campaign; subagent
   git authority should be tightened.

## Go-forward (Wave 2): selective, not exhaustive, with pre-registered stopping
Rationale: 6/60 does not justify abandoning the catalog, and the expensive
infrastructure is now built (marginal test cost is low). But two constraints
argue for selectivity rather than grinding all 54 remaining: (a) trial-count
deflation -- every test raises the DSR multiple-comparisons bar for any eventual
winner (already ~95 trials); (b) the FX cost tax is systematic but hits
high-turnover directional factors hardest, pointing toward low-turnover /
market-neutral / different-alpha mechanisms.

Wave 2 therefore tests the most structurally-different, uncrowded,
cost-advantaged candidates the failed factors do NOT represent, with a
pre-registered stopping rule to be set before the wave runs. Candidate pool
(untested, structurally distinct): cointegration/spread relative-value
(#30/#35/#36/#37, cointegration artifact already built), calendar/seasonal
(#31/#33/#34), dollar-factor and correlation-structure plays (#39/#40/#42), and
ML meta-labeling (#48-53, needs the unbuilt ML harness). Exact 6 selected via a
brainstorming pass; gated through strategy-lead.
