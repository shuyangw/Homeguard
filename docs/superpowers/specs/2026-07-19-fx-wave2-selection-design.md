# FX Catalog Campaign -- Wave 2 Selection + Pre-Registration

**Date:** 2026-07-19
**Status:** Approved (brainstorm), pending per-unit implementation plans
**Role:** This is both the Wave 2 roadmap (which 6 strategies, why, what infra) AND the pre-registration (the bar, trial-counting policy, and stopping rule, locked BEFORE any Wave 2 result). Wave 1 synthesis: `docs/strategies/research/20260719_fx_catalog_campaign_synthesis.md`.

## 1. Why Wave 2 is scoped this way

Wave 1 gated 6 strategies, all FAIL, robust to cost. But the tested set was the crowded, most-published factors (time-series and cross-sectional trend, plain and filtered carry, a metals ratio-reversion, the most-traded retail session breakout). Their failure is weak evidence about the structurally-different, uncrowded, cost-advantaged mechanisms that remain untested. Wave 2 tests six of those, deliberately spanning distinct mechanisms so that a uniform failure would be informative across the whole style space (not just one family repeated).

## 2. The 6 strategies

Three ride a new spread-execution engine (market-neutral relative-value); three run on the existing daily `forecast_panel` engine (different mechanisms, testable in parallel).

| # | Strategy | Mechanism | Infra | Universe / inputs |
|---|----------|-----------|-------|-------------------|
| 35 | AUD/NZD beta-weighted spread | Cointegration RV (market-neutral) | new spread engine | AUDUSD, NZDUSD (both present) |
| 37 | Cointegration scanner | General cointegration RV | new spread engine | G10 pairs; cointegration artifact selects legs |
| 30 | Relative-vol pair (XAU/XAG) | Vol-differential RV | new spread engine | XAUUSD, XAGUSD (present) |
| 39 | PCA dollar-factor residual | Statistical residual reversion (market-neutral) | READY | 22-pair panel; pca_dollar artifact |
| 42 | RORO regime spread | Macro risk-on/off regime | READY | AUDJPY, CHFJPY, XAUUSD (present); regime artifact |
| 33 | Turn-of-month USD | Seasonal (month-end rebalancing flow) | READY | USD-major pairs |

Mechanism coverage: spread-RV (3), statistical-residual (1), macro-regime (1), seasonal (1). None overlaps the failed trend/carry/breakout family.

## 3. Decomposition (Wave 2 is two things)

1. **Spread-execution engine (sub-project, built first).** A beta-weighted 2-leg relative-value engine: hedge ratio from the cointegration artifact (or a fixed/rolling beta for #30/#35), net-forecast sizing, the SAME conservative cost model and walk-forward gate machinery already in use (charge spread on BOTH legs). Its own spec + plan + build (subagent-driven or strategy-lead-orchestrated per the build/verdict boundary). Gates strategies #35/#37/#30.
2. **The 6 strategy gatings.** Each strategy is implemented and gated. The 3 READY strategies (#39/#42/#33) do NOT depend on the spread engine and can be built + gated in PARALLEL with the engine, yielding early Wave 2 verdicts.

Each buildable unit (the engine, each strategy) gets its own implementation plan. Any phase that RUNS a backtest / walk-forward / gate producing a VERDICT is delegated to `strategy-lead` per the repo's build-vs-verdict boundary.

## 4. The bar (unified, North-Star-aligned)

**Primary gate: the combined statistical gate (methodology Section 2.5)** -- a positive, deflated out-of-sample Sharpe clearing PSR / DSR / PBO, net of realistic costs, on the walk-forward (36m/12m/12m, purge + embargo, both cost legs). This replaces Wave 1's mixed use of a "beat the S&P Sharpe" bar, because (a) Section 2.5 is the methodology's authoritative gate, and (b) a "beat S&P Sharpe" bar is inappropriate for market-neutral RV strategies, which are uncorrelated to equities by construction.

**Reported alongside (book-level context, per the North Star, non-gating):** S&P correlation, information ratio vs the S&P, and the strategy's marginal deflated cost-net contribution given cross-sleeve correlation. A market-neutral strategy with a modest but positive deflated Sharpe and near-zero equity correlation has genuine book value even if its standalone Sharpe is unremarkable -- this context is recorded so a PASS is judged at the book level, not on standalone Sharpe alone.

## 5. Trial-counting policy (resolves the Wave 1 open item)

Per the North Star ("Every specification run -- parameter, feature, universe, or holding-period change -- is a trial"), Wave 2 uses **every-spec counting**: the project-wide trial count N fed to the DSR deflation is the cumulative number of distinct strategy specifications gated across the project (Wave 1's specifications + each Wave 2 specification as it runs), NOT the current optimizer-combinations-only count (which returns ~0 and under-deflates the DSR, making the gate too easy).

`strategy-lead` owns implementing and enforcing this: fix `src/experiments/registry.py::n_trials_project_wide` (or the runners' trial-count inputs) so every gated specification increments N, and re-confirm the honest N before computing any Wave 2 DSR. This is integrity infrastructure the gate depends on; it must be correct before the first Wave 2 DSR is computed.

## 6. Pre-registered stopping rule (locked before any Wave 2 result)

- **If >=1 Wave 2 strategy clears the combined gate**, OR comes genuinely close (positive deflated Sharpe with low S&P correlation = real diversification value at the book level): that mechanism defines Wave 3, scoped to variants/neighbors of the surviving mechanism.
- **If all 6 Wave 2 strategies FAIL** the combined gate: across two waves the campaign will have tested 8+ distinct mechanisms (trend, cross-sectional momentum, carry, filtered carry, session breakout, spread-RV, statistical residual, macro-regime, seasonal, metals) spanning the full frequency and style spectrum, all failing after realistic costs. That is decisive evidence the retail G10 FX catalog is exhausted. The campaign DECLARES the finding and STOPS: no Wave 3, and specifically no ML-harness build (#48-53). An 8-mechanism, two-wave, cost-robust failure is a structural verdict about the asset class + cost regime, not a coverage gap that "we just haven't tried ML yet" would fill.

## 7. Out of scope / deferred

- The ML meta-labeling family (#48-53) and its harness: gated behind the Wave 2 stopping rule (only if something survives).
- Scandi triangle #36 (needs Brent oil data on top of the spread engine) and correlation-breakdown #40 (partial spread dependency): deferred; not in the 6.
- EM/data-blocked strategies (#18, #55) and the remaining INTRADAY strategies (#21-25): not in Wave 2.
