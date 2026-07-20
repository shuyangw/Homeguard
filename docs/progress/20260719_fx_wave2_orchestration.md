# FX Wave 2 Orchestration + Campaign Resolution - 2026-07-19

## Summary
Orchestrated the second (and final) wave of the 60-strategy FX catalog campaign after the user chose to continue selectively at 6/60-all-FAIL. Ran a cost-model audit + governance re-gate, then scoped and executed Wave 2 (6 structurally-different strategies) via brainstorm -> plan -> subagent-driven build -> strategy-lead gating. ALL 6 Wave 2 strategies FAIL the combined statistical gate. Per the pre-registered stopping rule, the retail G10 FX catalog is declared exhausted (12 gated strategies, 8+ mechanisms, all fail net of realistic costs, cost-robust). Campaign CLOSED: no Wave 3, no ML build.

## What happened (orchestrator-level narrative)
1. **Stepped back at 6/60-all-FAIL** (user prompt). Reframed the finding bar-independently: all 6 had negative net-of-cost OOS Sharpe, so "beat S&P" was not the issue -- there was no gross edge after costs.
2. **Cost-model audit:** our major-tier round-trip (2.0-2.4 pip) is ~2-3x conservative vs IBKR (~0.6-1.0 pip). But recalibrating does not rescue any strategy.
3. **Governance re-gate (strategy-lead):** the 2 Wave-1 enhanced verdicts had been gated OUTSIDE strategy-lead (a compliance gap; the strategy_lead_gate hook does not fire inside subagents). Re-gated all 6 Wave-1 strategies at base + IBKR-optimistic cost -> 6/6-FAIL confirmed ROBUST to cost; registry trail rebuilt; near-misses (#3/#4/weekly-seatbelt) flip sign at optimistic cost but none clear the binding gate.
4. **Debt cleanup:** fixed the fx_clock intraday-DST-gap crash; routed the trial-count policy (every-spec, honest growing N) into strategy-lead's gating; adopted a behavioral no-push rule for subagents (git-hook fix unsafe given the shared Dropbox/Windows .git/config).
5. **Wave 2 scoped + pre-registered** (spec 2026-07-19-fx-wave2-selection-design.md): 6 structurally-different strategies, combined statistical gate as the binding bar, honest every-spec N, stopping rule. Track A (3 READY) + Track B (3 spread-RV needing a new engine).
6. **Track A gated (strategy-lead):** #33 Turn-of-Month REJECT, #39 PCA Dollar-Residual REJECT, #42 RORO WEAK (+0.06 gross, dead after cost/deflation -- campaign high-water mark). 2 plumbing bugs fixed (PCA empty-window, RORO USD-conversion leg).
7. **Track B built (subagent-driven, isolated worktree):** beta-weighted spread engine + #35/#37/#30. Reviews caught + fixed real bugs before any verdict: simulator NaN-poisoning (Critical) + bankruptcy re-check; #37 cost-gate 4x-not-2x + watchlist staleness; #35 dead-stop (Critical false-PASS risk); #30 runner mis-pairing. Opus whole-branch review judged the engine SOUND.
8. **Track B gated (strategy-lead):** #35/#37/#30 all REJECT.

## Wave 2 verdicts (all REJECT, combined statistical gate)
| # | Strategy | Mechanism | OOS Sharpe (1x/1.5x) | DSR | PBO | S&P corr |
|---|----------|-----------|---------------------|-----|-----|----------|
| 33 | Turn-of-Month USD | seasonal | -0.28 / -0.36 | 0 | 0.84 | 0.03 |
| 39 | PCA Dollar-Residual | statistical/market-neutral | -0.12 / -0.22 | 0 | 0.38 | 0.02 |
| 42 | RORO Regime Spread | macro-regime | +0.06 / -0.03 | 0 | 0.17 | 0.00 |
| 35 | AudNzdPairs | cointegration RV | -0.24 / -0.30 | 0 | 0.82 | 0.04 |
| 37 | CointScanner | cointegration RV | -0.24 / -0.31 | 0 | 0.45 | -0.01 |
| 30 | VolRatioPair | vol-ratio RV | -0.48 / -0.54 | 0 | 0.43 | 0.14 |

## The finding (campaign conclusion)
12 gated strategies across two waves span trend, cross-sectional momentum, carry (plain + filtered), metals RV, session breakout, seasonal, statistical dollar-residual, macro-regime, and cointegration/vol-ratio relative-value -- the full frequency (weekly carry to 1-minute breakout) and style spectrum. All fail the combined statistical gate net of realistic costs, and the failure is robust to an IBKR-optimistic cost bound. This is a genuine, well-earned NEGATIVE finding: the retail G10 FX daily/session/RV catalog does not contain a deployable edge after costs in the 2011-2026 sample. Per the North Star, surfacing this is a completed objective, not a failure to engineer around.

## Durable assets built (reusable beyond FX)
Spot-FX daily engine, intraday minute-bar order engine (OCO/bracket/trailing), beta-weighted spread simulator, FX session/DST clock, tier-1 EUR/GBP event calendar, S&P benchmark harness, carry-unwind score, 8 computed artifacts, cost-sensitivity walk-forward harness.

## Commits (this session, all pushed)
- `618c1ae` synthesis; `5a49980` fx_clock DST fix; `81df693`/`1fe7618`/`4194396` cost re-gate; `950b39f`/`172e964` Wave 2 Track A; `95aedeb`/`8e80d49` Track B spec+plan; `6c12358..e7b0d5b` Track B engine+strategies (FF-merged); `84c9075` Track B gate verdicts.

## Known Issues / Remaining Work
- CLOSED per the stopping rule. No further catalog testing.
- Non-blocking debt left: #37 candidate-pair helper duplicates the Cointegration artifact's rule (cosmetic); n_trials_project_wide counts optimizer-combos-only (strategy-lead computed honest N explicitly instead) -- only matters if the campaign ever reopens.
- Governance: 2 unauthorized subagent-to-main pushes occurred earlier in the campaign; behavioral no-push rule held for all Wave 2 subagents.

## Validation
- 17 Track B unit tests pass on main; Track A 24 tests pass. All gates run under strategy-lead with sentinel + registry trail + honest N (grew to 111). Engine soundness confirmed by opus whole-branch review. Verdicts reproduced under both cost legs.
