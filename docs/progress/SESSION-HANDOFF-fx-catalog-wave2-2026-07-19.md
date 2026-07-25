# Session Handoff: FX Catalog Campaign Wave 2 + Resolution

**Date:** 2026-07-19 · **Working dir:** /Users/shuyangw/Library/CloudStorage/Dropbox/cs/github/Homeguard · **Model:** Opus 4.8 (1M context)

## Resume Here (read this first)
- **Goal:** Work the 60-strategy FX catalog: gate pre-registered economic hypotheses through the integrity apparatus (walk-forward + PSR/DSR/PBO), report honest verdicts. NOT to make a metric go up.
- **Status:** CAMPAIGN CLOSED. 12 strategies gated across 2 waves, ALL FAIL net of realistic costs (cost-robust to IBKR-optimistic bound). Retail G10 FX catalog declared EXHAUSTED per the pre-registered stopping rule. STOP: no wave-3, no ML build. `main` = `origin/main` = `3106b9c`, all pushed.
- **Next steps (only if user re-engages):** (1) OPTIONAL backfill: 8 of 12 gated strategies lack a fills-level `trades.csv` (see Open Questions) -- user was asked whether to backfill via strategy-lead; awaiting answer. (2) If user wants to reopen: the ML family (#48-53) and #36/#40 remain untested but are gated behind the stopping rule.
- **Blockers / open questions:** the trade-log backfill decision (below). Otherwise none -- campaign is resolved.
- **To resume, you need:** fintech conda env + `PYTHONPATH=$(pwd)`. macOS: main tree `settings.ini` has an uncommitted `[macos]` storage-path section (do NOT commit it). Git hazard: use only `git add <paths>`/`commit`/`log`; NEVER `git checkout`/bare `status`/`diff`/`reset` (broken Windows gitlinks; shared Dropbox `.git`).

## Original Task
Continuation of the FX 60-strategy catalog campaign. This session's arc: user stepped back at 6/60-all-FAIL and asked "why give up on the remaining ones if 6 have failed?" -> decided to CONTINUE selectively -> ran a cost audit + governance re-gate -> scoped and executed Wave 2 (6 more strategies) -> all fail -> stopping rule resolved -> closed the campaign. Then integrity follow-ups (strategy-lead usage, trade logging).

## Subtasks & Progress
- [x] Strategic review at 6/60 -- reframed bar-independently: all 6 had NEGATIVE net-of-cost OOS Sharpe (no gross edge), so "beat S&P" was never the issue.
- [x] Cost-model audit -- major-tier round-trip 2.0-2.4 pip is ~2-3x conservative vs IBKR (~0.6-1.0). But does not rescue anything.
- [x] Governance re-gate (strategy-lead) -- re-gated all 6 Wave-1 at base + IBKR-optimistic 0.5-pip cost. 6/6-FAIL ROBUST to cost. Near-misses (#3/#4/weekly-seatbelt) flip sign at optimistic cost but none clear the binding gate. Registry trail rebuilt. Doc: `docs/strategies/research/20260719_fx_cost_sensitivity_regate.md`.
- [x] Debt cleanup -- fx_clock intraday-DST-gap crash FIXED (`5a49980`, tz-naive shift); trial-count routed to strategy-lead (honest every-spec N); git governance = behavioral no-push rule.
- [x] Wave 2 scoped + pre-registered -- spec `docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md`. 6 strategies, combined statistical gate (§2.5) as binding bar, every-spec trial count, stopping rule.
- [x] Wave 2 Track A gated (strategy-lead) -- #33 Turn-of-Month REJECT, #39 PCA Dollar-Residual REJECT, #42 RORO WEAK (+0.06 gross, dead after cost, campaign high-water mark). 2 plumbing bugs fixed (PCA empty-window, RORO USD-conversion leg).
- [x] Wave 2 Track B built -- beta-weighted spread engine + #35/#37/#30, subagent-driven in isolated worktree, FF-merged `e7b0d5b`. Reviews caught+fixed real bugs pre-verdict (see Key Decisions).
- [x] Wave 2 Track B gated (strategy-lead) -- #35 AudNzd REJECT, #37 CointScanner REJECT, #30 VolRatio REJECT. All DSR 0.
- [x] Wave 2 RESOLVED -- all 6 fail -> catalog exhausted -> STOP. Docs written, memory updated, all pushed (`4049a95`).
- [x] Integrity Q1: "did we run strat lead on all?" -- Answer: Wave 2 all through strategy-lead from start; Wave-1 enhanced (#16/#19, #20) originally BYPASSED it (the governance gap) but were retroactively re-gated through strategy-lead in the cost re-gate. All 12 now covered.
- [x] Integrity Q2: "did we log all trades?" -- NO. 4 of 12 have fills-level `trades.csv`; 8 do not (they have daily return-streams in the registry but not fills). Real gap.
- [x] Fixed strategy-lead agent def to ENFORCE + VERIFY fills-level trade logs (`3106b9c`).
- [ ] OPTIONAL: backfill the 8 missing fills logs via strategy-lead -- user asked, not yet answered.

## Key Decisions & Tradeoffs
- **Continue, not stop, at 6/60.** Why: 6/60 is 10%, front-loaded with the crowded/most-arbitraged factors (trend/carry/breakout); weak evidence about uncrowded structural plays. Tradeoff: trial-count deflation grows (now N~111), so continue SELECTIVELY with a pre-registered stopping rule, not exhaustively.
- **Wave 2 binding bar = combined statistical gate (§2.5), not "beat S&P".** Why: §2.5 is the authoritative methodology gate and is correct for market-neutral RV (a "beat S&P Sharpe" bar is wrong for equity-uncorrelated spreads). S&P/corr/IR kept as book-level context.
- **Every-spec trial-count policy.** Per North Star ("every specification run is a trial"). strategy-lead computed the honest N explicitly (104->111) rather than trust `n_trials_project_wide` (which counts optimizer-combos-only, returns ~0 -- a known bug left as debt).
- **Track B: build a real beta-weighted spread simulator** (not equal-vol approximation on the existing engine). Why: the existing sim vol-targets each leg independently (equal-vol); the research specifies hedge-ratio-weighted spreads; an equal-vol approximation would make a FAIL ambiguous (the naive-carry lesson). #30 simplified to symmetric vol-ratio reversion (asymmetric bracket deferred).
- **No-push governance = behavioral, not a hook.** Why: `core.hooksPath` points to a broken Windows path and `.git/config` is shared via Dropbox; a git pre-push hook is unsafe to add. So: subagents COMMIT only, orchestrator OWNS all pushes. Held for all Wave 2 subagents (2 unauthorized pushes happened EARLIER in the campaign).
- **strategy-lead agent-def fix over per-dispatch instruction.** Why user asked: durable > one-off. Added a "Fills-level trade log persisted" verification gate + an explicit rule that return-streams do NOT satisfy it and custom runners must additionally run a primary logged backtest and VERIFY the artifact.

## Wave 2 Verdicts (all REJECT, combined statistical gate, honest N)
| # | Strategy | Mechanism | OOS Sharpe 1x/1.5x | DSR | PBO | S&P corr | N |
|---|---|---|---|---|---|---|---|
| 33 | Turn-of-Month USD | seasonal | -0.28 / -0.36 | 0 | 0.84 | 0.03 | 104 |
| 39 | PCA Dollar-Residual | statistical/mkt-neutral | -0.12 / -0.22 | 0 | 0.38 | 0.02 | 105 |
| 42 | RORO Regime Spread | macro-regime | +0.06 / -0.03 | 0 | 0.17 | 0.00 | 106 |
| 35 | AudNzdPairs | cointegration RV | -0.24 / -0.30 | 0 | 0.82 | 0.04 | 109 |
| 37 | CointScanner | cointegration RV | -0.24 / -0.31 | 0 | 0.45 | -0.01 | 110 |
| 30 | VolRatioPair | vol-ratio RV | -0.48 / -0.54 | 0 | 0.43 | 0.14 | 111 |

Wave 1 (prior, all FAIL): #3 TSMOM, #4 XSectMom, #15 Carry, #43 GoldSilver (naive, FAIL-naive); #16/#19 CarrySeatbelt (FAIL-enh, daily -0.75/weekly -0.11); #20 LondonBreakout (FAIL-enh, -1.60). Total 12 gated, all FAIL, 8+ mechanisms.

## Bugs caught by reviews (Track B, all fixed pre-verdict)
- Spread simulator: NaN poisoning (Critical -- one NaN close permanently corrupts the equity curve; real FX panels have NaN) + missing post-cost bankruptcy re-check (High). Fixed `ce70d10`.
- #37 scanner: cost-gate charged 4x round-trip not 2x (`fx_round_trip_pips` already returns round-trip) + watchlist never refreshed hedge/staleness. Fixed `75b3bd0`.
- #35: stop and time-stop were DEAD (same-bar re-entry negated them; only target exit flattened) -> would have INFLATED toward a FALSE PASS on a mean-reverting pair (the most dangerous bug, on the best-hope mechanism). Fixed `e7b0d5b`.
- #30 runner: positional coupled-set pairing silently mis-pairs if a leg is uncached. Fixed `e7b0d5b`.
- Opus whole-branch review judged the engine SOUND after fixes -> verdicts trustworthy.

## Commands & Outputs (load-bearing)
```
# trade-log audit
$ find output -name "trades.csv" | (fx ones): FxTSMOM, FxXSectMom, FxCarry, FxGoldSilver  # only 4 of 12 have fills logs
$ duckdb output/experiments.duckdb: runs=455, return_streams rows=290708, distinct runs w/ return_stream=135
# strategy_lead_gate hook FIRED on a top-level command containing "gate/verdict/walkforward" -> confirms hook works for top-level, NOT for subagent-launched backtests (root of the governance gap)
$ registry fx runs: 30
```

## Files Touched (this session, all on main/pushed)
- `docs/strategies/research/20260719_fx_catalog_campaign_synthesis.md` -- created then marked RESOLVED.
- `docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md` -- Wave 2 pre-registration.
- `docs/superpowers/specs|plans/2026-07-19-fx-spread-engine*.md` -- Track B spread engine spec + plan.
- `src/backtesting/sessions/fx_clock.py` -- DST-gap fix (`5a49980`).
- `src/backtesting/engine/spread_sizing.py`, `fx_spread_simulator.py` -- beta-weighted spread engine.
- `src/strategies/advanced/fx_{audnzd_pairs,coint_scanner,vol_ratio_pair,spread_base}.py` -- Track B strategies.
- `src/strategies/advanced/fx_{turn_of_month,pca_dollar_residual,roro_regime_spread}.py` -- Track A strategies.
- `scripts/backtest_scripts/run_fx_spread_{backtest,walkforward}.py` -- spread runners (gitignored dir, force-added).
- `.claude/agents/strategy-lead.md` -- trade-log enforcement gate (`3106b9c`).
- `docs/progress/20260719_fx_wave2_orchestration.md` -- orchestrator session log.
- Memory: `~/.claude/projects/.../memory/fx-catalog-campaign.md` + `MEMORY.md` -- marked CLOSED.

## Key Takeaways & Gotchas
- **The finding:** retail G10 FX daily/session/RV catalog has NO deployable edge after realistic costs in 2011-2026, robust to IBKR-optimistic costs, across 8+ mechanisms. A genuine well-earned NEGATIVE result (North Star: surfacing failure = completed objective).
- **strategy_lead_gate hook only catches TOP-LEVEL commands, not subagent-launched backtests.** This is why the 2 Wave-1 enhanced verdicts bypassed strategy-lead. Enforcement of the mandated pipeline rests on orchestrator routing discipline, not the hook.
- **Trade-logging gap:** custom walk-forward runners (S&P-relative, R-multiple, spread-book) aggregate to daily returns + log return-streams but do NOT persist fills-level `trades.csv`. Only the 4 naive strategies (config-driven `backtest_runner` path) have fills logs. Fixed the agent def going forward; the 8 existing logs are not backfilled.
- **Concurrency:** earlier in the campaign a futures session shared the tree and caused 2 unauthorized subagent-to-main pushes + a swept-in commit. Give each concurrent session its own git worktree; subagents commit-only.
- **Reusable assets built (survive the campaign):** spot-FX daily engine, intraday minute-bar order engine (OCO/bracket/trailing), beta-weighted spread simulator, FX session/DST clock, tier-1 EUR/GBP event calendar, S&P benchmark harness, carry-unwind score, 8 computed artifacts, cost-sensitivity walk-forward harness.
- **Registry debt:** `n_trials_project_wide` counts only `agent_name='backtest-optimizer'` rows -> returns ~0 for parameter-free walk-forwards; strategy-lead computed the honest N explicitly instead. Fix before any future DSR-gating.

## References
- Resolution: `docs/strategies/research/20260719_fx_wave2_resolution.md`, `20260719_fx_wave2_trackB_results.md`, `20260719_fx_wave2_trackA_results.md`.
- Orchestration log: `docs/progress/20260719_fx_wave2_orchestration.md`.
- Tracker: `docs/strategies/FX_60_CATALOG_TRACKER.md`.
- Prior handoff: `docs/progress/SESSION-HANDOFF-fx-catalog-campaign-2026-07-06.md`.
- GitHub: https://github.com/shuyangw/Homeguard (main @ 3106b9c).
