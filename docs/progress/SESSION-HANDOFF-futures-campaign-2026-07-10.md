# Session Handoff: Futures Strategy Testability Campaign (SP-B1 + gate verdicts)

**Date:** 2026-07-10 | **Working dir:** C:\Users\qwqw1\Dropbox\cs\github\Homeguard | **main @ 3830f3e == origin/main** (github.com/shuyangw/Homeguard)

## Resume Here (read this first)
- **Goal:** Make the ~53-strategy futures catalog (2 Downloads deep-dive docs) testable in Homeguard's walk-forward PSR/DSR/PBO gate, then find anything that beats the incumbent carry_idm (OOS Sharpe 0.765). 5 sub-projects: A (daily wrappers), E (external data), B (intraday engine), C (spread engine), D (options-IV).
- **Status:** SP-A + SP-E + SP-B1 DONE + merged to main. Gate verdicts RUN on everything with data. HONEST BOTTOM LINE: nothing beats carry_idm (0.765). Only #13 carry-trend PASSES (0.357, weaker + a re-expression). #26 VIX roll-down (+0.564) is the one positive lead but PBO NaN (needs deflation). Remaining: SP-C (spread engine), SP-D (options-IV), SP-B2 (rest of intraday).
- **Next steps (user was asked to pick, no answer yet):** (1) START SP-C -- yield-curve steepener #35 first (my recommendation: [A]-tier, DV01-neutral, genuinely orthogonal rates curve, uncorrelated with all that failed); OR (2) finish the VIX #26 deflation (the one positive result, needs a real return-stream deflation + cost + best-of-N); OR (3) SP-D / SP-B2.
- **Blockers / open questions:** Binance perp funding is GEO-BLOCKED (HTTP 451) from this location -> #49 funding carry is unit-tested only, no real data. The workflow per sub-project: brainstorm -> writing-plans -> subagent-driven-development in an ISOLATED git worktree (`.worktrees/<name>`), then finishing-a-development-branch merges as a fast-forward.
- **To resume, you need:** conda env `fintech` (run `conda run -n fintech ...` OR `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe`); storage at H:\Stock_Data via `get_local_storage_dir()`; Git Bash; targeted git only (Dropbox worktree-gitlink hazard makes bare `git status`/`git diff`/`git checkout`/`git reset --hard` FATAL -- use `git add <paths>`, `git commit`, `git log`, and merge via `git merge --ff-only` or ref-update). Run walk-forward with `POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1`.

## Original Task (this session)
User: "Run verdicts before SPB" then "Continue to SPB". I.e. run the actual walk-forward gate on the now-testable SP-A/SP-E strategies, THEN build SP-B (the intraday session engine). Earlier this session: user asked "why do we need options IV" (answered: only #28 needs it, overlaps #26 VIX; kept SP-D anyway per "No let's keep both").

## Subtasks & Progress
- [x] **Gate verdicts (2026-07-09)** -- ran `walk_forward_carver` on all data-having strats. Recorded in the SP-A/SP-E trial ledgers. NOTHING beats carry_idm 0.765.
- [x] **Gate-regression fix** -- SP-A's pre-registration gate had broken the walk-forward runner (over-fired on internal per-window configs). Fixed with `validate_prereg` flag + regression test (commit 723304a).
- [x] **SP-B1 brainstorm -> spec -> plan -> SDD (8 tasks) -> merge** -- intraday session engine + overnight drift + pre-FOMC. All merged (ff to 66bf019).
- [x] Session log + memory updated + pushed (3830f3e).
- [ ] **SP-C** (spread engine, yield steepener #35 etc.) -- NOT STARTED.
- [ ] **SP-D** (options-IV, #28) -- NOT STARTED. Must carry a correlated-re-expression check vs #26 VIX.
- [ ] **SP-B2** (other ~10 intraday: into-close #6, settlement #45, gap-fade #22, EIA #41, NFP #42) -- NOT STARTED.
- [ ] **VIX #26 deflation** -- the one positive lead (+0.564) needs a proper return-stream deflation + cost model + best-of-N before it can be trusted. PBO is NaN (single-config return stream).

## Gate Verdict Results (walk-forward 36/12/12, IDM on, 2010-2026, 13 windows)
Gate = PSR>=0.95 AND DSR>=0.95 AND PBO<0.25 AND 1.5x cost. Benchmark carry_idm 0.765 / PBO 0.189 / PASS.
| # | Strategy | OOS 1x | PBO | verdict |
|---|---|---|---|---|
| 13 | carry-trend gate | 0.357 | 0.189 | PASS (but < carry, a carry+trend re-expression) |
| 10 | curve-slope XS | 0.846 | 0.690 | WEAK (highest raw Sharpe but overfit -- XS-carry failure mode) |
| 23 | reversal | 0.297 | 0.805 | WEAK |
| 3 | XS commodity mom | 0.209 | 0.579 | WEAK |
| 15 | same-month season | 0.180 | 0.281 | WEAK |
| 37 | CoT tilt | -0.124 | 0.141 | REJECT |
| 16 | turn-of-month | -0.274 | 0.217 | REJECT (mis-sampled: daily signal on weekly-rebalance runner -- unreliable) |
| 26/27 | VIX roll-down (return stream) | +0.564 | NaN | needs deflation (the one positive lead) |
| 49 | funding carry | -- | -- | NO DATA (Binance 451) |
Recorded in docs/strategies/research/20260707_FUTURES_SP_{A,E}_TRIALS.md.

## SP-B1 Real Results (return-stream engine, gated like VIX)
| # | Strategy | Window (ET) | OOS 1x/1.5x | PBO | verdict |
|---|---|---|---|---|---|
| 21/25 | overnight drift | 16:00->09:30 | 0.792 / 0.671 | 0.513 | WEAK (positive premium, window-unstable) |
| 21 | hour-slice | 02:00->05:00 | -0.023 / -0.277 | 0.87 | REJECT (drift not in this window) |
| 39 | pre-FOMC | 14:00(F-1)->14:00(F) | n_windows=0 | NaN | UNGRADEABLE (8 events/yr never fills a 12mo/10-sample window) |
NONE clears the gate. Recorded in docs/strategies/research/20260710_FUTURES_SP_B_TRIALS.md.

## Key Decisions & Tradeoffs
- **SP-B is a return-stream engine, NOT a contract/margin sim.** Why: same simplification that made the VIX sleeve clean; sufficient for testability; cost is still modeled. Tradeoff: no dollar-P&L/margin (fine, deploys nothing).
- **SP-B scoped to SP-B1 = engine + overnight + pre-FOMC** (user chose "engine + overnight drift + pre-FOMC"). The other ~10 intraday strats are SP-B2 follow-ons, only if the engine + first strats validate (they did not clear the gate, but the engine works).
- **Session-bars cache** extracts ratio-adjusted 1-min closes at 5 ET times per root ONCE (avoids re-adjusting 5.5M bars per run -- the A1 lesson). Drop all-NaN Sunday rows (ES opens Sunday 18:00 ET) so the trading-day index is correct (weekend overnight representable).
- **Kept SP-D (options-IV) despite #28 overlapping #26 VIX** -- user said keep both; #28 gets a mandatory correlated-re-expression check vs #26.
- **Isolated git worktree per sub-project** (SP-E and SP-B both) -- fully solved the SP-A cross-session collision (a concurrent FX session was committing to main during SP-A). Worktree = clean fast-forward merges, no interleaving.

## Discussion Summary
- Verdicts confirmed the catalog's own skepticism: contested/marginal signals mostly do not clear. carry_idm stays the best deployable book. Running verdicts BEFORE building SP-B surfaced the pre-registration gate regression (the walk-forward runner was silently broken since SP-A merged).
- SP-B1 built via SDD, 8 tasks. The per-task review process caught 2 real bugs the offline tests missed: (1) all-NaN Sunday rows polluting the trading-day index (would drop the Friday->Monday weekend overnight); (2) a date-vs-Timestamp mismatch between the date-keyed cache/simulator and the DateOffset gate windows (centralized the fix in aggregate_returns -> DatetimeIndex). Also caught 2 wrong test fixtures in the plan (NaN at an unread cell; a self-contradictory holiday date).
- Overnight drift: a REAL positive premium (0.79 Sharpe) but PBO 0.51 -> window-unstable, fails the gate. Hour-slice (SR-917 ~02:00-05:00 ET approximation) is negative -> the drift is not concentrated there. Pre-FOMC: the walk-forward gate architecturally can't judge an ~8-events/yr stream (n_windows=0); decay split is small-n noise (and sqrt(252) annualization inflates the sparse-stream Sharpe, e.g. the "6.54" is not a real Sharpe).
- Final Opus review: MERGE-READY, 2 Important deferred to SP-B2 (they change no current verdict). Signs long, never flipped despite marginal/negative smokes.

## Commands & Outputs (load-bearing)
```
# gate verdict sweep (after fixing the runner)
$ .../python.exe -m scripts.backtest_scripts.run_spae_verdicts   # (scripts/backtest_scripts is gitignored, one-off)
carry_idm 0.765 PASS; #13 0.357 PASS; #10 0.846 PBO0.690 WEAK; #37 -0.124 REJECT; ...

# SP-B session-bars caches
build_session_bars_cache('ES'/'NQ') -> ES/NQ 4044 dates each (after dropping 831 all-NaN Sunday rows)

# SP-B smokes
overnight: 8086 trades, 1x 0.792/pbo 0.513, 1.5x 0.671/pbo 0.457
hour-slice: -0.023 / -0.277, pbo 0.87
pre-FOMC: 252 trades, gate n_windows=0, decay pre 0.25 / post 6.54 (noise)

# merges (both clean fast-forwards, no force-push)
verdicts+fix: 70e3faa..4dba5f7 (then SP-B off 4dba5f7)
SP-B: 4dba5f7..66bf019 -> main; log: 66bf019..3830f3e
```

## Files Touched (this session, all on main)
- `src/backtesting/engine/futures_backtest.py` -- added `validate_prereg` param (gate-regression fix).
- `scripts/backtest_scripts/run_carver_walkforward.py` -- `_run_window` passes `validate_prereg=False`.
- `tests/strategies/futures/test_pre_registration.py` -- regression test for the flag.
- SP-B NEW: `src/backtesting/sessions/equity_index_clock.py` (ET->UTC), `src/backtesting/session/{session_bars,session_simulator,session_walkforward}.py`, `src/strategies/advanced/{overnight_drift_strategy,prefomc_strategy}.py`, `src/data/futures/paths.py` (+session_bars_dir), tests under `tests/backtesting/session/` + `tests/strategies/futures/{test_overnight_drift,test_prefomc}.py`.
- Docs: `docs/strategies/research/20260707_FUTURES_SP_{A,E}_TRIALS.md` (verdict results), `docs/strategies/research/20260710_FUTURES_SP_B_TRIALS.md`, `docs/progress/20260710_FUTURES_SP_B.md`, specs+plans under `docs/superpowers/{specs,plans}/` (LOCAL-ONLY: docs/* is gitignored except negated subdirs; superpowers docs are NOT committed).
- Session-bars caches on disk: H:\Stock_Data\futures\session_bars\{ES,NQ}.parquet.

## Key Takeaways & Gotchas
- **docs/* is gitignored except negated subdirs.** superpowers specs/plans and SDD ledgers (.superpowers/sdd/) are LOCAL working docs, never committed. docs/progress/ and docs/strategies/research/ ARE tracked.
- **SDD ledgers use distinct paths** (progress-sp-a/e/b.md) to avoid the shared-ledger collision that happened in SP-A. Controller stays in the MAIN dir (plan/ledger/briefs there, gitignored); subagents build in the worktree; task-brief/review-package run from MAIN (shared object db resolves branch SHAs). See [[feedback_concurrent_sessions_worktrees]].
- **The walk-forward return-stream gate can't judge sparse-event strategies** (n_windows=0 for ~8/yr). Relevant for SP-B2 event strats (#41 EIA, #42 NFP) -- they will need a different gate (full-sample or event-study), not the daily walk-forward.
- **sqrt(252) annualization overstates sparse-stream Sharpe** -- do not read pre-FOMC's "6.54" as real.
- **SP-B2 hardening notes** (deferred, change no current verdict): aggregate_returns vol-normalizes over the FULL sample (mild in-sample; switch to trailing vol before promoting any session strat); the cost model uses regular_hours=True half-tick slippage even for the off-hours hour-slice (understates its cost; thread a per-window liquidity flag for off-hours strats); simulator return dict keyed by exit_date only (safe as used, footgun for >1-trade/day/root).
- **Reuse, do not rebuild:** SP-C should model each spread as a synthetic instrument + reuse the daily engine + A templates (per the umbrella design); only SP-B needed a new simulator. Yield futures (2YY/5YY/10Y/30Y) are cash-settled, constant $10/bp DV01 -> clean 1:1 steepener.

## References
- Umbrella campaign design: docs/superpowers/specs/2026-07-07-futures-strategy-testability-campaign-design.md (LOCAL)
- Comprehensive futures review: docs/strategies/research/20260705_FUTURES_STRATEGY_EXPLORATION_REVIEW.md
- Catalog source docs: C:\Users\qwqw1\Downloads\20260706_FUTURES_STRATEGY_DEEP_DIVE.md + compass_artifact_wf-...text_markdown.md
- Memory: project_futures_testability_campaign.md, feedback_concurrent_sessions_worktrees.md
