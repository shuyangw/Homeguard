# Futures Variation Campaign + Governance/Gitlink Fixes - 2026-07-11

## Summary
Second half of the 2026-07-11 session. Merged the strategy-lead-governed retest to main,
raised the per-strategy variation budget 2 -> 15, root-cause-fixed the long-standing Dropbox
git-hazard, then ran an EXHAUSTIVE pre-registered variation campaign on the whole futures
catalog. Final verdict unchanged and now exhaustively documented: across 23 graded retest
combos + 28 variations (honest cumulative N grown 66 -> 94), ZERO clear the combined
statistical gate. The apparatus (PBO especially) correctly rejected every high-Sharpe
overfit candidate. Companion to the earlier `20260711_FUTURES_RETEST.md`.

## Changes Made
- **Governance (strategy-lead budget 2 -> 15)** `233eb9e`: raised the Phase 6.5/7 per-strategy
  iteration ceiling, gated by the honest trial-count tax (every variation increments the
  growing project-wide DSR N, so a weak strategy cannot be iterated into passing). Added the
  bug/mis-spec-fix exception (uncounted), a diminishing-returns early stop, and replaced the
  retired Sharpe<3.0/CAGR<20% magic-number Phase 6 checklist with the combined gate (2.5).
- **Dropbox git-hazard ROOT-CAUSE FIX** `b093ef8` (main) / `b77f620` (branch): the recurring
  `fatal: not a git repository: .../.git/worktrees/ramp-equity-fix` was two worktree dirs
  accidentally committed as gitlinks (mode 160000, in HEAD): `.claude/worktrees/ramp-equity-fix`
  and `.claude/worktrees/sip-validation`. `git worktree prune` could not touch them (tracked
  tree entries, not live registrations). Untracked both + gitignored `.claude/worktrees/` and
  `.worktrees/`. Bare `git status`/`git diff` now work on main (verified exit 0). The
  "targeted-git-only" CLAUDE.md caution is now largely moot for main.
- **Retest merge** `315897b`: fast-forward-reconciled the diverged retest branch (Gate 0
  deflation fix + sleeve drivers + #16 fix + run-status retry) into main and pushed.
- **Variation campaign** (branch `feat/futures-variations`, NOT merged):
  - Gate 0-bis `cc5ae98`: repointed the experiment registry to `get_local_storage_dir()/
    experiments/experiments.duckdb` (persists across worktrees; `HOMEGUARD_EXPERIMENTS_DB`
    override for test isolation) and seeded the honest cumulative baseline N=66.
  - Phase A build-out `b1e3bcd`/`31d9bfb`: sleeve knob exposure (overnight entry/exit_col;
    crush window/entry_z/etc via `sp_retest_*` CLI, verified to reproduce documented Sharpes
    exactly), 4 carver variant YAMLs, `FuturesCarryVolRegimeStrategy` overlay (lookahead-
    reviewed + load-bearing falsification test).
  - Waves 1-2 `61ff376`/`70d845d`/`8a68acc`/`5fc551f`/`7753954`/`0450104`: 28 pre-registered
    mechanism-motivated variations. Carry M5 no-trade band in `FuturesPortfolioSimulator`;
    overnight M7 vol overlay; `FuturesCarryTrendGraded`; momentum lookback/horizon knobs; VIX
    1.5x-cost apparatus fix.
  - Close-out docs: `docs/strategies/research/20260711_FUTURES_VARIATIONS_{TODO,SUMMARY}.md`.
    Corrected the retest summary's count in-place (26/18 -> 23 graded + 4 Tier-3 ungradeable).

## Commits
- main: `233eb9e` (budget), `b093ef8` (gitlink fix), `315897b` (retest merge)
- `feat/futures-variations`: `cc5ae98`, `b1e3bcd`, `31d9bfb`, `62fa56d`, `61ff376`, `70d845d`,
  `8a68acc`, `5fc551f`, `7753954`, `0450104`, `b77f620`

## Known Issues / Remaining Work
- **`feat/futures-variations` is a clean fast-forward onto main (315897b), NOT merged** --
  awaiting decision. Touches the core futures engine (no-trade band), so surfaced before merge.
- Closest campaign miss overnight-drift M7 (DSR 1.0000/Sharpe 1.2145) REJECTS on PBO 0.4838;
  it masks ~29% of nights -- an overfit caught by PBO, not an edge. Not pursued (North Star).
- main has PRE-EXISTING uncommitted changes not from this work (`config/backtesting/fx_*.yaml`,
  `settings.ini`, `.superpowers/sdd/progress.md`) -- left untouched.
- Operational gotcha: subagents are NOT auto-notified when a backgrounded shell finishes, so a
  strategy-lead that backgrounds a run and returns will hang. Run gates synchronously or poll
  RunStatus to DONE within the turn.
- Carried-forward engine follow-ups: convergence stop-exit slippage + SpreadTrade.exit_reason;
  #36 book-corr vs RAMP; overnight M3 blocked on a missing YM session-bars cache.

## Validation
- Gate 0-bis: deflation bites at N=66 (synthetic 0.63 -> DSR ~2.5e-05); N grows as rows append;
  a genuinely strong record (Sharpe 1.18 / ~3000 obs) can still pass.
- Every variation gated once (PSR/DSR-at-N/PBO/1.5x-cost), FULL data range, appended to the
  persistent registry, recorded in the ledger iterations table (runs 1-28).
- Knob exposure verified behavior-preserving (defaults reproduce 0.7924 / 0.136 exactly).
- 20+ new/updated unit tests green; each build unit code-reviewed (lookahead checks on overlays).
- Zero variations clear both DSR and PBO, re-confirmed at final N=94.
