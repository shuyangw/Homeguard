# Futures Campaign Comprehensive Retest (SP-A..SP-E) - 2026-07-11

## Summary
Ran the first strategy-lead-governed, honestly-deflated re-validation of the entire
futures catalog (26 gradeable strategy/root/pair combos) in an isolated worktree.
Fixed a repo-wide DSR deflation inconsistency (Gate 0) plus two real bugs found along
the way. Verdict: ZERO of 26 clear the combined statistical gate, carry incumbent
included (OOS Sharpe 0.7646, DSR 0.824 < 0.95). This confirms the pre-registered honest
expectation and is the completed objective, not a failure to engineer around.

## Changes Made (all on branch `feat/futures-retest`, isolated worktree)
- **Gate 0.1 deflation-consistency fix** (`170946b`): `run_carver_walkforward.py`,
  `gate_session_stream` (session_walkforward.py), `run_fx_walkforward.py`,
  `satellite_blend.blend_books` all computed DSR against a single-element `[oos_sharpe]`
  list (collapses `expected_max_sharpe` to ~0 -> DSR == undeflated PSR). Now deflate
  against the real `CAMPAIGN_TRIAL_SHARPES` distribution, mirroring `gate_return_stream`.
- **Gate 0.2 honest, growing N** (`170946b`): added
  `walkforward_common.get_campaign_trial_distribution()` -- sources n_trials + the
  trial-Sharpe distribution from `output/experiments.duckdb` (static 40/29 baseline +
  every subsequently-appended run), safe fallback to constants, never raises.
- **Gate 0.3 committed Path-2 sleeve drivers** (`b8cdec7`): ten `sp_retest_*.py` drivers
  + `sp_retest_common.py` + `sp_retest_trade_log.py`. Code review caught and fixed one
  CRITICAL issue: the calendar/processing/ratio drivers originally claimed a
  converge/structural/time-stop exit-reason breakdown that `SpreadTrade` does not persist;
  replaced with `convergence_exit_summary` that reports only what `trades.csv` contains
  (no fabricated diagnostics).
- **#16 caveat-fix** (`095c627`): the walk-forward runner hardcoded `rebalance: "weekly"`
  for every window regardless of the config's declared frequency, mis-sampling
  turn-of-month's daily signal. Threaded the config's declared rebalance through the call
  chain. Sign flipped -0.274 -> +0.0815 (legitimate bias correction, reviewed before
  re-gating); still fails.
- **RunStatus reliability fix** (`757318f`): added 5-attempt retry-with-backoff to
  `RunStatus._write` for the Dropbox-indexer `WinError 5 PermissionError` lock class that
  `registry.py` already handled; was intermittently killing Tier 2 multi-root loops.
- **Data repair** (not a code change): deleted corrupted cache
  `H:\Stock_Data\futures\roll_volume\MNQ\2020.parquet` (regenerable) that broke
  `test_futures_walkforward.py`.
- **Durable portfolio summary**: `docs/strategies/research/20260711_FUTURES_RETEST_PORTFOLIO_SUMMARY.md`
  (tracked copy of the gitignored worktree report).

## Commits
Branch `feat/futures-retest` (code, in worktree -- NOT yet merged to main):
- `170946b` fix(gate0): thread honest, growing DSR deflation into all un-deflated gate paths
- `b8cdec7` feat(gate0.3): committed Path-2 sleeve drivers for the futures retest
- `095c627` fix(gate0-caveat): #16 FuturesTurnOfMonth daily-rebalance mis-sampling
- `757318f` fix(run-status): retry-with-backoff on the atomic status-file rename
- (this session log + durable summary committed on top)

Branch `main` (ledger only, committed directly by strategy-lead during the run):
- `22209b6` docs(futures-retest): Gate 0 complete + Tier 1 carver strategies re-gated
- `2122c80` docs(futures-retest): Tier 2/Tier 3 complete -- zero strategies clear the gate

## Known Issues / Remaining Work
- **MERGE PENDING (needs user):** `main` and `feat/futures-retest` have DIVERGED (strategy-lead
  committed the ledger to main; code is on the branch -> non-ff). A merge is blocked by a
  pre-existing broken worktree gitlink on main (`.git/worktrees/ramp-equity-fix` ->
  `fatal: not a git repository` on any working-tree op), the Dropbox gitlink hazard.
  Recommended reconciliation (targeted, with user go-ahead): `git worktree prune` to clear
  the stale `ramp-equity-fix` registration, verify status is safe, then merge
  `feat/futures-retest` -> main (disjoint files, no conflicts), push, then remove the worktree.
- Worktree `.worktrees/futures-retest` LEFT INTACT for review -- the per-strategy
  `docs/reports/futures/*_READINESS.md`, `output/*_gate.json`, and `output/run_status/` files
  are gitignored and worktree-local; they vanish on worktree removal.
- Engine follow-ups (out of scope, flagged in the portfolio summary): (1) `simulate_convergence`
  doesn't differentiate stop-exit slippage per Section 11.5; (2) `SpreadTrade` lacks an
  `exit_reason` field (Section 11.9 breakdown unavailable for #31-#34); (3) #36 book-correlation
  vs RAMP not run; (4) #31 NG RollCalendar F1/F2 fix deprioritized. None change any verdict.
- 4 pre-existing `test_walkforward_idm_threading.py` failures (stale fixture missing
  `validate_prereg` kwarg) predate this branch; left as-is.

## Validation
- Gate 0 verified: on a synthetic positive-Sharpe stream, `gate_return_stream` and
  `gate_session_stream` now yield DSR < PSR (deflation bites); VIX/session/spreads/walkforward
  suites at accepted baseline (38 passed / 4 pre-existing unrelated failures).
- Carry incumbent OOS Sharpe 0.7646 matches the long-documented 0.765 walk-forward figure
  exactly -- confirms Gate 0 changed only the gate math, not signal generation.
- All 26 gradeable combos have a row in the retest TODO iterations table with full metrics;
  each Path-1 strategy has a `docs/reports/futures/<STRAT>_READINESS.md`, each Path-2 sleeve a
  `returns.csv` + `gate.json`; runs appended to `output/experiments.duckdb`.
