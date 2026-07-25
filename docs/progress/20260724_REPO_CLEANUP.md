# Repo Cleanup Canvas + Execution - 2026-07-24

## Summary
Full-codebase canvas for dead/unused code, followed by execution: untracked
campaign debris backed up to `../Homeguard_legacy` and deleted, ~8.7k tracked
LoC of verified-dead code removed on `chore/repo-cleanup`, plus a .gitignore
fix that stops `__pycache__` polluting `git status`.

## Canvas Findings (verification results)

Kept after verification (looked dead, is not):
- `src/backtesting_v2/` -- v1 `from_signals` delegates to it
  (`portfolio_simulator.py:1075`); it is the live sim path
- `src/research/regime_momentum_lab/` -- harness for the open RAMP
  "hybrid signal+regime-overlay" lead; TODO.md notes a rename in flight
- `src/experiments/`, `src/monitoring/`, `src/streaming/`,
  `src/visualization/`, `src/features/`, `src/strategies/{research,
  implementations,universe,opex,options/csp}`, `tools/options-downloader`
  -- all have active consumers or are asset-class dev infra
- `scripts/backtest_scripts/_archived/` -- deliberately archived with README
- `models/bayesian_reversion_model.pkl`, tracked FX `output/` verdict
  artifacts -- intentional

Key correction found during verification: the options `_archived` trees were
NOT shelved to any git ref (scanned all local+remote branches) -- deleting
them without backup would have been permanent loss. Hence the legacy-folder
backup below.

## Changes Made

Backup (before any deletion): `C:\Users\qwqw1\Dropbox\cs\github\Homeguard_legacy`
(518 files, 7.4MB, source paths preserved):
- `src/strategies/options/_archived/` + `tests/strategies/options/_archived/`
  (~29k lines, options campaign legs -- unrecoverable from git)
- 161 untracked `scripts/backtest_scripts/` one-offs (kept the 2 with recent
  mtime: `run_spae_verdicts.py`, `test_futures_sharpe.py`)
- Root strays (`__main__`, `TODO.md.tmp.*`, `settings.ini.bak`, stray
  `python -m` logs)

Deleted without backup (junk): `archive/` (pycache only), `.tmp/`, `cache/`
(empty history), `offline_charts/`, `.coverage`.

Tracked removals (branch `chore/repo-cleanup`):
- **NautilusTrader adapter** (3,258 LoC): single commit 2025-12-12, dep never
  in requirements.txt, import guarded off, zero refs anywhere
- **Root committed `python -m` logs**: `ramp_phase4_wave3_readiness` + 2 others
- **RAMP CSP drivers** (`scripts/backtest/ramp_csp_*.py`): campaign closed
  2026-04-02 "do not revive"; CSP engine + tests kept
- **25 stale one-off scripts** (5,202 LoC): 17 Alpaca-era `scripts/trading`
  phase-validation/demo scripts (superseded by IBKR smoke + contract tests),
  `scripts/debug/` (5), `scripts/probe/` (3)
- **.gitignore fix**: `!src/data/artifacts/**` re-include was overriding the
  global `__pycache__/` rule
- **Committed 10 outstanding session-handoff logs** (Jul 2-21)

## Commits
- `bf201e4` chore: remove unused NautilusTrader adapter layer
- `d21c414` chore: remove accidentally committed python -m output logs from root
- `c57b3fa` chore: remove RAMP CSP campaign backtest drivers
- `cf2c4cf` chore: remove superseded one-off validation and probe scripts
- `2a4fb91` fix(gitignore): stop artifacts ** re-include from exposing __pycache__
- `cc07c09` docs(progress): commit outstanding session handoff logs (Jul 2-21)

## Validation
- `import src.backtesting.adapters` / `src.backtesting.engine` OK after
  nautilus removal
- pytest `tests/backtesting tests/trading tests/data` (fintech env):
  2217 passed, 7 failed, 7 skipped
- ALL 7 failures verified to fail identically on main (re-run there before
  merge) -- pre-existing local data/env gaps, none caused by this cleanup:
  - `test_rolling_mode` x3 (missing AAPL 2024-01 slice on H:)
  - `test_fx_carry_seatbelt_configs` x2 + `test_fx_spread_backtest`
    (fx_daily/ cache never built on this machine; builder script intact)
  - `test_walkforward_parallel::test_parallel_equals_serial`,
    `test_sentiment_analyzer::test_get_daily_sentiment`
- Excluded from collection (deps not in local env, unrelated to cleanup):
  `test_dukascopy_fx` (dukascopy_python), `test_holidays_calendar` (holidays)

## Known Issues / Remaining Work
- TEST-HYGIENE BUG (pre-existing, surfaced during validation): running
  `tests/trading`/`tests/data` writes stray logger files named
  `scripts.data.build_*` into the repo root and rewrites the REAL
  `config/trading/strategy_toggle.yaml` (metadata only, flags unchanged --
  restored). Tests should redirect both to tmp_path.
- `.superpowers/sdd/progress.md` modified by another in-flight session -- left alone
- `settings.ini` still tracked despite machine-local paths (user call)
- `docs/planning/` vs `docs/plans/` consolidation not done (user call)
- `logs/` (368MB) + `output/` (293MB) runtime dirs not pruned (user call)
- Tier-3 items intentionally KEPT: `regime_momentum_lab`,
  `scripts/backtest_scripts/_archived/`
