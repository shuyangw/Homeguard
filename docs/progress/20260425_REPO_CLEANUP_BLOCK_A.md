# Repo Cleanup Block A - 2026-04-25

## Summary

Executed Block A of the Homeguard repository cleanup plan: 8 commits, **25,327 LOC deleted across 127 files**, ~14% reduction in repo surface area. Removed legacy `PaperTradingBot` orchestrator chain, deprecated `_fetch_vix_yfinance` VIX shims, ghost strategy directories, broken `Makefile`, loose investigative scripts, stale `examples/`, and the unused GUI + Web UI layers (Flet desktop app + FastAPI/React web interface). Top of `src/` now reads as a clean inventory of what runs in production.

Plan source: `C:\Users\qwqw1\Downloads\homeguard_cleanup_plan.md`. Local fact-check corrected three plan claims (CSCMSignalLogger NOT dead, base_strategies has live callers, several test scripts depend on chain being deleted) before execution.

## Changes Made

- **Pre-0 (commit `3eedb79`)**: Added `After=homeguard-gateway.service` + `Wants=homeguard-gateway.service` to `infra/ec2/services/homeguard-omr.service` and `homeguard-ramp.service`. Defensive hygiene -- `deploy_multi_strategy_streaming.sh` already disables both, but `setup_multi_strategy.sh` would re-enable them.

- **Phase 0a (commit `aacd444`)**: Removed ghost dirs `src/strategies/{advanced_strategies,custom}/` and four `scripts/validate_{phase2,risk_management,risk_simple,gui_readiness}.py` files. -938 LOC.

- **Phase 5 (commit `8166cde`)**: Deleted root `Makefile`. Targets referenced files long gone (`backtest_scripts\RUN_ALL_BASIC.bat`, `src/run_ingestion.py`); mixed Win/Linux syntax. Equivalent commands documented in CLAUDE.md.

- **Phase 4 (commit `a8542a8`)**: Deleted 7 loose scripts (`test_vix_*.py`, `test_sweep_tearsheet.py`, `debug_streaming_buffer.py`, `show_news_*.py`, `sentiment_grid_search.py`) and the entire `examples/` directory (8 files, all written 2025-11/12 against pre-refactor APIs). -2,186 LOC. Held `scripts/backtest_omr_ramp_v2_visualization.py` for now.

- **Phase 0b (commit `03c2b5a`)**: After Phases 0a+4 cleared the six callers, deleted `src/strategies/base_strategies/` re-export shim. -19 LOC.

- **Phase 1.7 (commit `f6cb58d`)**: Deleted three `_fetch_vix_yfinance` shim tests in `tests/trading/test_adapters.py` first, then the methods themselves at `omr_live_adapter.py:598-601` and `momentum_live_adapter.py:435-438`. Column normalization tests preserved (they call `fetch_market_data`, not the shim). -103 LOC.

- **Phase 1 (commit `dd43777`)**: Removed legacy PaperTradingBot orchestrator chain. Updated `scripts/trading/test_e2e_imports.py` (dropped `PaperTradingBot` import probe + `test_trading_strategies()` function and main() call) and deleted `scripts/trading/test_omr_strategy_integration.py` first. Then deleted `scripts/ops/start_trading_bot.py`, `src/trading/core/paper_trading_bot.py`, and the entire `src/trading/strategies/` package (including `omr_live_strategy.py`). Cleaned exports/docstrings: `src/trading/core/__init__.py` (drop PaperTradingBot), `src/trading/__init__.py` (drop docstring mention), `src/trading/brokers/broker_factory.py` (replace example with ExecutionEngine). -1,409 LOC.

- **Phase 2 (commit `2f57805`)**: Removed `src/gui/` (Flet desktop app, ~9.4k LOC), `src/web/` (FastAPI backend + React frontend), GUI-coupled tests (`tests/gui/`, `tests/test_gui_*.py`, `tests/integration/test_phase1_backend.py`, `tests/validate_optimization_refactoring.py`), the entire `tests/legacy/` directory (test_sprint1-4, all GUI consumers), and 7 launchers in `scripts/ops/` (`run_gui*`, `start_gui*`, `start_web_ui*`). Dropped `fastapi`/`uvicorn`/`python-multipart` from `requirements.txt`. Removed `## Web UI` section and `gui/`/`web/` entries from CLAUDE.md's `src/` packages listing. **-20,615 LOC** (largest single commit).

## Commits

- `3eedb79` fix(infra): add gateway dependency to omr/ramp service unit files
- `aacd444` chore: remove ghost strategy directories and stale validation scripts
- `8166cde` chore: remove broken Makefile (stale paths, mixed syntax)
- `a8542a8` chore: remove loose investigative scripts and stale examples/
- `03c2b5a` chore: remove deprecated base_strategies re-export shim
- `f6cb58d` refactor(adapters): remove deprecated _fetch_vix_yfinance shims
- `dd43777` refactor: remove legacy PaperTradingBot orchestrator chain
- `2f57805` refactor: remove unused GUI and Web UI presentation layers

## Known Issues / Remaining Work

- **Block A not pushed to origin/main**. Awaiting user approval per project convention. Recommended next steps: `git push origin main`, then watch `homeguard-multi.service` on EC2 for 24 hours.
- **Block B (archive research strategies)** -- separate dedicated session per the plan. Moves ICT, BMSB, DSTS, EVR, FRS, HV-ORB, HurstMR, ML-Crypto-MR, OpexPinning, OPEX subsystem from `src/strategies/advanced/` into `src/strategies/archive/` with a registry split and pytest `archived` marker. Not started.
- **Block C (`backtesting_v2/` fold-back)** -- deferred until next backtesting infrastructure work.
- **Phase 7 (MA adapter cleanup)** -- skipped; `ma_live_adapter.py` is the simplest reference for the adapter pattern.
- **Phase 1.5 (CSCMSignalLogger)** -- intentionally skipped per user direction. `log_signal()` (hourly snapshots) is still actively called from `cscm_live_adapter.py:732`. Decision-log integration only replaced `log_rebalance()`.
- **Pre-existing test failures unchanged** -- 8 failed, 7 errored across `tests/trading/` (Dropbox file-lock flake on `test_close_overnight_positions_no_positions`, IBKR no-connection errors, EOD report tests referencing non-existent `LivePaperTrading` class with invalid `flush_interval_hours` kwarg, Windows chmod limitation). Net zero new regressions from this work.

## Validation

- Per-phase: `python -c "import src; import src.trading; import src.strategies"` succeeded after every phase
- Per-phase: `python scripts/trading/run_live_paper_trading.py --help` rendered correctly after every phase
- Phase 1.7: `pytest tests/trading/test_adapters.py -q` -- 49 passed, 1 pre-existing flake. The 3 deleted shim tests gone as intended.
- Phase 1: `grep -rn "PaperTradingBot\|OMRLiveStrategy"` across src/ infra/ scripts/ tests/ returned zero matches in production code (only historical references in docs/guides/ retrospectives, which were left intact).
- Phase 2: `grep -rn "from src\.gui\|from src\.web\|from gui\.\|from web\."` returned zero matches across all `.py` files after `tests/legacy/` was also deleted.
- After all of Block A: `pytest tests/trading/decision_log/` -- 38/38 passed (decision-log subsystem unaffected).
- `git diff --stat 3eedb79^..HEAD` -- 127 files changed, 6 insertions, 25,327 deletions.

Smoke test against IBKR (`scripts/trading/smoke_test_ibkr_paper.py --mode full`) was NOT run during this session (after-hours; IBKR Gateway not reachable from local). Recommend running on EC2 before pushing to verify the surviving live-trading call chain still works end-to-end.
