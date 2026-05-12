# RAMP metrics fix + methodology rollout phases 1-3 - 2026-05-12

## Summary

Fixed the `homeguard-multi` crash loop that had stopped all RAMP metrics, then landed phases 1-3 of the new authoritative backtesting methodology rollout (place doc, update governance, point existing agents at it).

## Changes Made

- **`src/trading/brokers/ibkr/connection.py:198-225`** -- subscribe to IBKR account updates in `_connect()` via `self._ib.client.reqAccountUpdates(True, acct)` plus a 1-second wait so `ib.portfolio()` returns the populated PortfolioItem list when callers read it right after `start()`. Without this, `IBKRBroker.get_stock_positions()` returned empty, the startup reconciliation in `scripts/trading/run_live_paper_trading.py:1131-1209` compared the 20 RAMP positions in state against an empty broker result, logged `POSITION MISMATCH DETECTED`, and exited 1 -- crash loop. First attempt used the sync wrapper `ib.reqAccountUpdates(acct)`, which raised "event loop already running" because it calls `run_until_complete` from inside the connection manager's running loop; corrected to the low-level client method.
- **`docs/methodology/backtesting.md`** -- new 915-line authoritative methodology covering bias prevention, statistical framework (PSR / DSR / PBO with correct formulas), walk-forward (purge + embargo as separate concepts), cost models per asset class, stopping conditions, portfolio integration, reproducibility, and an experiment-registry schema. Replaces inline rules scattered across agents.
- **`CLAUDE.md`** -- Backtesting section now points at the methodology as authoritative ("when agent prompts and this file conflict, this file wins").
- **`.claude/rules/strategy-pipeline.md`** -- inline "Backtest integrity" and "Overfitting thresholds" sections replaced by a section-pointer table. Magic-number thresholds ("Sharpe > 3.0 REJECT", etc.) explicitly retired.
- **`.gitignore`** -- allow-list `docs/methodology/` (was caught by `docs/*` deny rule).
- **`.claude/agents/backtest-driver.md`** -- opens with "read methodology Sections 1-4, 8-10"; Result Validation Thresholds table replaced by a pointer to the combined statistical gate (Section 2.5).
- **`.claude/agents/backtest-optimizer.md`** -- opens with "read Sections 1-3, 5, 8-9"; the old DSR approximation `Sharpe * (1 - ln(N)/(2T))` is explicitly retired in favor of the real Bailey & Lopez de Prado formula in Section 2.3, with project-wide trial count from the registry.
- **`.claude/agents/trading-lead.md`** -- orchestrator-level pointer to Sections 1, 5, 6, 10; Overfitting prevention table replaced by the combined gate; dispatch template now tells subagents which methodology sections to read.
- **`.claude/agents/trade-log-analyzer.md`** -- pointer to Section 10 for paths/env; stale 900MB memory alarm rewritten to 3GB per actual t4g.medium capacity.
- **`.claude/agents/code-reviewer.md`** -- pointer to Sections 1 (bias prevention) and 7 (PIT) when reviewing strategy or backtest code.

## Commits

- `aea1443` fix(ibkr): subscribe to account updates so portfolio() populates  (first attempt, sync wrapper -- introduced "event loop already running" regression)
- `8be75a9` fix(ibkr): use low-level client.reqAccountUpdates to avoid loop nesting  (correction)
- `8c9fcbd` docs(methodology): add authoritative backtesting methodology
- `592bc7c` docs(agents): point quant agents at the methodology

## Known Issues / Remaining Work

- **Per-position metrics** (`hg_position_qty{symbol=...}`, `hg_position_unrealized_pnl_usd{symbol=...}`) did not emit on the post-fix restart; only top-level `hg_strategy_equity_usd`, `hg_portfolio_*` are flowing. Total metric count is 8 vs the historical ~50. Likely needs a strategy tick (next RAMP rebalance at 15:55 ET) or a market-data subscription line per symbol -- the "symbol limit exceeded (405)" and "code 10167 delayed market data" warnings in the journal suggest the market-data tier is constraining position-level pricing. Not a blocker for live trading but worth investigating before next session.
- **Methodology phases 4-8 are planned but not started** (see `C:\Users\qwqw1\.claude\plans\effervescent-wandering-seal.md`):
  - Phase 4: experiment registry (`src/experiments/`, `output/experiments.duckdb`)
  - Phase 5: statistical helpers (`src/backtesting/statistics/` -- PSR, DSR, PBO)
  - Phase 6: extend `WalkForwardValidator` with `purge_days` and `embargo_pct` parameters
  - Phase 7: cost model registry (`src/backtesting/costs/` per asset class)
  - Phase 8: new agent definitions (`strategy-architect`, `strategy-implementer`, `portfolio-integrator`) -- pending naming decision A/B/C in plan file
- **Pre-existing test failures unrelated to this session**: `tests/trading/brokers/ibkr/test_config_and_errors.py::TestIBKRConfig::test_defaults`, `test_paper_detection`, `test_gateway_type_label` expect client_id=1 but code has 10. Also 5 `test_contracts.py` tests error on missing `ibkr_connection` fixture. None caused by this session's edits; worth a cleanup PR.
- **Pre-existing uncommitted changes** in working tree (model bumps for code-architect / code-explorer / codebase-analyzer; data acquisition migration scripts) were left alone -- they aren't part of this session.

## Validation

- Diagnostic on EC2 (separate clientId 97-99) confirmed: `ib.positions()` returned all 20 RAMP positions immediately; `ib.portfolio()` returned 0 until `client.reqAccountUpdates(True, acct)` was called, then 20 within ~250ms. Validates root cause and fix.
- Local `pytest tests/trading/brokers/ibkr/` excluding the 1 pre-existing failure: 118 passed.
- After deploying `8be75a9` to EC2: `systemctl is-active homeguard-multi` = `active`; journal shows `LIVE PAPER TRADING - CONTINUOUS MODE` (past the reconcile gate that was failing); `hg_strategy_equity_usd{strategy="ramp"}` = $102,158.61 and `hg_portfolio_equity_usd{broker="ibkr"}` = $1,014,605.58 are emitting.
- `grep -rn "Sharpe > 3.0\|Sharpe > 1.5\|CAGR > 20%" .claude/` returns only commentary lines that explicitly retire the old thresholds -- no live rules remaining.
