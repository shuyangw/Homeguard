# Strategy Pipeline Rules

These rules load alongside CLAUDE.md at session start. They apply to the orchestrator. Subagents do NOT read this file — the orchestrator must include relevant rules in every dispatch prompt.

## Strategy testing goes through strategy-lead — even under superpowers

Superpowers (brainstorming / writing-plans / subagent-driven-development) governs BUILDING (engines, wrappers, data plumbing). It does NOT exempt strategy TESTING from `strategy-lead`. Any phase that runs a backtest / walk-forward / statistical gate (PSR/DSR/PBO) / smoke producing a strategy VERDICT must be delegated to `strategy-lead`, which owns the integrity gates and the backtest sentinel. When a superpowers plan contains a verdict/gate/smoke phase, split it: build tasks run via subagent-driven-development; verdict phases go to `strategy-lead`. This is hard-enforced by the `PreToolUse` hook `.claude/hooks/strategy_lead_gate.py` (registered in `.claude/settings.json`): backtest/gate/smoke commands are DENIED unless `strategy-lead` has created the `.claude/.strategy-lead-active` sentinel. A blocked backtest is the guard, not a bug — route through `strategy-lead`.

## Session recovery protocol

1. On EVERY session start (fresh or resumed), read TODO.md FIRST
2. If `.claude-recovery.log` exists, read it for interrupt context
3. Any phase marked `[~]` was interrupted — restart it from the beginning
4. Read output files from completed phases to rebuild context — don't re-run them
5. For interrupted optimizer runs: check `output/optimization/<strategy>/` for the progress chronicle
6. For interrupted backtests: check `docs/reports/<strategy>/` for partial reports

## Agent dispatch rules

### Read-only agents (code-explorer, code-architect, code-reviewer)
- Return text only — orchestrator MUST write their output to `docs/agent-learnings/<strategy>/`
- Their analysis is gone if you don't save it

### Self-writing agents (backtest-driver, backtest-optimizer)
- backtest-driver writes to `docs/reports/<strategy>/`
- backtest-optimizer writes to `output/optimization/<strategy>/`
- Orchestrator reads and extracts key metrics into TODO.md
- Verify output files exist before marking phase complete

### All agents
- Subagents have NO context from your session — prompts must be self-contained
- Include ALL relevant Homeguard file paths in every dispatch prompt
- Include the applicable integrity rules (from strategy-lead.md Section 2) in every dispatch
- Include data frequency from the strategy spec in every backtest/optimizer dispatch
- Include data coverage rule: use FULL available data range, never a subset window
- After return: read the output files, don't trust the summary alone

## TODO.md discipline

- Mark `[~]` BEFORE starting a phase (so recovery knows what's in progress)
- Mark `[x]` AFTER verifying output files exist
- Never skip filling in the backtest iterations table — record ALL metrics
- Never overwrite previous results — increment run numbers

## Backtest integrity (authoritative)

**`docs/methodology/backtesting.md` is the single source of truth.** When this file and the methodology disagree, the methodology wins. Read the relevant section directly; do not paraphrase from memory.

| Topic | Methodology section |
|---|---|
| Bias prevention (lookahead, survivorship, selection, normalization, vol-target, full-data coverage) | Section 1 |
| Statistical gate (Sharpe, PSR, DSR with correct formula, PBO, combined gate) | Section 2 |
| Walk-forward (purging, embargo -- embargo is NOT the feature lookback) | Section 3 |
| Cost models by asset class + 1.5x cost-sensitivity gate | Section 4 |
| Stopping conditions (statistical floor, diminishing returns, overfitting trip, parameter sensitivity) | Section 5 |
| Data quality, point-in-time, news timestamps, data-snapshot reproducibility | Section 7 |
| Reproducibility identity fields (git SHA, config SHA, snapshot date, env hash, seeds) | Section 8 |
| Experiment registry schema (`output/experiments.duckdb`) | Section 9 |
| Homeguard paths, regime detectors, brokers, services, EC2 env | Section 10 |
| Exit logic, stops, profit-taking, MAE/MFE, asset-class profit-taking rules | Section 11 |
| Required diagnostic outputs (trade-level metrics, capacity, regime transitions, parameter stability, benchmark/IR) | Section 12 |

Inline magic-number thresholds ("Sharpe > 3.0 REJECT", "CAGR > 20% INVESTIGATE") are removed -- they were replaced by the combined statistical gate in methodology Section 2.5.

## Metrics -- record all of these for every backtest run

Sharpe, PSR, DSR (using project-wide cumulative trial count), PBO, CAGR, Max DD, Max DD duration, Calmar, win rate (monthly), profit factor, trade count, avg hold time, regime robustness, cost sensitivity (1.5x), IS/OOS Sharpe ratio, backtest window, data frequency. Append to the experiment registry per methodology Section 9.3.

## Trade logging -- MANDATORY for every backtest, EVERY asset class

Every backtest run MUST persist a simulated-trade log (the fills/position changes), not just aggregate metrics -- equity, crypto, AND futures. This is methodology Section 12 and is non-negotiable. The equity/crypto path does this via `backtest_runner` -> `TradeLogger` (gated on `output.save_trades`, default True); the futures path does it via `run_futures_backtest(..., log_trades=True)` writing `output/backtests/futures/<strategy>/<start>_to_<end>/{trades,equity,margin_utilization}.csv`. When adding a NEW backtest engine or asset-class path, wiring trade-log persistence is a REQUIRED part of the task -- a run that produces only metrics/equity and discards its fills is incomplete and must be rejected in review. Validation-harness internals (e.g. per-window walk-forward runs) may suppress logging, but the primary/representative backtest for a strategy MUST produce one.

## Run-status logging -- MANDATORY for every long/background run

Any long-running or backgrounded run (walk-forward, cache build, multi-hour backtest) MUST be wrapped in `RunStatus` (`src/utils/run_status.py`). It writes a JSON status file under `output/run_status/<name>_<ts>.json` with a background heartbeat, so a killed run leaves a stale `RUNNING` sentinel plus its last `heartbeat_at` -- telling us it died and roughly when. NEVER rely on the process's own stdout log to explain a death: a `SIGKILL`'d process cannot self-log, and a shell-level `echo "exit $?"` dies with the shell (this is exactly why the 2026-07-03 carry re-baseline kill left no captured reason). When launching such a run, also: (a) prefer a completion sentinel/`--json` output you can check afterward, and (b) do NOT switch git branches or mutate the working tree while it runs (parallel workers re-import code from disk on spawn). After any run that was killed, read its `output/run_status/` file for the last-alive time BEFORE guessing the cause.

## Homeguard file structure

- Strategy specs: `docs/strategies/production/<n>.md`
- Strategy code: `src/strategies/advanced/<n>.py`
- Strategy base classes: `src/backtesting/base/`
- Strategy configs: `config/strategies/<n>.yaml`
- Backtest configs: `config/backtesting/`
- Backtest engine: `src/backtesting/engine/`
- Walk-forward chunking: `src/backtesting/chunking/`
- Optimization framework: `src/backtesting/optimization/`
- Regime detection: `src/backtesting/regimes/detector.py`
- Reporting: `src/backtesting/reporting/`
- Position sizing/risk: `src/backtesting/utils/`
- Symbol universes: `config/universes/`
- Backtest scripts: `scripts/backtest_scripts/`
- Tests: `tests/strategies/test_<n>.py`
- Backtesting guidelines: `docs/guidelines/`
- Phase analysis output: `docs/agent-learnings/<strategy>/`
- Backtest reports: `docs/reports/<strategy>/`
- Optimization output: `output/optimization/<strategy>/`
- Backtest run data: `output/backtests/<strategy>/`
- Infra patterns: `docs/architecture/infra_patterns.md`
- Implementation skill: `.claude/skills/implement-strategy/`
- Recovery log: `.claude-recovery.log`
