# Strategy Pipeline Rules

These rules load alongside CLAUDE.md at session start. They apply to the orchestrator. Subagents do NOT read this file — the orchestrator must include relevant rules in every dispatch prompt.

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
- Include the applicable integrity rules (from trading-lead.md Section 2) in every dispatch
- Include data frequency from the strategy spec in every backtest/optimizer dispatch
- Include data coverage rule: use FULL available data range, never a subset window
- After return: read the output files, don't trust the summary alone

## TODO.md discipline

- Mark `[~]` BEFORE starting a phase (so recovery knows what's in progress)
- Mark `[x]` AFTER verifying output files exist
- Never skip filling in the backtest iterations table — record ALL metrics
- Never overwrite previous results — increment run numbers

## Backtest integrity (enforced at every phase)

### No future data leakage
- All signals must use shift(1) or equivalent
- Train/test splits must be temporal, never random
- Features computed from only past data at each point
- Regime detection uses only data available at decision time
- Any data join checked for look-ahead
- Universe selection must not use future knowledge (survivorship bias)

### Overfitting thresholds
- Sharpe > 3.0 → REJECT
- Sharpe > 1.5 → VERIFY with Deflated Sharpe Ratio
- CAGR > 20% → INVESTIGATE
- Max DD < 5% on volatile assets → SUSPICIOUS
- Trades < 30 → INSUFFICIENT
- IS vs OOS gap > 30% → REJECT (strong overfitting)
- >70% returns from single regime → FRAGILE

### Full data coverage is mandatory
- Backtests MUST use the FULL available data range for every symbol, not a cherry-picked window
- Determine earliest available date per symbol; start backtest there (minus lookback)
- Symbols with shorter histories enter the universe when data becomes available
- Reserve most recent 1 year as true out-of-sample
- Short-window results (e.g., 2-3 years) are NOT evidence of viability
- Lesson: ramp-long-calls showed Sharpe 0.698 (2022-2024) but -0.767 over full timeline (2018-2024)

### Transaction costs are mandatory
- Every backtest must include realistic transaction costs (see cost tier table in trading-lead.md)
- Final validation must test at 1.5x costs
- If Sharpe < 0.5 at 1.5x costs → edge too thin for live
- Slippage model from `src/backtesting/engine/` must be active

### Walk-forward is mandatory for optimized parameters
- Minimum 3 rolling windows via `src/backtesting/chunking/`
- Report per-window OOS performance
- Walk-forward OOS degradation > 30% from IS → edge likely not real

### Parameter discipline
- Target ≤3 tunable parameters
- Every parameter needs economic rationale
- Neighbors +/-10-20% must also work (no cliff edges)
- Magic numbers are red flags

### Data frequency validation
- Strategy spec declares required frequency (daily/1min/5min)
- If strategy logic is intraday but spec says daily → flag mismatch before implementation
- Pass data frequency to every backtest/optimizer dispatch
- Factor runtime into session pacing (1min data = much longer runs)

## Metrics — record all of these for every backtest run

Sharpe, DSR-adjusted Sharpe, CAGR, Max DD, Max DD duration, Calmar (CAGR/MaxDD), win rate (monthly), profit factor, trade count, avg hold time, regime robustness, cost sensitivity (1.5x Sharpe), IS/OOS gap, backtest window, data frequency.

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
