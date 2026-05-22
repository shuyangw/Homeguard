# Homeguard - Algorithmic Trading Framework

General coding standards (encoding, git workflow, testing, defensive mindset, output efficiency, GUI) are in `~/.claude/CLAUDE.md`. This file covers Homeguard-specific guidelines only.

## Architecture

### src/ packages
```
strategies/      OMR, RAMP, CSCM, ORB + base classes, strategy registry (lazy loading)
trading/         ExecutionEngine, MultiStrategyRunner, BrokerInterface/DataProviderInterface (ISP)
                 IBKRBroker (primary for stocks/options via ib_async), AlpacaBroker (data + fallback),
                 Coinbase (crypto). Routing in config/trading/broker_routing.yaml
data/            CompositeDataProvider (Alpaca -> yfinance -> cache), acquisition manager, DuckDB loader
streaming/       LiveDataProvider, Alpaca WebSocket, 500-bar LRU buffer per symbol
backtesting/     BacktestEngine, PortfolioSimulator (Numba JIT), optimization (grid/Bayesian/genetic),
                 regime detection (5 states), walk-forward validation, reporting
backtesting_v2/  Next-gen backtesting (in development)
discord_bot/     Claude-powered read-only monitoring (slash commands)
discord_cscm/    CSCM-specific Discord alerts
screening/       Stock screener via Alpaca + yfinance fundamentals
utils/           Logger (ASCII-only, Rich), timezone (tz.now()), VIX fallbacks, TTL caching
settings/        get_local_storage_dir(), .env (API keys), settings.ini (paths), YAML configs
visualization/   QuantStats tearsheets, matplotlib/plotly charts
```

### Key data flows
- Backtesting: YAML config -> BacktestEngine -> DuckDB/Parquet -> PortfolioSimulator -> CSV/HTML
- Live trading: Strategy -> ExecutionEngine -> AlpacaBroker (REST) or IBKR (TCP) -> state JSON
- Data: Alpaca REST (primary) -> yfinance (fallback) -> Parquet cache (last resort)
- Streaming: Alpaca WebSocket -> LiveDataProvider -> live strategies
- Infra: Terraform -> EC2 t4g.medium ARM64 + Lambda/EventBridge (market hours start/stop)

Full component graph with edges/protocols: `docs/architecture/composer_diagram.json`

## Orientation (gaining context on unfamiliar areas)

1. Read the README.md or CLAUDE.md in the relevant `src/` subdirectory first
2. Read `__init__.py` for public API and exports
3. For strategy specs: `docs/strategies/`
4. For architecture deep dives: `docs/architecture/ARCHITECTURE_OVERVIEW.md`
5. For live trading state: `.claude/live_trading.md`
6. Use `find` and `grep` to explore -- don't read large files speculatively

## Role & Mindset

**You are an experienced algorithmic trader** with expertise in statistics, stochastic processes, systems architecture, market microstructure, and portfolio theory.

- Always consider 2-3 approaches with trade-offs before recommending one
- Be realistic: assess statistical significance, alpha decay, overfitting risk
- Challenge assumptions: is the pattern real? Will the backtest hold OOS?
- Propose simpler alternatives first: can a rule-based filter get 80% of the benefit?
- Think in probabilities, not certainties

Detailed overfitting thresholds and backtest integrity rules: `.claude/rules/strategy-pipeline.md` (auto-loaded)

## Environment

**CRITICAL**: All Python code execution MUST use the `fintech` conda environment.
- Location: `C:\Users\qwqw1\anaconda3\envs\fintech`
- Activate: `conda activate fintech`
- Details: [`.claude/environment.md`](.claude/environment.md)

## Data Handling

**CRITICAL**: Follow canonical schema. Use `get_local_storage_dir()` for paths.
- **Storage**: `from src.settings import get_local_storage_dir` - NEVER hardcode paths
- **Schema**: 8 columns (timestamp, open, high, low, close, volume, trade_count, vwap) - lowercase, float64
- **Download**: `python scripts/data/download_symbols.py --csv <file> --skip-existing`
- **Symbol lists**: `config/universes/sp500-2025.csv`, `russell1000-2025.csv`, `russell2000-2025.csv`
- Details: [`.claude/data_handling.md`](.claude/data_handling.md)

## Project Organization

- No script files in root directory
- Production scripts go in `src/`, `tests/`, `scripts/`
- **Experimental/one-off scripts** go in `scripts/backtest_scripts/` or `scripts/scratch/` (gitignored)
- Documentation co-located with modules
- Details: [`.claude/project_structure.md`](.claude/project_structure.md)
- Code standards and anti-overengineering rules: [`.claude/code_standards.md`](.claude/code_standards.md)

## Logging Standards

**CRITICAL**: Use centralized logging module (`src/utils/logger.py`) for all output.
- Never use `print()` statements
- **ALWAYS log exceptions** - Never silently swallow errors
- Use `logger.error()` for all caught exceptions
- Homeguard logger does NOT support `%s` positional args -- use f-strings
- Color-coded output (green=success, red=error, etc.)
- Details: [`.claude/logging.md`](.claude/logging.md)

## Backtesting

**CRITICAL**: Methodology is authoritative at [`docs/methodology/backtesting.md`](docs/methodology/backtesting.md). When agent prompts and this file conflict, this file wins. Read the relevant sections before any quantitative work.

- **ALWAYS use the config-driven backtesting system** - don't write ad-hoc scripts
- Run backtests via: `python -m src.backtest_runner --config config/backtesting/ma_single.yaml`
- Bias prevention, statistical gates (PSR/DSR/PBO), walk-forward purge/embargo, cost models, stopping conditions: see `docs/methodology/backtesting.md` Sections 1-5
- Use market calendar for trading day filtering
- Apply proper risk management
- Operational details: [`.claude/backtesting.md`](.claude/backtesting.md)

### Existing Backtest Tools (CHECK BEFORE CREATING NEW)

| Tool | Location | Purpose |
|------|----------|---------|
| **Standard Report** | `scripts/backtest/run_standard_report.py` | Monthly Sharpe/drawdown reports for any strategy |
| **Config-Driven Runner** | `python -m src.backtest_runner` | Main backtest runner with YAML configs |
| **Walk-Forward** | `config/backtesting/lgbm_walk_forward.yaml` | Out-of-sample validation |

**Standard Report Generator**: `src/backtesting/reporting/standard_report.py`
- Usage: `python scripts/backtest/run_standard_report.py --strategy <name> --symbols <list>`
- Outputs: Console, Markdown, CSV to `settings.ini` output directory

**Adding New Backtest Tools**: Add to `src/backtesting/`, register in `__init__.py`, document in this table. One-off scripts go in `scripts/backtest_scripts/` (gitignored).

## Existing Data & Screening Tools

| Tool | Location | Purpose |
|------|----------|---------|
| **Stock Screener** | `src/screening/` | Stock screener using Alpaca APIs. Docs: `src/screening/README.md` |
| **YFinance Fundamentals** | `src/data/yfinance/` | Market cap, P/E, sector, dividends. Docs: `src/data/yfinance/README.md` |
| **Equity Downloader** | `scripts/data/download_symbols.py` | Download OHLCV data via `src/data/acquisition/` plugin registry |

## Risk Management

**CRITICAL**: All backtests MUST use proper position sizing.
- Default: 10% per trade (moderate risk profile)
- Never use 99% capital per trade
- Five position sizing methods available
- Details: [`.claude/risk_management.md`](.claude/risk_management.md)

## Type Safety (CRITICAL)

**CHECK TYPES WITH EVERY CODE CHANGE!** Verify return types, parameter types, dict vs attribute access.
- Details: [`.claude/type_issues.md`](.claude/type_issues.md)

## Live Trading

**CRITICAL**: Watch for common live trading issues.
- **Type mismatches** - API data comes as strings; always convert explicitly
- **VIX data resilience** - Must have fallbacks for VIX fetch failures
- **Bayesian model coverage** - Model must be trained with ALL trading universe symbols
- **Market hours** - OMR: entry 3:50 PM, exit 9:31 AM ET. RAMP: rebalance 3:55 PM ET.
- **Timezone handling** - ALWAYS use `from src.utils.timezone import tz` and `tz.now()` instead of `datetime.now()`. EC2 instances run in UTC.
- Details: [`.claude/live_trading.md`](.claude/live_trading.md)

### IBKR Smoke Test (RUN AFTER ANY TRADING-CHAIN CHANGE)

**`scripts/trading/smoke_test_ibkr_paper.py`** -- end-to-end validation of the live trading call chain against IBKR paper.

**When to run:** after ANY change to `IBKRBroker`, `AlpacaBroker`, `ExecutionEngine`, `BrokerInterface` / `StockTradingInterface` / `OrderManagementInterface`, `broker_routing.yaml`, `IBKRConfig`, or any live adapter's order-submission path. ~25s, idempotent, safe after-hours.

```bash
# On EC2 (typical, after a deploy):
ssh ec2 'cd ~/Homeguard && source venv/bin/activate && python scripts/trading/smoke_test_ibkr_paper.py'

# Modes: --mode direct (broker only) | engine (ExecutionEngine only) | full (default, both)
```

**What it validates:**
- **Part 1** (`--mode direct`): `IBKRBroker.place_stock_order` / `get_order` / `cancel_order` lifecycle directly
- **Part 2** (`--mode engine`, default `full`): `ExecutionEngine.execute_order` / `cancel_order` -- the EXACT call chain RAMP/OMR/MP use in prod. This is the layer where the 2026-04-24 silent-failure regression lived (`ExecutionEngine` calling deprecated `broker.place_order` on `IBKRBroker` which lacks the shim).

Uses clientId=99 by default (the running `homeguard-multi` service holds clientId=10). Limit prices are 50%/200% of market so orders never fill. Every order placed is cancelled before exit and clean state is verified.

**Companion contract test** (auto-runs in pytest, no IBKR connection needed): `tests/trading/brokers/test_broker_contract.py` parametrizes over `(AlpacaBroker, IBKRBroker)` × every method `ExecutionEngine` calls. Catches "broker missing required method" regressions at commit time -- this would have caught the 2026-04-24 bug before deploy.

## Production Strategies (EC2)

| Strategy | Service | Schedule | Description |
|----------|---------|----------|-------------|
| **OMR** | `homeguard-multi` | Entry 3:50 PM, Exit 9:31 AM | Overnight mean reversion on leveraged ETFs (disabled in `strategy_toggle.yaml`) |
| **RAMP** | `homeguard-multi` | Rebalance 3:55 PM | Regime-aware momentum protection on S&P 500 (IBKR paper, enabled) |
| **CSCM** | `homeguard-cscm` | Weekly (Sunday 0:00 UTC) | Cross-sectional crypto momentum with BTC regime filter |

`homeguard-multi` runs `scripts/trading/run_live_paper_trading.py --strategy ramp`, which creates the RAMP adapter and routes it to its broker per `config/trading/broker_routing.yaml` (IBKR paper, port 4002). The standalone `homeguard-omr` / `homeguard-ramp` unit files still exist but are `disabled` and should not be enabled — they have been superseded by `homeguard-multi`.

Note: `--strategy multi` mode exists in the runner but only launches one strategy (priority order: OMR > MP > RAMP) since true concurrent multi-strategy support is not yet implemented. Until that lands, use `--strategy ramp` (or another explicit strategy name) so the unit is explicit about what it runs, independent of `strategy_toggle.yaml` state.

**RAMP Strategy Details** (Deployed 2025-12-08):
- Universe: S&P 500 stocks, dynamic 1/N position sizing
- 5 market regimes (STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR)
- Walk-forward validated: **0.846 Sharpe ratio out-of-sample** (2022-2024)
- Docs: `docs/strategies/production/RAMP_STRATEGY.md` (walk-forward validation history in `docs/archive/strategies/20251212_RAMP_WALK_FORWARD_VALIDATION.md`)

## Agents, Commands & Skills

Slash commands: `/code-review` (pre-commit review), `/feature-dev` (guided implementation)
Strategy skill: `.claude/skills/implement-strategy/`

| Agent | Location | Use for |
|-------|----------|---------|
| Trade Log Analyzer | `.claude/agents/trade-log-analyzer.md` | Diagnose live trading issues |
| Backtest Driver | `.claude/agents/backtest-driver.md` | Autonomous backtest execution |
| Backtest Optimizer | `.claude/agents/backtest-optimizer.md` | Parameter optimization runs |
| Codebase Analyzer | `.claude/agents/codebase-analyzer.md` | Code quality and coverage gaps |
| Strategy Lead | `.claude/agents/strategy-lead.md` | Strategy pipeline orchestration (formerly trading-lead) |
| Live Ops | `.claude/agents/live-ops.md` | Routine EC2 ops (status, metrics, journal, restart) -- state-changing recipes confirm first |
| Code Architect | `.claude/agents/code-architect.md` | Architecture analysis |
| Code Explorer | `.claude/agents/code-explorer.md` | Codebase navigation/discovery |
| Code Reviewer | `.claude/agents/code-reviewer.md` | Pre-commit code review |

**EC2 Management Scripts** (Windows):
- `infra\ec2\local_start_instance.bat` / `local_stop_instance.bat` - Start/stop EC2
- `infra\ec2\check_bot.bat` - Check bot status
- `infra\ec2\view_logs.bat` - Stream live logs
- `infra\ec2\daily_health_check.bat` - 6-point health check

**EC2 Instance Aliases** (SSH):
- `bot-status`, `bot-logs`, `bot-logs-recent`, `bot-update`, `bot-restart`

## Architecture & Infrastructure Documentation

- **ALWAYS** update `docs/architecture/` when adding/removing/moving modules
- **ALWAYS** update `docs/INFRASTRUCTURE_OVERVIEW.md` when modifying AWS resources
- Update `infra/terraform/README.md` when changing Terraform configuration
- Details: [`.claude/documentation.md`](.claude/documentation.md)

## Sensitive Data - Homeguard Specific

| Data Type | Storage Location | Template File |
|-----------|------------------|---------------|
| API Keys (Alpaca, Discord, Anthropic) | `.env` | `.env.example` |
| App Settings | `settings.ini` | `settings.ini.example` |
| EC2 Config (IP, instance ID) | `.env` | `.env.example` |
| IBKR Config (host, port, credentials) | `.env` | `.env.example` |
| Strategy enable/shutdown toggle (runtime state) | `config/trading/strategy_toggle.yaml` | `config/trading/strategy_toggle.example.yaml` |

Shell scripts: `source infra/ec2/load_env.sh`. Batch scripts: `call infra\ec2\load_env.bat`.

## Session Work Logs

After completing a significant implementation session (new features, infra changes, migrations, multi-step fixes), write a timestamped summary to `docs/progress/YYYYMMDD_<TOPIC>.md`.

**When to write:** At the end of a session that produced commits -- not for pure research/discussion sessions.

**Format:**
```markdown
# <Topic> - YYYY-MM-DD

## Summary
1-3 sentence overview of what was accomplished.

## Changes Made
- **File/area**: what changed and why (bulleted, concise)

## Commits
- `<short-hash>` <commit message>

## Known Issues / Remaining Work
- What's left, what broke, what needs follow-up

## Validation
- How the changes were verified (tests run, manual checks, EC2 deploy)
```

**Rules:**
- Keep it factual and concise -- this is a reference doc, not a narrative
- Include commit hashes so future sessions can trace changes
- Always list remaining work so the next session knows where to pick up
- Do not duplicate content already in commit messages -- focus on context and decisions that aren't captured in code

## Common Type Issues

- DataFrame.xs() type hints, VectorBT incomplete stubs, SQL injection prevention
- Details: [`.claude/type_issues.md`](.claude/type_issues.md)

## Memory Efficiency - Backtests

- Load data year-by-year or symbol-by-symbol, not all at once
- Use `StreamingDataLoader` for chunked processing

## When to Consult Detailed Guides

- **Backtesting**: [`.claude/backtesting.md`](.claude/backtesting.md)
- **Code standards**: [`.claude/code_standards.md`](.claude/code_standards.md)
- **Live trading**: [`.claude/live_trading.md`](.claude/live_trading.md)
- **Tests**: [`.claude/testing.md`](.claude/testing.md)
- **Risk**: [`.claude/risk_management.md`](.claude/risk_management.md)
- **Types**: [`.claude/type_issues.md`](.claude/type_issues.md)
- **Project structure**: [`.claude/project_structure.md`](.claude/project_structure.md)
- **Logging**: [`.claude/logging.md`](.claude/logging.md)
- **Documentation**: [`.claude/documentation.md`](.claude/documentation.md)
- **Git**: [`.claude/git_workflow.md`](.claude/git_workflow.md)
- **Screening**: [`src/screening/README.md`](src/screening/README.md)
- **Fundamentals**: [`src/data/yfinance/README.md`](src/data/yfinance/README.md)
