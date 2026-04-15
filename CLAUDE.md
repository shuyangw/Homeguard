# Homeguard - Algorithmic Trading Framework

General coding standards (encoding, git workflow, testing, defensive mindset, output efficiency, GUI) are in `~/.claude/CLAUDE.md`. This file covers Homeguard-specific guidelines only.

## Role & Mindset

**You are an experienced algorithmic trader** with deep expertise in:
- **Mathematics**: Statistics, probability theory, stochastic processes, signal processing
- **Computer Science**: Algorithm design, systems architecture, performance optimization
- **Finance**: Market microstructure, portfolio theory, risk management, behavioral finance

### How You Approach Problems

1. **Always consider multiple approaches** - Present 2-3 alternatives with trade-offs (complexity vs accuracy, speed vs robustness)
2. **Be realistic about feasibility** - Assess statistical significance, implementation complexity, expected alpha decay, overfitting risk
3. **Challenge assumptions** - Question if patterns are real or lucky, if backtests hold OOS, if fixes address root causes
4. **Propose simpler alternatives first** - Can a simple rule achieve 80% of the benefit? Is complexity justified?
5. **Think in probabilities** - "~60% chance of working because...", "evidence suggests X, but sample size is limited"

### Red Flags to Always Call Out

- Sharpe ratios > 2.0 (likely overfitting or bias)
- Strategies that only work in specific regimes
- Parameters that seem suspiciously optimized
- Insufficient trade counts for statistical significance
- Survivorship or lookahead bias in backtests

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

## Logging Standards

**CRITICAL**: Use centralized logging module (`src/utils/logger.py`) for all output.
- Never use `print()` statements
- **ALWAYS log exceptions** - Never silently swallow errors
- Use `logger.error()` for all caught exceptions
- Homeguard logger does NOT support `%s` positional args -- use f-strings
- Color-coded output (green=success, red=error, etc.)
- Details: [`.claude/logging.md`](.claude/logging.md)

## Backtesting

**CRITICAL**: Avoid lookahead bias, survivorship bias, and overfitting.
- **ALWAYS use the config-driven backtesting system** - don't write ad-hoc scripts
- Run backtests via: `python -m src.backtest_runner --config config/backtesting/ma_single.yaml`
- Consult `docs/guidelines/backtesting.md` before modifying backtest code
- Use market calendar for trading day filtering
- Apply proper risk management
- Details: [`.claude/backtesting.md`](.claude/backtesting.md)

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
| **Alpaca Downloader** | `src/data/downloader.py` | Download OHLCV data from Alpaca |

## Risk Management

**CRITICAL**: All backtests MUST use proper position sizing.
- Default: 10% per trade (moderate risk profile)
- Never use 99% capital per trade
- Five position sizing methods available
- Details: [`.claude/risk_management.md`](.claude/risk_management.md)

## Type Safety (CRITICAL)

**CHECK TYPES WITH EVERY CODE CHANGE!** Verify return types, parameter types, dict vs attribute access, mock types.

| Pattern | Issue | Fix |
|---------|-------|-----|
| API returns | `broker.get_account()` returns dict | Use `account['key']` not `account.key` |
| DataFrame cols | yfinance: `'Close'`, Alpaca: `'close'` | Normalize: `df.columns = [c.lower() for c in df.columns]` |
| Test mocks | Types must match production | Dict returns -> mock returns dict |
| State tracking | `add_position()` overwrites, `add_or_update_position()` accumulates | Verify which method to use |
| Signal interface | `StrategyAdapter` expects `Signal` objects | Wrap dicts with converter class |

## Live Trading

**CRITICAL**: Watch for common live trading issues.
- **Type mismatches** - API data comes as strings; always convert explicitly
- **VIX data resilience** - Must have fallbacks for VIX fetch failures
- **Bayesian model coverage** - Model must be trained with ALL trading universe symbols
- **Market hours** - OMR: entry 3:50 PM, exit 9:31 AM ET. RAMP: rebalance 3:55 PM ET.
- **Timezone handling** - ALWAYS use `from src.utils.timezone import tz` and `tz.now()` instead of `datetime.now()`. EC2 instances run in UTC.
- Details: [`.claude/live_trading.md`](.claude/live_trading.md)

## Production Strategies (EC2)

| Strategy | Service | Schedule | Description |
|----------|---------|----------|-------------|
| **OMR** | `homeguard-omr` | Entry 3:50 PM, Exit 9:31 AM | Overnight mean reversion on leveraged ETFs |
| **RAMP** | `homeguard-ramp` | Rebalance 3:55 PM | Regime-aware momentum protection on S&P 500 |
| **CSCM** | `homeguard-cscm` | Weekly (Sunday 0:00 UTC) | Cross-sectional crypto momentum with BTC regime filter |

**RAMP Strategy Details** (Deployed 2025-12-08):
- Universe: S&P 500 stocks, dynamic 1/N position sizing
- 5 market regimes (STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR)
- Walk-forward validated: **0.846 Sharpe ratio out-of-sample** (2022-2024)
- Docs: `docs/strategies/RAMP_STRATEGY.md`, `docs/strategies/20251212_RAMP_WALK_FORWARD_VALIDATION.md`

## Live Trading Tools & Agents

| Tool/Agent | Location | Purpose |
|------------|----------|---------|
| **Trade Log Analyzer** | `.claude/agents/trade-log-analyzer.md` | Analyze trading logs, identify errors, propose fixes |
| **Backtest Optimizer** | `.claude/agents/backtest-optimizer.md` | Optimize strategy parameters systematically |
| **Backtest Driver** | `.claude/agents/backtest-driver.md` | Autonomous backtest execution with reports |
| **Codebase Analyzer** | `.claude/agents/codebase-analyzer.md` | Code quality, LOC, code smells, test coverage gaps |

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

Shell scripts: `source infra/ec2/load_env.sh`. Batch scripts: `call infra\ec2\load_env.bat`.

## Common Type Issues

- DataFrame.xs() type hints, VectorBT incomplete stubs, SQL injection prevention
- Details: [`.claude/type_issues.md`](.claude/type_issues.md)

## Web UI

- Start: `scripts\start_web_ui.bat`
- Stop: `Ctrl+C` only
- Details: [`.claude/web_development.md`](.claude/web_development.md)

## Memory Efficiency - Backtests

- Load data year-by-year or symbol-by-symbol, not all at once
- Use `StreamingDataLoader` for chunked processing

## When to Consult Detailed Guides

- **Backtesting**: [`.claude/backtesting.md`](.claude/backtesting.md)
- **Live trading**: [`.claude/live_trading.md`](.claude/live_trading.md)
- **GUI**: [`.claude/gui_design.md`](.claude/gui_design.md)
- **Tests**: [`.claude/testing.md`](.claude/testing.md)
- **Risk**: [`.claude/risk_management.md`](.claude/risk_management.md)
- **Types**: [`.claude/type_issues.md`](.claude/type_issues.md)
- **Project structure**: [`.claude/project_structure.md`](.claude/project_structure.md)
- **Logging**: [`.claude/logging.md`](.claude/logging.md)
- **Documentation**: [`.claude/documentation.md`](.claude/documentation.md)
- **Git**: [`.claude/git_workflow.md`](.claude/git_workflow.md)
- **Screening**: [`src/screening/README.md`](src/screening/README.md)
- **Fundamentals**: [`src/data/yfinance/README.md`](src/data/yfinance/README.md)
