# Homeguard Backtesting Framework - Architecture Overview

**Version**: 1.7
**Last Updated**: 2026-05-17
**Status**: Current

---

## Executive Summary

Homeguard is a professional-grade backtesting framework for algorithmic trading strategies. Built with Python, it provides a modular, extensible architecture that separates concerns across four main layers: Data, Strategy, Backtesting Engine, and Visualization/Reporting. The runtime production interface is CLI + systemd services -- no GUI or web frontend.

**Key Characteristics**:
- **Modular Design**: Clear separation between components
- **Extensible**: Easy to add strategies, indicators, position sizing methods
- **Scalable**: Support for single-asset and multi-asset portfolios
- **Risk-First**: Built-in position sizing, stop losses, portfolio constraints
- **Production-Ready**: Validated with 50+ accuracy tests

---

## System Architecture

### 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              LAYER 4: VISUALIZATION & REPORTING             │
│  Charts, reports, and performance analytics                 │
│  - QuantStats tearsheets (50+ metrics)                      │
│  - Candlestick charts with trade markers                    │
│  - HTML reports and CSV export                              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Discord Bot (Optional)                  │   │
│  │  - Natural language queries via Claude               │   │
│  │  - Read-only observer for EC2 log inspection         │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│           LAYER 3: BACKTESTING ENGINE (CORE)                │
│  Orchestrates simulation, portfolio management, risk        │
│  - BacktestEngine (main orchestrator)                       │
│  - PortfolioSimulator (custom simulator)                    │
│  - RiskManager, PositionSizer, Metrics                      │
│  - SweepRunner (parallel multi-symbol execution)            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│           LAYER 2: STRATEGY IMPLEMENTATION                  │
│  Trading logic and signal generation                        │
│  - Base strategies (MA, Momentum, Mean Reversion)           │
│  - Advanced strategies (Vol-Targeted, Pairs Trading)        │
│  - BaseStrategy abstract class                              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│           LAYER 1: DATA INGESTION & STORAGE                 │
│  Data fetching, storage, and retrieval                      │
│  - Alpaca API client (REST + WebSocket streaming)           │
│  - LiveDataProvider (real-time streaming)                   │
│  - Parquet storage (partitioned by symbol/date)             │
│  - DuckDB query engine                                      │
│  - News/Sentiment pipeline (Alpaca News + FinBERT)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Overview

### Layer 1: Data Layer

**Purpose**: Fetch, store, and retrieve historical market data

**Key Components**:
- **DataAcquisitionManager** ([src/data/acquisition/manager.py](../../src/data/acquisition/manager.py))
  - Unified entry point for all data downloads
  - Plugin-based architecture (equities, crypto, futures, news)
  - Multi-threaded with retry logic and manifest tracking

- **BaseDownloader** ([src/data/acquisition/base.py](../../src/data/acquisition/base.py))
  - Shared infrastructure for all download plugins
  - Hive-partitioned parquet storage
  - Schema validation and thread-safe client management

- **DataLoader** ([src/backtesting/optimization/data_loader.py](../../src/backtesting/optimization/data_loader.py))
  - Loads data from Parquet via DuckDB
  - Market day filtering (weekends/holidays)
  - SQL-based queries for speed
  - For Polars/LRU-cached streaming loads, see `StreamingDataLoader` in `src/backtesting/engine/streaming_data_loader.py`

**Data Format**: Parquet files under `equities_1min/{symbol}/{date}.parquet`

**Dependencies**: Alpaca API, Pandas, PyArrow, DuckDB

#### Real-Time Streaming (src/streaming/)

- **LiveDataProvider** ([src/streaming/live_data_provider.py](../../src/streaming/live_data_provider.py))
  - WebSocket-based real-time market data
  - 32x faster than polling (167s -> 5s for RAMP)
  - Public API: `get_price()`, `get_bars()`, `get_quote()`, `get_vwap()`

- **StreamManager** - WebSocket connection management
- **BarBuffer** - In-memory cache (500 bars per symbol)
- **FallbackPoller** - REST API backup when streaming unavailable

**Feature Flag**: `USE_STREAMING=true` environment variable enables streaming

#### News & Sentiment (src/data/news/)

- **NewsDownloader** ([src/data/news/news_downloader.py](../../src/data/news/news_downloader.py))
  - Fetches news from Alpaca News API
  - Thread-safe parallel downloads
  - Parquet storage: `news/symbol={SYMBOL}/year={YYYY}/news.parquet`

- **SentimentAnalyzer** ([src/data/news/sentiment_analyzer.py](../../src/data/news/sentiment_analyzer.py))
  - FinBERT-based sentiment scoring
  - Score range: -1 (negative) to +1 (positive)

- **SentimentCache** - Caches computed sentiment scores

#### Stock Screening (src/screening/)

- **StockScreener** ([src/screening/screener.py](../../src/screening/screener.py))
  - Finviz-like stock screening using Alpaca APIs
  - Filters: price, volume, technical indicators, fundamentals
  - Fetches all ~10,000 tradable US equities by default
  - IEX feed (paper) or SIP feed (live)

- **ScreenerConfig** ([src/screening/filters.py](../../src/screening/filters.py))
  - Pydantic models: PriceFilter, VolumeFilter, TechnicalFilter, FundamentalFilter
  - Type-safe configuration with validation

- **ScreenerCache** ([src/screening/cache.py](../../src/screening/cache.py))
  - In-memory TTL cache (60s snapshots, 1hr historical)
  - Thread-safe multi-tier caching

- **TechnicalIndicators** ([src/screening/indicators.py](../../src/screening/indicators.py))
  - RSI, SMA, EMA, MACD, Bollinger Bands, ATR

**Usage**: Strategies call screener to dynamically select trading universe

#### YFinance Fundamentals (src/data/yfinance/)

- **YFinanceFundamentalsProvider** ([src/data/yfinance/provider.py](../../src/data/yfinance/provider.py))
  - Fundamental data not available from Alpaca
  - Market cap, P/E, PEG, ROE, dividends, sector, beta
  - 40+ fundamental fields

- **FundamentalsCache** ([src/data/yfinance/cache.py](../../src/data/yfinance/cache.py))
  - Persistent parquet cache (24-hour TTL)
  - Survives restarts

**Integration**: Pass provider to StockScreener for fundamental filters

#### Data Validation Framework (src/data/validation/)

Multi-domain layered validation framework for the local data store. Currently
implemented for futures; equities/crypto/fx/options have placeholder packages
ready to be filled in. Each domain registers checks across four layers:

- **Layer 1 — Structural**: section presence, partition key validity, file
  count per symbol, schema match, parquet integrity sampling
- **Layer 2 — Statistical**: per-symbol bar density, OHLCV invariants, date
  floor, freshness, gap detection, volume sanity, derived-signal sanity (SOFR,
  Treasury yields)
- **Layer 3 — Cross-source**: definitions completeness vs per-contract,
  derived SOFR vs 2Y direct read, yield curve smoothness
- **Layer 4 — External**: known events captured (e.g. CL April 2020 negative
  oil), ZN-vs-10Y correlation, optional yfinance and CME settlement
  cross-checks (opt-in via `--external-yfinance` / `--external-cme`)

**Core primitives** ([src/data/validation/core/](../../src/data/validation/core/)):
- `BaseCheck` ABC with class-level auto-registration via `__init_subclass__`
- `ValidationRunner` — domain/layer/name filtering, per-check exception
  containment, continue-on-CRITICAL aggregation
- `MarkdownReporter` — YAML frontmatter + machine-parseable body for
  regression diff against prior runs
- `ValidationResult` / `RunReport` (frozen dataclasses)

**Adaptation F gating** ([src/data/validation/futures/checks/adaptation_f.py](../../src/data/validation/futures/checks/adaptation_f.py))
is lazy-loaded — checks register only when the CLI is invoked with
`--adaptation-f`, so they don't pollute the default registry.

**CLI**: `python scripts/data/run_validation.py --domain futures --mode initial`

#### Derivation Pipelines (src/data/derivations/)

Reusable derivations that turn primary market data into engineered signals
the strategies and validation checks consume.

- **SOFR derivation** ([src/data/derivations/futures/sofr.py](../../src/data/derivations/futures/sofr.py))
  - `derive_sofr(date)` returns implied overnight SOFR from `SR1` front-month
    futures close (`100 - close`); listing 2018-05-07
- **Treasury yields** ([src/data/derivations/futures/yields.py](../../src/data/derivations/futures/yields.py))
  - `get_treasury_yield(tenor, date)` reads the on-the-run yield directly
    from CME Micro Yield futures (2YY, 5YY, 10Y, 30Y); listing 2022-08-15

ES realized volatility, VIX-equivalent (Cboe whitepaper), and per-asset-class
carry computation are deferred — see
[20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md](../progress/20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md)
for the rationale and re-entry plan.

---

### Layer 2: Strategy Layer

**Purpose**: Implement trading logic and generate entry/exit signals

**Key Components**:
- **Strategy** ([src/backtesting/base/strategy.py](../../src/backtesting/base/strategy.py))
  - Abstract base class for all strategies
  - Defines interface: `generate_signals(data) -> (entries, exits)`

- **MultiSymbolStrategy** ([src/backtesting/base/strategy.py](../../src/backtesting/base/strategy.py))
  - Base for strategies that trade multiple symbols simultaneously
  - Interface: `generate_signals_multi(data_dict) -> signals_dict`

- **PairsStrategy** ([src/backtesting/base/pairs_strategy.py](../../src/backtesting/base/pairs_strategy.py))
  - Base for pairs trading strategies (market-neutral)
  - Inherits from `MultiSymbolStrategy`
  - Enforces synchronized execution of both legs
  - Automatically routes to `PairsPortfolio`

**Strategy Categories**:

1. **Production Strategies** ([src/strategies/advanced/](../../src/strategies/advanced/))
   - `OvernightMeanReversion` (OMR): Overnight gap trading with Bayesian model - **LIVE**
   - `MomentumProtectionStrategy` (MP): Cross-sectional momentum with VIX protection
   - `RAMPStrategy` (RAMP): Regime-Aware Momentum Protection - **LIVE** (deployed Dec 2025)
   - `HVORBStrategy` (HV ORB): High Volatility Opening Range Breakout - **RESEARCH**
   - `ICTStrategy` (ICT): Smart Money Concepts / Institutional Chart Techniques - **RESEARCH**
   - Supporting modules: `bayesian_reversion_model.py`, `market_regime_detector.py`, `orb_numba_core.py`

2. **Research Strategies** ([src/strategies/research/](../../src/strategies/research/))
   - `MovingAverageCrossover`: Fast MA > Slow MA
   - `TripleMovingAverage`: Three-level MA crossover
   - `MomentumStrategy`: Trend-following momentum
   - `BreakoutStrategy`: Price breakout trading
   - `MeanReversion`: Bollinger Band reversion
   - `RSIMeanReversion`: RSI oversold/overbought
   - `VolatilityTargetedMomentum`: Vol-scaled momentum
   - `CrossSectionalMomentum`: Multi-asset momentum ranking
   - `PairsTrading`: Statistical arbitrage (cointegration-based)

3. **Base Strategies** ([src/strategies/base_strategies/](../../src/strategies/base_strategies/))
   - **Deprecated**: Re-exports from `research/` for backward compatibility
   - New code should import from `src.strategies.research.*`

**Strategy Flow**:
```python
# 1. Strategy receives OHLCV data
data = pd.DataFrame(OHLCV)

# 2. Strategy generates signals
entries, exits = strategy.generate_signals(data)

# 3. Engine executes signals
portfolio = engine.run_with_data(strategy, data)
```

**Dependencies**: Pandas, NumPy, Technical Indicators

---

### Layer 3: Backtesting Engine Layer

**Purpose**: Execute simulations, manage risk, calculate performance

**Core Engine** ([src/backtesting/engine/](../../src/backtesting/engine/)):

- **BacktestEngine** ([backtest_engine.py](../../src/backtesting/engine/backtest_engine.py))
  - **Primary orchestrator** for all backtests
  - Routes to single-symbol or multi-asset mode
  - Loads data, executes strategy, returns portfolio
  - Validates data integrity (handles duplicates, NaN, etc.)

- **PortfolioSimulator** ([portfolio_simulator.py](../../src/backtesting/engine/portfolio_simulator.py))
  - **Custom simulator** (replaces VectorBT dependency)
  - Bar-by-bar portfolio simulation
  - Risk management integration
  - Trade logging and metrics calculation
  - Returns `Portfolio` object with equity curve, trades, stats

- **SweepRunner** ([sweep_runner.py](../../src/backtesting/optimization/sweep_runner.py))
  - Runs strategy across multiple symbols **in parallel**
  - ThreadPoolExecutor for concurrent execution
  - Callbacks for progress tracking
  - Result aggregation

- **MultiAssetPortfolio** ([multi_asset_portfolio.py](../../src/backtesting/engine/multi_asset_portfolio.py))
  - Handles **simultaneous positions** across multiple symbols
  - Portfolio weighting schemes (Equal Weight, Risk Parity, etc.)
  - Rebalancing logic
  - Portfolio-level metrics

- **PairsPortfolio** ([pairs_portfolio.py](../../src/backtesting/engine/pairs_portfolio.py))
  - **Synchronized execution** for pairs trading strategies
  - Both legs trade simultaneously (market-neutral)
  - Position sizing via `PairsPositionSizer` classes
  - Automatic routing from `BacktestEngine` when `PairsStrategy` detected
  - Trade logging with pair-specific attributes

**Optimization** ([src/backtesting/engine/](../../src/backtesting/engine/) | [Detailed Docs](../planning/OPTIMIZATION_MODULE.md)):

- **BacktestEngine.optimize()** ([backtest_engine.py](../../src/backtesting/engine/backtest_engine.py))
  - Grid search parameter optimization
  - Tests all parameter combinations via `itertools.product()`
  - Supports Sharpe Ratio, Total Return, Max Drawdown metrics
  - Returns best parameters, value, and portfolio

- **SweepRunner.optimize_across_universe()** ([sweep_runner.py](../../src/backtesting/optimization/sweep_runner.py))
  - Universe-wide parameter optimization
  - Finds parameters optimal across multiple symbols
  - Aggregation metrics: median/mean Sharpe, returns, win rate
  - Parallel execution support

**Risk Management** ([src/backtesting/utils/](../../src/backtesting/utils/)):

- **RiskManager** ([risk_manager.py](../../src/backtesting/utils/risk_manager.py))
  - Tracks open positions
  - Enforces stop losses
  - Manages portfolio constraints

- **PositionSizer** ([position_sizer.py](../../src/backtesting/utils/position_sizer.py))
  - **5 Position Sizing Methods**:
    1. Fixed Percentage (e.g., 10% per trade)
    2. Fixed Dollar (e.g., $10,000 per trade)
    3. Volatility-Based (ATR-scaled)
    4. Kelly Criterion (optimal sizing)
    5. Risk Parity (equal risk contribution)

- **PairsPositionSizer** ([pairs_position_sizer.py](../../src/backtesting/utils/pairs_position_sizer.py))
  - **Position sizing for pairs trading** (both legs simultaneously)
  - **3 Sizing Strategies**:
    1. **DollarNeutral**: Equal dollar allocation (50/50 split)
    2. **VolatilityAdjusted**: Inverse volatility weighting
    3. **RiskParity**: Equal risk contribution (correlation-aware)
  - Factory function: `create_pairs_sizer(method, **kwargs)`
  - Returns `(shares1, shares2)` tuple

- **RiskConfig** ([risk_config.py](../../src/backtesting/utils/risk_config.py))
  - Configuration dataclass
  - Preset profiles: `conservative()`, `moderate()`, `aggressive()`, `disabled()`

**Utilities**:

- **Indicators** ([indicators.py](../../src/backtesting/utils/indicators.py))
  - 15+ technical indicators: SMA, EMA, RSI, ATR, MACD, Bollinger Bands, etc.

- **MarketCalendar** ([market_calendar.py](../../src/backtesting/utils/market_calendar.py))
  - NYSE trading calendar
  - Filters weekends and holidays

- **Metrics** ([metrics.py](../../src/backtesting/engine/metrics.py))
  - Performance metrics: Sharpe, Sortino, Calmar, Max Drawdown, etc.

**Dependencies**: Pandas, NumPy, QuantStats

---

### Layer 4: Visualization & Reporting Layer

**Purpose**: Generate charts, reports, and performance analytics

**Key Components** ([src/visualization/](../../src/visualization/)):

- **BacktestVisualizer** ([integration.py](../../src/visualization/integration.py))
  - Unified interface to visualization pipeline
  - Connects backtest results to charts and reports

- **QuantStatsReporter** ([reports/quantstats_reporter.py](../../src/visualization/reports/quantstats_reporter.py))
  - **QuantStats integration** for professional tearsheets
  - 50+ metrics: Returns, Sharpe, Sortino, Drawdown, etc.
  - Benchmark comparison (S&P 500, custom)
  - Monthly/yearly returns heatmaps
  - Rolling metrics charts

- **Charts** ([charts/](../../src/visualization/charts/))
  - **Candlestick** ([candlestick.py](../../src/visualization/charts/candlestick.py)): Interactive price charts
  - **mplfinance** ([mplfinance_chart.py](../../src/visualization/charts/mplfinance_chart.py)): Technical chart generation

- **ReportGenerator** ([reports/report_generator.py](../../src/visualization/reports/report_generator.py))
  - Summary reports (text, HTML, CSV)
  - Trade-by-trade logs
  - Performance summaries

- **OutputManager** ([utils/output_manager.py](../../src/visualization/utils/output_manager.py))
  - Manages output directory structure
  - File naming conventions

**Dependencies**: QuantStats, Matplotlib, mplfinance, Plotly

---

### Discord Bot (Optional Addon)

**Purpose**: Read-only observability for live trading via natural language queries

**Design Principles**:
- **Fully Isolated**: No imports from trading/backtesting modules
- **Read-Only**: Cannot modify files or control services
- **Optional**: Trading system operates independently; bot failure has zero impact

**Key Components** ([src/discord_bot/](../../src/discord_bot/)):

- **TradingInvestigator** ([investigator.py](../../src/discord_bot/investigator.py))
  - Claude-powered ReAct agent
  - Multi-step investigation via shell commands
  - Homeguard-specific system prompt

- **CommandExecutor** ([executor.py](../../src/discord_bot/executor.py))
  - Async subprocess execution
  - Read-only command whitelist
  - Timeout and output truncation

- **Security** ([security.py](../../src/discord_bot/security.py))
  - 50+ allowed read-only commands
  - 30+ blocked dangerous patterns
  - Output sanitization (masks secrets)

- **Discord Bot** ([main.py](../../src/discord_bot/main.py))
  - Commands: `!ask`, `!status`, `!signals`, `!trades`, `!logs`, `!errors`
  - Channel-restricted access
  - Deferred response pattern

**Deployment**: Separate systemd service (`homeguard-discord.service`)

**Dependencies**: discord.py, anthropic

---

## Data Flow

### CLI Execution Flow

```
User Command:
  python -m src.backtest_runner \
    --strategy MovingAverageCrossover \
    --symbols AAPL --start 2023-01-01 --end 2024-01-01

    v

backtest_runner.py (CLI entry point)
    v
BacktestEngine.run()
    ├─-> DataLoader.load_data() -> DuckDB query -> Parquet files
    ├─-> MarketCalendar.filter_market_days()
    ├─-> Strategy.generate_signals(data) -> (entries, exits)
    ├─-> PortfolioSimulator.simulate()
    │     ├─ For each bar:
    │     │   ├─ Check entry/exit signals
    │     │   ├─ PositionSizer.calculate_shares()
    │     │   ├─ RiskManager.check_constraints()
    │     │   ├─ Execute trade
    │     │   ├─ Update equity curve
    │     │   └─ TradeLogger.log_trade()
    │     └─ Return Portfolio object
    └─-> Calculate metrics
        ├─ Metrics.calculate_performance()
        └─ Return stats dict

    v

BacktestVisualizer.generate()
    ├─-> QuantStatsReporter.create_tearsheet()
    ├─-> Charts.generate_candlestick()
    └─-> ReportGenerator.create_summary()

    v

Output:
  - Tearsheet HTML/PDF
  - Trade log CSV
  - Candlestick charts PNG
  - Performance summary TXT
```

### Multi-Symbol Sweep Flow

```
SweepRunner.run_sweep(symbols=['AAPL', 'MSFT', 'GOOGL'])
    v
ThreadPoolExecutor (parallel execution)
    ├─ Worker 1: BacktestEngine.run('AAPL')
    ├─ Worker 2: BacktestEngine.run('MSFT')
    └─ Worker 3: BacktestEngine.run('GOOGL')
        │
        ├─-> Callback: on_symbol_start('AAPL')
        ├─-> Callback: on_symbol_complete('AAPL', portfolio)
        └─-> Callback: on_symbol_error('AAPL', error)

    v (all symbols complete)

ResultsAggregator.aggregate()
    ├─ Combine equity curves
    ├─ Aggregate metrics
    └─ Create comparison charts

    v

Return: List[Portfolio] + Aggregate Reports
```

---

## Technology Stack

### Core Dependencies

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Primary language | 3.13+ |
| **Pandas** | Data manipulation | Latest |
| **NumPy** | Numerical computation | Latest |
| **DuckDB** | Fast SQL queries on Parquet | Latest |
| **PyArrow** | Parquet file I/O | Latest |

### Backtesting & Analysis

| Technology | Purpose |
|------------|---------|
| **QuantStats** | Performance metrics and tearsheets |
| **VectorBT** | (Legacy, being phased out) |

### Visualization

| Technology | Purpose |
|------------|---------|
| **Matplotlib** | Charts and plots |
| **mplfinance** | Candlestick charts |
| **Plotly** | Interactive charts |

### Data Sources

| Technology | Purpose |
|------------|---------|
| **Alpaca API** | Market data provider |

### Utilities

| Technology | Purpose |
|------------|---------|
| **Rich** | Color-coded console logging |

---

## Configuration Management

### Global Configuration

**File**: `settings.ini`

**Contents**:
- OS-specific paths (Windows/macOS/Linux)
- Data storage directory
- Log output directory
- Tearsheet frequency (full, hourly, daily, weekly)

### Risk Configuration

**Class**: `RiskConfig` ([src/backtesting/utils/risk_config.py](../../src/backtesting/utils/risk_config.py))

**Preset Profiles**:
```python
RiskConfig.conservative()  # 5% per trade, 60% cash reserve
RiskConfig.moderate()      # 10% per trade, balanced
RiskConfig.aggressive()    # 20% per trade, high deployment
RiskConfig.disabled()      # 99% per trade (testing only)
```

### Visualization Configuration

**Class**: `VisualizationConfig` ([src/visualization/config.py](../../src/visualization/config.py))

**Options**:
- Log levels (minimal, info, debug)
- Enable/disable charts, logs, reports
- Output formatting

---

## Entry Points

### 1. CLI Backtest Runner

**File**: `src/backtest_runner.py`

**Usage**:
```bash
python -m src.backtest_runner \
  --strategy MovingAverageCrossover \
  --symbols AAPL MSFT \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --initial-capital 100000 \
  --fees 0.001
```

### 2. Data Ingestion

**Module**: `src/data/acquisition/`

**Usage**:
```bash
python -m src.data.acquisition --source equities --symbols AAPL,MSFT --start 2020-01-01
```

---

## Design Principles

### 1. Separation of Concerns
- Each layer has a single, well-defined responsibility
- Layers communicate through clearly defined interfaces
- No direct cross-layer dependencies (except via public APIs)

### 2. Extensibility
- Easy to add new strategies (inherit from `BaseStrategy`)
- Easy to add new indicators (add to `indicators.py`)
- Easy to add new position sizing methods (add to `PositionSizer`)

### 3. Risk-First Design
- Risk management is **built into** the engine, not optional
- Position sizing enforced by default
- Stop losses, max positions, capital constraints

### 4. Testability
- **50+ accuracy tests** validate engine correctness
- Synthetic data tests prove mathematical accuracy
- Lookahead bias prevention tests
- Data integrity tests

### 5. Performance
- DuckDB for fast Parquet queries
- Parallel symbol sweeping via ThreadPoolExecutor
- Efficient data structures (Pandas DataFrames)
- Caching for expensive operations

### 6. User Experience
- CLI interface backed by config-driven YAML
- Color-coded logging (Rich library)
- Progress tracking for long operations
- Professional reports (QuantStats tearsheets)

---

## Scalability

### Current Support
- **Single Symbol**: Fully supported
- **Multiple Symbols (Sweep)**: Parallel execution, 1-8 workers
- **Multi-Asset Portfolio**: Simultaneous positions across symbols
- **Timeframes**: 1-minute bars (primary), extensible to other frequencies

### Performance Benchmarks
- **Data Loading**: DuckDB -> ~1-2 seconds for 1 year of 1-minute data
- **Backtest Execution**: ~2-5 seconds per symbol per year
- **Parallel Sweep**: 3-8x speedup with 4-8 workers
- **Test Suite**: 50 tests in <5 seconds

---

## Security & Data Privacy

### API Keys
- Stored securely in `src/api_key.py`
- **Never committed to version control** (in `.gitignore`)

### Data Storage
- Local-only storage (no cloud uploads by default)
- Parquet files stored in configured directory
- No sensitive data logged

---

## Future Extensibility

### Easy to Add
- [+] New strategies (inherit `BaseStrategy`)
- [+] New indicators (add to `indicators.py`)
- [+] New position sizing methods (add to `PositionSizer`)
- [+] New data sources (implement API client interface)
- [+] New risk constraints (add to `RiskManager`)
- [+] New brokers (implement focused interfaces - ISP-compliant design)

### Planned Enhancements
-  Options trading support (interface ready: `OptionsTradingInterface`)
-  Futures trading support
-  Intraday rebalancing
-  Machine learning strategy integration
-  Additional broker integrations (TastyTrade, IBKR)

### Recently Deployed
- [+] **Config-driven backtesting** - YAML-based backtest configuration (November 2025)
  - Single command: `python -m src.backtest_runner --config path/to/config.yaml`
  - Supports all modes: single, sweep, optimize, walk-forward
  - Pydantic-validated configuration schema with inheritance (`extends:` directive)
  - Predefined date presets and symbol universes
  - Strategy registry with lazy loading (no import chain issues)
  - See [config/backtesting/](../../config/backtesting/) for example configs

- [+] **Broker interface refactoring** - ISP-compliant interface design (November 2025)
  - 6 focused interfaces: AccountInterface, MarketHoursInterface, MarketDataInterface, OrderManagementInterface, StockTradingInterface, OptionsTradingInterface
  - BrokerInterface is now a composite interface (backward compatible)
  - Backward-compatible method aliases preserve existing code
  - Ready for multi-broker support (Alpaca, TastyTrade, IBKR)
  - 39 new interface compliance tests
  - See [MODULE_REFERENCE.md](MODULE_REFERENCE.md#trading-system-layer) for details

- [+] **Live trading integration** - Paper trading deployed to AWS EC2 with automated scheduling (November 2025)
  - EC2 instance with Python 3.11 (t4g.medium ARM64, 4 GB RAM, 50 GB gp3 EBS)
  - Lambda-powered auto-start/stop (9 AM - 4:30 PM ET Mon-Fri; weekend up for CSCM Sun 00:00 UTC tick)
  - **Multi-strategy architecture** (current, May 2026):
    - `homeguard-multi.service`: runs `scripts/trading/run_live_paper_trading.py --strategy ramp`. Routes RAMP through **IBKR paper (port 4002)** per `config/trading/broker_routing.yaml`. OMR is disabled via `strategy_toggle.yaml`. Metrics exposed on a single port.
    - `homeguard-cscm.service`: Cross-Sectional Crypto Momentum (DemoBroker + Binance WS streaming, metrics port 8084), scheduled weekly (Sunday 0:00 UTC)
    - Legacy per-strategy units (`homeguard-omr.service`, `homeguard-ramp.service`) remain on disk as `disabled` and are superseded by `homeguard-multi`
  - SSH management scripts (Windows .bat and Unix .sh) with `.env`-based configuration
  - See [Infrastructure Overview](../INFRASTRUCTURE_OVERVIEW.md) for details

- [+] **Self-hosted monitoring stack** - VictoriaMetrics + Grafana + Loki on the trading host (April 2026)
  - `victoriametrics.service` (port 8428, 90d retention): scrapes each strategy's in-process metrics exporter every 15s
  - `grafana-server.service` (port 3000): auto-provisioned dashboards (Portfolio Overview, Strategy Breakdown, Incident Review)
  - `loki.service` (port 3100, 14d retention) + `promtail.service`: ships systemd journal to Loki for correlated log views
  - `node_exporter.service` (port 9100): host CPU/memory/disk/network metrics
  - Tailscale VPN for operator access (no public exposure of Grafana/VM/Loki)
  - `homeguard-weekly-report.timer` (Sun 00:30 UTC): QuantStats tearsheet rendered from VictoriaMetrics equity series
  - Metric naming contract in [docs/monitoring/METRIC_SPEC.md](../monitoring/METRIC_SPEC.md)
  - Design spec: [docs/superpowers/specs/2026-04-18-monitoring-system-design.md](../superpowers/specs/2026-04-18-monitoring-system-design.md)
  - Installed via `infra/ec2/setup/install_{victoriametrics,grafana,loki,node_exporter,tailscale}.sh` (not managed by Terraform)

- [+] **Strategy reorganization** - Separated production vs research strategies (December 2025)
  - Production strategies in `src/strategies/advanced/`: OMR, MP, RAMP
  - Research strategies moved to `src/strategies/research/`
  - Backward compatibility via re-exports in `base_strategies/`

- [+] **Real-time streaming platform** - WebSocket-based market data (December 2025)
  - `LiveDataProvider` for real-time bar/quote data
  - 32x performance improvement over polling
  - Smart fallback to REST API when streaming unavailable
  - Feature flag: `USE_STREAMING=true`
  - See [20251209_STREAMING_DATA_PLATFORM.md](20251209_STREAMING_DATA_PLATFORM.md)

- [+] **News & Sentiment pipeline** - Market sentiment analysis (December 2025)
  - Alpaca News API integration
  - FinBERT-based sentiment scoring
  - Premarket sentiment filters for strategies

- [+] **New research strategies** - HV ORB and ICT (December 2025)
  - `HVORBStrategy`: High Volatility Opening Range Breakout
  - `ICTStrategy`: Smart Money Concepts / Institutional Chart Techniques
  - Numba JIT compilation for ORB signal generation

---

## References

- **Module Reference**: [MODULE_REFERENCE.md](MODULE_REFERENCE.md)
- **Data Flow**: [DATA_FLOW.md](DATA_FLOW.md)
- **Optimization Module**: [OPTIMIZATION_MODULE.md](../planning/OPTIMIZATION_MODULE.md)
- **Backtesting Guide**: [../guides/BACKTESTING_GUIDE.md](../guides/BACKTESTING_GUIDE.md)
- **Testing Guide**: [../testing/TEST_SUITE_QUICK_START.md](../testing/TEST_SUITE_QUICK_START.md)

---

**Last Updated**: 2026-05-17
**Maintainers**: Update this doc when adding/removing/moving major modules
**Review Frequency**: After any architectural changes
