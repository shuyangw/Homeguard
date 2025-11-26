# Homeguard Backtesting Framework - Architecture Overview

**Version**: 1.2
**Last Updated**: 2025-11-25
**Status**: Current

---

## Executive Summary

Homeguard is a professional-grade backtesting framework for algorithmic trading strategies. Built with Python, it provides a modular, extensible architecture that separates concerns across five main layers: Data, Engine, Strategy, Visualization, and GUI.

**Key Characteristics**:
- **Modular Design**: Clear separation between components
- **Extensible**: Easy to add strategies, indicators, position sizing methods
- **Scalable**: Support for single-asset and multi-asset portfolios
- **Risk-First**: Built-in position sizing, stop losses, portfolio constraints
- **Production-Ready**: Validated with 50+ accuracy tests

---

## System Architecture

### 5-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 5: GUI (PRESENTATION)              │
│  Flet-based graphical interface for non-technical users     │
│  - Setup View, Run View, Results View                       │
│  - Thread-safe worker communication via queues              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              LAYER 4: VISUALIZATION & REPORTING             │
│  Charts, reports, and performance analytics                 │
│  - QuantStats tearsheets (50+ metrics)                      │
│  - Candlestick charts with trade markers                    │
│  - HTML reports and CSV export                              │
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
│  - Alpaca API client                                        │
│  - Parquet storage (partitioned by symbol/date)             │
│  - DuckDB query engine                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Overview

### Layer 1: Data Layer

**Purpose**: Fetch, store, and retrieve historical market data

**Key Components**:
- **AlpacaClient** ([src/data_engine/api/alpaca_client.py](../../src/data_engine/api/alpaca_client.py))
  - Connects to Alpaca REST API
  - Fetches OHLCV bars (open, high, low, close, volume)
  - Rate limiting and error handling

- **ParquetStorage** ([src/data_engine/storage/parquet_storage.py](../../src/data_engine/storage/parquet_storage.py))
  - Stores data in Apache Parquet format
  - Partitioned by timeframe and symbol
  - 100x compression vs CSV

- **IngestionPipeline** ([src/data_engine/orchestration/ingestion_pipeline.py](../../src/data_engine/orchestration/ingestion_pipeline.py))
  - Multi-threaded data ingestion
  - Progress tracking
  - Batch processing for efficiency

- **DataLoader** ([src/backtesting/engine/data_loader.py](../../src/backtesting/engine/data_loader.py))
  - Loads data from Parquet via DuckDB
  - Market day filtering (weekends/holidays)
  - SQL-based queries for speed

**Data Format**: Parquet files under `equities_1min/{symbol}/{date}.parquet`

**Dependencies**: Alpaca API, Pandas, PyArrow, DuckDB

---

### Layer 2: Strategy Layer

**Purpose**: Implement trading logic and generate entry/exit signals

**Key Components**:
- **Strategy** ([src/backtesting/base/strategy.py](../../src/backtesting/base/strategy.py))
  - Abstract base class for all strategies
  - Defines interface: `generate_signals(data) → (entries, exits)`

- **MultiSymbolStrategy** ([src/backtesting/base/strategy.py](../../src/backtesting/base/strategy.py))
  - Base for strategies that trade multiple symbols simultaneously
  - Interface: `generate_signals_multi(data_dict) → signals_dict`

- **PairsStrategy** ([src/backtesting/base/pairs_strategy.py](../../src/backtesting/base/pairs_strategy.py))
  - Base for pairs trading strategies (market-neutral)
  - Inherits from `MultiSymbolStrategy`
  - Enforces synchronized execution of both legs
  - Automatically routes to `PairsPortfolio`

**Strategy Categories**:

1. **Base Strategies** ([src/strategies/base_strategies/](../../src/strategies/base_strategies/))
   - `MovingAverageCrossover`: Fast MA > Slow MA
   - `TripleMovingAverage`: Three-level MA crossover
   - `MomentumStrategy`: Trend-following momentum
   - `BreakoutStrategy`: Price breakout trading
   - `MeanReversion`: Bollinger Band reversion
   - `RSIMeanReversion`: RSI oversold/overbought

2. **Advanced Strategies** ([src/strategies/advanced/](../../src/strategies/advanced/))
   - `VolatilityTargetedMomentum`: Vol-scaled momentum
   - `OvernightMeanReversion`: Overnight gap trading
   - `CrossSectionalMomentum`: Multi-asset momentum ranking
   - `PairsTrading`: Statistical arbitrage (cointegration-based)

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

- **SweepRunner** ([sweep_runner.py](../../src/backtesting/engine/sweep_runner.py))
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

**Optimization** ([src/backtesting/engine/](../../src/backtesting/engine/) | [Detailed Docs](OPTIMIZATION_MODULE.md)):

- **BacktestEngine.optimize()** ([backtest_engine.py](../../src/backtesting/engine/backtest_engine.py:408))
  - Grid search parameter optimization
  - Tests all parameter combinations via `itertools.product()`
  - Supports Sharpe Ratio, Total Return, Max Drawdown metrics
  - Returns best parameters, value, and portfolio

- **SweepRunner.optimize_across_universe()** ([sweep_runner.py](../../src/backtesting/engine/sweep_runner.py:382))
  - Universe-wide parameter optimization
  - Finds parameters optimal across multiple symbols
  - Aggregation metrics: median/mean Sharpe, returns, win rate
  - Parallel execution support

- **OptimizationDialog** ([src/gui/views/optimization_dialog.py](../../src/gui/views/optimization_dialog.py))
  - GUI parameter grid specification
  - Supports int, float, bool, string parameters
  - Combination estimation and preview
  - CSV export of all results

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

### Layer 5: GUI Layer

**Purpose**: Provide graphical interface for non-technical users

**Key Components** ([src/gui/](../../src/gui/)):

- **BacktestApp** ([app.py](../../src/gui/app.py))
  - Main Flet application
  - View navigation and state management
  - Dark theme with bright text

**Views** ([gui/views/](../../src/gui/views/)):
- **SetupView** ([setup_view.py](../../src/gui/views/setup_view.py))
  - Strategy selection
  - Parameter configuration
  - Date range picker
  - Symbol selection

- **RunView** ([run_view.py](../../src/gui/views/run_view.py))
  - Real-time progress monitoring
  - Symbol-by-symbol updates
  - Worker thread status

- **ResultsView** ([results_view.py](../../src/gui/views/results_view.py))
  - Metrics table display
  - Performance summary
  - Link to reports and charts

- **ExecutionView** ([execution_view.py](../../src/gui/views/execution_view.py))
  - Advanced execution monitoring

**Workers** ([gui/workers/](../../src/gui/workers/)):
- **GUIBacktestController** ([gui_controller.py](../../src/gui/workers/gui_controller.py))
  - **Thread-safe wrapper** around SweepRunner
  - Queue-based communication between worker and UI
  - Prevents GUI freezing during long backtests

**Utilities** ([gui/utils/](../../src/gui/utils/)):
- Config management, symbol downloader, data inspector, trade inspector, run history

**Dependencies**: Flet, Threading, Queue

---

## Data Flow

### CLI Execution Flow

```
User Command:
  python -m src.backtest_runner \
    --strategy MovingAverageCrossover \
    --symbols AAPL --start 2023-01-01 --end 2024-01-01

    ↓

backtest_runner.py (CLI entry point)
    ↓
BacktestEngine.run()
    ├─→ DataLoader.load_data() → DuckDB query → Parquet files
    ├─→ MarketCalendar.filter_market_days()
    ├─→ Strategy.generate_signals(data) → (entries, exits)
    ├─→ PortfolioSimulator.simulate()
    │     ├─ For each bar:
    │     │   ├─ Check entry/exit signals
    │     │   ├─ PositionSizer.calculate_shares()
    │     │   ├─ RiskManager.check_constraints()
    │     │   ├─ Execute trade
    │     │   ├─ Update equity curve
    │     │   └─ TradeLogger.log_trade()
    │     └─ Return Portfolio object
    └─→ Calculate metrics
        ├─ Metrics.calculate_performance()
        └─ Return stats dict

    ↓

BacktestVisualizer.generate()
    ├─→ QuantStatsReporter.create_tearsheet()
    ├─→ Charts.generate_candlestick()
    └─→ ReportGenerator.create_summary()

    ↓

Output:
  - Tearsheet HTML/PDF
  - Trade log CSV
  - Candlestick charts PNG
  - Performance summary TXT
```

### Multi-Symbol Sweep Flow

```
SweepRunner.run_sweep(symbols=['AAPL', 'MSFT', 'GOOGL'])
    ↓
ThreadPoolExecutor (parallel execution)
    ├─ Worker 1: BacktestEngine.run('AAPL')
    ├─ Worker 2: BacktestEngine.run('MSFT')
    └─ Worker 3: BacktestEngine.run('GOOGL')
        │
        ├─→ Callback: on_symbol_start('AAPL')
        ├─→ Callback: on_symbol_complete('AAPL', portfolio)
        └─→ Callback: on_symbol_error('AAPL', error)

    ↓ (all symbols complete)

ResultsAggregator.aggregate()
    ├─ Combine equity curves
    ├─ Aggregate metrics
    └─ Create comparison charts

    ↓

Return: List[Portfolio] + Aggregate Reports
```

### GUI Execution Flow

```
User clicks "Run Backtest" in GUI
    ↓
GUIBacktestController.start()
    ├─ Create worker thread
    ├─ Start SweepRunner in background
    └─ Return immediately (UI responsive)

Worker Thread:
  SweepRunner.run_sweep()
    ├─→ Put progress updates in queue
    ├─→ on_symbol_complete → Queue.put(result)
    └─→ on_error → Queue.put(error)

Main Thread (UI):
  while running:
    ├─ Poll queue for updates
    ├─ Update progress bars
    ├─ Update status labels
    └─ Render results view when complete

    ↓

ResultsView displays:
  - Metrics table
  - Link to tearsheet
  - Link to charts
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

### GUI

| Technology | Purpose |
|------------|---------|
| **Flet** | Cross-platform GUI framework |

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

**File**: `src/run_ingestion.py`

**Usage**:
```bash
python -m src.run_ingestion
```

### 3. GUI Application

**File**: `src/gui/__main__.py`

**Usage**:
```bash
python -m gui
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
- Both CLI and GUI interfaces
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
- **Data Loading**: DuckDB → ~1-2 seconds for 1 year of 1-minute data
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
- ✅ New strategies (inherit `BaseStrategy`)
- ✅ New indicators (add to `indicators.py`)
- ✅ New position sizing methods (add to `PositionSizer`)
- ✅ New data sources (implement API client interface)
- ✅ New risk constraints (add to `RiskManager`)
- ✅ New brokers (implement focused interfaces - ISP-compliant design)

### Planned Enhancements
- 🚧 Options trading support (interface ready: `OptionsTradingInterface`)
- 📋 Futures trading support
- 📋 Intraday rebalancing
- 📋 Machine learning strategy integration
- 📋 Additional broker integrations (TastyTrade, IBKR)

### Recently Deployed
- ✅ **Config-driven backtesting** - YAML-based backtest configuration (November 2025)
  - Single command: `python -m src.backtest_runner --config path/to/config.yaml`
  - Supports all modes: single, sweep, optimize, walk-forward
  - Pydantic-validated configuration schema with inheritance (`extends:` directive)
  - Predefined date presets and symbol universes
  - Strategy registry with lazy loading (no import chain issues)
  - See [config/backtesting/](../../config/backtesting/) for example configs

- ✅ **Broker interface refactoring** - ISP-compliant interface design (November 2025)
  - 6 focused interfaces: AccountInterface, MarketHoursInterface, MarketDataInterface, OrderManagementInterface, StockTradingInterface, OptionsTradingInterface
  - BrokerInterface is now a composite interface (backward compatible)
  - Backward-compatible method aliases preserve existing code
  - Ready for multi-broker support (Alpaca, TastyTrade, IBKR)
  - 39 new interface compliance tests
  - See [MODULE_REFERENCE.md](MODULE_REFERENCE.md#trading-system-layer) for details

- ✅ **Live trading integration** - Paper trading deployed to AWS EC2 with automated scheduling (November 2025)
  - EC2 instance with Python 3.11 (t4g.small ARM64)
  - Lambda-powered auto-start/stop (9 AM - 4:30 PM ET Mon-Fri)
  - Systemd service with auto-restart capabilities
  - SSH management scripts and automated health monitoring
  - See [Infrastructure Overview](../INFRASTRUCTURE_OVERVIEW.md) for details

---

## References

- **Module Reference**: [MODULE_REFERENCE.md](MODULE_REFERENCE.md)
- **Data Flow**: [DATA_FLOW.md](DATA_FLOW.md)
- **Optimization Module**: [OPTIMIZATION_MODULE.md](OPTIMIZATION_MODULE.md)
- **Backtesting Guide**: [../guides/BACKTESTING_GUIDE.md](../guides/BACKTESTING_GUIDE.md)
- **Testing Guide**: [../testing/TEST_SUITE_QUICK_START.md](../testing/TEST_SUITE_QUICK_START.md)

---

**Last Updated**: 2025-11-25
**Maintainers**: Update this doc when adding/removing/moving major modules
**Review Frequency**: After any architectural changes
