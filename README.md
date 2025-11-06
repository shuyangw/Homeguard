# Homeguard

## Components

### Data Engine

The Data Engine is a production-ready data ingestion and storage system for financial market data.

**Key Features:**
- Multithreaded data ingestion (up to 16x speedup)
- Partitioned Parquet storage with DuckDB backend
- Real-time progress tracking with thread visibility
- Support for multiple data feeds (SIP, IEX, OTC)
- OS-aware configuration (Windows, macOS, Linux)

[Read the Data Engine Documentation](src/data_engine/README.md)

### Backtesting Engine

A comprehensive VectorBT-based backtesting framework for testing stock trading strategies with both **GUI** and **CLI** interfaces.

**Key Features:**
- **🖥️ Desktop GUI**: Modern Flet-based visual interface with real-time progress monitoring
- **Sweep Mode**: Test strategies across multiple stocks with parallel execution (up to 16 workers)
- VectorBT integration for high-performance vectorized backtesting
- 10 built-in strategies (MA, RSI, MACD, Breakout, Volatility-targeted, Pairs Trading, etc.)
- 11 predefined universes (FAANG, DOW30, NASDAQ100, sectors)
- Easy custom strategy development with base classes
- Parameter optimization with grid search
- Comprehensive performance metrics (Sharpe, drawdown, win rate, etc.)
- Automated CSV + HTML reporting
- CLI and programmatic API interfaces

**🖥️ GUI Quick Start (Recommended for Beginners):**

**Windows:**
```bash
# Double-click start_gui.bat in the root directory
# Or run from command line:
start_gui.bat
```

**macOS/Linux:**
```bash
# Make script executable (first time only)
chmod +x start_gui.sh

# Run the launcher
./start_gui.sh
```

The GUI provides:
- Visual strategy selector with dynamic parameter configuration
- Real-time progress cards for each symbol
- Color-coded results table with summary statistics
- Export to CSV/HTML with one click
- Support for 1-16 parallel workers

**Documentation:**
- [GUI User Guide](src/gui/docs/USER_GUIDE.md) - Complete usage instructions
- [GUI README](src/gui/README.md) - Developer documentation

**Quick Start (Windows):**
```bash
# Double-click any script in backtest_scripts/
backtest_scripts\RUN_QUICK_TEST.bat

# Or run a sweep across FAANG stocks
backtest_scripts\basic\01_simple_ma_crossover.bat
```

**Quick Start (Sweep Mode - Recommended):**
```bash
# Sweep backtest across FAANG (5 stocks) with parallel execution
python src\backtest_runner.py \
  --strategy MovingAverageCrossover \
  --universe FAANG \
  --sweep \
  --parallel \
  --start 2023-01-01 \
  --end 2024-01-01

# Sweep across DOW30 (30 stocks) and show top 10 results
python src\backtest_runner.py \
  --strategy BreakoutStrategy \
  --universe DOW30 \
  --sweep \
  --parallel \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --sort-by "Sharpe Ratio" \
  --top-n 10
```

**Pre-configured Scripts**: 25+ ready-to-run sweep batch scripts in `backtest_scripts/`

**Output**: Generates CSV + HTML reports with symbol rankings and summary statistics

[Read the Backtesting Documentation](docs/BACKTESTING_README.md) | [Sweep Mode Guide](docs/SWEEP_MIGRATION_GUIDE.md)

#### QuantStats Reporting

Generate professional tearsheet reports with 50+ metrics, benchmark comparisons, and publication-ready charts:

**Features:**
- Comprehensive performance metrics (Sharpe, Sortino, Calmar, etc.)
- Benchmark comparison (vs SPY, QQQ, or custom benchmarks)
- Risk metrics (VaR, CVaR, max drawdown duration)
- Monthly/yearly returns heatmaps
- Rolling metrics (Sharpe, volatility, beta)
- Drawdown analysis with underwater plot
- HTML reports with PDF export capability

**Usage:**
```python
from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover
from pathlib import Path

# Create engine
engine = BacktestEngine(
    initial_capital=50000,
    fees=0.002,
    benchmark='SPY'
)

# Run backtest with automatic report generation
strategy = MovingAverageCrossover(fast_window=10, slow_window=50)

# Option 1: Auto-generate output directory (uses settings.ini log_output_dir)
# Saved to: {settings.ini:log_output_dir}/{timestamp}_{strategy}_{symbol}/
portfolio = engine.run_and_report(
    strategy=strategy,
    symbols=['AAPL'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Option 2: Custom output directory
# portfolio = engine.run_and_report(
#     strategy=strategy,
#     symbols=['AAPL'],
#     start_date='2024-01-01',
#     end_date='2024-12-31',
#     output_dir=Path('/path/to/custom/directory')
# )
```

**Output Files** (saved to directory configured in settings.ini):
- `tearsheet.html` - Full interactive HTML report
- `quantstats_metrics.txt` - Text metrics summary
- `daily_returns.csv` - Daily returns data
- `equity_curve.csv` - Portfolio value over time

**Log Location:** Reports are saved to the `log_output_dir` configured in [settings.ini](settings.ini.example), not in the repository. See [docs/quantstats/LOG_LOCATION_UPDATE.md](docs/quantstats/LOG_LOCATION_UPDATE.md) for details

## Quick Start

### 1. Installation

**Requirements:**
- Python 3.13.5 or higher
- Alpaca Markets API account

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### 2. Configuration

**Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your Alpaca API credentials
```

**Configure storage paths:**
```bash
cp settings.ini.example settings.ini
# Edit settings.ini and set your preferred data storage directory
```

See [SETUP.md](SETUP.md) for detailed configuration instructions.

### 3. Run Data Ingestion

From the repository root:
```bash
python src\run_ingestion.py
```

This will download historical market data for configured symbols and store them in partitioned Parquet format.

## Project Structure

```
Homeguard/
├── src/
│   ├── data_engine/          # Data ingestion and storage
│   │   ├── api/              # API clients (Alpaca, etc.)
│   │   ├── storage/          # Parquet storage and metadata
│   │   ├── loaders/          # Symbol loaders
│   │   └── orchestration/    # Pipeline orchestration
│   ├── backtesting/          # Backtesting framework
│   │   ├── engine/           # Core backtesting engine
│   │   ├── base/             # Base strategy classes
│   │   └── utils/            # Indicators and utilities
│   ├── visualization/        # QuantStats-based performance visualization
│   │   ├── charts/           # Legacy chart generators (archived)
│   │   ├── reports/          # QuantStats report generation
│   │   └── utils/            # Visualization utilities
│   ├── strategies/           # Strategy implementations
│   │   ├── base_strategies/  # Built-in strategies
│   │   └── custom/           # Custom strategies
│   ├── run_ingestion.py      # Data ingestion runner
│   ├── backtest_runner.py    # Backtest execution CLI
│   └── config.py             # OS-aware configuration
├── examples/                  # Example scripts
├── backtest_scripts/         # Pre-configured backtest batch scripts
├── docs/                      # Architecture documentation
│   ├── quantstats/           # QuantStats reporting documentation
│   └── archived/             # Archived TradingView documentation
├── backtest_lists/           # Symbol lists for ingestion
├── tests/                     # Unit tests
├── settings.ini.example      # Configuration template
├── .env.example              # API credentials template
└── requirements.txt          # Python dependencies
```

## Documentation

### Setup and Configuration
- [Setup Instructions](SETUP.md) - Initial configuration and setup

### Data Engine
- [Data Engine Documentation](src/data_engine/README.md) - Detailed API and usage guide
- [Data Ingestion Pipeline Architecture](docs/DATA_INGESTION_PIPELINE.md) - System design and flow

### Backtesting
- [Backtesting Guide](docs/BACKTESTING_GUIDE.md) - Complete user guide and walkthrough
- [API Reference](docs/API_REFERENCE.md) - Detailed API documentation
- [Backtesting System README](docs/BACKTESTING_README.md) - Quick start and overview

### QuantStats Reporting
- **[QuantStats Documentation](docs/quantstats/README.md) - Complete QuantStats guide**
- [Metrics Explained](docs/quantstats/QUANTSTATS_METRICS_EXPLAINED.md) - Understanding performance metrics
- [Migration Guide](docs/quantstats/MIGRATION_GUIDE_FOR_USERS.md) - Upgrading from TradingView charts
- [Log Configuration](docs/quantstats/LOG_LOCATION_UPDATE.md) - Where reports are saved

## Features

- **High-Performance Data Ingestion**: Parallel processing with configurable workers
- **Efficient Storage**: Partitioned Parquet format optimized for time-series queries
- **Flexible Symbol Loading**: CSV, text files, or programmatic lists
- **Progress Monitoring**: Real-time tracking with time estimates
- **Error Resilience**: Per-symbol error isolation
- **Metadata Tracking**: Index membership and historical tracking
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Development

### Running Tests

The project includes a comprehensive unit test suite covering backtesting engine, strategies, and P&L calculations.

**Run all tests**:
```bash
make test

# Or without make:
python -m pytest tests/ -v
```

**Run specific test suites**:
```bash
make test-engine      # Backtest engine tests
make test-strategies  # Strategy tests
make test-pnl         # P&L calculation tests
```

See [tests/README.md](tests/README.md) for complete testing documentation.

### Makefile Commands

Quick access to common commands:

```bash
make test              # Run all unit tests
make test-verbose      # Run tests with detailed output
make backtest-quick    # Quick backtest verification
make backtest-all      # Run all basic backtests
make ingest            # Run data ingestion
make clean             # Clean Python cache files
make help              # Show all commands
```

See [MAKE_COMMANDS.md](MAKE_COMMANDS.md) for complete command reference.

### Contributing

When contributing:
1. Run tests before committing: `make test`
2. Ensure all tests pass
3. Follow coding guidelines in [CLAUDE.md](CLAUDE.md)
4. Update documentation as needed

## License

*(Add your license here)*
    