# Backtesting Engine

**A comprehensive backtesting framework for algorithmic trading strategies with optimization, regime analysis, and walk-forward validation.**

**Last Updated**: 2025-12-08

---

## Overview

### What It Does
- Executes historical backtests for trading strategies on OHLCV data
- Supports single-symbol, multi-symbol sweep, and pairs trading modes
- Provides parameter optimization (grid search, Bayesian, genetic algorithms)
- Detects and analyzes market regimes (bull, bear, sideways)
- Validates strategies via walk-forward analysis
- Generates performance reports and tearsheets

### Key Features
- **Multiple Execution Modes**: Single symbol, parallel sweep across universe, multi-asset portfolio
- **Custom Portfolio Simulator**: Bar-by-bar simulation with risk management integration
- **Optimization Suite**: Grid search, random search, Bayesian, genetic algorithms
- **Regime Analysis**: Automatic market regime detection and strategy performance segmentation
- **Walk-Forward Validation**: Out-of-sample testing to prevent overfitting
- **Config-Driven**: YAML configuration for reproducible backtests

### Use Cases
- Backtest new trading strategy ideas
- Optimize strategy parameters across a symbol universe
- Validate strategy robustness with walk-forward analysis
- Analyze strategy performance across market regimes
- Generate professional tearsheets for strategy evaluation

---

## Architecture

```
src/backtesting/
├── __init__.py                      # Public API: BacktestEngine, DataLoader, BaseStrategy
├── base/
│   ├── strategy.py                  # BaseStrategy, MultiSymbolStrategy abstract classes
│   └── pairs_strategy.py            # PairsStrategy for market-neutral trading
├── engine/
│   ├── backtest_engine.py           # Main orchestrator for all backtests
│   ├── data_loader.py               # Load historical data from Parquet/DuckDB
│   ├── portfolio_simulator.py       # Bar-by-bar portfolio simulation
│   ├── numba_sim.py                 # JIT-compiled simulator for performance
│   ├── sweep_runner.py              # Parallel execution across symbols
│   ├── metrics.py                   # Performance metrics calculation
│   ├── multi_symbol_metrics.py      # Aggregated metrics across symbols
│   ├── multi_asset_portfolio.py     # Simultaneous multi-symbol positions
│   ├── pairs_portfolio.py           # Pairs trading portfolio management
│   ├── multi_pair_portfolio.py      # Multi-pair portfolio tracking
│   ├── tearsheet_generator.py       # QuantStats tearsheet generation
│   ├── trade_logger.py              # Trade event logging
│   ├── benchmark_calculator.py      # Benchmark comparison (SPY, etc.)
│   ├── results_aggregator.py        # Aggregate results from sweeps
│   └── portfolio_aggregator.py      # Combine portfolio results
├── optimization/
│   ├── base_optimizer.py            # Abstract optimizer interface
│   ├── grid_search.py               # Exhaustive grid search
│   ├── random_search.py             # Random parameter sampling
│   ├── bayesian_optimizer.py        # Bayesian optimization
│   ├── genetic_optimizer.py         # Genetic algorithm optimization
│   ├── regime_aware.py              # Regime-aware optimization
│   ├── walk_forward.py              # Walk-forward optimization
│   ├── sweep_runner.py              # Optimization sweep runner
│   └── result_cache.py              # Cache optimization results
├── regimes/
│   ├── detector.py                  # Market regime detection (bull/bear/sideways)
│   ├── analyzer.py                  # Regime-specific performance analysis
│   └── exporter.py                  # Export regime analysis results
├── chunking/
│   └── walk_forward.py              # Walk-forward data chunking
├── reporting/
│   └── standard_report.py           # Standardized backtest reports
├── utils/
│   ├── risk_manager.py              # Position limits, stop losses
│   ├── position_sizer.py            # Position sizing methods
│   ├── pairs_position_sizer.py      # Pairs-specific sizing
│   ├── risk_config.py               # Risk configuration presets
│   ├── indicators.py                # Technical indicators (SMA, RSI, ATR, etc.)
│   └── market_calendar.py           # NYSE trading calendar
└── adapters/
    └── (strategy adapters for config-driven backtests)
```

### Design Philosophy

1. **Separation of Concerns**: Engine orchestrates, strategies generate signals, simulators execute
2. **Pluggable Components**: Easily swap optimizers, position sizers, risk managers
3. **Config-Driven**: YAML configs enable reproducible backtests without code changes
4. **Risk-First**: Risk management is integrated, not optional
5. **Performance**: Numba JIT compilation for critical paths

---

## Key Components

### BacktestEngine (`engine/backtest_engine.py`)

**Purpose**: Main orchestrator that coordinates data loading, strategy execution, and result generation.

**Key Methods**:
- `run()`: Execute single-symbol backtest
- `run_sweep()`: Parallel backtest across multiple symbols
- `optimize()`: Grid search parameter optimization
- `run_with_config()`: Config-driven backtest execution

**Usage**:
```python
from src.backtesting import BacktestEngine, DataLoader

engine = BacktestEngine()
portfolio = engine.run(
    strategy=my_strategy,
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    initial_capital=100000
)
```

### DataLoader (`engine/data_loader.py`)

**Purpose**: Load historical OHLCV data from Parquet files via DuckDB.

**Key Methods**:
- `load_data()`: Load data for a symbol and date range
- `load_multiple()`: Load data for multiple symbols

**Features**:
- Market calendar filtering (excludes weekends/holidays)
- Data validation and cleaning
- Efficient DuckDB queries on partitioned Parquet

### PortfolioSimulator (`engine/portfolio_simulator.py`)

**Purpose**: Bar-by-bar simulation with integrated risk management.

**Key Features**:
- Entry/exit signal execution
- Position sizing via PositionSizer
- Stop loss and take profit handling
- Trade logging with detailed metrics
- Equity curve tracking

### Optimization Suite (`optimization/`)

**Available Optimizers**:
| Optimizer | Use Case | Method |
|-----------|----------|--------|
| `GridSearchOptimizer` | Exhaustive search | All combinations |
| `RandomSearchOptimizer` | Large parameter spaces | Random sampling |
| `BayesianOptimizer` | Efficient exploration | Gaussian process |
| `GeneticOptimizer` | Complex landscapes | Evolutionary algorithm |

**Usage**:
```python
from src.backtesting.optimization import GridSearchOptimizer

optimizer = GridSearchOptimizer(engine, strategy_class)
best_params, best_value = optimizer.optimize(
    param_grid={'fast_period': [10, 20], 'slow_period': [50, 100]},
    metric='sharpe_ratio'
)
```

### Regime Analysis (`regimes/`)

**Purpose**: Detect market regimes and analyze strategy performance within each.

**Regimes Detected**:
- **Bull**: Sustained uptrend (SMA above threshold)
- **Bear**: Sustained downtrend (SMA below threshold)
- **Sideways**: Range-bound (low volatility, no trend)

---

## Data Flow

```
YAML Config / CLI Args
        ↓
   BacktestEngine
        ↓
   DataLoader → Parquet/DuckDB
        ↓
   Strategy.generate_signals()
        ↓
   PortfolioSimulator
        ├─→ PositionSizer
        ├─→ RiskManager
        └─→ TradeLogger
        ↓
   Portfolio (equity curve, trades, stats)
        ↓
   TearsheetGenerator / ReportGenerator
        ↓
   HTML/PDF Reports
```

---

## Public API

### Primary Exports

```python
from src.backtesting import BacktestEngine, DataLoader, BaseStrategy

# For optimization
from src.backtesting.optimization import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    GeneticOptimizer,
)

# For regime analysis
from src.backtesting.regimes import RegimeDetector, RegimeAnalyzer

# For risk management
from src.backtesting.utils import RiskManager, PositionSizer, RiskConfig
```

### Config-Driven Usage

```bash
# Run backtest from YAML config
python -m src.backtest_runner --config config/backtesting/ma_single.yaml
```

---

## Configuration

### YAML Config Files
Located in `config/backtesting/`:
- `ma_single.yaml` - Single symbol MA crossover
- `momentum_sweep.yaml` - Multi-symbol momentum sweep
- `lgbm_walk_forward.yaml` - Walk-forward validation example

### Risk Config Presets
```python
from src.backtesting.utils import RiskConfig

config = RiskConfig.moderate()  # 10% per trade, balanced
config = RiskConfig.conservative()  # 5% per trade, 60% cash reserve
config = RiskConfig.aggressive()  # 20% per trade
```

---

## Dependencies

### Internal (src/ modules)
- `src.data_engine` - Historical data storage
- `src.strategies` - Strategy implementations
- `src.visualization` - Charts and reports
- `src.settings` - Configuration loading

### External (pip packages)
- `pandas` - Data manipulation
- `numpy` - Numerical computation
- `duckdb` - Fast Parquet queries
- `quantstats` - Performance tearsheets
- `numba` - JIT compilation
- `scikit-learn` - Bayesian optimization
- `deap` - Genetic algorithms

---

## Testing

### Test Location
- `tests/backtesting/` - Unit tests
- `tests/engine/` - Engine-specific tests
- `tests/optimization/` - Optimizer tests

### Running Tests
```bash
pytest tests/backtesting/ -v
pytest tests/engine/ -v
```

---

## Related Documentation

- [Architecture Overview](../../docs/architecture/ARCHITECTURE_OVERVIEW.md)
- [Module Reference](../../docs/architecture/MODULE_REFERENCE.md)
- [Backtesting Guidelines](../../backtest_guidelines/guidelines.md)

---

## Changelog

- **2025-12-08**: Initial documentation created
- **2025-11-27**: Config-driven backtesting added
- **2025-11-15**: Numba JIT simulator added
- **2025-10-XX**: Initial backtesting engine
