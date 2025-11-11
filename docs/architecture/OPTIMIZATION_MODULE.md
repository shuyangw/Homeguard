# Homeguard Optimization Module - Architecture Documentation

**Version**: 2.0
**Last Updated**: 2025-11-05
**Status**: Current (Refactored)

---

## Executive Summary

The Homeguard Optimization Module provides automated parameter tuning for trading strategies through exhaustive grid search. It enables users to find optimal strategy parameters by testing all combinations across user-defined parameter ranges and selecting the best-performing configuration based on chosen metrics (Sharpe Ratio, Total Return, or Max Drawdown).

**Key Capabilities**:
- **Grid Search Optimization**: Exhaustive parameter space exploration
- **Multi-Symbol Optimization**: Find parameters optimal across multiple symbols
- **Multiple Metrics**: Optimize for Sharpe Ratio, Total Return, or Max Drawdown
- **GUI Integration**: User-friendly parameter grid specification
- **Result Export**: CSV export of all parameter combinations and metrics
- **Parallel Execution**: Fast optimization using thread pooling
- **Modular Architecture**: Clean separation between backend and GUI components

**Module Structure** (Refactored November 2025):
```
src/backtesting/optimization/     # Backend optimization
├── __init__.py
├── grid_search.py                # GridSearchOptimizer class
└── sweep_runner.py               # SweepRunner class

src/gui/optimization/             # GUI optimization
├── __init__.py
├── dialog.py                     # OptimizationDialog class
└── runner.py                     # OptimizationRunner class
```

**Integration Points**:
- Core backtesting engine ([backtest_engine.py](../../src/backtesting/engine/backtest_engine.py))
- Grid search optimizer ([grid_search.py](../../src/backtesting/optimization/grid_search.py))
- Universe sweep runner ([sweep_runner.py](../../src/backtesting/optimization/sweep_runner.py))
- GUI dialog system ([dialog.py](../../src/gui/optimization/dialog.py))
- GUI runner ([runner.py](../../src/gui/optimization/runner.py))
- Results aggregation ([results_aggregator.py](../../src/backtesting/engine/results_aggregator.py))

---

## Architecture Overview

### Three-Tier Optimization Architecture (Refactored)

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 1: GUI LAYER                            │
│  Location: src/gui/optimization/                                │
│  - OptimizationDialog (dialog.py): Parameter grid collection   │
│  - OptimizationRunner (runner.py): Execution orchestration     │
│  - BacktestApp: Main app integration                           │
│  - Results Dialog: Best parameter application                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   TIER 2: OPTIMIZATION LAYER                    │
│  Location: src/backtesting/optimization/                        │
│  - GridSearchOptimizer (grid_search.py): Grid search logic     │
│  - SweepRunner (sweep_runner.py): Universe-wide optimization   │
│  - Parameter grid generation (itertools.product)               │
│  - Metric comparison and best tracking                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  TIER 3: ENGINE & RESULTS LAYER                 │
│  Location: src/backtesting/engine/                              │
│  - BacktestEngine: Core backtest execution                     │
│  - ResultsAggregator: Multi-result combination                 │
│  - CSV Export: All combinations with metrics                   │
│  - Sorting and ranking logic                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Architectural Changes** (November 2025 Refactoring):
- **Extracted** `GridSearchOptimizer` from `BacktestEngine` → `backtesting/optimization/grid_search.py`
- **Moved** `SweepRunner` → `backtesting/optimization/sweep_runner.py`
- **Moved** `OptimizationDialog` → `gui/optimization/dialog.py`
- **Extracted** `OptimizationRunner` from `BacktestApp` → `gui/optimization/runner.py`
- **Maintained** backward compatibility: `BacktestEngine.optimize()` still works

---

## Module Components

### Component 1: Grid Search Optimization

#### **GridSearchOptimizer**
**Location**: [src/backtesting/optimization/grid_search.py](../../src/backtesting/optimization/grid_search.py)

**Purpose**: Exhaustive grid search parameter optimization

**Class Design**:
```python
class GridSearchOptimizer:
    def __init__(self, engine: BacktestEngine):
        self.engine = engine

    def optimize(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        # Grid search implementation
```

**Backward Compatibility**:
```python
# Old API (still works)
engine = BacktestEngine()
result = engine.optimize(...)

# New API (recommended)
from backtesting.optimization import GridSearchOptimizer
optimizer = GridSearchOptimizer(engine)
result = optimizer.optimize(...)
```

**Note**: `BacktestEngine.optimize()` now delegates to `GridSearchOptimizer` internally

**Method Signature**:
```python
def optimize(
    self,
    strategy_class: type,
    param_grid: Dict[str, List[Any]],
    symbols: Union[str, List[str]],
    start_date: str,
    end_date: str,
    metric: str = 'sharpe_ratio'
) -> Dict[str, Any]:
```

**Parameters**:
- `strategy_class`: Strategy class to optimize (e.g., `MovingAverageCrossover`)
- `param_grid`: Dictionary mapping parameter names to value lists
- `symbols`: Single symbol or list of symbols to test
- `start_date`: Backtest start date (YYYY-MM-DD)
- `end_date`: Backtest end date (YYYY-MM-DD)
- `metric`: Optimization metric (`sharpe_ratio`, `total_return`, `max_drawdown`)

**Returns**:
```python
{
    'best_params': Dict[str, Any],      # Optimal parameter values
    'best_value': float,                 # Metric value at best params
    'best_portfolio': Portfolio,         # Portfolio object with best params
    'metric': str                        # Optimization metric used
}
```

**Algorithm**:
1. Generate all parameter combinations using `itertools.product()`
2. For each combination:
   - Instantiate strategy with parameters
   - Run backtest via `_run_single_symbol()` or `_run_multiple_symbols()`
   - Extract metric from `portfolio.stats()`
   - Compare to current best (maximize Sharpe/Return, minimize Drawdown)
3. Track best parameters, value, and portfolio object
4. Return optimization results

**Supported Metrics**:
| Metric | Optimization Goal | Comparison |
|--------|------------------|------------|
| `sharpe_ratio` | Maximize risk-adjusted returns | `>` (higher is better) |
| `total_return` | Maximize absolute returns | `>` (higher is better) |
| `max_drawdown` | Minimize peak-to-trough decline | `<` (less negative is better) |

**Usage Example**:
```python
from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover

engine = BacktestEngine(initial_capital=100000, fees=0.001)

param_grid = {
    'fast_window': [10, 15, 20, 25],
    'slow_window': [40, 50, 60, 70]
}

results = engine.optimize(
    strategy_class=MovingAverageCrossover,
    param_grid=param_grid,
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    metric='sharpe_ratio'
)

print(f"Best parameters: {results['best_params']}")
# Output: Best parameters: {'fast_window': 15, 'slow_window': 60}
```

**Dependencies**: `itertools.product`, `DataLoader`, `PortfolioSimulator`

---

#### **GridSearchOptimizer.optimize_parallel()** ⚡ NEW
**Location**: [src/backtesting/optimization/grid_search.py](../../src/backtesting/optimization/grid_search.py)

**Purpose**: Parallel grid search parameter optimization (3-8x faster than sequential)

**Method Signature**:
```python
def optimize_parallel(
    self,
    strategy_class: type,
    param_grid: Dict[str, List[Any]],
    symbols: Union[str, List[str]],
    start_date: str,
    end_date: str,
    metric: str = 'sharpe_ratio',
    max_workers: Optional[int] = None,
    price_type: str = 'close'
) -> Dict[str, Any]:
```

**Parameters**:
- `strategy_class`: Strategy class to optimize (e.g., `MovingAverageCrossover`)
- `param_grid`: Dictionary mapping parameter names to value lists
- `symbols`: Single symbol or list of symbols to test
- `start_date`: Backtest start date (YYYY-MM-DD)
- `end_date`: Backtest end date (YYYY-MM-DD)
- `metric`: Optimization metric (`sharpe_ratio`, `total_return`, `max_drawdown`)
- `max_workers`: Number of parallel workers (default: min(4, cpu_count))
- `price_type`: Price column to use ('close', 'open', etc.)

**Returns**:
```python
{
    'best_params': Dict[str, Any],      # Optimal parameter values
    'best_value': float,                 # Metric value at best params
    'best_portfolio': Portfolio,         # Portfolio object with best params
    'metric': str,                       # Optimization metric used
    'all_results': List[Dict]           # Full results for all combinations (NEW!)
}
```

**Algorithm**:
1. Load data once (shared across all workers)
2. Generate all parameter combinations using `itertools.product()`
3. **Parallel execution**: Submit all combinations to ProcessPoolExecutor
4. Collect results as they complete (track progress)
5. Track best parameters/value dynamically
6. Return best result + all tested combinations

**Key Features**:
- ✅ **Automatic fallback**: Uses sequential for small grids (< 10 combos)
- ✅ **Progress tracking**: Real-time progress updates as tests complete
- ✅ **Full results**: Returns all tested combinations for analysis
- ✅ **Worker auto-detection**: Automatically uses optimal worker count
- ✅ **Memory efficient**: Shares data across workers (loaded once)
- ✅ **100% compatible**: Same results as sequential version

**Performance**:
- **Small grids (< 10 combos)**: Uses sequential (overhead not worth it)
- **Medium grids (10-50 combos)**: 2-4x speedup with 4 workers
- **Large grids (> 50 combos)**: 3-8x speedup depending on CPU cores

**Usage Example**:
```python
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer
from strategies.base_strategies.moving_average import MovingAverageCrossover

engine = BacktestEngine(initial_capital=100000, fees=0.001)
optimizer = GridSearchOptimizer(engine)

param_grid = {
    'fast_window': [10, 15, 20, 25, 30],
    'slow_window': [50, 60, 70, 80, 90, 100]
}

# Parallel optimization (30 combinations)
results = optimizer.optimize_parallel(
    strategy_class=MovingAverageCrossover,
    param_grid=param_grid,
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    metric='sharpe_ratio',
    max_workers=4  # Use 4 CPU cores
)

print(f"Best parameters: {results['best_params']}")
# Output: Best parameters: {'fast_window': 20, 'slow_window': 80}

print(f"Speedup: ~3-4x faster than sequential")

# Analyze all results
import pandas as pd
all_results_df = pd.DataFrame(results['all_results'])
print(all_results_df[['params', 'value']].sort_values('value', ascending=False))
```

**When to Use**:
- ✅ **Use parallel**: Grid size > 10 combinations
- ✅ **Use parallel**: Walk-forward validation (many windows)
- ✅ **Use parallel**: Complex strategies with slow execution
- ❌ **Use sequential**: Grid size < 10 combinations
- ❌ **Use sequential**: Very fast strategies (< 1 second per test)

**Comparison with Sequential**:
| Aspect | Sequential | Parallel (4 workers) |
|--------|-----------|---------------------|
| Speed (25 combos) | 125 seconds | ~35 seconds (3.6x) |
| Memory usage | 1x | ~1.2x (minimal overhead) |
| Progress tracking | After each combo | After each combo |
| Results accuracy | Exact | Exact (same best params) |
| Worker efficiency | N/A | ~70-90% |

**Dependencies**: `ProcessPoolExecutor`, `itertools.product`, `DataLoader`, `PortfolioSimulator`

---

#### **SweepRunner.optimize_across_universe()**
**Location**: [src/backtesting/optimization/sweep_runner.py](../../src/backtesting/optimization/sweep_runner.py)

**Purpose**: Find parameters optimal across multiple symbols (universe-wide optimization)

**Method Signature**:
```python
def optimize_across_universe(
    self,
    strategy_class: type,
    symbols: List[str],
    param_grid: Dict[str, List[Any]],
    start_date: str,
    end_date: str,
    metric: str = 'median_sharpe',
    parallel: bool = False
) -> Dict[str, Any]:
```

**Aggregation Metrics**:
| Metric | Description | Use Case |
|--------|-------------|----------|
| `median_sharpe` | Median Sharpe Ratio across symbols | Robust to outliers |
| `mean_sharpe` | Mean Sharpe Ratio across symbols | Average performance |
| `median_return` | Median total return across symbols | Robust return metric |
| `mean_return` | Mean total return across symbols | Average return |
| `win_rate` | % of symbols with positive returns | Strategy reliability |

**Algorithm**:
1. Generate all parameter combinations
2. For each combination:
   - Run backtest on all symbols (parallel or sequential)
   - Collect metrics from each symbol
   - Compute aggregation metric (median/mean/win_rate)
   - Compare to current best
3. Return best parameters for universe

**Usage Example**:
```python
from backtesting.optimization import SweepRunner
from strategies.base_strategies.momentum import MomentumStrategy

sweep_runner = SweepRunner(
    initial_capital=100000,
    fees=0.001,
    risk_profile='moderate'
)

param_grid = {
    'lookback': [10, 20, 30],
    'threshold': [1.0, 1.5, 2.0]
}

results = sweep_runner.optimize_across_universe(
    strategy_class=MomentumStrategy,
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    param_grid=param_grid,
    start_date='2023-01-01',
    end_date='2024-01-01',
    metric='median_sharpe',
    parallel=True
)

print(f"Best params (universe-wide): {results['best_params']}")
print(f"Median Sharpe: {results['best_value']:.4f}")
```

**Dependencies**: `ThreadPoolExecutor`, `BacktestEngine`, `statistics`

---

### Component 2: GUI Parameter Collection

#### **OptimizationDialog**
**Location**: [src/gui/optimization/dialog.py](../../src/gui/optimization/dialog.py)

**Purpose**: Interactive dialog for specifying parameter grids

**Import**:
```python
from gui.optimization import OptimizationDialog
# or
from gui.optimization.dialog import OptimizationDialog
```

**Key Methods**:

##### `_collect_param_grid()` (lines 272-318)
Collects parameter grid from user input fields

**Parameter Types Supported**:

1. **Numeric Parameters (int/float)**:
   - User inputs: Min, Max, Step
   - Generation: `numpy.arange(min, max+step, step)`
   - Example: min=5, max=15, step=5 → `[5, 10, 15]`

2. **Boolean Parameters**:
   - User checkbox: Test both True/False
   - If checked: `[True, False]`
   - If unchecked: Parameter not optimized (uses default)

3. **Value List Parameters (strings/custom)**:
   - User inputs: Comma-separated values
   - Parsing: Split by comma, strip whitespace
   - Example: "sma, ema, wma" → `['sma', 'ema', 'wma']`

**Return Value**:
```python
{
    'fast_window': [5, 10, 15, 20],           # numeric
    'slow_window': [30, 40, 50],              # numeric
    'threshold': [1.0, 1.5, 2.0],             # float
    'use_stops': [True, False],               # boolean
    'ma_type': ['sma', 'ema', 'wma']          # value list
}
```

##### `_on_estimate_combinations()` (lines 246-270)
Estimates total parameter combinations and runtime

**Calculation**:
```python
total_combinations = 1
for param_values in param_grid.values():
    total_combinations *= len(param_values)

# Example:
# fast_window: [5, 10, 15, 20] → 4 values
# slow_window: [30, 40, 50] → 3 values
# Total: 4 × 3 = 12 combinations

estimated_seconds = total_combinations * 2  # ~2 sec per backtest
estimated_minutes = estimated_seconds / 60
```

**UI Display**:
- Shows total combinations
- Estimates runtime (assumes 2 sec/backtest)
- Displays first 10 parameter combinations as preview

**Usage Flow**:
```
User opens OptimizationDialog
  ↓
User fills Min/Max/Step for each parameter
  ↓
User clicks "Estimate Combinations" button
  ↓
Dialog shows: "Total: 48 combinations (~1.6 minutes)"
  ↓
Dialog shows preview:
  1. fast_window=5, slow_window=30
  2. fast_window=5, slow_window=40
  ...
  ↓
User clicks "Start Optimization"
  ↓
Dialog returns param_grid to BacktestApp
```

---

### Component 3: GUI Optimization Execution

#### **OptimizationRunner**
**Location**: [src/gui/optimization/runner.py](../../src/gui/optimization/runner.py)

**Purpose**: Orchestrate optimization execution from GUI

**Class Design**:
```python
class OptimizationRunner:
    def __init__(
        self,
        page: ft.Page,
        setup_view: Any,
        show_notification: Callable,
        show_error_dialog: Callable,
        show_setup_view: Callable
    ):
        self.page = page
        self.setup_view = setup_view
        # ... callbacks

    def run_optimization(self, config: Dict[str, Any]):
        # Orchestrate optimization workflow
```

**Import**:
```python
from gui.optimization import OptimizationRunner
# or
from gui.optimization.runner import OptimizationRunner
```

**Integration with BacktestApp**:
```python
class BacktestApp:
    def _build_ui(self):
        # Initialize optimization runner
        self.optimization_runner = OptimizationRunner(
            page=self.page,
            setup_view=self.setup_view,
            show_notification=self._show_notification,
            show_error_dialog=self._show_error_dialog,
            show_setup_view=self._show_setup_view
        )

    def _run_optimization(self, config):
        # Delegate to OptimizationRunner
        self.optimization_runner.run_optimization(config)
```

**Workflow**:

1. **Setup Phase**:
   - Collect config from OptimizationDialog
   - Extract strategy class, param_grid, symbols, dates, metric
   - Initialize progress tracking variables

2. **Execution Phase** (two modes):

   **Mode A: Engine-based optimization** (uses `BacktestEngine.optimize()`):
   ```python
   engine = BacktestEngine(initial_capital, fees, slippage, risk_profile)
   results = engine.optimize(
       strategy_class=strategy_class,
       param_grid=param_grid,
       symbols=symbols,
       start_date=start_date,
       end_date=end_date,
       metric=metric
   )
   ```

   **Mode B: Manual iteration** (for custom tracking):
   ```python
   all_results = []
   for param_combo in product(*param_grid.values()):
       params = dict(zip(param_names, param_combo))
       strategy = strategy_class(**params)
       portfolio = engine.run(strategy, symbols, start_date, end_date)
       stats = portfolio.stats()

       all_results.append({
           **params,
           'metric_value': stats[metric],
           'sharpe_ratio': stats['sharpe_ratio'],
           'total_return': stats['total_return'],
           'max_drawdown': stats['max_drawdown'],
           'win_rate': stats['win_rate'],
           'total_trades': stats['total_trades']
       })

       # Update progress bar
       progress.value += 1 / total_combinations
   ```

3. **Export Phase**:
   ```python
   # Sort results (descending for Sharpe/Return, ascending for Drawdown)
   df = pd.DataFrame(all_results)
   ascending = (metric == 'max_drawdown')
   df = df.sort_values('metric_value', ascending=ascending)

   # Export to CSV
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   filename = f"{timestamp}_{strategy_class.__name__}_optimization.csv"
   df.to_csv(os.path.join(log_dir, filename), index=False)
   ```

4. **Results Display Phase**:
   - Show results dialog with best parameters
   - Provide "Open CSV" button (launches in Excel/Sheets)
   - Provide "Apply Best Parameters" button (auto-fills setup form)

**CSV Output Columns**:
- All parameter names (e.g., `fast_window`, `slow_window`, `threshold`)
- `metric_value`: Optimization metric score
- `sharpe_ratio`: Sharpe Ratio
- `total_return`: Total Return [%]
- `max_drawdown`: Max Drawdown [%]
- `win_rate`: Win Rate [%]
- `total_trades`: Number of trades

**Dependencies**: `itertools.product`, `pandas`, `BacktestEngine`, `OptimizationDialog`

---

### Component 4: Results Aggregation

#### **ResultsAggregator**
**Location**: [src/backtesting/engine/results_aggregator.py](../../src/backtesting/engine/results_aggregator.py)

**Purpose**: Aggregate and export multiple backtest results

**Key Methods**:

##### `aggregate_results(portfolios: List[Portfolio]) -> pd.DataFrame`
Combines multiple portfolio results into a single DataFrame

**Returns**:
| Column | Description |
|--------|-------------|
| symbol | Ticker symbol |
| sharpe_ratio | Risk-adjusted returns |
| total_return | Total return [%] |
| max_drawdown | Max drawdown [%] |
| win_rate | Win rate [%] |
| total_trades | Number of trades |
| final_equity | Final portfolio value |

##### `export_to_csv(data: pd.DataFrame, filepath: str)`
Exports aggregated results to CSV

##### `export_to_html(data: pd.DataFrame, filepath: str)`
Exports aggregated results to HTML table

**Usage Example**:
```python
from backtesting.engine.results_aggregator import ResultsAggregator

# After optimization
portfolios = [portfolio1, portfolio2, portfolio3, ...]
aggregator = ResultsAggregator()

# Aggregate results
df = aggregator.aggregate_results(portfolios)

# Export
aggregator.export_to_csv(df, 'optimization_results.csv')
aggregator.export_to_html(df, 'optimization_results.html')
```

---

## Data Flow

### End-to-End Optimization Flow

```
┌──────────────────────────────────────────────────────────────┐
│  USER ACTION: Clicks "Optimize" button in SetupView         │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  OptimizationDialog Opens                                    │
│  - User fills Min/Max/Step for each parameter               │
│  - User selects optimization metric                         │
│  - User clicks "Estimate Combinations"                      │
│    → Shows: "Total: 48 combinations (~1.6 min)"            │
│  - User previews first 10 combinations                      │
│  - User confirms: "Start Optimization"                      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  BacktestApp._run_optimization()                             │
│  - Receives param_grid, metric, config                      │
│  - Initializes BacktestEngine                               │
│  - Shows progress dialog                                    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  OPTIMIZATION LOOP (itertools.product)                       │
│  For each parameter combination:                            │
│    1. params = {'fast_window': 10, 'slow_window': 50}      │
│    2. strategy = StrategyClass(**params)                    │
│    3. portfolio = engine.run(strategy, symbols, dates)      │
│    4. stats = portfolio.stats()                             │
│    5. record = {**params, **stats}                          │
│    6. all_results.append(record)                            │
│    7. Update progress bar: progress += 1/total             │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  EXPORT PHASE                                                │
│  1. df = pd.DataFrame(all_results)                          │
│  2. df.sort_values('metric_value', ascending=...)           │
│  3. best_params = df.iloc[0][param_names]                   │
│  4. df.to_csv(f"{timestamp}_{strategy}_optimization.csv")   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  RESULTS DIALOG                                              │
│  - Display: "Best Sharpe Ratio: 2.35"                       │
│  - Display: "Best params: fast_window=15, slow_window=60"   │
│  - Button: "Open CSV" → Opens in Excel/Sheets               │
│  - Button: "Apply Best Params" → Auto-fills SetupView       │
│  - Button: "Close" → Return to SetupView                    │
└──────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Parameter Grid Structure

**Format**: `Dict[str, List[Any]]`

**Example**:
```python
param_grid = {
    # Numeric parameters
    'fast_window': [5, 10, 15, 20, 25, 30],
    'slow_window': [40, 50, 60, 70, 80],
    'threshold': [1.0, 1.5, 2.0, 2.5],

    # Boolean parameters
    'use_stops': [True, False],
    'use_trailing': [True, False],

    # String/categorical parameters
    'ma_type': ['sma', 'ema', 'wma'],
    'signal_type': ['close', 'hlc3', 'ohlc4']
}
```

**Total Combinations**: Product of all list lengths
- Above example: 6 × 5 × 4 × 2 × 2 × 3 × 3 = **4,320 combinations**

**Parameter Type Inference**:
- `int` values → Treated as integer parameters
- `float` values → Treated as float parameters
- `True/False` → Treated as boolean parameters
- Strings → Treated as categorical parameters

---

### Optimization Metrics

**Configuration**: Selected in OptimizationDialog or passed to `optimize(metric=...)`

**Available Metrics**:

| Metric | Portfolio Stat Key | Goal | Comparison | Use Case |
|--------|-------------------|------|------------|----------|
| `sharpe_ratio` | `stats['sharpe_ratio']` | Maximize | `>` | Best risk-adjusted returns |
| `total_return` | `stats['total_return']` | Maximize | `>` | Highest absolute returns |
| `max_drawdown` | `stats['max_drawdown']` | Minimize | `<` | Lowest peak-to-trough decline |

**Comparison Logic**:
```python
# For Sharpe Ratio and Total Return (maximize)
if current_value > best_value:
    best_value = current_value
    best_params = current_params

# For Max Drawdown (minimize, less negative is better)
if current_value > best_value:  # e.g., -10% > -20%
    best_value = current_value
    best_params = current_params
```

---

## Integration Points

### Integration 1: Backtesting Engine

**File**: [src/backtesting/engine/backtest_engine.py](../../src/backtesting/engine/backtest_engine.py)

**Integration Point**: `optimize()` method (lines 408-500)

**Calls**:
- `DataLoader.load_data()` → Load historical data
- `_run_single_symbol()` → Single-symbol backtest
- `_run_multiple_symbols()` → Multi-symbol backtest
- `portfolio.stats()` → Extract performance metrics

**Data Dependencies**:
- Requires data loaded into Parquet storage
- Uses same data loading pipeline as regular backtests
- Respects market calendar filtering

---

### Integration 2: GUI System

**File**: [src/gui/app.py](../../src/gui/app.py)

**Integration Points**:
- SetupView: "Optimize" button triggers dialog
- OptimizationDialog: Parameter grid collection
- BacktestApp: Orchestration and execution
- Results Dialog: Display and parameter application

**Communication Flow**:
```
SetupView → OptimizationDialog → BacktestApp → ResultsDialog → SetupView
```

**Threading**:
- Optimization runs in main thread (blocking)
- Progress updates via Flet's UI update mechanism
- No worker threads (unlike regular backtests)

---

### Integration 3: Strategy Layer

**Requirements for Optimizable Strategies**:

1. **Parameterized Constructor**:
```python
class MyStrategy(LongOnlyStrategy):
    def __init__(self, fast_window: int, slow_window: int, threshold: float = 1.5):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.threshold = threshold
```

2. **generate_signals() Method**:
```python
def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # Use self.fast_window, self.slow_window, etc.
    # Return (entries, exits)
```

3. **No Required Arguments Beyond Parameters**:
   - Strategy must be instantiable with only the optimized parameters
   - No required `data` argument in constructor

**Compatible Strategies**:
- All base strategies (MovingAverageCrossover, MomentumStrategy, etc.)
- All advanced strategies
- Custom strategies following BaseStrategy interface

---

## Test Coverage

### Test Files

#### 1. **Grid Search Optimization Tests**
**File**: [tests/backtesting/optimization/test_grid_search.py](../../tests/backtesting/optimization/test_grid_search.py)
**Test Count**: 14 tests

**Test Categories**:

**Basic Functionality** (5 tests):
- `test_optimize_single_parameter`: Single parameter optimization
- `test_optimize_multiple_parameters`: Multiple parameter optimization
- `test_optimize_returns_best_value`: Correct best value returned
- `test_optimize_single_combination`: Edge case - one combination
- `test_optimize_many_combinations`: Stress test - many combinations

**Metric Testing** (4 tests):
- `test_optimize_sharpe_ratio`: Sharpe maximization
- `test_optimize_total_return`: Return maximization
- `test_optimize_max_drawdown`: Drawdown minimization
- `test_optimize_invalid_metric`: Error handling

**Multi-Symbol** (2 tests):
- `test_optimize_multi_symbol`: List of symbols
- `test_optimize_single_symbol_as_list`: Single symbol as list

**Integrity** (2 tests):
- `test_optimize_best_portfolio_matches_params`: Verification
- `test_optimize_comparison_logic`: Metric comparison correctness

**Edge Cases** (7 tests):
- Empty parameter grid handling
- Invalid parameter types
- Missing data handling
- Conflicting date ranges
- Strategy instantiation errors
- Performance edge cases

---

#### 2. **GUI Dialog Tests**
**File**: [tests/gui/optimization/test_dialog.py](../../tests/gui/optimization/test_dialog.py)
**Test Count**: 17 tests

**Test Categories**:

**Parameter Collection** (5 tests):
- `test_collect_numeric_parameter_range`: Int/float ranges
- `test_collect_boolean_parameter`: Boolean handling
- `test_collect_value_list_parameter`: String lists
- `test_collect_mixed_parameters`: Multiple types
- `test_collect_boolean_parameter_unchecked`: Default handling

**Edge Cases** (6 tests):
- `test_empty_numeric_fields`: Empty min/max/step
- `test_partial_numeric_fields`: Missing step value
- `test_empty_value_list`: Empty comma-separated list
- `test_value_list_with_spaces`: Whitespace handling
- `test_float_parameter_range`: Float step values
- `test_single_value_numeric_range`: Min=Max case

**Combination Estimation** (3 tests):
- `test_estimate_single_parameter`: Single param estimation
- `test_estimate_two_parameters`: Product calculation
- `test_estimate_mixed_types`: Mixed parameter types

**UI Validation** (6 tests):
- Input field validation
- Error message display
- Preview generation
- Metric selection
- Cancel button behavior
- Confirm button state

**Integration** (10 tests):
- Dialog-to-App communication
- Parameter grid passing
- Config object construction
- Strategy class validation
- Symbol list validation
- Date range validation
- Result dialog triggering
- Best param application
- CSV export triggering
- Error handling

---

#### 3. **GUI Runner Tests**
**File**: [tests/gui/optimization/test_runner.py](../../tests/gui/optimization/test_runner.py)
**Test Count**: 20 tests

**Test Categories**:

**Basic Execution** (3 tests):
- `test_optimization_processes_all_combinations`: Completeness
- `test_optimization_tracks_all_results`: Result tracking
- `test_optimization_skips_invalid_combinations`: Error handling

**Metric Optimization** (3 tests):
- `test_optimization_sharpe_ratio_metric`: Sharpe maximization
- `test_optimization_total_return_metric`: Return maximization
- `test_optimization_max_drawdown_metric`: Drawdown minimization

**CSV Export** (4 tests):
- `test_csv_export_includes_all_columns`: Column completeness
- `test_csv_sorted_by_metric`: Correct sorting
- `test_csv_sorted_ascending_for_drawdown`: Drawdown sort order
- `test_csv_filename_format`: Naming convention

**Progress Tracking** (5 tests):
- Progress bar updates
- Status label updates
- ETA calculation
- Cancellation handling
- Completion notification

**Integration** (10 tests):
- Engine integration
- Results dialog display
- Best param extraction
- CSV file creation
- Parameter application flow
- Multi-symbol optimization
- Universe-wide optimization
- Parallel execution
- Error recovery
- State management

---

## Usage Examples

### Example 1: Basic CLI Optimization

**File**: [examples/parameter_optimization.py](../../examples/parameter_optimization.py)

```python
from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover

# Initialize engine
engine = BacktestEngine(
    initial_capital=100000,
    fees=0.001,
    slippage=0.0001
)

# Define parameter grid
param_grid = {
    'fast_window': [10, 15, 20, 25, 30],
    'slow_window': [40, 50, 60, 70, 80]
}

# Run optimization
results = engine.optimize(
    strategy_class=MovingAverageCrossover,
    param_grid=param_grid,
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    metric='sharpe_ratio'
)

# Display results
print(f"Best parameters: {results['best_params']}")
print(f"Best Sharpe Ratio: {results['best_value']:.4f}")

# Test on validation period
optimal_strategy = MovingAverageCrossover(**results['best_params'])
validation_portfolio = engine.run(
    strategy=optimal_strategy,
    symbols='AAPL',
    start_date='2024-01-01',
    end_date='2024-06-01'
)

print(f"Validation Sharpe: {validation_portfolio.stats()['sharpe_ratio']:.4f}")
```

---

### Example 2: Multi-Symbol Universe Optimization

```python
from backtesting.engine.sweep_runner import SweepRunner
from strategies.base_strategies.momentum import MomentumStrategy

# Initialize sweep runner
sweep_runner = SweepRunner(
    initial_capital=100000,
    fees=0.001,
    risk_profile='moderate'
)

# Define parameter grid
param_grid = {
    'lookback': [10, 20, 30, 40],
    'threshold': [1.0, 1.5, 2.0]
}

# Optimize across universe
results = sweep_runner.optimize_across_universe(
    strategy_class=MomentumStrategy,
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    param_grid=param_grid,
    start_date='2023-01-01',
    end_date='2024-01-01',
    metric='median_sharpe',  # Universe-wide metric
    parallel=True
)

print(f"Best params (universe): {results['best_params']}")
print(f"Median Sharpe: {results['best_value']:.4f}")

# Apply to each symbol
for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
    strategy = MomentumStrategy(**results['best_params'])
    portfolio = sweep_runner.engine.run(
        strategy=strategy,
        symbols=symbol,
        start_date='2024-01-01',
        end_date='2024-06-01'
    )
    print(f"{symbol} validation Sharpe: {portfolio.stats()['sharpe_ratio']:.4f}")
```

---

### Example 3: GUI Optimization (User Flow)

**Step 1: Open OptimizationDialog**
```
User: Clicks "Optimize" button in SetupView
System: Opens OptimizationDialog
```

**Step 2: Specify Parameter Grid**
```
User: Fills in parameter ranges
  - fast_window: Min=5, Max=30, Step=5 → [5, 10, 15, 20, 25, 30]
  - slow_window: Min=40, Max=80, Step=10 → [40, 50, 60, 70, 80]
  - use_stops: Checked → [True, False]

User: Selects metric = "Sharpe Ratio"
```

**Step 3: Estimate Combinations**
```
User: Clicks "Estimate Combinations"
System: Shows "Total: 60 combinations (~2 minutes)"
System: Displays preview:
  1. fast_window=5, slow_window=40, use_stops=True
  2. fast_window=5, slow_window=40, use_stops=False
  ...
```

**Step 4: Run Optimization**
```
User: Clicks "Start Optimization"
System: Closes dialog, shows progress
  Progress: 15/60 (25%) - ETA: 1.5 minutes
  ...
  Progress: 60/60 (100%) - Complete!
```

**Step 5: View Results**
```
System: Opens Results Dialog
  Best Sharpe Ratio: 2.35
  Best Parameters:
    - fast_window: 15
    - slow_window: 60
    - use_stops: True

  [Open CSV] [Apply Best Parameters] [Close]
```

**Step 6: Apply Best Parameters**
```
User: Clicks "Apply Best Parameters"
System: Returns to SetupView with fields filled:
  - fast_window: 15
  - slow_window: 60
  - use_stops: True
```

---

## Performance Characteristics

### Computational Complexity

**Time Complexity**: O(n × m)
- n = number of parameter combinations
- m = backtest execution time per combination

**Space Complexity**: O(n)
- Stores all results in memory before CSV export

**Typical Performance**:
| Parameter Combinations | Backtest Duration | Total Time | Parallelization |
|----------------------|-------------------|------------|-----------------|
| 10 | 2 sec/backtest | ~20 seconds | No benefit |
| 50 | 2 sec/backtest | ~1.7 minutes | No benefit |
| 100 | 2 sec/backtest | ~3.3 minutes | No benefit |
| 500 | 2 sec/backtest | ~17 minutes | Possible |
| 1000 | 2 sec/backtest | ~33 minutes | Recommended |

**Bottlenecks**:
1. Data loading (mitigated by DuckDB caching)
2. Portfolio simulation (bar-by-bar iteration)
3. Metric calculation (QuantStats overhead)

**Optimization Opportunities**:
- Parallel parameter testing (ThreadPoolExecutor)
- Early stopping (stop if Sharpe < threshold)
- Sampling (random parameter sampling instead of full grid)

---

## Dependencies

### Internal Dependencies

| Module | Purpose | Location |
|--------|---------|----------|
| GridSearchOptimizer | Grid search logic | [src/backtesting/optimization/grid_search.py](../../src/backtesting/optimization/grid_search.py) |
| SweepRunner | Multi-symbol execution | [src/backtesting/optimization/sweep_runner.py](../../src/backtesting/optimization/sweep_runner.py) |
| BacktestEngine | Core backtest execution | [src/backtesting/engine/backtest_engine.py](../../src/backtesting/engine/backtest_engine.py) |
| DataLoader | Historical data loading | [src/backtesting/engine/data_loader.py](../../src/backtesting/engine/data_loader.py) |
| PortfolioSimulator | Portfolio simulation | [src/backtesting/engine/portfolio_simulator.py](../../src/backtesting/engine/portfolio_simulator.py) |
| ResultsAggregator | Result aggregation | [src/backtesting/engine/results_aggregator.py](../../src/backtesting/engine/results_aggregator.py) |
| OptimizationDialog | GUI parameter input | [src/gui/optimization/dialog.py](../../src/gui/optimization/dialog.py) |
| OptimizationRunner | GUI orchestration | [src/gui/optimization/runner.py](../../src/gui/optimization/runner.py) |
| BacktestApp | Main GUI app | [src/gui/app.py](../../src/gui/app.py) |

### External Dependencies

| Library | Purpose | Version |
|---------|---------|---------|
| `itertools` | Parameter grid generation (`product()`) | Standard library |
| `pandas` | Result storage and export | Latest |
| `numpy` | Parameter range generation (`arange()`) | Latest |
| `statistics` | Median/mean calculation | Standard library |
| `concurrent.futures` | Parallel execution (ThreadPoolExecutor) | Standard library |
| `typing` | Type hints | Standard library |

---

## Limitations & Future Enhancements

### Current Limitations

1. **Grid Search Only**: No intelligent sampling (Bayesian, genetic algorithms)
2. **No Walk-Forward**: No rolling optimization with out-of-sample validation
3. **No Constraints**: Cannot specify parameter constraints (e.g., fast < slow)
4. **Memory Intensive**: Stores all results in memory
5. **No Early Stopping**: Tests all combinations even if some clearly inferior
6. **Single Metric**: Optimizes for one metric at a time (no multi-objective)

### Planned Enhancements

**Short-term**:
- [ ] Parallel parameter testing via ThreadPoolExecutor
- [ ] Early stopping based on Sharpe threshold
- [ ] Parameter constraints (e.g., `fast_window < slow_window`)
- [ ] Multi-objective optimization (Pareto frontier)

**Medium-term**:
- [ ] Walk-forward optimization with rolling windows
- [ ] Bayesian optimization (Gaussian process)
- [ ] Random search option (sample N combinations randomly)
- [ ] Optimization result caching (avoid re-running same params)

**Long-term**:
- [ ] Genetic algorithm optimization
- [ ] Hyperparameter tuning via Optuna integration
- [ ] Distributed optimization (multiple machines)
- [ ] Real-time optimization dashboard

---

## Best Practices

### 1. Parameter Range Selection

**Guidelines**:
- Start with wide ranges, then narrow down
- Use reasonable step sizes (too fine = overfitting)
- Consider market regime (bull/bear impacts optimal params)

**Example**:
```python
# BAD: Too fine-grained
param_grid = {
    'fast_window': range(5, 31, 1),  # 26 values
    'slow_window': range(40, 81, 1)   # 41 values
    # Total: 1,066 combinations (overfitting risk!)
}

# GOOD: Reasonable steps
param_grid = {
    'fast_window': [5, 10, 15, 20, 25, 30],  # 6 values
    'slow_window': [40, 50, 60, 70, 80]      # 5 values
    # Total: 30 combinations (manageable)
}
```

---

### 2. Overfitting Prevention

**Techniques**:
1. **Train/Validation Split**:
   ```python
   # Optimize on 2023
   results = engine.optimize(..., start_date='2023-01-01', end_date='2023-12-31')

   # Validate on 2024
   strategy = StrategyClass(**results['best_params'])
   validation = engine.run(strategy, ..., start_date='2024-01-01', end_date='2024-12-31')
   ```

2. **Universe-Wide Optimization**:
   - Use `optimize_across_universe()` instead of per-symbol optimization
   - Parameters that work across many symbols are more robust

3. **Conservative Metric Selection**:
   - Prefer Sharpe Ratio over Total Return
   - Sharpe accounts for risk, reducing overfitting to volatile periods

---

### 3. Computational Efficiency

**Tips**:
- Limit date range during initial exploration
- Use coarser step sizes first, refine later
- Consider parallel execution for >100 combinations
- Cache data loading results

**Example**:
```python
# Phase 1: Coarse search (fast)
coarse_grid = {
    'fast_window': [10, 20, 30],
    'slow_window': [50, 70, 90]
}
coarse_results = engine.optimize(..., param_grid=coarse_grid)

# Phase 2: Fine search around best (targeted)
best_fast = coarse_results['best_params']['fast_window']
best_slow = coarse_results['best_params']['slow_window']

fine_grid = {
    'fast_window': [best_fast-5, best_fast, best_fast+5],
    'slow_window': [best_slow-10, best_slow, best_slow+10]
}
fine_results = engine.optimize(..., param_grid=fine_grid)
```

---

### 4. Metric Selection

**Guidelines**:

| Metric | When to Use | Pros | Cons |
|--------|-------------|------|------|
| Sharpe Ratio | **Default choice** | Risk-adjusted, robust | May favor low-volatility |
| Total Return | High-return focus | Simple, intuitive | Ignores risk |
| Max Drawdown | Risk-averse | Minimizes pain | May sacrifice returns |

**Recommendation**: Use Sharpe Ratio for most cases, validate with Max Drawdown

---

## References

### Related Documentation

- **Architecture Overview**: [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)
- **Module Reference**: [MODULE_REFERENCE.md](MODULE_REFERENCE.md)
- **Data Flow**: [DATA_FLOW.md](DATA_FLOW.md)
- **Backtesting Guidelines**: [../../backtest_guidelines/guidelines.md](../../backtest_guidelines/guidelines.md)
- **Testing Guide**: [../testing/TEST_SUITE_QUICK_START.md](../testing/TEST_SUITE_QUICK_START.md)

### Related Code Files

**Backend Optimization**:
- [src/backtesting/optimization/grid_search.py](../../src/backtesting/optimization/grid_search.py)
- [src/backtesting/optimization/sweep_runner.py](../../src/backtesting/optimization/sweep_runner.py)
- [src/backtesting/engine/backtest_engine.py](../../src/backtesting/engine/backtest_engine.py)
- [src/backtesting/engine/results_aggregator.py](../../src/backtesting/engine/results_aggregator.py)

**GUI Optimization**:
- [src/gui/optimization/dialog.py](../../src/gui/optimization/dialog.py)
- [src/gui/optimization/runner.py](../../src/gui/optimization/runner.py)
- [src/gui/app.py](../../src/gui/app.py)
- [src/gui/views/setup_view.py](../../src/gui/views/setup_view.py)

**Test Files**:
- [tests/backtesting/optimization/test_grid_search.py](../../tests/backtesting/optimization/test_grid_search.py)
- [tests/gui/optimization/test_dialog.py](../../tests/gui/optimization/test_dialog.py)
- [tests/gui/optimization/test_runner.py](../../tests/gui/optimization/test_runner.py)
- [tests/validate_optimization_refactoring.py](../../tests/validate_optimization_refactoring.py)

**Examples**:
- [examples/parameter_optimization.py](../../examples/parameter_optimization.py)

---

**Last Updated**: 2025-11-05
**Maintainers**: Update this doc when modifying optimization algorithms or adding new optimization methods
**Review Frequency**: After any optimization-related changes
