# Regime-Based Testing Architecture

**Version:** 2.1
**Status:** Active (CLI + programmatic API; GUI/Level-4 file-export was removed when `src/gui/` was deleted)
**Last Updated:** 2026-05-17

## Overview

The Regime-Based Testing system provides advanced validation techniques to assess strategy robustness across different market conditions. This prevents overfitting and identifies failure conditions before deploying strategies to live trading.

## Motivation

### The Problem with Long Backtests

Running a single backtest over multiple years can hide critical weaknesses:

1. **Hidden Regime Bias**: A strategy may perform well in bull markets but fail catastrophically in bear markets
2. **Overfitting**: Parameters optimized on the entire dataset may not perform well on future data
3. **Statistical Artifacts**: One exceptional year can mask consistent mediocrity
4. **Regime Changes**: Market conditions change, and strategies must adapt

### The Solution

Break up long backtests into:
1. **Walk-Forward Validation**: Train/test on rolling windows to prevent overfitting
2. **Regime Analysis**: Analyze performance across market conditions (bull/bear/volatile/calm)
3. **Robustness Scoring**: Quantify consistency across regimes

## Architecture

### Module Structure

```
src/backtesting/
├── chunking/
│   ├── __init__.py
│   └── walk_forward.py          # Walk-forward validation
│
└── regimes/
    ├── __init__.py
    ├── detector.py               # Regime detection (trend, volatility, drawdown)
    └── analyzer.py               # Performance analysis by regime

tests/backtesting/
├── chunking/
│   ├── __init__.py
│   └── test_walk_forward.py     # 10 tests
│
└── regimes/
    ├── __init__.py
    ├── test_detector.py          # 20 tests
    └── test_analyzer.py          # 13 tests

backtest_scripts/
└── regime_analysis_example.py    # Proof-of-concept examples
```

## Components

### 1. Walk-Forward Validation (`backtesting/chunking/walk_forward.py`)

Prevents overfitting by testing on truly out-of-sample data.

**Classes:**

- `WalkForwardWindow`: Represents a single train/test window
- `WalkForwardResults`: Contains aggregated walk-forward results
- `WalkForwardValidator`: Executes walk-forward validation

**Key Method:**

```python
def validate(
    strategy_class: type,
    param_grid: Dict[str, List[Any]],
    symbols: Union[str, List[str]],
    start_date: str,
    end_date: str,
    metric: str = 'sharpe_ratio'
) -> WalkForwardResults:
    """
    Run walk-forward validation.

    For each window:
    1. Optimize parameters on training period
    2. Test with optimal params on out-of-sample period
    3. Track degradation

    Returns only out-of-sample performance (true performance).
    """
```

**Algorithm:**

1. Generate rolling train/test windows
   - Example: Train on 12 months, test on 3 months, step forward 3 months
2. For each window:
   - Optimize parameters on training data using `GridSearchOptimizer`
   - Test with best parameters on out-of-sample test data
   - Calculate performance degradation
3. Aggregate all out-of-sample results
4. Report degradation and warn if >20%

**Degradation Warnings:**

- `< 10%`: Low degradation - strategy appears robust [+]
- `10-20%`: Moderate degradation - acceptable range
- `> 20%`: High degradation - strategy may be overfit! [!]

### 2. Regime Detection (`backtesting/regimes/detector.py`)

Automatically detects different market regimes from price data.

**Regime Types:**

1. **Trend Regimes**:
   - `BULL`: Strong upward trend (>5% over lookback period)
   - `BEAR`: Strong downward trend (<-5% over lookback period)
   - `SIDEWAYS`: No clear trend (-5% to +5%)

2. **Volatility Regimes**:
   - `HIGH_VOL`: Above 70th percentile of rolling volatility
   - `LOW_VOL`: Below 70th percentile of rolling volatility

3. **Drawdown Regimes**:
   - `DRAWDOWN`: Drawdown worsening (>10% below high water mark)
   - `RECOVERY`: Recovering from drawdown
   - `CALM`: Near high water mark

**Classes:**

- `TrendDetector`: Detects bull/bear/sideways markets
- `VolatilityDetector`: Detects high/low volatility periods
- `DrawdownDetector`: Detects drawdown/recovery/calm phases

**Example Usage:**

```python
from backtesting.regimes import TrendDetector

detector = TrendDetector(lookback_days=60, threshold_pct=5.0)
regimes = detector.detect(market_prices)

for regime in regimes:
    print(f"{regime.regime.value}: {regime.start_date} to {regime.end_date}")
```

### 3. Regime Analysis (`backtesting/regimes/analyzer.py`)

Analyzes strategy performance across detected regimes.

**Classes:**

- `RegimePerformance`: Performance metrics for a specific regime
- `RegimeAnalysisResults`: Aggregated results across all regimes
- `RegimeAnalyzer`: Main analysis engine

**Key Metrics:**

- **Sharpe Ratio by Regime**: Risk-adjusted returns in each regime
- **Total Return by Regime**: Raw returns in each regime
- **Robustness Score** (0-100): Consistency across regimes
  - `>= 70`: Excellent - highly consistent
  - `50-70`: Good - reasonably consistent
  - `< 50`: Poor - varies significantly by regime

**Example Usage:**

```python
from backtesting.regimes import RegimeAnalyzer

analyzer = RegimeAnalyzer(
    trend_lookback=60,
    vol_lookback=20,
    drawdown_threshold=10.0
)

results = analyzer.analyze(
    portfolio_returns=returns,
    market_prices=market_prices,
    trades=None  # Optional
)

print(f"Robustness Score: {results.robustness_score:.1f}/100")
print(f"Best Regime: {results.best_regime}")
print(f"Worst Regime: {results.worst_regime}")
```

## Integration with Existing System

### Dependencies

```
RegimeAnalyzer
    └─> RegimeDetectors (Trend, Volatility, Drawdown)

WalkForwardValidator
    └─> GridSearchOptimizer
        └─> BacktestEngine
```

### Data Flow

```
Market Data (prices)
    │
    ├─> TrendDetector ────┐
    ├─> VolatilityDetector ├──> RegimePeriods
    └─> DrawdownDetector ──┘
                │
                ▼
        Portfolio Returns ──> RegimeAnalyzer ──> RegimeAnalysisResults
                                                      │
                                                      ├─> Performance by regime
                                                      ├─> Robustness score
                                                      └─> Best/worst regimes

Strategy Parameters
    │
    └─> WalkForwardValidator ──> WalkForwardResults
            │                         │
            ├─> Train: GridSearchOptimizer  ├─> In-sample performance
            └─> Test: BacktestEngine        ├─> Out-of-sample performance
                                            └─> Degradation %
```

## Usage Examples

### Example 0: Programmatic Engine Integration

```python
from backtesting.engine.backtest_engine import BacktestEngine
from strategies.research.moving_average import MovingAverageCrossover

# Create engine with regime analysis enabled
engine = BacktestEngine(
    initial_capital=10000,
    fees=0.001,
    enable_regime_analysis=True  # Enable automatic regime analysis
)

# Run backtest
strategy = MovingAverageCrossover(fast_window=20, slow_window=100)
portfolio = engine.run(
    strategy=strategy,
    symbols=['AAPL'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Regime analysis automatically printed to console
```

### Example 1: Walk-Forward Validation

```python
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.chunking import WalkForwardValidator
from strategies.research.moving_average import MovingAverageCrossover

# Create engine
engine = BacktestEngine(initial_capital=10000, fees=0.001)

# Create validator
validator = WalkForwardValidator(
    engine=engine,
    train_months=12,  # 12 months training
    test_months=3,    # 3 months testing
    step_months=3     # Step forward 3 months
)

# Run validation
results = validator.validate(
    strategy_class=MovingAverageCrossover,
    param_grid={
        'fast_window': [10, 20, 30],
        'slow_window': [50, 100, 150]
    },
    symbols='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31',
    metric='sharpe_ratio'
)

# Check results
print(f"Out-of-sample Sharpe: {results.out_of_sample_sharpe:.2f}")
print(f"Degradation: {results.degradation_pct:.1f}%")
```

### Example 2: Regime Analysis

```python
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.regimes import RegimeAnalyzer
from strategies.research.moving_average import MovingAverageCrossover

# Run backtest
engine = BacktestEngine(initial_capital=10000, fees=0.001)
strategy = MovingAverageCrossover(fast_window=20, slow_window=100)

portfolio = engine.run(
    strategy=strategy,
    symbols=['AAPL'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    price_type='close'
)

# Get returns
returns = portfolio.returns()

# Load market data for regime detection (use StreamingDataLoader directly --
# BacktestEngine does NOT expose a `.data_loader` attribute)
from backtesting.engine.streaming_data_loader import StreamingDataLoader

loader = StreamingDataLoader()
market_df = loader.load_symbol('AAPL', '2020-01-01', '2023-12-31').collect().to_pandas()
market_prices = market_df.set_index('timestamp')['close']

# Analyze by regime
analyzer = RegimeAnalyzer()
results = analyzer.analyze(
    portfolio_returns=returns,
    market_prices=market_prices
)

print(f"Robustness Score: {results.robustness_score:.1f}/100")
```

### Example 3: Combined Analysis (Production Readiness Assessment)

```python
# Step 1: Walk-forward validation
wf_results = validator.validate(...)

# Step 2: Regime analysis on out-of-sample returns
regime_results = analyzer.analyze(
    portfolio_returns=wf_results.oos_returns,
    market_prices=market_prices
)

# Step 3: Production readiness assessment
is_robust = abs(wf_results.degradation_pct) < 20
is_consistent = regime_results.robustness_score >= 60

if is_robust and is_consistent:
    print("[+] PASS: Strategy is production-ready!")
elif is_robust:
    print("[!] CONDITIONAL: Low overfitting but inconsistent across regimes")
elif is_consistent:
    print("[!] CONDITIONAL: Consistent but may be overfit")
else:
    print("[-] FAIL: Strategy not production-ready")
```

## Proof-of-Concept Script

**Location:** `backtest_scripts/regime_analysis_example.py`

**Features:**
- Example 1: Walk-forward validation demonstration
- Example 2: Regime-based analysis demonstration
- Example 3: Combined analysis with production readiness assessment

**Run:**
```bash
conda activate fintech
python backtest_scripts/regime_analysis_example.py
```

## Test Coverage

**Total Tests:** 43

### Walk-Forward Tests (10)
- Window generation
- Validation execution
- Sharpe calculation
- Results aggregation

### Regime Detector Tests (20)
- Trend detection (bull/bear/sideways)
- Volatility detection (high/low)
- Drawdown detection (drawdown/recovery/calm)
- Edge cases (empty data, insufficient data)

### Regime Analyzer Tests (13)
- Performance calculation by regime
- Robustness scoring
- Extremes detection
- Edge cases

**Run Tests:**
```bash
conda activate fintech
pytest tests/backtesting/chunking/ tests/backtesting/regimes/ -v
```

## Best Practices

### When to Use Walk-Forward Validation

[+] **Always use when:**
- Optimizing strategy parameters
- Backtesting over > 2 years
- Testing for production deployment
- Publishing research results

[-] **Don't use when:**
- Quick exploratory analysis
- Fixed parameters (no optimization)
- Very short backtest periods (< 1 year)

### When to Use Regime Analysis

[+] **Use when:**
- Strategy shows high variability in results
- Backtesting across multiple market cycles
- Comparing strategies for robustness
- Identifying strategy weaknesses

### Interpretation Guidelines

**Walk-Forward Degradation:**
- `< 10%`: Excellent - low overfitting risk
- `10-20%`: Good - acceptable degradation
- `20-30%`: Warning - potential overfitting
- `> 30%`: Severe - strategy likely overfit

**Robustness Score:**
- `80-100`: Exceptional - works in all conditions
- `60-80`: Good - reasonably consistent
- `40-60`: Fair - some regime dependency
- `< 40`: Poor - highly regime-dependent

**Combined Assessment:**

| Degradation | Robustness | Assessment |
|------------|-----------|------------|
| < 20% | > 60 | [+] Production Ready |
| < 20% | < 60 | [!] Regime-Specific (consider adaptive sizing) |
| > 20% | > 60 | [!] Overfit but Consistent (longer training) |
| > 20% | < 60 | [-] Not Production Ready |

## Implementation Status

The CLI/programmatic regime testing surface is the supported entry point:

- [+] `enable_regime_analysis` parameter on `BacktestEngine` (transparent integration)
- [+] Walk-forward validation module (`backtesting/chunking/walk_forward.py`)
- [+] Regime detection (trend, volatility, drawdown) and analyzer modules (`backtesting/regimes/`)
- [+] Proof-of-concept scripts in `backtest_scripts/`
- [+] Comprehensive test suite (43 tests across walk-forward + regime modules)

**Removed:** The original Level 2 (GUI checkbox) and Level 4 (Regime Analysis tab, HTML/CSV/JSON export wired into the GUI controller) integrations were retired together with `src/gui/`. The underlying regime detector/analyzer remain available programmatically and via CLI scripts; teams that previously consumed the exported CSV/HTML/JSON should call `RegimeAnalyzer.analyze(...)` directly and serialize from the returned dataclasses.

## References

### Related Documentation
- [Optimization Module](../planning/OPTIMIZATION_MODULE.md) - GridSearchOptimizer integration
- [Backtesting Guide](../guides/BACKTESTING_GUIDE.md) - Core engine
- [Test Suite Quick Start](../testing/TEST_SUITE_QUICK_START.md) - Testing standards

### Academic References
- Walk-Forward Analysis: Bailey et al. (2014) "Probabilistic Sharpe Ratio"
- Regime Detection: Nystrup et al. (2020) "Multi-Period Portfolio Selection"
- Overfitting: López de Prado (2018) "Advances in Financial Machine Learning"

## Changelog

### Version 2.1 (May 2026)
- Retired GUI-bound Level 2 / Level 4 integrations together with `src/gui/`
- Updated examples to use `src/strategies/research/` paths and the `StreamingDataLoader` (`BacktestEngine` does not expose a `.data_loader` attribute)
- Consolidated Implementation Status into a single CLI/programmatic surface

### Version 2.0 (November 2025) [superseded]
- Level 4 GUI display and file export -- removed in 2.1 when `src/gui/` was deleted
- Multi-symbol support with dropdown selector -- removed in 2.1

### Version 1.0 (November 2025)
- Initial implementation of Level 3 (Advanced CLI Tools)
- Walk-forward validation system
- Regime detection and analysis
- Proof-of-concept examples
- Comprehensive test suite (43 tests)
- Level 1 (engine flag) and Level 2 (GUI checkbox -- since removed)
