# Backtesting Examples

This directory contains practical examples demonstrating the backtesting framework.

## Available Examples

### 1. Basic Backtest (`basic_backtest.py`)

**What it demonstrates:**
- Loading data
- Creating a strategy
- Running a simple backtest
- Viewing results and statistics

**Run it:**
```bash
cd src
python ../examples/basic_backtest.py
```

**What you'll learn:**
- How to initialize the BacktestEngine
- How to create a strategy instance
- How to run a backtest
- How to access portfolio statistics

---

### 2. Parameter Optimization (`parameter_optimization.py`)

**What it demonstrates:**
- Defining parameter grids
- Running grid search optimization
- Comparing optimal vs default parameters
- Validating on out-of-sample data

**Run it:**
```bash
cd src
python ../examples/parameter_optimization.py
```

**What you'll learn:**
- How to optimize strategy parameters
- How to define parameter search spaces
- How to validate optimized parameters
- How to avoid overfitting

---

### 3. Multi-Strategy Comparison (`multi_strategy_comparison.py`)

**What it demonstrates:**
- Testing multiple strategies on the same data
- Comparing performance metrics
- Identifying the best strategy
- Generating comparison reports

**Run it:**
```bash
cd src
python ../examples/multi_strategy_comparison.py
```

**What you'll learn:**
- How to run multiple strategies
- How to compare strategy performance
- How to use PerformanceMetrics for analysis
- How to identify optimal strategies for different objectives

---

### 4. Custom Strategy Development (`custom_strategy_example.py`)

**What it demonstrates:**
- Creating a custom strategy from scratch
- Implementing custom indicators
- Testing and optimizing custom strategies
- Validating custom strategy performance

**Run it:**
```bash
cd src
python ../examples/custom_strategy_example.py
```

**What you'll learn:**
- How to create custom strategies
- How to combine technical indicators
- How to implement entry/exit logic
- How to optimize custom strategy parameters

---

## Running Examples

### Prerequisites

1. Ensure you have data ingested for the required symbols and date ranges
2. All examples use AAPL data from 2022-2024
3. Run from the `src/` directory to ensure proper imports

### Modifying Examples

Feel free to modify these examples:

- Change symbols: Replace `'AAPL'` with other symbols
- Change date ranges: Adjust `start_date` and `end_date`
- Change parameters: Experiment with different strategy parameters
- Add strategies: Import and test additional strategies

### Example Modifications

#### Test Multiple Symbols

```python
# In any example, change:
symbols=['AAPL']

# To:
symbols=['AAPL', 'MSFT', 'GOOGL']
```

#### Test Different Date Range

```python
# Change:
start_date='2023-01-01',
end_date='2024-01-01'

# To:
start_date='2022-01-01',
end_date='2023-06-01'
```

#### Test Different Strategy Parameters

```python
# Change:
strategy = MovingAverageCrossover(fast_window=20, slow_window=50)

# To:
strategy = MovingAverageCrossover(fast_window=10, slow_window=30, ma_type='ema')
```

---

## Example Output

Each example produces formatted output showing:

1. **Configuration**: Strategy parameters, date ranges, symbols
2. **Progress**: Real-time backtest execution updates
3. **Results**: Performance metrics and statistics
4. **Visualizations**: Optional plots (close window to continue)

### Sample Output Format

```
================================================================================
Basic Backtest Example
================================================================================

1. Creating Moving Average Crossover strategy...
   Strategy: MovingAverageCrossover(fast_window=20, slow_window=50, ma_type='sma')

2. Initializing backtest engine...
   Engine initialized

3. Running backtest...

================================================================================
Running backtest: MovingAverageCrossover(fast_window=20, slow_window=50, ma_type='sma')
Symbols: AAPL
Period: 2023-01-01 to 2024-01-01
Initial capital: $100,000.00
Fees: 0.10%
================================================================================

Loaded 98520 bars for 1 symbol(s) from 2023-01-01 to 2024-01-01

================================================================================
BACKTEST RESULTS
================================================================================
Total Return:       25.34%
Annual Return:      25.34%
Sharpe Ratio:       1.82
Max Drawdown:       -8.45%
Win Rate:           58.33%
Total Trades:       24
Final Value:        $125,340.00
================================================================================
```

---

## Creating Your Own Examples

You can create your own examples by following this template:

```python
"""
My Custom Example

Description of what this example demonstrates.
"""

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover


def main():
    print("=" * 80)
    print("My Custom Example")
    print("=" * 80)

    # Your example code here
    strategy = MovingAverageCrossover()
    engine = BacktestEngine(initial_capital=100000)

    portfolio = engine.run(
        strategy,
        ['AAPL'],
        '2023-01-01',
        '2024-01-01'
    )

    print(portfolio.stats())


if __name__ == '__main__':
    main()
```

---

## Next Steps

After running these examples:

1. Read the [Backtesting Guide](../docs/BACKTESTING_GUIDE.md)
2. Review the [API Reference](../docs/API_REFERENCE.md)
3. Create your own custom strategy
4. Run backtests on your strategies
5. Optimize and validate your strategies

---

## Getting Help

If you encounter issues:

1. Ensure you're in the `src/` directory when running
2. Verify data is available for the symbols and date ranges
3. Check that all dependencies are installed
4. Review error messages for specific issues

For common issues, see the Troubleshooting section in the [Backtesting Guide](../docs/BACKTESTING_GUIDE.md).
