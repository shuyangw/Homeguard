# Homeguard Unit Tests

Comprehensive unit test suite for the Homeguard backtesting system.

## Installation

Install test dependencies:

```bash
pip install pytest==8.3.5
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
# Backtest engine tests
pytest tests/test_backtest_engine.py -v

# Strategy tests
pytest tests/test_strategies.py -v

# P&L calculation tests
pytest tests/test_pnl_calculations.py -v
```

### Run Specific Test Class

```bash
pytest tests/test_backtest_engine.py::TestSingleSymbolBacktest -v
```

### Run Specific Test Method

```bash
pytest tests/test_backtest_engine.py::TestSingleSymbolBacktest::test_simple_backtest_executes -v
```

### Run Tests Matching Pattern

```bash
# Run all tests with "pnl" in the name
pytest tests/ -k "pnl" -v

# Run all tests with "signal" in the name
pytest tests/ -k "signal" -v
```

## Test Organization

```
tests/
├── __init__.py
├── README.md                       # This file
├── conftest.py                     # Pytest fixtures (test data)
├── test_backtest_engine.py         # Backtest engine core tests (10 test classes, 30+ tests)
├── test_strategies.py              # Strategy implementation tests (7 test classes, 25+ tests)
└── test_pnl_calculations.py        # P&L and metrics tests (6 test classes, 25+ tests)
```

## Test Coverage

### Backtest Engine Tests (`test_backtest_engine.py`)

- **TestBacktestEngineInitialization**: Engine initialization and configuration
- **TestSingleSymbolBacktest**: Single symbol backtesting functionality
- **TestMultiSymbolBacktest**: Multi-symbol portfolio backtesting
- **TestSignalGeneration**: Signal generation and handling
- **TestBacktestEdgeCases**: Edge cases and error handling

### Strategy Tests (`test_strategies.py`)

- **TestMovingAverageCrossover**: MA crossover strategy
- **TestMeanReversion**: Bollinger Bands strategy
- **TestRSIMeanReversion**: RSI strategy
- **TestMomentumStrategy**: MACD momentum strategy
- **TestBreakoutStrategy**: Breakout strategy
- **TestTripleMovingAverage**: Triple MA strategy

### P&L Calculation Tests (`test_pnl_calculations.py`)

- **TestBasicPnLCalculations**: Basic profit/loss calculations
- **TestPerformanceMetrics**: Sharpe ratio, drawdown, win rate
- **TestTradeMetrics**: Trade counts, profit factor, avg win/loss
- **TestFeeImpact**: Impact of fees on returns
- **TestCapitalManagement**: Capital allocation and conservation

## Test Fixtures

Test fixtures are defined in `conftest.py` and provide synthetic price data:

- `simple_price_data`: Trending upward price data (100 days)
- `oscillating_price_data`: Oscillating price data (100 days)
- `multi_symbol_data`: Multi-symbol data with 3 symbols (50 days each)
- `flat_price_data`: Flat price data with minimal movement (50 days)

## Writing New Tests

When adding new functionality, add corresponding tests:

1. **Create test class**:
```python
class TestNewFeature:
    """Test new feature functionality."""

    def test_basic_functionality(self, simple_price_data):
        """Test that new feature works."""
        # Arrange
        engine = BacktestEngine()

        # Act
        result = engine.new_feature()

        # Assert
        assert result is not None
```

2. **Test edge cases**:
```python
    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            engine = BacktestEngine(invalid_param=-1)
```

3. **Run your new tests**:
```bash
pytest tests/test_backtest_engine.py::TestNewFeature -v
```

## Test-Driven Development

Follow the TDD workflow:

1. **Run tests before changes**: `pytest tests/`
2. **Make changes**
3. **Run tests frequently**: `pytest tests/test_backtest_engine.py -v`
4. **Fix failures**: Iterate until all tests pass
5. **Run full suite**: `pytest tests/`

Never commit code with failing tests!

## Continuous Integration

Tests should be run:

- Before every commit
- Before every pull request
- After modifying:
  - Backtesting engine code
  - Strategy implementations
  - P&L calculations
  - Data handling code

## Test Performance

Current test performance (approximate):

- **Full test suite**: ~30-60 seconds
- **Engine tests**: ~15-25 seconds
- **Strategy tests**: ~10-20 seconds
- **P&L tests**: ~15-25 seconds

Individual tests should run in < 1 second each.

## Troubleshooting

### pytest not found

Install pytest:
```bash
pip install pytest==8.3.5
```

### Import errors

Ensure you're running from the repo root:
```bash
cd C:\Users\qwqw1\Dropbox\cs\github\Homeguard
pytest tests/
```

### VectorBT errors

Some tests require VectorBT. Install it:
```bash
pip install vectorbt==0.28.1
```

### Slow tests

If tests are slow, run specific tests:
```bash
pytest tests/test_backtest_engine.py::TestSingleSymbolBacktest -v
```

## Best Practices

1. **Fast tests**: Keep individual tests under 1 second
2. **Isolated tests**: Each test should be independent
3. **Clear names**: Test names should describe what they test
4. **Good fixtures**: Use fixtures for common test data
5. **Assert messages**: Include descriptive assertion messages
6. **Test edge cases**: Don't just test the happy path

## Adding Test Dependencies

If you add a new testing dependency, update `requirements.txt`:

```bash
echo "new-package==1.0.0" >> requirements.txt
```

Then install:

```bash
pip install -r requirements.txt
```
