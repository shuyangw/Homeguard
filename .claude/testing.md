# Unit Testing Requirements

## When to Run Tests

**ALWAYS run unit tests when modifying:**
- Backtesting engine code (`src/backtesting/engine/`)
- Strategy implementations (`src/strategies/`)
- P&L or metrics calculations (`src/backtesting/engine/metrics.py`)
- Data loaders or data processing (`src/backtesting/engine/data_loader.py`)
- **Report generation** (`portfolio_aggregator.py`, `results_aggregator.py`, `tearsheet_generator.py`, `trade_logger.py`)
- **Chart data generation** (equity curves, performance metrics, HTML charts)
- Any core backtesting functionality

**How to run tests:**
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_backtest_engine.py

# Run with verbose output
pytest tests/ -v

# Run tests matching a pattern
pytest tests/ -k "test_pnl"
```

## Test-First Development (TDD)

**CRITICAL**: When adding NEW functionality, write tests FIRST before implementing the code.

### Why Test-First?

1. **Clarifies requirements**: Writing tests forces you to think through expected behavior
2. **Prevents scope creep**: Tests define exactly what the feature should do
3. **Catches bugs early**: Implementation is guided by failing tests
4. **Documents behavior**: Tests serve as executable specifications

### Test-First Workflow for NEW Features

1. **Define the expected behavior**:
   - What inputs does the function accept?
   - What outputs should it produce?
   - What edge cases need handling?

2. **Write failing tests FIRST**:
   ```python
   def test_new_feature_basic():
       """Test basic functionality."""
       result = new_feature(input)
       assert result == expected_output

   def test_new_feature_edge_case():
       """Test edge case behavior."""
       result = new_feature(edge_input)
       assert result == edge_expected
   ```

3. **Run tests - they should FAIL**:
   ```bash
   pytest tests/test_new_feature.py -v
   # Expected: FAILED (function doesn't exist yet)
   ```

4. **Implement the minimal code to pass tests**:
   - Write just enough code to make tests pass
   - Don't over-engineer

5. **Run tests - they should PASS**:
   ```bash
   pytest tests/test_new_feature.py -v
   # Expected: PASSED
   ```

6. **Refactor if needed**, keeping tests green

### Example: Adding Stop Loss Feature

**Step 1: Write tests first**
```python
# tests/engine/test_stop_loss.py
def test_stop_loss_triggers_at_threshold():
    """Stop loss should trigger when price drops below threshold."""
    # Arrange
    entry_price = 100.0
    stop_loss_pct = -0.10  # 10% stop
    current_price = 89.0   # 11% drop

    # Act
    should_stop = check_stop_loss(entry_price, current_price, stop_loss_pct)

    # Assert
    assert should_stop == True

def test_stop_loss_not_triggered_above_threshold():
    """Stop loss should NOT trigger if price is above threshold."""
    entry_price = 100.0
    stop_loss_pct = -0.10
    current_price = 95.0   # Only 5% drop

    should_stop = check_stop_loss(entry_price, current_price, stop_loss_pct)

    assert should_stop == False
```

**Step 2: Run tests (they fail)**
```bash
pytest tests/engine/test_stop_loss.py -v
# FAILED - check_stop_loss not defined
```

**Step 3: Implement feature**
```python
def check_stop_loss(entry_price, current_price, stop_loss_pct):
    return (current_price - entry_price) / entry_price <= stop_loss_pct
```

**Step 4: Run tests (they pass)**
```bash
pytest tests/engine/test_stop_loss.py -v
# PASSED
```

## Test-Driven Development Workflow (Modifying Existing Code)

When modifying code with unit tests:

1. **Before making changes**:
   - Run existing tests to ensure they pass: `pytest tests/`
   - Identify which tests cover the code you're modifying

2. **While making changes**:
   - Run relevant tests frequently during development
   - If tests fail, iterate on your logic until tests pass
   - Do NOT commit code with failing tests

3. **If tests fail**:
   - Read the failure message carefully
   - Understand WHY the test is failing
   - Fix the logic (not the test, unless the test is incorrect)
   - Re-run tests until all pass
   - Iterate as many times as needed

4. **After making changes**:
   - Run full test suite: `pytest tests/`
   - Ensure ALL tests pass, not just the ones you modified
   - Add new tests if you added new functionality

## Test Coverage Requirements

**Required test coverage areas:**
- **Engine functionality**: Backtest execution, signal handling, portfolio management
- **Strategy behavior**: Signal generation, parameter validation, edge cases
- **P&L calculations**: Returns, fees, drawdowns, trade metrics
- **Data handling**: NaN handling, type conversion, multi-symbol data
- **Edge cases**: Empty signals, minimal data, extreme parameters

## Writing New Tests

When adding new functionality, write tests that cover:

1. **Happy path**: Normal, expected usage
2. **Edge cases**: Boundary conditions
3. **Error conditions**: Invalid inputs
4. **Numerical accuracy**: Calculations are correct

## Test File Organization

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures (test data generators)
├── test_backtest_engine.py        # Engine core functionality tests
├── test_strategies.py             # Strategy implementation tests
├── test_pnl_calculations.py       # P&L and metrics calculation tests
├── test_report_generation.py      # Report/chart generation tests (CRITICAL)
└── test_data_loader.py            # Data loading and processing tests (future)
```

## Report Generation Testing

**CRITICAL**: Always write unit tests for report generation components to prevent subtle bugs like incorrect Sharpe ratio calculations.

### What to Test

When modifying **ANY** of these components, you MUST run/update tests:
- `src/backtesting/engine/portfolio_aggregator.py` - Portfolio aggregation metrics
- `src/backtesting/engine/results_aggregator.py` - HTML report generation
- `src/backtesting/engine/tearsheet_generator.py` - QuantStats tearsheets
- `src/backtesting/engine/trade_logger.py` - CSV trade exports

**Test file**: `tests/test_report_generation.py`

### Critical Tests

1. **Sharpe Ratio with Misaligned Date Ranges** - Catches bugs where ffill creates artificial flat segments
2. **Returns Calculation from Raw Data** - Verifies returns calculated from raw equity, not ffill'd
3. **Aggregate Metrics Consistency** - Tests aggregate metrics match individual symbol metrics
4. **Chart Data Testing** - Verifies chart structure and data integrity
5. **Edge Cases** - Empty portfolios, single symbol, flat equity (no trades)

### Running Report Generation Tests

```bash
# Run all report generation tests
pytest tests/test_report_generation.py -v

# Run specific test class
pytest tests/test_report_generation.py::TestPortfolioAggregation -v

# Run specific critical test
pytest tests/test_report_generation.py::TestPortfolioAggregation::test_misaligned_sharpe_ratio -v
```

### When to Run These Tests

**ALWAYS run before committing changes to:**
- Portfolio aggregation logic (Sharpe, returns, volatility calculations)
- Chart data generation (equity curves, performance charts)
- HTML template modifications (tearsheet layout, chart rendering)
- Metrics calculation (any statistical computations)

## Test Maintenance

- **Keep tests updated**: When functionality changes, update corresponding tests
- **Don't skip tests**: If a test is consistently failing, fix the code or update the test appropriately
- **Test coverage**: Aim for high coverage of critical paths (backtesting, P&L, signals)
- **Fast tests**: Keep unit tests fast (< 1 second each) by using small fixtures
