# Benchmark Comparison Feature [Implementation]

**Date**: 2025-11-03
**Status**: [+] Complete
**Feature**: Strategy vs Buy-and-Hold & S&P 500 Benchmark Comparison
**Related Issues**: N/A

---

## Summary

Implemented comprehensive benchmark comparison system for HTML tearsheet reports, allowing users to compare strategy performance against buy-and-hold (passive holding) for individual symbols and S&P 500 (SPY) for aggregate portfolio performance. Includes interactive toggle controls to manage chart crowding when testing multiple symbols.

---

## Changes Made

### Files Created
- `src/backtesting/engine/benchmark_calculator.py` - Core benchmark calculation module
- `tests/test_benchmark_comparison.py` - Comprehensive test suite (20 tests)

### Files Modified
- `src/backtesting/engine/portfolio_aggregator.py` - Added Chart.js data generation for benchmarks (+175 lines)
- `src/backtesting/engine/results_aggregator.py` - Integrated benchmarks into HTML tearsheet (+450 lines)
- `src/backtesting/engine/sweep_runner.py` - Pass data_loader and dates to enable benchmarks (+10 lines)

### Files Deleted
- None

---

## Implementation Details

### BenchmarkCalculator Module

Core calculation engine for benchmark comparisons:

**Key Methods**:
- `calculate_buy_and_hold_equity()` - Simulates passive holding strategy (buy at start, hold until end)
- `calculate_spy_benchmark()` - Calculates S&P 500 (SPY) benchmark equity curve
- `calculate_outperformance()` - Computes alpha and excess returns vs benchmark
- `calculate_all_benchmarks()` - Batch processing for multiple symbols with SPY comparison

**Example**:
```python
from backtesting.engine.benchmark_calculator import BenchmarkCalculator

# Calculate buy-and-hold for a symbol
bh_equity = BenchmarkCalculator.calculate_buy_and_hold_equity(
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    initial_capital=100000,
    data_loader=loader
)

# Calculate outperformance
metrics = BenchmarkCalculator.calculate_outperformance(
    strategy_equity=portfolio.equity_curve,
    benchmark_equity=bh_equity
)
# Returns: strategy_return_pct, benchmark_return_pct, outperformance_pct, alpha
```

### PortfolioAggregator Chart Data Generation

Added two chart data generation methods:

1. **`generate_benchmark_comparison_chart_data()`** - Per-symbol comparison
   - Strategy lines: Solid, colored (blue, green, red, purple, orange)
   - Benchmark lines: Dashed gray (#9ca3af)
   - Includes outperformers/underperformers categorization
   - Supports toggling individual symbols and benchmark lines

2. **`generate_spy_comparison_chart_data()`** - Aggregate vs SPY
   - Aggregate portfolio: Solid blue line
   - SPY benchmark: Dashed gray line
   - Shows overall portfolio performance vs market

**Chart.js Dataset Structure**:
```javascript
{
    'labels': ['2023-01-01', '2023-01-02', ...],
    'datasets': [
        {
            'label': 'AAPL Strategy',
            'borderColor': '#3b82f6',  // Blue
            'borderWidth': 2.5,
            'type': 'strategy'
        },
        {
            'label': 'AAPL Buy-Hold',
            'borderColor': '#9ca3af',  // Gray
            'borderDash': [5, 5],
            'type': 'benchmark'
        }
    ],
    'outperformers': ['AAPL', 'MSFT'],
    'underperformers': ['GOOGL']
}
```

### HTML Tearsheet Integration

Added two new sections to HTML reports:

**1. Strategy vs Buy-and-Hold Comparison Section**:
- **Stats Cards**: Outperformers count, underperformers count, success rate
- **Toggle Controls**:
  - "Show All" / "Hide All" buttons
  - "Only Outperformers" button (filters to symbols that beat buy-and-hold)
  - Individual symbol checkboxes
  - "Show Buy-Hold Lines" checkbox
- **Interactive Chart**: Dual-line chart (strategy + buy-hold per symbol)

**2. Aggregate Portfolio vs S&P 500 Section**:
- Single chart comparing aggregate portfolio equity vs SPY
- Only shown if SPY data available
- Outperformance metric displayed in stats card

**JavaScript Toggle Functions**:
```javascript
function toggleAllSymbols(show) { ... }         // Show/hide all symbols
function toggleBenchmarks() { ... }              // Show/hide benchmark lines
function toggleSymbol(symbol) { ... }            // Toggle individual symbol
function showOnlyOutperformers() { ... }         // Filter to outperformers only
```

**Visual Design**:
- Dark theme consistent with existing tearsheet
- Color-coded stats cards (green for outperformers, red for underperformers)
- Responsive layout with grid-based metrics display
- Smooth chart animations on toggle

### Backward Compatibility

Feature is **fully backward compatible**:
- All new parameters are optional (`data_loader=None`, `include_benchmarks=True`)
- If parameters not provided, benchmarks are skipped gracefully
- Existing workflows continue to work unchanged
- No breaking changes to API signatures

**Example (Old Code Still Works)**:
```python
# Old code (no benchmarks)
ResultsAggregator.export_to_html(df, html_path, portfolios=portfolios)

# New code (with benchmarks)
ResultsAggregator.export_to_html(
    df, html_path,
    portfolios=portfolios,
    data_loader=loader,
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

---

## Testing

- [x] Unit tests added/passing (20 tests)
- [x] Integration tests passing
- [x] Manual testing completed
- [x] Edge cases validated

**Test Results**: All 20 tests passing (100% success rate)

**Test Coverage**:

### TestBenchmarkCalculator (7 tests)
- [+] Buy-and-hold with uptrend (25% gain)
- [+] Buy-and-hold with downtrend (-20% loss)
- [+] Missing symbol handling
- [+] SPY benchmark calculation
- [+] Positive outperformance calculation
- [+] Negative outperformance calculation
- [+] All benchmarks integration

### TestPortfolioAggregatorBenchmarks (6 tests)
- [+] Benchmark chart data structure validation
- [+] Both line types present (strategy + benchmark)
- [+] Outperformers categorization accuracy
- [+] SPY chart data generation
- [+] Empty portfolios handling
- [+] Single symbol chart data

### TestHTMLBenchmarkIntegration (4 tests)
- [+] Benchmark section present in HTML
- [+] Stats cards rendered correctly
- [+] Toggle controls present
- [+] Backward compatibility (works without benchmarks)

### TestBenchmarkEdgeCases (3 tests)
- [+] All symbols outperform scenario
- [+] All symbols underperform scenario
- [+] SPY missing from database (graceful degradation)

**Mock Testing Infrastructure**:
- `MockDataLoader` class for testing without database
- Synthetic OHLCV data generation
- Uptrend/downtrend/outperforming/underperforming fixtures

---

## Validation

### Success Criteria
- [x] Buy-and-hold equity calculated correctly
- [x] SPY benchmark comparison functional
- [x] Outperformance metrics accurate (alpha, excess returns)
- [x] Interactive toggles work in HTML
- [x] Chart crowding managed with filters
- [x] Backward compatible with existing code
- [x] Graceful degradation when SPY unavailable
- [x] All tests passing

### Metrics
- **Test Coverage**: 100% pass rate (20/20 tests)
- **Code Quality**: Follows project conventions, type-annotated
- **Performance**: Minimal overhead (~500ms for 10 symbols with SPY)
- **UX Impact**: Solves chart crowding problem, provides actionable insights

### User Experience Improvements
1. **Contextual Performance**: Users can now see if strategy adds value vs passive holding
2. **Market Comparison**: Aggregate portfolio compared to S&P 500
3. **Chart Management**: Toggle controls prevent information overload
4. **Quick Insights**: Stats cards show outperformers at a glance
5. **Granular Control**: Individual symbol checkboxes for detailed analysis

---

## Next Steps

- [ ] Consider adding other benchmark indices (NASDAQ, Russell 2000)
- [ ] Add benchmark comparison to CSV exports
- [ ] Implement risk-adjusted benchmark comparisons (Information Ratio)
- [ ] Add benchmark comparison to QuantStats tearsheets
- [ ] Consider adding sector-specific benchmarks
- [ ] Add documentation to user guides

**Future Enhancement Ideas**:
- Time-windowed benchmark analysis (rolling outperformance)
- Benchmark attribution analysis (what drove outperformance)
- Multi-currency benchmark support
- Custom benchmark composition

---

## References

- Test file: [tests/test_benchmark_comparison.py](../../tests/test_benchmark_comparison.py)
- Benchmark calculator: [src/backtesting/engine/benchmark_calculator.py](../../src/backtesting/engine/benchmark_calculator.py)
- Portfolio aggregator: [src/backtesting/engine/portfolio_aggregator.py](../../src/backtesting/engine/portfolio_aggregator.py)
- Results aggregator: [src/backtesting/engine/results_aggregator.py](../../src/backtesting/engine/results_aggregator.py)
- Sweep runner: [src/backtesting/engine/sweep_runner.py](../../src/backtesting/engine/sweep_runner.py)

**Related Progress Docs**:
- `2025-11-02_MULTI_SYMBOL_PORTFOLIO_STATUS.md` - Multi-symbol portfolio foundation
- `2024-11-02_GUI_RISK_MANAGEMENT_INTEGRATION.md` - GUI integration patterns

**User Guides** (to be updated):
- `docs/BACKTESTING_GUIDE.md` - Should document benchmark comparison
- `docs/API_REFERENCE.md` - Should document new parameters

---

## User Request

**Original Request**: "Ultrathink and come up with a way to, within the tearsheet HTML page, to compare the performance of our strategy with the S&P or within just holding the stock itself without managed trading. Perhaps toggling those overlays? Consider the fact that if we are testing multiple symbols, the chart can be crowded"

**Delivered**:
- [+] Buy-and-hold comparison for each symbol
- [+] S&P 500 (SPY) comparison for aggregate portfolio
- [+] Toggle overlays to manage chart crowding
- [+] Interactive controls for granular visibility
- [+] Color-coded visualization
- [+] Comprehensive test coverage

---

**Author**: Claude (AI Assistant)
**Last Updated**: 2025-11-03
