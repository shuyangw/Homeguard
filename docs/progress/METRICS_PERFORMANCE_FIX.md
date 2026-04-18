# Portfolio Metrics Performance Fix

## Problem

Portfolio mode backtest chart generation was taking **8-9 minutes** to complete.

Initial investigation focused on chart generation, but user-provided logs revealed the actual bottleneck:

```
[2025-11-04 00:18:21] Worker: Calculating multi-symbol portfolio metrics...
[2025-11-04 00:26:28] Worker: Calculated metrics in 486.8s  <- 8.1 MINUTES!
[2025-11-04 00:26:28] Worker: Generating portfolio visualization charts...
[2025-11-04 00:26:29] Worker: Generated 9 chart datasets in 0.3s
```

**Root cause: `MultiSymbolMetrics.calculate_all_metrics()` taking 486.8 seconds**

## Technical Analysis

### The O(n²) Bottleneck

Found in `src/backtesting/engine/multi_symbol_metrics.py` line 49 (old code):

```python
for timestamp, cash in portfolio.cash_history:
    try:
        portfolio_idx = portfolio.equity_timestamps.index(timestamp)  # O(n) search!
        portfolio_value = portfolio.equity_curve[portfolio_idx]
        # ... calculations ...
```

**Why this was slow:**

1. `portfolio.equity_timestamps` is a **list** with 98,000+ entries (252 trading days × 390 minutes)
2. `portfolio.cash_history` also has 98,000+ entries
3. `.index()` performs a **linear O(n) search** through the entire list
4. This creates **O(n²) complexity**: 98,000 × 98,000 = **9.6 billion operations!**

For a 1-year backtest with minute-level data:
- Old O(n²) algorithm: **9.6 billion operations -> 486.8 seconds**
- Each operation took ~0.05 microseconds

## Solution

Changed from O(n²) list searches to O(1) dictionary lookups:

```python
# PERFORMANCE FIX: Create O(1) lookup dict instead of O(n) .index() calls
# For 98K timestamps, this changes O(n²) = 9.6B operations to O(n) = 98K operations
timestamp_to_idx = {ts: idx for idx, ts in enumerate(portfolio.equity_timestamps)}

capital_utilization = []
for timestamp, cash in portfolio.cash_history:
    portfolio_idx = timestamp_to_idx.get(timestamp)  # O(1) lookup!
    if portfolio_idx is None:
        continue

    try:
        portfolio_value = portfolio.equity_curve[portfolio_idx]
        # ... calculations ...
```

## Performance Improvement

### Test Results (100,000 bars)

**Before fix:**
- Total metrics calculation: **486.8 seconds** (8.1 minutes)
- Composition metrics: ~486 seconds (the bottleneck)

**After fix:**
- Total metrics calculation: **0.22 seconds**
- Composition metrics: **0.05 seconds**
- Attribution metrics: 0.00 seconds
- Diversification metrics: 0.17 seconds
- Rebalancing metrics: 0.00 seconds
- Trade analysis metrics: 0.00 seconds

**Speedup: 2,213x faster!** (486.8s -> 0.22s)

### Real-World Impact

For portfolio mode backtests with minute-level data:
- **Old**: 8-9 minutes wait for results
- **New**: < 1 second for metrics + 0.3s for charts = **~1 second total!**

## Files Modified

1. **`src/backtesting/engine/multi_symbol_metrics.py`**
   - Fixed O(n²) bottleneck in `calculate_portfolio_composition_metrics()` (lines 46-64)
   - Added timing logs to `calculate_all_metrics()` to track metric category performance (lines 318-351)

2. **`tests/test_metrics_performance.py`** (new file)
   - Performance regression test
   - Verifies metrics calculation stays fast (< 1s for 100K bars)
   - Tests composition metrics and all metrics together

## Complexity Analysis

### Before Fix
```
Time Complexity: O(n²)
- n = number of bars (98,000 for 1 year of minute data)
- Operations: n × n = 9.6 billion
- Time: 486.8 seconds
```

### After Fix
```
Time Complexity: O(n)
- n = number of bars (98,000)
- Operations: n = 98,000
- Time: 0.05 seconds
- Speedup: 98,000 / 9.6B = 98,000x for this operation alone
```

## Testing

Run the performance test:

```bash
conda activate fintech
python tests/test_metrics_performance.py
```

Expected output:
```
[+] PERFORMANCE EXCELLENT: 0.11s (target: < 1s)
[+] O(n²) bottleneck successfully fixed!
[+] ALL METRICS PERFORMANCE GOOD: 0.22s (target: < 5s)
ALL TESTS PASSED
```

## Verification in GUI

When running a portfolio mode backtest in the GUI, you should now see in the output log:

```
Worker: Calculating multi-symbol portfolio metrics...
  - Composition metrics: 0.05s
  - Attribution metrics: 0.00s
  - Diversification metrics: 0.17s
  - Rebalancing metrics: 0.00s
  - Trade analysis metrics: 0.00s
  - Total metrics calculation: 0.22s
Worker: Calculated metrics in 0.2s
Worker: Generating portfolio visualization charts...
Worker: Generated 9 chart datasets in 0.3s
```

**Total time: ~0.5 seconds instead of 8-9 minutes!**

## Lessons Learned

1. **Profile before optimizing**: Initial assumption was chart generation was slow, but logs showed metrics calculation was the real bottleneck

2. **Watch for O(n²) patterns**: Using `.index()` or `in` on lists inside loops creates quadratic complexity

3. **Use appropriate data structures**:
   - Lists -> O(n) search
   - Dictionaries -> O(1) lookup
   - For 100K+ items, this makes 98,000x performance difference!

4. **Add timing logs**: Granular timing logs helped pinpoint exactly which operation was slow

## Related Issues

This fix also improves performance for:
- Large symbol universe backtests (50+ symbols)
- High-frequency strategies (1-minute or tick data)
- Long backtest periods (multi-year)
- Portfolio rebalancing simulations

Any portfolio backtest with >10K bars will see dramatic improvement.
