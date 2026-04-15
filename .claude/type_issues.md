# Common Pylance Type Issues and Solutions

## DataFrame.xs() Type Annotation

**Issue**: Pylance incorrectly infers `df.xs()` return type as `DataFrame | Series[Any]` causing type errors when the result is known to be a DataFrame.

**When it occurs**: When using `.xs()` to extract a cross-section from a MultiIndex DataFrame, leaving at least one index level remaining.

**Solution**: Add explicit type annotation with type ignore comment:

```python
# Correct pattern
result: pd.DataFrame = df.xs(symbol, level='symbol')  # type: ignore[assignment]
```

**Where applied**:
- `src/backtesting/engine/backtest_engine.py:95` - `_run_single_symbol()`
- `src/backtesting/engine/backtest_engine.py:127` - `_run_multiple_symbols()`
- `src/backtesting/engine/data_loader.py:105` - `load_single_symbol()`

## VectorBT Incomplete Type Stubs

**Issue**: VectorBT library has incomplete type stubs, causing Pylance to report unknown attributes.

**Common cases**:
- `portfolio.trades.records_readable` - attribute not recognized
- `portfolio.stats()` - can return None but not annotated
- `client.get_stock_bars().df` - .df attribute not recognized

**Solution**: Add type ignore comments for known-good VectorBT API calls:

```python
# For unknown attributes
trades = portfolio.trades.records_readable  # type: ignore[attr-defined]

# For None return values
stats = portfolio.stats()
if stats is None:
    return {}

# For conversion operations that may return Series
value = float(stats.get('Sharpe Ratio', 0))  # type: ignore[arg-type]
```

**Where applied**:
- `src/backtesting/engine/metrics.py` - All methods accessing `portfolio.trades.records_readable`
- `src/backtesting/engine/backtest_engine.py:246` - None check in `optimize()`
- `src/backtesting/engine/backtest_engine.py:289` - None check in `_print_summary()`
- `src/backtesting/engine/backtest_engine.py:250-254` - Float conversions in `optimize()`
- `src/data_engine/api/alpaca_client.py` - `.df` attribute access

## SQL Injection Prevention

**Issue**: Direct string interpolation in SQL queries can cause syntax errors or security vulnerabilities.

**Solution**: Escape single quotes in user-provided strings before SQL interpolation:

```python
# For single symbol
escaped_symbol = symbol.replace("'", "''")
query = f"WHERE symbol = '{escaped_symbol}'"

# For multiple symbols
escaped_symbols = [symbol.replace("'", "''") for symbol in symbols]
symbols_str = "', '".join(escaped_symbols)
query = f"WHERE symbol IN ('{symbols_str}')"
```

**Where applied**:
- `src/backtesting/engine/data_loader.py:59` - `load_symbols()`
- `src/backtesting/engine/data_loader.py:133` - `get_date_range()`

## VectorBT DataFrame Type Requirements

**Issue**: VectorBT's Numba-compiled functions require precise numeric types. When concatenating DataFrames with misaligned indices, pandas introduces NaN values and converts dtypes to `object`, causing Numba typing errors.

**Error signature**:
```
numba.core.errors.TypingError: Failed in nopython mode pipeline
non-precise type array(pyobject, 2d, F)
```

**When it occurs**:
- Multi-symbol backtests where symbols have different trading times or data gaps
- Concatenating signal DataFrames (entries/exits) across symbols
- Concatenating price DataFrames with non-aligned timestamps

**Solution**: After concatenating DataFrames for VectorBT, always:

1. **For boolean signals (entries/exits)**: Fill NaN with False and convert to bool
```python
entries_df = pd.concat(all_entries, axis=1, keys=symbols)
entries_df = entries_df.fillna(False).astype(bool)

exits_df = pd.concat(all_exits, axis=1, keys=symbols)
exits_df = exits_df.fillna(False).astype(bool)
```

2. **For numeric data (prices)**: Forward-fill then backward-fill gaps
```python
prices_df = pd.concat(all_prices, axis=1, keys=symbols)
prices_df = prices_df.ffill().bfill()
```

**Why this works**:
- `fillna(False)` replaces NaN values with False (no signal at that timestamp)
- `.astype(bool)` ensures precise boolean type for Numba
- `ffill()` forward-fills price gaps (uses last known price)
- `bfill()` backward-fills any remaining gaps at the start

**Where applied**:
- `src/backtesting/engine/backtest_engine.py:100-101` - `_run_single_symbol()`
- `src/backtesting/engine/backtest_engine.py:139-141` - `_run_multiple_symbols()`
- `src/backtesting/engine/backtest_engine.py:182-185` - `run_with_data()`

**Prevention**: Always ensure DataFrames passed to VectorBT have:
- No NaN values (or handle them explicitly)
- Precise dtypes (bool for signals, float64 for prices)
- Aligned indices across all symbols

**Rule**: Before every call to `vbt.Portfolio.from_signals()`, clean the data:
```python
entries = entries.fillna(False).astype(bool)
exits = exits.fillna(False).astype(bool)
# For multi-symbol price DataFrames:
prices_df = prices_df.ffill().bfill()
```
