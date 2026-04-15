# Backtesting Best Practices

**CRITICAL**: When modifying, creating, or reviewing backtesting logic, ALWAYS refer to `backtest_guidelines/guidelines.md` for detailed guidance on avoiding common pitfalls.

## Required Reading Before Backtesting Work

Before making ANY changes to backtesting code, you MUST:

1. **Consult the Guidelines**: Review `backtest_guidelines/guidelines.md` to understand common pitfalls
2. **Avoid Lookahead Bias**: Ensure your code never uses future data to make past decisions
3. **Avoid Survivorship Bias**: Verify the dataset includes delisted/failed entities
4. **Avoid Overfitting**: Don't create overly complex strategies with too many parameters
5. **Validate Assumptions**: Double-check all timing, indexing, and data availability assumptions

## Backtesting Code Areas

The guidelines apply to any code in these areas:
- `src/backtesting/engine/` - Core backtesting engine
- `src/backtesting/utils/` - Backtesting utilities and indicators
- `src/strategies/` - Strategy implementations
- Any code that processes historical data for trading simulations

## When Making Changes

When editing backtesting logic:
1. First, identify which pitfalls from `backtest_guidelines/guidelines.md` are relevant
2. Verify your changes don't introduce lookahead bias, overfitting, or other issues
3. Test with edge cases mentioned in the guidelines
4. Document any assumptions about data timing or availability

**Remember**: A backtest that looks "too good to be true" usually is. Prioritize correctness over impressive-looking results.

## Performance Optimization

**CRITICAL**: When loading or processing large datasets (>10M rows), use these optimization techniques:

### Use Polars for Large File Processing

Polars is significantly faster than pandas for large datasets. Always prefer Polars for:
- Loading large parquet files (100M+ rows)
- Groupby operations on large datasets
- Data transformations before passing to backtest engine

```python
import polars as pl

# Fast parquet loading (10-20x faster than pandas)
df = pl.read_parquet('large_file.parquet')

# Efficient groupby with aggregation
grouped = df.group_by(['symbol', 'date']).agg([
    pl.col('close').alias('close_prices')
])

# Convert to pandas only when needed for compatibility
df_pandas = df.to_pandas()
```

### Key Polars Gotchas

- `partition_by('col', as_dict=True)` returns **tuple keys** like `('AAPL',)`, not strings
- Extract string: `symbol = key[0] if isinstance(key, tuple) else key`
- Use `maintain_order=True` when order matters for time-series data

### Avoid BashOutput Polling

When running long backtest jobs in background:
- Do NOT poll BashOutput every few seconds - this wastes context
- Let the job run to completion, then check output once
- Use `run_in_background=True` and wait for completion
- **Exception**: Check output if the user explicitly asks for status/progress

## Market Calendar and Trading Days

**CRITICAL**: Backtests only execute on NYSE trading days (excludes weekends, holidays, special closures).

- **DataLoader**: Auto-filters to trading days by default (use `filter_market_days=False` only for debugging)
- **Utilities**: `from backtesting.utils.market_calendar import MarketCalendar, is_trading_day`
- **No web requests**: `pandas_market_calendars` uses static calendar data

## Using Existing Backtesting Frameworks

**CRITICAL**: When asked to run a backtest, ALWAYS use the existing config-driven backtesting system rather than writing ad-hoc scripts.

### Config-Driven Backtesting (Preferred)

The project has a comprehensive config-driven backtesting system. Use it via:

```bash
# Run a backtest using a YAML config file
python -m src.backtest_runner --config config/backtesting/ma_single.yaml
```

### Available Backtest Modes

1. **Single Mode**: Run one backtest on one symbol
2. **Sweep Mode**: Run the same strategy across multiple symbols
3. **Optimize Mode**: Search for optimal strategy parameters
4. **Walk-Forward Mode**: Time-series cross-validation

### Config File Structure

Example config (`config/backtesting/ma_single.yaml`):

```yaml
mode: single

strategy:
  name: MovingAverageCrossover  # From strategy registry
  parameters:
    fast_window: 10
    slow_window: 50

symbols:
  list: [SPY]  # Or use: universe: "production.conservative"

dates:
  start: "2023-01-01"
  end: "2024-01-01"
  # Or use: preset: "full_periods.five_years"

backtest:
  initial_capital: 100000
  fees: 0.001
  slippage: 0.0005

risk:
  position_sizing_method: fixed_percent
  position_size_pct: 0.10
  max_positions: 5

output:
  save_trades: true
  quantstats: true
  visualize: true
```

### Key Modules

- **Config Schema**: `src/settings/schema.py` - Pydantic models for validation
- **Config Loader**: `src/settings/loader.py` - YAML loading with inheritance
- **Defaults**: `src/settings/defaults.py` - Default values, presets, universes
- **Runner**: `src/backtest_runner.py` - CLI entry point
- **Strategy Registry**: `src/strategies/registry.py` - Available strategies

### Symbol Universes

Available via `symbols.universe` in configs:
- `production.conservative`: SPY, QQQ, IWM
- `production.moderate`: SPY, QQQ, IWM, XLK, XLF
- `technology.mag7`: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- `leveraged.triple_long`: TQQQ, UPRO, SOXL, TECL, FAS
- See `src/settings/defaults.py` for full list

### Date Presets

Available via `dates.preset` in configs:
- `full_periods.five_years`: 2020-01-01 to 2024-12-31
- `regimes.covid_crash`: 2020-02-01 to 2020-04-30
- `optimization.fast_test`: 2023-01-01 to 2024-12-31
- See `src/settings/defaults.py` for full list

### When to Create New Configs vs Use Existing

- **Use existing configs**: For standard backtests with minor parameter tweaks
- **Create new configs**: For new strategy/symbol combinations or special test scenarios
- **Store configs in**: `configs/examples/` for reusable examples, `configs/` for one-off tests
