# Skill: Implement RAMP Options Strategy

**Methodology**: Consult `docs/methodology/backtesting.md` Sections **1** (bias prevention -- every signal must use `.shift(1)` or equivalent; no full-sample statistics in features; no `bfill` on price data) and **7** (point-in-time data conventions -- fundamentals lag, news timestamps, index membership) before writing strategy code. Section **11** also applies when Section 11 lands and the strategy has any non-time-based exit (stops, targets, trailing rules): the trade log must include `mae_pct`, `mfe_pct`, `hit_stop`, `hit_target`, `exit_reason`, `bars_held` per 11.6.

## When to Use
When implementing a new options strategy that overlays the existing RAMP equity strategy.

## Two Strategy Patterns

### Pattern A: Signal-Based Equity Strategy
- Extend `BaseStrategy` / `LongOnlyStrategy` / `MultiSymbolStrategy` from `src/backtesting/base/strategy.py`
- Strategy code: `src/strategies/advanced/<name>.py`
- Config: `config/strategies/<name>.yaml`
- Backtest config: `config/backtesting/<name>.yaml`
- Tests: `tests/strategies/test_<name>.py`
- Run via: `python -m src.backtest_runner --config config/backtesting/<name>.yaml`

### Pattern B: Options Strategy (Use This for RAMP Options)
- No base class to extend - use callback-driven engine architecture
- Strategy code: `src/strategies/options/<type>/` (modular package)
- Config: `config/strategies/<name>.yaml`
- Tests: `tests/strategies/options/<type>/`
- Run via: Custom runner class

## Options Strategy File Layout

```
src/strategies/options/<type>/
    __init__.py              # Module docstring
    position.py              # Position + Trade dataclasses
    contract_selector.py     # Options chain filtering (delta, DTE, OI, spread)
    mark_to_market.py        # Price updates (chain lookup + Black-Scholes fallback)
    engine.py                # Event-driven backtest engine (callback architecture)
    metrics.py               # Strategy-specific metrics
    ramp_integration.py      # Wires RAMPSignals + MarketRegimeDetector into engine

config/strategies/<name>.yaml
    strategy:                # capital, allocation, profit/loss targets, slippage, fees
    contract_selection:      # delta range, DTE, OI, spread filters
    ramp:                    # VIX threshold, SPY drawdown threshold
    dates:                   # in_sample/out_of_sample ranges
    validation:              # thresholds for pass/fail

tests/strategies/options/<type>/
    __init__.py
    test_position.py
    test_contract_selector.py
    test_mark_to_market.py
    test_engine.py
    test_metrics.py
    test_ramp_integration.py
```

## Reference Implementation: CSP (ramp-csp)

See `src/strategies/options/csp/` for the complete reference.

### Callback Architecture Pattern
The engine does NOT own data. It receives callbacks:
- `get_regime(date) -> (str, float)` - market regime + confidence
- `get_crash_protection(date) -> bool` - RAMP crash protection active?
- `get_top_n_symbols(date) -> List[str]` - RAMP momentum ranking
- `get_chain(symbol, date) -> pd.DataFrame` - options chain data
- `get_underlying_price(symbol, date) -> float` - current stock price

### No Future Data Leakage
All callbacks filter with `<= d`: `spy_series[spy_series.index.date <= d]`

### RAMP Signal Integration
```python
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.ramp_strategy import RAMPSignals
```

### Options Data
```python
from src.strategies.options.data_loader import OptionsDataLoader
# Data layout: options_combined/root={SYMBOL}/year={YYYY}/month={MM}/data.parquet
# Use: loader.get_eod_chain(symbol, date) -> pd.DataFrame
```

### Equity Price Data
```python
from src.settings.settings import get_local_storage_dir
# Cache: get_local_storage_dir() / "equities_daily_cache.parquet"
```

## Key Conventions
- Logger: `from src.utils.logger import get_logger` then `logger = get_logger()`
- Never use print()
- ASCII only (no emojis, no Unicode)
- Environment: `fintech` conda environment
- Canonical OHLCV schema: timestamp, open, high, low, close, volume, trade_count, vwap
