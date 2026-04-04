# Homeguard Infrastructure Patterns

**Date**: 2026-03-30
**Purpose**: Reference for implementing new strategies. Created during strategy #1 (ramp-csp) pipeline.

---

## Strategy Implementation Patterns

### Pattern A: Signal-Based Equity Strategies (OMR, momentum, mean-reversion)

**Base class**: Extend `BaseStrategy`, `LongOnlyStrategy`, `LongShortStrategy`, or `MultiSymbolStrategy` from `src/backtesting/base/strategy.py`

**File layout**:
- Strategy code: `src/strategies/advanced/<name>.py`
- Config: `config/strategies/<name>.yaml` (strategy params)
- Backtest config: `config/backtesting/<name>.yaml` (backtest params: capital, fees, dates, etc.)
- Tests: `tests/strategies/test_<name>.py`
- Backtest script: `scripts/backtest_scripts/<name>_backtest.py` (one-off, gitignored)

**Signal flow**:
```
Strategy.generate_signals(data) -> (entries: pd.Series[bool], exits: pd.Series[bool])
  |
  v
BacktestEngine.run(strategy, symbols, start_date, end_date)
  |
  v
Portfolio (from_signals) -> equity_curve, trades, stats
```

**Running**: `python -m src.backtest_runner --config config/backtesting/<name>.yaml`

### Pattern B: Options Strategies (CSP, covered calls, wheels)

**No base class**: Options strategies use callback-driven engines because they have fundamentally different mechanics (no simple boolean entry/exit signals).

**File layout**:
- Strategy code: `src/strategies/options/<type>/` (modular: engine, position, selector, metrics, integration)
- Config: `config/strategies/<name>.yaml`
- Tests: `tests/strategies/options/<type>/`
- Data loader: `src/strategies/options/data_loader.py` (shared options data loader)

**Signal flow**:
```
Integration Runner (wires data + RAMP signals)
  |
  +--> Callbacks: get_regime(), get_crash_protection(), get_top_n_symbols(), get_chain(), get_underlying_price()
  |
  v
Options Engine (event-driven, day-by-day)
  |
  +--> For each day: manage exits, scan entries, record daily snapshot
  |
  v
Result (closed_trades, daily_snapshots, equity_curve)
```

**Running**: Via custom runner class (e.g., `CSPBacktestRunner.run(start_date, end_date)`)

---

## Data Infrastructure

### Equity Prices
- **Source**: Alpaca API via `scripts/data/download_symbols.py`
- **Storage**: `get_local_storage_dir()` from `src/settings`
- **Cache**: `equities_daily_cache.parquet` for daily data
- **Canonical schema**: timestamp, open, high, low, close, volume, trade_count, vwap (lowercase, float64)
- **Loader**: `StreamingDataLoader` from `src/backtesting/engine/streaming_data_loader.py`

### Options Data
- **Source**: ThetaData API via `src/data/options/thetadata_client.py`
- **Storage**: Hive-partitioned parquet: `options_combined/root={SYMBOL}/year={YYYY}/month={MM}/data.parquet`
- **Loader**: `OptionsDataLoader` from `src/strategies/options/data_loader.py`
- **Key columns**: strike, expiry, delta, bid, ask, mid_price, implied_vol, open_interest, days_to_expiry, option_type, underlying_price

### Symbol Universes
- Location: `config/universes/`
- Files: `sp500-2025.csv`, `russell1000-2025.csv`, `russell2000-2025.csv`
- Format: CSV with `Symbol` column

---

## Backtesting Infrastructure

### General Equity Engine: `src/backtesting/engine/backtest_engine.py`
- `BacktestEngine(initial_capital, fees, slippage, freq, market_hours_only, benchmark, risk_config, enable_regime_analysis, allow_shorts, timeframe)`
- Modes: single, multi_asset, rolling
- Uses `Portfolio` (bar-by-bar simulation with optional Numba JIT)

### Risk Management: `src/backtesting/utils/`
- `RiskConfig`: position sizing configuration
- `PositionSizer`: FixedPercentage, FixedDollar, VolatilityBased, KellyCriterion
- `RiskManager`: stop-loss, trailing stops, time stops

### Optimization: `src/backtesting/optimization/`
- Grid search, random search, Bayesian, genetic
- `WalkForwardOptimizer`: rolling train/test with IS/OOS gap analysis
- Built for equity `BacktestEngine` - options strategies need custom wrappers

### Walk-Forward: `src/backtesting/chunking/walk_forward.py`
- Rolling window splitting utilities

### Regime Detection
- **5-regime (RAMP)**: `src/strategies/advanced/market_regime_detector.py` -> STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR
- **3-regime (generic)**: `src/backtesting/regimes/detector.py` -> Bull, Bear, Sideways + volatility + drawdown regimes

### Reporting: `src/backtesting/reporting/standard_report.py`
- `StandardReportGenerator`: monthly breakdown, overall Sharpe/drawdown
- Output formats: console, markdown, CSV

---

## Configuration Patterns

### Strategy Config (`config/strategies/<name>.yaml`)
Strategy-specific parameters. Read by the strategy or its runner.

### Backtest Config (`config/backtesting/<name>.yaml`)
Backtest execution parameters. Read by `src.backtest_runner`.
```yaml
mode: single
strategy:
  name: StrategyClassName
  parameters: {key: value}
symbols:
  list: [SYM1, SYM2]
dates:
  start: "YYYY-MM-DD"
  end: "YYYY-MM-DD"
backtest:
  initial_capital: 100000
  fees: 0.001
  slippage: 0.0005
risk:
  enabled: true
  position_sizing_method: fixed_percent
  position_size_pct: 0.10
output:
  save_trades: true
  save_reports: true
```

---

## Logging and Environment

- **Logger**: `from src.utils.logger import logger` or `get_logger()`
- **Never use print()** - always logger
- **Environment**: `fintech` conda environment for all Python execution
- **Platform**: Windows (cp1252 encoding) - ASCII only in all code and docs
