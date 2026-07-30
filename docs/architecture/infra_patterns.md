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
- **WARNING**: `OptionsDataLoader` renames `gamma_eod -> gamma` and
  `open_interest_eod -> open_interest`, stripping the `_eod` leak marker, and its
  `get_eod_chain()` hardcodes 16:00 while the registered snapshot is 15:45. Do
  not use it for slate work -- use the canonical layer below.

### Options Canonical + Derived Tables (`src/data/options/`)
All hive-partitioned as `root={SYMBOL}/year={YYYY}/month={MM}/data.parquet`
under `<storage>/options/`.

| Table | Module | Grain | Notes |
|---|---|---|---|
| `options_chain_eod` | `canonical.py` | contract x session | Snapshot at the registered **15:45 ET** minute, clamped to the real close on half-days. Marks are `mid` from valid quotes only; `close`/`vwap` are trade prints and are never marks. SPY 2017-01..2025-12, QQQ 2012-06..2025-12. |
| `options_iv_surface` | `iv_surface_build.py` | session x expiry | **M7.** Raw-SVI params, parity-implied forward/discount, fit diagnostics, and a REASON CODE. Refused slices are present with null params -- a refusal is a positive record, not a missing row. |
| `options_iv_smooth` | `iv_surface_build.py` | contract x session | **M7.** `iv_smooth`, `delta_smooth`, `iv_source`, `surface_reason`, `extrapolated`. Join to `options_chain_eod` on (session_date, expiry, strike, right). Required by P1 below \|delta\| 0.10. |

- **M7 fitter**: `src/data/options/iv_surface.py`. Dividend-awareness comes from
  the put-call-parity forward solved off the session's own quotes (point-in-time,
  no dividend series); the discount factor comes from
  `src/data/rates/fred_reader.py` (DGS1MO..DGS2). Registered choices and the
  D-047/030 gate threshold: `docs/strategies/research/options-slate/20260730_m7_prereg.md`.
- **Build**: `python scripts/data/build_m7_surface.py --jobs 8 --shard i/n`
- **Validate**: `python scripts/data/validate_m7_surface.py`

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

## Operations agents

Two agents handle EC2 / live-system interaction. They are deliberately split by capability:

- **`trade-log-analyzer`** -- diagnostics-only, read-only. Analyzes today's logs in ET, identifies errors, and *proposes* (does not implement) fixes. Best for "what went wrong today" questions. Never modifies state.
- **`live-ops`** -- routine ops with state-changing capability. Canned recipes for status, metrics, journal tails, instance start/stop, dashboard sync, service restarts. State changes require explicit user yes/no confirmation. Best for "do X to the system" tasks. Never modifies code, strategy configs, or trading state.

Both load EC2 identifiers from `.env` (`EC2_INSTANCE_ID`, `EC2_IP`, `EC2_USER`, `EC2_SSH_KEY_PATH`, `EC2_REGION`) rather than hardcoding. The strategy-pipeline agents (`strategy-lead` and its specialists) are separate from these ops agents -- they don't touch live infrastructure.
