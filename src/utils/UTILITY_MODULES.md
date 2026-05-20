# Utility Modules

**Shared utilities for logging, timezone handling, VIX data, trade logging, and backtest result caching.**

**Last Updated**: 2026-05-17

---

## Overview

### What It Does
- Provides centralized colored logging via Rich console
- Handles timezone-aware timestamps for trading (ET default)
- Manages VIX data fetching with fallback providers
- Persists structured trade logs (JSON Lines) for live trading audit
- Caches backtest results and configurations for quick retrieval

### Key Features
- **Colored Logging**: Rich-based console output with semantic colors
- **Timezone Management**: Consistent ET (Eastern Time) handling
- **VIX Provider**: Multiple fallback sources for VIX data
- **TradeLogWriter**: Append-only JSON Lines trade log with per-day rotation
- **Backtest CacheManager**: Persistent on-disk cache keyed by config hash

### Use Cases
- Log trading events with color-coded output
- Get consistent timestamps across platforms (Windows, macOS, EC2)
- Fetch VIX data reliably for strategy calculations
- Record every live entry/exit with realized P&L for later analysis
- Skip re-running an identical backtest by loading the cached results

---

## Architecture

```
src/utils/
|-- __init__.py              # Exports the logger surface
|-- logger.py                # Rich-based colored logging + file logging
|-- timezone.py              # ET timezone handling (tz singleton + helpers)
|-- vix_provider.py          # VIX data with fallbacks (yfinance -> FRED -> cache)
|-- cache_manager.py         # Persistent backtest-result cache (config-hashed)
`-- trading_logger.py        # Live-trading log setup + TradeLogWriter (JSONL)
```

### Design Philosophy

1. **Single Source of Truth**: All console logging through `logger.py`
2. **Timezone Safety**: ET-aware timestamps for trading
3. **Resilience**: Multiple fallbacks for external data
4. **No Print Statements**: Use logger, never `print()`
5. **Logging Never Blocks Trading**: TradeLogWriter swallows its own I/O errors

---

## Key Components

### Logger (`logger.py`)

**Purpose**: Centralized colored logging using Rich console.

**Styles**:
| Method | Color | Prefix | Use For |
|--------|-------|--------|---------|
| `success()` | Green | `[+]` | Successful operations |
| `profit()`  | Green | `[^]` | Profitable trades |
| `error()`   | Red   | `[X]` | Errors |
| `loss()`    | Red   | `[v]` | Losing trades |
| `warning()` | Yellow | `[!]` | Warnings |
| `info()`    | Cyan  | `[i]` | Information |
| `header()`  | Magenta | None | Section headers |
| `metric()`  | Blue  | None | Statistics |
| `neutral()` | White | None | Neutral text |
| `dim()`     | Dim   | None | Secondary info |

**Usage**:
```python
from src.utils.logger import logger, get_logger

logger.success("Trade executed successfully")
logger.error("Failed to load data")
logger.info("Loading symbols...")
logger.header("=" * 60)
logger.metric(f"Sharpe Ratio: {sharpe:.2f}")

# Module-scoped logger
mod_logger = get_logger(__name__)

# File logging via Logger class
from pathlib import Path
from src.utils.logger import Logger
file_logger = Logger(log_file=Path("output/log.txt"))
```

**Module-Level Convenience Functions** (re-exported from `src.utils`):
```python
from src.utils import success, error, warning, info

success("Done!")
error("Failed!")
```

### TimezoneManager (`timezone.py`)

**Purpose**: Consistent ET (Eastern Time) handling for trading applications.

**Key Methods (via the `tz` singleton)**:
- `tz.now()`: Current datetime in ET
- `tz.today()`: Today's date in ET
- `tz.timestamp()`: Formatted timestamp string
- `tz.iso_timestamp()`: ISO 8601 timestamp
- `tz.date_str()`: Date for filenames (YYYYMMDD)
- `tz.from_utc(dt)`: Convert UTC to ET
- `tz.to_utc(dt)`: Convert ET to UTC

**Critical for**:
- EC2 instances running in UTC
- Consistent log timestamps
- Market hours calculations

**Usage**:
```python
from src.utils.timezone import tz

# Get current time in ET
now = tz.now()
print(now)  # 2026-05-15 15:50:23-04:00

# Format for logging
timestamp = tz.timestamp()      # "2026-05-15 15:50:23"
iso = tz.iso_timestamp()        # "2026-05-15T15:50:23.123456-04:00"
date = tz.date_str()            # "20260515"

# Convert UTC to ET
from datetime import datetime
utc_dt = datetime.utcnow()
et_dt = tz.from_utc(utc_dt)

# Validate DataFrame timezone
from src.utils.timezone import assert_et_timezone
assert_et_timezone(df, "broker output")  # Raises if not ET
```

**Convenience Functions**:
```python
from src.utils.timezone import now, today, timestamp, from_utc

current = now()
date = today()
ts = timestamp()
```

### VIX Provider (`vix_provider.py`)

**Purpose**: Fetch VIX data with multiple fallback sources.

**Fallback Chain**:
1. yfinance `^VIX` (primary, real-time)
2. FRED `VIXCLS` (fallback, end-of-day, official CBOE)
3. Persisted cache (last resort, stale but better than nothing)

**Features**:
- Automatic retry on failure
- Persistent JSON cache of last-known-good values
- Configurable max-age before warning on stale cache

**Usage**:
```python
from src.utils.vix_provider import VIXProvider

provider = VIXProvider()

# Historical / lookback window
vix_df = provider.get_vix_data(lookback_days=252)

# Latest spot VIX
current_vix = provider.get_current_vix()
print(f"VIX: {current_vix:.2f}")
```

### Backtest CacheManager (`cache_manager.py`)

**Purpose**: Persistent on-disk cache for backtest results, keyed by a stable hash of the config dictionary.

**Cache layout** (defaults to an OS-specific cache directory under `Homeguard/backtests/`):
```
cache_dir/
  configs/<config_hash>.json     # Config metadata
  results/<config_hash>.pkl      # Pickled results DataFrame
  portfolios/<config_hash>_<symbol>.pkl  # Optional pickled portfolios
  metadata.json                  # Index of cached runs
```

**Key Methods**:
- `cache_results(config, results_df, portfolios=None, description="")` -> `config_hash`
- `get_cached_results(config)` -> dict with `config`, `results_df`, `portfolios`, or `None`
- `get_cached_results_by_hash(config_hash)` -> same shape
- `is_cached(config)` -> `bool`
- `list_cached_runs(limit=50)` -> list of run metadata dicts
- `clear_cache(older_than_days=None)` -> count cleared
- `get_cache_size()` -> dict with `total_size_mb`, `file_count`, `num_cached_runs`, `cache_dir`
- `get_last_run_settings()` -> dict reconstructing the most recent run's config

**Usage**:
```python
from src.utils.cache_manager import CacheManager

cache = CacheManager()

config = {
    "strategy_class": MovingAverageCrossover,
    "strategy_params": {"fast_period": 10, "slow_period": 50},
    "symbols": ["SPY"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 100_000,
}

if cache.is_cached(config):
    cached = cache.get_cached_results(config)
    results_df = cached["results_df"]
else:
    results_df = run_backtest(config)  # your runner
    cache.cache_results(config, results_df, description="MA 10/50 on SPY")

print(cache.get_cache_size())
for run in cache.list_cached_runs(limit=5):
    print(run["timestamp"], run["strategy"], run["date_range"])
```

> Note: this cache is specifically for backtest results -- it is NOT a generic in-memory TTL cache.
> Use functools / a dict if you need an in-process TTL cache; we don't currently ship one in `src/utils/`.

### Trading Logger (`trading_logger.py`)

Two distinct surfaces live in this module:

#### 1. Live-trading log setup (file + console handlers)

```python
from src.utils.trading_logger import setup_trading_logs, cleanup_old_logs

main_logger, exec_logger = setup_trading_logs(
    log_dir="/home/ec2-user/logs",
    log_level="INFO",
)

main_logger.info("Bot started")
exec_logger.info("BUY TQQQ 100 @ $45.32")

# Periodically clean old logs
cleanup_old_logs(keep_days=30)
```

`get_trading_logger()` and `get_execution_logger()` are the individual factories. Both write to rotating files with EST timestamps (via `ESTFormatter`).

#### 2. `TradeLogWriter` -- structured JSONL trade log

Writes one JSON object per line to `trades_YYYYMMDD.jsonl` for downstream
P&L analysis.

```python
from src.utils.trading_logger import (
    TradeLogWriter,
    get_trade_log_writer,
    read_trade_log,
    compute_lifetime_realized_pnl,
)

# Singleton
trade_logger = get_trade_log_writer()  # uses /home/ec2-user/logs by default

trade_logger.log_entry(
    strategy="ramp",
    symbol="AAPL",
    qty=50,
    price=188.42,
    order_id="ibkr-12345",
    order_type="market",
    metadata={"regime": "WEAK_BULL", "rank": 3},
)

trade_logger.log_exit(
    strategy="ramp",
    symbol="AAPL",
    qty=50,
    exit_price=190.10,
    order_id="ibkr-12346",
    entry_price=188.42,
    entry_time="2026-05-14T15:55:00-04:00",
)

# Read back today's log
trades_today = read_trade_log()                 # list[dict]
trades_specific = read_trade_log("20260514")

# Lifetime realized P&L for a strategy (cached per-file by mtime)
ramp_pnl = compute_lifetime_realized_pnl("ramp")
```

`TradeLogWriter` swallows its own I/O errors -- a write failure logs via the
system logger but never raises, so a broken log disk cannot block live
trading.

---

## Data Flow

```
Application Code
        v
  Logger / tz / TradeLogWriter / CacheManager / VIXProvider
        v
  +----------------+--------------------------+
  | Console Output | File Output              |
  | (Rich colored) | (rotating .log / .jsonl) |
  +----------------+--------------------------+
                                v
                       CacheManager: pickle/json on disk
```

---

## Public API

### Logger Exports (re-exported from `src.utils`)

```python
from src.utils import (
    Logger,      # Class for file logging
    get_logger,  # Get logger instance
    success,
    profit,
    error,
    loss,
    warning,
    info,
    header,
    metric,
    neutral,
    dim,
    separator,
    blank,
)
```

### Timezone Exports

```python
from src.utils.timezone import (
    tz,                  # TimezoneManager instance
    now,                 # Current ET datetime
    today,               # Today's date
    timestamp,           # Formatted timestamp
    iso_timestamp,       # ISO timestamp
    date_str,            # Date string
    time_str,            # Time string
    datetime_str,        # Datetime string
    from_utc,            # Convert UTC to ET
    set_timezone,        # Change timezone
    ensure_et_index,     # Convert DataFrame to ET
    assert_et_timezone,  # Validate DataFrame timezone
)
```

### Trading Logger Exports

```python
from src.utils.trading_logger import (
    ESTFormatter,
    get_trading_logger,
    get_execution_logger,
    setup_trading_logs,
    cleanup_old_logs,
    TradeLogWriter,
    get_trade_log_writer,
    read_trade_log,
    compute_lifetime_realized_pnl,
)
```

### Cache Manager Exports

```python
from src.utils.cache_manager import CacheManager
```

---

## Configuration

### Logger Configuration

The logger is configured via Rich theme with forced terminal output for systemd compatibility:

```python
# Forced color output even in non-interactive terminals
console = Console(force_terminal=True)
```

### Timezone Configuration

Default timezone is `US/Eastern`. Can be changed:

```python
from src.utils.timezone import set_timezone
set_timezone("US/Pacific")  # Change to Pacific
```

### TradeLogWriter Configuration

The default `log_dir` (`/home/ec2-user/logs`) targets the EC2 host. Override it
when running locally:

```python
TradeLogWriter(log_dir="C:/tmp/homeguard-logs")
```

### CacheManager Configuration

By default the cache lives under an OS-appropriate cache directory
(`~/Library/Caches/Homeguard/backtests` on macOS, `%LOCALAPPDATA%/Temp/Homeguard/backtests`
on Windows, `~/.cache/Homeguard/backtests` on Linux). Override with
`CacheManager(cache_dir=Path(...))`.

---

## Dependencies

### Internal (src/ modules)
- `src.settings` - Directory paths
- `src.utils.timezone` - Used by `trading_logger.ESTFormatter` for EST timestamps

### External (pip packages)
- `rich` - Colored console output
- `pytz` - Timezone handling
- `pandas` - DataFrames (cache_manager, vix_provider)
- `yfinance` - Primary VIX source
- `requests` - FRED fallback for VIX

---

## Best Practices

### Logging

```python
# ALWAYS use logger, NEVER print()
from src.utils.logger import logger

# Good
logger.info("Loading data...")
logger.error("Failed to fetch data")

# Bad
print("Loading data...")  # NO!
```

### Timezone

```python
# ALWAYS use tz.now(), NEVER datetime.now()
from src.utils.timezone import tz

# Good
now = tz.now()

# Bad - gives UTC on EC2!
now = datetime.now()  # NO!
```

### Error Handling

```python
# ALWAYS log exceptions with logger.error()
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")  # Good
    # Don't silently swallow errors!
```

### Trade Logging

```python
# In live adapters, log every entry AND every exit so realized P&L is recoverable
from src.utils.trading_logger import get_trade_log_writer

trade_logger = get_trade_log_writer()
trade_logger.log_entry(strategy="ramp", symbol=sym, qty=qty, price=fill_price, order_id=oid)
# ... later ...
trade_logger.log_exit(strategy="ramp", symbol=sym, qty=qty, exit_price=exit_fill, order_id=oid2,
                      entry_price=entry_price, entry_time=entry_iso)
```

---

## Testing

### Test Location
- `tests/utils/` - Unit tests

### Running Tests
```bash
pytest tests/utils/ -v
```

---

## Related Documentation

- [Architecture Overview](../../docs/architecture/ARCHITECTURE_OVERVIEW.md)
- [CLAUDE.md - Logging Standards](../../CLAUDE.md)
- [Live Trading System](../trading/LIVE_TRADING_SYSTEM.md)

---

## Changelog

- **2026-05-17**: Corrected docs to match real APIs -- replaced fabricated `TradingLogger`/`CacheManager` (TTL) sections with the actual `TradeLogWriter` (JSONL) and backtest-result `CacheManager` APIs.
- **2025-12-08**: Initial documentation created
- **2025-11-XX**: TimezoneManager added
- **2025-10-XX**: VIX provider with fallbacks
- **2025-09-XX**: Initial logger module
