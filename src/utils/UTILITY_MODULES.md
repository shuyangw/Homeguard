# Utility Modules

**Shared utilities for logging, timezone handling, caching, and data providers used across the application.**

**Last Updated**: 2025-12-08

---

## Overview

### What It Does
- Provides centralized colored logging via Rich console
- Handles timezone-aware timestamps for trading (ET default)
- Manages VIX data fetching with fallback providers
- Provides caching utilities for performance

### Key Features
- **Colored Logging**: Rich-based console output with semantic colors
- **Timezone Management**: Consistent ET (Eastern Time) handling
- **VIX Provider**: Multiple fallback sources for VIX data
- **Cache Manager**: Generic caching with TTL support

### Use Cases
- Log trading events with color-coded output
- Get consistent timestamps across platforms (Windows, macOS, EC2)
- Fetch VIX data reliably for strategy calculations
- Cache expensive computations

---

## Architecture

```
src/utils/
├── __init__.py              # Logger exports
├── logger.py                # Rich-based colored logging
├── timezone.py              # ET timezone handling
├── vix_provider.py          # VIX data with fallbacks
├── cache_manager.py         # Generic TTL caching
└── trading_logger.py        # Trade-specific logging
```

### Design Philosophy

1. **Single Source of Truth**: All logging through one module
2. **Timezone Safety**: ET-aware timestamps for trading
3. **Resilience**: Multiple fallbacks for external data
4. **No Print Statements**: Use logger, never print()

---

## Key Components

### Logger (`logger.py`)

**Purpose**: Centralized colored logging using Rich console.

**Styles**:
| Method | Color | Prefix | Use For |
|--------|-------|--------|---------|
| `success()` | Green | `[+]` | Successful operations |
| `profit()` | Green | `[^]` | Profitable trades |
| `error()` | Red | `[X]` | Errors |
| `loss()` | Red | `[v]` | Losing trades |
| `warning()` | Yellow | `[!]` | Warnings |
| `info()` | Cyan | `[i]` | Information |
| `header()` | Magenta | None | Section headers |
| `metric()` | Blue | None | Statistics |
| `neutral()` | White | None | Neutral text |
| `dim()` | Dim | None | Secondary info |

**Usage**:
```python
from src.utils.logger import logger

logger.success("Trade executed successfully")
logger.error("Failed to load data")
logger.info("Loading symbols...")
logger.header("=" * 60)
logger.metric(f"Sharpe Ratio: {sharpe:.2f}")

# With file logging
from src.utils.logger import Logger
file_logger = Logger(log_file=Path("output/log.txt"))
```

**Module-Level Functions**:
```python
from src.utils import success, error, warning, info

success("Done!")
error("Failed!")
```

### TimezoneManager (`timezone.py`)

**Purpose**: Consistent ET (Eastern Time) handling for trading applications.

**Key Methods**:
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
print(now)  # 2025-12-08 15:50:23-05:00

# Format for logging
timestamp = tz.timestamp()      # "2025-12-08 15:50:23"
iso = tz.iso_timestamp()        # "2025-12-08T15:50:23.123456-05:00"
date = tz.date_str()            # "20251208"

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
1. Yahoo Finance (yfinance)
2. CBOE direct download
3. Local cache (stale data)

**Features**:
- Automatic retry on failure
- Cache for resilience
- Mock VIX from SPY volatility as last resort

**Usage**:
```python
from src.utils.vix_provider import VIXProvider

provider = VIXProvider()
vix_df = provider.get_vix_data(
    start="2023-01-01",
    end="2024-01-01",
    use_cache=True
)

# Get latest VIX value
current_vix = provider.get_current_vix()
print(f"VIX: {current_vix:.2f}")
```

### Cache Manager (`cache_manager.py`)

**Purpose**: Generic TTL-based caching for expensive operations.

**Features**:
- In-memory caching with TTL
- Thread-safe operations
- Automatic expiration

**Usage**:
```python
from src.utils.cache_manager import CacheManager

cache = CacheManager(ttl_seconds=300)  # 5 minute TTL

# Store data
cache.set("key", expensive_computation())

# Retrieve
data = cache.get("key")
if data is None:
    data = expensive_computation()
    cache.set("key", data)

# Clear
cache.clear()
cache.delete("key")
```

### Trading Logger (`trading_logger.py`)

**Purpose**: Specialized logging for trade events and performance.

**Features**:
- Trade event logging with timestamps
- Performance metric tracking
- CSV export for trades

**Usage**:
```python
from src.utils.trading_logger import TradingLogger

trade_logger = TradingLogger(output_dir=Path("logs/"))

trade_logger.log_trade(
    symbol="AAPL",
    side="BUY",
    quantity=100,
    price=150.00,
    order_id="abc123"
)

trade_logger.log_metric("daily_pnl", 1500.00)
trade_logger.export_trades("trades.csv")
```

---

## Data Flow

```
Application Code
        ↓
  Logger / TimezoneManager / CacheManager
        ↓
  ┌────────────────┬─────────────────┐
  │ Console Output │ File Output     │
  │ (Rich colored) │ (Plain text)    │
  └────────────────┴─────────────────┘
```

---

## Public API

### Logger Exports

```python
from src.utils import (
    Logger,      # Class for file logging
    get_logger,  # Get logger instance
    success,     # Success message
    profit,      # Profit message
    error,       # Error message
    loss,        # Loss message
    warning,     # Warning message
    info,        # Info message
    header,      # Header message
    metric,      # Metric message
    neutral,     # Neutral message
    dim,         # Dim message
    separator,   # Separator line
    blank,       # Blank line
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
set_timezone('US/Pacific')  # Change to Pacific
```

---

## Dependencies

### Internal (src/ modules)
- `src.settings` - Directory paths

### External (pip packages)
- `rich` - Colored console output
- `pytz` - Timezone handling
- `yfinance` - VIX data (in vix_provider)

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

- **2025-12-08**: Initial documentation created
- **2025-11-XX**: TimezoneManager added
- **2025-10-XX**: VIX provider with fallbacks
- **2025-09-XX**: Initial logger module
