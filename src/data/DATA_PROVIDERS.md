# Data Providers

**Market data acquisition layer with provider abstraction, fallback chains, and persistent caching.**

**Last Updated**: 2025-12-08

---

## Overview

### What It Does
- Provides historical OHLCV market data from multiple sources
- Implements fallback chain pattern for data resilience
- Caches data persistently to reduce API calls
- Downloads bulk data from Alpaca API with parallel processing

### Key Features
- **Provider Abstraction**: `DataProviderInterface` for any data source
- **Fallback Chain**: Try Alpaca -> yfinance -> cache automatically
- **Cache-First Strategy**: Check cache before API calls for intraday
- **Bulk Downloads**: Multi-threaded downloader with retry logic
- **Schema Enforcement**: Standardized lowercase columns, ET timezone

### Use Cases
- Fetch historical data for backtesting
- Provide market data to live trading strategies
- Download and maintain local data archive
- Handle API failures gracefully with cached data

---

## Architecture

```
src/data/
├── __init__.py                  # Public API: News, Sentiment
├── acquisition/                 # Unified data acquisition (replaces downloader.py)
│   ├── __init__.py              # DataAcquisitionManager, DownloadResult
│   ├── base.py                  # BaseDownloader with threading, retry, storage
│   ├── manager.py               # Orchestrator for all data downloads
│   ├── schemas.py               # Canonical schema definitions
│   ├── manifest.py              # Download tracking manifest
│   └── plugins/                 # Source-specific plugins
│       ├── alpaca_equities.py   # Equity OHLCV from Alpaca
│       ├── alpaca_crypto.py     # Crypto OHLCV from Alpaca
│       ├── alpaca_news.py       # News from Alpaca
│       └── databento_futures.py # Futures from Databento
└── providers/
    ├── __init__.py              # Provider exports
    ├── base.py                  # DataProviderInterface abstract class
    ├── alpaca.py                # Alpaca data provider
    ├── yfinance.py              # yfinance fallback provider
    ├── composite.py             # Fallback chain orchestrator
    ├── cache.py                 # Persistent parquet cache
    └── factory.py               # Provider factory
```

### Design Philosophy

1. **Interface-Based**: All providers implement `DataProviderInterface`
2. **Graceful Degradation**: Return `None` on failure, don't raise exceptions
3. **Cache as Safety Net**: Stale data better than no data for trading
4. **Schema Standardization**: All providers return same format (lowercase columns, ET timezone)
5. **Composition**: CompositeProvider chains providers with caching

---

## Key Components

### DataProviderInterface (`providers/base.py`)

**Purpose**: Abstract interface for all market data providers.

**Contract**:
- Index: `DatetimeIndex` with `America/New_York` timezone
- Columns: `open`, `high`, `low`, `close`, `volume` (lowercase)
- Returns: `pd.DataFrame` or `None` on failure

**Key Methods**:
- `get_historical_bars(symbol, start, end, timeframe)`: Single symbol
- `get_historical_bars_batch(symbols, start, end, timeframe)`: Multiple symbols
- `name`: Provider name for logging
- `is_available()`: Check provider availability
- `supports_timeframe(timeframe)`: Check timeframe support

**Usage**:
```python
from src.data.providers.base import DataProviderInterface

class MyProvider(DataProviderInterface):
    @property
    def name(self) -> str:
        return "MyProvider"

    def get_historical_bars(self, symbol, start, end, timeframe='1D', force_refresh=False):
        # Fetch data, return DataFrame or None
        pass

    def get_historical_bars_batch(self, symbols, start, end, timeframe='1D', force_refresh=False):
        # Fetch batch, return Dict[symbol, DataFrame]
        pass
```

### CompositeDataProvider (`providers/composite.py`)

**Purpose**: Orchestrates fallback chain with caching.

**Key Features**:
- Try providers in priority order
- Cache-first for intraday data (reduces API calls)
- Stale cache as last resort
- Track data source for logging
- Support `force_refresh` to bypass cache at execution time

**Usage**:
```python
from src.data.providers import AlpacaDataProvider, YFinanceDataProvider
from src.data.providers.composite import CompositeDataProvider

providers = [AlpacaDataProvider(broker), YFinanceDataProvider()]
composite = CompositeDataProvider(
    providers,
    cache_enabled=True,
    cache_first_for_intraday=True,
    intraday_cache_ttl_minutes=5
)

# Will try Alpaca first, then yfinance, then cache
df = composite.get_historical_bars('TQQQ', start, end, '1Min')

# At execution time, force fresh data
df = composite.get_historical_bars('TQQQ', start, end, '1Min', force_refresh=True)

# Check where data came from
source, fetch_time = composite.get_source_info()
print(f"Data from: {source}")  # "Alpaca", "yfinance", or "cache"
```

### DataCache (`providers/cache.py`)

**Purpose**: Persistent parquet-based cache for market data.

**Storage Layout**:
```
{storage_dir}/cache/market_data/
├── metadata.json        # Cache index with timestamps
├── daily/
│   ├── AAPL.parquet     # Daily bars per symbol
│   └── SPY.parquet
└── intraday/
    ├── TQQQ_1Min.parquet    # Intraday by symbol+timeframe
    └── SOXL_1Min.parquet
```

**Features**:
- Parquet format for efficient storage
- TTL-based expiration with stale fallback
- Metadata tracking (timestamp, row count, date range)
- Per-symbol and per-timeframe clearing

**Usage**:
```python
from src.data.providers.cache import DataCache

cache = DataCache()  # Uses default storage dir

# Store data
cache.store('AAPL', '1D', df)

# Retrieve (warns if stale)
df = cache.retrieve('AAPL', '1D', max_age_hours=24)

# Get cache stats
stats = cache.get_stats()
# {'daily_files': 10, 'intraday_files': 25, 'metadata_entries': 35}

# Clear cache
cache.clear(symbol='AAPL')  # Clear one symbol
cache.clear()  # Clear all
```

### DataAcquisitionManager (`acquisition/manager.py`)

**Purpose**: Unified entry point for all data downloads (replaces legacy AlpacaDownloader).

**Features**:
- Plugin-based architecture (equities, crypto, futures, news)
- Multi-threaded downloads (default 6 threads)
- Retry logic with exponential backoff (3 retries/symbol)
- End-of-run retry rounds (3 rounds for failures)
- Skip-existing support
- Canonical 8-column schema enforcement
- Hive partitioned output format
- Download manifest tracking

**Usage**:
```python
from src.data.acquisition import DataAcquisitionManager

manager = DataAcquisitionManager()
result = manager.download(
    source="equities",
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2017-01-01",
    end_date="2024-12-31",
    skip_existing=True,
    num_threads=6,
)

print(f"Downloaded {result.total_rows} rows")
print(f"Success rate: {result.success_rate:.1f}%")
print(f"Failed: {result.failed_symbols}")
```

**Command Line**:
```bash
# Download equities from CSV
python scripts/download_symbols.py --csv config/universes/sp500-2025.csv --skip-existing

# Download specific symbols
python scripts/download_symbols.py --symbols AAPL,MSFT,GOOGL

# Download crypto
python scripts/download_crypto.py --skip-existing

# Unified CLI
python -m src.data.acquisition --source equities --symbols AAPL,MSFT --start 2020-01-01
python -m src.data.acquisition --status
```

---

## Data Flow

```
API Request (symbol, timeframe, date range)
        v
┌────────────────────────────────────────┐
│        CompositeDataProvider           │
├────────────────────────────────────────┤
│  1. Check cache (if cache-first mode)  │
│  2. Try AlpacaDataProvider             │
│  3. Try YFinanceDataProvider           │
│  4. Return stale cache as last resort  │
└────────────────────────────────────────┘
        v
  Cache successful result
        v
  Return DataFrame (or None on total failure)
```

---

## Public API

### Acquisition Exports

```python
from src.data.acquisition import DataAcquisitionManager, DownloadResult

# Download data
manager = DataAcquisitionManager()
result = manager.download(source="equities", symbols=["AAPL"], start_date="2020-01-01")
```

### Provider Exports

```python
from src.data.providers import (
    DataProviderInterface,
    AlpacaDataProvider,
    YFinanceDataProvider,
    CompositeDataProvider,
    DataCache
)
```

---

## Configuration

### Storage Paths (from settings.ini)

```ini
[storage]
local_storage_dir = F:\Stock_Data  # Windows
# local_storage_dir = /Users/shuyangw/Library/CloudStorage/Dropbox/cs/stonk/data  # macOS
# local_storage_dir = /home/ec2-user/stock_data  # EC2
```

### Directory Structure

```
{local_storage_dir}/
├── equities_1min/
│   └── symbol={SYMBOL}/
│       └── year={YYYY}/
│           └── month={MM}/
│               └── data.parquet
├── equities_1hour/
│   └── (same structure)
├── equities_1day/
│   └── (same structure)
└── cache/
    └── market_data/
        ├── metadata.json
        ├── daily/
        └── intraday/
```

### Canonical Schema

All data MUST follow this schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `datetime64[us, UTC]` | Bar timestamp |
| `open` | `float64` | Opening price |
| `high` | `float64` | High price |
| `low` | `float64` | Low price |
| `close` | `float64` | Closing price |
| `volume` | `float64` | Volume traded |
| `trade_count` | `float64` | Number of trades |
| `vwap` | `float64` | Volume-weighted avg price |

---

## Dependencies

### Internal (src/ modules)
- `src.api_key` - Alpaca credentials
- `src.settings` - Storage paths
- `src.utils.logger` - Logging
- `src.utils.timezone` - Timezone utilities

### External (pip packages)
- `alpaca-py` - Alpaca API client
- `yfinance` - Yahoo Finance fallback
- `pandas` - DataFrames
- `pyarrow` - Parquet I/O

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `DataProviderError` | Generic data fetch error | Try fallback provider |
| `SymbolNotFoundError` | Symbol doesn't exist | Skip symbol |
| `DataUnavailableError` | API temporarily down | Use cached data |
| Rate limit exceeded | Too many API calls | Use cache-first mode |

---

## Testing

### Test Location
- `tests/data/` - Unit tests

### Running Tests
```bash
pytest tests/data/ -v
```

---

## Related Documentation

- [Architecture Overview](../../docs/architecture/ARCHITECTURE_OVERVIEW.md)
- [Module Reference](../../docs/architecture/MODULE_REFERENCE.md)
- [Data Acquisition](acquisition/__init__.py)
- [CLAUDE.md - Data Handling](../../CLAUDE.md)

---

## Changelog

- **2025-12-08**: Initial documentation created
- **2025-11-XX**: CompositeDataProvider with cache-first
- **2025-10-XX**: AlpacaDownloader with parallel downloads
- **2025-09-XX**: Initial provider abstraction
