# MarketDataProvider Abstraction Layer - Design Document

**Date**: 2025-12-05
**Status**: Proposed
**Author**: Claude Code

---

## Executive Summary

This document describes a new `MarketDataProvider` abstraction layer that decouples market data fetching from trading adapters. The design follows the proven `VIXProvider` pattern (multi-source fallback with caching) and maintains full backward compatibility.

### Problem Statement

The OMR strategy has no trades because:
1. **Alpaca IEX feed only covers ~2% of market volume**
2. **Leveraged ETFs (TECL, FAS, TNA, LABU, etc.) have sparse/missing intraday data on IEX**
3. **yfinance provides complete coverage but is tightly coupled in adapters**

### Validation Data (Dec 5, 2025)

| Symbol | Alpaca IEX Bars | yfinance Bars | IEX Coverage |
|--------|-----------------|---------------|--------------|
| TECL | 15 | 352 | 4.3% |
| FAS | 6 | 279 | 2.2% |
| TNA | 87 | 390 | 22.3% |
| LABU | 28 | 365 | 7.7% |
| TQQQ | 138 | 390 | 35.4% |
| SOXL | 142 | 390 | 36.4% |

**Key Finding**: IEX has no morning data before 1:00 PM for most leveraged ETFs. The OMR strategy requires 9:30 AM open prices.

---

## Goals

1. **Backward Compatible** - Existing code works unchanged
2. **Follow VIXProvider Pattern** - Multi-source fallback with caching
3. **Decouple yfinance** - Remove direct imports from trading adapters
4. **Opt-in Migration** - New provider is optional, not forced

---

## Architecture

### Current Data Flow

```
┌─────────────────┐     ┌─────────────────┐
│  OMR Strategy   │     │  MP Strategy    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ OMRLiveAdapter  │     │ MPLiveAdapter   │
│                 │     │ import yfinance │  <- Tight coupling
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│           AlpacaBroker                   │
│  - get_historical_bars() -> IEX only     │
│  - ~2% market coverage                   │
└─────────────────────────────────────────┘
```

### Proposed Data Flow

```
┌─────────────────┐     ┌─────────────────┐
│  OMR Strategy   │     │  MP Strategy    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│      StrategyAdapter (base class)        │
│  - Optional: data_provider parameter     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│      CompositeDataProvider (NEW)         │
│  - Fallback chain: Alpaca -> yfinance     │
│  - Persistent cache as last resort       │
│  - Follows VIXProvider pattern           │
└────────┬───────────────────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ AlpacaProvider  │     │ YFinanceProvider│
│  (wraps broker) │     │  (new impl)     │
└─────────────────┘     └─────────────────┘
```

---

## File Structure

```
src/data/providers/
├── __init__.py           # Package exports
├── base.py               # DataProviderInterface abstract class
├── alpaca.py             # AlpacaDataProvider (wraps AlpacaBroker)
├── yfinance.py           # YFinanceDataProvider (new)
├── composite.py          # CompositeDataProvider (fallback chain)
├── cache.py              # DataCache (persistent parquet storage)
└── factory.py            # create_data_provider() factory

config/
└── data_providers.yaml   # Provider configuration

tests/data/providers/
├── test_base.py          # Interface contract tests
├── test_alpaca.py        # Alpaca provider tests
├── test_yfinance.py      # yfinance provider tests
├── test_composite.py     # Fallback chain tests
└── test_cache.py         # Cache tests
```

---

## Interface Definitions

### DataProviderInterface (base.py)

```python
"""
Data Provider Interface - Abstract base for all market data providers.

All implementations must:
1. Return data in standardized format (lowercase columns, ET timezone)
2. Handle errors gracefully (return None, don't raise)
3. Support both single-symbol and batch operations
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class DataProviderInterface(ABC):
    """
    Abstract interface for market data providers.

    Contract:
    - Index: DatetimeIndex with America/New_York timezone
    - Columns: open, high, low, close, volume (lowercase)
    - Returns: pd.DataFrame or None on failure (enables fallback)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging (e.g., 'Alpaca', 'yfinance')."""
        pass

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV bars for a single symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TQQQ')
            start: Start datetime (timezone-aware recommended)
            end: End datetime (timezone-aware recommended)
            timeframe: '1D', '1Min', '5Min', '1Hour'

        Returns:
            DataFrame with OHLCV data, or None on failure
        """
        pass

    @abstractmethod
    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical bars for multiple symbols.

        Args:
            symbols: List of symbols
            start: Start datetime
            end: End datetime
            timeframe: Timeframe string

        Returns:
            Dict mapping symbol -> DataFrame (empty dict on total failure)
        """
        pass

    def is_available(self) -> bool:
        """Check if provider is currently available."""
        return True

    def supports_timeframe(self, timeframe: str) -> bool:
        """Check if provider supports the given timeframe."""
        return True


class DataProviderError(Exception):
    """Base exception for data provider errors."""
    pass


class SymbolNotFoundError(DataProviderError):
    """Raised when a symbol cannot be found."""
    pass


class DataUnavailableError(DataProviderError):
    """Raised when data is temporarily unavailable."""
    pass
```

### AlpacaDataProvider (alpaca.py)

```python
"""
Alpaca Data Provider - Wraps existing AlpacaBroker.

This is a thin adapter that delegates to AlpacaBroker without modifying it.
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from src.data.providers.base import DataProviderInterface
from src.trading.brokers.alpaca_broker import AlpacaBroker
from src.utils.logger import logger


class AlpacaDataProvider(DataProviderInterface):
    """
    Data provider wrapping AlpacaBroker.

    No changes to AlpacaBroker are required - this is a pure wrapper.
    """

    def __init__(self, broker: AlpacaBroker):
        """
        Initialize with existing broker instance.

        Args:
            broker: Configured AlpacaBroker instance
        """
        self._broker = broker

    @property
    def name(self) -> str:
        return "Alpaca"

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Optional[pd.DataFrame]:
        """Fetch bars via AlpacaBroker, returning None on failure."""
        try:
            df = self._broker.get_historical_bars(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=timeframe
            )

            if df is None or df.empty:
                logger.warning(f"[Alpaca] No data for {symbol}")
                return None

            # AlpacaBroker already returns correct format
            return df

        except Exception as e:
            logger.error(f"[Alpaca] Failed to fetch {symbol}: {e}")
            return None

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Dict[str, pd.DataFrame]:
        """Batch fetch - delegates to single-symbol calls."""
        results = {}

        for symbol in symbols:
            df = self.get_historical_bars(symbol, start, end, timeframe)
            if df is not None:
                results[symbol] = df

        return results

    def is_available(self) -> bool:
        """Check Alpaca connection."""
        try:
            return self._broker.test_connection()
        except Exception:
            return False
```

### YFinanceDataProvider (yfinance.py)

```python
"""
YFinance Data Provider - Fetches data from Yahoo Finance.

Handles yfinance quirks:
- MultiIndex columns for multi-symbol downloads
- Capitalized column names ('Close' -> 'close')
- Timezone normalization to Eastern Time
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from src.data.providers.base import DataProviderInterface
from src.utils.logger import logger


class YFinanceDataProvider(DataProviderInterface):
    """
    Data provider using yfinance (Yahoo Finance).

    Normalizes output to standard format:
    - Lowercase column names
    - Eastern Time timezone
    - Single-level column index
    """

    @property
    def name(self) -> str:
        return "yfinance"

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Optional[pd.DataFrame]:
        """Fetch bars via yfinance."""
        try:
            import yfinance as yf

            # Convert dates to yfinance format
            start_str = start.strftime('%Y-%m-%d')
            end_dt = end + timedelta(days=1)  # Include end date
            end_str = end_dt.strftime('%Y-%m-%d')

            # Map timeframe to yfinance interval
            interval = self._map_timeframe(timeframe)

            # For intraday, use period instead of start/end
            if interval in ['1m', '5m', '15m', '1h']:
                df = yf.download(
                    symbol,
                    period='1d',
                    interval=interval,
                    progress=False,
                    auto_adjust=True
                )
            else:
                df = yf.download(
                    symbol,
                    start=start_str,
                    end=end_str,
                    interval=interval,
                    progress=False,
                    auto_adjust=True
                )

            if df is None or df.empty:
                logger.warning(f"[yfinance] No data for {symbol}")
                return None

            # Normalize to standard format
            df = self._normalize_dataframe(df, symbol)

            logger.debug(f"[yfinance] {symbol}: {len(df)} bars")
            return df

        except ImportError:
            logger.error("[yfinance] yfinance package not installed")
            return None
        except Exception as e:
            logger.error(f"[yfinance] Failed to fetch {symbol}: {e}")
            return None

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Dict[str, pd.DataFrame]:
        """Batch fetch using yfinance multi-symbol download."""
        try:
            import yfinance as yf

            interval = self._map_timeframe(timeframe)

            # yfinance batch download
            if interval in ['1m', '5m', '15m', '1h']:
                df = yf.download(
                    symbols,
                    period='1d',
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    group_by='ticker'
                )
            else:
                start_str = start.strftime('%Y-%m-%d')
                end_str = (end + timedelta(days=1)).strftime('%Y-%m-%d')
                df = yf.download(
                    symbols,
                    start=start_str,
                    end=end_str,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    group_by='ticker'
                )

            if df is None or df.empty:
                return {}

            # Parse multi-symbol result
            results = {}

            if isinstance(df.columns, pd.MultiIndex):
                for symbol in symbols:
                    try:
                        symbol_df = df[symbol].copy()
                        symbol_df = self._normalize_dataframe(symbol_df, symbol)
                        if not symbol_df.empty:
                            results[symbol] = symbol_df
                    except KeyError:
                        logger.warning(f"[yfinance] No data for {symbol}")
            else:
                # Single symbol returned
                df = self._normalize_dataframe(df, symbols[0])
                if not df.empty:
                    results[symbols[0]] = df

            logger.info(f"[yfinance] Batch: {len(results)}/{len(symbols)} symbols")
            return results

        except Exception as e:
            logger.error(f"[yfinance] Batch fetch failed: {e}")
            return {}

    def _map_timeframe(self, timeframe: str) -> str:
        """Map internal timeframe to yfinance interval."""
        mapping = {
            '1D': '1d', '1d': '1d', 'D': '1d',
            '1Min': '1m', '1min': '1m',
            '5Min': '5m', '5min': '5m',
            '15Min': '15m', '15min': '15m',
            '1Hour': '1h', '1hour': '1h', '1H': '1h',
        }
        return mapping.get(timeframe, '1d')

    def _normalize_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """Normalize yfinance DataFrame to standard format."""
        result = pd.DataFrame(index=df.index)

        # Handle MultiIndex columns (e.g., ('Close', 'AAPL'))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Map columns to lowercase
        column_map = {
            'Open': 'open', 'open': 'open',
            'High': 'high', 'high': 'high',
            'Low': 'low', 'low': 'low',
            'Close': 'close', 'close': 'close',
            'Volume': 'volume', 'volume': 'volume',
        }

        for orig, new in column_map.items():
            if orig in df.columns and new not in result.columns:
                result[new] = df[orig]

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in result.columns:
                logger.warning(f"[yfinance] Missing column {col} for {symbol}")
                return pd.DataFrame()

        # Normalize timezone to Eastern Time
        if result.index.tz is None:
            result.index = result.index.tz_localize('America/New_York')
        else:
            result.index = result.index.tz_convert('America/New_York')

        return result

    def supports_timeframe(self, timeframe: str) -> bool:
        """yfinance intraday is limited to ~60 days history."""
        return timeframe in ['1D', '1d', 'D', '1Min', '1min', '5Min', '5min',
                            '15Min', '15min', '1Hour', '1hour', '1H']
```

### CompositeDataProvider (composite.py)

```python
"""
Composite Data Provider - Orchestrates fallback chain.

Follows VIXProvider pattern:
1. Try primary provider
2. On failure, try fallback providers in order
3. Cache successful results for resilience
4. Use cache as last resort
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.data.providers.base import DataProviderInterface
from src.data.providers.cache import DataCache
from src.utils.logger import logger


class CompositeDataProvider(DataProviderInterface):
    """
    Composite provider with fallback chain and caching.

    Usage:
        providers = [AlpacaDataProvider(broker), YFinanceDataProvider()]
        composite = CompositeDataProvider(providers, cache_enabled=True)

        # Will try Alpaca first, then yfinance, then cache
        df = composite.get_historical_bars('TQQQ', start, end, '1Min')
    """

    def __init__(
        self,
        providers: List[DataProviderInterface],
        cache_enabled: bool = True,
        cache_max_age_hours: int = 24
    ):
        """
        Initialize composite provider.

        Args:
            providers: List of providers in priority order
            cache_enabled: Enable persistent caching
            cache_max_age_hours: Maximum age for cached data before warning
        """
        self._providers = providers
        self._cache_enabled = cache_enabled
        self._cache = DataCache() if cache_enabled else None
        self._cache_max_age_hours = cache_max_age_hours

        # Track last successful source
        self.last_source: Optional[str] = None
        self.last_fetch_time: Optional[datetime] = None

        provider_names = [p.name for p in providers]
        logger.info(f"[Composite] Initialized with providers: {provider_names}")

    @property
    def name(self) -> str:
        return "Composite"

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Optional[pd.DataFrame]:
        """Fetch bars with fallback chain."""

        # Try each provider in order
        for provider in self._providers:
            if not provider.is_available():
                logger.debug(f"[Composite] {provider.name} unavailable, skipping")
                continue

            if not provider.supports_timeframe(timeframe):
                logger.debug(f"[Composite] {provider.name} doesn't support {timeframe}")
                continue

            df = provider.get_historical_bars(symbol, start, end, timeframe)

            if df is not None and not df.empty:
                self.last_source = provider.name
                self.last_fetch_time = datetime.now()

                # Cache successful result
                if self._cache_enabled and self._cache:
                    self._cache.store(symbol, timeframe, df)

                logger.success(f"[Composite] {symbol} from {provider.name} ({len(df)} bars)")
                return df
            else:
                logger.warning(f"[Composite] {provider.name} failed for {symbol}")

        # All providers failed - try cache as last resort
        if self._cache_enabled and self._cache:
            cached = self._cache.retrieve(
                symbol, timeframe,
                max_age_hours=self._cache_max_age_hours
            )
            if cached is not None:
                self.last_source = "cache"
                logger.warning(f"[Composite] {symbol} from cache (stale data)")
                return cached

        logger.error(f"[Composite] All sources failed for {symbol}")
        return None

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Dict[str, pd.DataFrame]:
        """Batch fetch with per-symbol fallback."""
        results = {}
        remaining = list(symbols)

        # Try batch fetch from each provider
        for provider in self._providers:
            if not remaining:
                break

            if not provider.is_available():
                continue

            if not provider.supports_timeframe(timeframe):
                continue

            batch_results = provider.get_historical_bars_batch(
                remaining, start, end, timeframe
            )

            for symbol, df in batch_results.items():
                if df is not None and not df.empty:
                    results[symbol] = df
                    remaining.remove(symbol)

                    # Cache result
                    if self._cache_enabled and self._cache:
                        self._cache.store(symbol, timeframe, df)

        # Try cache for remaining symbols
        if remaining and self._cache_enabled and self._cache:
            for symbol in list(remaining):
                cached = self._cache.retrieve(
                    symbol, timeframe,
                    max_age_hours=self._cache_max_age_hours
                )
                if cached is not None:
                    results[symbol] = cached
                    remaining.remove(symbol)

        if remaining:
            logger.warning(f"[Composite] Failed to fetch: {remaining}")

        logger.info(f"[Composite] Batch complete: {len(results)}/{len(symbols)} symbols")
        return results

    def get_source_info(self) -> Tuple[Optional[str], Optional[datetime]]:
        """Get info about last data source used."""
        return self.last_source, self.last_fetch_time
```

### DataCache (cache.py)

```python
"""
Persistent Data Cache for Market Data.

Uses parquet format for efficient storage.
Provides TTL-based expiration with stale data fallback.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

from src.settings import get_local_storage_dir
from src.utils.logger import logger
from src.utils.timezone import tz


class DataCache:
    """
    Persistent cache for market data.

    Storage layout:
        cache_dir/
            metadata.json           # Cache index
            daily/
                AAPL.parquet       # Daily bars per symbol
            intraday/
                TQQQ_1Min.parquet  # Intraday bars per symbol+timeframe
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize data cache.

        Args:
            cache_dir: Cache directory (default: {storage}/cache/market_data)
        """
        if cache_dir is None:
            cache_dir = get_local_storage_dir() / "cache" / "market_data"

        self.cache_dir = Path(cache_dir)
        self.daily_dir = self.cache_dir / "daily"
        self.intraday_dir = self.cache_dir / "intraday"
        self.metadata_file = self.cache_dir / "metadata.json"

        # Create directories
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.intraday_dir.mkdir(parents=True, exist_ok=True)

        self._metadata = self._load_metadata()

    def store(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """Store data in cache."""
        try:
            if data is None or data.empty:
                return False

            cache_path = self._get_cache_path(symbol, timeframe)
            data.to_parquet(cache_path)

            # Update metadata
            key = f"{symbol}_{timeframe}"
            self._metadata[key] = {
                'timestamp': tz.iso_timestamp(),
                'rows': len(data),
                'start': str(data.index.min()),
                'end': str(data.index.max())
            }
            self._save_metadata()

            return True

        except Exception as e:
            logger.error(f"[Cache] Failed to store {symbol}: {e}")
            return False

    def retrieve(
        self,
        symbol: str,
        timeframe: str,
        max_age_hours: int = 24
    ) -> Optional[pd.DataFrame]:
        """Retrieve data from cache (returns stale data with warning)."""
        try:
            cache_path = self._get_cache_path(symbol, timeframe)

            if not cache_path.exists():
                return None

            # Check age and warn if stale
            key = f"{symbol}_{timeframe}"
            if key in self._metadata:
                cached_time = datetime.fromisoformat(
                    self._metadata[key]['timestamp'].replace('Z', '+00:00')
                )
                age = tz.now() - cached_time
                age_hours = age.total_seconds() / 3600

                if age_hours > max_age_hours:
                    logger.warning(f"[Cache] {symbol} is stale ({age_hours:.1f}h old)")

            df = pd.read_parquet(cache_path)
            return df

        except Exception as e:
            logger.error(f"[Cache] Failed to retrieve {symbol}: {e}")
            return None

    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path."""
        is_daily = timeframe.upper() in ['1D', 'D', 'DAILY']

        if is_daily:
            return self.daily_dir / f"{symbol}.parquet"
        else:
            return self.intraday_dir / f"{symbol}_{timeframe}.parquet"

    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"[Cache] Failed to save metadata: {e}")
```

### Factory Function (factory.py)

```python
"""
Data Provider Factory - Creates providers from configuration.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from src.data.providers.base import DataProviderInterface
from src.data.providers.alpaca import AlpacaDataProvider
from src.data.providers.yfinance import YFinanceDataProvider
from src.data.providers.composite import CompositeDataProvider
from src.trading.brokers.broker_interface import BrokerInterface
from src.utils.logger import logger


def create_data_provider(
    broker: Optional[BrokerInterface] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None
) -> DataProviderInterface:
    """
    Create a data provider based on configuration.

    Default behavior (no config):
        - If broker provided: CompositeDataProvider([Alpaca, yfinance])
        - If no broker: YFinanceDataProvider only

    Args:
        broker: Optional broker for Alpaca provider
        config: Optional configuration dict
        config_path: Optional path to YAML config file

    Returns:
        Configured DataProviderInterface

    Usage:
        # Default: Alpaca -> yfinance fallback
        provider = create_data_provider(broker=my_broker)

        # From YAML config
        provider = create_data_provider(
            broker=my_broker,
            config_path='config/data_providers.yaml'
        )
    """
    # Load config from file if path provided
    if config_path and config is None:
        config = _load_yaml_config(config_path)

    if config is None:
        config = {}

    # Get provider settings
    dp_config = config.get('data_providers', config)
    provider_names = dp_config.get('providers', ['alpaca', 'yfinance'])
    cache_config = dp_config.get('cache', {})
    cache_enabled = cache_config.get('enabled', True)
    cache_max_age = cache_config.get('max_age_hours', 24)

    # Build provider list
    providers: List[DataProviderInterface] = []

    for name in provider_names:
        name_lower = name.lower()

        if name_lower == 'alpaca':
            if broker is not None:
                providers.append(AlpacaDataProvider(broker))
            else:
                logger.warning("Alpaca provider requested but no broker provided")

        elif name_lower == 'yfinance':
            providers.append(YFinanceDataProvider())

        else:
            logger.warning(f"Unknown provider: {name}")

    if not providers:
        logger.warning("No providers configured, using yfinance")
        providers.append(YFinanceDataProvider())

    # Single provider - no composite needed
    if len(providers) == 1:
        return providers[0]

    return CompositeDataProvider(
        providers=providers,
        cache_enabled=cache_enabled,
        cache_max_age_hours=cache_max_age
    )


def _load_yaml_config(path: str) -> Dict:
    """Load YAML configuration file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        return {}
```

---

## Configuration

### config/data_providers.yaml

```yaml
# Market Data Provider Configuration
#
# Providers are tried in order; first success is used.
# Cache provides fallback if all providers fail.

data_providers:
  # Provider priority order
  providers:
    - alpaca      # Primary: Alpaca API (IEX feed for paper, SIP for live)
    - yfinance    # Fallback: Yahoo Finance (complete coverage)

  # Persistent cache settings
  cache:
    enabled: true
    max_age_hours: 24
    storage_dir: null  # Default: {storage}/cache/market_data

  # Per-provider settings
  alpaca:
    feed: auto  # 'iex' for paper, 'sip' for live (auto-detect)

  yfinance:
    rate_limit_per_second: 2
    retry_on_failure: true
```

---

## Migration Plan

### Phase 1: Create Provider Package (Zero Risk)

**Scope**: Create all new files, no changes to existing code.

Files to create:
- `src/data/providers/__init__.py`
- `src/data/providers/base.py`
- `src/data/providers/alpaca.py`
- `src/data/providers/yfinance.py`
- `src/data/providers/composite.py`
- `src/data/providers/cache.py`
- `src/data/providers/factory.py`
- `config/data_providers.yaml`

Tests to create:
- `tests/data/providers/test_base.py`
- `tests/data/providers/test_yfinance.py`
- `tests/data/providers/test_composite.py`
- `tests/data/providers/test_cache.py`

**Risk**: None - only new files

### Phase 2: Add Provider to OMR Adapter (Low Risk)

**Scope**: Add optional `data_provider` parameter to `OMRLiveAdapter.__init__()`.

```python
# src/trading/adapters/omr_live_adapter.py

def __init__(
    self,
    broker: BrokerInterface,
    symbols: Optional[List[str]] = None,
    # ... existing params ...
    data_provider: Optional[DataProviderInterface] = None  # NEW
):
    # ... existing init ...

    # NEW: Store optional provider
    self._data_provider = data_provider
```

Modify `fetch_market_data()` to use provider when available:

```python
def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
    # NEW: Use provider if available
    if self._data_provider is not None:
        return self._fetch_via_provider()

    # EXISTING: Use broker directly (unchanged)
    return self._fetch_via_broker()

def _fetch_via_provider(self) -> Dict[str, pd.DataFrame]:
    """NEW: Fetch via data provider with fallback."""
    # ... implementation using self._data_provider
```

**Risk**: Low - optional param, existing callers unchanged

### Phase 3: Migrate Momentum Adapter (Medium Risk)

**Scope**: Remove direct yfinance coupling.

Changes:
1. Remove `import yfinance as yf` (line 12)
2. Remove `_fetch_vix_yfinance()` method - use `VIXProvider` instead
3. Add optional `data_provider` parameter
4. Update `fetch_market_data()` to use provider

**Risk**: Medium - modifies working production code

### Phase 4: Base Adapter Enhancement (Low Risk)

**Scope**: Add provider to `StrategyAdapter` base class.

```python
# src/trading/adapters/strategy_adapter.py

def __init__(
    self,
    strategy: StrategySignals,
    broker: BrokerInterface,
    symbols: List[str],
    # ... existing params ...
    data_provider: Optional[DataProviderInterface] = None  # NEW
):
    # ... existing init ...
    self._data_provider = data_provider
```

**Risk**: Low - optional param, subclasses can override

---

## Testing Strategy

### Unit Tests

```python
# tests/data/providers/test_yfinance.py

class TestYFinanceProvider:
    def test_normalize_multiindex_columns(self):
        """Test flattening of yfinance MultiIndex columns."""
        pass

    def test_normalize_timezone_to_et(self):
        """Test timezone conversion to America/New_York."""
        pass

    def test_map_timeframe(self):
        """Test internal to yfinance timeframe mapping."""
        pass

    def test_returns_none_on_empty(self):
        """Test graceful handling of empty responses."""
        pass


# tests/data/providers/test_composite.py

class TestCompositeProvider:
    def test_fallback_on_primary_failure(self):
        """Test fallback to secondary provider."""
        pass

    def test_cache_stores_success(self):
        """Test successful data is cached."""
        pass

    def test_cache_fallback_on_all_fail(self):
        """Test cache as last resort."""
        pass

    def test_source_tracking(self):
        """Test last_source is set correctly."""
        pass
```

### Integration Tests

```python
# tests/data/providers/test_integration.py

class TestProviderIntegration:
    def test_fetch_liquid_symbol(self):
        """Test TQQQ fetches from both providers."""
        pass

    def test_fetch_sparse_symbol(self):
        """Test TECL falls back to yfinance."""
        pass

    def test_omr_adapter_with_provider(self):
        """Test OMR adapter works with injected provider."""
        pass
```

---

## Success Criteria

1. **OMR can fetch complete intraday data** for all 19 leveraged ETF symbols
2. **Existing code works unchanged** when `data_provider` is not specified
3. **yfinance import removed** from `momentum_live_adapter.py`
4. **All tests pass** including new provider tests
5. **Cache provides resilience** when all providers fail temporarily

---

## Appendix: Reference Patterns

### VIXProvider Pattern (src/utils/vix_provider.py)

The VIXProvider demonstrates the fallback pattern to follow:

```python
class VIXProvider:
    def get_vix_data(self, lookback_days=252):
        # 1. Try primary (yfinance)
        vix_data = self._fetch_yfinance(...)
        if vix_data is not None and len(vix_data) >= min_required:
            self.last_source = "yfinance"
            self._persist_latest(vix_data)
            return vix_data

        # 2. Try fallback (FRED)
        vix_data = self._fetch_fred(...)
        if vix_data is not None and len(vix_data) >= min_required:
            self.last_source = "FRED"
            self._persist_latest(vix_data)
            return vix_data

        # 3. Try cache (last resort)
        vix_data = self._load_persisted_data()
        if vix_data is not None:
            self.last_source = "cache"
            return vix_data

        # All failed
        return None
```

### Data Schema Contract

All providers must normalize to this schema:

| Component | Requirement |
|-----------|-------------|
| Index | `DatetimeIndex` with `tz=America/New_York` |
| Columns | `open`, `high`, `low`, `close`, `volume` (lowercase) |
| Column Types | `float64` (including volume) |
| On Failure | Return `None` (not raise exception) |
