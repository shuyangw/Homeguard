"""
VIX Data Provider with Multi-Source Fallback.

Provides VIX data with resilient fallback chain:
1. yfinance (primary) - Yahoo Finance ^VIX
2. FRED API (fallback) - Federal Reserve VIXCLS series
3. Persisted cache (last resort) - Last known good VIX value

Usage:
    from src.data.vix_provider import VIXProvider

    vix_provider = VIXProvider()
    vix_data = vix_provider.get_vix_data(lookback_days=252)
    current_vix = vix_provider.get_current_vix()
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import pytz

from src.utils.logger import logger
from src.utils.timezone import tz
from src.settings import get_local_storage_dir, get_models_dir


class VIXProvider:
    """
    Multi-source VIX data provider with persistent caching.

    Fallback chain:
    1. yfinance ^VIX (primary, real-time)
    2. FRED VIXCLS (fallback, end-of-day, official CBOE data)
    3. Persisted cache (last resort, stale but better than nothing)
    """

    # Cache file for persisted VIX data
    CACHE_FILENAME = "vix_cache.json"

    # Maximum age for cached data before warning (hours)
    MAX_CACHE_AGE_HOURS = 24

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize VIX provider.

        Args:
            cache_dir: Directory for VIX cache file (default: data/trading/)
        """
        if cache_dir is None:
            cache_dir = get_local_storage_dir() / "data" / "trading"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / self.CACHE_FILENAME

        # Track which source was used
        self.last_source: Optional[str] = None
        self.last_fetch_time: Optional[datetime] = None

    def get_vix_data(
        self,
        lookback_days: int = 252,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get VIX historical data with fallback chain.

        Args:
            lookback_days: Number of days of history (default: 252 = 1 year)
            end_date: End date for data (default: now)

        Returns:
            DataFrame with VIX data (columns: close), or None if all sources fail
        """
        if end_date is None:
            end_date = tz.now()

        start_date = end_date - timedelta(days=lookback_days + 30)  # Buffer for weekends

        # Try each source in order
        vix_data = None

        # Minimum required data points (80% of requested trading days)
        min_required = int(lookback_days * 0.7 * 0.8)  # ~70% are trading days, want 80% of those

        # 1. Try yfinance (primary)
        vix_data = self._fetch_yfinance(start_date, end_date)
        if vix_data is not None and len(vix_data) >= min_required:
            self.last_source = "yfinance"
            self._persist_latest(vix_data)
            return vix_data

        # 2. Try FRED API (fallback)
        logger.warning("[VIX] yfinance failed or insufficient data, trying FRED API...")
        vix_data = self._fetch_fred(start_date, end_date)
        if vix_data is not None and len(vix_data) >= min_required:
            self.last_source = "FRED"
            self._persist_latest(vix_data)
            return vix_data

        # 3. Try persisted cache (last resort)
        logger.warning("[VIX] FRED failed, trying persisted cache...")
        vix_data = self._load_persisted_data()
        if vix_data is not None:
            self.last_source = "cache"
            logger.warning(f"[VIX] Using cached data from {self.last_fetch_time}")
            return vix_data

        # All sources failed
        logger.error("[VIX] All VIX data sources failed!")
        self.last_source = None
        return None

    def get_current_vix(self) -> Optional[float]:
        """
        Get the current (latest) VIX value.

        Returns:
            Current VIX value or None if unavailable
        """
        # Try to get fresh data (small lookback for speed)
        vix_data = self.get_vix_data(lookback_days=5)

        if vix_data is not None and not vix_data.empty:
            return float(vix_data['close'].iloc[-1])

        # Fallback to cached current value
        cached = self._load_cached_current()
        if cached is not None:
            logger.warning(f"[VIX] Using cached VIX value: {cached:.2f}")
            return cached

        return None

    def _fetch_yfinance(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch VIX data from yfinance."""
        try:
            import yfinance as yf

            logger.info("[VIX] Fetching from yfinance...")

            vix_data = yf.download(
                '^VIX',
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )

            if vix_data is None or vix_data.empty:
                logger.warning("[VIX] yfinance returned empty data")
                return None

            # Normalize to standard format
            df = self._normalize_dataframe(vix_data, 'yfinance')

            logger.success(f"[VIX] yfinance: {len(df)} days, latest: {df['close'].iloc[-1]:.2f}")
            self.last_fetch_time = tz.now()
            return df

        except Exception as e:
            logger.error(f"[VIX] yfinance failed: {e}")
            return None

    def _fetch_fred(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch VIX data from FRED API (VIXCLS series)."""
        try:
            import pandas_datareader as pdr

            logger.info("[VIX] Fetching from FRED (VIXCLS)...")

            # Convert to naive datetime for FRED API (it doesn't handle timezones)
            if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                start_naive = start_date.replace(tzinfo=None)
            else:
                start_naive = start_date

            if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                end_naive = end_date.replace(tzinfo=None)
            else:
                end_naive = end_date

            # VIXCLS is the official CBOE VIX Close
            vix_data = pdr.DataReader(
                'VIXCLS',
                'fred',
                start_naive,
                end_naive + timedelta(days=1)
            )

            if vix_data is None or vix_data.empty:
                logger.warning("[VIX] FRED returned empty data")
                return None

            # Drop NaN values (weekends/holidays)
            vix_data = vix_data.dropna()

            if vix_data.empty:
                logger.warning("[VIX] FRED data was all NaN")
                return None

            # Normalize to standard format
            df = self._normalize_dataframe(vix_data, 'FRED')

            logger.success(f"[VIX] FRED: {len(df)} days, latest: {df['close'].iloc[-1]:.2f}")
            self.last_fetch_time = tz.now()
            return df

        except ImportError:
            logger.warning("[VIX] pandas_datareader not installed")
            return None
        except Exception as e:
            logger.error(f"[VIX] FRED failed: {e}")
            return None

    def _normalize_dataframe(
        self,
        df: pd.DataFrame,
        source: str
    ) -> pd.DataFrame:
        """Normalize DataFrame to standard format with 'close' column."""
        result = pd.DataFrame(index=df.index)

        if source == 'yfinance':
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                result['close'] = df[('Close', '^VIX')].values
            elif 'Close' in df.columns:
                result['close'] = df['Close'].values
            else:
                result['close'] = df.iloc[:, 0].values

        elif source == 'FRED':
            # FRED returns single column 'VIXCLS'
            if 'VIXCLS' in df.columns:
                result['close'] = df['VIXCLS'].values
            else:
                result['close'] = df.iloc[:, 0].values

        else:
            # Generic - take first column
            result['close'] = df.iloc[:, 0].values

        # Ensure timezone-aware index (ET)
        if result.index.tz is None:
            result.index = result.index.tz_localize('America/New_York')
        else:
            result.index = result.index.tz_convert('America/New_York')

        return result

    def _persist_latest(self, vix_data: pd.DataFrame) -> None:
        """Persist latest VIX data to cache file."""
        try:
            if vix_data is None or vix_data.empty:
                return

            # Store last 30 days for quick recovery
            recent_data = vix_data.tail(30)

            cache_data = {
                "timestamp": tz.iso_timestamp(),
                "source": self.last_source,
                "current_vix": float(recent_data['close'].iloc[-1]),
                "latest_date": str(recent_data.index[-1]),
                "data": {
                    str(idx): float(val)
                    for idx, val in recent_data['close'].items()
                }
            }

            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"[VIX] Persisted {len(recent_data)} days to cache")

        except Exception as e:
            logger.error(f"[VIX] Failed to persist cache: {e}")

    def _load_persisted_data(self) -> Optional[pd.DataFrame]:
        """Load VIX data from persisted cache."""
        try:
            if not self.cache_file.exists():
                logger.warning("[VIX] No cache file found")
                return None

            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)

            # Check cache age
            cache_time = datetime.fromisoformat(cache_data['timestamp'].replace('Z', '+00:00'))
            if cache_time.tzinfo is None:
                cache_time = pytz.timezone('America/New_York').localize(cache_time)

            age_hours = (tz.now() - cache_time).total_seconds() / 3600

            if age_hours > self.MAX_CACHE_AGE_HOURS:
                logger.warning(f"[VIX] Cache is {age_hours:.1f} hours old (stale)")

            # Reconstruct DataFrame
            data = cache_data.get('data', {})
            if not data:
                return None

            df = pd.DataFrame({
                'close': list(data.values())
            }, index=pd.to_datetime(list(data.keys()), utc=True))

            # Convert to Eastern Time (cache may have mixed timezones)
            df.index = df.index.tz_convert('America/New_York')

            self.last_fetch_time = cache_time
            logger.success(f"[VIX] Loaded {len(df)} days from cache (age: {age_hours:.1f}h)")

            return df

        except Exception as e:
            logger.error(f"[VIX] Failed to load cache: {e}")
            return None

    def _load_cached_current(self) -> Optional[float]:
        """Load just the current VIX value from cache."""
        try:
            if not self.cache_file.exists():
                return None

            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)

            return cache_data.get('current_vix')

        except Exception as e:
            logger.error(f"[VIX] Failed to load cached current: {e}")
            return None

    def get_source_info(self) -> Tuple[Optional[str], Optional[datetime]]:
        """Get info about the last data source used."""
        return self.last_source, self.last_fetch_time


# Singleton instance for easy access
_vix_provider: Optional[VIXProvider] = None


def get_vix_provider() -> VIXProvider:
    """Get singleton VIX provider instance."""
    global _vix_provider
    if _vix_provider is None:
        _vix_provider = VIXProvider()
    return _vix_provider


if __name__ == "__main__":
    # Test the provider
    logger.info("Testing VIX Provider")
    logger.info("=" * 60)

    provider = VIXProvider()

    # Test full data fetch
    vix_data = provider.get_vix_data(lookback_days=30)
    if vix_data is not None:
        logger.success(f"Got {len(vix_data)} days of VIX data")
        logger.info(f"Source: {provider.last_source}")
        logger.info(f"Latest: {vix_data['close'].iloc[-1]:.2f}")

    # Test current VIX
    current = provider.get_current_vix()
    if current is not None:
        logger.success(f"Current VIX: {current:.2f}")
