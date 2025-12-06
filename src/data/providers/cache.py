"""
Persistent Data Cache for Market Data.

Uses parquet format for efficient storage.
Provides TTL-based expiration with stale data fallback.
"""

import json
from datetime import datetime
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

    def clear(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """Clear cache entries."""
        if symbol is None and timeframe is None:
            # Clear all
            for f in self.daily_dir.glob("*.parquet"):
                f.unlink()
            for f in self.intraday_dir.glob("*.parquet"):
                f.unlink()
            self._metadata = {}
            self._save_metadata()
        elif symbol:
            # Clear specific symbol
            for f in self.daily_dir.glob(f"{symbol}*.parquet"):
                f.unlink()
            for f in self.intraday_dir.glob(f"{symbol}*.parquet"):
                f.unlink()
            self._metadata = {
                k: v for k, v in self._metadata.items()
                if not k.startswith(symbol)
            }
            self._save_metadata()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        daily_count = len(list(self.daily_dir.glob("*.parquet")))
        intraday_count = len(list(self.intraday_dir.glob("*.parquet")))

        return {
            'daily_files': daily_count,
            'intraday_files': intraday_count,
            'metadata_entries': len(self._metadata)
        }
