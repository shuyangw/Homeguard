"""
Crypto Data Provider - Reads from local parquet storage.

Wraps the existing Alpaca crypto data stored in Hive-partitioned parquet format.
Supports symbols like 'BTC/USD', 'ETH/USD', etc.

Data is stored at: {LOCAL_STORAGE}/crypto_1day/, crypto_1hour/, crypto_1min/
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from src.data.providers.base import DataProviderInterface
from src.settings import get_local_storage_dir
from src.utils.logger import logger


# Timeframe to directory suffix mapping
_TIMEFRAME_DIR_MAP = {
    '1D': 'crypto_1day',
    '1d': 'crypto_1day',
    'day': 'crypto_1day',
    '1H': 'crypto_1hour',
    '1h': 'crypto_1hour',
    'hour': 'crypto_1hour',
    '1Min': 'crypto_1min',
    '1min': 'crypto_1min',
    'minute': 'crypto_1min',
}


def _normalize_symbol(symbol: str) -> str:
    """Convert API format (BTC/USD) to filesystem format (BTC_USD)."""
    return symbol.replace('/', '_')


def _denormalize_symbol(symbol: str) -> str:
    """Convert filesystem format (BTC_USD) to API format (BTC/USD)."""
    return symbol.replace('_', '/')


class CryptoDataProvider(DataProviderInterface):
    """
    Data provider for crypto assets using local parquet storage.

    Reads from Hive-partitioned parquet files downloaded by CryptoDownloader.
    Storage format: symbol=XXX/year=YYYY/month=MM/data.parquet

    Symbols use API format (BTC/USD) externally but filesystem format (BTC_USD) internally.

    Example:
        provider = CryptoDataProvider()
        df = provider.get_historical_bars('BTC/USD', start, end, '1D')
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the crypto data provider.

        Args:
            base_dir: Base storage directory. Defaults to settings.ini LOCAL_STORAGE_DIR.
        """
        self._base_dir = Path(base_dir) if base_dir else get_local_storage_dir()

    @property
    def name(self) -> str:
        return "Crypto"

    def _get_data_dir(self, timeframe: str) -> Path:
        """Get the data directory for a specific timeframe."""
        dir_suffix = _TIMEFRAME_DIR_MAP.get(timeframe, 'crypto_1day')
        return self._base_dir / dir_suffix

    def get_available_symbols(self, timeframe: str = '1D') -> Set[str]:
        """
        Get list of available crypto symbols.

        Args:
            timeframe: Timeframe to check ('1D', '1H', '1Min')

        Returns:
            Set of symbol names in API format (e.g., {'BTC/USD', 'ETH/USD'})
        """
        data_dir = self._get_data_dir(timeframe)
        symbols = set()

        if not data_dir.exists():
            return symbols

        for symbol_dir in data_dir.glob('symbol=*'):
            fs_symbol = symbol_dir.name.replace('symbol=', '')
            # Check if there's at least one parquet file
            if list(symbol_dir.glob('**/data*.parquet')):
                symbols.add(_denormalize_symbol(fs_symbol))

        return symbols

    def _load_symbol_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        Load data for a single symbol from parquet files.

        Args:
            symbol: Symbol in API format (e.g., 'BTC/USD')
            start: Start datetime
            end: End datetime
            timeframe: Timeframe string

        Returns:
            DataFrame with OHLCV data or None if not found
        """
        fs_symbol = _normalize_symbol(symbol)
        data_dir = self._get_data_dir(timeframe)
        symbol_dir = data_dir / f'symbol={fs_symbol}'

        if not symbol_dir.exists():
            logger.debug(f"[Crypto] No data directory for {symbol}")
            return None

        # Find all relevant parquet files based on date range
        start_year = start.year
        start_month = start.month
        end_year = end.year
        end_month = end.month

        parquet_files = []
        for year in range(start_year, end_year + 1):
            year_dir = symbol_dir / f'year={year}'
            if not year_dir.exists():
                continue

            # Determine month range for this year
            m_start = start_month if year == start_year else 1
            m_end = end_month if year == end_year else 12

            for month in range(m_start, m_end + 1):
                month_dir = year_dir / f'month={month}'
                if month_dir.exists():
                    parquet_file = month_dir / 'data.parquet'
                    if parquet_file.exists():
                        parquet_files.append(parquet_file)

        if not parquet_files:
            logger.debug(f"[Crypto] No parquet files found for {symbol} in date range")
            return None

        # Load and concatenate all parquet files
        dfs = []
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"[Crypto] Error reading {pf}: {e}")
                continue

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)

        # Convert timestamp to datetime and filter by date range
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            # Convert start/end to timezone-aware for comparison
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize('UTC')
            if end_ts.tz is None:
                end_ts = end_ts.tz_localize('UTC')

            df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]

            # Set timestamp as index with ET timezone
            df = df.set_index('timestamp')
            df.index = df.index.tz_convert('America/New_York')

        # Ensure lowercase column names
        df.columns = [c.lower() for c in df.columns]

        # Sort by index
        df = df.sort_index()

        if df.empty:
            return None

        return df

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV bars for a crypto symbol.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD', 'ETH/USD')
            start: Start datetime
            end: End datetime
            timeframe: '1D', '1H', or '1Min'
            force_refresh: Ignored (data is from local storage)

        Returns:
            DataFrame with OHLCV data, or None on failure
        """
        try:
            df = self._load_symbol_data(symbol, start, end, timeframe)

            if df is None or df.empty:
                logger.warning(f"[Crypto] No data for {symbol}")
                return None

            logger.debug(f"[Crypto] Loaded {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"[Crypto] Failed to load {symbol}: {e}")
            return None

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical bars for multiple crypto symbols.

        Args:
            symbols: List of crypto symbols (e.g., ['BTC/USD', 'ETH/USD'])
            start: Start datetime
            end: End datetime
            timeframe: Timeframe string
            force_refresh: Ignored (data is from local storage)

        Returns:
            Dict mapping symbol -> DataFrame
        """
        results = {}

        for symbol in symbols:
            df = self.get_historical_bars(symbol, start, end, timeframe, force_refresh)
            if df is not None:
                results[symbol] = df

        logger.info(f"[Crypto] Loaded {len(results)}/{len(symbols)} symbols")
        return results

    def is_available(self) -> bool:
        """Check if crypto data storage exists."""
        for tf in ['1D', '1H', '1Min']:
            data_dir = self._get_data_dir(tf)
            if data_dir.exists():
                return True
        return False

    def supports_timeframe(self, timeframe: str) -> bool:
        """Check if the timeframe is supported."""
        return timeframe in _TIMEFRAME_DIR_MAP

    def get_date_range(self, symbol: str, timeframe: str = '1D') -> Optional[tuple]:
        """
        Get the available date range for a symbol.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            timeframe: Timeframe string

        Returns:
            Tuple of (min_date, max_date) or None if no data
        """
        # Load a sample to get date range
        df = self._load_symbol_data(
            symbol,
            datetime(2000, 1, 1),  # Far past
            datetime(2100, 1, 1),  # Far future
            timeframe
        )

        if df is None or df.empty:
            return None

        return (df.index.min(), df.index.max())
