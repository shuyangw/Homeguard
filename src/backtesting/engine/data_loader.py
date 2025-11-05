"""
Data loader for backtesting engine that reads from parquet storage.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional
from datetime import datetime

from config import settings, OS_ENVIRONMENT
from utils import logger
from backtesting.utils.market_calendar import MarketCalendar


class DataLoader:
    """
    Loads OHLCV data from parquet storage for backtesting.
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None, filter_market_days: bool = True):
        """
        Initialize the data loader.

        Args:
            base_path: Base directory for data storage (defaults to OS-specific setting)
            filter_market_days: If True, automatically filter out weekends and holidays (default: True)
        """
        if base_path is None:
            base_path = settings[OS_ENVIRONMENT]["local_storage_dir"]
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "equities_1min"
        self.filter_market_days = filter_market_days
        self.market_calendar = MarketCalendar('NYSE') if filter_market_days else None

    def load_symbols(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        timeframe: str = '1min'
    ) -> pd.DataFrame:
        """
        Load OHLCV data for one or more symbols.

        Args:
            symbols: Single symbol string or list of symbols (e.g., 'AAPL' or ['AAPL', 'MSFT'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: Data timeframe (default: '1min', currently only 1min supported)

        Returns:
            DataFrame with MultiIndex (symbol, timestamp) and OHLCV columns
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        escaped_symbols = [symbol.replace("'", "''") for symbol in symbols]
        symbols_str = "', '".join(escaped_symbols)
        query = f"""
        SELECT *
        FROM read_parquet('{self.data_path.as_posix()}/**/*.parquet', hive_partitioning=1)
        WHERE symbol IN ('{symbols_str}')
          AND timestamp >= '{start_dt}'
          AND timestamp <= '{end_dt}'
        ORDER BY symbol, timestamp
        """

        con = duckdb.connect(database=':memory:')
        df = con.execute(query).df()
        con.close()

        if df.empty:
            raise ValueError(f"No data found for symbols {symbols} between {start_date} and {end_date}")

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.drop(columns=['year', 'month'], errors='ignore')
        df = df.set_index(['symbol', 'timestamp']).sort_index()

        # Filter for market trading days (excludes weekends and holidays)
        if self.filter_market_days and self.market_calendar is not None:
            original_len = len(df)
            df = self.market_calendar.filter_trading_days(df)
            filtered_count = original_len - len(df)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} non-trading day bars (weekends/holidays)")

        logger.success(f"Loaded {len(df)} bars for {len(symbols)} symbol(s) from {start_date} to {end_date}")

        return df

    def load_single_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = '1min'
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a single symbol with timestamp as index.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: Data timeframe (default: '1min')

        Returns:
            DataFrame with timestamp index and OHLCV columns
        """
        df = self.load_symbols([symbol], start_date, end_date, timeframe)
        result: pd.DataFrame = df.xs(symbol, level='symbol')  # type: ignore[assignment]

        return result

    def get_available_symbols(self) -> List[str]:
        """
        Get list of all symbols available in storage.

        Returns:
            List of symbol strings
        """
        if not self.data_path.exists():
            return []

        symbol_dirs = [d.name.replace('symbol=', '') for d in self.data_path.iterdir() if d.is_dir()]

        return sorted(symbol_dirs)

    def check_symbols_availability(self, symbols: Union[str, List[str]]) -> dict:
        """
        Check which symbols are available in storage and which are missing.

        Args:
            symbols: Single symbol string or list of symbols

        Returns:
            dict with 'available' and 'missing' lists
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        available_symbols = set(self.get_available_symbols())
        requested_symbols = set(symbols)

        return {
            'available': sorted(list(requested_symbols & available_symbols)),
            'missing': sorted(list(requested_symbols - available_symbols))
        }

    def get_date_range(self, symbol: str) -> tuple:
        """
        Get the available date range for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (start_date, end_date) as datetime objects
        """
        escaped_symbol = symbol.replace("'", "''")
        query = f"""
        SELECT MIN(timestamp) as start_date, MAX(timestamp) as end_date
        FROM read_parquet('{self.data_path.as_posix()}/**/*.parquet', hive_partitioning=1)
        WHERE symbol = '{escaped_symbol}'
        """

        con = duckdb.connect(database=':memory:')
        result = con.execute(query).df()
        con.close()

        if result.empty or result['start_date'].isna().all():
            raise ValueError(f"No data found for symbol {symbol}")

        return result['start_date'].iloc[0], result['end_date'].iloc[0]
