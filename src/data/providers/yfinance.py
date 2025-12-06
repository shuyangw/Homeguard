"""
YFinance Data Provider - Fetches data from Yahoo Finance.

Handles yfinance quirks:
- MultiIndex columns for multi-symbol downloads
- Capitalized column names ('Close' -> 'close')
- Timezone normalization to Eastern Time

Resilience features:
- Rate limiting (configurable delay between requests)
- Exponential backoff on 429 errors
- Time-bounded retries (must complete before deadline, e.g., 3:54 PM)
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from src.data.providers.base import DataProviderInterface
from src.utils.logger import logger
from src.utils.timezone import tz


class YFinanceDataProvider(DataProviderInterface):
    """
    Data provider using yfinance (Yahoo Finance).

    Normalizes output to standard format:
    - Lowercase column names
    - Eastern Time timezone
    - Single-level column index

    Rate limiting and retry behavior:
    - Default 0.5s delay between requests
    - Exponential backoff on 429 errors (1s, 2s, 4s, 8s)
    - Max 3 retries per request
    - Time-bounded: stops retrying if deadline approaching
    """

    def __init__(
        self,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
        base_backoff: float = 1.0,
        deadline_buffer_seconds: int = 60
    ):
        """
        Initialize YFinance provider with rate limiting.

        Args:
            rate_limit_delay: Seconds to wait between requests (default: 0.5)
            max_retries: Maximum retry attempts on 429 errors (default: 3)
            base_backoff: Base delay for exponential backoff (default: 1.0s)
            deadline_buffer_seconds: Stop retrying this many seconds before deadline (default: 60)
        """
        self._rate_limit_delay = rate_limit_delay
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._deadline_buffer = deadline_buffer_seconds
        self._last_request_time: Optional[datetime] = None

    @property
    def name(self) -> str:
        return "yfinance"

    def _wait_for_rate_limit(self) -> None:
        """Wait if needed to respect rate limit."""
        if self._last_request_time is not None:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self._rate_limit_delay:
                sleep_time = self._rate_limit_delay - elapsed
                time.sleep(sleep_time)
        self._last_request_time = datetime.now()

    def _should_stop_retrying(self, deadline: Optional[datetime] = None) -> bool:
        """Check if we should stop retrying due to deadline."""
        if deadline is None:
            return False

        now = tz.now()
        buffer = timedelta(seconds=self._deadline_buffer)
        return now >= (deadline - buffer)

    def _get_default_deadline(self) -> Optional[datetime]:
        """
        Get default deadline (3:49 PM ET) to not interfere with OMR trade at 3:50 PM.

        Returns None if current time is after 3:49 PM (no deadline needed).
        """
        now = tz.now()
        deadline = now.replace(hour=15, minute=49, second=0, microsecond=0)

        # If we're already past 3:49 PM, no deadline needed today
        if now >= deadline:
            return None

        return deadline

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False,  # Accepted for API consistency, ignored (yfinance always fresh)
        deadline: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch bars via yfinance with rate limiting and retries.

        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe ('1D', '1Min', etc.)
            force_refresh: Accepted for API consistency, ignored (yfinance always fetches fresh)
            deadline: Stop retrying after this time (default: 3:54 PM ET)
        """
        if deadline is None:
            deadline = self._get_default_deadline()

        for attempt in range(self._max_retries + 1):
            # Check deadline before each attempt
            if self._should_stop_retrying(deadline):
                logger.warning(f"[yfinance] Stopping retries for {symbol} - approaching deadline")
                return None

            try:
                self._wait_for_rate_limit()
                return self._fetch_single(symbol, start, end, timeframe)

            except Exception as e:
                error_str = str(e).lower()

                # Check for rate limit error
                if '429' in error_str or 'rate' in error_str or 'too many' in error_str:
                    if attempt < self._max_retries:
                        backoff = self._base_backoff * (2 ** attempt)
                        logger.warning(
                            f"[yfinance] Rate limited on {symbol}, "
                            f"backing off {backoff:.1f}s (attempt {attempt + 1}/{self._max_retries})"
                        )

                        # Check if backoff would exceed deadline
                        if deadline and self._should_stop_retrying(deadline):
                            logger.warning(f"[yfinance] Skipping backoff - deadline approaching")
                            return None

                        time.sleep(backoff)
                        continue

                # Non-rate-limit error or max retries exceeded
                logger.error(f"[yfinance] Failed to fetch {symbol}: {e}")
                return None

        return None

    def _fetch_single(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Internal fetch without retry logic."""
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

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False,  # Accepted for API consistency, ignored (yfinance always fresh)
        deadline: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Batch fetch using yfinance multi-symbol download with rate limiting.

        Args:
            symbols: List of stock symbols
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe
            force_refresh: Accepted for API consistency, ignored (yfinance always fetches fresh)
            deadline: Stop retrying after this time (default: 3:54 PM ET)
        """
        if deadline is None:
            deadline = self._get_default_deadline()

        for attempt in range(self._max_retries + 1):
            # Check deadline before each attempt
            if self._should_stop_retrying(deadline):
                logger.warning(f"[yfinance] Stopping batch retries - approaching deadline")
                return {}

            try:
                self._wait_for_rate_limit()
                return self._fetch_batch(symbols, start, end, timeframe)

            except Exception as e:
                error_str = str(e).lower()

                # Check for rate limit error
                if '429' in error_str or 'rate' in error_str or 'too many' in error_str:
                    if attempt < self._max_retries:
                        backoff = self._base_backoff * (2 ** attempt)
                        logger.warning(
                            f"[yfinance] Rate limited on batch, "
                            f"backing off {backoff:.1f}s (attempt {attempt + 1}/{self._max_retries})"
                        )

                        if deadline and self._should_stop_retrying(deadline):
                            logger.warning(f"[yfinance] Skipping backoff - deadline approaching")
                            return {}

                        time.sleep(backoff)
                        continue

                logger.error(f"[yfinance] Batch fetch failed: {e}")
                return {}

        return {}

    def _fetch_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """Internal batch fetch without retry logic."""
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
