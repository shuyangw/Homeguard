"""
Symbol downloader utility for GUI - downloads missing symbols from Alpaca API.
"""

from typing import List, Callable, Optional
from datetime import datetime, timedelta
from alpaca.data import TimeFrame
from alpaca.data.enums import DataFeed

from data_engine.api.alpaca_client import AlpacaClient
from data_engine.storage.parquet_storage import ParquetStorage
from gui.utils.error_logger import log_info, log_error, log_exception


class SymbolDownloader:
    """
    Downloads missing symbols from Alpaca API and stores them in parquet format.

    Uses IEX feed by default for free tier compatibility.
    """

    def __init__(self, feed: DataFeed = DataFeed.IEX):
        """
        Initialize symbol downloader.

        Args:
            feed: Data feed to use (default: DataFeed.IEX for free tier)
        """
        self.feed = feed
        self.api_client = None
        self.storage = None

    def download_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> dict:
        """
        Download data for missing symbols.

        Args:
            symbols: List of symbols to download
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            progress_callback: Optional callback(symbol, current, total) for progress updates

        Returns:
            dict with 'successful' and 'failed' lists
        """
        if not symbols:
            return {'successful': [], 'failed': []}

        # Initialize clients
        try:
            self.api_client = AlpacaClient(feed=self.feed)
            self.storage = ParquetStorage()
        except Exception as e:
            log_exception(e, "Failed to initialize download clients")
            return {
                'successful': [],
                'failed': [(sym, f"Initialization error: {str(e)}") for sym in symbols]
            }

        successful = []
        failed = []
        total = len(symbols)

        log_info(f"Downloading {total} missing symbols from Alpaca IEX feed")
        log_info(f"Date range: {start_date} to {end_date}")

        for idx, symbol in enumerate(symbols, start=1):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(symbol, idx, total)

                log_info(f"[{idx}/{total}] Downloading {symbol}...")

                # Fetch data from Alpaca
                data = self.api_client.fetch_bars(
                    symbol=symbol,
                    start_date_str=start_date,
                    end_date_str=end_date,
                    timeframe=TimeFrame.Minute,
                    feed=self.feed
                )

                if data.empty:
                    error_msg = "No data returned from API (symbol may not exist on IEX)"
                    log_error(f"{symbol}: {error_msg}")
                    failed.append((symbol, error_msg))
                    continue

                # Store to parquet
                self.storage.store(data, TimeFrame.Minute)
                log_info(f"{symbol}: Downloaded {len(data)} bars")
                successful.append(symbol)

            except Exception as e:
                error_msg = str(e)
                log_error(f"{symbol}: Failed - {error_msg}")
                failed.append((symbol, error_msg))

        log_info(f"Download complete: {len(successful)} successful, {len(failed)} failed")

        return {
            'successful': successful,
            'failed': failed
        }

    def download_symbol_for_date_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> bool:
        """
        Download a single symbol for a specific date range.

        Args:
            symbol: Symbol to download
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            True if successful, False otherwise
        """
        result = self.download_symbols([symbol], start_date, end_date)
        return len(result['successful']) > 0

    @staticmethod
    def get_default_date_range() -> tuple:
        """
        Get default date range for downloading symbols (last 2 years).

        Returns:
            Tuple of (start_date, end_date) as strings
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years

        return (
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
