"""
Symbol downloader utility for GUI - downloads missing symbols from Alpaca API.

Uses the unified data acquisition module (src.data.acquisition) for downloading
and storage in canonical hive-partitioned parquet format.
"""

from typing import List, Callable, Optional
from datetime import datetime, timedelta

from src.data.acquisition import DataAcquisitionManager
from gui.utils.error_logger import log_info, log_error, log_exception


class SymbolDownloader:
    """
    Downloads missing symbols from Alpaca API and stores them in parquet format.

    Uses the unified DataAcquisitionManager with the 'equities' plugin.
    """

    def __init__(self):
        """Initialize symbol downloader."""
        self.manager = DataAcquisitionManager()

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

        total = len(symbols)
        log_info(f"Downloading {total} missing symbols via acquisition module")
        log_info(f"Date range: {start_date} to {end_date}")

        try:
            result = self.manager.download(
                source="equities",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            log_exception(e, "Failed to download symbols")
            return {
                'successful': [],
                'failed': [(sym, f"Download error: {str(e)}") for sym in symbols]
            }

        # Convert DownloadResult to the dict format expected by the GUI
        successful = []
        failed_list = list(result.failed_symbols)

        # Determine which symbols succeeded (those not in failed list)
        failed_symbol_names = {sym for sym, _ in failed_list}
        successful = [s for s in symbols if s not in failed_symbol_names]

        # Fire progress callback for each symbol if provided
        if progress_callback:
            for idx, symbol in enumerate(symbols, start=1):
                progress_callback(symbol, idx, total)

        log_info(f"Download complete: {len(successful)} successful, {len(failed_list)} failed")

        return {
            'successful': successful,
            'failed': failed_list
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
