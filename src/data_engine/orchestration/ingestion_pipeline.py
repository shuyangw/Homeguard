"""
Orchestration module for managing multithreaded data ingestion pipelines.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from alpaca.data import TimeFrame
import time
import threading

from src.data_engine.api.alpaca_client import AlpacaClient
from src.data_engine.storage.parquet_storage import ParquetStorage, get_thread_count
from src.data_engine.storage.metadata_store import MetadataStore
from src.data_engine.loaders.symbol_loader import SymbolLoader


class IngestionPipeline:
    """
    Manages the end-to-end ingestion pipeline for stock data.
    """

    def __init__(self, max_workers=None):
        """
        Initialize the ingestion pipeline.

        Args:
            max_workers: Number of threads to use (defaults to OS-specific setting)
        """
        self.max_workers = max_workers or get_thread_count()
        self.api_client = AlpacaClient()
        self.storage = ParquetStorage()
        self.metadata_store = MetadataStore()

    def _fetch_and_store_symbol(self, symbol, start_date, end_date, timeframe):
        thread_id = threading.get_ident()
        try:
            print(f"[Thread-{thread_id}] Processing {symbol}...")
            data = self.api_client.fetch_bars(symbol, start_date, end_date, timeframe)
            self.storage.store(data, timeframe)
            return (symbol, True, None)
        except Exception as e:
            error_msg = str(e)
            print(f"[Thread-{thread_id}] Error processing {symbol}: {error_msg}")
            return (symbol, False, error_msg)

    def ingest_symbols(self, symbols, start_date, end_date, timeframe=TimeFrame.Minute,
                       index_name=None):
        """
        Ingest data for multiple symbols using multithreading.

        Args:
            symbols: List of stock symbols to fetch
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: TimeFrame object (default: TimeFrame.Minute)
            index_name: Optional name of index for metadata storage

        Returns:
            dict: Summary with 'successful' and 'failed' lists
        """
        if index_name:
            self.metadata_store.store(index_name, symbols)

        total_symbols = len(symbols)
        print(f"\nUsing {self.max_workers} threads for data ingestion")
        print(f"Total symbols to process: {total_symbols}")

        successful = []
        failed = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._fetch_and_store_symbol, symbol, start_date, end_date, timeframe): symbol
                for symbol in symbols
            }

            for completed_count, future in enumerate(as_completed(future_to_symbol), start=1):
                symbol, success, error_msg = future.result()

                if success:
                    successful.append(symbol)
                else:
                    failed.append((symbol, error_msg))

                elapsed_time = time.time() - start_time
                progress_pct = (completed_count / total_symbols) * 100
                remaining = total_symbols - completed_count
                avg_time_per_symbol = elapsed_time / completed_count
                estimated_remaining = avg_time_per_symbol * remaining

                print(f"Progress: {completed_count}/{total_symbols} ({progress_pct:.1f}%)")
                print(f"  Successful: {len(successful)} | Failed: {len(failed)} | Remaining: {remaining}")
                print(f"  Elapsed: {self._format_time(elapsed_time)} | Est. remaining: {self._format_time(estimated_remaining)}")

        total_time = time.time() - start_time
        print(f"\nTotal ingestion time: {self._format_time(total_time)}")

        self._print_summary(successful, failed, total_symbols)

        return {
            'successful': successful,
            'failed': failed
        }

    def _format_time(self, seconds):
        """
        Format seconds into human-readable time string.

        Args:
            seconds: Time in seconds

        Returns:
            str: Formatted time string (e.g., '2h 15m 30s' or '45m 20s' or '30s')
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"

    def ingest_from_csv(self, csv_path, start_date, end_date, timeframe=TimeFrame.Minute,
                        index_name=None, symbol_column='Symbol'):
        """
        Load symbols from a CSV file and ingest their data.

        Args:
            csv_path: Path to the CSV file containing stock symbols (can be str or Path)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: TimeFrame object (default: TimeFrame.Minute)
            index_name: Optional name of index for metadata storage. If None, derives from CSV filename
            symbol_column: Name of the column containing symbols (default: 'Symbol')

        Returns:
            dict: Summary with 'successful' and 'failed' lists
        """
        csv_path = Path(csv_path)

        if index_name is None:
            index_name = csv_path.stem.replace('-', '_').replace(' ', '_').lower()

        symbols = SymbolLoader.from_csv(csv_path, symbol_column=symbol_column)

        print(f"\n{'='*60}")
        print(f"Starting data ingestion for {len(symbols)} symbols from {csv_path.name}")
        print(f"Index name: {index_name}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Timeframe: {timeframe}")
        print(f"{'='*60}\n")

        result = self.ingest_symbols(symbols, start_date, end_date, timeframe, index_name)

        print(f"\n{'='*60}")
        print(f"Data ingestion complete for {index_name}!")
        print(f"{'='*60}")

        return result

    def _print_summary(self, successful, failed, total):
        """
        Print ingestion summary.

        Args:
            successful: List of successful symbols
            failed: List of tuples (symbol, error_message) for failed symbols
            total: Total number of symbols
        """
        print(f"\n{'='*60}")
        print(f"Ingestion Summary:")
        print(f"  Successful: {len(successful)}/{total}")
        print(f"  Failed: {len(failed)}/{total}")
        if failed:
            print(f"\nFailed symbols:")
            for symbol, error in failed[:10]:  # Show first 10 failures
                print(f"  - {symbol}: {error}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")
        print(f"{'='*60}")
