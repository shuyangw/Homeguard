"""
Storage module for persisting stock data to Parquet format using DuckDB.
"""

import duckdb
from pathlib import Path
from alpaca.data import TimeFrame
import threading

from src.config import settings, OS_ENVIRONMENT


def get_base_path():
    """
    Get the base storage path from settings based on detected OS.

    Returns:
        Path object for the base storage directory
    """
    base_path = settings[OS_ENVIRONMENT]["local_storage_dir"]
    return Path(base_path)


def get_thread_count():
    """
    Get the number of threads to use for API calls based on detected OS.

    Returns:
        int: Number of threads
    """
    return int(settings[OS_ENVIRONMENT]["api_threads"])


class ParquetStorage:
    """
    Handles storage of stock data in partitioned Parquet format.
    """

    def __init__(self, base_path=None):
        """
        Initialize the storage manager.

        Args:
            base_path: Base directory for data storage (defaults to OS-specific setting)
        """
        self.base_path = Path(base_path) if base_path else get_base_path()

    def store(self, data, timeframe=TimeFrame.Minute):
        """
        Store a DataFrame in a partitioned Parquet dataset using DuckDB.
        Stores data in the format: base_path/equities_1min/symbol=AAPL/year=2024/month=01/data.parquet

        Args:
            data: The pandas DataFrame to store (must have 'symbol' and 'timestamp' in the index).
            timeframe: The TimeFrame object used to fetch the data (default: TimeFrame.Minute).
        """
        if data.empty:
            print("Received empty DataFrame. Nothing to store.")
            return

        # --- 1. Preprocess Data for Partitioning ---

        # Reset the index to turn 'symbol' and 'timestamp' into regular columns
        processed_data = data.reset_index()

        # Ensure timestamp is timezone-aware (it should be from Alpaca)
        # and create year/month columns for partitioning
        processed_data['timestamp'] = processed_data['timestamp'].dt.tz_convert('UTC')  # Standardize to UTC
        processed_data['year'] = processed_data['timestamp'].dt.year
        processed_data['month'] = processed_data['timestamp'].dt.month

        # --- 2. Store Data using DuckDB ---

        # Use equities_1min for minute-level data (adjust as needed for other timeframes)
        thread_id = threading.get_ident()
        storage_path = self.base_path / "equities_1min"
        storage_path.mkdir(parents=True, exist_ok=True)

        print(f"[Thread-{thread_id}] Storing data in: {storage_path}")

        con = duckdb.connect(database=':memory:')
        con.register('data_to_write', processed_data)

        query = f"""
        COPY data_to_write
        TO '{storage_path.as_posix()}'
        (FORMAT 'parquet', PARTITION_BY (symbol, year, month), OVERWRITE_OR_IGNORE 1);
        """

        con.execute(query)
        con.close()

        # A bit of hard-coding to get the symbol for the print message
        symbol = processed_data['symbol'].iloc[0]
        print(f"[Thread-{thread_id}] Successfully stored data for {symbol}.")
