"""
DEPRECATED: This file is maintained for backward compatibility only.

The code has been refactored into separate modules for better organization:

- API Client: data_engine.api.alpaca_client
- Storage: data_engine.storage.parquet_storage
- Metadata: data_engine.storage.metadata_store
- Symbol Loading: data_engine.loaders.symbol_loader
- Orchestration: data_engine.orchestration.ingestion_pipeline

For new code, please import from the appropriate modules above.
For running the ingestion pipeline, use: src/run_ingestion.py

This file simply re-exports all functions for backward compatibility.
"""

from alpaca.data import TimeFrame
from pathlib import Path

# Import all functions from refactored modules
from data_engine.api.alpaca_client import fetch_data, AlpacaClient
from data_engine.storage.parquet_storage import store_data, get_base_path, get_thread_count, ParquetStorage
from data_engine.storage.metadata_store import store_metadata, load_metadata, MetadataStore
from data_engine.loaders.symbol_loader import load_index_symbols, SymbolLoader
from data_engine.orchestration.ingestion_pipeline import (
    fetch_and_store_symbol,
    store_data_with_metadata,
    ingest_index_from_csv,
    IngestionPipeline
)

# Re-export all for backward compatibility
__all__ = [
    'fetch_data',
    'store_data',
    'get_base_path',
    'get_thread_count',
    'store_metadata',
    'load_metadata',
    'load_index_symbols',
    'fetch_and_store_symbol',
    'store_data_with_metadata',
    'ingest_index_from_csv',
    'AlpacaClient',
    'ParquetStorage',
    'MetadataStore',
    'SymbolLoader',
    'IngestionPipeline',
]


if __name__ == "__main__":
    """
    Legacy main function - use src/run_ingestion.py instead.
    """
    print("=" * 60)
    print("DEPRECATION WARNING:")
    print("This file is deprecated. Please use: python src/run_ingestion.py")
    print("=" * 60)
    print()

    # Define date range for data ingestion
    start_date = "2016-01-01"
    end_date = "2025-01-01"
    timeframe = TimeFrame.Minute

    # Path to index CSV files
    base_path = Path(__file__).parent.parent.parent / "backtest_lists"

    # Ingest S&P 500 data
    sp500_csv = base_path / "sp500-2025.csv"
    ingest_index_from_csv(
        csv_path=sp500_csv,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        index_name='s_and_p_500'  # Optional: will use 'sp500_2025' if not provided
    )

    # Example: Ingest other indices (uncomment and add CSV files as needed)
    # nasdaq100_csv = base_path / "nasdaq100-2025.csv"
    # ingest_index_from_csv(
    #     csv_path=nasdaq100_csv,
    #     start_date=start_date,
    #     end_date=end_date,
    #     timeframe=timeframe
    # )
    #
    # djia_csv = base_path / "djia-2025.csv"
    # ingest_index_from_csv(
    #     csv_path=djia_csv,
    #     start_date=start_date,
    #     end_date=end_date,
    #     timeframe=timeframe
    # )
