"""
Data Engine Package

A modular data ingestion and storage system for stock market data.
"""

from src.data_engine.api.alpaca_client import AlpacaClient
from src.data_engine.storage.parquet_storage import ParquetStorage, get_base_path, get_thread_count
from src.data_engine.storage.metadata_store import MetadataStore
from src.data_engine.loaders.symbol_loader import SymbolLoader
from src.data_engine.orchestration.ingestion_pipeline import IngestionPipeline

__all__ = [
    'AlpacaClient',
    'ParquetStorage',
    'get_base_path',
    'get_thread_count',
    'MetadataStore',
    'SymbolLoader',
    'IngestionPipeline',
]
