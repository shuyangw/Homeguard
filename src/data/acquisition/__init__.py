"""
Unified Data Acquisition Module.

Consolidates all data downloading into a plugin-based architecture with
shared infrastructure for threading, retry, storage, and manifest tracking.

Usage:
    from src.data.acquisition import DataAcquisitionManager

    manager = DataAcquisitionManager()
    result = manager.download(source="futures", symbols=["ES", "NQ"])

CLI:
    python -m src.data.acquisition --source futures --symbols ES,NQ --start 2020-01-01
"""

from src.data.acquisition.base import BaseDownloader, DownloadResult
from src.data.acquisition.manager import DataAcquisitionManager
from src.data.acquisition.schemas import (
    CANONICAL_OHLCV_SCHEMA,
    FUTURES_TRADES_SCHEMA,
    SchemaValidationError,
    validate_schema,
)

__all__ = [
    "BaseDownloader",
    "DataAcquisitionManager",
    "DownloadResult",
    "CANONICAL_OHLCV_SCHEMA",
    "FUTURES_TRADES_SCHEMA",
    "SchemaValidationError",
    "validate_schema",
]
