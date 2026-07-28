"""
Options Data Module.

Provides data infrastructure for options chain data from ThetaData API.
Used by OpEx Pinning strategy for GEX calculations.

Components:
- ThetaDataClient: REST API client for fetching options data
- OptionsDataStore: Parquet-based persistent storage
- ThetaDataAdapter: Reads existing ThetaData parquet files
- Schema: Dataclasses for options snapshots and GEX data
- canonical: Canonicalization layer over the 1-minute options store
  (registered 15:45 ET snapshot guard, V3 quote validity, V6 `_eod` lag)
"""

from src.data.options.canonical import (
    CANONICAL_COLUMNS,
    SNAPSHOT_TIME_ET,
    UnexpectedRightError,
    add_eod_lag,
    build_chain_eod,
    build_chain_eod_frame,
    canonicalize_frame,
    iter_canonical_batches,
    snapshot_from_bars,
    trading_days_between,
)

from src.data.options.options_schema import (
    OptionSnapshot,
    StrikeGEX,
    OptionsChain,
    DailyGEXSummary,
)
from src.data.options.thetadata_client import ThetaDataClient
from src.data.options.options_store import OptionsDataStore
from src.data.options.thetadata_adapter import ThetaDataAdapter

__all__ = [
    "OptionSnapshot",
    "StrikeGEX",
    "OptionsChain",
    "DailyGEXSummary",
    "ThetaDataClient",
    "OptionsDataStore",
    "ThetaDataAdapter",
    "CANONICAL_COLUMNS",
    "SNAPSHOT_TIME_ET",
    "UnexpectedRightError",
    "add_eod_lag",
    "build_chain_eod",
    "build_chain_eod_frame",
    "canonicalize_frame",
    "iter_canonical_batches",
    "snapshot_from_bars",
    "trading_days_between",
]
