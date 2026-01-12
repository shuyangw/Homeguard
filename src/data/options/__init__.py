"""
Options Data Module.

Provides data infrastructure for options chain data from ThetaData API.
Used by OpEx Pinning strategy for GEX calculations.

Components:
- ThetaDataClient: REST API client for fetching options data
- OptionsDataStore: Parquet-based persistent storage
- ThetaDataAdapter: Reads existing ThetaData parquet files
- Schema: Dataclasses for options snapshots and GEX data
"""

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
]
