"""Alpaca equities plugin for the unified data acquisition module."""

from datetime import datetime
from typing import Any, Optional

import pandas as pd
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest

from src.api_key import API_KEY, API_SECRET
from src.data.acquisition.base import BaseDownloader
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlpacaEquitiesPlugin(BaseDownloader):
    """Downloads equity OHLCV data from Alpaca."""

    def __init__(
        self,
        feed: Optional[DataFeed] = None,
        adjustment: Optional[Adjustment] = None,
        storage_subdir_override: Optional[str] = None,
        **kwargs: Any,
    ):
        self._feed = feed
        self._adjustment = adjustment
        self._storage_subdir_override = storage_subdir_override
        super().__init__(**kwargs)

    def _create_client(self) -> Any:
        return StockHistoricalDataClient(API_KEY, API_SECRET)

    def _fetch_symbol_data(
        self, client: Any, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.strptime(start, "%Y-%m-%d"),
            end=datetime.strptime(end, "%Y-%m-%d"),
            feed=self._feed,
            adjustment=self._adjustment,
        )
        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return pd.DataFrame(columns=CANONICAL_OHLCV_SCHEMA)

        df = df.reset_index()
        df = df[CANONICAL_OHLCV_SCHEMA]
        return df

    def _get_schema(self) -> list[str]:
        return CANONICAL_OHLCV_SCHEMA

    def _get_storage_subdir(self) -> str:
        return self._storage_subdir_override or "equities_1min"

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol
