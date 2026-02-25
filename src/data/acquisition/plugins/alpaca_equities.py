"""Alpaca equities plugin for the unified data acquisition module."""

from datetime import datetime
from typing import Any

import pandas as pd
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockBarsRequest

from src.api_key import API_KEY, API_SECRET
from src.data.acquisition.base import BaseDownloader
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlpacaEquitiesPlugin(BaseDownloader):
    """Downloads equity OHLCV data from Alpaca."""

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
        return "equities_1min"

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol
