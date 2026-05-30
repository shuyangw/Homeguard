"""Alpaca crypto plugin for the unified data acquisition module."""

from datetime import datetime
from typing import Any

import pandas as pd
from alpaca.data import CryptoHistoricalDataClient, TimeFrame
from alpaca.data.requests import CryptoBarsRequest

from src.api_key import API_KEY, API_SECRET
from src.data.acquisition.base import BaseDownloader
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA
from src.settings import CRYPTO_ALPACA_1MIN
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlpacaCryptoPlugin(BaseDownloader):
    """Downloads crypto OHLCV data from Alpaca."""

    def _create_client(self) -> Any:
        return CryptoHistoricalDataClient(API_KEY, API_SECRET)

    def _fetch_symbol_data(
        self, client: Any, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.strptime(start, "%Y-%m-%d"),
            end=datetime.strptime(end, "%Y-%m-%d"),
        )
        bars = client.get_crypto_bars(request)
        df = bars.df

        if df.empty:
            return pd.DataFrame(columns=CANONICAL_OHLCV_SCHEMA)

        df = df.reset_index()

        # Fill missing columns with NaN
        for col in CANONICAL_OHLCV_SCHEMA:
            if col not in df.columns:
                df[col] = float("nan")

        df = df[CANONICAL_OHLCV_SCHEMA]
        return df

    def _get_schema(self) -> list[str]:
        return CANONICAL_OHLCV_SCHEMA

    def _get_storage_subdir(self) -> str:
        return CRYPTO_ALPACA_1MIN

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "_")
