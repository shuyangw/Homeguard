"""Alpaca news plugin for the unified data acquisition module.

Thin wrapper - delegates to existing NewsDownloader for the complex
weekly-chunked pagination logic.
"""

from typing import Any

import pandas as pd

from src.data.acquisition.base import BaseDownloader
from src.data.news.news_downloader import NewsDownloader as LegacyNewsDownloader
from src.settings import NEWS_ALPACA
from src.utils.logger import get_logger

logger = get_logger(__name__)

NEWS_SCHEMA = ["timestamp", "headline", "summary", "source", "url", "symbol"]


class AlpacaNewsPlugin(BaseDownloader):
    """Downloads news data from Alpaca. Wraps existing NewsDownloader."""

    def _create_client(self) -> Any:
        return None  # NewsDownloader manages its own client

    def _fetch_symbol_data(
        self, client: Any, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        downloader = LegacyNewsDownloader(start_date=start, end_date=end)
        count = downloader.download_symbol(symbol)
        logger.info(f"{symbol}: downloaded {count} articles via legacy downloader")
        return pd.DataFrame(columns=NEWS_SCHEMA)

    def _get_schema(self) -> list[str]:
        return NEWS_SCHEMA

    def _get_storage_subdir(self) -> str:
        return NEWS_ALPACA

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol
