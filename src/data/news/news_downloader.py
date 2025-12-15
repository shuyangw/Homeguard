"""
Alpaca News Downloader Module.

Provides a NewsDownloader class for downloading historical news data from Alpaca API
with robust error handling, rate limiting, and schema compliance.

Usage:
    from src.data.news import NewsDownloader

    downloader = NewsDownloader(start_date='2024-01-01')
    result = downloader.download_symbols(['AAPL', 'MSFT'])
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

from src.api_key import API_KEY, API_SECRET
from src.settings import get_local_storage_dir
from src.data.news.news_schema import (
    NewsArticle,
    save_news_parquet,
    get_news_output_path,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Thread-local storage for Alpaca clients
_thread_local = threading.local()


@dataclass
class NewsDownloadResult:
    """Result of a news download operation."""
    total_symbols: int
    succeeded: int
    failed: int
    total_articles: int
    elapsed_seconds: float
    failed_symbols: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Return success rate as percentage."""
        if self.total_symbols == 0:
            return 0.0
        return (self.succeeded / self.total_symbols) * 100


def _get_news_client() -> NewsClient:
    """Get thread-local Alpaca NewsClient."""
    if not hasattr(_thread_local, 'news_client'):
        _thread_local.news_client = NewsClient(API_KEY, API_SECRET)
    return _thread_local.news_client


def _get_thread_id() -> str:
    """Get short thread ID for logging."""
    name = threading.current_thread().name
    if '_' in name:
        return f"T{name.split('_')[-1]}"
    return name


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable time string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class NewsDownloader:
    """
    Downloads historical news data from Alpaca API.

    Stores data in Hive-partitioned Parquet format:
    {storage_dir}/news/symbol={SYMBOL}/year={YYYY}/news.parquet

    Args:
        start_date: Start date for news download (default: 2020-01-01)
        end_date: End date for news download (default: today)
        max_workers: Number of parallel download threads (default: 4)
        include_content: Whether to include full article content (default: False)
        max_retries: Maximum retries per symbol (default: 3)
        retry_delay: Base delay between retries in seconds (default: 2.0)

    Example:
        downloader = NewsDownloader(start_date='2024-01-01')
        result = downloader.download_symbols(['AAPL', 'MSFT', 'GOOGL'])
        print(f"Downloaded {result.total_articles} articles")
    """

    def __init__(
        self,
        start_date: str = '2020-01-01',
        end_date: Optional[str] = None,
        max_workers: int = 4,
        include_content: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        self.max_workers = max_workers
        self.include_content = include_content
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.storage_dir = get_local_storage_dir()
        self._lock = threading.Lock()
        self._total_articles = 0

    def download_symbols(
        self,
        symbols: List[str],
        skip_existing: bool = True,
    ) -> NewsDownloadResult:
        """
        Download news for multiple symbols.

        Args:
            symbols: List of stock symbols
            skip_existing: Skip symbols that already have data

        Returns:
            NewsDownloadResult with download statistics
        """
        start_time = time.time()
        logger.info(f"Starting news download for {len(symbols)} symbols")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")

        # Filter existing if requested
        if skip_existing:
            symbols = self._filter_existing(symbols)
            if not symbols:
                logger.info("All symbols already have news data, skipping")
                return NewsDownloadResult(
                    total_symbols=0,
                    succeeded=0,
                    failed=0,
                    total_articles=0,
                    elapsed_seconds=0.0
                )

        succeeded = 0
        failed_symbols: List[Tuple[str, str]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_symbol, symbol): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    articles_count = future.result()
                    if articles_count >= 0:
                        succeeded += 1
                        logger.debug(f"[{_get_thread_id()}] {symbol}: {articles_count} articles")
                    else:
                        failed_symbols.append((symbol, "Unknown error"))
                except Exception as e:
                    failed_symbols.append((symbol, str(e)))
                    logger.error(f"[{_get_thread_id()}] {symbol} failed: {e}")

        elapsed = time.time() - start_time

        result = NewsDownloadResult(
            total_symbols=len(symbols),
            succeeded=succeeded,
            failed=len(failed_symbols),
            total_articles=self._total_articles,
            elapsed_seconds=elapsed,
            failed_symbols=failed_symbols,
        )

        logger.info(
            f"News download complete: {result.succeeded}/{result.total_symbols} symbols, "
            f"{result.total_articles} articles in {_format_time(elapsed)}"
        )

        if failed_symbols:
            logger.warning(f"Failed symbols: {[s[0] for s in failed_symbols]}")

        return result

    def _filter_existing(self, symbols: List[str]) -> List[str]:
        """Filter out symbols that already have news data for the date range."""
        filtered = []
        for symbol in symbols:
            # Check if most recent year's data exists
            year = self.end_date.year
            output_path = get_news_output_path(self.storage_dir, symbol, year)
            if not Path(output_path).exists():
                filtered.append(symbol)
            else:
                logger.debug(f"Skipping {symbol}: news data already exists")

        if len(filtered) < len(symbols):
            logger.info(f"Skipping {len(symbols) - len(filtered)} symbols with existing data")

        return filtered

    def _download_symbol(self, symbol: str) -> int:
        """
        Download news for a single symbol with retries.

        Returns:
            Number of articles downloaded, or -1 on failure
        """
        for attempt in range(self.max_retries):
            try:
                return self._download_symbol_impl(symbol)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"[{_get_thread_id()}] {symbol} attempt {attempt + 1} failed: {e}, "
                        f"retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"[{_get_thread_id()}] {symbol} failed after {self.max_retries} attempts: {e}")
                    raise

        return -1

    def _download_symbol_impl(self, symbol: str) -> int:
        """Download news for a single symbol (no retry logic)."""
        client = _get_news_client()

        # Fetch news from Alpaca API
        request = NewsRequest(
            symbols=symbol,
            start=self.start_date,
            end=self.end_date,
            include_content=self.include_content,
            limit=50,  # Per page limit
        )

        all_articles: List[NewsArticle] = []

        # Paginate through all results
        news_set = client.get_news(request)

        # Extract news from NewsSet - data is in news_set.data['news']
        news_list = news_set.data.get('news', []) if hasattr(news_set, 'data') else []

        for article in news_list:
            news_article = NewsArticle.from_alpaca(article, symbol)
            all_articles.append(news_article)

        # Handle pagination
        while news_set.next_page_token:
            request = NewsRequest(
                symbols=symbol,
                start=self.start_date,
                end=self.end_date,
                include_content=self.include_content,
                limit=50,
                page_token=news_set.next_page_token,
            )
            news_set = client.get_news(request)
            news_list = news_set.data.get('news', []) if hasattr(news_set, 'data') else []
            for article in news_list:
                news_article = NewsArticle.from_alpaca(article, symbol)
                all_articles.append(news_article)

        if not all_articles:
            logger.debug(f"[{_get_thread_id()}] {symbol}: No news articles found")
            return 0

        # Convert to DataFrame
        df = pd.DataFrame([a.to_dict() for a in all_articles])

        # Save by year (Hive partitioned)
        df['year'] = df['timestamp'].dt.year
        articles_saved = 0

        for year, year_df in df.groupby('year'):
            output_path = get_news_output_path(self.storage_dir, symbol, int(year))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Drop the year column before saving
            year_df = year_df.drop(columns=['year'])
            save_news_parquet(year_df, output_path)
            articles_saved += len(year_df)

        # Update total count thread-safely
        with self._lock:
            self._total_articles += articles_saved

        return articles_saved

    def download_symbol(self, symbol: str) -> int:
        """
        Download news for a single symbol (public method).

        Args:
            symbol: Stock symbol

        Returns:
            Number of articles downloaded
        """
        return self._download_symbol(symbol)


def download_news_cli(
    symbols: List[str],
    start_date: str = '2020-01-01',
    end_date: Optional[str] = None,
    skip_existing: bool = True,
    max_workers: int = 4,
    include_content: bool = False,
) -> NewsDownloadResult:
    """
    CLI-friendly wrapper for news download.

    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD, default: today)
        skip_existing: Skip symbols with existing data
        max_workers: Number of parallel download threads
        include_content: Include full article content

    Returns:
        NewsDownloadResult with download statistics
    """
    downloader = NewsDownloader(
        start_date=start_date,
        end_date=end_date,
        max_workers=max_workers,
        include_content=include_content,
    )
    return downloader.download_symbols(symbols, skip_existing=skip_existing)
