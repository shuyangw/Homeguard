"""
Sentiment Cache for Pre-computed Sentiment Scores.

Stores and retrieves sentiment analysis results in Parquet format,
enabling efficient batch processing and backtest integration.

Usage:
    from src.data.news import SentimentCache

    cache = SentimentCache()

    # Compute and cache sentiment for news data
    cache.compute_and_save('AAPL', 2024)

    # Load cached sentiment for backtesting
    df = cache.load('AAPL', 2024)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.settings import get_local_storage_dir
from src.data.news.news_schema import (
    SENTIMENT_SCHEMA,
    load_news_parquet,
    get_news_output_path,
    get_sentiment_output_path,
)
from src.data.news.sentiment_analyzer import (
    SentimentAnalyzer,
    get_sentiment_analyzer,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentCache:
    """
    Cache for pre-computed sentiment analysis results.

    Stores sentiment scores in Hive-partitioned Parquet format:
    {storage_dir}/sentiment/symbol={SYMBOL}/year={YYYY}/sentiment.parquet

    Args:
        storage_dir: Base storage directory (default: from settings)
        analyzer: SentimentAnalyzer instance (default: create new)

    Example:
        cache = SentimentCache()

        # Process all news for a symbol/year
        cache.compute_and_save('AAPL', 2024)

        # Load for backtesting
        sentiment = cache.load('AAPL', 2024)
        print(sentiment[sentiment['sentiment_label'] == 'positive'])
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
    ):
        self.storage_dir = storage_dir or get_local_storage_dir()
        self._analyzer = analyzer

    @property
    def analyzer(self) -> SentimentAnalyzer:
        """Get or create sentiment analyzer (lazy init)."""
        if self._analyzer is None:
            self._analyzer = get_sentiment_analyzer()
        return self._analyzer

    def get_news_path(self, symbol: str, year: int) -> str:
        """Get path to news parquet file."""
        return get_news_output_path(self.storage_dir, symbol, year)

    def get_sentiment_path(self, symbol: str, year: int) -> str:
        """Get path to sentiment parquet file."""
        return get_sentiment_output_path(self.storage_dir, symbol, year)

    def has_news(self, symbol: str, year: int) -> bool:
        """Check if news data exists for symbol/year."""
        return Path(self.get_news_path(symbol, year)).exists()

    def has_sentiment(self, symbol: str, year: int) -> bool:
        """Check if sentiment data exists for symbol/year."""
        return Path(self.get_sentiment_path(symbol, year)).exists()

    def load_news(self, symbol: str, year: int) -> pd.DataFrame:
        """Load news data for symbol/year."""
        path = self.get_news_path(symbol, year)
        if not Path(path).exists():
            raise FileNotFoundError(f"News data not found: {path}")
        return load_news_parquet(path)

    def load(self, symbol: str, year: int) -> pd.DataFrame:
        """
        Load cached sentiment data for symbol/year.

        Args:
            symbol: Stock symbol
            year: Year

        Returns:
            DataFrame with sentiment data

        Raises:
            FileNotFoundError: If sentiment data not found
        """
        path = self.get_sentiment_path(symbol, year)
        if not Path(path).exists():
            raise FileNotFoundError(f"Sentiment data not found: {path}")

        # Use ParquetFile to read single file directly (avoids dataset inference
        # which can cause schema merge conflicts in Hive-partitioned directories)
        pf = pq.ParquetFile(path)
        table = pf.read()
        return table.to_pandas()

    def load_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Load sentiment data for a date range across years.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with sentiment data for the range
        """
        dfs = []
        for year in range(start_date.year, end_date.year + 1):
            if self.has_sentiment(symbol, year):
                try:
                    df = self.load(symbol, year)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load sentiment for {symbol}/{year}: {e}")

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)

        # Filter to date range
        result['timestamp'] = pd.to_datetime(result['timestamp'], utc=True)

        # Convert dates to UTC-aware timestamps (handle both tz-aware and naive)
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize('UTC')
        else:
            start_ts = start_ts.tz_convert('UTC')
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize('UTC')
        else:
            end_ts = end_ts.tz_convert('UTC')

        mask = (result['timestamp'] >= start_ts) & (result['timestamp'] <= end_ts)
        return result[mask].reset_index(drop=True)

    def compute_and_save(
        self,
        symbol: str,
        year: int,
        overwrite: bool = False,
    ) -> int:
        """
        Compute sentiment for news data and save to cache.

        Args:
            symbol: Stock symbol
            year: Year to process
            overwrite: Whether to overwrite existing sentiment data

        Returns:
            Number of articles processed
        """
        # Check if already exists
        sentiment_path = self.get_sentiment_path(symbol, year)
        if Path(sentiment_path).exists() and not overwrite:
            logger.debug(f"Sentiment already exists for {symbol}/{year}, skipping")
            return 0

        # Load news data
        if not self.has_news(symbol, year):
            logger.warning(f"No news data found for {symbol}/{year}")
            return 0

        news_df = self.load_news(symbol, year)
        if news_df.empty:
            logger.debug(f"Empty news data for {symbol}/{year}")
            return 0

        logger.info(f"Computing sentiment for {symbol}/{year}: {len(news_df)} articles")

        # Use headlines only for sentiment analysis
        # Summaries dilute sentiment signal (more neutral/factual language)
        headlines = news_df['headline'].fillna('').tolist()
        results = self.analyzer.analyze_batch(headlines)

        # Build sentiment DataFrame
        sentiment_df = pd.DataFrame({
            'id': news_df['id'].values,
            'timestamp': news_df['timestamp'].values,
            'symbol': news_df['symbol'].values,
            'headline': news_df['headline'].values,
            'sentiment_score': [r.sentiment_score for r in results],
            'sentiment_positive': [r.positive_prob for r in results],
            'sentiment_negative': [r.negative_prob for r in results],
            'sentiment_neutral': [r.neutral_prob for r in results],
            'sentiment_label': [r.sentiment_label for r in results],
            'confidence': [r.confidence for r in results],
        })

        # Ensure timestamp is UTC
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], utc=True)

        # Save to Parquet - build columns explicitly to avoid dictionary encoding
        Path(sentiment_path).parent.mkdir(parents=True, exist_ok=True)

        # Build PyArrow arrays explicitly with correct types
        arrays = [
            pa.array(sentiment_df['id'].astype(str).tolist(), type=pa.string()),
            pa.array(sentiment_df['timestamp'].tolist(), type=pa.timestamp('us', tz='UTC')),
            pa.array(sentiment_df['symbol'].astype(str).tolist(), type=pa.string()),
            pa.array(sentiment_df['headline'].astype(str).tolist(), type=pa.string()),
            pa.array(sentiment_df['sentiment_score'].tolist(), type=pa.float64()),
            pa.array(sentiment_df['sentiment_positive'].tolist(), type=pa.float64()),
            pa.array(sentiment_df['sentiment_negative'].tolist(), type=pa.float64()),
            pa.array(sentiment_df['sentiment_neutral'].tolist(), type=pa.float64()),
            pa.array(sentiment_df['sentiment_label'].astype(str).tolist(), type=pa.string()),
            pa.array(sentiment_df['confidence'].tolist(), type=pa.float64()),
        ]

        table = pa.Table.from_arrays(arrays, schema=SENTIMENT_SCHEMA)
        pq.write_table(
            table,
            sentiment_path,
            compression='snappy',
            use_dictionary=True
        )

        logger.info(f"Saved sentiment data to {sentiment_path}")
        return len(sentiment_df)

    def compute_symbol(
        self,
        symbol: str,
        start_year: int,
        end_year: int,
        overwrite: bool = False,
    ) -> Dict[int, int]:
        """
        Compute sentiment for a symbol across multiple years.

        Args:
            symbol: Stock symbol
            start_year: Starting year
            end_year: Ending year (inclusive)
            overwrite: Whether to overwrite existing data

        Returns:
            Dict mapping year to number of articles processed
        """
        results = {}
        for year in range(start_year, end_year + 1):
            count = self.compute_and_save(symbol, year, overwrite=overwrite)
            results[year] = count
        return results

    def list_available_news(self) -> List[Dict[str, Any]]:
        """
        List all available news data.

        Returns:
            List of dicts with 'symbol' and 'year' keys
        """
        news_base = Path(self.storage_dir) / 'news'
        if not news_base.exists():
            return []

        available = []
        for symbol_dir in news_base.glob('symbol=*'):
            symbol = symbol_dir.name.replace('symbol=', '')
            for year_dir in symbol_dir.glob('year=*'):
                year = int(year_dir.name.replace('year=', ''))
                news_path = year_dir / 'news.parquet'
                if news_path.exists():
                    available.append({
                        'symbol': symbol,
                        'year': year,
                        'news_path': str(news_path),
                        'has_sentiment': self.has_sentiment(symbol, year),
                    })

        return sorted(available, key=lambda x: (x['symbol'], x['year']))

    def list_missing_sentiment(self) -> List[Dict[str, Any]]:
        """
        List news data that doesn't have sentiment computed.

        Returns:
            List of dicts with 'symbol' and 'year' for missing sentiment
        """
        available = self.list_available_news()
        return [x for x in available if not x['has_sentiment']]

    def get_daily_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        aggregation: str = 'mean',
    ) -> pd.DataFrame:
        """
        Get daily aggregated sentiment for backtesting.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            aggregation: How to aggregate ('mean', 'median', 'last')

        Returns:
            DataFrame with daily sentiment indexed by date
        """
        df = self.load_range(symbol, start_date, end_date)

        if df.empty:
            return pd.DataFrame()

        # Extract date
        df['date'] = df['timestamp'].dt.date

        # Aggregate by date
        if aggregation == 'mean':
            agg_func = 'mean'
        elif aggregation == 'median':
            agg_func = 'median'
        elif aggregation == 'last':
            # Sort by timestamp and take last
            df = df.sort_values('timestamp')
            agg_func = 'last'
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        daily = df.groupby('date').agg({
            'sentiment_score': agg_func,
            'sentiment_positive': agg_func,
            'sentiment_negative': agg_func,
            'confidence': agg_func,
            'id': 'count',  # Article count
        }).rename(columns={'id': 'article_count'})

        daily.index = pd.to_datetime(daily.index)
        return daily
