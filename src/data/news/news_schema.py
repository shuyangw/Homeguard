"""
News Data Schema Definitions.

Defines the canonical schema for news data storage in Parquet format,
matching the Alpaca News API response structure.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Canonical news schema for Parquet storage
NEWS_SCHEMA = pa.schema([
    ('id', pa.string()),                    # Unique article ID from Alpaca
    ('timestamp', pa.timestamp('us', tz='UTC')),  # Publication time
    ('symbol', pa.string()),                # Primary symbol
    ('symbols', pa.list_(pa.string())),     # All related symbols
    ('headline', pa.string()),              # Article headline
    ('summary', pa.string()),               # Article summary (may be truncated)
    ('source', pa.string()),                # News source (e.g., 'benzinga')
    ('url', pa.string()),                   # Original article URL
    ('author', pa.string()),                # Article author (if available)
    ('content', pa.string()),               # Full content (if available)
])

# Sentiment schema (computed separately)
SENTIMENT_SCHEMA = pa.schema([
    ('id', pa.string()),                    # Article ID (foreign key to news)
    ('timestamp', pa.timestamp('us', tz='UTC')),
    ('symbol', pa.string()),
    ('headline', pa.string()),
    ('sentiment_score', pa.float64()),      # -1 (negative) to +1 (positive)
    ('sentiment_positive', pa.float64()),   # Probability of positive
    ('sentiment_negative', pa.float64()),   # Probability of negative
    ('sentiment_neutral', pa.float64()),    # Probability of neutral
    ('sentiment_label', pa.string()),       # 'positive', 'negative', 'neutral'
    ('confidence', pa.float64()),           # Max probability
])


@dataclass
class NewsArticle:
    """Represents a single news article."""
    id: str
    timestamp: datetime
    symbol: str
    symbols: List[str]
    headline: str
    summary: str
    source: str
    url: str
    author: Optional[str] = None
    content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'symbols': self.symbols,
            'headline': self.headline,
            'summary': self.summary,
            'source': self.source,
            'url': self.url,
            'author': self.author or '',
            'content': self.content or '',
        }

    @classmethod
    def from_alpaca(cls, article: Any, primary_symbol: str) -> 'NewsArticle':
        """
        Create NewsArticle from Alpaca News API response.

        Args:
            article: Alpaca News object
            primary_symbol: The symbol this article was queried for

        Returns:
            NewsArticle instance
        """
        return cls(
            id=str(article.id),
            timestamp=article.created_at,
            symbol=primary_symbol,
            symbols=list(article.symbols) if article.symbols else [primary_symbol],
            headline=article.headline or '',
            summary=(article.summary or '')[:2000],  # Truncate long summaries
            source=article.source or 'unknown',
            url=article.url or '',
            author=getattr(article, 'author', None),
            content=getattr(article, 'content', None),
        )


@dataclass
class SentimentResult:
    """Represents sentiment analysis result for an article."""
    id: str
    timestamp: datetime
    symbol: str
    headline: str
    sentiment_score: float      # -1 to +1
    sentiment_positive: float   # 0 to 1
    sentiment_negative: float   # 0 to 1
    sentiment_neutral: float    # 0 to 1
    sentiment_label: str        # 'positive', 'negative', 'neutral'
    confidence: float           # Max of the three probabilities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'headline': self.headline,
            'sentiment_score': self.sentiment_score,
            'sentiment_positive': self.sentiment_positive,
            'sentiment_negative': self.sentiment_negative,
            'sentiment_neutral': self.sentiment_neutral,
            'sentiment_label': self.sentiment_label,
            'confidence': self.confidence,
        }


def validate_news_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame matches the news schema.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_cols = ['id', 'timestamp', 'symbol', 'headline', 'source']

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise ValueError("'timestamp' column must be datetime type")

    return True


def save_news_parquet(
    df: pd.DataFrame,
    output_path: str,
    schema: pa.Schema = NEWS_SCHEMA
) -> None:
    """
    Save news DataFrame to Parquet with schema enforcement.

    Args:
        df: DataFrame with news data
        output_path: Path to save Parquet file
        schema: PyArrow schema to use
    """
    # Ensure timestamp is UTC
    if 'timestamp' in df.columns:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Convert to PyArrow Table with schema
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    # Write with compression
    pq.write_table(
        table,
        output_path,
        compression='snappy',
        use_dictionary=True
    )

    logger.debug(f"Saved {len(df)} news articles to {output_path}")


def load_news_parquet(input_path: str) -> pd.DataFrame:
    """
    Load news DataFrame from Parquet.

    Args:
        input_path: Path to Parquet file

    Returns:
        DataFrame with news data
    """
    table = pq.read_table(input_path)
    return table.to_pandas()


def get_news_output_path(
    base_dir: str,
    symbol: str,
    year: int,
    month: Optional[int] = None
) -> str:
    """
    Get the output path for news data (Hive partitioned).

    Args:
        base_dir: Base storage directory
        symbol: Stock symbol
        year: Year
        month: Month (optional, for monthly partitioning)

    Returns:
        Full path to Parquet file
    """
    from pathlib import Path

    if month:
        path = Path(base_dir) / 'news' / f'symbol={symbol}' / f'year={year}' / f'month={month}' / 'news.parquet'
    else:
        path = Path(base_dir) / 'news' / f'symbol={symbol}' / f'year={year}' / 'news.parquet'

    return str(path)


def get_sentiment_output_path(
    base_dir: str,
    symbol: str,
    year: int
) -> str:
    """
    Get the output path for sentiment data.

    Args:
        base_dir: Base storage directory
        symbol: Stock symbol
        year: Year

    Returns:
        Full path to Parquet file
    """
    from pathlib import Path

    path = Path(base_dir) / 'sentiment' / f'symbol={symbol}' / f'year={year}' / 'sentiment.parquet'
    return str(path)
