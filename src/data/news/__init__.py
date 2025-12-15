"""
News Data Module.

Provides infrastructure for downloading news data from Alpaca API,
computing sentiment with FinBERT, and caching results for backtesting.
"""

from src.data.news.news_schema import (
    NEWS_SCHEMA,
    SENTIMENT_SCHEMA,
    NewsArticle,
    SentimentResult,
    validate_news_dataframe,
    save_news_parquet,
    load_news_parquet,
    get_news_output_path,
    get_sentiment_output_path,
)
from src.data.news.news_downloader import (
    NewsDownloader,
    NewsDownloadResult,
)
from src.data.news.sentiment_analyzer import (
    SentimentScore,
    SentimentAnalyzer,
    MockSentimentAnalyzer,
    get_sentiment_analyzer,
)
from src.data.news.sentiment_cache import SentimentCache

__all__ = [
    # Schema
    'NEWS_SCHEMA',
    'SENTIMENT_SCHEMA',
    # Data classes
    'NewsArticle',
    'SentimentResult',
    'SentimentScore',
    # Schema functions
    'validate_news_dataframe',
    'save_news_parquet',
    'load_news_parquet',
    'get_news_output_path',
    'get_sentiment_output_path',
    # Downloader
    'NewsDownloader',
    'NewsDownloadResult',
    # Sentiment
    'SentimentAnalyzer',
    'MockSentimentAnalyzer',
    'get_sentiment_analyzer',
    'SentimentCache',
]
