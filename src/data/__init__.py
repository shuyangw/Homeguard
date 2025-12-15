"""
Data Package.

Contains modules for market data acquisition and management.
"""

from src.data.downloader import AlpacaDownloader, DownloadResult, Timeframe
from src.data.crypto_downloader import CryptoDownloader
from src.data.news import (
    NewsDownloader,
    NewsDownloadResult,
    SentimentAnalyzer,
    MockSentimentAnalyzer,
    SentimentScore,
    SentimentCache,
    get_sentiment_analyzer,
)

__all__ = [
    # Market Data
    'AlpacaDownloader',
    'CryptoDownloader',
    'DownloadResult',
    'Timeframe',
    # News
    'NewsDownloader',
    'NewsDownloadResult',
    # Sentiment
    'SentimentAnalyzer',
    'MockSentimentAnalyzer',
    'SentimentScore',
    'SentimentCache',
    'get_sentiment_analyzer',
]
