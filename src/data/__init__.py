"""
Data Package.

Contains modules for market data acquisition and management.

Data downloading is handled by the unified acquisition module:
    from src.data.acquisition import DataAcquisitionManager
"""

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
