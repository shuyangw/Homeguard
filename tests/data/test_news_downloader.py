"""
Unit Tests for News Downloader and Schema.

Tests news article parsing, schema validation, and download functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

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


class TestNewsArticle:
    """Test NewsArticle dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        article = NewsArticle(
            id='12345',
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol='AAPL',
            symbols=['AAPL', 'MSFT'],
            headline='Apple announces new product',
            summary='Apple Inc. has announced...',
            source='benzinga',
            url='https://example.com/article',
            author='John Doe',
            content='Full article content here...',
        )

        result = article.to_dict()

        assert result['id'] == '12345'
        assert result['symbol'] == 'AAPL'
        assert result['headline'] == 'Apple announces new product'
        assert result['source'] == 'benzinga'
        assert result['author'] == 'John Doe'
        assert result['symbols'] == ['AAPL', 'MSFT']

    def test_to_dict_none_values(self):
        """Test conversion with None optional values."""
        article = NewsArticle(
            id='12345',
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol='AAPL',
            symbols=['AAPL'],
            headline='Test headline',
            summary='Test summary',
            source='benzinga',
            url='https://example.com',
            author=None,
            content=None,
        )

        result = article.to_dict()

        # None values should become empty strings
        assert result['author'] == ''
        assert result['content'] == ''

    def test_from_alpaca(self):
        """Test creation from Alpaca API response."""
        mock_article = Mock()
        mock_article.id = 12345
        mock_article.created_at = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        mock_article.symbols = ['AAPL', 'MSFT']
        mock_article.headline = 'Test Headline'
        mock_article.summary = 'Test summary'
        mock_article.source = 'benzinga'
        mock_article.url = 'https://example.com'

        article = NewsArticle.from_alpaca(mock_article, 'AAPL')

        assert article.id == '12345'
        assert article.symbol == 'AAPL'
        assert article.symbols == ['AAPL', 'MSFT']
        assert article.headline == 'Test Headline'

    def test_from_alpaca_missing_symbols(self):
        """Test creation when symbols list is empty."""
        mock_article = Mock()
        mock_article.id = 12345
        mock_article.created_at = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        mock_article.symbols = None
        mock_article.headline = 'Test Headline'
        mock_article.summary = None
        mock_article.source = None
        mock_article.url = None

        article = NewsArticle.from_alpaca(mock_article, 'AAPL')

        assert article.symbols == ['AAPL']
        assert article.summary == ''
        assert article.source == 'unknown'
        assert article.url == ''

    def test_summary_truncation(self):
        """Test that long summaries are truncated."""
        mock_article = Mock()
        mock_article.id = 12345
        mock_article.created_at = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        mock_article.symbols = ['AAPL']
        mock_article.headline = 'Test'
        mock_article.summary = 'x' * 3000  # Very long summary
        mock_article.source = 'benzinga'
        mock_article.url = 'https://example.com'

        article = NewsArticle.from_alpaca(mock_article, 'AAPL')

        assert len(article.summary) == 2000  # Truncated to 2000 chars


class TestSentimentResult:
    """Test SentimentResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SentimentResult(
            id='12345',
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol='AAPL',
            headline='Apple stock rises',
            sentiment_score=0.85,
            sentiment_positive=0.90,
            sentiment_negative=0.05,
            sentiment_neutral=0.05,
            sentiment_label='positive',
            confidence=0.90,
        )

        d = result.to_dict()

        assert d['sentiment_score'] == 0.85
        assert d['sentiment_label'] == 'positive'
        assert d['confidence'] == 0.90


class TestValidation:
    """Test DataFrame validation."""

    def test_validate_valid_dataframe(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({
            'id': ['1', '2'],
            'timestamp': pd.to_datetime(['2024-01-15', '2024-01-16']),
            'symbol': ['AAPL', 'AAPL'],
            'headline': ['Headline 1', 'Headline 2'],
            'source': ['benzinga', 'reuters'],
        })

        assert validate_news_dataframe(df) == True

    def test_validate_missing_column(self):
        """Test validation with missing required column."""
        df = pd.DataFrame({
            'id': ['1', '2'],
            'timestamp': pd.to_datetime(['2024-01-15', '2024-01-16']),
            'symbol': ['AAPL', 'AAPL'],
            # Missing 'headline' and 'source'
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_news_dataframe(df)

    def test_validate_wrong_timestamp_type(self):
        """Test validation with non-datetime timestamp."""
        df = pd.DataFrame({
            'id': ['1', '2'],
            'timestamp': ['2024-01-15', '2024-01-16'],  # String, not datetime
            'symbol': ['AAPL', 'AAPL'],
            'headline': ['Headline 1', 'Headline 2'],
            'source': ['benzinga', 'reuters'],
        })

        with pytest.raises(ValueError, match="timestamp.*datetime"):
            validate_news_dataframe(df)


class TestParquetIO:
    """Test Parquet save/load operations."""

    def test_save_and_load(self):
        """Test saving and loading news DataFrame."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / 'test_news.parquet'

            df = pd.DataFrame({
                'id': ['1', '2'],
                'timestamp': pd.to_datetime(['2024-01-15 10:30', '2024-01-16 11:00'], utc=True),
                'symbol': ['AAPL', 'AAPL'],
                'symbols': [['AAPL'], ['AAPL', 'MSFT']],
                'headline': ['Headline 1', 'Headline 2'],
                'summary': ['Summary 1', 'Summary 2'],
                'source': ['benzinga', 'reuters'],
                'url': ['http://a.com', 'http://b.com'],
                'author': ['Author 1', 'Author 2'],
                'content': ['Content 1', 'Content 2'],
            })

            save_news_parquet(df, str(output_path))

            assert output_path.exists()

            loaded = load_news_parquet(str(output_path))

            assert len(loaded) == 2
            assert loaded['headline'].tolist() == ['Headline 1', 'Headline 2']


class TestPathGeneration:
    """Test path generation functions."""

    def test_news_output_path_yearly(self):
        """Test yearly news output path."""
        path = get_news_output_path('/data', 'AAPL', 2024)

        assert 'news' in path
        assert 'symbol=AAPL' in path
        assert 'year=2024' in path
        assert path.endswith('news.parquet')

    def test_news_output_path_monthly(self):
        """Test monthly news output path."""
        path = get_news_output_path('/data', 'AAPL', 2024, month=6)

        assert 'month=6' in path

    def test_sentiment_output_path(self):
        """Test sentiment output path."""
        path = get_sentiment_output_path('/data', 'AAPL', 2024)

        assert 'sentiment' in path
        assert 'symbol=AAPL' in path
        assert 'year=2024' in path
        assert path.endswith('sentiment.parquet')


class TestNewsDownloadResult:
    """Test NewsDownloadResult dataclass."""

    def test_success_rate(self):
        """Test success rate calculation."""
        result = NewsDownloadResult(
            total_symbols=10,
            succeeded=8,
            failed=2,
            total_articles=150,
            elapsed_seconds=30.0,
        )

        assert result.success_rate == 80.0

    def test_success_rate_zero_symbols(self):
        """Test success rate with zero symbols."""
        result = NewsDownloadResult(
            total_symbols=0,
            succeeded=0,
            failed=0,
            total_articles=0,
            elapsed_seconds=0.0,
        )

        assert result.success_rate == 0.0


class TestNewsDownloader:
    """Test NewsDownloader class."""

    @patch('src.data.news.news_downloader.get_local_storage_dir')
    def test_initialization(self, mock_storage):
        """Test downloader initialization."""
        mock_storage.return_value = '/tmp/test_data'

        downloader = NewsDownloader(
            start_date='2024-01-01',
            end_date='2024-06-30',
            max_workers=2,
            include_content=True,
        )

        assert downloader.start_date == datetime(2024, 1, 1)
        assert downloader.end_date == datetime(2024, 6, 30)
        assert downloader.max_workers == 2
        assert downloader.include_content == True

    @patch('src.data.news.news_downloader.get_local_storage_dir')
    def test_initialization_default_end_date(self, mock_storage):
        """Test default end date is today."""
        mock_storage.return_value = '/tmp/test_data'

        downloader = NewsDownloader(start_date='2024-01-01')

        assert downloader.end_date.date() == datetime.now().date()

    @patch('src.data.news.news_downloader.get_local_storage_dir')
    @patch('src.data.news.news_downloader._get_news_client')
    def test_download_symbol_no_articles(self, mock_client_func, mock_storage):
        """Test downloading symbol with no articles."""
        mock_storage.return_value = '/tmp/test_data'

        # Mock the client - data is in news_set.data['news']
        mock_client = Mock()
        mock_news_set = Mock()
        mock_news_set.data = {'news': []}
        mock_news_set.next_page_token = None
        mock_client.get_news.return_value = mock_news_set
        mock_client_func.return_value = mock_client

        downloader = NewsDownloader(start_date='2024-01-01')
        count = downloader._download_symbol_impl('AAPL')

        assert count == 0

    @patch('src.data.news.news_downloader.get_local_storage_dir')
    @patch('src.data.news.news_downloader._get_news_client')
    @patch('src.data.news.news_downloader.save_news_parquet')
    def test_download_symbol_with_articles(self, mock_save, mock_client_func, mock_storage):
        """Test downloading symbol with articles."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_storage.return_value = tmp_dir

            # Create mock article
            mock_article = Mock()
            mock_article.id = 12345
            mock_article.created_at = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
            mock_article.symbols = ['AAPL']
            mock_article.headline = 'Test Headline'
            mock_article.summary = 'Test summary'
            mock_article.source = 'benzinga'
            mock_article.url = 'https://example.com'

            # Mock the client - data is in news_set.data['news']
            mock_client = Mock()
            mock_news_set = Mock()
            mock_news_set.data = {'news': [mock_article]}
            mock_news_set.next_page_token = None
            mock_client.get_news.return_value = mock_news_set
            mock_client_func.return_value = mock_client

            downloader = NewsDownloader(start_date='2024-01-01')
            count = downloader._download_symbol_impl('AAPL')

            assert count == 1
            mock_save.assert_called_once()


class TestSchemaDefinitions:
    """Test PyArrow schema definitions."""

    def test_news_schema_fields(self):
        """Test NEWS_SCHEMA has expected fields."""
        field_names = [field.name for field in NEWS_SCHEMA]

        assert 'id' in field_names
        assert 'timestamp' in field_names
        assert 'symbol' in field_names
        assert 'symbols' in field_names
        assert 'headline' in field_names
        assert 'summary' in field_names
        assert 'source' in field_names
        assert 'url' in field_names

    def test_sentiment_schema_fields(self):
        """Test SENTIMENT_SCHEMA has expected fields."""
        field_names = [field.name for field in SENTIMENT_SCHEMA]

        assert 'id' in field_names
        assert 'sentiment_score' in field_names
        assert 'sentiment_positive' in field_names
        assert 'sentiment_negative' in field_names
        assert 'sentiment_neutral' in field_names
        assert 'sentiment_label' in field_names
        assert 'confidence' in field_names
