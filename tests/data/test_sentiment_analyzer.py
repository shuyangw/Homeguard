"""
Unit Tests for Sentiment Analyzer and Cache.

Tests sentiment score calculation, caching, and integration.
Uses MockSentimentAnalyzer to avoid loading the actual model.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.data.news.sentiment_analyzer import (
    SentimentScore,
    SentimentAnalyzer,
    MockSentimentAnalyzer,
    get_sentiment_analyzer,
)
from src.data.news.sentiment_cache import SentimentCache


class TestSentimentScore:
    """Test SentimentScore dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = SentimentScore(
            text='Apple stock rises',
            sentiment_label='positive',
            sentiment_score=0.85,
            positive_prob=0.90,
            negative_prob=0.05,
            neutral_prob=0.05,
            confidence=0.90,
        )

        result = score.to_dict()

        assert result['text'] == 'Apple stock rises'
        assert result['sentiment_label'] == 'positive'
        assert result['sentiment_score'] == 0.85
        assert result['sentiment_positive'] == 0.90
        assert result['sentiment_negative'] == 0.05
        assert result['confidence'] == 0.90

    def test_sentiment_score_range(self):
        """Test that sentiment score is in valid range."""
        # Positive sentiment
        score = SentimentScore(
            text='Great earnings',
            sentiment_label='positive',
            sentiment_score=0.8,
            positive_prob=0.9,
            negative_prob=0.1,
            neutral_prob=0.0,
            confidence=0.9,
        )
        assert -1 <= score.sentiment_score <= 1

        # Negative sentiment
        score = SentimentScore(
            text='Stock crashes',
            sentiment_label='negative',
            sentiment_score=-0.7,
            positive_prob=0.1,
            negative_prob=0.8,
            neutral_prob=0.1,
            confidence=0.8,
        )
        assert -1 <= score.sentiment_score <= 1


class TestMockSentimentAnalyzer:
    """Test MockSentimentAnalyzer for testing without model."""

    def test_analyze_single(self):
        """Test single text analysis."""
        analyzer = MockSentimentAnalyzer(seed=42)
        result = analyzer.analyze("Apple stock rises")

        assert result.text == "Apple stock rises"
        assert result.sentiment_label in ['positive', 'negative', 'neutral']
        assert -1 <= result.sentiment_score <= 1
        assert 0 <= result.positive_prob <= 1
        assert 0 <= result.negative_prob <= 1
        assert 0 <= result.neutral_prob <= 1
        assert 0 <= result.confidence <= 1

    def test_analyze_batch(self):
        """Test batch analysis."""
        analyzer = MockSentimentAnalyzer(seed=42)
        texts = [
            "Stock rises on earnings",
            "Company reports loss",
            "Market remains stable",
        ]

        results = analyzer.analyze_batch(texts)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.text == texts[i]
            assert result.sentiment_label in ['positive', 'negative', 'neutral']

    def test_analyze_empty_batch(self):
        """Test batch analysis with empty list."""
        analyzer = MockSentimentAnalyzer(seed=42)
        results = analyzer.analyze_batch([])
        assert results == []

    def test_reproducibility_with_seed(self):
        """Test that seeded analyzer gives reproducible results."""
        analyzer1 = MockSentimentAnalyzer(seed=123)
        analyzer2 = MockSentimentAnalyzer(seed=123)

        text = "Test headline"

        result1 = analyzer1.analyze(text)
        result2 = analyzer2.analyze(text)

        assert result1.sentiment_score == result2.sentiment_score
        assert result1.sentiment_label == result2.sentiment_label

    def test_analyze_dataframe(self):
        """Test DataFrame analysis."""
        analyzer = MockSentimentAnalyzer(seed=42)

        df = pd.DataFrame({
            'headline': ['Stock rises', 'Stock falls', 'Market stable'],
            'other_col': [1, 2, 3],
        })

        result = analyzer.analyze_dataframe(df, text_column='headline')

        assert 'sentiment_score' in result.columns
        assert 'sentiment_label' in result.columns
        assert 'sentiment_positive' in result.columns
        assert 'sentiment_negative' in result.columns
        assert 'sentiment_neutral' in result.columns
        assert 'sentiment_confidence' in result.columns
        assert len(result) == 3

    def test_analyze_dataframe_empty(self):
        """Test DataFrame analysis with empty DataFrame."""
        analyzer = MockSentimentAnalyzer(seed=42)
        df = pd.DataFrame()

        result = analyzer.analyze_dataframe(df)
        assert result.empty

    def test_analyze_dataframe_custom_prefix(self):
        """Test DataFrame analysis with custom prefix."""
        analyzer = MockSentimentAnalyzer(seed=42)

        df = pd.DataFrame({
            'headline': ['Test'],
        })

        result = analyzer.analyze_dataframe(df, prefix='sent_')

        assert 'sent_score' in result.columns
        assert 'sent_label' in result.columns


class TestGetSentimentAnalyzer:
    """Test factory function."""

    def test_get_mock_analyzer(self):
        """Test getting mock analyzer."""
        analyzer = get_sentiment_analyzer(use_mock=True, seed=42)
        assert isinstance(analyzer, MockSentimentAnalyzer)

    def test_get_real_analyzer_returns_sentiment_analyzer(self):
        """Test getting real analyzer type."""
        analyzer = get_sentiment_analyzer(use_mock=False)
        assert isinstance(analyzer, SentimentAnalyzer)


class TestSentimentAnalyzerLazyInit:
    """Test SentimentAnalyzer lazy initialization."""

    def test_not_initialized_at_creation(self):
        """Test that model is not loaded at creation time."""
        analyzer = SentimentAnalyzer()
        assert analyzer._initialized == False

    def test_initialization_deferred(self):
        """Test that initialization is deferred until first use."""
        analyzer = SentimentAnalyzer()
        # Should not raise even without transformers installed
        # because model loading is deferred


class TestSentimentCache:
    """Test SentimentCache functionality."""

    def test_path_generation(self):
        """Test path generation for news and sentiment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SentimentCache(storage_dir=tmp_dir)

            news_path = cache.get_news_path('AAPL', 2024)
            sentiment_path = cache.get_sentiment_path('AAPL', 2024)

            assert 'news' in news_path
            assert 'symbol=AAPL' in news_path
            assert 'year=2024' in news_path

            assert 'sentiment' in sentiment_path
            assert 'symbol=AAPL' in sentiment_path
            assert 'year=2024' in sentiment_path

    def test_has_news_false(self):
        """Test has_news returns False when no data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SentimentCache(storage_dir=tmp_dir)
            assert cache.has_news('AAPL', 2024) == False

    def test_has_sentiment_false(self):
        """Test has_sentiment returns False when no data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SentimentCache(storage_dir=tmp_dir)
            assert cache.has_sentiment('AAPL', 2024) == False

    def test_list_available_news_empty(self):
        """Test list_available_news with no data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SentimentCache(storage_dir=tmp_dir)
            result = cache.list_available_news()
            assert result == []

    def test_list_missing_sentiment(self):
        """Test list_missing_sentiment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SentimentCache(storage_dir=tmp_dir)

            # Create mock news data
            news_dir = Path(tmp_dir) / 'news' / 'symbol=AAPL' / 'year=2024'
            news_dir.mkdir(parents=True)
            (news_dir / 'news.parquet').touch()

            missing = cache.list_missing_sentiment()
            assert len(missing) == 1
            assert missing[0]['symbol'] == 'AAPL'
            assert missing[0]['year'] == 2024
            assert missing[0]['has_sentiment'] == False

    @patch('src.data.news.sentiment_cache.load_news_parquet')
    @patch('src.data.news.sentiment_cache.pq.write_table')
    def test_compute_and_save(self, mock_write, mock_load_news):
        """Test compute_and_save with mock analyzer."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create news directory
            news_dir = Path(tmp_dir) / 'news' / 'symbol=AAPL' / 'year=2024'
            news_dir.mkdir(parents=True)
            (news_dir / 'news.parquet').touch()

            # Mock news data
            mock_news_df = pd.DataFrame({
                'id': ['1', '2'],
                'timestamp': pd.to_datetime(['2024-01-15', '2024-01-16'], utc=True),
                'symbol': ['AAPL', 'AAPL'],
                'headline': ['Stock rises', 'Stock falls'],
            })
            mock_load_news.return_value = mock_news_df

            # Create cache with mock analyzer
            mock_analyzer = MockSentimentAnalyzer(seed=42)
            cache = SentimentCache(storage_dir=tmp_dir, analyzer=mock_analyzer)

            count = cache.compute_and_save('AAPL', 2024)

            assert count == 2
            mock_write.assert_called_once()

    def test_load_range_empty(self):
        """Test load_range with no data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SentimentCache(storage_dir=tmp_dir)
            result = cache.load_range(
                'AAPL',
                datetime(2024, 1, 1),
                datetime(2024, 12, 31)
            )
            assert result.empty

    @patch('src.data.news.sentiment_cache.pq.read_table')
    def test_get_daily_sentiment(self, mock_read):
        """Test daily sentiment aggregation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create sentiment directory
            sent_dir = Path(tmp_dir) / 'sentiment' / 'symbol=AAPL' / 'year=2024'
            sent_dir.mkdir(parents=True)
            (sent_dir / 'sentiment.parquet').touch()

            # Mock sentiment data
            mock_df = pd.DataFrame({
                'id': ['1', '2', '3'],
                'timestamp': pd.to_datetime([
                    '2024-01-15 09:30:00',
                    '2024-01-15 10:30:00',  # Same day
                    '2024-01-16 09:30:00',
                ], utc=True),
                'symbol': ['AAPL', 'AAPL', 'AAPL'],
                'headline': ['H1', 'H2', 'H3'],
                'sentiment_score': [0.5, 0.7, -0.3],
                'sentiment_positive': [0.6, 0.8, 0.1],
                'sentiment_negative': [0.1, 0.1, 0.4],
                'sentiment_label': ['positive', 'positive', 'negative'],
                'confidence': [0.6, 0.8, 0.4],
            })

            mock_table = Mock()
            mock_table.to_pandas.return_value = mock_df
            mock_read.return_value = mock_table

            cache = SentimentCache(storage_dir=tmp_dir)
            result = cache.get_daily_sentiment(
                'AAPL',
                datetime(2024, 1, 1),
                datetime(2024, 12, 31),
                aggregation='mean'
            )

            # Should have 2 days
            assert len(result) == 2
            assert 'sentiment_score' in result.columns
            assert 'article_count' in result.columns

            # Jan 15 should have 2 articles averaged
            jan15 = result.loc[pd.Timestamp('2024-01-15')]
            assert jan15['article_count'] == 2
            assert jan15['sentiment_score'] == pytest.approx(0.6, abs=0.01)


class TestProbabilityConstraints:
    """Test that probabilities sum to 1 and are in valid range."""

    def test_mock_probabilities_sum_to_one(self):
        """Test that mock probabilities sum to approximately 1."""
        analyzer = MockSentimentAnalyzer(seed=42)

        for _ in range(10):
            result = analyzer.analyze("Test text")
            prob_sum = result.positive_prob + result.negative_prob + result.neutral_prob
            assert abs(prob_sum - 1.0) < 0.001

    def test_confidence_equals_max_prob(self):
        """Test that confidence equals the max probability."""
        analyzer = MockSentimentAnalyzer(seed=42)

        for _ in range(10):
            result = analyzer.analyze("Test text")
            max_prob = max(result.positive_prob, result.negative_prob, result.neutral_prob)
            assert result.confidence == pytest.approx(max_prob, abs=0.001)


class TestSentimentScoreCalculation:
    """Test sentiment score calculation logic."""

    def test_positive_minus_negative(self):
        """Test that sentiment_score = positive_prob - negative_prob."""
        analyzer = MockSentimentAnalyzer(seed=42)

        for _ in range(10):
            result = analyzer.analyze("Test text")
            expected_score = result.positive_prob - result.negative_prob
            assert result.sentiment_score == pytest.approx(expected_score, abs=0.001)

    def test_score_range(self):
        """Test sentiment score is always between -1 and 1."""
        analyzer = MockSentimentAnalyzer(seed=42)

        for _ in range(100):
            result = analyzer.analyze("Random text for testing")
            assert -1 <= result.sentiment_score <= 1
