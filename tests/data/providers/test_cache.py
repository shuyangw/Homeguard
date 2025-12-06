"""Tests for DataCache."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.data.providers.cache import DataCache


class TestDataCache:
    """Test DataCache functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [102.0, 103.0],
            'volume': [1000.0, 1100.0]
        }, index=pd.DatetimeIndex([
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0)
        ]))

    def test_store_and_retrieve(self, temp_cache_dir, sample_data):
        """Test basic store and retrieve."""
        cache = DataCache(cache_dir=temp_cache_dir)

        success = cache.store('AAPL', '1D', sample_data)
        assert success is True

        retrieved = cache.retrieve('AAPL', '1D')
        assert retrieved is not None
        assert len(retrieved) == 2

    def test_retrieve_nonexistent_returns_none(self, temp_cache_dir):
        """Test retrieving non-existent data returns None."""
        cache = DataCache(cache_dir=temp_cache_dir)

        result = cache.retrieve('INVALID', '1D')
        assert result is None

    def test_store_none_returns_false(self, temp_cache_dir):
        """Test storing None returns False."""
        cache = DataCache(cache_dir=temp_cache_dir)

        success = cache.store('AAPL', '1D', None)
        assert success is False

    def test_store_empty_returns_false(self, temp_cache_dir):
        """Test storing empty DataFrame returns False."""
        cache = DataCache(cache_dir=temp_cache_dir)

        success = cache.store('AAPL', '1D', pd.DataFrame())
        assert success is False

    def test_daily_vs_intraday_paths(self, temp_cache_dir, sample_data):
        """Test daily and intraday use different directories."""
        cache = DataCache(cache_dir=temp_cache_dir)

        cache.store('AAPL', '1D', sample_data)
        cache.store('AAPL', '1Min', sample_data)

        daily_path = cache._get_cache_path('AAPL', '1D')
        intraday_path = cache._get_cache_path('AAPL', '1Min')

        assert 'daily' in str(daily_path)
        assert 'intraday' in str(intraday_path)
        assert daily_path != intraday_path

    def test_metadata_updated_on_store(self, temp_cache_dir, sample_data):
        """Test metadata is updated after store."""
        cache = DataCache(cache_dir=temp_cache_dir)

        cache.store('AAPL', '1D', sample_data)

        assert 'AAPL_1D' in cache._metadata
        assert 'timestamp' in cache._metadata['AAPL_1D']
        assert cache._metadata['AAPL_1D']['rows'] == 2

    def test_clear_all(self, temp_cache_dir, sample_data):
        """Test clearing all cache entries."""
        cache = DataCache(cache_dir=temp_cache_dir)

        cache.store('AAPL', '1D', sample_data)
        cache.store('MSFT', '1D', sample_data)

        cache.clear()

        assert cache.retrieve('AAPL', '1D') is None
        assert cache.retrieve('MSFT', '1D') is None
        assert len(cache._metadata) == 0

    def test_clear_symbol(self, temp_cache_dir, sample_data):
        """Test clearing specific symbol."""
        cache = DataCache(cache_dir=temp_cache_dir)

        cache.store('AAPL', '1D', sample_data)
        cache.store('MSFT', '1D', sample_data)

        cache.clear(symbol='AAPL')

        assert cache.retrieve('AAPL', '1D') is None
        assert cache.retrieve('MSFT', '1D') is not None

    def test_get_stats(self, temp_cache_dir, sample_data):
        """Test cache statistics."""
        cache = DataCache(cache_dir=temp_cache_dir)

        cache.store('AAPL', '1D', sample_data)
        cache.store('AAPL', '1Min', sample_data)

        stats = cache.get_stats()

        assert stats['daily_files'] == 1
        assert stats['intraday_files'] == 1
        assert stats['metadata_entries'] == 2

    def test_persists_across_instances(self, temp_cache_dir, sample_data):
        """Test cache persists across instances."""
        cache1 = DataCache(cache_dir=temp_cache_dir)
        cache1.store('AAPL', '1D', sample_data)

        # Create new instance pointing to same directory
        cache2 = DataCache(cache_dir=temp_cache_dir)
        result = cache2.retrieve('AAPL', '1D')

        assert result is not None
        assert len(result) == 2
