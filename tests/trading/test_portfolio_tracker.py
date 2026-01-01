"""
Unit tests for Portfolio Tracker.

Tests:
- Equity hours detection
- Value updates (scheduled and forced)
- State persistence
- Value history tracking
- Performance summary
"""

import pytest
from datetime import datetime, time, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile


class TestPortfolioTrackerEquityHours:
    """Test equity hours detection."""

    @pytest.fixture
    def tracker(self):
        """Create tracker with mocked adapter."""
        with patch('src.trading.portfolio.tracker.tz') as mock_tz:
            mock_adapter = Mock()
            mock_adapter._get_total_portfolio_value.return_value = 10000.0
            mock_adapter._get_current_positions.return_value = {}
            mock_adapter._get_account_balance.return_value = 10000.0
            mock_adapter.get_status.return_value = {'regime': 'bullish'}

            from src.trading.portfolio.tracker import PortfolioTracker
            tracker = PortfolioTracker(mock_adapter)

            # Use yield to keep mock active during test
            yield tracker, mock_tz

    def test_is_equity_hours_during_market(self, tracker):
        """Test equity hours returns True during market hours."""
        tracker_instance, mock_tz = tracker

        # Monday at 10:30 AM EST - use a proper mock that supports weekday()
        mock_now = MagicMock()
        mock_now.weekday.return_value = 0  # Monday
        mock_now.time.return_value = time(10, 30)
        mock_tz.now.return_value = mock_now

        result = tracker_instance.is_equity_hours()

        assert result is True

    def test_is_equity_hours_before_open(self, tracker):
        """Test equity hours returns False before market open."""
        tracker_instance, mock_tz = tracker

        # Monday at 8:00 AM EST (before 9:30 AM)
        mock_now = Mock()
        mock_now.weekday.return_value = 0  # Monday
        mock_now.time.return_value = time(8, 0)
        mock_tz.now.return_value = mock_now

        result = tracker_instance.is_equity_hours()

        assert result is False

    def test_is_equity_hours_after_close(self, tracker):
        """Test equity hours returns False after market close."""
        tracker_instance, mock_tz = tracker

        # Monday at 5:00 PM EST (after 4:00 PM)
        mock_now = Mock()
        mock_now.weekday.return_value = 0  # Monday
        mock_now.time.return_value = time(17, 0)
        mock_tz.now.return_value = mock_now

        result = tracker_instance.is_equity_hours()

        assert result is False

    def test_is_equity_hours_weekend(self, tracker):
        """Test equity hours returns False on weekends."""
        tracker_instance, mock_tz = tracker

        # Saturday at 10:30 AM EST
        mock_now = Mock()
        mock_now.weekday.return_value = 5  # Saturday
        mock_now.time.return_value = time(10, 30)
        mock_tz.now.return_value = mock_now

        result = tracker_instance.is_equity_hours()

        assert result is False


class TestPortfolioTrackerValueUpdates:
    """Test value update behavior."""

    @pytest.fixture
    def tracker(self):
        """Create tracker with mocked adapter."""
        mock_adapter = Mock()
        mock_adapter._get_total_portfolio_value.return_value = 10000.0
        mock_adapter._get_current_positions.return_value = {}
        mock_adapter._get_account_balance.return_value = 10000.0
        mock_adapter.get_status.return_value = {'regime': 'bullish'}

        with patch('src.trading.portfolio.tracker.tz') as mock_tz:
            mock_now = Mock()
            mock_now.weekday.return_value = 0  # Monday
            mock_now.time.return_value = time(10, 30)
            mock_tz.now.return_value = mock_now

            from src.trading.portfolio.tracker import PortfolioTracker

            # Use temp directory for state files
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(PortfolioTracker, 'CACHE_DIR', Path(tmpdir)):
                    tracker = PortfolioTracker(mock_adapter)
                    tracker._mock_tz = mock_tz
                    yield tracker

    def test_update_value_during_equity_hours(self, tracker):
        """Test that value updates during equity hours."""
        # Set up for equity hours
        mock_now = Mock()
        mock_now.weekday.return_value = 0
        mock_now.time.return_value = time(10, 30)
        mock_now.isoformat.return_value = '2024-01-08T10:30:00'
        mock_now.__sub__ = Mock(return_value=timedelta(hours=1))
        tracker._mock_tz.now.return_value = mock_now
        tracker._last_update = None

        value = tracker.update_value()

        assert value == 10000.0
        assert tracker._current_value == 10000.0

    def test_update_value_skips_outside_equity_hours(self, tracker):
        """Test that value updates are skipped outside equity hours."""
        # Set up for outside equity hours
        mock_now = Mock()
        mock_now.weekday.return_value = 5  # Saturday
        mock_now.time.return_value = time(10, 30)
        tracker._mock_tz.now.return_value = mock_now
        tracker._current_value = 9000.0

        value = tracker.update_value()

        # Should return cached value without updating
        assert value == 9000.0

    def test_update_value_force_ignores_equity_hours(self, tracker):
        """Test that forced updates ignore equity hours."""
        # Set up for outside equity hours
        mock_now = Mock()
        mock_now.weekday.return_value = 5  # Saturday
        mock_now.time.return_value = time(10, 30)
        mock_now.isoformat.return_value = '2024-01-13T10:30:00'
        tracker._mock_tz.now.return_value = mock_now

        value = tracker.update_value(force=True)

        assert value == 10000.0
        assert tracker._current_value == 10000.0


class TestPortfolioTrackerStatus:
    """Test status reporting."""

    @pytest.fixture
    def tracker(self):
        """Create tracker with mocked adapter."""
        mock_adapter = Mock()
        mock_adapter._get_total_portfolio_value.return_value = 10000.0
        mock_adapter._get_current_positions.return_value = {
            'BTC/USD': Decimal('0.1'),
            'ETH/USD': Decimal('1.0')
        }
        mock_adapter._get_account_balance.return_value = 5000.0
        mock_adapter.get_status.return_value = {'regime': 'bullish'}

        with patch('src.trading.portfolio.tracker.tz') as mock_tz:
            mock_now = Mock()
            mock_now.weekday.return_value = 0
            mock_now.time.return_value = time(10, 30)
            mock_tz.now.return_value = mock_now

            from src.trading.portfolio.tracker import PortfolioTracker

            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(PortfolioTracker, 'CACHE_DIR', Path(tmpdir)):
                    tracker = PortfolioTracker(mock_adapter)
                    tracker._current_value = 10000.0
                    yield tracker

    def test_get_status_returns_all_fields(self, tracker):
        """Test that status includes all expected fields."""
        status = tracker.get_status()

        assert 'current_value' in status
        assert 'last_update' in status
        assert 'is_equity_hours' in status
        assert 'positions' in status
        assert 'positions_count' in status
        assert 'cash' in status
        assert 'regime' in status

    def test_get_status_positions_converted_to_float(self, tracker):
        """Test that positions are converted to float."""
        status = tracker.get_status()

        for symbol, qty in status['positions'].items():
            assert isinstance(qty, float)


class TestPortfolioTrackerHistory:
    """Test value history tracking."""

    @pytest.fixture
    def tracker(self):
        """Create tracker with mocked adapter."""
        mock_adapter = Mock()
        mock_adapter._get_total_portfolio_value.return_value = 10000.0
        mock_adapter._get_current_positions.return_value = {}
        mock_adapter._get_account_balance.return_value = 10000.0
        mock_adapter.get_status.return_value = {'regime': 'bullish'}

        with patch('src.trading.portfolio.tracker.tz') as mock_tz:
            mock_now = Mock()
            mock_now.weekday.return_value = 0
            mock_now.time.return_value = time(10, 30)
            mock_now.isoformat.return_value = '2024-01-08T10:30:00'
            mock_now.__sub__ = Mock(return_value=timedelta(hours=1))
            mock_tz.now.return_value = mock_now

            from src.trading.portfolio.tracker import PortfolioTracker

            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(PortfolioTracker, 'CACHE_DIR', Path(tmpdir)):
                    with patch.object(PortfolioTracker, 'STATE_FILE', Path(tmpdir) / 'state.pkl'):
                        with patch.object(PortfolioTracker, 'VALUE_HISTORY_FILE', Path(tmpdir) / 'history.parquet'):
                            tracker = PortfolioTracker(mock_adapter)
                            tracker._mock_tz = mock_tz
                            yield tracker

    def test_update_adds_to_history(self, tracker):
        """Test that updates add entries to history."""
        tracker._last_update = None

        tracker.update_value(force=True)

        assert len(tracker._value_history) == 1
        assert tracker._value_history[0]['value'] == 10000.0

    def test_get_value_history_returns_dataframe(self, tracker):
        """Test that get_value_history returns DataFrame."""
        import pandas as pd

        tracker._value_history = [
            {'timestamp': '2024-01-08T10:00:00', 'value': 9500, 'cash': 5000, 'positions_count': 2},
            {'timestamp': '2024-01-08T11:00:00', 'value': 10000, 'cash': 5000, 'positions_count': 2},
        ]

        df = tracker.get_value_history()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'value' in df.columns

    def test_get_value_history_filters_by_days(self, tracker):
        """Test that history can be filtered by days."""
        import pandas as pd
        from datetime import datetime, timedelta

        now = datetime.now()
        tracker._value_history = [
            {'timestamp': (now - timedelta(days=10)).strftime('%Y-%m-%dT%H:%M:%S'), 'value': 9000, 'cash': 5000, 'positions_count': 2, 'is_equity_hours': True},
            {'timestamp': (now - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S'), 'value': 10000, 'cash': 5000, 'positions_count': 2, 'is_equity_hours': True},
        ]

        df = tracker.get_value_history(days=5)

        # Should only include the recent entry
        assert len(df) == 1


class TestPortfolioTrackerPerformance:
    """Test performance summary."""

    @pytest.fixture
    def tracker(self):
        """Create tracker with history."""
        mock_adapter = Mock()
        mock_adapter._get_total_portfolio_value.return_value = 12000.0
        mock_adapter._get_current_positions.return_value = {}
        mock_adapter._get_account_balance.return_value = 12000.0
        mock_adapter.get_status.return_value = {'regime': 'bullish'}

        from src.trading.portfolio.tracker import PortfolioTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PortfolioTracker, 'CACHE_DIR', Path(tmpdir)):
                tracker = PortfolioTracker(mock_adapter)

                # Add sample history
                tracker._value_history = [
                    {'timestamp': '2024-01-01T10:00:00', 'value': 10000, 'cash': 5000, 'positions_count': 2},
                    {'timestamp': '2024-01-02T10:00:00', 'value': 11000, 'cash': 5000, 'positions_count': 2},
                    {'timestamp': '2024-01-03T10:00:00', 'value': 10500, 'cash': 5000, 'positions_count': 2},
                    {'timestamp': '2024-01-04T10:00:00', 'value': 12000, 'cash': 5000, 'positions_count': 2},
                ]

                yield tracker

    def test_get_performance_summary(self, tracker):
        """Test performance summary calculation."""
        summary = tracker.get_performance_summary()

        assert 'first_value' in summary
        assert 'current_value' in summary
        assert 'peak_value' in summary
        assert 'total_return_pct' in summary
        assert 'max_drawdown_pct' in summary

        assert summary['first_value'] == 10000
        assert summary['current_value'] == 12000
        assert summary['peak_value'] == 12000
        assert summary['total_return_pct'] == pytest.approx(20.0)  # (12000-10000)/10000 * 100

    def test_get_performance_summary_insufficient_data(self):
        """Test performance summary with insufficient data."""
        mock_adapter = Mock()

        from src.trading.portfolio.tracker import PortfolioTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PortfolioTracker, 'CACHE_DIR', Path(tmpdir)):
                tracker = PortfolioTracker(mock_adapter)
                tracker._value_history = []

                summary = tracker.get_performance_summary()

                assert 'error' in summary


class TestPortfolioTrackerReset:
    """Test reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        mock_adapter = Mock()

        from src.trading.portfolio.tracker import PortfolioTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            state_file = cache_dir / 'state.pkl'
            history_file = cache_dir / 'history.parquet'

            with patch.object(PortfolioTracker, 'CACHE_DIR', cache_dir):
                with patch.object(PortfolioTracker, 'STATE_FILE', state_file):
                    with patch.object(PortfolioTracker, 'VALUE_HISTORY_FILE', history_file):
                        tracker = PortfolioTracker(mock_adapter)
                        tracker._current_value = 10000.0
                        tracker._value_history = [{'value': 10000}]
                        tracker._update_count = 5

                        tracker.reset()

                        assert tracker._current_value == 0.0
                        assert tracker._value_history == []
                        assert tracker._update_count == 0
                        assert tracker._last_update is None
