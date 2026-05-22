"""
Unit Tests for End-of-Day Report Generation.

Ensures that end-of-day reports are generated exactly once per session,
not repeatedly on every check cycle after market close.
"""

import pytest
from datetime import datetime, time as dt_time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

from scripts.trading.run_live_paper_trading import TradingSessionTracker


class TestEndOfDayReport:
    """Test end-of-day report generation."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_tracker(self, temp_log_dir):
        """Create session tracker with temp directory."""
        return TradingSessionTracker(
            log_dir=temp_log_dir,
            strategy_name="TEST",
        )

    def _make_runner(self, temp_log_dir, mock_adapter):
        """Construct a LiveTradingRunner for EOD tests.

        Strategy name is derived from the adapter class name via the
        runner's _persist_strategy logic (e.g. TestLiveAdapter -> 'test').
        We provide the adapter mock with a class whose __name__ ends in
        'LiveAdapter' so the derivation produces 'test'.
        """
        from scripts.trading.run_live_paper_trading import LiveTradingRunner
        return LiveTradingRunner(
            adapter=mock_adapter,
            check_interval=15,
            log_dir=temp_log_dir,
        )

    def test_eod_report_generates_once_after_market_close(self, session_tracker, temp_log_dir):
        """Test that EOD report is generated exactly once after 4 PM.

        Constructs the runner under real `tz.now()` so it can build its
        session paths, then patches only `tz.now` for the EOD check.
        """
        mock_adapter = Mock()
        mock_adapter.__class__.__name__ = 'TestLiveAdapter'
        trader = self._make_runner(temp_log_dir, mock_adapter)
        # Re-target the runner's session tracker at the test fixture so
        # the summary_file path matches what we'll create below.
        trader.session_tracker = session_tracker

        with patch('scripts.trading.run_live_paper_trading.tz.now') as mock_now:
            mock_now.return_value.time.return_value = dt_time(16, 5)  # 4:05 PM ET

            # First check - should generate report (file doesn't exist)
            assert trader._check_for_end_of_day() is True

            # Simulate report generation by creating the file
            session_tracker.summary_file.parent.mkdir(parents=True, exist_ok=True)
            session_tracker.summary_file.write_text("Test report")

            # Second/third checks - should NOT regenerate
            assert trader._check_for_end_of_day() is False
            assert trader._check_for_end_of_day() is False

    def test_eod_report_not_generated_during_market_hours(self, temp_log_dir):
        """Test that EOD report is not generated before 4 PM."""
        mock_adapter = Mock()
        mock_adapter.__class__.__name__ = 'TestLiveAdapter'
        trader = self._make_runner(temp_log_dir, mock_adapter)

        with patch('scripts.trading.run_live_paper_trading.tz.now') as mock_now:
            mock_now.return_value.time.return_value = dt_time(14, 0)  # 2:00 PM ET
            assert trader._check_for_end_of_day() is False

    def test_eod_check_uses_correct_filename(self, session_tracker):
        """Test that EOD check uses the session timestamp filename, not just date."""
        # Verify the summary file includes timestamp
        summary_filename = session_tracker.summary_file.name

        # Should match format: YYYYMMDD_HHMMSS_STRATEGY_summary.md
        assert "_summary.md" in summary_filename

        # Should have more than just date (YYYYMMDD)
        # Date is 8 chars, but filename should be longer due to timestamp
        parts = summary_filename.split('_')
        assert len(parts) >= 4  # [YYYYMMDD, HHMMSS, STRATEGY, summary.md]

    def test_multiple_sessions_same_day_separate_reports(self, temp_log_dir):
        """Test that multiple sessions on same day create separate reports."""
        import time

        # Create first session
        tracker1 = TradingSessionTracker(
            log_dir=temp_log_dir,
            strategy_name="TEST",
        )

        # Sleep > 1s — session_datetime is HHMMSS, second-resolution
        time.sleep(1.1)

        # Create second session
        tracker2 = TradingSessionTracker(
            log_dir=temp_log_dir,
            strategy_name="TEST",
        )

        # Should have different summary files
        assert tracker1.summary_file != tracker2.summary_file

        # Both should be in same date directory
        assert tracker1.summary_file.parent == tracker2.summary_file.parent


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
