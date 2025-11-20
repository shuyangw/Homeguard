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
            strategy_name="TEST",
            log_dir=temp_log_dir,
            flush_interval_hours=1.0
        )

    def test_eod_report_generates_once_after_market_close(self, session_tracker, temp_log_dir):
        """Test that EOD report is generated exactly once after 4 PM."""
        # Mock time to be 4:05 PM EST (after market close)
        with patch('scripts.trading.run_live_paper_trading.datetime') as mock_dt:
            # First check at 4:05 PM - should return True (no report exists yet)
            mock_now_est = MagicMock()
            mock_now_est.time.return_value = dt_time(16, 5)  # 4:05 PM
            mock_now_est.strftime.return_value = '20251119'

            mock_dt.now.return_value.astimezone.return_value = mock_now_est

            # Create the parent class instance for testing
            from scripts.trading.run_live_paper_trading import LivePaperTrading

            mock_broker = Mock()
            mock_adapter = Mock()

            trader = LivePaperTrading(
                adapter=mock_adapter,
                log_dir=temp_log_dir,
                strategy_name="TEST",
                check_interval=15
            )

            # First check - should generate report (file doesn't exist)
            assert trader._check_for_end_of_day() == True

            # Simulate report generation by creating the file
            session_tracker.summary_file.parent.mkdir(parents=True, exist_ok=True)
            session_tracker.summary_file.write_text("Test report")

            # Second check - should NOT generate report (file exists)
            assert trader._check_for_end_of_day() == False

            # Third check - should still NOT generate report
            assert trader._check_for_end_of_day() == False

    def test_eod_report_not_generated_during_market_hours(self, temp_log_dir):
        """Test that EOD report is not generated before 4 PM."""
        with patch('scripts.trading.run_live_paper_trading.datetime') as mock_dt:
            # Mock time to be 2:00 PM EST (during market hours)
            mock_now_est = MagicMock()
            mock_now_est.time.return_value = dt_time(14, 0)  # 2:00 PM

            mock_dt.now.return_value.astimezone.return_value = mock_now_est

            from scripts.trading.run_live_paper_trading import LivePaperTrading

            mock_broker = Mock()
            mock_adapter = Mock()

            trader = LivePaperTrading(
                adapter=mock_adapter,
                log_dir=temp_log_dir,
                strategy_name="TEST",
                check_interval=15
            )

            # Should not generate report during market hours
            assert trader._check_for_end_of_day() == False

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
            strategy_name="TEST",
            log_dir=temp_log_dir,
            flush_interval_hours=1.0
        )

        # Small delay to ensure different timestamp
        time.sleep(0.1)

        # Create second session
        tracker2 = TradingSessionTracker(
            strategy_name="TEST",
            log_dir=temp_log_dir,
            flush_interval_hours=1.0
        )

        # Should have different summary files
        assert tracker1.summary_file != tracker2.summary_file

        # Both should be in same date directory
        assert tracker1.summary_file.parent == tracker2.summary_file.parent


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
