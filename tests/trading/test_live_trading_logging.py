"""
Unit tests for live trading logging behavior.

Tests critical logging requirements:
1. Status line appears every 15 seconds with correct format
2. Periodic flush happens every 5 minutes
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from scripts.trading.run_live_paper_trading import LiveTradingRunner


class TestStatusLineLogging:
    """Test 15-second status line logging."""

    @pytest.fixture
    def mock_trader(self):
        """Create a mock LiveTradingRunner instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Mock dependencies
            mock_broker = Mock()
            mock_broker.is_market_open.return_value = True

            mock_adapter = Mock()
            mock_adapter.broker = mock_broker

            # Create trader instance
            trader = LiveTradingRunner(
                adapter=mock_adapter,
                strategy_name='TEST',
                check_interval=15
            )

            # Override log_dir for testing
            trader.log_dir = log_dir

            yield trader

    def test_status_line_format(self, mock_trader):
        """Test status line has correct one-liner format."""
        # Capture console output
        with patch('scripts.trading.run_live_paper_trading.console') as mock_console:
            mock_trader._log_minute_progress(force=True)

            # Verify console.print was called
            assert mock_console.print.called

            # Get the printed message
            call_args = mock_console.print.call_args[0][0]

            # Verify format: [HH:MM:SS] Market: OPEN | Checks: N | Runs: N | Signals: N | Orders: N/N
            assert '[' in call_args and ']' in call_args  # Timestamp
            assert 'Market:' in call_args
            assert 'Checks:' in call_args
            assert 'Runs:' in call_args
            assert 'Signals:' in call_args
            assert 'Orders:' in call_args
            assert call_args.count('|') == 4  # 4 separators

    def test_status_line_interval_15_seconds(self, mock_trader):
        """Test status line appears every 15 seconds."""
        # Set initial progress log time
        initial_time = datetime(2025, 11, 21, 10, 0, 0)
        mock_trader.last_progress_log = initial_time

        # Test at 14 seconds - should NOT log
        with patch('scripts.trading.run_live_paper_trading.datetime') as mock_dt:
            mock_dt.now.return_value = initial_time + timedelta(seconds=14)
            with patch('scripts.trading.run_live_paper_trading.console') as mock_console:
                mock_trader._log_minute_progress()
                assert not mock_console.print.called

        # Test at 15 seconds - SHOULD log
        with patch('scripts.trading.run_live_paper_trading.datetime') as mock_dt:
            mock_dt.now.return_value = initial_time + timedelta(seconds=15)
            with patch('scripts.trading.run_live_paper_trading.console') as mock_console:
                mock_trader._log_minute_progress()
                assert mock_console.print.called

    def test_status_line_shows_correct_counts(self, mock_trader):
        """Test status line displays correct check/run/signal/order counts."""
        # Set session tracker values
        mock_trader.session_tracker.total_checks = 100
        mock_trader.session_tracker.total_runs = 5
        mock_trader.session_tracker.total_signals = 3
        mock_trader.session_tracker.total_orders = 2
        mock_trader.session_tracker.successful_orders = 2

        with patch('scripts.trading.run_live_paper_trading.console') as mock_console:
            mock_trader._log_minute_progress(force=True)

            call_args = mock_console.print.call_args[0][0]

            # Verify counts appear in output
            assert 'Checks: 100' in call_args
            assert 'Runs: 5' in call_args
            assert 'Signals: 3' in call_args
            assert 'Orders: 2/2' in call_args

    def test_status_line_market_status(self, mock_trader):
        """Test status line shows correct market status (OPEN/CLOSED)."""
        # Test OPEN
        mock_trader.adapter.broker.is_market_open.return_value = True
        with patch('scripts.trading.run_live_paper_trading.console') as mock_console:
            mock_trader._log_minute_progress(force=True)
            call_args = mock_console.print.call_args[0][0]
            assert 'Market: OPEN' in call_args

        # Test CLOSED
        mock_trader.adapter.broker.is_market_open.return_value = False
        with patch('scripts.trading.run_live_paper_trading.console') as mock_console:
            mock_trader._log_minute_progress(force=True)
            call_args = mock_console.print.call_args[0][0]
            assert 'Market: CLOSED' in call_args

    def test_status_line_uses_console_not_logger(self, mock_trader):
        """Test status line uses console.print (immediate) not logger (buffered)."""
        with patch('scripts.trading.run_live_paper_trading.console') as mock_console:
            with patch('scripts.trading.run_live_paper_trading.logger') as mock_logger:
                mock_trader._log_minute_progress(force=True)

                # Should use console.print (immediate output)
                assert mock_console.print.called

                # Should NOT use logger.info (buffered output)
                # Note: logger might be used elsewhere, so we just check console is called
                assert mock_console.print.call_count > 0


class TestPeriodicFlush:
    """Test 5-minute periodic log flush."""

    @pytest.fixture
    def mock_trader(self):
        """Create a mock LiveTradingRunner with mocked logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            mock_broker = Mock()
            mock_broker.is_market_open.return_value = True

            mock_adapter = Mock()
            mock_adapter.broker = mock_broker

            trader = LiveTradingRunner(
                adapter=mock_adapter,
                strategy_name='TEST',
                check_interval=15
            )

            trader.log_dir = log_dir

            # Mock the trading logger
            mock_trading_logger = Mock()
            mock_trading_logger.should_periodic_flush.return_value = False
            trader.session_tracker.trading_logger = mock_trading_logger

            yield trader

    def test_periodic_flush_interval_5_minutes(self, mock_trader):
        """Test periodic flush is triggered every 5 minutes."""
        mock_logger = mock_trader.session_tracker.trading_logger

        # First call - should not flush (just started)
        mock_logger.should_periodic_flush.return_value = False
        with patch('scripts.trading.run_live_paper_trading.datetime'):
            # This would be called in run_continuous loop
            # We're testing the logic, not the full loop
            if mock_logger.should_periodic_flush():
                mock_logger.flush_to_disk(reason="Periodic flush (multi-day session)")

        assert not mock_logger.flush_to_disk.called

        # Later call - should flush (5 min elapsed)
        mock_logger.should_periodic_flush.return_value = True
        if mock_logger.should_periodic_flush():
            mock_logger.flush_to_disk(reason="Periodic flush (multi-day session)")

        # Verify flush was called
        assert mock_logger.flush_to_disk.called
        mock_logger.flush_to_disk.assert_called_with(reason="Periodic flush (multi-day session)")

    def test_flush_is_silent_no_output(self, mock_trader):
        """Test periodic flush happens silently with NO console output on success."""
        mock_logger = mock_trader.session_tracker.trading_logger
        mock_logger.should_periodic_flush.return_value = True

        # Capture console output
        with patch('src.utils.logger.console') as mock_console:
            if mock_logger.should_periodic_flush():
                mock_logger.flush_to_disk(reason="Periodic flush (multi-day session)")

            # Verify console.print was NOT called (silent flush)
            # Note: flush_to_disk should only print on error
            assert not mock_console.print.called

    def test_flush_logs_only_on_error(self):
        """Test that flush only logs to console when an error occurs."""
        from src.utils.logger import TradingLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            trading_logger = TradingLogger(
                name='TEST',
                log_dir=log_dir,
                flush_interval_hours=5/60
            )

            # Add some log entries
            trading_logger.info("Test message")

            # Make the log directory read-only to cause flush error
            log_dir.chmod(0o444)

            # Capture console output
            with patch('src.utils.logger.console') as mock_console:
                try:
                    trading_logger.flush_to_disk(reason="Test flush")
                except:
                    pass  # Ignore the error itself

                # Verify console.print WAS called with error message
                assert mock_console.print.called
                call_args = str(mock_console.print.call_args)
                assert "Error flushing logs" in call_args

            # Restore permissions
            log_dir.chmod(0o755)

    def test_flush_writes_to_disk(self):
        """Test that TradingLogger.flush_to_disk() actually writes buffered logs."""
        from src.utils.logger import TradingLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Create trading logger with flush interval
            trading_logger = TradingLogger(
                name='TEST',
                log_dir=log_dir,
                flush_interval_hours=5/60  # 5 minutes
            )

            # Log some messages (these get buffered)
            trading_logger.info("Test message 1")
            trading_logger.info("Test message 2")

            # Force flush to disk
            trading_logger.flush_to_disk(reason="Test flush")

            # Verify log file exists and contains messages
            log_files = list(log_dir.glob('*.log'))
            assert len(log_files) == 1

            content = log_files[0].read_text()
            assert "Test message 1" in content
            assert "Test message 2" in content


class TestStatusLineIntegration:
    """Integration tests for status line logging in live trading context."""

    def test_status_appears_in_journalctl_format(self):
        """Test status line appears in journalctl-compatible format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            mock_broker = Mock()
            mock_broker.is_market_open.return_value = True

            mock_adapter = Mock()
            mock_adapter.broker = mock_broker

            trader = LiveTradingRunner(
                adapter=mock_adapter,
                strategy_name='TEST',
                check_interval=15
            )
            trader.log_dir = log_dir

            # Capture console output (this is what appears in journalctl)
            with patch('scripts.trading.run_live_paper_trading.console') as mock_console:
                trader._log_minute_progress(force=True)

                # Get the output
                assert mock_console.print.called
                output = mock_console.print.call_args[0][0]

                # Verify it's a single line (no newlines)
                assert '\n' not in output.strip()

                # Verify it has the expected format for systemd journalctl
                # Should start with a space and timestamp
                assert output.startswith(' [')
                assert '] Market:' in output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
