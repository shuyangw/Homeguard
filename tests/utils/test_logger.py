"""
Unit tests for centralized logging module.

Tests the Logger, CSVLogger, and TradingLogger classes.
"""

import pytest
import tempfile
import csv
from pathlib import Path
from datetime import datetime

from src.utils.logger import (
    Logger,
    CSVLogger,
    TradingLogger,
    get_logger,
    get_trading_logger
)


class TestLogger:
    """Test base Logger class."""

    def test_logger_initialization_no_file(self):
        """Test logger can be initialized without file logging."""
        logger = Logger()
        assert logger.log_file is None
        assert logger.console is not None

    def test_logger_initialization_with_file(self):
        """Test logger can be initialized with file logging."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = Path(f.name)

        try:
            logger = Logger(log_file=log_file)
            assert logger.log_file == log_file

            # Log something
            logger.info("Test message")

            # Verify file was written
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_logger_success_method(self):
        """Test success logging method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = Path(f.name)

        try:
            logger = Logger(log_file=log_file)
            logger.success("Success message")

            content = log_file.read_text()
            assert "Success message" in content

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_logger_error_method(self):
        """Test error logging method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = Path(f.name)

        try:
            logger = Logger(log_file=log_file)
            logger.error("Error message")

            content = log_file.read_text()
            assert "Error message" in content

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_logger_warning_method(self):
        """Test warning logging method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = Path(f.name)

        try:
            logger = Logger(log_file=log_file)
            logger.warning("Warning message")

            content = log_file.read_text()
            assert "Warning message" in content

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_logger_info_method(self):
        """Test info logging method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = Path(f.name)

        try:
            logger = Logger(log_file=log_file)
            logger.info("Info message")

            content = log_file.read_text()
            assert "Info message" in content

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_get_logger_function(self):
        """Test get_logger() convenience function."""
        logger = get_logger()
        assert isinstance(logger, Logger)
        assert logger.log_file is None

    def test_get_logger_with_file(self):
        """Test get_logger() with file parameter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = Path(f.name)

        try:
            logger = get_logger(log_file=log_file)
            assert isinstance(logger, Logger)
            assert logger.log_file == log_file

        finally:
            if log_file.exists():
                log_file.unlink()


class TestCSVLogger:
    """Test CSVLogger class."""

    def test_csv_logger_initialization(self):
        """Test CSVLogger initialization creates file with headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / 'test.csv'
            headers = ['timestamp', 'symbol', 'price']

            csv_logger = CSVLogger(csv_file, headers)

            # Verify file was created
            assert csv_file.exists()

            # Verify headers were written
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                first_row = next(reader)
                assert first_row == headers

    def test_csv_logger_creates_directory(self):
        """Test CSVLogger creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / 'subdir' / 'nested' / 'test.csv'
            headers = ['col1', 'col2']

            csv_logger = CSVLogger(csv_file, headers)

            # Verify file and directories were created
            assert csv_file.exists()
            assert csv_file.parent.exists()

    def test_csv_logger_log_row(self):
        """Test CSVLogger.log_row() appends data correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / 'test.csv'
            headers = ['timestamp', 'symbol', 'price']

            csv_logger = CSVLogger(csv_file, headers)

            # Log some rows
            csv_logger.log_row(['2025-11-14T10:00:00', 'TQQQ', 45.23])
            csv_logger.log_row(['2025-11-14T10:01:00', 'SQQQ', 12.34])

            # Read back and verify
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 3  # Header + 2 data rows
            assert rows[0] == headers
            assert rows[1] == ['2025-11-14T10:00:00', 'TQQQ', '45.23']
            assert rows[2] == ['2025-11-14T10:01:00', 'SQQQ', '12.34']

    def test_csv_logger_multiple_writes(self):
        """Test CSVLogger handles multiple log_row() calls correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / 'test.csv'
            headers = ['id', 'value']

            csv_logger = CSVLogger(csv_file, headers)

            # Write 10 rows
            for i in range(10):
                csv_logger.log_row([i, i * 100])

            # Verify all rows were written
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 11  # Header + 10 data rows

            # Verify data
            for i in range(10):
                assert rows[i + 1] == [str(i), str(i * 100)]


class TestTradingLogger:
    """Test TradingLogger class."""

    def test_trading_logger_initialization_no_dir(self):
        """Test TradingLogger can be initialized without log directory."""
        trading_logger = TradingLogger('TestStrategy')
        assert trading_logger.name == 'TestStrategy'
        assert trading_logger.log_dir is None
        assert isinstance(trading_logger.logger, Logger)

    def test_trading_logger_initialization_with_dir(self):
        """Test TradingLogger initialization with log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            # Use buffer_logs=False to enable immediate file writing
            trading_logger = TradingLogger('TestStrategy', log_dir, buffer_logs=False)

            assert trading_logger.name == 'TestStrategy'
            assert trading_logger.log_dir == log_dir
            assert isinstance(trading_logger.logger, Logger)

            # Log something to trigger file creation
            trading_logger.info("Test message")

            # Verify log file was created (immediate write with buffer_logs=False)
            log_files = list(log_dir.glob('*.log'))
            assert len(log_files) == 1
            assert 'TestStrategy' in log_files[0].name

    def test_trading_logger_add_csv_logger(self):
        """Test adding CSV loggers to TradingLogger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trading_logger = TradingLogger('TestStrategy', log_dir)

            csv_file = log_dir / 'trades.csv'
            headers = ['timestamp', 'symbol', 'side']

            trading_logger.add_csv_logger('trades', csv_file, headers)

            # Verify CSV logger was added
            assert 'trades' in trading_logger.csv_loggers
            assert isinstance(trading_logger.csv_loggers['trades'], CSVLogger)

            # Verify CSV file was created with headers
            assert csv_file.exists()
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                first_row = next(reader)
                assert first_row == headers

    def test_trading_logger_log_trade_success(self):
        """Test log_trade() for successful trade."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trading_logger = TradingLogger('TestStrategy', log_dir)

            # Add trades CSV logger
            csv_file = log_dir / 'trades.csv'
            headers = ['timestamp', 'symbol', 'side', 'qty', 'price',
                      'order_type', 'status', 'order_id', 'error']
            trading_logger.add_csv_logger('trades', csv_file, headers)

            # Log a successful trade
            trading_logger.log_trade(
                symbol='TQQQ',
                side='buy',
                qty=100,
                price=45.23,
                status='SUCCESS',
                order_type='market',
                order_id='abc123'
            )

            # Verify CSV was written
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 2  # Header + 1 data row
            assert rows[1][1] == 'TQQQ'  # symbol
            assert rows[1][2] == 'buy'   # side
            assert rows[1][3] == '100'   # qty
            assert rows[1][4] == '45.23' # price
            assert rows[1][5] == 'market' # order_type
            assert rows[1][6] == 'SUCCESS' # status
            assert rows[1][7] == 'abc123' # order_id
            assert rows[1][8] == ''       # error (empty)

    def test_trading_logger_log_trade_failed(self):
        """Test log_trade() for failed trade."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trading_logger = TradingLogger('TestStrategy', log_dir)

            # Add trades CSV logger
            csv_file = log_dir / 'trades.csv'
            headers = ['timestamp', 'symbol', 'side', 'qty', 'price',
                      'order_type', 'status', 'order_id', 'error']
            trading_logger.add_csv_logger('trades', csv_file, headers)

            # Log a failed trade
            trading_logger.log_trade(
                symbol='SQQQ',
                side='sell',
                qty=50,
                price=12.34,
                status='FAILED',
                error='Insufficient funds'
            )

            # Verify CSV was written
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 2  # Header + 1 data row
            assert rows[1][1] == 'SQQQ'  # symbol
            assert rows[1][2] == 'sell'  # side
            assert rows[1][6] == 'FAILED' # status
            assert rows[1][8] == 'Insufficient funds' # error

    def test_trading_logger_log_market_check(self):
        """Test log_market_check() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trading_logger = TradingLogger('TestStrategy', log_dir)

            # Add market checks CSV logger
            csv_file = log_dir / 'market_checks.csv'
            headers = ['timestamp', 'market_open', 'check_number']
            trading_logger.add_csv_logger('market_checks', csv_file, headers)

            # Log some market checks
            trading_logger.log_market_check(True, 1)
            trading_logger.log_market_check(False, 2)
            trading_logger.log_market_check(True, 3)

            # Verify CSV was written
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 4  # Header + 3 data rows
            assert rows[1][1] == 'True'   # market_open
            assert rows[1][2] == '1'      # check_number
            assert rows[2][1] == 'False'  # market_open
            assert rows[2][2] == '2'      # check_number
            assert rows[3][1] == 'True'   # market_open
            assert rows[3][2] == '3'      # check_number

    def test_trading_logger_convenience_methods(self):
        """Test that TradingLogger exposes base logger methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            # Use buffer_logs=False to enable immediate file writing
            trading_logger = TradingLogger('TestStrategy', log_dir, buffer_logs=False)

            # These should not raise exceptions
            trading_logger.success("Success message")
            trading_logger.error("Error message")
            trading_logger.warning("Warning message")
            trading_logger.info("Info message")
            trading_logger.separator("=", 80)
            trading_logger.blank()

            # Verify log file has content (immediate write with buffer_logs=False)
            log_files = list(log_dir.glob('*.log'))
            assert len(log_files) == 1
            content = log_files[0].read_text()
            assert "Success message" in content
            assert "Error message" in content
            assert "Warning message" in content
            assert "Info message" in content

    def test_get_trading_logger_function(self):
        """Test get_trading_logger() convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trading_logger = get_trading_logger('TestStrategy', log_dir)

            assert isinstance(trading_logger, TradingLogger)
            assert trading_logger.name == 'TestStrategy'
            assert trading_logger.log_dir == log_dir

    def test_trading_logger_multiple_csv_loggers(self):
        """Test TradingLogger with multiple CSV loggers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trading_logger = TradingLogger('TestStrategy', log_dir)

            # Add multiple CSV loggers
            trades_file = log_dir / 'trades.csv'
            market_file = log_dir / 'market.csv'

            trading_logger.add_csv_logger(
                'trades',
                trades_file,
                ['timestamp', 'symbol', 'side', 'qty', 'price',
                 'order_type', 'status', 'order_id', 'error']
            )

            trading_logger.add_csv_logger(
                'market_checks',
                market_file,
                ['timestamp', 'market_open', 'check_number']
            )

            # Log to both
            trading_logger.log_trade(
                symbol='TQQQ',
                side='buy',
                qty=100,
                price=45.23,
                status='SUCCESS'
            )

            trading_logger.log_market_check(True, 1)

            # Verify both files exist and have data
            assert trades_file.exists()
            assert market_file.exists()

            with open(trades_file, 'r', newline='', encoding='utf-8') as f:
                trades_rows = list(csv.reader(f))
            assert len(trades_rows) == 2  # Header + 1 row

            with open(market_file, 'r', newline='', encoding='utf-8') as f:
                market_rows = list(csv.reader(f))
            assert len(market_rows) == 2  # Header + 1 row


class TestIntegration:
    """Integration tests for complete logging workflow."""

    def test_full_trading_session_simulation(self):
        """Test complete trading session logging workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Create trading logger with buffer_logs=False for immediate file writing
            trading_logger = get_trading_logger('OMR', log_dir, buffer_logs=False)

            # Add CSV loggers
            trades_file = log_dir / 'trades.csv'
            market_file = log_dir / 'market_checks.csv'

            trading_logger.add_csv_logger(
                'trades',
                trades_file,
                ['timestamp', 'symbol', 'side', 'qty', 'price',
                 'order_type', 'status', 'order_id', 'error']
            )

            trading_logger.add_csv_logger(
                'market_checks',
                market_file,
                ['timestamp', 'market_open', 'check_number']
            )

            # Simulate trading session
            trading_logger.info("Starting trading session")

            # Market checks
            for i in range(3):
                trading_logger.log_market_check(True, i + 1)

            # Successful trade
            trading_logger.log_trade(
                symbol='TQQQ',
                side='buy',
                qty=100,
                price=45.23,
                status='SUCCESS',
                order_id='trade001'
            )

            # Failed trade
            trading_logger.log_trade(
                symbol='SQQQ',
                side='sell',
                qty=50,
                price=12.34,
                status='FAILED',
                error='Insufficient buying power'
            )

            trading_logger.info("Trading session complete")

            # Verify all files exist
            assert trades_file.exists()
            assert market_file.exists()
            log_files = list(log_dir.glob('*.log'))
            assert len(log_files) == 1

            # Verify trades CSV
            with open(trades_file, 'r', newline='', encoding='utf-8') as f:
                trades_rows = list(csv.reader(f))
            assert len(trades_rows) == 3  # Header + 2 trades

            # Verify market checks CSV
            with open(market_file, 'r', newline='', encoding='utf-8') as f:
                market_rows = list(csv.reader(f))
            assert len(market_rows) == 4  # Header + 3 checks

            # Verify log file content
            log_content = log_files[0].read_text()
            assert "Starting trading session" in log_content
            assert "Trading session complete" in log_content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
