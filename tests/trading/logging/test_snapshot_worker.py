"""Tests for PortfolioSnapshotWorker daemon thread."""

import pytest
from unittest.mock import MagicMock
from datetime import time as dt_time

from src.trading.logging.snapshot_worker import PortfolioSnapshotWorker
from src.trading.logging.portfolio_logger import PortfolioLogger


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_account.return_value = {
        "equity": 100000.0,
        "cash": 50000.0,
        "buying_power": 100000.0,
    }
    broker.get_positions.return_value = [
        {
            "symbol": "AAPL",
            "quantity": 10,
            "current_price": 150.0,
            "unrealized_pnl": 50.0,
            "market_value": 1500.0,
        }
    ]
    return broker


@pytest.fixture
def portfolio_logger(tmp_path):
    return PortfolioLogger(log_dir=tmp_path / "logs")


class TestSnapshotWorker:
    def test_take_snapshot_writes_to_csv(self, mock_broker, portfolio_logger):
        worker = PortfolioSnapshotWorker(
            mock_broker, portfolio_logger, interval_minutes=15
        )
        worker._take_snapshot()

        latest = portfolio_logger.get_latest_snapshot()
        assert latest is not None
        assert latest["equity"] == 100000.0
        mock_broker.get_account.assert_called_once()
        mock_broker.get_positions.assert_called_once()

    def test_take_snapshot_handles_broker_error(
        self, mock_broker, portfolio_logger
    ):
        mock_broker.get_account.side_effect = Exception("API timeout")
        worker = PortfolioSnapshotWorker(
            mock_broker, portfolio_logger, interval_minutes=15
        )
        worker._take_snapshot()  # Should not raise
        assert portfolio_logger.get_latest_snapshot() is None

    def test_is_market_hours(self, mock_broker, portfolio_logger):
        worker = PortfolioSnapshotWorker(
            mock_broker, portfolio_logger, interval_minutes=15
        )
        # During market hours on a weekday
        assert worker._is_market_hours(dt_time(10, 0), weekday=0) is True
        # Before market open
        assert worker._is_market_hours(dt_time(8, 0), weekday=0) is False
        # After market close
        assert worker._is_market_hours(dt_time(16, 30), weekday=0) is False
        # Weekend
        assert worker._is_market_hours(dt_time(10, 0), weekday=5) is False
        # Boundary: exactly at open
        assert worker._is_market_hours(dt_time(9, 30), weekday=0) is True
        # Boundary: exactly at close (exclusive)
        assert worker._is_market_hours(dt_time(16, 0), weekday=0) is False

    def test_start_stop(self, mock_broker, portfolio_logger):
        worker = PortfolioSnapshotWorker(
            mock_broker, portfolio_logger, interval_minutes=15
        )
        worker.start()
        assert worker._running is True
        assert worker._thread is not None
        assert worker._thread.daemon is True
        worker.stop()
        assert worker._running is False
