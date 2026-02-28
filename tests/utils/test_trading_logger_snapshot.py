"""Tests for TradeLogWriter account snapshot enhancement."""

import json
import pytest
from pathlib import Path

from src.utils.trading_logger import TradeLogWriter


@pytest.fixture
def trade_logger(tmp_path):
    return TradeLogWriter(log_dir=str(tmp_path))


class TestTradeLogWithSnapshot:
    def test_log_entry_with_snapshot(self, trade_logger):
        snapshot = {"equity": 100000.0, "cash": 50000.0, "buying_power": 100000.0}
        trade_logger.log_entry(
            strategy="ramp", symbol="AAPL", qty=10, price=150.0,
            account_snapshot=snapshot
        )

        log_file = trade_logger._get_log_file()
        with open(log_file) as f:
            record = json.loads(f.readline())

        assert record["account_snapshot"] == snapshot
        assert record["symbol"] == "AAPL"
        assert record["strategy"] == "ramp"

    def test_log_exit_with_snapshot(self, trade_logger):
        snapshot = {"equity": 102000.0, "cash": 52000.0, "buying_power": 104000.0}
        trade_logger.log_exit(
            strategy="omr", symbol="TQQQ", qty=100, exit_price=70.0,
            entry_price=68.0, account_snapshot=snapshot
        )

        log_file = trade_logger._get_log_file()
        with open(log_file) as f:
            record = json.loads(f.readline())

        assert record["account_snapshot"] == snapshot
        assert record["pnl_dollars"] == 200.0

    def test_log_entry_without_snapshot_still_works(self, trade_logger):
        trade_logger.log_entry(
            strategy="ramp", symbol="AAPL", qty=10, price=150.0
        )

        log_file = trade_logger._get_log_file()
        with open(log_file) as f:
            record = json.loads(f.readline())

        assert "account_snapshot" not in record
        assert record["symbol"] == "AAPL"
