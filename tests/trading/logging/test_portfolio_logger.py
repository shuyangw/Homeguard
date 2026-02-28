"""Tests for PortfolioLogger CSV snapshot writing and query methods."""

import csv
from datetime import date

import pandas as pd
import pytest

from src.trading.logging.portfolio_logger import PortfolioLogger


def _make_account(equity=100000, cash=50000, buying_power=200000):
    return {
        "equity": equity,
        "cash": cash,
        "buying_power": buying_power,
    }


def _make_positions():
    return [
        {
            "symbol": "AAPL",
            "quantity": 10,
            "current_price": 150.50,
            "unrealized_pnl": 55.0,
            "market_value": 1505.0,
        },
        {
            "symbol": "MSFT",
            "quantity": 5,
            "current_price": 310.00,
            "unrealized_pnl": -20.0,
            "market_value": 1550.0,
        },
    ]


class TestLogSnapshot:
    """Tests for writing portfolio snapshots."""

    def test_creates_csv_with_header(self, tmp_path):
        pl = PortfolioLogger(tmp_path)
        pl.log_snapshot(_make_account(), _make_positions())

        with open(pl.csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        assert len(reader) == 2  # header + 1 data row
        assert reader[0] == [
            "timestamp",
            "equity",
            "cash",
            "buying_power",
            "num_positions",
            "total_unrealized_pnl",
            "positions",
        ]

    def test_appends_multiple_snapshots(self, tmp_path):
        pl = PortfolioLogger(tmp_path)
        pl.log_snapshot(_make_account(equity=100000), [])
        pl.log_snapshot(_make_account(equity=101000), [])

        with open(pl.csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        assert len(reader) == 3  # header + 2 data rows
        assert reader[1][1] == "100000.0"
        assert reader[2][1] == "101000.0"

    def test_positions_serialized(self, tmp_path):
        pl = PortfolioLogger(tmp_path)
        positions = _make_positions()
        pl.log_snapshot(_make_account(), positions)

        with open(pl.csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        positions_cell = reader[1][6]
        assert "AAPL:10@150.5" in positions_cell
        assert "MSFT:5@310.0" in positions_cell
        assert ";" in positions_cell

    def test_empty_positions(self, tmp_path):
        pl = PortfolioLogger(tmp_path)
        pl.log_snapshot(_make_account(), [])

        with open(pl.csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        assert len(reader) == 2
        assert reader[1][4] == "0"  # num_positions
        assert reader[1][5] == "0"  # total_unrealized_pnl
        assert reader[1][6] == ""   # positions string


class TestGetLatestSnapshot:
    """Tests for get_latest_snapshot query method."""

    def test_get_latest_snapshot_returns_last(self, tmp_path):
        pl = PortfolioLogger(tmp_path)
        pl.log_snapshot(_make_account(equity=100000), [])
        pl.log_snapshot(_make_account(equity=105000), [])

        latest = pl.get_latest_snapshot()
        assert latest is not None
        assert float(latest["equity"]) == 105000.0

    def test_get_latest_snapshot_none_when_empty(self, tmp_path):
        pl = PortfolioLogger(tmp_path)
        latest = pl.get_latest_snapshot()
        assert latest is None


class TestGetSnapshotHistory:
    """Tests for get_snapshot_history query method."""

    def test_get_snapshot_history_returns_dataframe(self, tmp_path):
        pl = PortfolioLogger(tmp_path)
        pl.log_snapshot(_make_account(), _make_positions())
        pl.log_snapshot(_make_account(), [])

        df = pl.get_snapshot_history()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_get_snapshot_history_empty(self, tmp_path):
        pl = PortfolioLogger(tmp_path)
        df = pl.get_snapshot_history()
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestGetDailySummary:
    """Tests for get_daily_summary query method."""

    def test_get_daily_summary(self, tmp_path):
        pl = PortfolioLogger(tmp_path)
        pl.log_snapshot(_make_account(equity=100000), [])
        pl.log_snapshot(_make_account(equity=102000), [])
        pl.log_snapshot(_make_account(equity=101000), [])

        # Read the CSV to determine which date the snapshots were logged on
        df = pd.read_csv(pl.csv_path)
        snapshot_date = pd.to_datetime(df["timestamp"].iloc[0]).date()

        summary = pl.get_daily_summary(target_date=snapshot_date)
        assert summary["snapshot_count"] == 3
        assert summary["equity_first"] == 100000.0
        assert summary["equity_last"] == 101000.0
        assert summary["equity_high"] == 102000.0
        assert summary["equity_low"] == 100000.0
