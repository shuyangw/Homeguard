"""Comprehensive tests for trade -> OHLCV-1m reconstruction."""

import numpy as np
import pandas as pd
import pytest

from src.data.acquisition.aggregators import trades_to_ohlcv_1m
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA


def _make_trades(timestamps, prices, sizes):
    """Helper to build a trades DataFrame from parallel lists."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "price": prices,
        "size": sizes,
    })


class TestBasicReconstruction:
    def test_basic_ohlcv_reconstruction(self):
        trades = _make_trades(
            timestamps=[
                "2024-01-02 09:30:00",
                "2024-01-02 09:30:15",
                "2024-01-02 09:30:45",
                "2024-01-02 09:31:05",
                "2024-01-02 09:31:30",
            ],
            prices=[100.0, 102.0, 99.0, 101.0, 103.0],
            sizes=[10, 20, 15, 5, 25],
        )
        result = trades_to_ohlcv_1m(trades)
        assert len(result) == 2  # Two 1-minute bars

        bar_0930 = result.iloc[0]
        assert bar_0930["open"] == 100.0
        assert bar_0930["high"] == 102.0
        assert bar_0930["low"] == 99.0
        assert bar_0930["close"] == 99.0  # Last trade at 09:30:45
        assert bar_0930["volume"] == 45  # 10 + 20 + 15
        assert bar_0930["trade_count"] == 3

        bar_0931 = result.iloc[1]
        assert bar_0931["open"] == 101.0
        assert bar_0931["high"] == 103.0
        assert bar_0931["low"] == 101.0
        assert bar_0931["close"] == 103.0

    def test_single_trade_per_minute(self):
        trades = _make_trades(
            timestamps=["2024-01-02 09:30:30"],
            prices=[100.5],
            sizes=[10],
        )
        result = trades_to_ohlcv_1m(trades)
        assert len(result) == 1
        bar = result.iloc[0]
        assert bar["open"] == 100.5
        assert bar["high"] == 100.5
        assert bar["low"] == 100.5
        assert bar["close"] == 100.5
        assert bar["volume"] == 10
        assert bar["trade_count"] == 1


class TestEdgeCases:
    def test_empty_minutes_excluded(self):
        trades = _make_trades(
            timestamps=[
                "2024-01-02 09:30:00",
                "2024-01-02 09:33:00",  # 2-minute gap
            ],
            prices=[100.0, 101.0],
            sizes=[10, 5],
        )
        result = trades_to_ohlcv_1m(trades)
        assert len(result) == 2  # Only bars for 09:30 and 09:33
        timestamps = result["timestamp"].tolist()
        assert pd.Timestamp("2024-01-02 09:31:00", tz="UTC") not in timestamps
        assert pd.Timestamp("2024-01-02 09:32:00", tz="UTC") not in timestamps

    def test_empty_dataframe_returns_empty(self):
        trades = pd.DataFrame(columns=["timestamp", "price", "size"])
        result = trades_to_ohlcv_1m(trades)
        assert len(result) == 0
        assert list(result.columns) == CANONICAL_OHLCV_SCHEMA

    def test_duplicate_trades_deduplicated(self):
        trades = _make_trades(
            timestamps=[
                "2024-01-02 09:30:00",
                "2024-01-02 09:30:00",  # Exact duplicate timestamp+price+size
                "2024-01-02 09:30:30",
            ],
            prices=[100.0, 100.0, 101.0],
            sizes=[10, 10, 5],
        )
        result = trades_to_ohlcv_1m(trades)
        assert len(result) == 1
        bar = result.iloc[0]
        assert bar["volume"] == 15  # 10 + 5 (deduplicated, not 10+10+5)
        assert bar["trade_count"] == 2  # 2 unique trades


class TestVWAP:
    def test_vwap_calculation(self):
        trades = _make_trades(
            timestamps=[
                "2024-01-02 09:30:00",
                "2024-01-02 09:30:30",
            ],
            prices=[100.0, 200.0],
            sizes=[10, 30],
        )
        result = trades_to_ohlcv_1m(trades)
        bar = result.iloc[0]
        # vwap = (100*10 + 200*30) / (10+30) = 7000/40 = 175.0
        assert bar["vwap"] == pytest.approx(175.0)

    def test_vwap_zero_volume(self):
        trades = _make_trades(
            timestamps=["2024-01-02 09:30:00"],
            prices=[100.0],
            sizes=[0],
        )
        result = trades_to_ohlcv_1m(trades)
        bar = result.iloc[0]
        # Zero volume -> vwap should be NaN (not crash)
        assert np.isnan(bar["vwap"])

    def test_vwap_single_trade(self):
        trades = _make_trades(
            timestamps=["2024-01-02 09:30:00"],
            prices=[150.0],
            sizes=[100],
        )
        result = trades_to_ohlcv_1m(trades)
        assert result.iloc[0]["vwap"] == pytest.approx(150.0)


class TestSchema:
    def test_output_schema_matches_canonical(self):
        trades = _make_trades(
            timestamps=["2024-01-02 09:30:00"],
            prices=[100.0],
            sizes=[10],
        )
        result = trades_to_ohlcv_1m(trades)
        assert list(result.columns) == CANONICAL_OHLCV_SCHEMA

    def test_output_dtypes(self):
        trades = _make_trades(
            timestamps=["2024-01-02 09:30:00", "2024-01-02 09:30:30"],
            prices=[100.0, 101.0],
            sizes=[10, 20],
        )
        result = trades_to_ohlcv_1m(trades)
        for col in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
            assert result[col].dtype == np.float64, f"{col} should be float64"

    def test_trade_count_accuracy(self):
        trades = _make_trades(
            timestamps=[
                "2024-01-02 09:30:00",
                "2024-01-02 09:30:10",
                "2024-01-02 09:30:20",
                "2024-01-02 09:30:30",
                "2024-01-02 09:30:40",
            ],
            prices=[100.0, 101.0, 99.0, 102.0, 100.5],
            sizes=[10, 20, 30, 40, 50],
        )
        result = trades_to_ohlcv_1m(trades)
        assert result.iloc[0]["trade_count"] == 5.0


class TestTimestampHandling:
    def test_timestamp_alignment(self):
        trades = _make_trades(
            timestamps=[
                "2024-01-02 09:30:15",  # Should floor to 09:30
                "2024-01-02 09:30:45",
                "2024-01-02 09:31:30",  # Should floor to 09:31
            ],
            prices=[100.0, 101.0, 102.0],
            sizes=[10, 20, 30],
        )
        result = trades_to_ohlcv_1m(trades)
        assert result.iloc[0]["timestamp"] == pd.Timestamp(
            "2024-01-02 09:30:00", tz="UTC"
        )
        assert result.iloc[1]["timestamp"] == pd.Timestamp(
            "2024-01-02 09:31:00", tz="UTC"
        )

    def test_overnight_session(self):
        trades = _make_trades(
            timestamps=[
                "2024-01-02 23:59:30",
                "2024-01-03 00:00:30",
            ],
            prices=[100.0, 101.0],
            sizes=[10, 20],
        )
        result = trades_to_ohlcv_1m(trades)
        assert len(result) == 2
        assert result.iloc[0]["timestamp"] == pd.Timestamp(
            "2024-01-02 23:59:00", tz="UTC"
        )
        assert result.iloc[1]["timestamp"] == pd.Timestamp(
            "2024-01-03 00:00:00", tz="UTC"
        )


class TestRobustness:
    def test_large_volume_no_overflow(self):
        trades = _make_trades(
            timestamps=["2024-01-02 09:30:00", "2024-01-02 09:30:01"],
            prices=[5000.0, 5001.0],
            sizes=[1_000_000, 2_000_000],
        )
        result = trades_to_ohlcv_1m(trades)
        assert result.iloc[0]["volume"] == 3_000_000.0

    def test_reconstruction_deterministic(self):
        trades = _make_trades(
            timestamps=[
                "2024-01-02 09:30:00",
                "2024-01-02 09:30:15",
                "2024-01-02 09:30:30",
                "2024-01-02 09:31:00",
            ],
            prices=[100.0, 102.0, 99.0, 101.0],
            sizes=[10, 20, 15, 5],
        )
        result1 = trades_to_ohlcv_1m(trades)
        result2 = trades_to_ohlcv_1m(trades)
        pd.testing.assert_frame_equal(result1, result2)

    def test_unsorted_trades_handled(self):
        trades = _make_trades(
            timestamps=[
                "2024-01-02 09:30:30",
                "2024-01-02 09:30:00",  # Out of order
                "2024-01-02 09:30:15",
            ],
            prices=[99.0, 100.0, 102.0],
            sizes=[15, 10, 20],
        )
        result = trades_to_ohlcv_1m(trades)
        bar = result.iloc[0]
        # After sorting: 09:30:00 (100), 09:30:15 (102), 09:30:30 (99)
        assert bar["open"] == 100.0
        assert bar["close"] == 99.0
