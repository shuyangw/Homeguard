"""Tests for ContinuousContractDataLoader."""

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.data.continuous_contract_loader import ContinuousContractDataLoader


def test_class_importable():
    loader = ContinuousContractDataLoader()
    assert loader is not None


def _write_pcm_fixture(root: Path, year: int, month: int, rows: list[dict]) -> None:
    """Write a per-contract fixture parquet."""
    d = root / "futures_per_contract_1min" / f"year={year}" / f"month={month}"
    d.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    )
    df.write_parquet(d / "data.parquet")


def test_active_contract_picks_highest_volume_outright(tmp_path, monkeypatch):
    rows = [
        # ESM4 dominates 2024-06-03
        {"timestamp": datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
         "open": 5400.0, "high": 5400.0, "low": 5400.0, "close": 5400.0,
         "volume": 1_000_000, "symbol": "ESM4"},
        {"timestamp": datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
         "open": 5400.0, "high": 5400.0, "low": 5400.0, "close": 5400.0,
         "volume": 5_000, "symbol": "ESU4"},
        # Spread should be ignored even if volume is high
        {"timestamp": datetime(2024, 6, 3, 14, 2, tzinfo=timezone.utc),
         "open": -1.0, "high": -1.0, "low": -1.0, "close": -1.0,
         "volume": 2_000_000, "symbol": "ESM4-ESU4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    monkeypatch.setattr(
        "src.data.continuous_contract_loader._storage_root",
        lambda: tmp_path,
    )
    loader = ContinuousContractDataLoader()
    active = loader._active_contract_per_day("ES", date(2024, 6, 1), date(2024, 6, 30))
    assert active.shape == (1, 2)
    row = active.row(0, named=True)
    assert row["date"] == date(2024, 6, 3)
    assert row["active"] == "ESM4"


def test_detect_roll_dates(tmp_path, monkeypatch):
    # Build 5 trading days: ESM4 dominates first 3, ESU4 takes over on day 4
    rows = []
    for day, sym, vol in [
        (3, "ESM4", 1_000_000),
        (4, "ESM4", 800_000),
        (5, "ESM4", 600_000),
        (6, "ESU4", 700_000),  # roll happens here
        (7, "ESU4", 900_000),
    ]:
        rows.append({
            "timestamp": datetime(2024, 6, day, 14, 0, tzinfo=timezone.utc),
            "open": 5400.0, "high": 5400.0, "low": 5400.0, "close": 5400.0,
            "volume": vol, "symbol": sym,
        })
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    monkeypatch.setattr(
        "src.data.continuous_contract_loader._storage_root",
        lambda: tmp_path,
    )
    rolls = ContinuousContractDataLoader().detect_roll_dates(
        "ES", date(2024, 6, 1), date(2024, 6, 30),
    )
    assert rolls == [date(2024, 6, 6)]


def test_load_raw_passthrough(tmp_path, monkeypatch):
    # Write minimal continuous .v.0 data
    d = tmp_path / "futures_1min" / "symbol=ES" / "year=2024" / "month=6"
    d.mkdir(parents=True)
    pl.DataFrame({
        "timestamp": [
            datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
        ],
        "open": [5400.0, 5400.5],
        "high": [5401.0, 5401.0],
        "low": [5399.5, 5400.0],
        "close": [5400.5, 5400.75],
        "volume": [100, 120],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    ).write_parquet(d / "data.parquet")

    monkeypatch.setattr(
        "src.data.continuous_contract_loader._storage_root",
        lambda: tmp_path,
    )
    df = ContinuousContractDataLoader().load("ES", method="raw")
    assert df.shape == (2, 6)
    assert df["close"].to_list() == [5400.5, 5400.75]


def test_load_ratio_adjusted(tmp_path, monkeypatch):
    # Synthetic .v.0: 3 days. Roll on day 2 with a discontinuity 100 -> 110.
    d = tmp_path / "futures_1min" / "symbol=ZZ" / "year=2024" / "month=1"
    d.mkdir(parents=True)
    pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc),  # pre-roll: close=100
            datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),  # roll day: close=110
            datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc),  # post-roll: close=112
        ],
        "open": [100.0, 110.0, 110.0],
        "high": [100.0, 110.0, 112.0],
        "low": [100.0, 110.0, 110.0],
        "close": [100.0, 110.0, 112.0],
        "volume": [50, 50, 50],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    ).write_parquet(d / "data.parquet")

    # Per-contract: ZZF4 dominates day 1; ZZG4 dominates from day 2
    rows = [
        {"timestamp": datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc),
         "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
         "volume": 1000, "symbol": "ZZF4"},
        {"timestamp": datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),
         "open": 110.0, "high": 110.0, "low": 110.0, "close": 110.0,
         "volume": 1000, "symbol": "ZZG4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 1, rows)
    monkeypatch.setattr(
        "src.data.continuous_contract_loader._storage_root",
        lambda: tmp_path,
    )
    df = ContinuousContractDataLoader().load("ZZ", method="ratio_adjusted")
    # Day 1 close should be 100 * (110/100) = 110.0; day 2 = 110; day 3 = 112
    assert df["close"].to_list() == pytest.approx([110.0, 110.0, 112.0], abs=1e-6)


def test_load_panama_adjusted(tmp_path, monkeypatch):
    d = tmp_path / "futures_1min" / "symbol=YY" / "year=2024" / "month=1"
    d.mkdir(parents=True)
    pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc),  # close=100
            datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),  # close=110 (roll)
            datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc),  # close=112
        ],
        "open": [100.0, 110.0, 110.0],
        "high": [100.0, 110.0, 112.0],
        "low": [100.0, 110.0, 110.0],
        "close": [100.0, 110.0, 112.0],
        "volume": [50, 50, 50],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    ).write_parquet(d / "data.parquet")
    rows = [
        {"timestamp": datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc),
         "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
         "volume": 1000, "symbol": "YYF4"},
        {"timestamp": datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),
         "open": 110.0, "high": 110.0, "low": 110.0, "close": 110.0,
         "volume": 1000, "symbol": "YYG4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 1, rows)
    monkeypatch.setattr(
        "src.data.continuous_contract_loader._storage_root",
        lambda: tmp_path,
    )
    df = ContinuousContractDataLoader().load("YY", method="panama_adjusted")
    # diff = 110 - 100 = 10; day 1 close adjusted = 100 + 10 = 110; day 2 = 110; day 3 = 112
    assert df["close"].to_list() == pytest.approx([110.0, 110.0, 112.0], abs=1e-6)
