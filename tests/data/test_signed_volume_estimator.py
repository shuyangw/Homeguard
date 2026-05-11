"""Tests for signed volume estimator."""
from datetime import date, datetime, timezone

import polars as pl
import pytest

from src.data.signed_volume_estimator import estimate_signed_volume_from_bars


def test_signed_volume_tick_rule(tmp_path, monkeypatch):
    """Up bars (close > prior close) -> positive signed volume.
    Down bars -> negative. First bar of the day: no prior -> 0."""
    d = tmp_path / "futures_1min" / "symbol=ES" / "year=2024" / "month=6"
    d.mkdir(parents=True)
    pl.DataFrame({
        "timestamp": [
            datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),  # first bar
            datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),  # up
            datetime(2024, 6, 3, 14, 2, tzinfo=timezone.utc),  # down
            datetime(2024, 6, 3, 14, 3, tzinfo=timezone.utc),  # up
        ],
        "open": [5400.0, 5400.5, 5401.0, 5400.5],
        "high": [5400.5, 5401.0, 5401.0, 5401.0],
        "low":  [5399.5, 5400.0, 5400.0, 5400.0],
        "close": [5400.5, 5401.0, 5400.5, 5401.0],
        "volume": [100, 200, 150, 250],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    ).write_parquet(d / "data.parquet")

    monkeypatch.setattr(
        "src.data.signed_volume_estimator._storage_root",
        lambda: tmp_path,
    )
    df = estimate_signed_volume_from_bars("ES", date(2024, 6, 3))
    # Bar 1: first -> tick_sign=0 -> signed=0
    # Bar 2: 5401.0 > 5400.5 -> tick_sign=+1 -> signed=+200
    # Bar 3: 5400.5 < 5401.0 -> tick_sign=-1 -> signed=-150
    # Bar 4: 5401.0 > 5400.5 -> tick_sign=+1 -> signed=+250
    signed = df["signed_volume"].to_list()
    assert signed[0] == 0
    assert signed[1] == 200
    assert signed[2] == -150
    assert signed[3] == 250


def test_signed_volume_missing_data_returns_empty(tmp_path, monkeypatch):
    """If no data file for the date, return empty DataFrame."""
    monkeypatch.setattr(
        "src.data.signed_volume_estimator._storage_root",
        lambda: tmp_path,
    )
    df = estimate_signed_volume_from_bars("ES", date(2024, 6, 3))
    assert df.is_empty()


def test_signed_volume_aggregate_matches_input_volume(tmp_path, monkeypatch):
    """The sum of |signed_volume| should equal the total minute-bar volume
    for non-first-bar rows. Verifies no doubling or missing rows."""
    d = tmp_path / "futures_1min" / "symbol=NQ" / "year=2024" / "month=3"
    d.mkdir(parents=True)
    pl.DataFrame({
        "timestamp": [
            datetime(2024, 3, 4, 14, 0, tzinfo=timezone.utc),
            datetime(2024, 3, 4, 14, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 4, 14, 2, tzinfo=timezone.utc),
        ],
        "open": [18000.0, 18001.0, 18000.5],
        "high": [18001.0, 18002.0, 18001.0],
        "low":  [17999.0, 18000.0, 18000.0],
        "close": [18001.0, 18000.5, 18000.0],
        "volume": [100, 200, 300],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    ).write_parquet(d / "data.parquet")

    monkeypatch.setattr(
        "src.data.signed_volume_estimator._storage_root",
        lambda: tmp_path,
    )
    df = estimate_signed_volume_from_bars("NQ", date(2024, 3, 4))
    # First bar has signed=0; subsequent bars: |signed| should equal their volume
    abs_signed_sum = sum(abs(v) for v in df["signed_volume"].to_list()[1:])
    input_vol_sum = sum(df["volume"].to_list()[1:])
    assert abs_signed_sum == input_vol_sum
