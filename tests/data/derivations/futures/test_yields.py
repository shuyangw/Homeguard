"""Tests for Treasury yield reads from Micro Yield futures."""
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.data.derivations.futures.yields import (
    TENOR_TO_SYMBOL,
    get_treasury_yield,
)


def test_tenor_mapping():
    assert TENOR_TO_SYMBOL == {
        "2Y": "2YY", "5Y": "5YY", "10Y": "10Y", "30Y": "30Y",
    }


def test_get_treasury_yield_synthetic(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    out = (
        storage / "futures_1min" / "symbol=10Y"
        / "year=2024" / "month=6" / "data.parquet"
    )
    out.parent.mkdir(parents=True)
    df = pl.DataFrame({
        "timestamp": pl.datetime_range(
            start=pl.datetime(2024, 6, 15, 13, 0, time_zone="UTC"),
            end=pl.datetime(2024, 6, 15, 21, 0, time_zone="UTC"),
            interval="8h",
            eager=True,
        ),
        "open": [4.20, 4.23],
        "high": [4.20, 4.23],
        "low": [4.20, 4.23],
        "close": [4.20, 4.228],
        "volume": [100, 100],
    })
    df.write_parquet(out)

    monkeypatch.setattr(
        "src.data.derivations.futures.yields._storage_root",
        lambda: storage,
    )

    y = get_treasury_yield("10Y", date(2024, 6, 15))
    assert y == pytest.approx(4.228, abs=0.001)


def test_pre_listing_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.data.derivations.futures.yields._storage_root",
        lambda: tmp_path,
    )
    with pytest.raises(ValueError, match="Micro Yield.*listing"):
        get_treasury_yield("10Y", date(2022, 8, 14))


def test_unknown_tenor_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.data.derivations.futures.yields._storage_root",
        lambda: tmp_path,
    )
    with pytest.raises(KeyError):
        get_treasury_yield("7Y", date(2024, 6, 15))


def test_missing_data_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.data.derivations.futures.yields._storage_root",
        lambda: tmp_path,
    )
    with pytest.raises(ValueError, match="no .* data"):
        get_treasury_yield("10Y", date(2024, 6, 15))
