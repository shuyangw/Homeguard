"""Tests for CarryCalculator."""
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.data.carry_calculator import CarryCalculator


def _write_pcm_fixture(root: Path, year: int, month: int, rows: list[dict]) -> None:
    d = root / "futures_per_contract_1min" / f"year={year}" / f"month={month}"
    d.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    )
    df.write_parquet(d / "data.parquet")


def test_find_front_second_close(tmp_path, monkeypatch):
    rows = [
        {"timestamp": datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
         "open": 5300.0, "high": 5300.0, "low": 5300.0, "close": 5300.0,
         "volume": 1_000_000, "symbol": "ESM4"},
        {"timestamp": datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
         "open": 5362.0, "high": 5362.0, "low": 5362.0, "close": 5362.0,
         "volume": 50_000, "symbol": "ESU4"},
        # Less liquid third contract
        {"timestamp": datetime(2024, 6, 3, 14, 2, tzinfo=timezone.utc),
         "open": 5420.0, "high": 5420.0, "low": 5420.0, "close": 5420.0,
         "volume": 100, "symbol": "ESZ4"},
        # Spread should be ignored
        {"timestamp": datetime(2024, 6, 3, 14, 3, tzinfo=timezone.utc),
         "open": -62.0, "high": -62.0, "low": -62.0, "close": -62.0,
         "volume": 2_000_000, "symbol": "ESM4-ESU4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    monkeypatch.setattr(
        "src.data.carry_calculator._storage_root",
        lambda: tmp_path,
    )
    front_sym, front_c, second_sym, second_c = CarryCalculator()._find_front_second_close(
        "ES", date(2024, 6, 3),
    )
    assert front_sym == "ESM4"
    assert front_c == 5300.0
    assert second_sym == "ESU4"
    assert second_c == 5362.0
