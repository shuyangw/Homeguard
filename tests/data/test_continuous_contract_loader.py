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
