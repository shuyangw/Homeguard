"""Tests for FuturesRollManager."""
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.data.roll_detector import FuturesRollManager


def _write_pcm_fixture(root: Path, year: int, month: int, rows: list[dict]) -> None:
    d = root / "futures" / "databento" / "per_contract_1min" / f"year={year}" / f"month={month}"
    d.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    )
    df.write_parquet(d / "data.parquet")


def test_get_active_contract(tmp_path, monkeypatch):
    rows = [
        {"timestamp": datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
         "open": 5300.0, "high": 5300.0, "low": 5300.0, "close": 5300.0,
         "volume": 1_000_000, "symbol": "ESM4"},
        {"timestamp": datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
         "open": 5362.0, "high": 5362.0, "low": 5362.0, "close": 5362.0,
         "volume": 50_000, "symbol": "ESU4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    # The loader reads via src.data.futures.paths -> get_local_storage_dir, must be patched.
    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: tmp_path,
    )
    mgr = FuturesRollManager()
    assert mgr.get_active_contract("ES", date(2024, 6, 3)) == "ESM4"


def test_get_upcoming_rolls_returns_empty_v1():
    """v1 stub: get_upcoming_rolls always returns empty list."""
    mgr = FuturesRollManager()
    rolls = mgr.get_upcoming_rolls(["ES", "GC"], today=date(2024, 6, 3), lookahead_days=14)
    assert rolls == []
