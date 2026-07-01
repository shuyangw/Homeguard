"""Tests for aggregate open interest derivation."""
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.data.derivations.futures.open_interest import (
    STAT_TYPE_OPEN_INTEREST,
    _is_outright,
    aggregate_open_interest,
)


def _write_stats(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def test_outright_regex():
    assert _is_outright("ESH4", "ES")
    assert _is_outright("ESM24", "ES")
    assert _is_outright("CLH4", "CL")
    assert not _is_outright("ESH4-ESM4", "ES")  # spread
    assert not _is_outright("ZNH4", "ES")  # different root
    assert not _is_outright("ES H4", "ES")  # has space
    assert not _is_outright("ES", "ES")  # no month/year suffix


def test_aggregate_oi_simple(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    path = storage / "futures" / "databento" / "statistics" / "year=2024" / "month=6" / "data.parquet"
    _write_stats(path, [
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESM4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 1000},
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESU4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 500},
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESZ4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 200},
    ])

    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: storage,
    )
    assert aggregate_open_interest("ES", date(2024, 6, 15)) == 1700


def test_excludes_spreads(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    path = storage / "futures" / "databento" / "statistics" / "year=2024" / "month=6" / "data.parquet"
    _write_stats(path, [
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESM4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 1000},
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESM4-ESU4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 9999},
    ])
    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: storage,
    )
    assert aggregate_open_interest("ES", date(2024, 6, 15)) == 1000


def test_excludes_other_roots(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    path = storage / "futures" / "databento" / "statistics" / "year=2024" / "month=6" / "data.parquet"
    _write_stats(path, [
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESM4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 1000},
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "CLM4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 500},
    ])
    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: storage,
    )
    assert aggregate_open_interest("ES", date(2024, 6, 15)) == 1000


def test_excludes_other_stat_types(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    path = storage / "futures" / "databento" / "statistics" / "year=2024" / "month=6" / "data.parquet"
    _write_stats(path, [
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESM4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 1000},
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESM4", "stat_type": 3, "quantity": 99999},  # settlement
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESM4", "stat_type": 6, "quantity": 88888},  # volume
    ])
    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: storage,
    )
    assert aggregate_open_interest("ES", date(2024, 6, 15)) == 1000


def test_takes_latest_snapshot_per_contract(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    path = storage / "futures" / "databento" / "statistics" / "year=2024" / "month=6" / "data.parquet"
    _write_stats(path, [
        {"timestamp": datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc),
         "symbol": "ESM4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 900},
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESM4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 1000},
    ])
    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: storage,
    )
    assert aggregate_open_interest("ES", date(2024, 6, 15)) == 1000


def test_excludes_other_dates(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    path = storage / "futures" / "databento" / "statistics" / "year=2024" / "month=6" / "data.parquet"
    _write_stats(path, [
        {"timestamp": datetime(2024, 6, 14, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESM4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 500},
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "ESM4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 1000},
    ])
    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: storage,
    )
    assert aggregate_open_interest("ES", date(2024, 6, 15)) == 1000


def test_missing_partition_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="futures_statistics partition"):
        aggregate_open_interest("ES", date(2099, 1, 1))


def test_no_rows_returns_zero(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    path = storage / "futures" / "databento" / "statistics" / "year=2024" / "month=6" / "data.parquet"
    _write_stats(path, [
        {"timestamp": datetime(2024, 6, 15, 21, 0, tzinfo=timezone.utc),
         "symbol": "CLM4", "stat_type": STAT_TYPE_OPEN_INTEREST, "quantity": 500},
    ])
    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: storage,
    )
    assert aggregate_open_interest("ES", date(2024, 6, 15)) == 0
