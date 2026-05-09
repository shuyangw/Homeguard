"""Tests for Layer 2 statistical checks."""
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.data.validation.core.result import Severity
from src.data.validation.futures.checks.statistical import (
    DensityCheck,
    OhlcvInvariantsCheck,
    DateFloorCheck,
    SofrDerivationSanityCheck,
    TreasuryYieldsSanityCheck,
    EsRealizedVolDeferredCheck,
)


def _write_es_month(root: Path, year: int, month: int, n_bars_per_day: int):
    es_dir = root / "futures_1min" / "symbol=ES" / f"year={year}" / f"month={month}"
    es_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for day in range(1, 6):  # 5 trading days
        for bar in range(n_bars_per_day):
            ts = datetime(year, month, day, 13, bar % 60, tzinfo=timezone.utc)
            rows.append({
                "timestamp": ts,
                "open": 5400.0, "high": 5401.0, "low": 5399.0, "close": 5400.5,
                "volume": 100,
            })
    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    )
    df.write_parquet(es_dir / "data.parquet")


def test_density_check_pass(tmp_path: Path, monkeypatch):
    _write_es_month(tmp_path, 2024, 6, n_bars_per_day=900)
    monkeypatch.setattr(
        "src.data.validation.futures.checks.statistical._storage_root",
        lambda: tmp_path,
    )
    result = DensityCheck(symbol="ES").run()
    assert result.passed is True


def test_density_check_critical_when_far_below(tmp_path: Path, monkeypatch):
    # Bug-fix smoking gun: GC.c.0 had ~7 bars/day; expected 900-1200.
    _write_es_month(tmp_path, 2024, 6, n_bars_per_day=7)
    # Use ES symbol but expectations.EXPECTED_DENSITY says ES is (800, 1000)
    monkeypatch.setattr(
        "src.data.validation.futures.checks.statistical._storage_root",
        lambda: tmp_path,
    )
    result = DensityCheck(symbol="ES").run()
    assert result.passed is False
    assert result.severity == Severity.CRITICAL


def test_ohlcv_invariants_pass(tmp_path: Path, monkeypatch):
    _write_es_month(tmp_path, 2024, 6, n_bars_per_day=10)
    monkeypatch.setattr(
        "src.data.validation.futures.checks.statistical._storage_root",
        lambda: tmp_path,
    )
    r = OhlcvInvariantsCheck(symbol="ES").run()
    assert r.passed is True


def test_ohlcv_invariants_fail_on_low_above_high(tmp_path: Path, monkeypatch):
    es_dir = tmp_path / "futures_1min" / "symbol=ES" / "year=2024" / "month=6"
    es_dir.mkdir(parents=True)
    df = pl.DataFrame({
        "timestamp": [datetime(2024, 6, 17, 13, 0, tzinfo=timezone.utc)],
        "open": [5400.0], "high": [5399.0], "low": [5401.0],  # broken
        "close": [5400.0], "volume": [100],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    )
    df.write_parquet(es_dir / "data.parquet")
    monkeypatch.setattr(
        "src.data.validation.futures.checks.statistical._storage_root",
        lambda: tmp_path,
    )
    r = OhlcvInvariantsCheck(symbol="ES").run()
    assert r.passed is False
    assert r.severity == Severity.CRITICAL


def test_date_floor_check_pass(tmp_path: Path, monkeypatch):
    es_dir = tmp_path / "futures_1min" / "symbol=ES" / "year=2010" / "month=6"
    es_dir.mkdir(parents=True)
    df = pl.DataFrame({
        "timestamp": [datetime(2010, 6, 6, 13, 0, tzinfo=timezone.utc)],
        "open": [1000.0], "high": [1000.0], "low": [1000.0],
        "close": [1000.0], "volume": [100],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    )
    df.write_parquet(es_dir / "data.parquet")
    monkeypatch.setattr(
        "src.data.validation.futures.checks.statistical._storage_root",
        lambda: tmp_path,
    )
    r = DateFloorCheck().run()
    assert r.passed is True


def test_sofr_derivation_check_skips_when_data_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "src.data.validation.futures.checks.statistical._storage_root",
        lambda: tmp_path,
    )
    r = SofrDerivationSanityCheck().run()
    # No SR1 data - all sample dates fail; severity WARNING
    assert r.severity in (Severity.WARNING, Severity.CRITICAL)


def test_es_realized_vol_deferred_returns_info():
    r = EsRealizedVolDeferredCheck().run()
    assert r.passed is True  # not failed; deferred is intentional
    assert r.severity == Severity.INFO
    assert "deferred" in r.message.lower()
