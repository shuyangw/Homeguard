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


def test_compute_commodity(tmp_path, monkeypatch):
    """GC: GCQ4 (Aug) is volume-heavier so it's 'front'; GCM4 (Jun) is 'second'.
    second_month - front_month = -2 months (= -60 days). Algorithm uses ABSOLUTE
    months for days_to_second so days_to_second=60.
    carry = (2322.5 - 2346.1) / 2346.1 * (365/60) ≈ -0.0612"""
    rows = [
        {"timestamp": datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
         "open": 2322.5, "high": 2322.5, "low": 2322.5, "close": 2322.5,
         "volume": 1000, "symbol": "GCM4"},  # Jun 2024
        {"timestamp": datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
         "open": 2346.1, "high": 2346.1, "low": 2346.1, "close": 2346.1,
         "volume": 100000, "symbol": "GCQ4"},  # Aug 2024 (front)
    ]
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    monkeypatch.setattr(
        "src.data.carry_calculator._storage_root",
        lambda: tmp_path,
    )
    carry = CarryCalculator().compute("GC", "commodity", date(2024, 6, 3))
    assert carry == pytest.approx(-0.0612, abs=0.005)


def test_compute_equity_index(tmp_path, monkeypatch):
    """ES: front=5296.75, second=5359.50, ~90d to second.
    Equity formula: (front - second) / second * (365/90) ≈ -0.0475."""
    rows = [
        {"timestamp": datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
         "open": 5296.75, "high": 5296.75, "low": 5296.75, "close": 5296.75,
         "volume": 1_000_000, "symbol": "ESM4"},
        {"timestamp": datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
         "open": 5359.50, "high": 5359.50, "low": 5359.50, "close": 5359.50,
         "volume": 50_000, "symbol": "ESU4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    monkeypatch.setattr(
        "src.data.carry_calculator._storage_root",
        lambda: tmp_path,
    )
    carry = CarryCalculator().compute("ES", "equity_index", date(2024, 6, 3))
    assert carry == pytest.approx(-0.0475, abs=0.005)


def test_compute_bond_micro_yield(tmp_path, monkeypatch):
    """10Y Micro Yield: front close = 4.20% (yield), SOFR = 3.50%.
    duration=9, carry = 9 * (4.20 - 3.50) / 100 = 0.063."""
    rows = [
        {"timestamp": datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
         "open": 4.20, "high": 4.20, "low": 4.20, "close": 4.20,
         "volume": 100000, "symbol": "10YM4"},
        {"timestamp": datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
         "open": 4.22, "high": 4.22, "low": 4.22, "close": 4.22,
         "volume": 5000, "symbol": "10YU4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    monkeypatch.setattr(
        "src.data.carry_calculator._storage_root",
        lambda: tmp_path,
    )
    # Stub SOFR derivation
    monkeypatch.setattr(
        "src.data.carry_calculator.derive_sofr",
        lambda d: 3.5,
    )
    carry = CarryCalculator().compute("10Y", "bond", date(2024, 6, 3))
    # 9 * (4.20 - 3.50) / 100 = 0.063
    assert carry == pytest.approx(0.063, abs=0.005)


def test_compute_bond_standard_returns_zero(tmp_path, monkeypatch):
    """ZN (price-traded T-Note): no direct yield available -> v1 fallback returns 0."""
    rows = [
        {"timestamp": datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
         "open": 110.5, "high": 110.5, "low": 110.5, "close": 110.5,
         "volume": 100000, "symbol": "ZNM4"},
        {"timestamp": datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
         "open": 110.0, "high": 110.0, "low": 110.0, "close": 110.0,
         "volume": 5000, "symbol": "ZNU4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    monkeypatch.setattr(
        "src.data.carry_calculator._storage_root",
        lambda: tmp_path,
    )
    carry = CarryCalculator().compute("ZN", "bond", date(2024, 6, 3))
    assert carry == 0.0
