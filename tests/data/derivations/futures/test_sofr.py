"""Tests for SOFR derivation from SR1 front-month."""
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.data.derivations.futures.sofr import derive_sofr, sr1_front_month_symbol


def test_sr1_front_month_january():
    # SR1 front month for January 2024 -> SR1F4 (CME 'F' = January, '4' = year digit)
    assert sr1_front_month_symbol(date(2024, 1, 15)) == "SR1F4"


def test_sr1_front_month_december():
    assert sr1_front_month_symbol(date(2024, 12, 5)) == "SR1Z4"


def test_sr1_front_month_year_rollover():
    # Single year-digit convention rolls Z9 -> F0 etc.
    assert sr1_front_month_symbol(date(2025, 1, 1)) == "SR1F5"


def test_derive_sofr_synthetic(tmp_path: Path, monkeypatch):
    # Build a fake per-contract parquet with SR1 contracts, exercise derivation.
    # SR1F4 close = 95.50 -> implied SOFR = 4.50
    from datetime import datetime
    storage = tmp_path / "storage"
    pcm_dir = storage / "futures_per_contract_1min" / "year=2024" / "month=1"
    pcm_dir.mkdir(parents=True)

    df = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 15, 16, 0)],
        "open": [95.5], "high": [95.5], "low": [95.5], "close": [95.5],
        "volume": [100], "symbol": ["SR1F4"],
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    df.write_parquet(pcm_dir / "data.parquet")

    monkeypatch.setattr(
        "src.data.derivations.futures.sofr._storage_root",
        lambda: storage,
    )

    rate = derive_sofr(date(2024, 1, 15))
    assert rate == pytest.approx(4.50, abs=0.001)


def test_derive_sofr_pre_listing_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.data.derivations.futures.sofr._storage_root",
        lambda: tmp_path,
    )
    with pytest.raises(ValueError, match="SR1.*listing"):
        derive_sofr(date(2018, 5, 6))  # one day before SR1 listing


def test_derive_sofr_missing_data_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.data.derivations.futures.sofr._storage_root",
        lambda: tmp_path,
    )
    with pytest.raises(ValueError, match="no SR1"):
        derive_sofr(date(2024, 1, 15))
