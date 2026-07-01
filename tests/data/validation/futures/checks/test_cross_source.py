"""Tests for Layer 3 cross-source checks."""
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.data.validation.core.result import Severity
from src.data.validation.futures.checks.cross_source import (
    DefinitionsCompletenessCheck,
    SofrVs2YCheck,
    CarrySanityDeferredCheck,
)


def test_definitions_completeness_pass(tmp_path: Path, monkeypatch):
    # per-contract has ESH4; definitions also has ESH4
    pcm = tmp_path / "futures" / "databento" / "per_contract_1min" / "year=2024" / "month=3"
    pcm.mkdir(parents=True)
    pl.DataFrame({
        "timestamp": [datetime(2024, 3, 15, tzinfo=timezone.utc)],
        "open": [5400.0], "high": [5400.0], "low": [5400.0],
        "close": [5400.0], "volume": [100], "symbol": ["ESH4"],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    ).write_parquet(pcm / "data.parquet")

    defs = tmp_path / "futures" / "definitions" / "year=2024" / "month=3"
    defs.mkdir(parents=True)
    pl.DataFrame({
        "timestamp": [datetime(2024, 3, 1, tzinfo=timezone.utc)],
        "raw_symbol": ["ESH4"],
        "expiration": [datetime(2024, 3, 15, tzinfo=timezone.utc)],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("expiration").cast(pl.Datetime("ns", "UTC")),
    ).write_parquet(defs / "data.parquet")

    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: tmp_path,
    )
    r = DefinitionsCompletenessCheck().run()
    assert r.passed is True


def test_definitions_completeness_critical_when_missing(tmp_path: Path, monkeypatch):
    pcm = tmp_path / "futures" / "databento" / "per_contract_1min" / "year=2024" / "month=3"
    pcm.mkdir(parents=True)
    pl.DataFrame({
        "timestamp": [datetime(2024, 3, 15, tzinfo=timezone.utc)],
        "open": [5400.0], "high": [5400.0], "low": [5400.0],
        "close": [5400.0], "volume": [100], "symbol": ["ESH4"],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    ).write_parquet(pcm / "data.parquet")

    defs = tmp_path / "futures" / "definitions" / "year=2024" / "month=3"
    defs.mkdir(parents=True)
    pl.DataFrame({
        "timestamp": [datetime(2024, 3, 1, tzinfo=timezone.utc)],
        "raw_symbol": ["NOT-ESH4"],  # missing the actual contract
        "expiration": [datetime(2024, 3, 15, tzinfo=timezone.utc)],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("expiration").cast(pl.Datetime("ns", "UTC")),
    ).write_parquet(defs / "data.parquet")

    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: tmp_path,
    )
    r = DefinitionsCompletenessCheck().run()
    assert r.passed is False
    assert r.severity == Severity.CRITICAL


def test_sofr_vs_2y_skips_when_no_data(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "src.data.futures.paths.get_local_storage_dir",
        lambda: tmp_path,
    )
    r = SofrVs2YCheck().run()
    # No data at all - reports issue but doesn't crash
    assert r.severity in (Severity.WARNING, Severity.INFO)


def test_carry_sanity_deferred():
    r = CarrySanityDeferredCheck().run()
    assert r.passed is True
    assert r.severity == Severity.INFO
    assert "deferred" in r.message.lower()
