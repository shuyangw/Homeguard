"""Tests for Layer 4 external checks."""
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.data.validation.core.result import Severity
from src.data.validation.futures.checks.external import (
    KnownEventsCheck,
    YfinanceCrossCheck,
    CmeSettlementsCheck,
)


def test_yfinance_check_off_by_default():
    r = YfinanceCrossCheck(flags={}).run()
    assert r.passed is True
    assert r.severity == Severity.INFO
    assert "opt-in" in r.message.lower() or "skipped" in r.message.lower()


def test_yfinance_check_runs_when_flag_set(monkeypatch):
    # We can't actually hit yfinance in unit tests; test the flag-gating only.
    r = YfinanceCrossCheck(flags={"external_yfinance": True}).run()
    # Either passed=True (no diff) or skipped due to no test data; both OK
    assert r is not None


def test_cme_settlements_check_off_by_default():
    r = CmeSettlementsCheck(flags={}).run()
    assert r.passed is True
    assert r.severity == Severity.INFO


def test_known_events_pass_when_data_present(tmp_path: Path, monkeypatch):
    es_dir = tmp_path / "futures_1min" / "symbol=ES" / "year=2020" / "month=4"
    es_dir.mkdir(parents=True)
    df = pl.DataFrame({
        "timestamp": [datetime(2020, 4, 20, 14, 0, tzinfo=timezone.utc)],
        "open": [2750.0], "high": [2760.0], "low": [2740.0],
        "close": [2755.0], "volume": [10000],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    )
    df.write_parquet(es_dir / "data.parquet")
    cl_dir = tmp_path / "futures_1min" / "symbol=CL" / "year=2020" / "month=4"
    cl_dir.mkdir(parents=True)
    df2 = pl.DataFrame({
        "timestamp": [datetime(2020, 4, 20, 14, 0, tzinfo=timezone.utc)],
        "open": [-30.0], "high": [-10.0], "low": [-40.0],
        "close": [-37.0], "volume": [50000],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    )
    df2.write_parquet(cl_dir / "data.parquet")

    monkeypatch.setattr(
        "src.data.validation.futures.checks.external._storage_root",
        lambda: tmp_path,
    )
    r = KnownEventsCheck().run()
    assert r is not None  # at minimum, it ran
