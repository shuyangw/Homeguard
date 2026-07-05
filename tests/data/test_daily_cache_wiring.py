import polars as pl
from datetime import datetime, timezone, date

from src.data.continuous_contract_loader import ContinuousContractDataLoader


def _raw():
    return pl.DataFrame({
        "timestamp": [
            datetime(2020, 1, 2, tzinfo=timezone.utc),
            datetime(2020, 1, 3, tzinfo=timezone.utc),
        ],
        "open": [10.0, 11.0],
        "high": [10.0, 11.0],
        "low": [10.0, 11.0],
        "close": [10.0, 11.0],
        "volume": [5, 5],
    })


def test_uses_cache_when_present(monkeypatch, tmp_path):
    from src.data.futures import paths as fpaths
    monkeypatch.setattr(fpaths, "daily_raw_dir", lambda: tmp_path)

    _raw().write_parquet(tmp_path / "FAKE.parquet")

    ld = ContinuousContractDataLoader()

    def boom(*a, **k):
        raise AssertionError("load() should not be called when cache present")

    monkeypatch.setattr(ld, "load", boom)

    # no rolls for a 2-row fake -> ratio_adjust_daily returns the daily unchanged
    out = ld.aggregate_to_daily(
        "FAKE", method="ratio_adjusted", start=date(2020, 1, 1), end=date(2020, 1, 31)
    )
    assert out.height == 2


def test_falls_back_when_absent(monkeypatch, tmp_path):
    from src.data.futures import paths as fpaths
    monkeypatch.setattr(fpaths, "daily_raw_dir", lambda: tmp_path)  # empty dir, no cache file

    ld = ContinuousContractDataLoader()
    called = {"n": 0}

    def fake_load(root, method="ratio_adjusted", start=None, end=None):
        called["n"] += 1
        return pl.DataFrame({
            "timestamp": [datetime(2020, 1, 2, tzinfo=timezone.utc)],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1],
        })

    monkeypatch.setattr(ld, "load", fake_load)
    out = ld.aggregate_to_daily(
        "NOCACHE", method="ratio_adjusted", start=date(2020, 1, 1), end=date(2020, 1, 31)
    )
    assert called["n"] == 1  # fell back to load()
