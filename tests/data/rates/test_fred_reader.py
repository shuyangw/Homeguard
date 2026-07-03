from datetime import date
import polars as pl
import pytest
from src.data.rates.fred_reader import get_fred_series


def test_reads_value_on_exact_date():
    # DGS10 has a print for a known trading day; value is a plausible yield.
    v = get_fred_series("DGS10", date(2024, 6, 3))
    assert 0.0 < v < 20.0


def test_forward_fills_weekend_causally():
    # A Sunday has no print -> returns the prior Friday's value (latest <= d).
    sun = get_fred_series("DGS10", date(2024, 6, 2))   # Sunday
    fri = get_fred_series("DGS10", date(2024, 5, 31))  # Friday
    assert sun == fri


def test_raises_before_series_start():
    with pytest.raises(ValueError):
        get_fred_series("DGS10", date(1990, 1, 1))  # series starts 1995


def test_missing_series_raises(monkeypatch, tmp_path):
    import src.data.rates.fred_reader as fr
    monkeypatch.setattr(fr, "get_local_storage_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        get_fred_series("NOPE", date(2024, 1, 1))
