"""Tests for CBOE SKEW acquirer."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.data.leading_indicators import load_skew
import src.data.leading_indicators.skew as m


@pytest.mark.network
def test_skew_2024_window():
    """Smoke test: downloading a small window returns expected schema."""
    df = load_skew(datetime(2024, 1, 1), datetime(2024, 1, 31), cache=False)
    assert {'skew_close'}.issubset(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df['skew_close'].notna().all()
    assert (df['skew_close'] > 0).all()
    assert len(df) > 10


@pytest.mark.network
def test_skew_date_range_within_request():
    """Returned dates fall within [start, end]."""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 31)
    df = load_skew(start, end, cache=False)
    assert df.index.min() >= pd.Timestamp(start)
    assert df.index.max() <= pd.Timestamp(end)


@pytest.mark.network
def test_skew_cache_round_trip(tmp_path, monkeypatch):
    """Second call with cache returns identical data."""
    monkeypatch.setattr(m, 'CACHE_PATH', lambda: tmp_path / 'skew.parquet')
    df1 = load_skew(datetime(2024, 1, 1), datetime(2024, 1, 31), cache=True)
    df2 = load_skew(datetime(2024, 1, 1), datetime(2024, 1, 31), cache=True)
    pd.testing.assert_frame_equal(df1, df2)
