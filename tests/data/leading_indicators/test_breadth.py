"""Tests for NYSE A-D breadth acquirer."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.data.leading_indicators import load_breadth
import src.data.leading_indicators.breadth as m


@pytest.mark.network
def test_breadth_2024_window():
    """Smoke test: computing breadth over a small window returns expected schema."""
    df = load_breadth(datetime(2024, 1, 1), datetime(2024, 1, 31), cache=False)
    assert {'breadth_pct', 'n_constituents'}.issubset(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df['breadth_pct'].notna().all()
    assert (df['breadth_pct'] >= 0.0).all()
    assert (df['breadth_pct'] <= 1.0).all()
    assert (df['n_constituents'] > 0).all()
    assert len(df) > 10


@pytest.mark.network
def test_breadth_date_range_within_request():
    """Returned dates fall within [start, end]."""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 31)
    df = load_breadth(start, end, cache=False)
    assert df.index.min() >= pd.Timestamp(start)
    assert df.index.max() <= pd.Timestamp(end)


@pytest.mark.network
def test_breadth_cache_round_trip(tmp_path, monkeypatch):
    """Second call with cache returns identical data."""
    monkeypatch.setattr(m, 'CACHE_PATH', lambda: tmp_path / 'breadth.parquet')
    df1 = load_breadth(datetime(2024, 1, 1), datetime(2024, 1, 31), cache=True)
    df2 = load_breadth(datetime(2024, 1, 1), datetime(2024, 1, 31), cache=True)
    pd.testing.assert_frame_equal(df1, df2)
