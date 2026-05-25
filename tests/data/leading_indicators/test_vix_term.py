"""Tests for VIX term structure acquirer."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.data.leading_indicators import load_vix_term
import src.data.leading_indicators.vix_term as v


@pytest.mark.network
def test_vix_term_2024_window():
    """Smoke test: downloading a small window returns expected schema."""
    df = load_vix_term(datetime(2024, 1, 1), datetime(2024, 1, 31), cache=False)
    assert {'vix_close', 'vix3m_close', 'vix_term_ratio'}.issubset(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert (df['vix_term_ratio'] > 0).all()
    assert df['vix_close'].notna().all()
    assert df['vix3m_close'].notna().all()
    assert len(df) > 10


@pytest.mark.network
def test_vix_term_date_range_within_request():
    """Returned dates fall within [start, end]."""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 31)
    df = load_vix_term(start, end, cache=False)
    assert df.index.min() >= pd.Timestamp(start)
    assert df.index.max() <= pd.Timestamp(end)


@pytest.mark.network
def test_vix_term_cache_round_trip(tmp_path, monkeypatch):
    """Second call with cache returns identical data."""
    monkeypatch.setattr(v, 'CACHE_PATH', lambda: tmp_path / 'vix_term.parquet')
    df1 = load_vix_term(datetime(2024, 1, 1), datetime(2024, 1, 31), cache=True)
    df2 = load_vix_term(datetime(2024, 1, 1), datetime(2024, 1, 31), cache=True)
    pd.testing.assert_frame_equal(df1, df2)
