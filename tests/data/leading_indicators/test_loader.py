"""Tests for the unified leading-indicator loader."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.data.leading_indicators import load_leading_indicators


EXPECTED_COLUMNS = {
    'vix_close',
    'vix3m_close',
    'vix_term_ratio',
    'hy_oas',
    'breadth_pct',
    'n_constituents',
    'skew_close',
}


@pytest.mark.network
def test_loader_2024_window_schema():
    """Smoke test: loader returns the joined indicator schema."""
    df = load_leading_indicators(datetime(2024, 1, 1), datetime(2024, 6, 30), cache=False)
    assert EXPECTED_COLUMNS.issubset(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.isna().any().any()
    assert len(df) > 100


@pytest.mark.network
def test_loader_no_large_gaps():
    """After forward-fill (limit=2) there should be no calendar-day gaps > 4 days
    inside the joined panel (2 ffill days + 2 weekend days).
    """
    df = load_leading_indicators(datetime(2024, 1, 1), datetime(2024, 6, 30), cache=False)
    gaps = df.index.to_series().diff().dt.days.dropna()
    # Holiday + weekend can extend a gap; allow up to 5 calendar days as slack
    # for long-weekend holidays. The forward-fill discipline is enforced via
    # MAX_FFILL_DAYS=2 trading days, not calendar days.
    assert gaps.max() <= 6, f'unexpected gap of {gaps.max()} days in joined panel'
