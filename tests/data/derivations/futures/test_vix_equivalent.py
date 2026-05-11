"""Tests for VIX-equivalent derivation from ES realized vol."""
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from src.data.derivations.futures.vix_equivalent import (
    TRADING_DAYS_PER_YEAR,
    WINDOW_DAYS,
    derive_vix_equivalent,
)


class _FakeLoader:
    def __init__(self, daily_df: pl.DataFrame):
        self._df = daily_df

    def aggregate_to_daily(self, root, method="ratio_adjusted",
                          start=None, end=None):
        df = self._df
        if start is not None:
            df = df.filter(pl.col("timestamp").dt.date() >= start)
        if end is not None:
            df = df.filter(pl.col("timestamp").dt.date() <= end)
        return df


def _make_daily(closes: list[float], end_date: date) -> pl.DataFrame:
    n = len(closes)
    timestamps = [
        datetime(
            (end_date - timedelta(days=n - 1 - i)).year,
            (end_date - timedelta(days=n - 1 - i)).month,
            (end_date - timedelta(days=n - 1 - i)).day,
            tzinfo=timezone.utc,
        )
        for i in range(n)
    ]
    return pl.DataFrame({
        "timestamp": pl.Series("timestamp", timestamps),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1000] * n,
    })


def test_constant_returns_zero_vol():
    closes = [100.0] * (WINDOW_DAYS + 5)
    loader = _FakeLoader(_make_daily(closes, date(2024, 6, 30)))
    vol = derive_vix_equivalent(date(2024, 6, 30), loader=loader)
    assert vol == pytest.approx(0.0, abs=1e-9)


def test_known_daily_vol_annualizes_correctly():
    """1% daily vol annualizes to ~15.87% (1 * sqrt(252))."""
    closes = []
    price = 100.0
    sign = 1
    for _ in range(WINDOW_DAYS + 5):
        closes.append(price)
        price = price * (1 + sign * 0.01)
        sign = -sign
    loader = _FakeLoader(_make_daily(closes, date(2024, 6, 30)))
    vol = derive_vix_equivalent(date(2024, 6, 30), loader=loader)
    # Alternating +/-1% returns -> ~1% daily stdev -> ~15.87% annualized
    # (bias-corrected variance pushes this slightly higher; widen tolerance)
    assert vol == pytest.approx(16.0, abs=1.0)


def test_insufficient_history_returns_nan():
    closes = [100.0, 101.0, 102.0]
    loader = _FakeLoader(_make_daily(closes, date(2024, 6, 30)))
    vol = derive_vix_equivalent(date(2024, 6, 30), loader=loader)
    import math
    assert math.isnan(vol)


def test_invalid_window_raises():
    loader = _FakeLoader(_make_daily([100.0] * 30, date(2024, 6, 30)))
    with pytest.raises(ValueError, match="window_days"):
        derive_vix_equivalent(date(2024, 6, 30), window_days=1, loader=loader)


def test_window_size_parameter():
    closes = [100.0 + i * 0.5 for i in range(15)]
    loader = _FakeLoader(_make_daily(closes, date(2024, 6, 30)))
    vol = derive_vix_equivalent(date(2024, 6, 30), window_days=10, loader=loader)
    assert vol > 0  # monotone increasing has nonzero stdev


def test_high_vol_above_low_vol():
    end = date(2024, 6, 30)
    low_closes = [100.0 + (i % 2) * 0.1 for i in range(30)]
    high_closes = [100.0 + (i % 2) * 5.0 for i in range(30)]
    low_loader = _FakeLoader(_make_daily(low_closes, end))
    high_loader = _FakeLoader(_make_daily(high_closes, end))
    low_vol = derive_vix_equivalent(end, loader=low_loader)
    high_vol = derive_vix_equivalent(end, loader=high_loader)
    assert high_vol > low_vol
    assert high_vol > 10.0  # 5% daily swings annualize to very high vol
