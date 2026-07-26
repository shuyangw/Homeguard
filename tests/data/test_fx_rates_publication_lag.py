"""FRED publication lag: a monthly average must not be visible before it exists."""
import pandas as pd

from src.data.fx_rates import (
    _publication_lag_days, _MONTHLY_PUBLICATION_LAG_DAYS, _DAILY_PUBLICATION_LAG_DAYS,
)


def test_monthly_series_detected():
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=24, freq="MS"))
    assert _publication_lag_days(idx) == _MONTHLY_PUBLICATION_LAG_DAYS


def test_daily_series_detected():
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=200, freq="D"))
    assert _publication_lag_days(idx) == _DAILY_PUBLICATION_LAG_DAYS


def test_business_daily_series_detected_as_daily():
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=200, freq="B"))
    assert _publication_lag_days(idx) == _DAILY_PUBLICATION_LAG_DAYS


def test_monthly_lag_pushes_a_may_average_past_may():
    """The value FRED stamps 2026-05-01 is May's average, so it must not be
    usable during May -- that was the lookahead."""
    stamp = pd.Timestamp("2026-05-01")
    active = stamp + pd.Timedelta(days=_MONTHLY_PUBLICATION_LAG_DAYS)
    assert active > pd.Timestamp("2026-05-31"), "must not be visible inside its own month"
    assert active >= pd.Timestamp("2026-06-30"), "must also clear a realistic publication lag"


def test_two_point_daily_series_is_still_detected_as_daily():
    """Two observations are enough for one spacing diff. Requiring three made a
    short DAILY fixture fall through to the conservative MONTHLY lag, which
    NaN-ed the whole panel."""
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    assert _publication_lag_days(idx) == _DAILY_PUBLICATION_LAG_DAYS


def test_two_point_monthly_series_is_detected_as_monthly():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-02-01"])
    assert _publication_lag_days(idx) == _MONTHLY_PUBLICATION_LAG_DAYS


def test_single_observation_defaults_to_the_conservative_lag():
    idx = pd.DatetimeIndex(["2020-01-01"])
    assert _publication_lag_days(idx) == _MONTHLY_PUBLICATION_LAG_DAYS
