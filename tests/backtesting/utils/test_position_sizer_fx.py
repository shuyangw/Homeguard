import pytest

from src.backtesting.utils.position_sizer_fx import size_from_forecast_fx


def test_sizer_formula():
    # forecast 10 -> forecast/10 = 1.0
    # units = 1.0 * capital * vol_target * div_mult / (base_to_usd * ann_vol)
    units = size_from_forecast_fx(
        forecast=10.0, capital=100_000.0, vol_target=0.2,
        base_to_usd=1.10, daily_vol=0.01)
    ann_vol = 0.01 * (252 ** 0.5)
    expected = 1.0 * 100_000.0 * 0.2 / (1.10 * ann_vol)
    assert units == pytest.approx(expected)


def test_negative_forecast_gives_short():
    units = size_from_forecast_fx(-10.0, 100_000.0, 0.2, 1.10, 0.01)
    assert units < 0


def test_zero_vol_returns_zero():
    assert size_from_forecast_fx(10.0, 100_000.0, 0.2, 1.10, 0.0) == 0.0
