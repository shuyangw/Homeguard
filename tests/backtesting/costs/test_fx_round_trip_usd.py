import pytest

from src.backtesting.costs.fx import fx_round_trip_pips, fx_round_trip_usd


def test_major_usd_quote_cost():
    # EURUSD, quote USD -> quote_to_usd = 1.0, pip_size 0.0001
    units = 100_000.0
    cost = fx_round_trip_usd("EURUSD", units, price=1.10, quote_to_usd=1.0,
                             tier="major", session="ny")
    rt_pips = fx_round_trip_pips("major", "ny")
    assert cost == pytest.approx(rt_pips * 0.0001 * units * 1.0)


def test_jpy_quote_uses_2dp_pip():
    units = 100_000.0
    cost = fx_round_trip_usd("USDJPY", units, price=150.0, quote_to_usd=1 / 150.0,
                             tier="major", session="ny")
    rt_pips = fx_round_trip_pips("major", "ny")
    assert cost == pytest.approx(rt_pips * 0.01 * units * (1 / 150.0))


def test_metals_use_bps_of_notional():
    units = 100.0  # 100 oz gold
    cost = fx_round_trip_usd("XAUUSD", units, price=2000.0, quote_to_usd=1.0,
                             metals_bps=4.0)
    notional = 100.0 * 2000.0 * 1.0
    assert cost == pytest.approx(notional * 4.0 / 10_000)


def test_cost_uses_absolute_units():
    a = fx_round_trip_usd("EURUSD", 50_000.0, 1.1, 1.0)
    b = fx_round_trip_usd("EURUSD", -50_000.0, 1.1, 1.0)
    assert a == pytest.approx(b)
