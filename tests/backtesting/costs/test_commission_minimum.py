"""IBKR charges 0.20 bps of trade value per side with a $2 MINIMUM per order.

The rate was already modelled. The minimum was not, and it is the term that
bites: it stops binding only above $100,000 of notional per order, and a
retail-sized intraday FX trade sits well below that. At $25k capital with a
33-pip stop at 0.5% risk the notional is roughly $48k, so the true commission is
about twice the headline rate -- and commission is already the DOMINANT cost
term for tight majors, larger than the spread itself.

Ignoring the minimum therefore under-charges every small order, and it
under-charges the many-small-trades strategies hardest, which is precisely the
shape of an intraday spec.
"""
import pytest

from src.backtesting.costs.fx import (COMMISSION_MIN_USD, COMMISSION_RATE_BPS,
                                      effective_commission_bps,
                                      fx_round_trip_bps_at)

_LIQUID_HOUR = 2 * 24 + 13


def test_minimum_stops_binding_at_100k_notional():
    assert COMMISSION_MIN_USD / (COMMISSION_RATE_BPS / 1e4) == pytest.approx(100_000)


def test_large_orders_pay_the_headline_rate():
    for notional in (100_000, 500_000, 5_000_000):
        assert effective_commission_bps(notional) == pytest.approx(COMMISSION_RATE_BPS)


@pytest.mark.parametrize("notional,expected", [
    (50_000, 0.40),
    (25_000, 0.80),
    (10_000, 2.00),
    (5_000, 4.00),
])
def test_small_orders_pay_the_minimum(notional, expected):
    assert effective_commission_bps(notional) == pytest.approx(expected)


def test_unknown_notional_falls_back_to_the_headline_rate():
    """Stated rather than guessed: with no size we cannot know if the min binds."""
    assert effective_commission_bps(None) == pytest.approx(COMMISSION_RATE_BPS)


def test_effective_rate_is_monotone_in_size():
    sizes = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000]
    rates = [effective_commission_bps(s) for s in sizes]
    assert rates == sorted(rates, reverse=True)


def test_zero_or_negative_notional_raises():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            effective_commission_bps(bad)


def test_round_trip_cost_rises_as_order_size_falls():
    big = fx_round_trip_bps_at("EURUSD", _LIQUID_HOUR, notional_usd=500_000)
    small = fx_round_trip_bps_at("EURUSD", _LIQUID_HOUR, notional_usd=10_000)
    assert small > big
    # commission is charged twice, so the gap is 2x the per-side difference
    assert small - big == pytest.approx(
        2 * (effective_commission_bps(10_000) - effective_commission_bps(500_000)))


def test_omitting_notional_preserves_the_previous_behaviour():
    """Back-compat: existing callers pass no size and must be unaffected."""
    assert fx_round_trip_bps_at("EURUSD", _LIQUID_HOUR) == pytest.approx(
        fx_round_trip_bps_at("EURUSD", _LIQUID_HOUR, notional_usd=100_000))


def test_explicit_commission_override_still_wins():
    assert fx_round_trip_bps_at("EURUSD", _LIQUID_HOUR,
                                commission_bps_per_side=0.0,
                                notional_usd=1_000) == pytest.approx(
        fx_round_trip_bps_at("EURUSD", _LIQUID_HOUR, commission_bps_per_side=0.0))
