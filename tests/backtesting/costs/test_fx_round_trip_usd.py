"""FX round-trip cost: MEASURED spread + commission, in bps of notional.

Rewritten 2026-07-26. The previous assertions pinned the PIP-tier model, which
charged in pips -- a different fraction of price at different price levels, so it
under-charged high-priced crosses by up to 20x and over-charged majors by up to
7x. Those tests were faithfully asserting a broken model.
"""
import pytest

from src.backtesting.costs.fx import (
    _DEFAULT_COMMISSION_BPS_PER_SIDE, _MEASURED_RT_BPS, _UNMEASURED_RT_BPS,
    fx_round_trip_pips, fx_round_trip_usd,
)


def _bps(pair, units, price, q2u, **kw):
    cost = fx_round_trip_usd(pair, units, price=price, quote_to_usd=q2u, **kw)
    return cost / (units * price * q2u) * 1e4


def test_cost_is_measured_spread_plus_two_sided_commission():
    got = _bps("EURUSD", 100_000.0, 1.10, 1.0)
    want = _MEASURED_RT_BPS["EURUSD"] + 2 * _DEFAULT_COMMISSION_BPS_PER_SIDE
    assert got == pytest.approx(want)


def test_cost_is_scale_invariant_across_price_levels():
    """The pip model's core failure: a similar spread in bps must cost a similar
    fraction of notional whether the pair prints at 1.08 or 151."""
    a = _bps("EURUSD", 1_000_000.0, 1.083, 1.0)
    b = _bps("USDJPY", 1_000_000.0, 151.4, 1 / 151.4)
    assert abs(a - b) < 0.05


def test_high_priced_cross_is_no_longer_undercharged():
    """EURNOK was 20x too cheap under the pip model (0.21bps vs 4.32 measured)."""
    got = _bps("EURNOK", 1_000_000.0, 11.671, 1 / 11.671)
    assert got == pytest.approx(_MEASURED_RT_BPS["EURNOK"] + 0.40)
    assert got > 4.0


def test_silver_costs_more_than_gold():
    """Measured: XAG 10.41 bps vs XAU 1.63. The flat 4bps metals constant priced
    them identically, under-costing silver by 2.6x."""
    ag = _bps("XAGUSD", 1_000.0, 29.0, 1.0)
    au = _bps("XAUUSD", 100.0, 2380.0, 1.0)
    assert ag > au and ag > 10.0


def test_commission_is_overridable_for_a_cost_ladder():
    raw = _bps("EURUSD", 100_000.0, 1.10, 1.0, commission_bps_per_side=0.0)
    assert raw == pytest.approx(_MEASURED_RT_BPS["EURUSD"])
    dbl = _bps("EURUSD", 100_000.0, 1.10, 1.0, commission_bps_per_side=0.40)
    assert dbl == pytest.approx(_MEASURED_RT_BPS["EURUSD"] + 0.80)


def test_unmeasured_pair_falls_back_not_crashes():
    """Triangulated crosses (NOKSEK/NOKJPY/SEKJPY) have no quoted spread."""
    got = _bps("NOKSEK", 1_000_000.0, 0.98, 1.0)
    assert got == pytest.approx(_UNMEASURED_RT_BPS + 0.40)


def test_cost_uses_absolute_units():
    a = fx_round_trip_usd("EURUSD", 50_000.0, 1.1, 1.0)
    b = fx_round_trip_usd("EURUSD", -50_000.0, 1.1, 1.0)
    assert a == pytest.approx(b)


def test_pip_helper_retained_for_reference_only():
    """fx_round_trip_pips is no longer on the costing path; kept so historical
    reports stay reproducible."""
    assert fx_round_trip_pips("major", "ny") > 0
