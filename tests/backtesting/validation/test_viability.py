"""Statistical-viability screen.

A spec whose best-case-if-true Sharpe cannot clear the current deflated bar is
arithmetically incapable of passing, however good its thesis. Spending a trial
on it raises the bar for everything after it and buys nothing. Most of the 141
trials in the FX campaign were in that category.
"""
import math

import pytest

from src.backtesting.validation.viability import (ViabilityResult,
                                                  expected_cost_bps,
                                                  if_true_sharpe, screen_spec)


def test_formula_matches_the_worked_example():
    """sqrt(T) * (edge - cost) / per-trade vol."""
    got = if_true_sharpe(trades_per_year=200, gross_edge_bps=5.0,
                         cost_bps=3.5, per_trade_vol_bps=25.0)
    assert got == pytest.approx(math.sqrt(200) * 1.5 / 25.0)


def test_cost_exceeding_edge_gives_a_negative_sharpe():
    assert if_true_sharpe(100, 2.0, 3.0, 20.0) < 0


def test_more_trades_raise_the_if_true_sharpe_at_fixed_edge():
    few = if_true_sharpe(50, 5.0, 1.0, 20.0)
    many = if_true_sharpe(200, 5.0, 1.0, 20.0)
    assert many == pytest.approx(2.0 * few)


@pytest.mark.parametrize("kwargs", [
    dict(trades_per_year=0, gross_edge_bps=5.0, cost_bps=1.0, per_trade_vol_bps=20.0),
    dict(trades_per_year=100, gross_edge_bps=5.0, cost_bps=1.0, per_trade_vol_bps=0.0),
    dict(trades_per_year=-1, gross_edge_bps=5.0, cost_bps=1.0, per_trade_vol_bps=20.0),
])
def test_degenerate_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        if_true_sharpe(**kwargs)


def test_expected_cost_uses_the_measured_surface_for_the_hours_traded():
    """A spec trading only liquid hours must not be charged the weekly average."""
    liquid = expected_cost_bps(["EURUSD"], hours_of_week=[2 * 24 + 13])   # Wed 13:00 UTC
    illiquid = expected_cost_bps(["EURUSD"], hours_of_week=[2 * 24 + 21])  # Wed 21:00 UTC
    assert liquid < illiquid


def test_expected_cost_averages_across_the_pairs_traded():
    both = expected_cost_bps(["EURUSD", "USDNOK"], hours_of_week=[2 * 24 + 13])
    eur = expected_cost_bps(["EURUSD"], hours_of_week=[2 * 24 + 13])
    nok = expected_cost_bps(["USDNOK"], hours_of_week=[2 * 24 + 13])
    assert both == pytest.approx((eur + nok) / 2.0)
    assert eur < nok


def test_screen_fails_a_spec_that_cannot_reach_the_bar():
    res = screen_spec(name="thin", trades_per_year=60, gross_edge_bps=2.0,
                      per_trade_vol_bps=30.0, pairs=["EURUSD"],
                      hours_of_week=[2 * 24 + 13], sr_zero=1.14)
    assert isinstance(res, ViabilityResult)
    assert not res.viable
    assert res.if_true_sharpe < 1.14


def test_screen_passes_a_spec_that_clears_the_bar():
    res = screen_spec(name="fat", trades_per_year=250, gross_edge_bps=12.0,
                      per_trade_vol_bps=25.0, pairs=["EURUSD"],
                      hours_of_week=[2 * 24 + 13], sr_zero=1.14)
    assert res.viable and res.if_true_sharpe > 1.14


def test_screen_reports_the_cost_it_charged():
    res = screen_spec(name="x", trades_per_year=100, gross_edge_bps=5.0,
                      per_trade_vol_bps=20.0, pairs=["EURUSD"],
                      hours_of_week=[2 * 24 + 13], sr_zero=1.14)
    assert res.cost_bps == pytest.approx(
        expected_cost_bps(["EURUSD"], [2 * 24 + 13]))
    assert res.margin == pytest.approx(res.if_true_sharpe - 1.14)


def test_screen_is_stricter_when_the_spec_trades_illiquid_hours():
    common = dict(name="x", trades_per_year=200, gross_edge_bps=5.0,
                  per_trade_vol_bps=25.0, pairs=["EURUSD"], sr_zero=1.14)
    liquid = screen_spec(hours_of_week=[2 * 24 + 13], **common)
    illiquid = screen_spec(hours_of_week=[2 * 24 + 21], **common)
    assert liquid.if_true_sharpe > illiquid.if_true_sharpe


def test_small_notional_raises_the_screened_cost():
    """The $2 per-order commission minimum binds below $100k of notional."""
    common = dict(pairs=["EURUSD"], hours_of_week=[2 * 24 + 13])
    big = expected_cost_bps(**common, notional_usd=500_000)
    small = expected_cost_bps(**common, notional_usd=10_000)
    assert small > big


def test_n_legs_multiplies_the_charge():
    common = dict(pairs=["EURUSD"], hours_of_week=[2 * 24 + 13])
    assert expected_cost_bps(**common, n_legs=2) == pytest.approx(
        2 * expected_cost_bps(**common))


def test_n_legs_below_one_raises():
    with pytest.raises(ValueError):
        expected_cost_bps(["EURUSD"], [2 * 24 + 13], n_legs=0)


def test_defaults_are_the_flattering_assumption_and_are_documented():
    """Omitting notional and legs must not silently change existing results."""
    common = dict(pairs=["EURUSD"], hours_of_week=[2 * 24 + 13])
    assert expected_cost_bps(**common) == pytest.approx(
        expected_cost_bps(**common, notional_usd=100_000, n_legs=1))
