import pytest

from src.backtesting.costs.fx import (
    _DEFAULT_COMMISSION_BPS_PER_SIDE,
    _MEASURED_RT_BPS,
    _UNMEASURED_RT_BPS,
    fx_round_trip_bps_at,
    hour_of_week_multiplier,
    load_hour_of_week_surface,
)


def _commission() -> float:
    return 2.0 * _DEFAULT_COMMISSION_BPS_PER_SIDE


def test_surface_loads_and_covers_the_measured_pairs():
    surf = load_hour_of_week_surface()
    assert len(surf) > 1000
    assert {"EURUSD", "USDJPY", "GBPUSD", "USDCAD", "AUDUSD"} <= set(surf["pair"])


def test_each_pair_multiplier_is_quote_weighted_normalised():
    """The multiplier carries SHAPE only; level stays owned by _MEASURED_RT_BPS."""
    surf = load_hour_of_week_surface()
    for pair, grp in surf.groupby("pair"):
        weighted = (grp["spread_multiplier"] * grp["n_quotes"]).sum() / grp["n_quotes"].sum()
        assert weighted == pytest.approx(1.0, abs=0.01), pair


def test_surface_is_not_flat_for_a_major():
    """Regression on the synthetic-model defect: a constant is not a surface.

    The superseded spread_model emitted one value for all 168 hours plus a single
    spike hour. Real EURUSD spreads span an order of magnitude across the week.
    """
    mults = [hour_of_week_multiplier("EURUSD", h) for h in range(120)]
    assert max(mults) / min(mults) > 5.0


def test_cost_at_hour_scales_spread_but_not_commission():
    base = _MEASURED_RT_BPS["EURUSD"]
    mult = hour_of_week_multiplier("EURUSD", 40)
    assert fx_round_trip_bps_at("EURUSD", 40) == pytest.approx(base * mult + _commission())


def test_liquid_hour_is_cheaper_than_illiquid_hour():
    mults = {h: hour_of_week_multiplier("EURUSD", h) for h in range(120)}
    cheapest = min(mults, key=mults.get)
    dearest = max(mults, key=mults.get)
    assert fx_round_trip_bps_at("EURUSD", cheapest) < fx_round_trip_bps_at("EURUSD", dearest)


def test_unquoted_hour_charges_the_pairs_widest_observed_spread():
    """Weekend hours have no quotes. Silently charging the mean would be optimistic."""
    surf = load_hour_of_week_surface()
    observed = set(surf[surf["pair"] == "EURUSD"]["hour_of_week"])
    missing = [h for h in range(168) if h not in observed]
    assert missing, "expected unquoted weekend hours"
    widest = max(hour_of_week_multiplier("EURUSD", h) for h in observed)
    assert hour_of_week_multiplier("EURUSD", missing[0]) == pytest.approx(widest)


def test_unmeasured_pair_falls_back_to_flat_shape_and_unmeasured_level():
    assert hour_of_week_multiplier("EURPLN", 40) == 1.0
    assert fx_round_trip_bps_at("EURPLN", 40) == pytest.approx(
        _UNMEASURED_RT_BPS + _commission())


@pytest.mark.parametrize("bad_hour", [-1, 168, 999])
def test_out_of_range_hour_raises(bad_hour):
    with pytest.raises(ValueError):
        hour_of_week_multiplier("EURUSD", bad_hour)
