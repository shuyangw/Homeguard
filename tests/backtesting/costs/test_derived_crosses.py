"""EURGBP and GBPJPY costs derived from their measured USD legs.

Neither is quoted in the local tick archive nor in the Dukascopy sample, so both
took the flat 4.0 bps unmeasured fallback. The #20 re-gate showed that is not a
harmless default: GBPJPY was its largest leg at 1888 entries and the fallback
charged roughly 3.5x a defensible estimate.

A synthetic cross is the sum of its legs, so the legs' MEASURED spreads bound
the cross. That bound is conservative in the right direction: a bank quoting
EURGBP directly quotes tighter than someone crossing EURUSD and GBPUSD, so the
derived number over-charges rather than under-charges.
"""
import pytest

from src.backtesting.costs.fx import (_DERIVED_RT_BPS, _MEASURED_RT_BPS,
                                      _UNMEASURED_RT_BPS, DERIVED_CROSS_LEGS,
                                      fx_round_trip_bps_at,
                                      hour_of_week_multiplier)

_HOURS = [d * 24 + h for d in range(5) for h in range(24)]


@pytest.mark.parametrize("cross", ["EURGBP", "GBPJPY"])
def test_derived_level_is_the_sum_of_its_measured_legs(cross):
    a, b = DERIVED_CROSS_LEGS[cross]
    assert _DERIVED_RT_BPS[cross] == pytest.approx(
        _MEASURED_RT_BPS[a] + _MEASURED_RT_BPS[b])


@pytest.mark.parametrize("cross", ["EURGBP", "GBPJPY"])
def test_derived_crosses_are_cheaper_than_the_blanket_fallback(cross):
    assert _DERIVED_RT_BPS[cross] < _UNMEASURED_RT_BPS


@pytest.mark.parametrize("cross", ["EURGBP", "GBPJPY"])
def test_derived_crosses_are_not_flat_across_the_week(cross):
    mults = [hour_of_week_multiplier(cross, h) for h in _HOURS]
    assert max(mults) / min(mults) > 3.0


@pytest.mark.parametrize("cross", ["EURGBP", "GBPJPY"])
def test_derived_shape_lies_between_its_legs(cross):
    """The cross spread is a spread-weighted blend of the legs, so at any hour
    its multiplier cannot sit outside the two leg multipliers."""
    a, b = DERIVED_CROSS_LEGS[cross]
    for h in _HOURS:
        ma, mb = hour_of_week_multiplier(a, h), hour_of_week_multiplier(b, h)
        mc = hour_of_week_multiplier(cross, h)
        assert min(ma, mb) - 1e-6 <= mc <= max(ma, mb) + 1e-6, h


@pytest.mark.parametrize("cross", ["EURGBP", "GBPJPY"])
def test_cost_uses_the_derived_level_not_the_fallback(cross):
    """A liquid hour on a derived cross must beat the blanket unmeasured cost."""
    liquid = 2 * 24 + 13
    assert fx_round_trip_bps_at(cross, liquid) < _UNMEASURED_RT_BPS


def test_measured_pairs_are_untouched_by_the_derived_table():
    assert not (set(_DERIVED_RT_BPS) & set(_MEASURED_RT_BPS)), (
        "a derived value must never shadow a measured one")


def test_an_unrelated_pair_still_takes_the_fallback():
    assert fx_round_trip_bps_at("EURPLN", 2 * 24 + 13) > _UNMEASURED_RT_BPS - 1e-9
