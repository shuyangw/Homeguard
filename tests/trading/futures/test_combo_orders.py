"""Tests for FuturesComboOrderBuilder."""
import pytest

from src.trading.futures.combo_orders import (
    ComboLegSpec, ComboOrderSpec, ComboOrderRejected, FuturesComboOrderBuilder,
)


def test_build_calendar_roll_2_legs():
    """Calendar roll: SELL front, BUY back. Both legs same root."""
    builder = FuturesComboOrderBuilder()
    combo = builder.build_calendar_roll(
        symbol_root="ES", from_month="202403", to_month="202406", quantity=2,
    )
    assert isinstance(combo, ComboOrderSpec)
    assert len(combo.legs) == 2
    assert combo.exchange == "GLOBEX"
    # Leg 1: sell the from-month
    assert combo.legs[0].symbol_root == "ES"
    assert combo.legs[0].contract_month == "202403"
    assert combo.legs[0].action == "SELL"
    assert combo.legs[0].ratio == 2
    # Leg 2: buy the to-month
    assert combo.legs[1].symbol_root == "ES"
    assert combo.legs[1].contract_month == "202406"
    assert combo.legs[1].action == "BUY"
    assert combo.legs[1].ratio == 2


def test_build_calendar_roll_short_position_reverses_sides():
    """Negative quantity = short. Roll a short: BUY front (cover), SELL back (re-short)."""
    builder = FuturesComboOrderBuilder()
    combo = builder.build_calendar_roll(
        symbol_root="ES", from_month="202403", to_month="202406", quantity=-2,
    )
    assert combo.legs[0].action == "BUY"
    assert combo.legs[1].action == "SELL"
    assert combo.legs[0].ratio == 2  # ratio is absolute value of quantity
    assert combo.legs[1].ratio == 2


def test_build_inter_commodity_spread():
    """Inter-commodity spread (e.g. ES vs NQ): leg roots differ."""
    builder = FuturesComboOrderBuilder()
    combo = builder.build_inter_commodity_spread(
        leg_a_symbol="ES", leg_a_month="202406", leg_a_qty=3,
        leg_b_symbol="NQ", leg_b_month="202406", leg_b_qty=-2,
    )
    assert len(combo.legs) == 2
    assert combo.legs[0].symbol_root == "ES"
    assert combo.legs[0].action == "BUY"
    assert combo.legs[0].ratio == 3
    assert combo.legs[1].symbol_root == "NQ"
    assert combo.legs[1].action == "SELL"
    assert combo.legs[1].ratio == 2


def test_combo_order_rejected_exception_exists():
    """ComboOrderRejected is the exception type that bans separate-leg fallback."""
    with pytest.raises(ComboOrderRejected, match="manual"):
        raise ComboOrderRejected("IBKR rejected combo; operator must investigate manual")


def test_combo_leg_spec_immutable():
    """ComboLegSpec is frozen (cannot be mutated post-construction)."""
    import dataclasses
    leg = ComboLegSpec(symbol_root="ES", contract_month="202406", action="BUY", ratio=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        leg.action = "SELL"
