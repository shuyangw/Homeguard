"""Exit-side fill schema: reason, excursions, holding period, round-trip linkage.

Methodology Section 11.9 requires trade-level exit diagnostics. The engine
already computed an exit `reason` internally and discarded it at the Fill
boundary, so every persisted exit was indistinguishable from every other.
"""
import datetime as dt
import math

import pytest

from src.backtesting.engine.intraday_order_engine import Bar, Fill, Order, OrderEngine

_T0 = dt.datetime(2024, 1, 10, 8, 0)


def _bar(i, o, h, lo, c):
    return Bar(_T0 + dt.timedelta(minutes=i), o, h, lo, c)


def _long_at(engine, price=100.0, qty=1.0, stop=99.0, target=102.0,
             tp_fraction=1.0, trail_dist=None):
    return engine.open_position("buy", qty, price, _T0, stop, target,
                                tp_fraction, trail_dist)


def test_fill_is_backward_compatible_positionally():
    """Existing callers construct Fill with five positional fields."""
    f = Fill(1, _T0, 1.25, 2.0, "buy")
    assert (f.order_id, f.ts, f.price, f.qty, f.side) == (1, _T0, 1.25, 2.0, "buy")
    assert f.reason == ""


def test_entry_fill_has_no_exit_reason():
    eng = OrderEngine()
    eng.add_order(Order(side="buy", kind="stop", trigger=100.0, qty=1.0))
    fills = eng.match_resting_orders(_bar(1, 100.0, 101.0, 99.5, 100.5))
    assert len(fills) == 1 and fills[0].reason == ""


@pytest.mark.parametrize("hi,lo,expected", [
    (100.5, 98.0, "stop"),      # stop breached
    (103.0, 99.5, "target"),    # target reached, full size
])
def test_exit_reason_is_recorded(hi, lo, expected):
    eng = OrderEngine()
    _long_at(eng)
    eng.update_position(_bar(1, 100.0, hi, lo, 100.0))
    exits = [f for f in eng.fills if f.reason]
    assert [f.reason for f in exits] == [expected]


def test_flatten_records_its_reason():
    eng = OrderEngine()
    _long_at(eng)
    eng.flatten(100.5, _T0 + dt.timedelta(minutes=5), reason="flat_1600")
    assert [f.reason for f in eng.fills] == ["flat_1600"]


def test_mfe_and_mae_track_the_excursion_before_exit():
    eng = OrderEngine()
    _long_at(eng, price=100.0, stop=97.0, target=110.0)
    eng.update_position(_bar(1, 100.0, 104.0, 99.0, 103.0))   # +4 / -1
    eng.update_position(_bar(2, 103.0, 103.5, 98.5, 99.0))    # deeper adverse
    eng.flatten(99.0, _T0 + dt.timedelta(minutes=3))
    exit_fill = eng.fills[-1]
    assert exit_fill.mfe == pytest.approx(4.0)
    assert exit_fill.mae == pytest.approx(-1.5)


def test_mae_mfe_sign_convention_is_favourable_positive_for_shorts():
    eng = OrderEngine()
    eng.open_position("sell", 1.0, 100.0, _T0, 103.0, 90.0, 1.0, None)
    eng.update_position(_bar(1, 100.0, 101.0, 96.0, 97.0))    # short profits as price falls
    eng.flatten(97.0, _T0 + dt.timedelta(minutes=2))
    exit_fill = eng.fills[-1]
    assert exit_fill.mfe == pytest.approx(4.0)
    assert exit_fill.mae == pytest.approx(-1.0)


def test_bars_held_counts_bars_since_entry():
    eng = OrderEngine()
    _long_at(eng, stop=90.0, target=110.0)
    for i in range(1, 4):
        eng.update_position(_bar(i, 100.0, 100.5, 99.5, 100.0))
    eng.flatten(100.0, _T0 + dt.timedelta(minutes=4))
    assert eng.fills[-1].bars_held == 3


def test_exit_carries_entry_price_and_ts_for_round_trip_reconstruction():
    eng = OrderEngine()
    _long_at(eng, price=100.0, stop=99.0, target=102.0)
    eng.update_position(_bar(1, 100.0, 100.5, 98.0, 98.5))
    exit_fill = eng.fills[-1]
    assert exit_fill.entry_price == pytest.approx(100.0)
    assert exit_fill.entry_ts == _T0


def test_partial_then_trail_exits_share_one_trade_id():
    eng = OrderEngine()
    _long_at(eng, qty=2.0, stop=99.0, target=102.0, tp_fraction=0.5, trail_dist=1.0)
    eng.update_position(_bar(1, 100.0, 102.5, 99.5, 102.0))   # target -> partial
    eng.update_position(_bar(2, 102.0, 102.0, 100.0, 100.2))  # trail breached
    exits = [f for f in eng.fills if f.reason]
    assert len(exits) == 2
    assert exits[0].trade_id == exits[1].trade_id
    assert exits[0].reason == "target" and exits[1].reason == "trail"


def test_separate_positions_get_distinct_trade_ids():
    eng = OrderEngine()
    _long_at(eng)
    eng.flatten(100.0, _T0 + dt.timedelta(minutes=1))
    _long_at(eng)
    eng.flatten(100.0, _T0 + dt.timedelta(minutes=2))
    exits = [f for f in eng.fills if f.reason]
    assert exits[0].trade_id != exits[1].trade_id


def test_entry_fill_excursions_are_nan_not_zero():
    """Zero would be a claim about the excursion; NaN says it does not apply."""
    eng = OrderEngine()
    eng.add_order(Order(side="buy", kind="stop", trigger=100.0, qty=1.0))
    f = eng.match_resting_orders(_bar(1, 100.0, 101.0, 99.5, 100.5))[0]
    assert math.isnan(f.mae) and math.isnan(f.mfe)


def test_trade_ids_are_unique_ACROSS_engines():
    """Reproduces the deployment topology, which the single-engine test did not.

    The runner builds a fresh OrderEngine per FX trading day. With an
    engine-local counter every day's first position took the same trade_id: a
    real run produced 2 distinct values across 14838 fill rows.
    """
    ids = []
    for _ in range(3):
        eng = OrderEngine()
        _long_at(eng)
        eng.flatten(100.0, _T0 + dt.timedelta(minutes=1))
        ids.append([f.trade_id for f in eng.fills if f.reason][0])
    assert len(set(ids)) == 3, f"trade_ids collided across engines: {ids}"
