import datetime as dt

from src.backtesting.engine.intraday_order_engine import (
    Bar, Order, OrderEngine)


def _bar(o, h, l, c, minute=0):
    return Bar(dt.datetime(2024, 1, 2, 8, minute, tzinfo=dt.timezone.utc), o, h, l, c)


def test_buy_stop_fills_at_trigger_when_bar_straddles():
    eng = OrderEngine()
    oid = eng.add_order(Order(side="buy", kind="stop", trigger=1.2500, qty=1.0))
    eng.match_resting_orders(_bar(1.2480, 1.2510, 1.2475, 1.2505))  # high crosses T, open below
    f = eng.fills[-1]
    assert f.order_id == oid and abs(f.price - 1.2500) < 1e-12  # max(T, open)=T


def test_buy_stop_gap_through_fills_at_open():
    eng = OrderEngine()
    eng.add_order(Order(side="buy", kind="stop", trigger=1.2500, qty=1.0))
    eng.match_resting_orders(_bar(1.2520, 1.2530, 1.2515, 1.2525))  # opened above T
    assert abs(eng.fills[-1].price - 1.2520) < 1e-12  # max(T, open)=open


def test_sell_stop_fills_at_min_trigger_open():
    eng = OrderEngine()
    eng.add_order(Order(side="sell", kind="stop", trigger=1.2400, qty=1.0))
    eng.match_resting_orders(_bar(1.2390, 1.2395, 1.2380, 1.2385))  # opened below T
    assert abs(eng.fills[-1].price - 1.2390) < 1e-12  # min(T, open)=open


def test_no_fill_when_bar_does_not_reach_trigger():
    eng = OrderEngine()
    eng.add_order(Order(side="buy", kind="stop", trigger=1.2500, qty=1.0))
    eng.match_resting_orders(_bar(1.2470, 1.2490, 1.2460, 1.2480))
    assert eng.fills == []


def test_order_added_this_bar_not_eligible_until_next():
    eng = OrderEngine()
    b = _bar(1.2480, 1.2510, 1.2475, 1.2505)
    # simulate: order armed AT this bar's ts must not fill against the same bar
    oid = eng.add_order(Order(side="buy", kind="stop", trigger=1.2500, qty=1.0),
                        armed_at=b.ts)
    eng.match_resting_orders(b)
    assert eng.fills == []  # same-ts bar excluded
    b2 = Bar(b.ts + dt.timedelta(minutes=1), 1.2490, 1.2510, 1.2485, 1.2505)
    eng.match_resting_orders(b2)
    assert len(eng.fills) == 1 and eng.fills[0].order_id == oid
