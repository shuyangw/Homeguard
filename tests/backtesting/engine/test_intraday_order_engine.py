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


from src.backtesting.engine.intraday_order_engine import Position


def test_oco_one_leg_fill_cancels_sibling():
    eng = OrderEngine()
    grp = eng.add_oco(
        Order(side="buy", kind="stop", trigger=1.2500, qty=1.0),
        Order(side="sell", kind="stop", trigger=1.2400, qty=1.0))
    assert grp is not None
    eng.match_resting_orders(_bar(1.2480, 1.2510, 1.2475, 1.2505))  # buy leg triggers
    assert len(eng.fills) == 1
    assert eng.resting_orders == []  # sibling cancelled


def test_oco_wide_bar_spanning_both_legs_fills_only_one():
    eng = OrderEngine()
    eng.add_oco(
        Order(side="buy", kind="stop", trigger=1.2500, qty=1.0),
        Order(side="sell", kind="stop", trigger=1.2400, qty=1.0))
    # one wide bar reaches BOTH triggers; OCO is atomic -> at most one leg fills
    eng.match_resting_orders(_bar(1.2450, 1.2550, 1.2350, 1.2500))
    assert len(eng.fills) == 1
    assert eng.resting_orders == []  # both legs resolved (one filled, sibling cancelled)


def test_bracket_partial_take_then_trail_closes_remainder():
    eng = OrderEngine()
    # long 1.0 @ 1.2500, stop 1.2450, target 1.2550, take half, trail 0.0030
    eng.open_position(side="buy", qty=1.0, entry_price=1.2500,
                      entry_ts=dt.datetime(2024, 1, 2, 8, 5, tzinfo=dt.timezone.utc),
                      stop=1.2450, target=1.2550, tp_fraction=0.5, trail_dist=0.0030)
    # bar reaches target -> take half; high 1.2560 arms trail at 1.2560-0.0030=1.2530
    ev = eng.update_position(_bar(1.2540, 1.2560, 1.2535, 1.2555, minute=6))
    assert any(r == "target" for r, _, _ in ev)
    assert abs(eng.position.qty - 0.5) < 1e-12  # half remains
    # next bar pulls back through the trailed stop 1.2530 -> remainder closes
    ev2 = eng.update_position(_bar(1.2532, 1.2534, 1.2520, 1.2525, minute=7))
    assert any(r == "trail" for r, _, _ in ev2)
    assert eng.position is None


def test_both_stop_and_target_in_one_bar_resolves_to_stop():
    eng = OrderEngine()
    eng.open_position(side="buy", qty=1.0, entry_price=1.2500,
                      entry_ts=dt.datetime(2024, 1, 2, 8, 5, tzinfo=dt.timezone.utc),
                      stop=1.2450, target=1.2550, tp_fraction=0.5, trail_dist=0.0030)
    # bar spans BOTH stop and target -> adverse (stop) wins, full close
    ev = eng.update_position(_bar(1.2500, 1.2560, 1.2440, 1.2455, minute=6))
    assert any(r == "stop" for r, _, _ in ev)
    assert eng.position is None


def test_partial_take_and_trail_breach_same_bar_closes_remainder():
    eng = OrderEngine()
    eng.open_position(side="buy", qty=1.0, entry_price=1.2500,
                      entry_ts=dt.datetime(2024, 1, 2, 8, 5, tzinfo=dt.timezone.utc),
                      stop=1.2450, target=1.2550, tp_fraction=0.5, trail_dist=0.0030)
    # high 1.2560 arms trail at 1.2530; same bar's low 1.2500 breaches it -> close remainder now
    ev = eng.update_position(_bar(1.2510, 1.2560, 1.2500, 1.2505, minute=6))
    assert any(r == "target" for r, _, _ in ev)
    trail_ev = [(r, px, q) for r, px, q in ev if r == "trail"]
    assert len(trail_ev) == 1
    assert abs(trail_ev[0][1] - 1.2530) < 1e-12
    assert abs(trail_ev[0][2] - 0.5) < 1e-12
    assert eng.position is None


def test_trailing_arm_does_not_loosen_stop():
    eng = OrderEngine()
    eng.open_position(side="buy", qty=1.0, entry_price=1.2500,
                      entry_ts=dt.datetime(2024, 1, 2, 8, 5, tzinfo=dt.timezone.utc),
                      stop=1.2480, target=1.2550, tp_fraction=0.5, trail_dist=0.0080)
    # extreme 1.2555 - 0.0080 = 1.2475 would loosen; must ratchet, stop stays 1.2480
    ev = eng.update_position(_bar(1.2510, 1.2555, 1.2505, 1.2550, minute=6))
    assert any(r == "target" for r, _, _ in ev)
    assert eng.position is not None
    assert eng.position.stop >= 1.2480 - 1e-12
    assert abs(eng.position.stop - 1.2480) < 1e-12


def test_tp_fraction_one_fully_closes_at_target():
    eng = OrderEngine()
    eng.open_position(side="buy", qty=1.0, entry_price=1.2500,
                      entry_ts=dt.datetime(2024, 1, 2, 8, 5, tzinfo=dt.timezone.utc),
                      stop=1.2450, target=1.2550, tp_fraction=1.0, trail_dist=0.0030)
    ev = eng.update_position(_bar(1.2540, 1.2560, 1.2535, 1.2555, minute=6))
    assert any(r == "target" for r, _, _ in ev)
    assert eng.position is None
    ev2 = eng.update_position(_bar(1.2560, 1.2570, 1.2555, 1.2565, minute=7))
    assert ev2 == []


def test_post_partial_stopout_labeled_stop_when_no_trailing():
    eng = OrderEngine()
    eng.open_position(side="buy", qty=1.0, entry_price=1.2500,
                      entry_ts=dt.datetime(2024, 1, 2, 8, 5, tzinfo=dt.timezone.utc),
                      stop=1.2450, target=1.2550, tp_fraction=0.5, trail_dist=None)
    ev = eng.update_position(_bar(1.2540, 1.2560, 1.2535, 1.2555, minute=6))
    assert any(r == "target" for r, _, _ in ev)
    assert abs(eng.position.qty - 0.5) < 1e-12
    ev2 = eng.update_position(_bar(1.2455, 1.2460, 1.2440, 1.2445, minute=7))
    reasons = [r for r, _, _ in ev2]
    assert "stop" in reasons
    assert "trail" not in reasons
    assert eng.position is None


def test_run_is_causal_order_placed_on_bar_fills_next():
    eng = OrderEngine()

    class S:
        placed = False
        def on_bar(self, bar, engine):
            if not self.placed:
                engine.add_order(Order(side="buy", kind="stop", trigger=1.2500, qty=1.0),
                                 armed_at=bar.ts)
                self.placed = True

    bars = [_bar(1.2480, 1.2510, 1.2475, 1.2505, minute=0),   # would cross, but order not yet armed here
            _bar(1.2490, 1.2510, 1.2485, 1.2505, minute=1)]   # fills here
    eng.run(bars, S())
    assert len(eng.fills) == 1 and eng.fills[0].ts.minute == 1


def test_stop_out_records_entry_and_exit_fills():
    eng = OrderEngine()
    eng.add_order(Order(side="buy", kind="stop", trigger=1.2500, qty=1.0))
    eng.match_resting_orders(_bar(1.2480, 1.2510, 1.2475, 1.2505))  # entry fill
    entry = eng.fills[-1]
    assert entry.side == "buy"
    eng.open_position(side="buy", qty=1.0, entry_price=1.2500,
                      entry_ts=dt.datetime(2024, 1, 2, 8, 5, tzinfo=dt.timezone.utc),
                      stop=1.2450, target=1.2550, tp_fraction=0.5, trail_dist=None)
    ev = eng.update_position(_bar(1.2455, 1.2460, 1.2440, 1.2445, minute=6))
    assert any(r == "stop" for r, _, _ in ev)
    exit_fill = eng.fills[-1]
    assert len(eng.fills) == 2  # entry AND exit
    assert exit_fill.side == "sell"  # opposite the long entry
    assert abs(exit_fill.price - 1.2450) < 1e-12  # exit at stop price
    assert abs(exit_fill.qty - 1.0) < 1e-12


def test_target_take_records_entry_and_exit_fills():
    eng = OrderEngine()
    eng.add_order(Order(side="buy", kind="stop", trigger=1.2500, qty=1.0))
    eng.match_resting_orders(_bar(1.2480, 1.2510, 1.2475, 1.2505))  # entry fill
    eng.open_position(side="buy", qty=1.0, entry_price=1.2500,
                      entry_ts=dt.datetime(2024, 1, 2, 8, 5, tzinfo=dt.timezone.utc),
                      stop=1.2450, target=1.2550, tp_fraction=0.5, trail_dist=None)
    ev = eng.update_position(_bar(1.2540, 1.2560, 1.2535, 1.2555, minute=6))
    assert any(r == "target" for r, _, _ in ev)
    exit_fill = eng.fills[-1]
    assert len(eng.fills) == 2  # entry AND partial exit
    assert exit_fill.side == "sell"  # opposite the long entry
    assert abs(exit_fill.price - 1.2550) < 1e-12  # exit at target price
    assert abs(exit_fill.qty - 0.5) < 1e-12  # partial (tp_fraction)


def test_short_stop_out_exit_side_is_buy():
    eng = OrderEngine()
    eng.open_position(side="sell", qty=1.0, entry_price=1.2500,
                      entry_ts=dt.datetime(2024, 1, 2, 8, 5, tzinfo=dt.timezone.utc),
                      stop=1.2550, target=1.2450, tp_fraction=0.5, trail_dist=None)
    ev = eng.update_position(_bar(1.2545, 1.2560, 1.2540, 1.2555, minute=6))
    assert any(r == "stop" for r, _, _ in ev)
    exit_fill = eng.fills[-1]
    assert exit_fill.side == "buy"  # opposite the short entry
    assert abs(exit_fill.price - 1.2550) < 1e-12


def test_flatten_records_exit_fill():
    eng = OrderEngine()
    eng.open_position(side="buy", qty=1.0, entry_price=1.2500,
                      entry_ts=dt.datetime(2024, 1, 2, 8, 5, tzinfo=dt.timezone.utc),
                      stop=1.2450, target=1.2550, tp_fraction=0.5, trail_dist=None)
    ts = dt.datetime(2024, 1, 2, 8, 30, tzinfo=dt.timezone.utc)
    ev = eng.flatten(1.2520, ts, reason="eod")
    assert any(r == "eod" for r, _, _ in ev)
    exit_fill = eng.fills[-1]
    assert exit_fill.side == "sell"  # opposite the long entry
    assert abs(exit_fill.price - 1.2520) < 1e-12
    assert exit_fill.ts == ts


def test_cancel_all_and_flatten_from_strategy():
    eng = OrderEngine()

    class S:
        def on_bar(self, bar, engine):
            if bar.ts.minute == 0:
                engine.add_order(Order(side="buy", kind="stop", trigger=1.9999, qty=1.0),
                                 armed_at=bar.ts)  # never fills
            if bar.ts.minute == 1:
                engine.cancel_all_resting()

    bars = [_bar(1.2480, 1.2490, 1.2470, 1.2485, minute=0),
            _bar(1.2480, 1.2490, 1.2470, 1.2485, minute=1)]
    eng.run(bars, S())
    assert eng.resting_orders == []
