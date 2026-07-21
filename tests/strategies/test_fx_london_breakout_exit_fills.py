"""Regression: LondonBreakout must not treat engine EXIT fills as entry triggers.

Commit f34dd6e made OrderEngine append EXIT fills to engine.fills (for trade
logging). LondonBreakout reads engine.fills to decide entries, so without a
filter an exit fill opens a phantom reversed position. This drives a real day
(entry breakout -> intraday stop-out) through the real OrderEngine and asserts
the exit does NOT spawn a second position.
"""
import datetime as dt

import pandas as pd

from src.backtesting.engine.intraday_order_engine import (
    Bar, EXIT_ORDER_ID, OrderEngine)
from src.strategies.advanced.fx_london_breakout import LondonBreakoutStrategy

_DAY = dt.date(2024, 1, 2)  # January: Europe/London == UTC (GMT)


def _bar(hour, minute, o, h, l, c):
    ts = dt.datetime(2024, 1, 2, hour, minute, tzinfo=dt.timezone.utc)
    return Bar(ts, o, h, l, c)


def _day_bars():
    return [
        _bar(1, 0, 1.2020, 1.2050, 1.2010, 1.2030),   # asian: sets high 1.2050
        _bar(4, 0, 1.2030, 1.2035, 1.2000, 1.2010),   # asian: sets low 1.2000
        _bar(8, 0, 1.2030, 1.2040, 1.2025, 1.2035),   # arm OCO (range 50 pips)
        _bar(8, 1, 1.2045, 1.2060, 1.2040, 1.2055),   # buy-stop entry @1.2053
        _bar(8, 2, 1.2050, 1.2052, 1.1995, 1.2000),   # low breaks stop 1.2000
        _bar(16, 0, 1.2000, 1.2005, 1.1995, 1.2000),  # 16:00 flatten window
    ]


def _run(atr):
    eng = OrderEngine()
    strat = LondonBreakoutStrategy(
        pair="EURUSD", atr_d1={_DAY: atr}, releases=pd.DataFrame())
    opens = []
    real_open = eng.open_position

    def counting_open(*args, **kwargs):
        opens.append((args, kwargs))
        return real_open(*args, **kwargs)

    eng.open_position = counting_open
    eng.run(_day_bars(), strat)
    return eng, opens


def test_exit_fill_does_not_spawn_phantom_entry():
    eng, opens = _run(atr=0.0100)  # width 0.0050 in [0.25*atr, 0.80*atr]

    entry_fills = [f for f in eng.fills if f.order_id != EXIT_ORDER_ID]
    exit_fills = [f for f in eng.fills if f.order_id == EXIT_ORDER_ID]

    assert len(entry_fills) == 1  # exactly one real breakout entry
    assert entry_fills[0].side == "buy"
    assert len(opens) == 1        # engine opened exactly one position (no phantom)
    assert len(exit_fills) == 1   # only the real stop-out, no phantom exit
    assert eng.position is None
