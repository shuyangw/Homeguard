import datetime as dt

import numpy as np
import pandas as pd

from src.backtesting.engine.intraday_order_engine import OrderEngine, Bar
from src.strategies.advanced.fx_london_breakout import (
    LondonBreakoutStrategy, asian_range)


def _london_day_1m(day, hi, lo):
    # Build a 1m frame for one UTC day (Jan -> London==UTC): Asian 00:00-07:00
    # ranges [lo,hi]; then an 08:00-09:30 window that breaks above hi.
    idx = pd.date_range(f"{day} 00:00", f"{day} 16:00", freq="1min", tz="UTC")
    close = np.full(len(idx), (hi + lo) / 2.0)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
    asian = (df.index.hour < 7)
    df.loc[asian, "high"] = hi
    df.loc[asian, "low"] = lo
    return df


def test_asian_range_reads_0000_0700_london():
    df = _london_day_1m("2024-01-10", 1.2550, 1.2500)
    hi, lo = asian_range(df, dt.date(2024, 1, 10))
    assert abs(hi - 1.2550) < 1e-9 and abs(lo - 1.2500) < 1e-9


def test_width_filter_stands_down_when_too_wide():
    df = _london_day_1m("2024-01-10", 1.2600, 1.2500)  # width 100 pips
    strat = LondonBreakoutStrategy("GBPUSD", atr_d1={dt.date(2024, 1, 10): 0.0050})  # 0.8*ATR=40pips
    eng = OrderEngine()
    eng.run([Bar(ts.to_pydatetime(), r.open, r.high, r.low, r.close)
             for ts, r in df.iterrows()], strat)
    assert eng.fills == []  # width 100 > 0.8*50 -> no orders placed


def test_event_skip_day_places_no_orders():
    df = _london_day_1m("2024-01-10", 1.2530, 1.2500)  # width 30 pips, in-band vs ATR 50
    rel = pd.DataFrame([{"date": dt.date(2024, 1, 10), "name": "EZ", "currency": "EUR",
                         "release_utc": pd.Timestamp("2024-01-10 10:00", tz="UTC")}])
    strat = LondonBreakoutStrategy("GBPUSD", atr_d1={dt.date(2024, 1, 10): 0.0050}, releases=rel)
    eng = OrderEngine()
    eng.run([Bar(ts.to_pydatetime(), r.open, r.high, r.low, r.close)
             for ts, r in df.iterrows()], strat)
    assert eng.fills == []  # tier-1 release in 09:30-12:01 -> skip


def test_clean_upside_break_fills_buy_stop():
    df = _london_day_1m("2024-01-10", 1.2530, 1.2500)
    # inject a break above hi+3pip at 08:15
    brk = (df.index.hour == 8) & (df.index.minute == 15)
    df.loc[brk, ["high", "close"]] = 1.2540
    strat = LondonBreakoutStrategy("GBPUSD", atr_d1={dt.date(2024, 1, 10): 0.0050})
    eng = OrderEngine()
    eng.run([Bar(ts.to_pydatetime(), r.open, r.high, r.low, r.close)
             for ts, r in df.iterrows()], strat)
    assert any(f.side == "buy" for f in eng.fills)
