import datetime as dt

import numpy as np
import pandas as pd

from src.backtesting.costs.fx import _pip_size, fx_round_trip_bps_at
from src.backtesting.engine.intraday_order_engine import OrderEngine, Bar
from src.strategies.advanced.fx_london_breakout import LondonBreakoutStrategy

_DAY = dt.date(2024, 1, 10)
_PIP = _pip_size("GBPUSD")
_OFFSET_PIPS = 3.0


def _run(df, **kwargs):
    strat = LondonBreakoutStrategy("GBPUSD", **kwargs)
    eng = OrderEngine()
    eng.run([Bar(ts.to_pydatetime(), r.open, r.high, r.low, r.close)
             for ts, r in df.iterrows()], strat)
    return strat, eng


def _blank_day(day, lo, hi, end="16:00"):
    idx = pd.date_range(f"{day} 00:00", f"{day} {end}", freq="1min", tz="UTC")
    mid = (hi + lo) / 2.0
    df = pd.DataFrame({"open": mid, "high": mid, "low": mid, "close": mid}, index=idx)
    asian = df.index.hour < 7
    df.loc[asian, "high"] = hi
    df.loc[asian, "low"] = lo
    return df


def _rt_spread_r(initial_risk, entry_price):
    """Measured hour-of-week charge. Entry and exit both land in hour 8 UTC."""
    how = _DAY.weekday() * 24 + 8
    return (fx_round_trip_bps_at("GBPUSD", how) / 1e4 * entry_price) / initial_risk


def _long_stop_loss_day_r(width_pips):
    lo = 1.2500
    hi = lo + width_pips * _PIP
    df = _blank_day("2024-01-10", lo, hi)
    brk = (df.index.hour == 8) & (df.index.minute == 15)
    df.loc[brk, ["high", "close"]] = hi + 5 * _PIP
    stp = (df.index.hour == 8) & (df.index.minute == 20)
    df.loc[stp, ["open", "high", "low", "close"]] = lo - 1 * _PIP
    atr = 2 * width_pips * _PIP
    strat, _ = _run(df, atr_d1={_DAY: atr})
    initial_risk = width_pips * _PIP + _OFFSET_PIPS * _PIP
    entry_price = hi + _OFFSET_PIPS * _PIP
    return strat.day_r[_DAY], _rt_spread_r(initial_risk, entry_price)


def test_full_stop_loss_books_minus_one_r_qty_and_pip_independent():
    r30, rt30 = _long_stop_loss_day_r(30)
    r60, rt60 = _long_stop_loss_day_r(60)
    assert abs(r30 - (-1.0 - rt30)) < 1e-9
    assert abs(r60 - (-1.0 - rt60)) < 1e-9
    # The qty/pip-independent R core is exactly -1.0 for both widths.
    assert abs((r30 + rt30) - (r60 + rt60)) < 1e-12
    assert abs((r30 + rt30) - (-1.0)) < 1e-9


def test_short_full_stop_loss_books_minus_one_r():
    width_pips = 40
    lo = 1.2500
    hi = lo + width_pips * _PIP
    df = _blank_day("2024-01-10", lo, hi)
    brk = (df.index.hour == 8) & (df.index.minute == 15)
    df.loc[brk, ["low", "close"]] = lo - 5 * _PIP
    stp = (df.index.hour == 8) & (df.index.minute == 20)
    df.loc[stp, ["open", "high", "low", "close"]] = hi + 1 * _PIP
    strat, _ = _run(df, atr_d1={_DAY: 2 * width_pips * _PIP})
    initial_risk = width_pips * _PIP + _OFFSET_PIPS * _PIP
    entry_price = lo - _OFFSET_PIPS * _PIP
    assert abs(strat.day_r[_DAY] - (-1.0 - _rt_spread_r(initial_risk, entry_price))) < 1e-9


def test_target_then_trail_win_books_positive_r():
    width_pips = 30
    lo = 1.2500
    hi = lo + width_pips * _PIP
    df = _blank_day("2024-01-10", lo, hi)
    brk = (df.index.hour == 8) & (df.index.minute == 15)
    df.loc[brk, ["high", "close"]] = hi + 5 * _PIP
    run_up = (df.index.hour == 8) & (df.index.minute >= 16)
    high_level = hi + width_pips * _PIP + 10 * _PIP
    df.loc[run_up, ["open", "high", "low", "close"]] = high_level
    after = df.index.hour > 8
    df.loc[after, ["open", "high", "low", "close"]] = high_level
    strat, _ = _run(df, atr_d1={_DAY: 2 * width_pips * _PIP})
    assert strat.day_r[_DAY] > 0.0


def test_position_does_not_leak_across_day_and_books_under_entry_day():
    lo, hi = 1.2500, 1.2530
    day1 = _blank_day("2024-01-10", lo, hi, end="14:00")  # stops before 16:00
    brk = (day1.index.hour == 8) & (day1.index.minute == 15)
    day1.loc[brk, ["high", "close"]] = hi + 5 * _PIP
    day2 = _blank_day("2024-01-11", 1.2515, 1.2515)  # flat, no range/break
    df = pd.concat([day1, day2])
    strat, eng = _run(df, atr_d1={_DAY: 0.0050, dt.date(2024, 1, 11): 0.0050})
    assert _DAY in strat.day_r                       # booked under entry day
    assert dt.date(2024, 1, 11) not in strat.day_r   # not the close/next day
    assert strat._pos is None                        # no leak into next day
    assert eng.position is None


def test_cancel_fires_when_0930_bar_is_missing():
    lo, hi = 1.2500, 1.2530
    df = _blank_day("2024-01-10", lo, hi)
    missing = (df.index.hour == 9) & (df.index.minute == 30)
    df = df[~missing]                                # 09:29 -> 09:31 jump
    late = (df.index.hour == 9) & (df.index.minute == 40)
    df.loc[late, ["high", "close"]] = hi + 5 * _PIP  # break AFTER 09:30
    strat, eng = _run(df, atr_d1={_DAY: 0.0050})
    assert eng.fills == []                           # cancelled at 09:31, no fill


def test_wide_bar_spanning_both_oco_legs_fills_only_one_and_opens():
    # A single bar wide enough to trigger BOTH OCO legs must fill only ONE
    # (engine-level OCO atomicity): exactly one fill, one position opened.
    lo, hi = 1.2500, 1.2530
    df = _blank_day("2024-01-10", lo, hi, end="08:30")  # end before 16:00 flatten
    wide = (df.index.hour == 8) & (df.index.minute == 15)
    df.loc[wide, "high"] = hi + 5 * _PIP
    df.loc[wide, "low"] = lo - 5 * _PIP
    df.loc[wide, "open"] = (hi + lo) / 2.0
    df.loc[wide, "close"] = (hi + lo) / 2.0 + 2 * _PIP
    strat, eng = _run(df, atr_d1={_DAY: 0.0050})
    assert len(eng.fills) == 1  # atomic OCO: never both legs in one bar
    assert strat._pos is not None
    assert strat._pos.side == eng.fills[0].side


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
