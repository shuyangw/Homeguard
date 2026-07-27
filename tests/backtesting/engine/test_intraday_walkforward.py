"""Reusable intraday day-loop harness.

Twenty-one open catalog slots need the intraday engine, and until now the only
walk-forward plumbing around it was hard-wired to LondonBreakout. This is that
plumbing, extracted: group 1m bars into FX trading days, build a FRESH strategy
and engine per day, run, close out, and accumulate the per-day R-multiple.
"""
import datetime as dt

import pandas as pd
import pytest

from src.backtesting.engine.intraday_walkforward import (bars_for_day,
                                                         fx_day_values,
                                                         pair_daily_returns)


def _minutes(day: str, n: int = 120, price: float = 1.25) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 08:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": price, "high": price, "low": price,
                         "close": price}, index=idx)


def _two_days() -> pd.DataFrame:
    return pd.concat([_minutes("2024-01-10"), _minutes("2024-01-11")])


class _BooksFixedR:
    """Books a fixed R on the first bar of each day it sees."""

    def __init__(self, r=2.0):
        self.r = r
        self.day_r = {}
        self.finalized = []
        self._done = False

    def on_bar(self, bar, engine):
        if not self._done:
            self.day_r[bar.ts.date()] = self.r
            self._done = True

    def finalize_day(self, engine, bar):
        self.finalized.append(bar.ts)


class _NeverTrades:
    def __init__(self):
        self.day_r = {}

    def on_bar(self, bar, engine):
        pass

    def finalize_day(self, engine, bar):
        pass


def test_fx_day_values_splits_on_the_1700_et_boundary():
    idx = pd.date_range("2024-01-10 20:00", periods=4, freq="1h", tz="UTC")
    days = fx_day_values(idx)
    # 22:00 UTC is 17:00 ET in January, so the FX day rolls there.
    assert len(set(days)) == 2


def test_bars_for_day_preserves_ohlc_and_order():
    df = _minutes("2024-01-10", n=3)
    bars = bars_for_day(df)
    assert len(bars) == 3
    assert bars[0].ts < bars[-1].ts
    assert bars[0].close == pytest.approx(1.25)


def test_returns_are_risk_scaled_r_per_day():
    s = pair_daily_returns(_two_days(), lambda: _BooksFixedR(2.0), risk_frac=0.005)
    assert len(s) == 2
    assert all(v == pytest.approx(0.01) for v in s)


def test_days_without_trades_appear_as_zero_not_missing():
    """A flat day is a real observation; dropping it would inflate the Sharpe."""
    s = pair_daily_returns(_two_days(), _NeverTrades, risk_frac=0.005)
    assert len(s) == 2
    assert all(v == 0.0 for v in s)


def test_a_fresh_strategy_is_built_per_day():
    built = []

    def factory():
        st = _BooksFixedR()
        built.append(st)
        return st

    pair_daily_returns(_two_days(), factory, risk_frac=0.005)
    assert len(built) == 2, "state must not leak across trading days"


def test_finalize_day_is_called_with_the_last_bar_of_each_day():
    seen = []

    def factory():
        st = _BooksFixedR()
        seen.append(st)
        return st

    pair_daily_returns(_two_days(), factory, risk_frac=0.005)
    assert all(len(st.finalized) == 1 for st in seen)
    for st in seen:
        assert st.finalized[0].hour == 9  # 08:00 + 119 minutes


def test_empty_input_returns_an_empty_series():
    s = pair_daily_returns(pd.DataFrame(), _NeverTrades, risk_frac=0.005)
    assert s.empty


def test_fills_are_collected_when_a_sink_is_given():
    """The London walk-forward discarded eng.fills entirely, against the
    fill-logging mandate. Collection is available here by construction."""
    class _Trades(_BooksFixedR):
        def on_bar(self, bar, engine):
            super().on_bar(bar, engine)
            if engine.position is None and bar.ts.minute == 5:
                engine.open_position("buy", 1.0, bar.close, bar.ts,
                                     bar.close - 0.01, bar.close + 0.01, 1.0, None)

        def finalize_day(self, engine, bar):
            super().finalize_day(engine, bar)
            if engine.position is not None:      # the exit is what emits a Fill
                engine.flatten(bar.close, bar.ts, reason="eod")

    sink = []
    pair_daily_returns(_two_days(), _Trades, risk_frac=0.005,
                       pair="EURUSD", collect_fills=sink)
    assert sink, "no fills captured"
    assert {"pair", "date"} <= set(sink[0])


def test_no_sink_means_no_collection_overhead():
    s = pair_daily_returns(_two_days(), lambda: _BooksFixedR(1.0), risk_frac=0.01)
    assert len(s) == 2


# --- equivalence with the bespoke loop this harness replaces -----------------

def _breakout_days() -> pd.DataFrame:
    """Two FX days on which #20 actually trades: Asian range then an 08:15 break."""
    from src.backtesting.costs.fx import _pip_size
    pip = _pip_size("GBPUSD")
    frames = []
    for day, lo in (("2024-01-10", 1.2500), ("2024-01-11", 1.2600)):
        hi = lo + 30 * pip
        idx = pd.date_range(f"{day} 00:00", f"{day} 16:00", freq="1min", tz="UTC")
        mid = (hi + lo) / 2.0
        df = pd.DataFrame({"open": mid, "high": mid, "low": mid, "close": mid},
                          index=idx)
        asian = df.index.hour < 7
        df.loc[asian, "high"] = hi
        df.loc[asian, "low"] = lo
        brk = (df.index.hour == 8) & (df.index.minute == 15)
        df.loc[brk, ["high", "close"]] = hi + 5 * pip
        frames.append(df)
    return pd.concat(frames)


def _bespoke_loop(bars, make_strategy, risk_frac):
    """The loop as it existed inside the LondonBreakout runner, verbatim in shape."""
    from src.backtesting.engine.intraday_order_engine import OrderEngine
    day_r, trading_days = {}, []
    for day, sub in bars.groupby(fx_day_values(bars.index)):
        trading_days.append(day)
        strat = make_strategy()
        eng = OrderEngine()
        day_bars = bars_for_day(sub)
        eng.run(day_bars, strat)
        if eng.position is not None and day_bars:
            last = day_bars[-1]
            eng.flatten(last.close, last.ts, reason="eod_safety")
            strat._book_if_closed(eng, last)
        for k, v in strat.day_r.items():
            day_r[k] = day_r.get(k, 0.0) + v
    idx = sorted(set(trading_days) | set(day_r.keys()))
    return pd.Series({d: risk_frac * day_r.get(d, 0.0) for d in idx})


def test_harness_reproduces_the_bespoke_loop_exactly():
    """The abstraction is only trustworthy if it is behaviour-preserving."""
    from src.strategies.advanced.fx_london_breakout import LondonBreakoutStrategy
    from src.backtesting.costs.fx import _pip_size

    bars = _breakout_days()
    atr = {dt.date(2024, 1, 10): 60 * _pip_size("GBPUSD"),
           dt.date(2024, 1, 11): 60 * _pip_size("GBPUSD")}
    make = lambda: LondonBreakoutStrategy("GBPUSD", atr_d1=atr)   # noqa: E731

    old = _bespoke_loop(bars, make, risk_frac=0.005)
    new = pair_daily_returns(bars, make, risk_frac=0.005, pair="GBPUSD")

    assert list(old.index) == list(new.index)
    pd.testing.assert_series_equal(old, new)
    assert old.abs().sum() > 0, "fixture must actually trade, or this proves nothing"
