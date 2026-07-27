"""Reusable day-loop harness for intraday strategies.

The minute-bar order engine was already reusable; the walk-forward plumbing
around it was not. It lived inside the LondonBreakout runner, hard-wired to that
one strategy, while 21 open catalog slots need exactly the same loop: group 1m
bars into FX trading days, build a FRESH strategy and engine per day, run the
bars, close out at the end of the day, and accumulate the qty-independent
R-multiple booked per entry day.

Two things are fixed here relative to the code this was extracted from:

- **Fills are collectable.** The London walk-forward path discarded `eng.fills`
  outright and kept a separate "artifact backfill only" trade log that the gate
  never touched. Every simulated run must persist its fills, so a sink is
  available on the main path.
- **End-of-day close-out is the strategy's business.** The runner previously
  reached into `strategy._book_if_closed(...)`, a private method, and performed
  the flatten itself. A strategy now declares `finalize_day`, so how a position
  is closed at the day boundary belongs to the strategy that opened it.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from src.backtesting.engine.intraday_order_engine import Bar, OrderEngine


class IntradayStrategy(Protocol):
    """What the harness requires of a strategy it drives."""

    day_r: Dict[dt.date, float]

    def on_bar(self, bar: Bar, engine: OrderEngine) -> None: ...

    def finalize_day(self, engine: OrderEngine, bar: Bar) -> None:
        """Close out and book anything still open at the day boundary."""


def fx_day_values(utc_index: pd.DatetimeIndex) -> np.ndarray:
    """FX trading day (17:00-ET boundary) for a tz-aware UTC index, DST-safe.

    Mirrors ``fx_clock.fx_trading_day`` (shift NY wall time by +7h so 17:00 ET
    becomes local midnight, then take the calendar date) but operates on
    tz-NAIVE wall times, so the shift never re-localizes into a spring-forward
    gap. ``fx_clock``'s tz-aware ``DateOffset(hours=7)`` raises
    NonExistentTimeError on 1m data crossing the DST gap.
    """
    ny_naive = utc_index.tz_convert("America/New_York").tz_localize(None)
    return np.asarray(pd.DatetimeIndex(ny_naive + pd.Timedelta(hours=7)).date)


def bars_for_day(sub: pd.DataFrame) -> List[Bar]:
    ts = sub.index.to_pydatetime()
    o = sub["open"].to_numpy(dtype=float)
    h = sub["high"].to_numpy(dtype=float)
    lo = sub["low"].to_numpy(dtype=float)
    c = sub["close"].to_numpy(dtype=float)
    return [Bar(ts[i], o[i], h[i], lo[i], c[i]) for i in range(len(sub))]


def _fill_rows(fills, pair: str, day: dt.date) -> List[Dict[str, Any]]:
    return [{"pair": pair, "date": day, "ts": f.ts, "side": f.side,
             "price": f.price, "qty": f.qty, "order_id": f.order_id,
             "reason": f.reason, "trade_id": f.trade_id, "entry_ts": f.entry_ts,
             "entry_price": f.entry_price, "mae": f.mae, "mfe": f.mfe,
             "bars_held": f.bars_held} for f in fills]


def pair_daily_returns(bars_1m: pd.DataFrame,
                       make_strategy: Callable[[], IntradayStrategy],
                       risk_frac: float, *, pair: str = "",
                       collect_fills: Optional[List[Dict[str, Any]]] = None) -> pd.Series:
    """Daily return series for one pair, one strategy, over 1m bars.

    `make_strategy` is called once per FX trading day and must return a fresh
    instance: carrying strategy state across the day boundary is how an
    intraday backtest quietly acquires a lookahead.

    Days on which nothing traded are emitted as 0.0 rather than dropped. A flat
    day is a real observation, and omitting it would shrink the denominator and
    inflate the Sharpe.
    """
    if bars_1m is None or bars_1m.empty:
        return pd.Series(dtype=float)

    day_r: Dict[dt.date, float] = {}
    trading_days: List[dt.date] = []
    for day, sub in bars_1m.groupby(fx_day_values(bars_1m.index)):
        trading_days.append(day)
        strategy = make_strategy()
        engine = OrderEngine()
        day_bars = bars_for_day(sub)
        engine.run(day_bars, strategy)
        if day_bars:
            strategy.finalize_day(engine, day_bars[-1])
        if collect_fills is not None and engine.fills:
            collect_fills.extend(_fill_rows(engine.fills, pair, day))
        for booked_day, r in strategy.day_r.items():
            day_r[booked_day] = day_r.get(booked_day, 0.0) + r

    index = sorted(set(trading_days) | set(day_r.keys()))
    return pd.Series({d: risk_frac * day_r.get(d, 0.0) for d in index})


def portfolio_daily_returns(per_pair: Dict[str, pd.Series]) -> pd.Series:
    """Equal-weight combination of per-pair daily returns.

    Union of dates; a pair absent on a date contributes 0.0 for that date, which
    is what "not trading" means. Dividing by the FULL pair count rather than the
    count present on each date keeps the book's risk budget constant instead of
    silently levering up on days when only one pair is active.
    """
    live = {p: s for p, s in per_pair.items() if s is not None and not s.empty}
    if not live:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(live).sort_index()
    return frame.reindex(sorted(frame.index)).fillna(0.0).sum(axis=1) / len(live)
