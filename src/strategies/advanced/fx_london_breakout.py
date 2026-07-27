"""#20 London Open Breakout (Asian range break), intraday.

Per pair per FX day: define the Asian range (00:00-07:00 London), filter by
range width vs ATR(14,D1), skip tier-1 EUR/GBP release days, place an OCO
buy-stop/sell-stop bracket around the range in the 08:00-09:30 London window,
cancel unfilled at 09:30, manage a bracket exit (take half at 1x range, trail
the rest at 1x ATR(15m)), and flatten at 16:00 London. Drives the general
intraday order engine; fixed-fractional risk sizing.

P&L is reported as a qty-independent R-multiple series ``day_r`` (dict
date -> float), booked against the day the trade was ENTERED. One unit of R is
the initial per-trade risk ``abs(entry_fill_price - initial_stop_price)``, so a
full-stop loss books ``-1.0 - rt_spread_R`` regardless of the stop's pip width.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src.backtesting.costs.fx import (_pip_size, fx_round_trip_bps_at,
                                      fx_round_trip_pips)
from src.backtesting.engine.intraday_order_engine import EXIT_ORDER_ID, Order
from src.backtesting.sessions.fx_clock import EXCHANGE_TZ
from src.backtesting.utils.indicators import Indicators
from src.data.macro_calendar_tier1 import tier1_release_in_window

_LONDON = EXCHANGE_TZ["LONDON"]


class LondonBreakoutStrategy:
    def __init__(self, pair, atr_d1, risk_frac=0.005, tp_fraction=0.5,
                 offset_pips=3.0, tier="major", releases=None,
                 override_pips=None, cost_mult=1.0):
        self.pair = pair
        self.atr_d1 = atr_d1                # dict[date -> ATR(14) daily, price units]
        self.risk_frac = float(risk_frac)
        self.tp_fraction = float(tp_fraction)
        self.pip = _pip_size(pair)
        self.offset = offset_pips * self.pip
        self.tier = tier
        self.releases = releases
        self.override_pips = override_pips
        self.cost_mult = float(cost_mult)
        self.day_r: dict[dt.date, float] = {}
        # Default: MEASURED hour-of-week spread, charged at the hours the fills
        # actually land on. override_pips keeps the legacy flat tier charge as an
        # explicit escape hatch so a flat-cost leg stays reproducible.
        self._flat_rt_pips = (
            fx_round_trip_pips(tier, session="london", override_pips=override_pips)
            if override_pips is not None else None)
        self._entry_how = None
        self._day = None
        self._skip = False
        self._armed = False
        self._cancelled = False
        self._range = None
        self._asian_hi = None
        self._asian_lo = None
        self._trail_dist = None
        self._pos = None
        self._entry_day = None
        self._entry_qty = 0.0
        self._initial_risk = None
        self._fill_cursor = 0
        self._bars_ts: list = []
        self._bars_o: list = []
        self._bars_h: list = []
        self._bars_l: list = []
        self._bars_c: list = []

    def _local(self, ts):
        return pd.Timestamp(ts).tz_convert(_LONDON)

    @staticmethod
    def _hour_of_week(ts) -> int:
        """Monday 00:00 UTC = 0, matching the measured spread surface."""
        t = pd.Timestamp(ts).tz_convert("UTC")
        return int(t.dayofweek * 24 + t.hour)

    def _round_trip_r(self, entry_price, exit_ts) -> float:
        """Round-trip cost in R. Half the spread is paid at each fill's own hour."""
        if self._flat_rt_pips is not None:
            rt_price = self._flat_rt_pips * self.pip
        else:
            bps = 0.5 * (fx_round_trip_bps_at(self.pair, self._entry_how)
                         + fx_round_trip_bps_at(self.pair, self._hour_of_week(exit_ts)))
            rt_price = bps / 1e4 * entry_price
        return rt_price * self.cost_mult / self._initial_risk

    def on_bar(self, bar, engine):
        lt = self._local(bar.ts)
        day = lt.date()
        if day != self._day:
            self._start_day(day, bar, engine)
        self._accumulate(bar, lt)
        self._book_if_closed(engine, bar)
        self._maybe_open(bar, engine)
        if self._skip:
            return
        self._maybe_arm(bar, lt, engine)
        self._maybe_cancel(lt, engine)
        if lt.hour >= 16 and engine.position is not None:
            engine.flatten(bar.close, bar.ts, reason="flat_1600")
            self._book_if_closed(engine, bar)

    def _start_day(self, day, bar, engine):
        if self._pos is not None:
            if engine.position is self._pos:
                engine.flatten(bar.open, bar.ts, reason="stale_close")
            self._book(bar)
        self._day = day
        self._armed = False
        self._cancelled = False
        self._range = None
        self._asian_hi = None
        self._asian_lo = None
        self._trail_dist = None
        self._bars_ts = []
        self._bars_o = []
        self._bars_h = []
        self._bars_l = []
        self._bars_c = []
        self._skip = self._skip_day(day)

    def _skip_day(self, day) -> bool:
        if self.atr_d1.get(day) is None:
            return True
        return tier1_release_in_window(day, dt.time(9, 30), dt.time(12, 1),
                                       exchange="LONDON",
                                       currencies=("EUR", "GBP"),
                                       releases=self.releases)

    def _accumulate(self, bar, lt):
        self._bars_ts.append(pd.Timestamp(bar.ts))
        self._bars_o.append(bar.open)
        self._bars_h.append(bar.high)
        self._bars_l.append(bar.low)
        self._bars_c.append(bar.close)
        if lt.hour < 7:
            self._asian_hi = bar.high if self._asian_hi is None else max(self._asian_hi, bar.high)
            self._asian_lo = bar.low if self._asian_lo is None else min(self._asian_lo, bar.low)

    def _maybe_arm(self, bar, lt, engine):
        if self._armed:
            return
        if lt.hour < 8:
            return
        self._armed = True
        if lt.hour > 9 or (lt.hour == 9 and lt.minute >= 30):
            return
        if self._asian_hi is None or self._asian_lo is None:
            return
        hi, lo = self._asian_hi, self._asian_lo
        width = hi - lo
        atr = self.atr_d1.get(self._day)
        if atr is None or not (0.25 * atr <= width <= 0.80 * atr):
            return
        self._range = (hi, lo)
        self._trail_dist = self._compute_trail()
        self._arm_oco(hi, lo, engine)

    def _maybe_cancel(self, lt, engine):
        if self._cancelled:
            return
        if lt.hour > 9 or (lt.hour == 9 and lt.minute >= 30):
            engine.cancel_all_resting()
            self._cancelled = True

    def _arm_oco(self, hi, lo, engine):
        buy_trigger = hi + self.offset
        sell_trigger = lo - self.offset
        buy_dist = buy_trigger - lo
        sell_dist = hi - sell_trigger
        buy = Order(side="buy", kind="stop", trigger=buy_trigger,
                    qty=self.risk_frac / buy_dist)
        sell = Order(side="sell", kind="stop", trigger=sell_trigger,
                     qty=self.risk_frac / sell_dist)
        engine.add_oco(buy, sell)

    def _compute_trail(self):
        if len(self._bars_ts) < 2:
            return None
        df = pd.DataFrame(
            {"open": self._bars_o, "high": self._bars_h,
             "low": self._bars_l, "close": self._bars_c},
            index=pd.DatetimeIndex(self._bars_ts))
        bars15 = df.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        if len(bars15) < 2:
            return None
        atr = Indicators.atr(bars15["high"], bars15["low"], bars15["close"], 14)
        val = atr.iloc[-1]
        return None if pd.isna(val) else float(val)

    def _pick_fill(self, new_fills, bar):
        if len(new_fills) == 1:
            return new_fills[0]
        want = "buy" if bar.close >= bar.open else "sell"
        for fill in new_fills:
            if fill.side == want:
                return fill
        return new_fills[-1]

    def _maybe_open(self, bar, engine):
        if len(engine.fills) <= self._fill_cursor:
            return
        new_fills = engine.fills[self._fill_cursor:]
        self._fill_cursor = len(engine.fills)
        entry_fills = [f for f in new_fills if f.order_id != EXIT_ORDER_ID]
        if not entry_fills:
            return
        if engine.position is not None or self._pos is not None:
            return
        if not isinstance(self._range, tuple):
            return
        fill = self._pick_fill(entry_fills, bar)
        hi, lo = self._range
        width = hi - lo
        if fill.side == "buy":
            stop, target = lo, fill.price + width
        else:
            stop, target = hi, fill.price - width
        self._entry_qty = fill.qty
        self._initial_risk = abs(fill.price - stop)
        self._entry_day = self._day
        self._entry_how = self._hour_of_week(bar.ts)
        self._pos = engine.open_position(
            fill.side, fill.qty, fill.price, bar.ts, stop, target,
            self.tp_fraction, self._trail_dist)

    def finalize_day(self, engine, bar):
        """Close out anything still open at the FX day boundary, and book it.

        The runner used to do this itself and reach into `_book_if_closed`. How
        a position is closed at the boundary is the strategy's decision, so it
        lives here and the day-loop harness just calls it.
        """
        if engine.position is not None:
            engine.flatten(bar.close, bar.ts, reason="eod_safety")
            self._book_if_closed(engine, bar)

    def _book_if_closed(self, engine, bar):
        if self._pos is None or engine.position is self._pos:
            return
        self._book(bar)

    def _book(self, bar):
        pos = self._pos
        exit_r = pos.realized_pips / (self._entry_qty * self._initial_risk)
        rt_spread_r = self._round_trip_r(pos.entry_price, bar.ts)
        self.day_r[self._entry_day] = (
            self.day_r.get(self._entry_day, 0.0) + exit_r - rt_spread_r)
        self._pos = None
