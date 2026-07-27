"""General minute-bar order engine (no-lookahead).

Order book supporting stop/limit/OCO/bracket orders, partial fills, trailing
stops, and time-based controls. Fills are conservative: a triggered stop fills at
the worse of trigger and bar open (gap-through), a limit at the better of limit
and open. An order armed while reacting to bar_t is first eligible on bar_{t+1}.
Instrument-agnostic; a strategy drives it via on_bar callbacks (see run()).
"""
from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field
from typing import NamedTuple, Optional


class Bar(NamedTuple):
    ts: dt.datetime
    open: float
    high: float
    low: float
    close: float


class Fill(NamedTuple):
    order_id: int
    ts: dt.datetime
    price: float
    qty: float
    side: str
    # Exit-side diagnostics (methodology Section 11.9). Defaulted so the
    # five-field positional construction used for entries keeps working.
    # Entries carry reason="" and NaN excursions: zero would be a claim about
    # the excursion, NaN says the field does not apply.
    reason: str = ""
    trade_id: int = -1
    entry_ts: Optional[dt.datetime] = None
    entry_price: float = float("nan")
    mae: float = float("nan")
    mfe: float = float("nan")
    bars_held: int = -1


EXIT_ORDER_ID = -1  # engine-managed position exit (not a resting order)

# Trade ids must be unique across ENGINES, not just within one. Callers build a
# fresh OrderEngine per FX trading day, so an engine-local counter gave every
# day's first position the same id -- a real run produced 2 distinct values
# across 14838 fill rows, making the field useless for round-trip linkage.
_TRADE_ID_SEQ = itertools.count(1)


@dataclass
class Order:
    side: str            # "buy" | "sell"
    kind: str            # "stop" | "limit"
    trigger: float
    qty: float
    order_id: int = -1
    oco_group: Optional[int] = None
    armed_at: Optional[dt.datetime] = None
    filled: bool = False
    fill_price: Optional[float] = None
    fill_ts: Optional[dt.datetime] = None


class OrderEngine:
    def __init__(self):
        self.resting_orders: list[Order] = []
        self.fills: list[Fill] = []
        self._next_id = 0

    def add_order(self, order: Order, armed_at: Optional[dt.datetime] = None) -> int:
        order.order_id = self._next_id
        order.armed_at = armed_at
        self._next_id += 1
        self.resting_orders.append(order)
        return order.order_id

    def cancel(self, order_id: int) -> None:
        self.resting_orders = [o for o in self.resting_orders if o.order_id != order_id]

    def _match_order(self, o: Order, bar: Bar) -> Optional[float]:
        if o.side == "buy" and o.kind == "stop":
            return max(o.trigger, bar.open) if bar.high >= o.trigger else None
        if o.side == "sell" and o.kind == "stop":
            return min(o.trigger, bar.open) if bar.low <= o.trigger else None
        if o.side == "buy" and o.kind == "limit":
            return min(o.trigger, bar.open) if bar.low <= o.trigger else None
        if o.side == "sell" and o.kind == "limit":
            return max(o.trigger, bar.open) if bar.high >= o.trigger else None
        return None

    def match_resting_orders(self, bar: Bar) -> list[Fill]:
        new_fills: list[Fill] = []
        cancelled_groups: set[int] = set()
        for o in list(self.resting_orders):
            if o.armed_at is not None and bar.ts <= o.armed_at:
                continue  # armed this bar or later; not yet eligible
            if o.oco_group is not None and o.oco_group in cancelled_groups:
                continue  # sibling already filled this bar; OCO is atomic
            price = self._match_order(o, bar)
            if price is None:
                continue
            fill = Fill(o.order_id, bar.ts, price, o.qty, o.side)
            new_fills.append(fill)
            self.fills.append(fill)
            self.cancel(o.order_id)
            if o.oco_group is not None:
                cancelled_groups.add(o.oco_group)
        for grp in cancelled_groups:
            self.resting_orders = [o for o in self.resting_orders if o.oco_group != grp]
        return new_fills


@dataclass
class Position:
    side: str
    qty: float
    entry_price: float
    entry_ts: dt.datetime
    stop: float
    target: float
    tp_fraction: float
    trail_dist: Optional[float] = None
    took_partial: bool = False
    extreme: Optional[float] = None  # high-water (buy) / low-water (sell) for trailing
    realized_pips: float = 0.0
    trade_id: int = -1
    # Excursions in price units, signed so favourable is POSITIVE for both
    # sides. Seeded at 0.0: at entry the position has been neither ahead nor
    # behind, and a NaN seed would poison the running max/min.
    mfe: float = 0.0
    mae: float = 0.0
    bars_held: int = 0


def _add_oco(self, a: "Order", b: "Order") -> int:  # noqa: E301  (attached below)
    grp = self._next_id
    self._next_id += 1
    a.oco_group = grp
    b.oco_group = grp
    self.add_order(a)
    self.add_order(b)
    return grp


def _open_position(self, side, qty, entry_price, entry_ts, stop, target,
                   tp_fraction, trail_dist):
    self.position = Position(side=side, qty=qty, entry_price=entry_price,
                             entry_ts=entry_ts, stop=stop, target=target,
                             tp_fraction=tp_fraction, trail_dist=trail_dist,
                             extreme=entry_price, trade_id=next(_TRADE_ID_SEQ))
    return self.position


def _signed(side: str) -> float:
    return 1.0 if side == "buy" else -1.0


def _update_position(self, bar: "Bar") -> list:
    p = self.position
    if p is None:
        return []
    _track_excursion(p, bar)
    events: list = []
    sign = _signed(p.side)
    exit_side = "sell" if p.side == "buy" else "buy"
    hit_stop = (bar.low <= p.stop) if p.side == "buy" else (bar.high >= p.stop)
    hit_target = (bar.high >= p.target) if p.side == "buy" else (bar.low <= p.target)
    # Both-in-one-bar: adverse (stop) resolves first.
    if hit_stop:
        reason = "trail" if (p.took_partial and p.trail_dist is not None) else "stop"
        events.append((reason, p.stop, p.qty))
        self._record_exit_fill(EXIT_ORDER_ID, bar.ts, p.stop, p.qty, exit_side, reason)
        p.realized_pips += sign * (p.stop - p.entry_price) * p.qty
        self.position = None
        return events
    if hit_target and not p.took_partial:
        take = p.qty * p.tp_fraction
        events.append(("target", p.target, take))
        self._record_exit_fill(EXIT_ORDER_ID, bar.ts, p.target, take, exit_side, "target")
        p.realized_pips += sign * (p.target - p.entry_price) * take
        p.qty -= take
        if p.qty <= 1e-12:
            self.position = None
            return events
        p.took_partial = True
        p.extreme = bar.high if p.side == "buy" else bar.low
        if p.trail_dist is not None:
            if p.side == "buy":
                p.stop = max(p.stop, p.extreme - p.trail_dist)
            else:
                p.stop = min(p.stop, p.extreme + p.trail_dist)
            breached = (bar.low <= p.stop) if p.side == "buy" else (bar.high >= p.stop)
            if breached:
                events.append(("trail", p.stop, p.qty))
                self._record_exit_fill(EXIT_ORDER_ID, bar.ts, p.stop, p.qty, exit_side, "trail")
                p.realized_pips += sign * (p.stop - p.entry_price) * p.qty
                self.position = None
                return events
    # Ratchet trailing stop on the remainder.
    if p.took_partial and p.trail_dist is not None:
        if p.side == "buy":
            p.extreme = max(p.extreme, bar.high)
            p.stop = max(p.stop, p.extreme - p.trail_dist)
        else:
            p.extreme = min(p.extreme, bar.low)
            p.stop = min(p.stop, p.extreme + p.trail_dist)
    return events


def _flatten(self, price: float, ts: dt.datetime, reason: str = "flat") -> list:
    p = self.position
    if p is None:
        return []
    sign = _signed(p.side)
    exit_side = "sell" if p.side == "buy" else "buy"
    self._record_exit_fill(EXIT_ORDER_ID, ts, price, p.qty, exit_side, reason)
    p.realized_pips += sign * (price - p.entry_price) * p.qty
    self.position = None
    return [(reason, price, p.qty)]


def _record_exit_fill(self, order_id, ts, price, qty, side, reason="") -> None:
    p = self.position
    self.fills.append(Fill(
        order_id, ts, price, qty, side, reason=reason,
        trade_id=(p.trade_id if p else -1),
        entry_ts=(p.entry_ts if p else None),
        entry_price=(p.entry_price if p else float("nan")),
        mae=(p.mae if p else float("nan")),
        mfe=(p.mfe if p else float("nan")),
        bars_held=(p.bars_held if p else -1)))


def _track_excursion(p: "Position", bar: "Bar") -> None:
    """Update MAE/MFE, signed so favourable is POSITIVE for both sides."""
    sign = _signed(p.side)
    p.bars_held += 1
    best = sign * ((bar.high if p.side == "buy" else bar.low) - p.entry_price)
    worst = sign * ((bar.low if p.side == "buy" else bar.high) - p.entry_price)
    p.mfe = max(p.mfe, best)
    p.mae = min(p.mae, worst)


def _cancel_all_resting(self) -> None:
    self.resting_orders = []


def _run(self, bars, strategy) -> None:
    for bar in bars:
        if self.position is not None:
            self.update_position(bar)
        self.match_resting_orders(bar)
        strategy.on_bar(bar, self)


OrderEngine.position = None
OrderEngine.add_oco = _add_oco
OrderEngine.open_position = _open_position
OrderEngine.update_position = _update_position
OrderEngine.flatten = _flatten
OrderEngine._record_exit_fill = _record_exit_fill
OrderEngine.cancel_all_resting = _cancel_all_resting
OrderEngine.run = _run
