"""General minute-bar order engine (no-lookahead).

Order book supporting stop/limit/OCO/bracket orders, partial fills, trailing
stops, and time-based controls. Fills are conservative: a triggered stop fills at
the worse of trigger and bar open (gap-through), a limit at the better of limit
and open. An order armed while reacting to bar_t is first eligible on bar_{t+1}.
Instrument-agnostic; a strategy drives it via on_bar callbacks (see run()).
"""
from __future__ import annotations

import datetime as dt
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
                             extreme=entry_price)
    return self.position


def _signed(side: str) -> float:
    return 1.0 if side == "buy" else -1.0


def _update_position(self, bar: "Bar") -> list:
    p = self.position
    if p is None:
        return []
    events: list = []
    sign = _signed(p.side)
    hit_stop = (bar.low <= p.stop) if p.side == "buy" else (bar.high >= p.stop)
    hit_target = (bar.high >= p.target) if p.side == "buy" else (bar.low <= p.target)
    # Both-in-one-bar: adverse (stop) resolves first.
    if hit_stop:
        events.append(("stop" if not p.took_partial else "trail", p.stop, p.qty))
        p.realized_pips += sign * (p.stop - p.entry_price) * p.qty
        self.position = None
        return events
    if hit_target and not p.took_partial:
        take = p.qty * p.tp_fraction
        events.append(("target", p.target, take))
        p.realized_pips += sign * (p.target - p.entry_price) * take
        p.qty -= take
        p.took_partial = True
        p.extreme = bar.high if p.side == "buy" else bar.low
        if p.trail_dist is not None:
            p.stop = (p.extreme - p.trail_dist) if p.side == "buy" else (p.extreme + p.trail_dist)
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
    p.realized_pips += sign * (price - p.entry_price) * p.qty
    self.position = None
    return [(reason, price, p.qty)]


OrderEngine.position = None
OrderEngine.add_oco = _add_oco
OrderEngine.open_position = _open_position
OrderEngine.update_position = _update_position
OrderEngine.flatten = _flatten
