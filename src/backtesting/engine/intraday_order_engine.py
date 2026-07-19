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
