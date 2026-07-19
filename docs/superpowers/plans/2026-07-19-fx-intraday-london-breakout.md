# Intraday Order Engine + #20 London Open Breakout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a general no-lookahead minute-bar order engine plus the #20 London Open Breakout strategy and its walk-forward S&P gate, producing the campaign's first gated intraday result.

**Architecture:** A minute-bar loader feeds an event-driven order engine (order book with stop/limit/OCO/bracket orders, partial fills, trailing stops, time controls; conservative worst-of-trigger/open fills; strictly causal loop). The #20 strategy drives the engine per pair per FX day; per-day P&L aggregates into a daily return series gated by the existing walk-forward + S&P benchmark.

**Tech Stack:** Python 3.13 (conda env `fintech`), pandas, polars, numpy, pytest. Reuses `fx_clock`, `macro_calendar_tier1`, the FX cost model, `walkforward_common`, and `benchmark.py`. No new dependencies.

## Global Constraints

- Run Python via the `fintech` conda env. Test prefix: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest <path> -v` (conda already initialized in the shell).
- ASCII-only, no em dashes, no emojis, no `print()`; use `from src.utils import logger` where logging is genuinely needed.
- CAUSALITY is the load-bearing property: an order created inside `on_bar(bar_t)` is first eligible to fill on `bar_{t+1}`. Never same-bar fill. Every engine test includes this discipline.
- Fill model (exact): buy-stop trigger T fills when `bar.high >= T` at `max(T, bar.open)`; sell-stop T fills when `bar.low <= T` at `min(T, bar.open)`; buy-limit L fills when `bar.low <= L` at `min(L, bar.open)`; sell-limit L fills when `bar.high >= L` at `max(L, bar.open)`. A bar whose range spans BOTH an open position's stop and its target resolves to the STOP (adverse).
- Spread/slippage: half the round-trip spread (from `fx_round_trip_pips(tier, session)` converted to price via `_pip_size(pair)`) applied adversely on entry and on exit.
- Timestamps are tz-aware UTC throughout; session times come from `fx_clock` (never hardcode UTC offsets).
- #20 event-skip uses `tier1_release_in_window(day, time(9,30), time(12,1), ...)` (win_end 12:01 to catch a BOE noon decision).
- Git hazard (macOS/Dropbox): use ONLY `git add <explicit paths>`, `git commit`, `git log`. NEVER `git checkout`, bare `git status`/`git diff`, or `git reset`. Commit ONLY each task's own files by explicit path (the working tree may hold unrelated uncommitted changes; never `git add -A`/`.`).

---

### Task 1: Intraday 1-minute loader

**Files:**
- Create: `src/backtesting/data/fx_intraday_loader.py`
- Test: `tests/backtesting/data/test_fx_intraday_loader.py`

**Interfaces:**
- Consumes: `get_local_storage_dir` from `src.settings`.
- Produces:
  - `load_fx_1min(pair: str, start: datetime.date, end: datetime.date) -> pd.DataFrame` (tz-aware UTC DatetimeIndex named `timestamp`, columns open/high/low/close/volume, float; sorted, deduped).
  - `resample_ohlc(bars: pd.DataFrame, freq: str) -> pd.DataFrame` (right-closed/right-labeled OHLC resample, dropping empty buckets).

- [ ] **Step 1: Write the failing tests**

Create `tests/backtesting/data/test_fx_intraday_loader.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd

from src.backtesting.data.fx_intraday_loader import load_fx_1min, resample_ohlc


def test_resample_ohlc_15min_aggregates_correctly():
    idx = pd.date_range("2024-01-02 08:00", periods=30, freq="1min", tz="UTC")
    bars = pd.DataFrame({
        "open": np.arange(30, dtype=float), "high": np.arange(30, dtype=float) + 1.0,
        "low": np.arange(30, dtype=float) - 1.0, "close": np.arange(30, dtype=float) + 0.5,
        "volume": np.ones(30)}, index=idx)
    out = resample_ohlc(bars, "15min")
    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == 0.0 and first["high"] == 14.0 + 1.0
    assert first["low"] == 0.0 - 1.0 and first["close"] == 14.0 + 0.5


def test_load_fx_1min_real_data_is_utc_and_sorted():
    # GBPUSD 1m data is on disk for 2011-2026; load one short window.
    bars = load_fx_1min("GBPUSD", dt.date(2020, 6, 1), dt.date(2020, 6, 5))
    assert not bars.empty
    assert str(bars.index.tz) == "UTC"
    assert bars.index.is_monotonic_increasing
    assert not bars.index.has_duplicates
    assert list(bars.columns[:4]) == ["open", "high", "low", "close"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/data/test_fx_intraday_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backtesting.data.fx_intraday_loader'`.

- [ ] **Step 3: Write the implementation**

Create `src/backtesting/data/fx_intraday_loader.py`:

```python
"""1-minute spot-FX bar loader + OHLC resampler.

Reads the canonical 8-column 1m parquet cache (tz-aware UTC) for a pair over a
date range and resamples to coarser bars. Pure reads; no cleaning beyond sort +
dedupe (the 1m cache is already spike-cleaned upstream).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl

from src.settings import get_local_storage_dir
from src.utils import logger

_COLS = ["open", "high", "low", "close", "volume"]


def load_fx_1min(pair: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    base = Path(get_local_storage_dir()) / "fx" / "massive" / "1min" / f"symbol={pair}"
    if not base.exists() or not any(base.glob("**/*.parquet")):
        logger.warning(f"[load_fx_1min] no 1m data for {pair}")
        return pd.DataFrame(columns=_COLS)
    df = pl.scan_parquet(base / "**/*.parquet").collect().to_pandas()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    out = pd.DataFrame({c: df[c].astype(float) for c in _COLS})
    out.index = ts
    out.index.name = "timestamp"
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    out = out[(out.index >= lo) & (out.index < hi)]
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def resample_ohlc(bars: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = bars.resample(freq, label="right", closed="right").agg(agg)
    return out.dropna(subset=["open"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/data/test_fx_intraday_loader.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/data/fx_intraday_loader.py tests/backtesting/data/test_fx_intraday_loader.py
git commit -m "feat(fx): intraday 1-minute bar loader + OHLC resampler (sub-project 2b)"
```

---

### Task 2: Order engine core -- data types + stop/limit fills + no-lookahead loop

**Files:**
- Create: `src/backtesting/engine/intraday_order_engine.py`
- Test: `tests/backtesting/engine/test_intraday_order_engine.py`

**Interfaces:**
- Consumes: nothing (self-contained engine primitives).
- Produces:
  - `Bar` namedtuple `(ts, open, high, low, close)`.
  - `Order` dataclass: `side` ("buy"/"sell"), `kind` ("stop"/"limit"), `trigger` (float), `qty` (float), `order_id` (int), `oco_group` (int|None), plus mutable `filled` (bool), `fill_price` (float|None), `fill_ts` (optional).
  - `Fill` namedtuple `(order_id, ts, price, qty, side)`.
  - `OrderEngine` class with: `add_order(order) -> int`, `cancel(order_id)`, `resting_orders` (list), `fills` (list), and `_match_order(order, bar) -> float | None` (returns fill price or None) implementing the exact fill model. The public bar loop (`run`) arrives in Task 4; this task delivers `match_resting_orders(bar)` which fills eligible resting entry orders against `bar` and records fills, honoring the "added this bar is not yet eligible" rule via an `armed_before_ts` stamp.

- [ ] **Step 1: Write the failing tests**

Create `tests/backtesting/engine/test_intraday_order_engine.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/engine/test_intraday_order_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `src/backtesting/engine/intraday_order_engine.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/engine/test_intraday_order_engine.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/intraday_order_engine.py tests/backtesting/engine/test_intraday_order_engine.py
git commit -m "feat(fx): intraday order engine core -- stop/limit fills, no-lookahead"
```

---

### Task 3: OCO groups + bracket positions (partial take + trailing stop)

**Files:**
- Modify: `src/backtesting/engine/intraday_order_engine.py`
- Test: `tests/backtesting/engine/test_intraday_order_engine.py`

**Interfaces:**
- Consumes: `OrderEngine`, `Order`, `Bar`, `Fill` (Task 2).
- Produces:
  - `OrderEngine.add_oco(a: Order, b: Order) -> int` (assigns a shared new `oco_group` to both, adds both, returns the group id).
  - `Position` dataclass tracking an open position: `side`, `qty`, `entry_price`, `stop`, `target`, `tp_fraction`, `trail_dist` (float|None), `high_water`/`low_water` (for trailing), `closed_qty`, `realized_pips`.
  - `OrderEngine.open_position(side, qty, entry_price, entry_ts, stop, target, tp_fraction, trail_dist) -> Position` and `OrderEngine.update_position(bar) -> list[tuple]` which, per bar, applies the both-in-one-bar=adverse rule, fills the partial target, arms/ratchets the trailing stop on the remainder, and closes on stop/trail. Returns a list of `(reason, price, qty)` exit events. `OrderEngine.position` holds the current open `Position` or None.

- [ ] **Step 1: Write the failing tests**

Append to `tests/backtesting/engine/test_intraday_order_engine.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/engine/test_intraday_order_engine.py -v`
Expected: FAIL with `ImportError: cannot import name 'Position'` (and the new tests error).

- [ ] **Step 3: Add the implementation**

Append to `src/backtesting/engine/intraday_order_engine.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/engine/test_intraday_order_engine.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/intraday_order_engine.py tests/backtesting/engine/test_intraday_order_engine.py
git commit -m "feat(fx): OCO groups + bracket positions (partial take, trailing stop, adverse rule)"
```

---

### Task 4: Bar loop with time controls (`run`)

**Files:**
- Modify: `src/backtesting/engine/intraday_order_engine.py`
- Test: `tests/backtesting/engine/test_intraday_order_engine.py`

**Interfaces:**
- Consumes: `OrderEngine`, `Bar`, `Position` (Tasks 2-3).
- Produces: `OrderEngine.run(bars: list[Bar], strategy) -> None` where `strategy` exposes `on_bar(bar, engine)`. Loop order per bar: (1) `update_position(bar)` for exits, (2) `match_resting_orders(bar)` for entries, (3) `strategy.on_bar(bar, engine)` to place/cancel orders (armed at `bar.ts`, so eligible next bar). Also `cancel_all_resting()` and a `flatten` at a caller-driven time (the strategy calls these from `on_bar` when the bar's session time hits the control) -- no wall-clock logic in the engine; the strategy owns session timing via fx_clock.

- [ ] **Step 1: Write the failing tests**

Append to `tests/backtesting/engine/test_intraday_order_engine.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/engine/test_intraday_order_engine.py -v`
Expected: FAIL with `AttributeError: 'OrderEngine' object has no attribute 'run'`.

- [ ] **Step 3: Add the implementation**

Append to `src/backtesting/engine/intraday_order_engine.py`:

```python
def _cancel_all_resting(self) -> None:
    self.resting_orders = []


def _run(self, bars, strategy) -> None:
    for bar in bars:
        if self.position is not None:
            self.update_position(bar)
        self.match_resting_orders(bar)
        strategy.on_bar(bar, self)


OrderEngine.cancel_all_resting = _cancel_all_resting
OrderEngine.run = _run
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/engine/test_intraday_order_engine.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/intraday_order_engine.py tests/backtesting/engine/test_intraday_order_engine.py
git commit -m "feat(fx): engine bar loop (run) with strategy callback + cancel-all"
```

---

### Task 5: #20 London Open Breakout strategy

**Files:**
- Create: `src/strategies/advanced/fx_london_breakout.py`
- Test: `tests/strategies/test_fx_london_breakout.py`

**Interfaces:**
- Consumes: `OrderEngine`, `Order`, `Bar` (engine); `fx_clock` (`in_session_mask`, `fx_trading_day`, `EXCHANGE_TZ`); `macro_calendar_tier1.tier1_release_in_window`; `Indicators.atr` from `src.backtesting.utils.indicators`; `fx_round_trip_pips`/`_pip_size` from `src.backtesting.costs.fx`.
- Produces:
  - `LondonBreakoutStrategy(pair, atr_d1, risk_frac=0.005, tp_fraction=0.5, offset_pips=3.0, releases=None)` with an `on_bar(bar, engine)` method implementing #20, and a `day_pnl_pips` dict keyed by FX day accumulating realized pips.
  - `asian_range(bars_1m, fx_day) -> tuple[float, float] | None` (high/low of 00:00-07:00 London on that FX day; None if no bars).

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_fx_london_breakout.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/strategies/test_fx_london_breakout.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `src/strategies/advanced/fx_london_breakout.py`:

```python
"""#20 London Open Breakout (Asian range break), intraday.

Per pair per FX day: define the Asian range (00:00-07:00 London), filter by
range width vs ATR(14,D1), skip tier-1 EUR/GBP release days, place an OCO
buy-stop/sell-stop bracket around the range in the 08:00-09:30 London window,
cancel unfilled at 09:30, manage a bracket exit (take half at 1x range, trail
the rest at 1x ATR(15m)), and flatten at 16:00 London. Drives the general
intraday order engine; fixed-fractional risk sizing.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src.backtesting.costs.fx import _pip_size, fx_round_trip_pips
from src.backtesting.engine.intraday_order_engine import Order
from src.backtesting.sessions.fx_clock import EXCHANGE_TZ, in_session_mask
from src.data.macro_calendar_tier1 import tier1_release_in_window

_LONDON = EXCHANGE_TZ["LONDON"]


def asian_range(bars_1m: pd.DataFrame, fx_day: dt.date):
    mask = in_session_mask(bars_1m.index, "ASIAN_RANGE")
    day_local = bars_1m.index.tz_convert(_LONDON).date
    sel = bars_1m[mask.values & (day_local == fx_day)]
    if sel.empty:
        return None
    return float(sel["high"].max()), float(sel["low"].min())


class LondonBreakoutStrategy:
    def __init__(self, pair, atr_d1, risk_frac=0.005, tp_fraction=0.5,
                 offset_pips=3.0, tier="major", releases=None):
        self.pair = pair
        self.atr_d1 = atr_d1                # dict[date -> ATR(14) daily, price units]
        self.risk_frac = float(risk_frac)
        self.tp_fraction = float(tp_fraction)
        self.pip = _pip_size(pair)
        self.offset = offset_pips * self.pip
        self.tier = tier
        self.releases = releases
        self.day_pnl_pips: dict[dt.date, float] = {}
        self._day = None
        self._armed = False
        self._range = None
        self._half_spread = fx_round_trip_pips(tier) / 2.0 * self.pip

    def _local(self, ts):
        return pd.Timestamp(ts).tz_convert(_LONDON)

    def on_bar(self, bar, engine):
        lt = self._local(bar.ts)
        day = lt.date()
        if day != self._day:
            self._start_day(day, bar, engine)
        # cancel unfilled entries at 09:30 London
        if (lt.hour, lt.minute) == (9, 30):
            engine.cancel_all_resting()
        # flatten at 16:00 London
        if (lt.hour, lt.minute) >= (16, 0) and engine.position is not None:
            ev = engine.flatten(bar.close, bar.ts, reason="flat_1600")
            self._book(day, ev)

    def _start_day(self, day, bar, engine):
        self._day = day
        self._armed = False
        self._range = None
        if self._skip_day(day):
            return
        # placeholder; real range + orders armed at 08:00 (see _maybe_arm)
        self._range = "pending"

    def _skip_day(self, day) -> bool:
        atr = self.atr_d1.get(day)
        if atr is None:
            return True
        return tier1_release_in_window(day, dt.time(9, 30), dt.time(12, 1),
                                       exchange="LONDON", currencies=("EUR", "GBP"),
                                       releases=self.releases)
```

Note: the implementer completes `on_bar` so that at the first bar with local time >= 08:00 (and < 09:30) on a non-skip day, it computes the Asian range from the day's bars seen so far via a rolling min/max over the 00:00-07:00 mask, applies the width filter (`0.25*atr <= width <= 0.80*atr`), and if passing arms an OCO bracket: buy-stop at `hi + offset`, sell-stop at `lo - offset`, sized so risk-to-opposite-side equals `risk_frac` (qty = risk_frac / (stop_distance_in_price)); on an entry fill it opens a Position with stop = opposite range side, target = 1*width beyond entry, tp_fraction, trail_dist = 1*ATR(15m) computed from the day's 1m bars resampled to 15m; realized pips (net of `2*_half_spread` round trip) accumulate into `day_pnl_pips[day]`. The strategy tracks the Asian high/low incrementally from `on_bar` (never reading future bars). Keep all session-time comparisons in London local via `_local`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/strategies/test_fx_london_breakout.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/advanced/fx_london_breakout.py tests/strategies/test_fx_london_breakout.py
git commit -m "feat(fx): #20 London Open Breakout strategy driving the intraday engine"
```

---

### Task 6: Walk-forward runner + S&P gate + pre-registration

**Files:**
- Create: `scripts/backtest_scripts/run_fx_london_breakout_walkforward.py`
- Create: `config/backtesting/fx_london_breakout.yaml`
- Create: `docs/reports/fx/20260719_london_breakout_prereg.md`
- Test: `tests/backtesting/test_fx_london_breakout_runner.py`

**Interfaces:**
- Consumes: `load_fx_1min` (Task 1); `LondonBreakoutStrategy` (Task 5); `Indicators.atr`; `load_fx_daily_panel` (daily ATR source); `walkforward_common` helpers; `benchmark.py`; `RunStatus`; mirrors `run_fx_carry_seatbelt_walkforward.py`.
- Produces: `build_daily_returns(pairs, start, end) -> pd.Series` (run each pair through the engine per FX day, aggregate per-FX-day pips into a combined equal-risk daily return series) and a `run(...)` that feeds the existing walk-forward + S&P gate, writing `docs/reports/fx/FX_LONDON_BREAKOUT_WALK_FORWARD.md`.

- [ ] **Step 1: Write the pre-registration note**

Create `docs/reports/fx/20260719_london_breakout_prereg.md`:

```markdown
# #20 London Open Breakout Pre-Registration - 2026-07-19

Written and committed BEFORE any London Breakout walk-forward was run.

## Strategy
#20 London Open Breakout (Asian range break), intraday, pairs GBPUSD/EURUSD/
EURGBP/GBPJPY. Spec: docs/superpowers/specs/2026-07-19-fx-intraday-london-breakout-design.md.

## Success criterion (primary, relative)
Aggregate per-pair intraday P&L into a combined equal-risk daily return series;
run the existing FX walk-forward (36m/12m/12m, both 1.0x and 1.5x cost legs);
PASS if stitched OOS Sharpe (1x) exceeds the S&P 500 Sharpe over the same OOS
dates. Both cost legs reported.

## Diagnostics (non-gating)
PSR, DSR (project-wide trial count), PBO, IS/OOS Sharpe ratio, correlation and IR
vs S&P, S&P aligned day count.

## No absolute kill threshold
A form that fails the S&P bar is a failed base form; one bounded improvement
round (a #20 modification a-d) may follow only if it lands marginal.

## Known limitations
Conservative 1m fills (worst-of trigger/open, adverse both-in-one-bar); half-
spread slippage is a floor; approximate tier-1 event dates (2a).
```

- [ ] **Step 2: Write the config**

Create `config/backtesting/fx_london_breakout.yaml`:

```yaml
asset_class: fx
strategy:
  name: LondonBreakout
  pairs: [GBPUSD, EURUSD, EURGBP, GBPJPY]
  params: {risk_frac: 0.005, tp_fraction: 0.5, offset_pips: 3.0}
dates:
  start: "2011-01-01"
  end: "2026-04-01"
backtest:
  initial_capital: 100000.0
output:
  save_trades: true
```

- [ ] **Step 3: Write the failing test**

Create `tests/backtesting/test_fx_london_breakout_runner.py`:

```python
import datetime as dt

from src.backtesting_scripts_shim import build_daily_returns  # see note


def test_build_daily_returns_short_window_produces_series():
    s = build_daily_returns(["GBPUSD"], dt.date(2020, 6, 1), dt.date(2020, 6, 30))
    assert s is not None
    assert len(s) > 5
    assert s.index.is_monotonic_increasing
```

Note: the runner lives at `scripts/backtest_scripts/run_fx_london_breakout_walkforward.py`; expose `build_daily_returns` importably by adding `scripts` to the path in the test via `import sys; sys.path.insert(0, "scripts/backtest_scripts")` then `from run_fx_london_breakout_walkforward import build_daily_returns`. Replace the shim import above with that. (Do not create `src/backtesting_scripts_shim`.)

- [ ] **Step 4: Run the test to verify it fails**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/test_fx_london_breakout_runner.py -v`
Expected: FAIL (module not found).

- [ ] **Step 5: Write the runner**

Create `scripts/backtest_scripts/run_fx_london_breakout_walkforward.py` mirroring `run_fx_carry_seatbelt_walkforward.py`. It must:
- For each pair, `load_fx_1min` over [start, end]; compute daily ATR(14) via `Indicators.atr` on `load_fx_daily_panel` OHLC (map FX day -> prior-day ATR, no lookahead); precompute the tier-1 `releases` frame once via `generate_tier1_releases(start, end)`.
- Group the 1m bars by FX day (`fx_trading_day`), run `LondonBreakoutStrategy` through `OrderEngine.run` per day, collect `day_pnl_pips`.
- Convert each pair's per-day pips to a per-day return (pips * pip_value_fraction; since risk_frac sizing already normalizes, use realized_pips / entry-risk as an R-multiple times risk_frac, i.e. daily return = risk_frac * R_day), aggregate equal-weight across pairs into one daily return series (`build_daily_returns`).
- Feed that series to the existing walk-forward gate + S&P comparison (reuse the carry runner's window/stat/report code path), writing `docs/reports/fx/FX_LONDON_BREAKOUT_WALK_FORWARD.md` with the S&P verdict + diagnostics + episode notes. Wrap the run in `RunStatus`. Increment the project-wide DSR trial count.

- [ ] **Step 6: Run the test + the walk-forward**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/test_fx_london_breakout_runner.py -v`
Expected: PASS.
Then: `conda activate fintech; PYTHONPATH=$(pwd) python scripts/backtest_scripts/run_fx_london_breakout_walkforward.py`
Expected: writes the readiness report with the real PASS/FAIL-vs-S&P verdict (several minutes; 15 years of 1m bars x 4 pairs).

- [ ] **Step 7: Commit**

```bash
git add scripts/backtest_scripts/run_fx_london_breakout_walkforward.py config/backtesting/fx_london_breakout.yaml docs/reports/fx/20260719_london_breakout_prereg.md docs/reports/fx/FX_LONDON_BREAKOUT_WALK_FORWARD.md tests/backtesting/test_fx_london_breakout_runner.py
git commit -m "feat(fx): #20 London Breakout walk-forward runner + S&P gate report"
```

---

## Post-implementation (orchestrator, after all tasks)

- Confirm the engine suite (10 tests) + strategy suite (4) + loader (2) + runner (1) all pass.
- Record the #20 verdict (PASS/FAIL vs S&P) in `docs/strategies/FX_60_CATALOG_TRACKER.md` (strategy #20) and write the session results doc + progress log. This is the FIRST gated intraday result and the 6th gated strategy of the 60.
- If #20 fails the S&P bar, decide (per the pre-registration) whether to run one bounded #20 modification (a-d) or move to the next intraday strategy on the now-built engine.
