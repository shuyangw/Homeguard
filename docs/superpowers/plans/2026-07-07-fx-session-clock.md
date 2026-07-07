# FX Session Clock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure, DST-correct FX session/time-semantics module (`src/backtesting/sessions/fx_clock.py`) that converts exchange-local session times to/from UTC, labels UTC bars by trading session, defines the 17:00-ET FX trading day/week, and buckets bars by hour-of-week.

**Architecture:** One pure module over stdlib `zoneinfo`, operating on tz-aware UTC timestamps. An exchange registry and a named session-window registry (a `SessionWindow` dataclass) drive nine functions. Almost all logic runs on the always-unambiguous UTC->local path (`tz_convert`); only the scalar `local_to_utc` touches local->UTC, using pandas `tz_localize` policies.

**Tech Stack:** Python 3.13 (conda env `fintech`), stdlib `zoneinfo`, pandas, numpy, pytest. No new dependencies.

## Global Constraints

- Run Python via the `fintech` conda env. Test command prefix: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest <path> -v` (conda is already initialized in the shell).
- Pure module: no file, network, or config I/O. Deterministic functions only.
- Built on stdlib `zoneinfo` (`from zoneinfo import ZoneInfo`). Do NOT add a timezone dependency. Do NOT modify or import `src/utils/timezone.py`.
- All vectorized functions take a tz-aware UTC `pd.DatetimeIndex` and return a `pd.Series` indexed by it.
- ASCII-only, no em dashes, no emojis, no `print()`. This module needs no logging (pure functions); do not add any.
- Session windows use the half-open convention `[start, end)`. Times are exchange-LOCAL. Windows may cross local midnight (`end < start`).
- Hour-of-week is 0-167 with Monday 00:00 = 0.
- The FX trading-day boundary is 17:00 `America/New_York` (bars at/after 17:00 ET belong to the next FX day).
- Git hazard (macOS/Dropbox): use ONLY `git add <paths>`, `git commit`, `git log`. NEVER `git checkout`, bare `git status`/`git diff`, or `git reset`. These files are not under docs/, so normal `git add` works.

---

### Task 1: Package, registries, `SessionWindow`, and `local_to_utc`

**Files:**
- Create: `src/backtesting/sessions/__init__.py`
- Create: `src/backtesting/sessions/fx_clock.py`
- Create: `tests/backtesting/sessions/__init__.py`
- Test: `tests/backtesting/sessions/test_fx_clock.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `EXCHANGE_TZ: dict[str, ZoneInfo]` (keys `TOKYO`, `LONDON`, `NEW_YORK`).
  - `SessionWindow` dataclass `(exchange: str, start: datetime.time, end: datetime.time)`.
  - `SESSION_WINDOWS: dict[str, SessionWindow]` (keys `TOKYO`, `ASIAN_RANGE`, `LONDON`, `NEW_YORK`, `WMR_FIX_LONDON`).
  - `_zone_for(exchange: str) -> ZoneInfo` (resolves a registry key or a raw IANA string).
  - `local_to_utc(exchange: str, local_dt: datetime.datetime, nonexistent: str = "roll_forward", ambiguous: str = "first") -> pd.Timestamp`.

- [ ] **Step 1: Write the failing tests**

Create `tests/backtesting/sessions/__init__.py` (empty), then `tests/backtesting/sessions/test_fx_clock.py`:

```python
import datetime as dt

import pandas as pd
import pytest

from src.backtesting.sessions.fx_clock import (
    EXCHANGE_TZ, SESSION_WINDOWS, SessionWindow, local_to_utc)


def test_registries_present():
    assert set(EXCHANGE_TZ) == {"TOKYO", "LONDON", "NEW_YORK"}
    assert {"TOKYO", "ASIAN_RANGE", "LONDON", "NEW_YORK", "WMR_FIX_LONDON"} <= set(SESSION_WINDOWS)
    assert isinstance(SESSION_WINDOWS["LONDON"], SessionWindow)


def test_local_to_utc_dst_offsets():
    # London 08:00 is 08:00 UTC in winter (GMT), 07:00 UTC in summer (BST)
    assert local_to_utc("LONDON", dt.datetime(2024, 1, 15, 8, 0)) == pd.Timestamp("2024-01-15 08:00", tz="UTC")
    assert local_to_utc("LONDON", dt.datetime(2024, 6, 15, 8, 0)) == pd.Timestamp("2024-06-15 07:00", tz="UTC")
    # NY 17:00 is 22:00 UTC in winter (EST), 21:00 UTC in summer (EDT)
    assert local_to_utc("NEW_YORK", dt.datetime(2024, 1, 15, 17, 0)) == pd.Timestamp("2024-01-15 22:00", tz="UTC")
    assert local_to_utc("NEW_YORK", dt.datetime(2024, 6, 15, 17, 0)) == pd.Timestamp("2024-06-15 21:00", tz="UTC")


def test_local_to_utc_transition_policy():
    # 2024-03-31 spring-forward gap in London: 01:30 does not exist -> rolls forward
    got = local_to_utc("LONDON", dt.datetime(2024, 3, 31, 1, 30))
    assert got == pd.Timestamp("2024-03-31 01:00", tz="UTC")  # 02:00 BST == 01:00 UTC
    # 2024-10-27 fall-back overlap in London: 01:30 occurs twice -> first (BST) taken
    got2 = local_to_utc("LONDON", dt.datetime(2024, 10, 27, 1, 30))
    assert got2 == pd.Timestamp("2024-10-27 00:30", tz="UTC")  # first occurrence is BST (UTC+1)


def test_raw_iana_exchange_accepted():
    assert local_to_utc("Asia/Tokyo", dt.datetime(2024, 6, 15, 9, 0)) == pd.Timestamp("2024-06-15 00:00", tz="UTC")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/sessions/test_fx_clock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backtesting.sessions'`.

- [ ] **Step 3: Create the package and implementation**

Create `src/backtesting/sessions/__init__.py` (empty).

Create `src/backtesting/sessions/fx_clock.py`:

```python
"""FX session / DST clock: exchange-local <-> UTC, session masks, FX day/week,
hour-of-week. Pure functions over stdlib zoneinfo, operating on tz-aware UTC
timestamps. Centralizes DST correctness for all intraday FX work.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

EXCHANGE_TZ: dict[str, ZoneInfo] = {
    "TOKYO": ZoneInfo("Asia/Tokyo"),
    "LONDON": ZoneInfo("Europe/London"),
    "NEW_YORK": ZoneInfo("America/New_York"),
}


@dataclass(frozen=True)
class SessionWindow:
    exchange: str
    start: dt.time
    end: dt.time


SESSION_WINDOWS: dict[str, SessionWindow] = {
    "TOKYO": SessionWindow("TOKYO", dt.time(9, 0), dt.time(15, 0)),
    "ASIAN_RANGE": SessionWindow("LONDON", dt.time(0, 0), dt.time(7, 0)),
    "LONDON": SessionWindow("LONDON", dt.time(8, 0), dt.time(16, 30)),
    "NEW_YORK": SessionWindow("NEW_YORK", dt.time(8, 0), dt.time(17, 0)),
    "WMR_FIX_LONDON": SessionWindow("LONDON", dt.time(15, 58), dt.time(16, 2)),
}

_NONEXISTENT = {"roll_forward": "shift_forward", "roll_backward": "shift_backward"}
_AMBIGUOUS = {"first": True, "second": False}


def _zone_for(exchange: str) -> ZoneInfo:
    tz = EXCHANGE_TZ.get(exchange)
    return tz if tz is not None else ZoneInfo(exchange)


def local_to_utc(exchange: str, local_dt: dt.datetime,
                 nonexistent: str = "roll_forward",
                 ambiguous: str = "first") -> pd.Timestamp:
    ts = pd.Timestamp(local_dt).tz_localize(
        _zone_for(exchange),
        ambiguous=_AMBIGUOUS[ambiguous],
        nonexistent=_NONEXISTENT[nonexistent],
    )
    return ts.tz_convert("UTC")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/sessions/test_fx_clock.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/sessions/__init__.py src/backtesting/sessions/fx_clock.py tests/backtesting/sessions/
git commit -m "feat(fx): session-clock registries + local_to_utc (intraday sub-project 1)"
```

---

### Task 2: `session_window_utc` and `in_session_mask`

**Files:**
- Modify: `src/backtesting/sessions/fx_clock.py`
- Test: `tests/backtesting/sessions/test_fx_clock.py`

**Interfaces:**
- Consumes: `SESSION_WINDOWS`, `SessionWindow`, `_zone_for`, `local_to_utc`, `EXCHANGE_TZ` from Task 1.
- Produces:
  - `_resolve_window(window: str | SessionWindow) -> SessionWindow`.
  - `session_window_utc(window: str | SessionWindow, day: datetime.date) -> tuple[pd.Timestamp, pd.Timestamp]`.
  - `in_session_mask(utc_index: pd.DatetimeIndex, window: str | SessionWindow) -> pd.Series`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/backtesting/sessions/test_fx_clock.py`:

```python
from src.backtesting.sessions.fx_clock import session_window_utc, in_session_mask


def test_session_window_utc_dst_and_tokyo():
    # LONDON 08:00-16:30 -> 08:00/16:30 UTC winter, 07:00/15:30 UTC summer
    s, e = session_window_utc("LONDON", dt.date(2024, 1, 15))
    assert (s, e) == (pd.Timestamp("2024-01-15 08:00", tz="UTC"), pd.Timestamp("2024-01-15 16:30", tz="UTC"))
    s, e = session_window_utc("LONDON", dt.date(2024, 6, 15))
    assert (s, e) == (pd.Timestamp("2024-06-15 07:00", tz="UTC"), pd.Timestamp("2024-06-15 15:30", tz="UTC"))
    # TOKYO has no DST: always 00:00-06:00 UTC
    for d in (dt.date(2024, 1, 15), dt.date(2024, 6, 15)):
        assert session_window_utc("TOKYO", d) == (
            pd.Timestamp(f"{d} 00:00", tz="UTC"), pd.Timestamp(f"{d} 06:00", tz="UTC"))


def test_session_window_utc_offset_divergence():
    # 2024-03-20: NY on EDT (UTC-4), London still GMT (UTC+0) -> 4h gap not 5h
    ln_start, _ = session_window_utc("LONDON", dt.date(2024, 3, 20))
    ny_start, _ = session_window_utc("NEW_YORK", dt.date(2024, 3, 20))
    assert ln_start == pd.Timestamp("2024-03-20 08:00", tz="UTC")
    assert ny_start == pd.Timestamp("2024-03-20 12:00", tz="UTC")
    assert (ny_start - ln_start) == pd.Timedelta(hours=4)


def test_in_session_mask_flips_at_correct_utc_minute():
    # LONDON opens 08:00 London. Winter day: 08:00 UTC. Summer day: 07:00 UTC.
    idx_w = pd.date_range("2024-01-15 07:58", "2024-01-15 08:02", freq="1min", tz="UTC")
    m = in_session_mask(idx_w, "LONDON")
    assert not m.loc["2024-01-15 07:59"] and m.loc["2024-01-15 08:00"]
    idx_s = pd.date_range("2024-06-17 06:58", "2024-06-17 07:02", freq="1min", tz="UTC")
    m2 = in_session_mask(idx_s, "LONDON")
    assert not m2.loc["2024-06-17 06:59"] and m2.loc["2024-06-17 07:00"]


def test_in_session_mask_midnight_crossing():
    w = SessionWindow("LONDON", dt.time(22, 0), dt.time(2, 0))  # crosses local midnight
    idx = pd.date_range("2024-01-15 21:30", "2024-01-16 02:30", freq="30min", tz="UTC")
    m = in_session_mask(idx, w)  # London == UTC in January
    assert not m.loc["2024-01-15 21:30"]
    assert m.loc["2024-01-15 22:00"] and m.loc["2024-01-16 01:30"]
    assert not m.loc["2024-01-16 02:00"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/sessions/test_fx_clock.py -v`
Expected: FAIL with `ImportError: cannot import name 'session_window_utc'`.

- [ ] **Step 3: Add the implementation**

Append to `src/backtesting/sessions/fx_clock.py`:

```python
def _resolve_window(window: "str | SessionWindow") -> SessionWindow:
    if isinstance(window, SessionWindow):
        return window
    return SESSION_WINDOWS[window]


def session_window_utc(window: "str | SessionWindow",
                       day: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    w = _resolve_window(window)
    end_day = day if w.end > w.start else day + dt.timedelta(days=1)
    start = local_to_utc(w.exchange, dt.datetime.combine(day, w.start))
    end = local_to_utc(w.exchange, dt.datetime.combine(end_day, w.end))
    return start, end


def _seconds_of_day(t: dt.time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


def in_session_mask(utc_index: pd.DatetimeIndex,
                    window: "str | SessionWindow") -> pd.Series:
    w = _resolve_window(window)
    local = utc_index.tz_convert(_zone_for(w.exchange))
    sod = local.hour * 3600 + local.minute * 60 + local.second
    s, e = _seconds_of_day(w.start), _seconds_of_day(w.end)
    mask = (sod >= s) & (sod < e) if s <= e else (sod >= s) | (sod < e)
    return pd.Series(mask, index=utc_index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/sessions/test_fx_clock.py -v`
Expected: 8 passed (4 from Task 1 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/sessions/fx_clock.py tests/backtesting/sessions/test_fx_clock.py
git commit -m "feat(fx): session_window_utc + in_session_mask (DST-correct, vectorized)"
```

---

### Task 3: FX trading day / week -- `fx_trading_day`, `is_friday_fx`, `fx_trading_week_id`

**Files:**
- Modify: `src/backtesting/sessions/fx_clock.py`
- Test: `tests/backtesting/sessions/test_fx_clock.py`

**Interfaces:**
- Consumes: nothing beyond stdlib/pandas (operates directly on the UTC index).
- Produces:
  - `fx_trading_day(utc_index: pd.DatetimeIndex) -> pd.Series` (values `datetime.date`).
  - `is_friday_fx(utc_index: pd.DatetimeIndex) -> pd.Series` (bool).
  - `fx_trading_week_id(utc_index: pd.DatetimeIndex) -> pd.Series` (int, `iso_year * 100 + iso_week`, constant within one FX week).

- [ ] **Step 1: Write the failing tests**

Append to `tests/backtesting/sessions/test_fx_clock.py`:

```python
from src.backtesting.sessions.fx_clock import fx_trading_day, is_friday_fx, fx_trading_week_id


def test_fx_trading_day_17et_boundary():
    # Summer: boundary 21:00 UTC (17:00 EDT). 20:30 UTC -> same day; 21:30 UTC -> next day.
    idx = pd.DatetimeIndex(["2024-06-13 20:30", "2024-06-13 21:30"], tz="UTC")
    days = fx_trading_day(idx)
    assert days.iloc[0] == dt.date(2024, 6, 13)
    assert days.iloc[1] == dt.date(2024, 6, 14)
    # Winter: boundary 22:00 UTC (17:00 EST). 21:30 UTC -> still same day.
    idxw = pd.DatetimeIndex(["2024-01-15 21:30"], tz="UTC")
    assert fx_trading_day(idxw).iloc[0] == dt.date(2024, 1, 15)


def test_is_friday_fx_rolls_at_17et():
    # 2024-06-13 is Thursday; 21:30 UTC (17:30 EDT) rolls into Friday 2024-06-14.
    idx = pd.DatetimeIndex(["2024-06-13 20:30", "2024-06-13 21:30"], tz="UTC")
    ff = is_friday_fx(idx)
    assert not ff.iloc[0]  # still Thursday FX day
    assert ff.iloc[1]      # rolled into Friday FX day


def test_fx_trading_week_id_constant_mon_to_fri():
    idx = pd.DatetimeIndex(["2024-06-10 12:00", "2024-06-14 12:00"], tz="UTC")  # Mon..Fri
    wk = fx_trading_week_id(idx)
    assert wk.iloc[0] == wk.iloc[1]
    idx2 = pd.DatetimeIndex(["2024-06-14 12:00", "2024-06-17 12:00"], tz="UTC")  # Fri vs next Mon
    wk2 = fx_trading_week_id(idx2)
    assert wk2.iloc[0] != wk2.iloc[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/sessions/test_fx_clock.py -v`
Expected: FAIL with `ImportError: cannot import name 'fx_trading_day'`.

- [ ] **Step 3: Add the implementation**

Append to `src/backtesting/sessions/fx_clock.py`:

```python
def _fx_day_index(utc_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Shift NY-local time by +7h so the 17:00-ET boundary becomes midnight, then
    # the local calendar date is the FX trading day. DST handled by tz_convert.
    ny = utc_index.tz_convert("America/New_York") + pd.Timedelta(hours=7)
    return ny


def fx_trading_day(utc_index: pd.DatetimeIndex) -> pd.Series:
    shifted = _fx_day_index(utc_index)
    return pd.Series(shifted.date, index=utc_index)


def is_friday_fx(utc_index: pd.DatetimeIndex) -> pd.Series:
    shifted = _fx_day_index(utc_index)
    return pd.Series(shifted.dayofweek == 4, index=utc_index)


def fx_trading_week_id(utc_index: pd.DatetimeIndex) -> pd.Series:
    shifted = _fx_day_index(utc_index)
    iso = shifted.isocalendar()
    ids = (iso.year.to_numpy() * 100 + iso.week.to_numpy())
    return pd.Series(ids, index=utc_index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/sessions/test_fx_clock.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/sessions/fx_clock.py tests/backtesting/sessions/test_fx_clock.py
git commit -m "feat(fx): FX trading day/week + Friday flag (17:00-ET boundary)"
```

---

### Task 4: Hour-of-week -- `hour_of_week_utc` and `hour_of_week_anchored`

**Files:**
- Modify: `src/backtesting/sessions/fx_clock.py`
- Test: `tests/backtesting/sessions/test_fx_clock.py`

**Interfaces:**
- Consumes: `_zone_for` from Task 1.
- Produces:
  - `hour_of_week_utc(utc_index: pd.DatetimeIndex) -> pd.Series` (int 0-167, Monday 00:00 = 0).
  - `hour_of_week_anchored(utc_index: pd.DatetimeIndex, anchor: str = "Europe/London") -> pd.Series` (int 0-167 in anchor-local wall time).

- [ ] **Step 1: Write the failing tests**

Append to `tests/backtesting/sessions/test_fx_clock.py`:

```python
from src.backtesting.sessions.fx_clock import hour_of_week_utc, hour_of_week_anchored


def test_hour_of_week_utc_monday_zero():
    idx = pd.DatetimeIndex(["2024-06-10 00:00", "2024-06-10 08:00", "2024-06-11 00:00"], tz="UTC")
    how = hour_of_week_utc(idx)  # Mon 00:00 -> 0, Mon 08:00 -> 8, Tue 00:00 -> 24
    assert list(how) == [0, 8, 24]


def test_hour_of_week_anchored_is_dst_stable():
    # 08:00 London-local on a winter Monday (08:00 UTC) and a summer Monday (07:00 UTC).
    winter = pd.DatetimeIndex(["2024-01-15 08:00"], tz="UTC")  # Mon, London==UTC
    summer = pd.DatetimeIndex(["2024-06-17 07:00"], tz="UTC")  # Mon, London==UTC+1
    # Anchored (London) bucket is identical (Monday hour 8) across the DST change...
    assert hour_of_week_anchored(winter).iloc[0] == 8
    assert hour_of_week_anchored(summer).iloc[0] == 8
    # ...while the raw UTC hour-of-week differs by one (8 vs 7).
    assert hour_of_week_utc(winter).iloc[0] == 8
    assert hour_of_week_utc(summer).iloc[0] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/sessions/test_fx_clock.py -v`
Expected: FAIL with `ImportError: cannot import name 'hour_of_week_utc'`.

- [ ] **Step 3: Add the implementation**

Append to `src/backtesting/sessions/fx_clock.py`:

```python
def hour_of_week_utc(utc_index: pd.DatetimeIndex) -> pd.Series:
    how = utc_index.dayofweek * 24 + utc_index.hour
    return pd.Series(how, index=utc_index)


def hour_of_week_anchored(utc_index: pd.DatetimeIndex,
                          anchor: str = "Europe/London") -> pd.Series:
    local = utc_index.tz_convert(_zone_for(anchor))
    how = local.dayofweek * 24 + local.hour
    return pd.Series(how, index=utc_index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/sessions/test_fx_clock.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/sessions/fx_clock.py tests/backtesting/sessions/test_fx_clock.py
git commit -m "feat(fx): UTC + London-anchored hour-of-week bucketing"
```

---

## Post-implementation (orchestrator, after all tasks)

- Confirm the full module test suite passes (13 tests) and the module has no I/O / no print / no em dashes.
- This completes intraday sub-project 1 of 4. The next decision (per the campaign): sub-project 2 (intraday bar loader + session aggregations) builds on `in_session_mask` / `session_window_utc` / the hour-of-week helpers to compute actual session-window values (Asian-range high/low, rolling vol curves).
