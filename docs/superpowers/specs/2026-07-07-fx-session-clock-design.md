# FX Session Clock Design Spec

**Date:** 2026-07-07
**Status:** Approved (brainstorm), pending implementation plan
**Context:** Sub-project 1 of 4 in the intraday FX engine (the campaign pivot after two daily carry FAILs). The intraday engine decomposes into: (1) THIS session/DST clock, (2) intraday bar loader + session aggregations, (3) event-driven engine with OCO/bracket orders, (4) the ~24 intraday strategies (#20-25 and others). The clock is built first because the research flags exchange-local-vs-UTC / DST handling as the error that "silently corrupts a quarter of your backtest," and it is the pure, foundational layer everything else depends on. Building it first de-risks the whole intraday investment before committing to the larger engine.

## 1. Purpose

Provide one correct, DST-aware time-semantics layer for FX intraday work: convert between exchange-local session times and UTC, label UTC bars by trading session, define the FX trading day/week boundary, and bucket bars by hour-of-week for same-hour-distribution strategies. Every downstream intraday component (loader, engine, strategies) consumes this layer instead of hand-rolling timezone math, so DST correctness is centralized and tested once.

## 2. Constraints and inputs

- Input timestamps are tz-aware UTC (`datetime[ns, UTC]`), matching the on-disk 1-minute FX bars (`fx/massive/1min/symbol=*/**.parquet`). All vectorized functions take and return objects indexed by a tz-aware UTC `pd.DatetimeIndex`.
- Built on the Python standard library `zoneinfo` (Python 3.13). Do NOT add a new timezone dependency. The existing `src/utils/timezone.py` (a pytz-based live-ops display helper) is a different concern and is not modified or reused here.
- Pure module: no file, network, or config I/O. Deterministic functions only.
- ASCII-only, no em dashes, no emojis, no print().

## 3. Architecture

A single module `src/backtesting/sessions/fx_clock.py` in a new package `src/backtesting/sessions/`. Three parts:

### 3.1 Exchange registry
A mapping of exchange name to IANA timezone:
- `TOKYO` -> `Asia/Tokyo` (no DST, fixed UTC+9)
- `LONDON` -> `Europe/London` (GMT / BST)
- `NEW_YORK` -> `America/New_York` (EST / EDT)

Exposed as `EXCHANGE_TZ: dict[str, ZoneInfo]`.

### 3.2 Named session-window registry
`SESSION_WINDOWS: dict[str, SessionWindow]`, where `SessionWindow` is a small dataclass `(exchange: str, start: datetime.time, end: datetime.time)`. Each window is anchored to an exchange and expressed in that exchange's LOCAL time, so DST is handled automatically by tz conversion. Ships these canonical windows (from the research strategies):

| Name | Exchange | Local start | Local end | Source |
|---|---|---|---|---|
| `TOKYO` | Asia/Tokyo | 09:00 | 15:00 | #22 (equals 00:00-06:00 UTC Tokyo core) |
| `ASIAN_RANGE` | Europe/London | 00:00 | 07:00 | #20 (Asian range defined in London time) |
| `LONDON` | Europe/London | 08:00 | 16:30 | #20, #21, #23 |
| `NEW_YORK` | America/New_York | 08:00 | 17:00 | #21, #24 |
| `WMR_FIX_LONDON` | Europe/London | 15:58 | 16:02 | #23 (4-minute window around the 16:00 fix) |

The registry is extensible: a caller can add a window without changing the core functions. Windows that cross midnight local time (end < start) are supported (mask logic handles the wrap).

### 3.3 Functions
All pure functions over the two registries.

## 4. API

Signatures (exact, for the plan to use verbatim):

```python
def local_to_utc(exchange: str, local_dt: datetime.datetime,
                 nonexistent: str = "roll_forward",
                 ambiguous: str = "first") -> pd.Timestamp
```
Convert a naive local datetime in `exchange` to a tz-aware UTC `Timestamp`, DST-correct. `nonexistent="roll_forward"`: a local time inside a spring-forward gap maps to the next valid instant. `ambiguous="first"`: a local time inside a fall-back overlap takes the first (pre-transition, `fold=0`) occurrence. These policies are documented and unit-tested; they matter only for scalar boundary construction on transition dates.

```python
def session_window_utc(window_name: str, day: datetime.date) -> tuple[pd.Timestamp, pd.Timestamp]
```
Return `(utc_start, utc_end)` for a named window on a given calendar `day`, using that day's DST state. For a midnight-crossing window, `utc_end` is on the following UTC calendar day. Built on `local_to_utc`.

```python
def in_session_mask(utc_index: pd.DatetimeIndex, window_name: str) -> pd.Series
```
The workhorse. Boolean Series indexed by `utc_index` (tz-aware UTC), True where the bar falls in the named window. Implemented by converting the index to the window's exchange-local tz (`utc_index.tz_convert(tz)`) and comparing local wall-clock time to `[start, end)`. Inherently DST-correct (UTC -> local is never ambiguous) and vectorized. Handles midnight-crossing windows via an OR of the two half-open ranges. Half-open convention `[start, end)`.

```python
def fx_trading_day(utc_index: pd.DatetimeIndex) -> pd.Series
```
The FX trading day (a `datetime.date`) each bar belongs to, with the day boundary at 17:00 `America/New_York` (the standard FX rollover). Implemented by converting to NY local, adding 7 hours, and taking the local date. DST-correct. Bars from 17:00 ET onward belong to the NEXT calendar day's FX session.

```python
def is_friday_fx(utc_index: pd.DatetimeIndex) -> pd.Series
def fx_trading_week_id(utc_index: pd.DatetimeIndex) -> pd.Series
```
`is_friday_fx`: True where the FX trading day (17:00-ET convention) is a Friday (for #24 Friday-squaring). `fx_trading_week_id`: an integer/label constant within one FX trading week (Sunday 17:00 ET open through Friday 17:00 ET close), for week-scoped aggregations; derived from `fx_trading_day` via ISO week of the Thursday-anchored trading day.

```python
def hour_of_week_utc(utc_index: pd.DatetimeIndex) -> pd.Series
def hour_of_week_anchored(utc_index: pd.DatetimeIndex, anchor: str = "Europe/London") -> pd.Series
```
Integer 0-167 (Monday 00:00 = hour 0). `hour_of_week_utc` uses raw UTC wall time. `hour_of_week_anchored` uses `anchor`-local wall time (default `Europe/London`), so a fixed local session event maps to a DST-stable bucket year-round. The vol-curve / same-hour-distribution helpers for #22 and #25 default to the anchored form.

## 5. DST correctness (the core value)

The design keeps almost all logic on the always-unambiguous UTC -> local path (`tz_convert`), which is DST-correct by construction. The only ambiguity-prone path is the scalar `local_to_utc` (local -> UTC), used to build window boundaries; its `nonexistent` / `ambiguous` policies are explicit and tested. The load-bearing DST hazard the whole layer exists to prevent -- the London <-> New York offset being 4h or 6h instead of 5h during the ~2-week windows when one region has switched DST and the other has not -- is handled automatically because each exchange window is anchored to its own IANA zone and converted independently.

## 6. Testing plan

Tests live in `tests/backtesting/sessions/test_fx_clock.py` and pin real transition dates (2024 used for concrete values):

1. **London BST transitions:** 2024-03-31 (spring forward), 2024-10-27 (fall back). Assert the LONDON window's UTC bounds shift by exactly one hour across the transition (08:00 London = 08:00 UTC in winter, 07:00 UTC in summer).
2. **New York EDT transitions:** 2024-03-10, 2024-11-03. Assert the NEW_YORK window and the 17:00-ET FX-day boundary land on the correct UTC instant (21:00 UTC in summer, 22:00 UTC in winter).
3. **London/NY offset divergence:** a date in 2024-03-11..2024-03-30 (NY on EDT, London still GMT -> 4h offset). Assert LONDON and NEW_YORK window UTC bounds reflect the 4h gap, not the usual 5h.
4. **in_session_mask across a transition:** build a minute UTC index spanning a London DST change and assert the mask flips at the correct UTC minute on both sides.
5. **fx_trading_day boundary:** assert a bar at 21:30 UTC in summer (17:30 ET) belongs to the next FX day, and the same wall-clock UTC in winter (16:30 ET) belongs to the current day.
6. **hour_of_week_anchored DST stability:** assert the bucket for 08:00 London-local is identical in summer and winter, while `hour_of_week_utc` for the same local event differs by one.
7. **Tokyo no-DST sanity:** assert the TOKYO window maps to 00:00-06:00 UTC on any date, summer or winter.
8. **Midnight-crossing window:** a synthetic window (e.g. 22:00-02:00 local) masks correctly across the local-midnight wrap.
9. **local_to_utc transition policy:** a non-existent spring-forward local time rolls forward; an ambiguous fall-back local time takes the first occurrence.

## 7. Files

- Create `src/backtesting/sessions/__init__.py`
- Create `src/backtesting/sessions/fx_clock.py` (exchange registry, session-window registry, `SessionWindow` dataclass, the nine functions above)
- Create `tests/backtesting/sessions/__init__.py`
- Create `tests/backtesting/sessions/test_fx_clock.py`

## 8. Out of scope (deferred to later intraday sub-projects)

- Loading 1-minute bars and computing actual session-window aggregates (the Asian-range high/low VALUES, session ranges, rolling vol curves) -> sub-project 2 (intraday loader).
- OCO / bracket / stop-entry order types, time-based cancel/exit, minute-bar cost model -> sub-project 3 (event-driven engine).
- The intraday strategies themselves (#20-25 and others) -> sub-project 4.
- Event/calendar filters (tier-1 releases, Gotobi days, month-end): the event registries partly exist already; wiring them is a later concern.
- Any change to `src/utils/timezone.py` (unrelated live-ops helper).
