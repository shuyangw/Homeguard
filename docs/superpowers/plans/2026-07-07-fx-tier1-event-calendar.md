# Tier-1 EUR/GBP Event Calendar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a keyless, curated recurring-rule calendar of tier-1 EUR/GBP macro releases plus a window predicate, so intraday strategies (starting with #20) can skip event-driven days.

**Architecture:** A version-controlled rules yaml drives a pure module. A cadence expander turns each rule into dated occurrences over a range; a generator stamps each with a DST-correct UTC release time (via the FX session clock) and folds in the existing CB-decisions calendar; a predicate answers "is there a tier-1 EUR/GBP release in [window] on [day]?".

**Tech Stack:** Python 3.13 (conda env `fintech`), pandas, PyYAML, stdlib `calendar`/`datetime`, the merged `src.backtesting.sessions.fx_clock`, pytest. No new dependencies.

## Global Constraints

- Run Python via the `fintech` conda env. Test prefix: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest <path> -v` (conda already initialized in the shell).
- Keyless and pure: rules from the yaml only, no network/API. Deterministic functions.
- Release times are converted local->UTC via `src.backtesting.sessions.fx_clock.local_to_utc(tz, naive_dt)` (DST-correct); `tz` may be a raw IANA string (e.g. `Europe/Berlin`), which fx_clock accepts.
- Reuse `src.data.macro_calendar.load_cb_decisions() -> dict[str, list[date]]` (bank keys `ECB`, `BOE`) for `from_cb_decisions` rules; do not re-list those dates.
- Window semantics are half-open `[win_start, win_end)`, matching fx_clock.
- ASCII-only, no em dashes, no emojis, no `print()`. Pure module: no logging needed.
- Git hazard (macOS/Dropbox): use ONLY `git add <paths>`, `git commit`, `git log`. NEVER `git checkout`, bare `git status`/`git diff`, or `git reset`. Source/test/config files are not under docs/, so normal `git add` works.

---

### Task 1: Rules yaml, `load_tier1_rules`, and cadence expansion

**Files:**
- Create: `config/macro_calendar/eurgbp_tier1_releases.yaml`
- Create: `src/data/macro_calendar_tier1.py`
- Test: `tests/data/test_macro_calendar_tier1.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_tier1_rules() -> list[dict]` (the parsed rules).
  - `_expand_rule_dates(rule: dict, start: datetime.date, end: datetime.date) -> list[datetime.date]` (dated occurrences for one rule's cadence within the range; `from_cb_decisions` rules return `[]` here and are handled by the generator in Task 2).

- [ ] **Step 1: Write the rules yaml**

Create `config/macro_calendar/eurgbp_tier1_releases.yaml`:

```yaml
# Curated tier-1 EUR/GBP macro-release calendar (keyless, version-controlled).
# Recurring RULES, not scraped actuals: dates are approximate (a real release may
# shift a day or two). Consumers use this as a DAY-LEVEL skip filter, so approximate
# dates are acceptable. Release TIMES are structural per event and determine whether
# an event falls in a given session window. Cadence vocabulary understood by
# src/data/macro_calendar_tier1.py::_expand_rule_dates:
#   monthly:nth-weekday:<n>:<WD>        e.g. monthly:nth-weekday:3:WED
#   monthly:month-end-business-day
#   monthly:business-day:<n>
#   quarterly:nth-weekday:<n>:<WD>:<anchor-month 1-3>   (months where (month-anchor)%3==0)
rules:
  # ---- EUR ----
  - {name: EZ_FLASH_CPI,  currency: EUR, cadence: "monthly:month-end-business-day", time_local: "11:00", tz: "Europe/Berlin"}
  - {name: EZ_FLASH_PMI,  currency: EUR, cadence: "monthly:nth-weekday:4:WED",       time_local: "10:00", tz: "Europe/Berlin"}
  - {name: DE_IFO,        currency: EUR, cadence: "monthly:nth-weekday:4:MON",       time_local: "10:00", tz: "Europe/Berlin"}
  - {name: DE_ZEW,        currency: EUR, cadence: "monthly:nth-weekday:2:TUE",       time_local: "11:00", tz: "Europe/Berlin"}
  - {name: DE_FLASH_CPI,  currency: EUR, cadence: "monthly:month-end-business-day", time_local: "14:00", tz: "Europe/Berlin"}
  - {name: EZ_FLASH_GDP,  currency: EUR, cadence: "quarterly:nth-weekday:5:TUE:1",  time_local: "11:00", tz: "Europe/Berlin"}
  - {name: ECB_DECISION,  currency: EUR, from_cb_decisions: "ECB",                  time_local: "13:45", tz: "Europe/Berlin"}
  # ---- GBP ----
  - {name: UK_CPI,          currency: GBP, cadence: "monthly:nth-weekday:3:WED", time_local: "07:00", tz: "Europe/London"}
  - {name: UK_JOBS,         currency: GBP, cadence: "monthly:nth-weekday:2:TUE", time_local: "07:00", tz: "Europe/London"}
  - {name: UK_GDP,          currency: GBP, cadence: "monthly:nth-weekday:2:THU", time_local: "07:00", tz: "Europe/London"}
  - {name: UK_RETAIL_SALES, currency: GBP, cadence: "monthly:nth-weekday:3:FRI", time_local: "07:00", tz: "Europe/London"}
  - {name: UK_PMI,          currency: GBP, cadence: "monthly:business-day:1",    time_local: "09:30", tz: "Europe/London"}
  - {name: BOE_DECISION,    currency: GBP, from_cb_decisions: "BOE",             time_local: "12:00", tz: "Europe/London"}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/data/test_macro_calendar_tier1.py`:

```python
import datetime as dt

from src.data.macro_calendar_tier1 import load_tier1_rules, _expand_rule_dates


def test_load_tier1_rules_has_eur_and_gbp():
    rules = load_tier1_rules()
    names = {r["name"] for r in rules}
    assert {"EZ_FLASH_CPI", "UK_CPI", "ECB_DECISION", "BOE_DECISION"} <= names
    assert {r["currency"] for r in rules} == {"EUR", "GBP"}


def test_expand_nth_weekday():
    rule = {"cadence": "monthly:nth-weekday:3:WED"}
    dates = _expand_rule_dates(rule, dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    # 3rd Wednesday of Jan/Feb/Mar 2024
    assert dates == [dt.date(2024, 1, 17), dt.date(2024, 2, 21), dt.date(2024, 3, 20)]


def test_expand_month_end_business_day():
    rule = {"cadence": "monthly:month-end-business-day"}
    dates = _expand_rule_dates(rule, dt.date(2024, 3, 1), dt.date(2024, 3, 31))
    assert dates == [dt.date(2024, 3, 29)]  # 31st is Sun, 30th Sat -> Fri 29th


def test_expand_business_day_n():
    rule = {"cadence": "monthly:business-day:1"}
    dates = _expand_rule_dates(rule, dt.date(2024, 6, 1), dt.date(2024, 6, 30))
    assert dates == [dt.date(2024, 6, 3)]  # Jun 1 Sat, 2 Sun -> 1st biz day Mon 3rd


def test_expand_quarterly_only_anchor_months():
    rule = {"cadence": "quarterly:nth-weekday:5:TUE:1"}
    dates = _expand_rule_dates(rule, dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    # anchor month 1 -> Jan, Apr, Jul, Oct only
    assert [d.month for d in dates] == [1, 4, 7, 10]


def test_expand_from_cb_decisions_rule_returns_empty():
    rule = {"cadence": None, "from_cb_decisions": "ECB"}
    assert _expand_rule_dates(rule, dt.date(2024, 1, 1), dt.date(2024, 12, 31)) == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/data/test_macro_calendar_tier1.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.macro_calendar_tier1'`.

- [ ] **Step 4: Write the module (rules loader + cadence expander)**

Create `src/data/macro_calendar_tier1.py`:

```python
"""Tier-1 EUR/GBP macro-release calendar (keyless, curated recurring rules).

Expands the rules in config/macro_calendar/eurgbp_tier1_releases.yaml into dated,
UTC-stamped releases and answers whether a tier-1 EUR/GBP release falls in a given
session window on a day. Dates are approximate recurring-rule expansions (a real
release may shift a day or two); consumers use this as a day-level skip filter, so
that is acceptable. Release times are placed on the UTC timeline via the FX session
clock (DST-correct). ECB/BOE decision dates are reused from cb_decisions.yaml.
"""
from __future__ import annotations

import calendar
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

_PATH = Path(__file__).resolve().parents[2] / "config" / "macro_calendar" / "eurgbp_tier1_releases.yaml"
_WD = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def load_tier1_rules() -> list[dict]:
    with open(_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return list(raw.get("rules", []))


def _nth_weekday(year: int, month: int, n: int, wd: int) -> dt.date | None:
    first_wd = dt.date(year, month, 1).weekday()
    day = 1 + (wd - first_wd) % 7 + (n - 1) * 7
    if day > calendar.monthrange(year, month)[1]:
        return None
    return dt.date(year, month, day)


def _business_days(year: int, month: int) -> list[dt.date]:
    ndays = calendar.monthrange(year, month)[1]
    return [dt.date(year, month, d) for d in range(1, ndays + 1)
            if dt.date(year, month, d).weekday() < 5]


def _months_in_range(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _expand_rule_dates(rule: dict, start: dt.date, end: dt.date) -> list[dt.date]:
    if "from_cb_decisions" in rule:
        return []
    parts = str(rule["cadence"]).split(":")
    out: list[dt.date] = []
    for y, m in _months_in_range(start, end):
        d: dt.date | None = None
        if parts[:2] == ["monthly", "nth-weekday"]:
            d = _nth_weekday(y, m, int(parts[2]), _WD[parts[3]])
        elif parts[:2] == ["monthly", "month-end-business-day"]:
            d = _business_days(y, m)[-1]
        elif parts[:2] == ["monthly", "business-day"]:
            bd = _business_days(y, m)
            d = bd[int(parts[2]) - 1] if int(parts[2]) <= len(bd) else None
        elif parts[:2] == ["quarterly", "nth-weekday"]:
            if (m - int(parts[4])) % 3 != 0:
                continue
            d = _nth_weekday(y, m, int(parts[2]), _WD[parts[3]])
        else:
            raise ValueError(f"unknown cadence {rule['cadence']!r}")
        if d is not None and start <= d <= end:
            out.append(d)
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/data/test_macro_calendar_tier1.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add config/macro_calendar/eurgbp_tier1_releases.yaml src/data/macro_calendar_tier1.py tests/data/test_macro_calendar_tier1.py
git commit -m "feat(fx): tier-1 EUR/GBP release rules + cadence expander (sub-project 2a)"
```

---

### Task 2: `generate_tier1_releases`

**Files:**
- Modify: `src/data/macro_calendar_tier1.py`
- Test: `tests/data/test_macro_calendar_tier1.py`

**Interfaces:**
- Consumes: `load_tier1_rules`, `_expand_rule_dates` (Task 1); `fx_clock.local_to_utc`; `macro_calendar.load_cb_decisions`.
- Produces: `generate_tier1_releases(start: datetime.date, end: datetime.date) -> pd.DataFrame` with columns `date` (datetime.date), `name` (str), `currency` (str), `release_utc` (tz-aware UTC Timestamp), sorted by `release_utc`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_macro_calendar_tier1.py`:

```python
import pandas as pd
from src.data.macro_calendar_tier1 import generate_tier1_releases


def test_generate_has_columns_and_utc_and_sorted():
    df = generate_tier1_releases(dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    assert list(df.columns) == ["date", "name", "currency", "release_utc"]
    assert set(df["currency"]) == {"EUR", "GBP"}
    assert str(df["release_utc"].dt.tz) == "UTC"
    assert df["release_utc"].is_monotonic_increasing


def test_ez_flash_cpi_release_utc_is_dst_stable_in_london():
    # 11:00 Europe/Berlin -> 10:00 Europe/London year-round (constant 1h offset).
    df = generate_tier1_releases(dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    cpi = df[df["name"] == "EZ_FLASH_CPI"].copy()
    london = cpi["release_utc"].dt.tz_convert("Europe/London")
    assert set(london.dt.hour) == {10}  # always 10:00 London


def test_from_cb_decisions_ecb_dates_present():
    from src.data.macro_calendar import load_cb_decisions
    ecb = [d for d in load_cb_decisions().get("ECB", []) if d.year == 2025]
    if not ecb:
        return  # cb_decisions has no 2025 ECB dates in this environment
    df = generate_tier1_releases(dt.date(2025, 1, 1), dt.date(2025, 12, 31))
    ecb_rows = df[df["name"] == "ECB_DECISION"]
    assert set(ecb_rows["date"]) >= set(ecb)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/data/test_macro_calendar_tier1.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_tier1_releases'`.

- [ ] **Step 3: Add the implementation**

Append to `src/data/macro_calendar_tier1.py`:

```python
def generate_tier1_releases(start: dt.date, end: dt.date) -> pd.DataFrame:
    from src.backtesting.sessions.fx_clock import local_to_utc
    from src.data.macro_calendar import load_cb_decisions

    cb = load_cb_decisions()
    cols = ["date", "name", "currency", "release_utc"]
    rows: list[dict] = []
    for rule in load_tier1_rules():
        if "from_cb_decisions" in rule:
            dates = [d for d in cb.get(rule["from_cb_decisions"], []) if start <= d <= end]
        else:
            dates = _expand_rule_dates(rule, start, end)
        overrides = rule.get("overrides", {}) or {}
        for d in dates:
            t_local = overrides.get(d.isoformat(), rule["time_local"])
            hh, mm = (int(x) for x in str(t_local).split(":"))
            rel = local_to_utc(rule["tz"], dt.datetime(d.year, d.month, d.day, hh, mm))
            rows.append({"date": d, "name": rule["name"],
                         "currency": rule["currency"], "release_utc": rel})
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return pd.DataFrame(rows, columns=cols).sort_values("release_utc").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/data/test_macro_calendar_tier1.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/macro_calendar_tier1.py tests/data/test_macro_calendar_tier1.py
git commit -m "feat(fx): generate_tier1_releases (rules -> DST-correct UTC calendar)"
```

---

### Task 3: `tier1_release_in_window` predicate

**Files:**
- Modify: `src/data/macro_calendar_tier1.py`
- Test: `tests/data/test_macro_calendar_tier1.py`

**Interfaces:**
- Consumes: `generate_tier1_releases` (Task 2); `fx_clock.EXCHANGE_TZ`.
- Produces: `tier1_release_in_window(day: datetime.date, win_start: datetime.time, win_end: datetime.time, exchange: str = "LONDON", currencies: tuple[str, ...] = ("EUR", "GBP"), releases: pd.DataFrame | None = None) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_macro_calendar_tier1.py`:

```python
from src.data.macro_calendar_tier1 import tier1_release_in_window


def _releases_for(name, day, time_local, tz, currency):
    # Build a one-row releases frame for isolated predicate testing.
    from src.backtesting.sessions.fx_clock import local_to_utc
    hh, mm = (int(x) for x in time_local.split(":"))
    rel = local_to_utc(tz, dt.datetime(day.year, day.month, day.day, hh, mm))
    return pd.DataFrame([{"date": day, "name": name, "currency": currency, "release_utc": rel}])


def test_ez_release_inside_0930_1200_window():
    day = dt.date(2024, 2, 15)
    rel = _releases_for("EZ_FLASH_CPI", day, "11:00", "Europe/Berlin", "EUR")  # 10:00 London
    assert tier1_release_in_window(day, dt.time(9, 30), dt.time(12, 0), releases=rel) is True


def test_uk_0700_release_outside_window():
    day = dt.date(2024, 2, 15)
    rel = _releases_for("UK_CPI", day, "07:00", "Europe/London", "GBP")  # 07:00 London
    assert tier1_release_in_window(day, dt.time(9, 30), dt.time(12, 0), releases=rel) is False


def test_uk_pmi_0930_is_inside_half_open_lower_edge():
    day = dt.date(2024, 2, 15)
    rel = _releases_for("UK_PMI", day, "09:30", "Europe/London", "GBP")
    assert tier1_release_in_window(day, dt.time(9, 30), dt.time(12, 0), releases=rel) is True


def test_currency_filter_excludes_other_ccy():
    day = dt.date(2024, 2, 15)
    rel = _releases_for("EZ_FLASH_CPI", day, "11:00", "Europe/Berlin", "EUR")
    assert tier1_release_in_window(day, dt.time(9, 30), dt.time(12, 0),
                                   currencies=("GBP",), releases=rel) is False


def test_dst_stable_true_in_summer_and_winter():
    # EZ 11:00 Berlin -> 10:00 London in both seasons -> inside window both times.
    for day in (dt.date(2024, 1, 15), dt.date(2024, 7, 15)):
        rel = _releases_for("EZ_FLASH_CPI", day, "11:00", "Europe/Berlin", "EUR")
        assert tier1_release_in_window(day, dt.time(9, 30), dt.time(12, 0), releases=rel) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/data/test_macro_calendar_tier1.py -v`
Expected: FAIL with `ImportError: cannot import name 'tier1_release_in_window'`.

- [ ] **Step 3: Add the implementation**

Append to `src/data/macro_calendar_tier1.py`:

```python
def tier1_release_in_window(day: dt.date, win_start: dt.time, win_end: dt.time,
                            exchange: str = "LONDON",
                            currencies: tuple[str, ...] = ("EUR", "GBP"),
                            releases: "pd.DataFrame | None" = None) -> bool:
    from zoneinfo import ZoneInfo
    from src.backtesting.sessions.fx_clock import EXCHANGE_TZ

    if releases is None:
        releases = generate_tier1_releases(day, day)
    if releases.empty:
        return False
    sub = releases[(releases["currency"].isin(currencies)) & (releases["date"] == day)]
    if sub.empty:
        return False
    tz = EXCHANGE_TZ.get(exchange) or ZoneInfo(exchange)
    local = sub["release_utc"].dt.tz_convert(tz)
    sod = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
    s = win_start.hour * 3600 + win_start.minute * 60 + win_start.second
    e = win_end.hour * 3600 + win_end.minute * 60 + win_end.second
    return bool(((sod >= s) & (sod < e)).any())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/data/test_macro_calendar_tier1.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/macro_calendar_tier1.py tests/data/test_macro_calendar_tier1.py
git commit -m "feat(fx): tier1_release_in_window predicate (the #20 event-skip filter)"
```

---

## Post-implementation (orchestrator, after all tasks)

- Confirm the full module test suite passes (14 tests), pure/no-print/no-em-dash.
- This completes intraday sub-project 2a. Next: sub-project 2b (the #20 London Open Breakout vertical slice) consumes `tier1_release_in_window` for its event-skip filter, plus the fx session clock and a minimal intraday loader + OCO-bracket engine.
