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
