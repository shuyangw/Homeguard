"""Generate the EIA Weekly Petroleum Status Report release calendar.

EIA releases Wednesday 10:30 ET. When a US federal holiday falls on the
Monday, Tuesday, or Wednesday of the release week, the release shifts to
Thursday. This produces the release DATE set (times/intraday handling belong
to the SP-B consumer #41)."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml
from pandas.tseries.holiday import USFederalHolidayCalendar

_CAL_DIR = Path(__file__).resolve().parents[2] / "config" / "macro_calendar"


def eia_release_dates(start_year: int, end_year: int) -> list[date]:
    cal = USFederalHolidayCalendar()
    hols = {d.date() for d in cal.holidays(start=f"{start_year}-01-01", end=f"{end_year}-12-31")}
    out: list[date] = []
    for wed in pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="W-WED"):
        wed_d = wed.date()
        week_days = {wed_d - timedelta(days=2), wed_d - timedelta(days=1), wed_d}  # Mon/Tue/Wed
        shift = bool(week_days & hols)
        out.append(wed_d + timedelta(days=1) if shift else wed_d)
    return out


def write_eia_yaml(start_year: int = 2010, end_year: int = 2026) -> Path:
    dates = eia_release_dates(start_year, end_year)
    _CAL_DIR.mkdir(parents=True, exist_ok=True)
    path = _CAL_DIR / "eia.yaml"
    path.write_text(yaml.safe_dump({
        "event_type": "eia",
        "description": "EIA Weekly Petroleum Status Report release dates (Wed, shift to Thu on holiday weeks)",
        "dates": [d.isoformat() for d in dates],
    }, sort_keys=False))
    return path


if __name__ == "__main__":
    print(write_eia_yaml())
