"""One-shot generator for config/macro_calendar/{fomc,nfp,cpi}.yaml.

Run when the macro calendar needs refreshing (e.g., new Fed schedule released).
The output YAMLs are committed to the repo; this script is the source of truth
for how they were generated.

Usage:
    python scripts/data/generate_macro_calendar_yamls.py
"""
from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path

import yaml

CALENDAR_DIR = Path(__file__).resolve().parents[2] / "config" / "macro_calendar"

# FOMC meeting dates, 2010-2026.
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm (historical)
# Schedule for 2024+ from Fed's annual announcement.
# 2010-2019 dates compiled from FOMC historical archives.
FOMC_DATES_HISTORICAL = [
    # 2010 (8 meetings)
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23",
    "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (includes emergency cuts)
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (scheduled per fed announcement)
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]


def compute_nfp_dates(start_year: int, end_year: int) -> list[date]:
    """NFP releases on the first Friday of each month (deterministic)."""
    out: list[date] = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            cal = calendar.monthcalendar(y, m)
            first_friday = next(
                week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY]
            )
            out.append(date(y, m, first_friday))
    return out


def compute_cpi_proxy_dates(start_year: int, end_year: int) -> list[date]:
    """CPI release proxy: 10th of each month.

    Real BLS release dates vary by 1-3 days; consumers should treat this as
    an approximate signal-window center, not an exact release timestamp.
    """
    out: list[date] = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            out.append(date(y, m, 10))
    return out


def _write_yaml(path: Path, event_type: str, description: str,
                dates: list[date]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_type": event_type,
        "description": description,
        "dates": [d.isoformat() for d in dates],
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


def main() -> int:
    fomc = [date.fromisoformat(s) for s in FOMC_DATES_HISTORICAL]
    nfp = compute_nfp_dates(2010, 2026)
    cpi = compute_cpi_proxy_dates(2010, 2026)

    _write_yaml(
        CALENDAR_DIR / "fomc.yaml", "fomc",
        "FOMC meeting dates 2010-2026 from federalreserve.gov", fomc,
    )
    _write_yaml(
        CALENDAR_DIR / "nfp.yaml", "nfp",
        "NFP release dates 2010-2026 (first Friday of each month)", nfp,
    )
    _write_yaml(
        CALENDAR_DIR / "cpi.yaml", "cpi",
        "CPI release proxy dates 2010-2026 (10th of each month). "
        "Real BLS release varies by 1-3 days; treat as approximate "
        "signal-window center, not exact timestamp.",
        cpi,
    )
    print(f"Wrote {CALENDAR_DIR}/fomc.yaml ({len(fomc)} dates)")
    print(f"Wrote {CALENDAR_DIR}/nfp.yaml ({len(nfp)} dates)")
    print(f"Wrote {CALENDAR_DIR}/cpi.yaml ({len(cpi)} dates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
