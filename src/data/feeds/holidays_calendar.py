"""Holiday calendar feed using the holidays library (keyless).

EU uses Germany (DE) as the euro-area proxy for TARGET-adjacent holidays,
as Germany's market calendar aligns with the TARGET2 settlement system.
"""
from __future__ import annotations
from datetime import date
import holidays as _holidays

REQUIRES_KEY = None
_COUNTRY_CODE = {"US": "US", "UK": "GB", "JP": "JP", "EU": "DE", "AU": "AU"}
COUNTRIES = set(_COUNTRY_CODE)


def holiday_set(country: str, years: range) -> set[date]:
    """Return a set of holiday dates for the given country and years.

    Args:
        country: Country code (one of COUNTRIES)
        years: Range of years (e.g., range(2020, 2021))

    Returns:
        Set of datetime.date objects representing holidays
    """
    code = _COUNTRY_CODE[country]
    cal = _holidays.country_holidays(code, years=list(years))
    return set(cal.keys())
