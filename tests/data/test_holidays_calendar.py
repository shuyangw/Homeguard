from datetime import date
from src.data.feeds.holidays_calendar import holiday_set


def test_us_christmas_present():
    hs = holiday_set("US", range(2020, 2021))
    assert date(2020, 12, 25) in hs
