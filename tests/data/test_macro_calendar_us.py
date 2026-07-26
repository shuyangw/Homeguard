import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.macro_calendar_us import (all_release_timestamps, release_dates,
                                        release_timestamps)

_FX_1MIN = Path("/Users/shuyangw/Library/CloudStorage/Dropbox/Stock_Data/fx_1min"
                "/symbol=EURUSD")


def test_both_calendars_cover_the_campaign_window():
    for event in ("cpi", "nfp"):
        dates = release_dates(event)
        assert len(dates) > 150
        assert dates[0].year == 2011 and dates[-1].year >= 2026
        assert dates == sorted(dates)


def test_one_release_per_month():
    for event in ("cpi", "nfp"):
        months = [(d.year, d.month) for d in release_dates(event)]
        assert len(months) == len(set(months))


def test_release_time_is_0830_new_york_with_correct_dst():
    """08:30 ET is 13:30 UTC on standard time and 12:30 UTC on daylight time."""
    winter = release_timestamps("nfp", dt.date(2024, 1, 1), dt.date(2024, 2, 1))
    summer = release_timestamps("nfp", dt.date(2024, 7, 1), dt.date(2024, 8, 1))
    assert winter[0].hour == 13 and winter[0].minute == 30
    assert summer[0].hour == 12 and summer[0].minute == 30


def test_range_filter_is_half_open():
    d = release_dates("cpi")[0]
    assert len(release_timestamps("cpi", d, d)) == 0
    assert len(release_timestamps("cpi", d, d + dt.timedelta(days=1))) == 1


def test_nfp_never_lands_on_a_weekend():
    assert all(d.weekday() < 5 for d in release_dates("nfp"))


def test_february_cpi_takes_the_later_date_not_the_revision():
    """The annual seasonal-adjustment revision precedes the main release by 2 days."""
    feb_2011 = [d for d in release_dates("cpi")
                if d.year == 2011 and d.month == 2]
    assert feb_2011 == [dt.date(2011, 2, 17)]


def test_supersedes_the_rule_generated_proxy():
    """Documents why this module exists: the old cpi.yaml was a 10th-of-month proxy."""
    import yaml
    old = yaml.safe_load(open("config/macro_calendar/cpi.yaml"))
    old_by_month = {}
    for s in old["dates"]:
        d = dt.date.fromisoformat(s)
        old_by_month.setdefault((d.year, d.month), d)
    new = {(d.year, d.month): d for d in release_dates("cpi")}
    common = set(old_by_month) & set(new)
    exact = sum(1 for k in common if old_by_month[k] == new[k])
    assert exact / len(common) < 0.25       # proxy agreed with reality ~14% of months


def test_unknown_event_type_raises():
    with pytest.raises(ValueError):
        release_timestamps("gdp")


def test_all_releases_are_sorted_and_labelled():
    df = all_release_timestamps(dt.date(2024, 1, 1), dt.date(2025, 1, 1))
    assert set(df["event_type"]) == {"cpi", "nfp", "fomc"}
    assert df["timestamp_utc"].is_monotonic_increasing


def test_fomc_starts_at_2013_and_fires_at_1400_et():
    """2011-2012 is deliberately absent: the statement time alternated by meeting
    type in that era and no press-conference flag is available."""
    dates = release_dates("fomc")
    assert min(dates).year == 2013
    # Windows chosen to sit strictly inside one US DST regime: the March and
    # November meetings can fall either side of a switch.
    winter = release_timestamps("fomc", dt.date(2024, 1, 1), dt.date(2024, 3, 1))
    summer = release_timestamps("fomc", dt.date(2024, 6, 1), dt.date(2024, 9, 1))
    assert len(winter) and len(summer)
    assert all(ts.hour == 19 and ts.minute == 0 for ts in winter)
    assert all(ts.hour == 18 and ts.minute == 0 for ts in summer)


@pytest.mark.skipif(not _FX_1MIN.exists(), reason="local 1m FX data not present")
@pytest.mark.parametrize("event", ["nfp", "cpi", "fomc"])
def test_every_calendar_lands_on_its_volatility_spike(event):
    """The load-bearing check: these instants must be where the market moves.

    A calendar off by a day or an hour still produces trades, silently, on
    non-events. Rather than trust the 08:30 / 14:00 ET conventions, confirm
    them: the release minute must dominate the day's typical minute.
    """
    stamps = release_timestamps(event, dt.date(2018, 1, 1), dt.date(2024, 1, 1))
    at_release, background = [], []
    for ts in stamps:
        f = _FX_1MIN / f"year={ts.year}" / f"month={ts.month}" / "data.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["timestamp", "close"]).set_index("timestamp")
        day = df[df.index.date == ts.date()]
        if len(day) < 500:
            continue
        ret = np.abs(np.log(day["close"]).diff())
        window = ret.loc[ts:ts + pd.Timedelta(minutes=1)]
        if window.empty or not np.isfinite(window.iloc[0]):
            continue
        at_release.append(window.iloc[0])
        background.append(ret.median())
    assert len(at_release) > 30, f"{event}: too few usable release days"
    assert np.mean(at_release) / np.mean(background) > 5.0
