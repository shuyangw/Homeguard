import datetime as dt

import pandas as pd
import pytest

from src.backtesting.sessions.fx_clock import (
    EXCHANGE_TZ, SESSION_WINDOWS, SessionWindow, local_to_utc, session_window_utc, in_session_mask)


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
