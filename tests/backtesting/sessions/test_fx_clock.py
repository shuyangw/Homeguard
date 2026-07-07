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
