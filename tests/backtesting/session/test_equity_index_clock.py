from datetime import date, time
import pandas as pd
from src.backtesting.sessions.equity_index_clock import (
    et_to_utc, RTH_CLOSE, RTH_OPEN, SLICE_START)


def test_et_to_utc_summer_edt():
    # 2015-06-01 16:00 ET (EDT, UTC-4) -> 20:00 UTC
    assert et_to_utc(date(2015, 6, 1), RTH_CLOSE) == pd.Timestamp("2015-06-01 20:00", tz="UTC")


def test_et_to_utc_winter_est():
    # 2015-01-05 16:00 ET (EST, UTC-5) -> 21:00 UTC
    assert et_to_utc(date(2015, 1, 5), RTH_CLOSE) == pd.Timestamp("2015-01-05 21:00", tz="UTC")


def test_et_to_utc_open_and_slice_times():
    # 09:30 EDT -> 13:30 UTC; 02:00 EDT -> 06:00 UTC
    assert et_to_utc(date(2015, 6, 1), RTH_OPEN) == pd.Timestamp("2015-06-01 13:30", tz="UTC")
    assert et_to_utc(date(2015, 6, 1), SLICE_START) == pd.Timestamp("2015-06-01 06:00", tz="UTC")
