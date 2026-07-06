import pandas as pd
from scripts.data.build_fx_daily_cache import resample_fx_minute_to_daily


def _ts(*parts):
    # UTC timestamp helper
    return pd.Timestamp(*parts, tz="UTC")


def test_1700_et_boundary_splits_days():
    # 2024-06-03 is a Monday. 17:00 ET = 21:00 UTC (EDT, UTC-4).
    # A bar at 20:59 UTC (16:59 ET Mon) belongs to FX-day Monday.
    # A bar at 21:01 UTC (17:01 ET Mon) belongs to FX-day Tuesday.
    df = pd.DataFrame(
        {
            "timestamp": [
                _ts("2024-06-03 20:58"),
                _ts("2024-06-03 20:59"),
                _ts("2024-06-03 21:01"),
            ],
            "open": [1.10, 1.11, 1.12],
            "high": [1.10, 1.11, 1.12],
            "low": [1.10, 1.11, 1.12],
            "close": [1.10, 1.11, 1.12],
        }
    )
    daily = resample_fx_minute_to_daily(df)
    import datetime as dt

    assert daily.loc[dt.date(2024, 6, 3), "close"] == 1.11  # last before 17:00 ET
    assert daily.loc[dt.date(2024, 6, 4), "close"] == 1.12  # rolled into Tuesday


def test_dst_fallback_weekend_still_splits_days_correctly():
    # US DST fall-back in 2024 happened 2024-11-03 02:00 ET (inside the
    # Fri-17:00-ET to Sun-17:00-ET market closure, so no real bar ever sees it).
    # Fri 2024-11-01 16:59 ET (still EDT, UTC-4) = 20:59 UTC.
    # Mon 2024-11-04 09:00 ET (now EST, UTC-5) = 14:00 UTC.
    df = pd.DataFrame(
        {
            "timestamp": [
                _ts("2024-11-01 20:59"),
                _ts("2024-11-04 14:00"),
            ],
            "open": [1.20, 1.21],
            "high": [1.20, 1.21],
            "low": [1.20, 1.21],
            "close": [1.20, 1.21],
        }
    )
    daily = resample_fx_minute_to_daily(df)
    import datetime as dt

    assert list(daily.index) == [dt.date(2024, 11, 1), dt.date(2024, 11, 4)]
    assert daily.loc[dt.date(2024, 11, 1), "close"] == 1.20
    assert daily.loc[dt.date(2024, 11, 4), "close"] == 1.21
