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
            "close": [1.10, 1.11, 1.12],
        }
    )
    daily = resample_fx_minute_to_daily(df)
    import datetime as dt

    assert daily.loc[dt.date(2024, 6, 3), "close"] == 1.11  # last before 17:00 ET
    assert daily.loc[dt.date(2024, 6, 4), "close"] == 1.12  # rolled into Tuesday
