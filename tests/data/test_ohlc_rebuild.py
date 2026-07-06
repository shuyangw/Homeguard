import pandas as pd
from scripts.data.build_fx_daily_cache import resample_fx_minute_to_daily


def _minute_df():
    ts = pd.to_datetime([
        "2020-06-01 18:00:00+00:00", "2020-06-01 19:00:00+00:00",
        "2020-06-01 20:00:00+00:00",
    ], utc=True)
    return pd.DataFrame({
        "timestamp": ts,
        "open": [1.10, 1.11, 1.09],
        "high": [1.12, 1.13, 1.10],
        "low": [1.08, 1.10, 1.05],
        "close": [1.11, 1.09, 1.06],
    })


def test_resample_carries_ohlc():
    out = resample_fx_minute_to_daily(_minute_df())
    row = out.iloc[0]
    assert row["open"] == 1.10       # first
    assert row["high"] == 1.13       # max
    assert row["low"] == 1.05        # min
    assert row["close"] == 1.06      # last
