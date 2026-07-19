import datetime as dt

from src.backtesting.data.fx_intraday_loader import load_fx_1min


def test_load_fx_1min_real_data_is_utc_and_sorted():
    # GBPUSD 1m data is on disk for 2011-2026; load one short window.
    bars = load_fx_1min("GBPUSD", dt.date(2020, 6, 1), dt.date(2020, 6, 5))
    assert not bars.empty
    assert str(bars.index.tz) == "UTC"
    assert bars.index.is_monotonic_increasing
    assert not bars.index.has_duplicates
    assert list(bars.columns[:4]) == ["open", "high", "low", "close"]
