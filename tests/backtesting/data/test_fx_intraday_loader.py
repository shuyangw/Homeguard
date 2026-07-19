import datetime as dt

import numpy as np
import pandas as pd

from src.backtesting.data.fx_intraday_loader import load_fx_1min, resample_ohlc


def test_resample_ohlc_15min_aggregates_correctly():
    idx = pd.date_range("2024-01-02 08:00", periods=30, freq="1min", tz="UTC")
    bars = pd.DataFrame({
        "open": np.arange(30, dtype=float), "high": np.arange(30, dtype=float) + 1.0,
        "low": np.arange(30, dtype=float) - 1.0, "close": np.arange(30, dtype=float) + 0.5,
        "volume": np.ones(30)}, index=idx)
    out = resample_ohlc(bars, "15min")
    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == 0.0 and first["high"] == 14.0 + 1.0
    assert first["low"] == 0.0 - 1.0 and first["close"] == 14.0 + 0.5


def test_load_fx_1min_real_data_is_utc_and_sorted():
    # GBPUSD 1m data is on disk for 2011-2026; load one short window.
    bars = load_fx_1min("GBPUSD", dt.date(2020, 6, 1), dt.date(2020, 6, 5))
    assert not bars.empty
    assert str(bars.index.tz) == "UTC"
    assert bars.index.is_monotonic_increasing
    assert not bars.index.has_duplicates
    assert list(bars.columns[:4]) == ["open", "high", "low", "close"]
