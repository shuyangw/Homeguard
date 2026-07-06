from datetime import date

import pandas as pd
import polars as pl

from scripts.data.build_fx_daily_cache import build_fx_daily_cache, resample_fx_minute_to_daily


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


def _write_source_minute_parquet(src_root, pair, year, month):
    ts = pd.to_datetime([
        "2020-06-01 18:00:00+00:00", "2020-06-01 19:00:00+00:00",
        "2020-06-01 20:00:00+00:00",
    ], utc=True)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [1.10, 1.11, 1.09],
        "high": [1.12, 1.13, 1.10],
        "low": [1.08, 1.10, 1.05],
        "close": [1.11, 1.09, 1.06],
        "volume": [100.0, 110.0, 90.0],
        "trade_count": [10, 11, 9],
        "vwap": [1.105, 1.10, 1.075],
    })
    dst = src_root / f"symbol={pair}" / f"year={year}" / f"month={month}"
    dst.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(df).write_parquet(dst / "data.parquet")


def test_build_fx_daily_cache_writes_aggregated_ohlc(tmp_path):
    src_root = tmp_path / "fx_1min"
    out_root = tmp_path / "fx_daily"
    _write_source_minute_parquet(src_root, "EURUSD", 2020, 6)

    written = build_fx_daily_cache(
        ["EURUSD"], date(2020, 6, 1), date(2020, 6, 2),
        src_root=src_root, out_root=out_root)

    assert written == ["EURUSD"]
    out_file = out_root / "symbol=EURUSD" / "year=2020" / "month=6" / "data.parquet"
    assert out_file.exists()

    out_df = pl.read_parquet(out_file).to_pandas()
    assert list(out_df.columns) == ["fx_date", "open", "high", "low", "close"]
    row = out_df.iloc[0]
    assert row["open"] == 1.10        # first
    assert row["high"] == 1.13        # max
    assert row["low"] == 1.05         # min
    assert row["close"] == 1.06       # last
