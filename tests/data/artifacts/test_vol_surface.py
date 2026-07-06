import numpy as np
import pandas as pd
from src.data.artifacts.vol_surface import build_surface, hour_of_week


def test_surface_has_168_rows():
    ts = pd.date_range("2020-01-06", periods=168 * 3, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "close": 1.0 + np.arange(len(ts)) * 1e-4})
    surf = build_surface(df)
    assert len(surf) == 168
    assert set(surf.columns) >= {"hour_of_week", "median_abs_ret", "mad"}


def test_surface_sparse_coverage_fills_zero():
    # 2020-01-06 is a Monday -> hour_of_week 0 for 00:00 and 1 for 01:00.
    ts = pd.to_datetime(
        [
            "2020-01-06 00:00",
            "2020-01-06 00:01",
            "2020-01-06 00:02",
            "2020-01-13 00:00",
            "2020-01-13 00:01",
        ],
        utc=True,
    )
    df = pd.DataFrame({"timestamp": ts, "close": [1.0, 1.001, 0.999, 1.002, 0.998]})
    surf = build_surface(df)
    assert len(surf) == 168

    covered = surf[surf["hour_of_week"] == 0].iloc[0]
    assert covered["median_abs_ret"] >= 0

    uncovered = surf[surf["hour_of_week"] == 50].iloc[0]
    assert uncovered["median_abs_ret"] == 0.0
    assert uncovered["mad"] == 0.0


def test_hour_of_week_known_timestamp():
    # 2020-01-08 is a Wednesday (dayofweek == 2).
    ts = pd.Timestamp("2020-01-08 03:00", tz="UTC")
    assert hour_of_week(ts) == 2 * 24 + 3 == 51
