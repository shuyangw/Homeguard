import numpy as np
import pandas as pd
from src.data.artifacts.vol_surface import build_surface


def test_surface_has_168_rows():
    ts = pd.date_range("2020-01-06", periods=168 * 3, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "close": 1.0 + np.arange(len(ts)) * 1e-4})
    surf = build_surface(df)
    assert len(surf) == 168
    assert set(surf.columns) >= {"hour_of_week", "median_abs_ret", "mad"}
