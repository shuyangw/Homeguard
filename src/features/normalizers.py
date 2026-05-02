"""Stateless normalizer primitives.

See package docstring in src.features.__init__.py for the stateless contract.
"""
import numpy as np
import pandas as pd


def log_transform(series: pd.Series) -> pd.Series:
    """Natural log of series. Non-positive values produce NaN."""
    arr = series.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan)
    mask = arr > 0
    out[mask] = np.log(arr[mask])
    return pd.Series(out, index=series.index)
