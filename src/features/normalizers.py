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


def log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Log returns: log(p_t / p_{t-periods}). Time-additive.

    First `periods` values are NaN.
    """
    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    log_prices = log_transform(prices)
    return log_prices - log_prices.shift(periods)
