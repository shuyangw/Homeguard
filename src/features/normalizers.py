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


def zscore_rolling(series: pd.Series, window: int) -> pd.Series:
    """Standard rolling z-score: (x - rolling_mean) / rolling_std.

    Provided for legacy migration only; new code should use
    robust_zscore_rolling. Zero-variance windows produce NaN (no epsilon
    hack).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    rolling_std_safe = rolling_std.where(rolling_std > 0)
    return (series - rolling_mean) / rolling_std_safe


def _mad_zscore_window(window_values: np.ndarray) -> float:
    """Compute robust z-score for the most recent value of a window."""
    finite = window_values[~np.isnan(window_values)]
    if len(finite) == 0:
        return np.nan
    last = window_values[-1]
    if np.isnan(last):
        return np.nan
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    if mad == 0:
        return np.nan
    return (last - med) / (1.4826 * mad)


def robust_zscore_rolling(series: pd.Series, window: int) -> pd.Series:
    """MAD-based rolling z-score: (x - rolling_median) / (1.4826 * rolling_MAD).

    The 1.4826 factor scales MAD to be a consistent estimator of sigma under
    normal data. Zero-MAD windows produce NaN. Default for new code.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return series.rolling(window=window, min_periods=window).apply(
        _mad_zscore_window, raw=True
    )


def robust_zscore_cross_sectional(series: pd.Series) -> pd.Series:
    """MAD-based cross-sectional z-score across the index of a single
    timepoint's values.

    Operates over the index (e.g., across symbols), not over time. Zero-MAD
    inputs produce all NaN.
    """
    arr = series.to_numpy(dtype=float)
    finite = arr[~np.isnan(arr)]
    if len(finite) == 0:
        return pd.Series(arr, index=series.index)
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    if mad == 0:
        return pd.Series(np.full_like(arr, np.nan), index=series.index)
    return (series - med) / (1.4826 * mad)


def winsorize(series: pd.Series,
              lower_q: float = 0.01,
              upper_q: float = 0.99) -> pd.Series:
    """Clip values outside [lower_q, upper_q] quantiles to the quantile bounds.

    Quantiles are computed on non-NaN values; NaNs are preserved.
    """
    if not (0.0 <= lower_q <= 1.0):
        raise ValueError(f"lower_q must be in [0, 1], got {lower_q}")
    if not (0.0 <= upper_q <= 1.0):
        raise ValueError(f"upper_q must be in [0, 1], got {upper_q}")
    if lower_q >= upper_q:
        raise ValueError(f"lower_q ({lower_q}) must be < upper_q ({upper_q})")
    lower_bound = series.quantile(lower_q)
    upper_bound = series.quantile(upper_q)
    return series.clip(lower=lower_bound, upper=upper_bound)


def rank_transform(series: pd.Series, method: str = 'average') -> pd.Series:
    """Rank values to [0, 1] uniform. Wraps pandas .rank(pct=True).

    NaNs preserved. method: 'average' | 'min' | 'max' | 'first' | 'dense'.
    """
    return series.rank(method=method, pct=True)
