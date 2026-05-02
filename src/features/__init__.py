"""
src.features -- stateless data processing primitives.

All functions in this package are pure: same input -> same output, no side
effects, no global state, no I/O, no logging beyond raised exceptions for
schema violations. Inputs are pandas Series or DataFrame; outputs are pandas
Series or DataFrame with the input's index preserved. NaN propagates;
insufficient-data and zero-variance windows produce NaN, not exceptions.

Public API is flat at the package level. The canonical import form is:

    from src.features import close_to_close_rv, robust_zscore_rolling, ...

Submodule imports (from src.features.normalizers import ..., from
src.features.volatility import ...) remain valid.
"""

from src.features.normalizers import (
    log_transform,
    log_returns,
    zscore_rolling,
    robust_zscore_rolling,
    robust_zscore_cross_sectional,
)

__all__ = [
    'log_transform',
    'log_returns',
    'zscore_rolling',
    'robust_zscore_rolling',
    'robust_zscore_cross_sectional',
]
