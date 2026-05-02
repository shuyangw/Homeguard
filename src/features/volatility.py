"""Stateless realized-volatility estimators.

See package docstring in src.features.__init__.py for the stateless contract.
"""
import numpy as np
import pandas as pd


def close_to_close_rv(returns: pd.Series,
                      window: int,
                      annualization_factor: float = 252) -> pd.Series:
    """Close-to-close realized volatility:
        rolling_std(returns, window) * sqrt(annualization_factor)

    Inputs are returns (not prices). Uses min_periods=window. Annualization
    examples: 252 (daily), 252*390 (1-min US equities), 252*78 (5-min US
    equities), 365 (daily crypto).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    rolling_std = returns.rolling(window=window, min_periods=window).std()
    return rolling_std * np.sqrt(annualization_factor)
