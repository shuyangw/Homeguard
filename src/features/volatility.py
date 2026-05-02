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


def parkinson_rv(ohlc_df: pd.DataFrame,
                 window: int,
                 annualization_factor: float = 252) -> pd.Series:
    """Parkinson range-based volatility from high/low prices.

    Assumes zero drift. Requires lowercase 'high' and 'low' columns.
    Formula: sqrt( (1 / (4 * ln(2))) * mean(ln(H/L)^2) * annualization )
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    high = ohlc_df['high']
    low = ohlc_df['low']
    log_hl_sq = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    sq_var = factor * log_hl_sq.rolling(window=window, min_periods=window).mean()
    return np.sqrt(sq_var * annualization_factor)
