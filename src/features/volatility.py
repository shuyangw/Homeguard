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


def garman_klass_rv(ohlc_df: pd.DataFrame,
                    window: int,
                    annualization_factor: float = 252) -> pd.Series:
    """Garman-Klass volatility from OHLC prices.

    More efficient than Parkinson under zero drift. Requires lowercase
    'open', 'high', 'low', 'close' columns.
    Formula:
      sigma^2 = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
      Annualized vol = sqrt(rolling_mean(sigma^2) * annualization)
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    open_ = ohlc_df['open']
    high = ohlc_df['high']
    low = ohlc_df['low']
    close = ohlc_df['close']
    log_hl_sq = np.log(high / low) ** 2
    log_co_sq = np.log(close / open_) ** 2
    sigma_sq = 0.5 * log_hl_sq - (2.0 * np.log(2.0) - 1.0) * log_co_sq
    rolling = sigma_sq.rolling(window=window, min_periods=window).mean()
    return np.sqrt(rolling * annualization_factor)
