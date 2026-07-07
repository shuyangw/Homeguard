"""S&P 500 benchmark helpers for the relative success criterion.

The FX strategy passes only if its OOS Sharpe beats the S&P's over the SAME OOS
dates. These helpers load the cached SPX daily series (from the keyless
equity_index_yfinance plugin) and compute Sharpe / correlation / information
ratio over an arbitrary date index. All accept an injected `sp_returns` so tests
run without I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.settings import get_local_storage_dir

_TRADING_DAYS = 252


def load_sp500_daily_returns() -> pd.Series:
    fp = get_local_storage_dir() / "alt_data" / "equity_index" / "SPX" / "daily.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            f"S&P benchmark parquet missing at {fp}; populate it via "
            f"src.data.acquisition.plugins.equity_index_yfinance.fetch_index('SPX', ...)")
    df = pd.read_parquet(fp)
    s = pd.Series(df["close"].values,
                  index=pd.to_datetime(df["date"].values)).sort_index()
    return s.pct_change().dropna()


def _annualized_sharpe(returns: pd.Series) -> float:
    if returns.size < 2:
        return float("nan")
    std = float(returns.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return float("nan")
    return float(returns.mean()) / std * np.sqrt(_TRADING_DAYS)


def sp500_sharpe_over_dates(dates, sp_returns=None) -> float:
    if sp_returns is None:
        sp_returns = load_sp500_daily_returns()
    aligned = sp_returns.reindex(pd.to_datetime(pd.Index(dates))).dropna()
    return _annualized_sharpe(aligned)


def correlation_over_dates(strat_returns: pd.Series, sp_returns=None) -> float:
    if sp_returns is None:
        sp_returns = load_sp500_daily_returns()
    joined = pd.concat([strat_returns.rename("s"), sp_returns.rename("b")],
                       axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    return float(joined["s"].corr(joined["b"]))


def information_ratio_vs_sp500(strat_returns: pd.Series, sp_returns=None) -> float:
    if sp_returns is None:
        sp_returns = load_sp500_daily_returns()
    joined = pd.concat([strat_returns.rename("s"), sp_returns.rename("b")],
                       axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    active = joined["s"] - joined["b"]
    std = float(active.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return float("nan")
    return float(active.mean()) / std * np.sqrt(_TRADING_DAYS)
