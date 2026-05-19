"""Performance metrics for the Phase B harness.

Pure functions over pandas Series / list[DailyRecord]. No engine internals.
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(daily_returns: pd.Series) -> float:
    """Annualized Sharpe assuming 252 trading days, rf = 0.

    Returns 0.0 when std is 0 (constant returns) to avoid div-by-zero.
    """
    rets = daily_returns.dropna()
    if len(rets) < 2:
        return 0.0
    std = rets.std(ddof=1)
    if std == 0.0 or not np.isfinite(std) or std < 1e-15:
        return 0.0
    return (rets.mean() * TRADING_DAYS_PER_YEAR) / (std * np.sqrt(TRADING_DAYS_PER_YEAR))


def cagr(equity_curve: pd.Series) -> float:
    """Compound annual growth rate from start to end of the equity curve.

    Uses trading-days/252 as the year count (matches existing reports).
    """
    eq = equity_curve.dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    years = len(eq) / TRADING_DAYS_PER_YEAR
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1


def max_drawdown(equity_curve: pd.Series) -> float:
    """Peak-to-trough drawdown over the curve, as a negative fraction.

    Returns 0.0 for monotonically rising curves.
    """
    eq = equity_curve.dropna()
    if len(eq) < 2:
        return 0.0
    running_max = eq.cummax()
    drawdowns = (eq - running_max) / running_max
    return float(drawdowns.min())
