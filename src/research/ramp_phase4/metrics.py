"""Performance metrics for the Phase B harness.

Pure functions over pandas Series / list[DailyRecord]. No engine internals.
"""

from __future__ import annotations

from typing import Dict, List
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


def avg_daily_turnover(records: List) -> float:
    """Average of (turnover_usd / portfolio_value) across days where portfolio_value > 0."""
    ratios = []
    for r in records:
        if r.portfolio_value > 0:
            ratios.append(r.turnover_usd / r.portfolio_value)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def cost_drag_pct(records: List) -> float:
    """Total cost / total gross return, as a fraction.

    Returns 0.0 if gross return is non-positive.
    """
    if not records:
        return 0.0
    total_cost = sum(r.cost_usd for r in records)
    initial_value = records[0].portfolio_value
    if initial_value <= 0:
        return 0.0
    net_return = 1.0
    for r in records:
        net_return *= (1 + r.daily_return)
    net_return -= 1.0
    gross_return = net_return + total_cost / initial_value
    if gross_return <= 0:
        return 0.0
    return (total_cost / initial_value) / gross_return


def regime_attribution(records: List) -> Dict[str, Dict[str, float]]:
    """Per-regime aggregates: days and net_return (compounded)."""
    by_regime: Dict[str, Dict[str, float]] = {}
    for r in records:
        if r.regime not in by_regime:
            by_regime[r.regime] = {'days': 0, '_compound': 1.0}
        by_regime[r.regime]['days'] += 1
        by_regime[r.regime]['_compound'] *= (1 + r.daily_return)
    out: Dict[str, Dict[str, float]] = {}
    for regime, d in by_regime.items():
        out[regime] = {
            'days': int(d['days']),
            'net_return': float(d['_compound'] - 1.0),
        }
    return out
