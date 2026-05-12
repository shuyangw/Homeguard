"""Sharpe ratio per docs/methodology/backtesting.md Section 2.1."""
from __future__ import annotations

import numpy as np


def sharpe(returns, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio.

    Args:
        returns: per-period returns (np.ndarray, pd.Series, or list of floats).
            Daily returns by default; pass periods_per_year=12 for monthly.
        periods_per_year: 252 for daily, 12 for monthly, 52 for weekly.
            Annualization is by sqrt(periods_per_year) per Section 2.1.
        risk_free: per-period risk-free rate. Defaults to 0 per Homeguard
            convention (Section 2.1). Pass an annualized rate / periods_per_year
            for absolute Sharpe.

    Returns: annualized Sharpe. Returns 0.0 when the sample is empty or all
    identical (sigma = 0).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return 0.0
    excess = r - risk_free
    sigma = excess.std(ddof=1)
    # Treat near-zero sigma as zero: numpy's std of a near-constant float
    # array returns ~1e-19 rather than exactly 0, which would otherwise
    # blow up the ratio. The threshold is well below any economically
    # meaningful per-period volatility.
    if not np.isfinite(sigma) or sigma < 1e-12:
        return 0.0
    return float(excess.mean() / sigma * np.sqrt(periods_per_year))
