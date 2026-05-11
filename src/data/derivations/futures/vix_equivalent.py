"""VIX-equivalent regime feature from ES realized volatility.

Computes a 21-day annualized realized volatility from ratio-adjusted ES daily
closes and scales it to VIX percent-units. This is a coarse proxy for the
forward-looking VIX (which uses S&P 500 options implied vol); the two correlate
well for regime classification, but the realized-vol proxy lags by ~1-3 days at
vol spikes (since it's backward-looking).

Use this when:
- VIX history isn't available locally and a Yahoo round-trip is undesirable
- A strategy needs a regime feature derivable from in-house futures data only
- Cross-checking VIX with a settlement-grade source

Don't use this when:
- Forward-looking IV is required (use real VIX from VX futures or options chain)
- Sub-daily resolution is required (this is end-of-day)
"""
from __future__ import annotations

import math
from datetime import date, timedelta

from src.data.continuous_contract_loader import ContinuousContractDataLoader

WINDOW_DAYS = 21
TRADING_DAYS_PER_YEAR = 252


def derive_vix_equivalent(
    d: date,
    window_days: int = WINDOW_DAYS,
    loader: ContinuousContractDataLoader | None = None,
) -> float:
    """Compute VIX-equivalent annualized realized volatility from ES.

    Args:
        d: As-of date (inclusive); values prior to `d` are used to compute vol.
        window_days: Trailing window size in trading days. Default 21
            (approximates VIX's 30-calendar-day horizon).
        loader: Optional pre-constructed loader (for testing / dependency
            injection). Default constructs a fresh ContinuousContractDataLoader.

    Returns:
        Annualized realized volatility in VIX percent-units (e.g., 20.0 means
        20% annualized vol). Returns NaN if fewer than `window_days + 1`
        consecutive daily closes are available ending on or before `d`.

    Raises:
        ValueError: If `window_days < 2`.
    """
    if window_days < 2:
        raise ValueError(f"window_days must be >= 2; got {window_days}")

    loader = loader or ContinuousContractDataLoader()
    start = d - timedelta(days=window_days * 3)
    daily = loader.aggregate_to_daily(
        "ES", method="ratio_adjusted", start=start, end=d,
    )

    if daily.is_empty() or daily.height < window_days + 1:
        return float("nan")

    closes = daily["close"].to_list()
    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]
    if len(log_returns) < window_days:
        return float("nan")

    window = log_returns[-window_days:]
    mean = sum(window) / len(window)
    variance = sum((r - mean) ** 2 for r in window) / (len(window) - 1)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
