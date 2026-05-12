"""Equities cost model per docs/methodology/backtesting.md Section 4.1.

Round-trip cost = spread cost + commission + market impact (when relevant).

Tier defaults are MIDPOINTS of the methodology's bps ranges; callers
should override when they have a better empirical estimate for the
specific universe (e.g., RAMP large-cap S&P 500 hits the low end).
"""
from __future__ import annotations

import math
from typing import Literal, Optional

EquityTier = Literal["large_cap_liquid", "mid_cap", "leveraged_etf", "small_cap"]

# (lo_bps, hi_bps) per Section 4.1.
_TIER_RANGE_BPS: dict[str, tuple[float, float]] = {
    "large_cap_liquid": (5.0, 15.0),
    "mid_cap": (15.0, 30.0),
    "leveraged_etf": (15.0, 30.0),
    "small_cap": (50.0, 200.0),
}

# Midpoint by default; conservative without being punitive.
EQUITY_TIER_BPS: dict[str, float] = {
    tier: (lo + hi) / 2.0 for tier, (lo, hi) in _TIER_RANGE_BPS.items()
}


def equities_round_trip_bps(tier: EquityTier, override_bps: Optional[float] = None) -> float:
    """Total round-trip cost in basis points.

    Args:
        tier: liquidity tier per Section 4.1
        override_bps: if you have a measured value, pass it directly

    Returns:
        Round-trip cost in bps (spread + commission). Market impact for
        large orders is additive; see equities_market_impact_bps.
    """
    if override_bps is not None:
        return float(override_bps)
    if tier not in EQUITY_TIER_BPS:
        raise ValueError(f"Unknown equity tier {tier!r}. Choices: {list(EQUITY_TIER_BPS)}")
    return EQUITY_TIER_BPS[tier]


def equities_market_impact_bps(
    order_shares: float,
    daily_volume: float,
    daily_vol_pct: float,
    eta: float = 0.5,
) -> float:
    """Square-root market impact in bps per Section 4.1.

        impact = sigma_daily * eta * sqrt(X / V_daily)

    Args:
        order_shares: order size in shares (X)
        daily_volume: average daily volume in shares (V_daily)
        daily_vol_pct: daily return volatility as a fraction (e.g. 0.015 = 1.5%)
        eta: empirical coefficient. Methodology uses ~0.5.

    Returns:
        Impact in basis points. Returns 0 if daily_volume <= 0 (caller fed
        a bad input -- we don't punish that with NaN).

    Section 4.1 also requires: single order <= 5% of ADV, participation
    rate <= 25% of contemporaneous minute volume. Those are policy, not
    cost -- enforce upstream in execution.
    """
    if daily_volume <= 0 or order_shares <= 0:
        return 0.0
    impact_pct = daily_vol_pct * eta * math.sqrt(order_shares / daily_volume)
    return impact_pct * 10_000.0  # to bps
