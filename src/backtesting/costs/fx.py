"""FX cost model per docs/methodology/backtesting.md Section 4.3.

Spread costs in pips, session-aware. Retail spread-only tier absorbs
commission in the spread.
"""
from __future__ import annotations

from typing import Literal, Optional

FxTier = Literal["major", "minor", "exotic"]
Session = Literal["london_ny_overlap", "london", "ny", "asia", "weekend"]

# (min_pips, max_pips) per pair tier, methodology range; we use the midpoint.
_TIER_RANGE_PIPS: dict[str, tuple[float, float]] = {
    "major": (0.5, 1.5),   # EUR/USD, USD/JPY, GBP/USD, etc.
    "minor": (1.5, 3.5),   # EUR/JPY, GBP/JPY, AUD/CAD, etc.
    "exotic": (3.5, 8.0),  # USD/TRY, USD/MXN, USD/ZAR, EM crosses
}

FX_PIP_TIERS: dict[str, float] = {
    tier: (lo + hi) / 2.0 for tier, (lo, hi) in _TIER_RANGE_PIPS.items()
}

# Session multiplier on the base pip spread. London/NY overlap is tightest.
_SESSION_MULT: dict[str, float] = {
    "london_ny_overlap": 1.0,
    "london": 1.2,
    "ny": 1.2,
    "asia": 1.8,
    "weekend": 3.0,
}


def fx_round_trip_pips(
    tier: FxTier,
    session: Session = "london_ny_overlap",
    override_pips: Optional[float] = None,
) -> float:
    """Total round-trip spread cost in pips.

    Args:
        tier: pair liquidity tier
        session: trading session
        override_pips: if you have a measured value, pass it directly

    Returns:
        Round-trip spread in pips. Round-trip = cross the spread twice.
    """
    if override_pips is not None:
        return float(override_pips) * 2  # caller passed per-side; doubled
    if tier not in FX_PIP_TIERS:
        raise ValueError(f"Unknown FX tier {tier!r}. Choices: {list(FX_PIP_TIERS)}")
    if session not in _SESSION_MULT:
        raise ValueError(f"Unknown session {session!r}. Choices: {list(_SESSION_MULT)}")
    return FX_PIP_TIERS[tier] * _SESSION_MULT[session] * 2  # round-trip
