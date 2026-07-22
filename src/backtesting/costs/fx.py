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


_METALS_BASES = {"XAU", "XAG"}

# Per-SIDE half-spread in bps of USD notional for emerging-market currencies.
# EM pairs are far wider than the G10 major/minor pip tiers and the pip path
# under-costs them (a 1-pip major spread on USDMXN is ~0.5bp vs the ~3bp reality).
# Priced in bps of notional (like metals), not pips. Values pre-registered in
# docs/superpowers/specs/2026-07-21-fx-em-wave-preregistration.md Section 4.
_EM_HALF_SPREAD_BPS: dict[str, float] = {
    "MXN": 3.0, "ZAR": 6.0, "PLN": 4.0, "HUF": 5.0,
    "CNH": 5.0, "TRY": 15.0, "INR": 8.0, "BRL": 10.0,
}


def _em_half_bps(pair: str) -> Optional[float]:
    """Per-side half-spread bps if either leg is an EM currency, else None.
    Both-EM crosses take the wider (max) leg."""
    legs = [b for b in (pair[:3], pair[3:]) if b in _EM_HALF_SPREAD_BPS]
    return max(_EM_HALF_SPREAD_BPS[b] for b in legs) if legs else None


def _pip_size(pair: str) -> float:
    """0.01 for JPY-quoted pairs, 0.0001 otherwise."""
    return 0.01 if pair[3:] == "JPY" else 0.0001


def fx_round_trip_usd(pair: str, units_traded: float, price: float,
                      quote_to_usd: float, tier: FxTier = "major",
                      session: Session = "ny", metals_bps: float = 4.0) -> float:
    """Total round-trip USD cost for trading abs(units_traded) base units.

    Currency pairs: spread (pips) x pip_size x units x quote->USD. Metals
    (XAU/XAG) have no standard pip -> priced as metals_bps of USD notional,
    which is scale-invariant and how metal spreads are actually quoted.
    """
    qty = abs(units_traded)
    if pair[:3] in _METALS_BASES:
        notional_usd = qty * price * quote_to_usd
        return notional_usd * metals_bps / 10_000.0
    em_half = _em_half_bps(pair)
    if em_half is not None:
        notional_usd = qty * price * quote_to_usd
        return notional_usd * (2.0 * em_half) / 10_000.0  # round trip = 2 x per-side
    rt_pips = fx_round_trip_pips(tier, session)
    return rt_pips * _pip_size(pair) * qty * quote_to_usd
