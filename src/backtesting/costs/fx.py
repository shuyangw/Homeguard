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

# MEASURED round-trip spreads in BPS OF NOTIONAL (2026-07-26). Source: real
# Dukascopy bid/ask, 25 pairs x Jun 2015/2020/2024, ~30k quotes per pair-month;
# see docs/strategies/research/20260726_fx_measured_spreads.md and rebuild with
# scripts/data/measure_fx_spreads.py. Baked in as constants rather than read from
# the artifact because the artifact lives in gitignored local storage, and a cost
# model that silently changes with local data is not reproducible.
#
# These REPLACE the pip-tier model, which charged in PIPS -- a different fraction
# of price at different price levels (0.92bps on EURUSD at 1.08 versus 0.086bps on
# EURNOK at 11.67). That systematically UNDER-charged high-priced crosses: the
# Nordic block was 14-20x too cheap and silver 2.6x too cheap, while majors were
# 2.4-7x too expensive.
_MEASURED_RT_BPS: dict[str, float] = {
    "EURUSD": 0.32, "USDJPY": 0.33, "EURJPY": 0.61, "USDCNH": 0.73,
    "GBPUSD": 0.80, "USDCAD": 0.95, "AUDJPY": 0.98, "CHFJPY": 1.20,
    "USDCHF": 1.41, "EURCHF": 1.53, "AUDUSD": 1.62, "XAUUSD": 1.63,
    "USDTRY": 1.69, "NZDJPY": 1.79, "NZDUSD": 2.05, "AUDNZD": 2.15,
    "USDSEK": 3.43, "EURSEK": 3.70, "USDNOK": 3.96, "EURNOK": 4.32,
    "USDMXN": 4.32, "USDPLN": 4.90, "USDZAR": 5.54, "USDHUF": 5.65,
    "XAGUSD": 10.41,
}

# Fallback for pairs with no direct quote (the triangulated Nordic crosses, and
# EM pairs Dukascopy does not carry). Set at the measured median of comparable
# pairs rather than a pip constant.
_UNMEASURED_RT_BPS = 4.0

# Commission charged ON TOP of the spread, per side, in bps of notional.
# UNVERIFIED: the IBKR published schedule could not be retrieved (both pricing
# URLs return HTTP 403 to automated fetches on 2026-07-26). 0.20 bps/side is the
# widely-cited IDEALPRO figure and is used as a documented DEFAULT, not a checked
# fact. Verify against the current schedule before relying on a marginal result,
# and override via `commission_bps_per_side` rather than editing this constant.
_DEFAULT_COMMISSION_BPS_PER_SIDE = 0.20

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
                      session: Session = "ny", metals_bps: float = 4.0,
                      commission_bps_per_side: Optional[float] = None) -> float:
    """Total round-trip USD cost for trading abs(units_traded) base units.

    Cost = (MEASURED round-trip spread + 2 x commission) bps of USD notional.
    Both legs are in bps so the model is scale-invariant across price levels --
    the failure of the old pip-based tier model. `tier`, `session` and
    `metals_bps` are retained for signature compatibility and are UNUSED.
    """
    qty = abs(units_traded)
    notional_usd = qty * price * quote_to_usd
    if commission_bps_per_side is None:
        commission_bps_per_side = _DEFAULT_COMMISSION_BPS_PER_SIDE
    spread_rt = _MEASURED_RT_BPS.get(pair, _UNMEASURED_RT_BPS)
    total_rt_bps = spread_rt + 2.0 * float(commission_bps_per_side)
    return notional_usd * total_rt_bps / 10_000.0
