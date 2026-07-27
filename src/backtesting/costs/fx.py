"""FX cost model per docs/methodology/backtesting.md Section 4.3.

Spread costs in pips, session-aware. Retail spread-only tier absorbs
commission in the spread.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    import pandas as pd

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

# Crosses with no direct quotes in either source, DERIVED from their measured USD
# legs. A synthetic cross is the sum of its legs, so the legs bound the cross.
#
# The bound errs in the safe direction: a bank quoting EURGBP directly quotes
# TIGHTER than someone crossing EURUSD and GBPUSD, so these over-charge rather
# than under-charge. They are still far better than the blanket 4.0 fallback,
# which the #20 re-gate showed is not a harmless default -- GBPJPY was its
# largest leg at 1888 entries and the fallback charged roughly 3.5x this.
#
# Kept in a SEPARATE table from _MEASURED_RT_BPS so provenance stays legible: a
# derived number must never be mistaken for, or shadow, a measured one.
DERIVED_CROSS_LEGS: dict[str, tuple[str, str]] = {
    "EURGBP": ("EURUSD", "GBPUSD"),
    "GBPJPY": ("GBPUSD", "USDJPY"),
}
_DERIVED_RT_BPS: dict[str, float] = {
    cross: _MEASURED_RT_BPS[a] + _MEASURED_RT_BPS[b]
    for cross, (a, b) in DERIVED_CROSS_LEGS.items()
}

# Commission charged ON TOP of the spread, per side. CONFIRMED 2026-07-26 from
# the account's own schedule: 0.20 bps of trade value, with a $2 USD MINIMUM per
# order, at the entry tier (under $1bn monthly volume).
#
# The minimum is the term that bites. It stops binding only above $100,000 of
# notional per order, and a retail-sized intraday trade sits well below that: at
# $25k capital with a 33-pip stop at 0.5% risk the notional is about $48k, so the
# true commission is roughly TWICE the headline rate. Commission is already the
# dominant cost component for tight majors -- larger than the spread itself -- so
# ignoring the minimum under-charges every small order, and under-charges
# many-small-trades strategies hardest. That is exactly an intraday spec's shape.
COMMISSION_RATE_BPS = 0.20
COMMISSION_MIN_USD = 2.0
_DEFAULT_COMMISSION_BPS_PER_SIDE = COMMISSION_RATE_BPS


def effective_commission_bps(notional_usd: Optional[float],
                             rate_bps: float = COMMISSION_RATE_BPS,
                             min_usd: float = COMMISSION_MIN_USD) -> float:
    """Per-side commission in bps once the per-order minimum is applied.

    `notional_usd=None` means the order size is unknown, in which case the
    headline rate is returned. That is a statement of ignorance, not a claim
    that the minimum does not bind: a caller that knows its size should pass it.
    """
    if notional_usd is None:
        return rate_bps
    if notional_usd <= 0:
        raise ValueError(f"notional_usd must be positive, got {notional_usd}")
    return max(notional_usd * rate_bps / 1e4, min_usd) / notional_usd * 1e4

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


# MEASURED hour-of-week spread SHAPE, as a multiplier on the per-pair level in
# _MEASURED_RT_BPS (quote-weighted mean multiplier = 1.0, so the level above is
# unchanged in aggregate). Committed to the repo, not read from local storage, for
# the same reproducibility reason the levels are baked in; rebuild with
# scripts/data/build_fx_hour_of_week_cost.py.
#
# This REPLACES the hour-blind intraday path. A flat spread is wrong for any
# strategy that concentrates its trades in particular hours, and real dispersion
# is large: 10-19x across the week on the majors.
_HOW_SURFACE_PATH = Path(__file__).resolve().parents[3] / "config" / "costs" / "fx_hour_of_week_spread.csv"
_HOW_CACHE: dict[str, dict[int, float]] = {}
_HOW_WIDEST: dict[str, float] = {}
_HOURS_PER_WEEK = 168


def load_hour_of_week_surface() -> "pd.DataFrame":
    import pandas as pd
    return pd.read_csv(_HOW_SURFACE_PATH)


def _how_tables() -> tuple[dict[str, dict[int, float]], dict[str, float]]:
    if not _HOW_CACHE:
        surf = load_hour_of_week_surface()
        for pair, grp in surf.groupby("pair"):
            _HOW_CACHE[pair] = dict(zip(grp["hour_of_week"], grp["spread_multiplier"]))
            _HOW_WIDEST[pair] = float(grp["spread_multiplier"].max())
    return _HOW_CACHE, _HOW_WIDEST


def _spread_rt_bps(pair: str) -> float:
    """Round-trip spread level, measured where possible, then derived, then flat."""
    if pair in _MEASURED_RT_BPS:
        return _MEASURED_RT_BPS[pair]
    if pair in _DERIVED_RT_BPS:
        return _DERIVED_RT_BPS[pair]
    return _UNMEASURED_RT_BPS


def hour_of_week_multiplier(pair: str, hour_of_week: int) -> float:
    """Spread multiplier for `pair` at `hour_of_week` (Monday 00:00 UTC = 0).

    Unquoted hours (the weekend close) charge the pair's WIDEST observed hour:
    a strategy trading into a market with no measured quotes should not be
    handed the average. Unmeasured pairs get a flat 1.0 -- no shape information,
    stated rather than invented.

    A derived cross inherits a spread-weighted blend of its legs' shapes, which
    is exact given the synthetic construction: the cross spread at hour h is the
    sum of the leg spreads at hour h. Both leg multipliers are already
    normalised to mean 1.0 and the weights sum to 1, so the blend is too.
    """
    if not 0 <= int(hour_of_week) < _HOURS_PER_WEEK:
        raise ValueError(f"hour_of_week must be in [0, {_HOURS_PER_WEEK}), got {hour_of_week}")
    by_pair, widest = _how_tables()
    if pair in by_pair:
        return float(by_pair[pair].get(int(hour_of_week), widest[pair]))
    if pair in DERIVED_CROSS_LEGS:
        a, b = DERIVED_CROSS_LEGS[pair]
        lvl_a, lvl_b = _MEASURED_RT_BPS[a], _MEASURED_RT_BPS[b]
        return ((lvl_a * hour_of_week_multiplier(a, hour_of_week)
                 + lvl_b * hour_of_week_multiplier(b, hour_of_week))
                / (lvl_a + lvl_b))
    return 1.0


def fx_round_trip_bps_at(pair: str, hour_of_week: int,
                         commission_bps_per_side: Optional[float] = None,
                         notional_usd: Optional[float] = None) -> float:
    """Round-trip cost in bps of notional at a specific hour of the week.

    Only the spread scales with the hour. Commission is flat per side in bps,
    but its EFFECTIVE rate rises as order size falls, because of the $2 per-order
    minimum: pass `notional_usd` to have that applied. An explicit
    `commission_bps_per_side` overrides both.
    """
    if commission_bps_per_side is None:
        commission_bps_per_side = effective_commission_bps(notional_usd)
    spread_rt = _spread_rt_bps(pair) * hour_of_week_multiplier(pair, hour_of_week)
    return spread_rt + 2.0 * float(commission_bps_per_side)


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
    total_rt_bps = _spread_rt_bps(pair) + 2.0 * float(commission_bps_per_side)
    return notional_usd * total_rt_bps / 10_000.0
