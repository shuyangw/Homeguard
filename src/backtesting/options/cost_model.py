"""M3 -- `cost_model_v2`, the one options cost model (handoff spec v2 Section 2).

Registered signature:

    cost(legs, regime_state, width_source, tier) -> (entry_cost, exit_cost, fees)

No strategy may define its own costs.

==========================================================================
THE UNIT TRAP -- read this before touching a number in this file
==========================================================================
Two conventions are in play and they differ by a factor of TWO.

  * The doc chain quotes fill conventions as a fraction of the FULL quoted
    width paid beyond mid, per transit: "single leg, normal: 25%".
  * `src/backtesting/costs/options.py` (pre-existing, methodology Section 4.5)
    models `alpha` as the fraction of the HALF-spread:
        fill = mid +/- alpha * (1/2)(ask - bid)

Therefore:

    alpha = 2 * width_fraction          25% of width  ==  alpha 0.50
    width_fraction = alpha / 2          alpha 1.0     ==  50% of width

Worked example, bid 1.00 / ask 1.20 (width 0.20, half-spread 0.10):
    25% of width  = 0.25 * 0.20 = $0.05 per share = $5.00 per contract
    alpha 0.50    = 0.50 * 0.10 = $0.05 per share = $5.00 per contract
Same number. `alpha_from_width_fraction` is the ONLY sanctioned bridge, and
`tests/backtesting/options/test_cost_model.py` reconciles it against the
existing module explicitly. That module is EXTENDED here, never forked.

==========================================================================
Parameterization
==========================================================================
Widths come from `spread_census` (V11, 31 roots x year x moneyness x DTE x
delta x vol-state, with spread percentiles) -- the measured store, not the
spec's assumed table. The assumed table is the DOCUMENTED FALLBACK for cells
the census does not cover, and every call records which was used per leg.

Fill FRACTIONS (25% / 3-6% / stressed 15-25%) remain assumptions at v2. They
are superseded only by own IBKR fill statistics once anything trades live.

Every cost is reportable at 1.0x and +/-50% via `cost_multiplier`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from enum import Enum
from typing import Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backtesting.costs.options import options_slippage_per_contract
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONTRACT_MULTIPLIER = 100.0

# --------------------------------------------------------------------------
# Registered fill fractions (fraction of FULL quoted width, per transit)
# --------------------------------------------------------------------------

#: Single leg, normal conditions.
SINGLE_LEG_WIDTH_FRAC = 0.25

#: Native multi-leg combo (2-4 legs), normal conditions -- per leg.
COMBO_WIDTH_FRAC: Tuple[float, float] = (0.03, 0.06)

#: Stressed tier: combos are treated at "single-leg grade", registered as a band.
STRESSED_COMBO_WIDTH_FRAC: Tuple[float, float] = (0.15, 0.25)

#: Native combo order types cover 2-4 legs. Anything wider legs in separately.
NATIVE_COMBO_MAX_LEGS = 4

#: Registered stressed trigger on the realized-vol percentile (strictly above).
STRESSED_RV_PERCENTILE = 0.80

STRESSED_REGIMES = frozenset({"UNPREDICTABLE", "BEAR"})
KNOWN_REGIMES = frozenset(
    {"STRONG_BULL", "WEAK_BULL", "SIDEWAYS", "UNPREDICTABLE", "BEAR"}
)

# --------------------------------------------------------------------------
# Fee stack (per contract, ONE WAY)
# --------------------------------------------------------------------------

SEC_S31_EFFECTIVE_DATE = _date(2026, 4, 4)
SEC_S31_RATE_PER_DOLLAR = 20.60 / 1e6


@dataclass(frozen=True)
class FeeSchedule:
    """Registered fee stack. Defaults are the MIDPOINT of each quoted range."""

    commission: float = 0.50          # tiered/Lite, quoted $0.35-0.65
    occ_clearing: float = 0.025       # Options Clearing Corporation
    orf: float = 0.023                # Options Regulatory Fee
    cat: float = 0.0003               # Consolidated Audit Trail
    taf_sell_only: float = 0.00329    # FINRA Trading Activity Fee, sells only
    sec_s31_rate: float = SEC_S31_RATE_PER_DOLLAR
    sec_s31_from: _date = SEC_S31_EFFECTIVE_DATE


def per_contract_fees(
    side: Literal["buy", "sell"],
    session: _date,
    proceeds_usd: float = 0.0,
    fees: Optional[FeeSchedule] = None,
) -> float:
    """One-way, per-contract fee total in dollars.

    `proceeds_usd` is the SALE proceeds of ONE contract; it only matters for
    SEC Section 31, which applies to sells from 2026-04-04.
    """
    f = fees or FeeSchedule()
    total = f.commission + f.occ_clearing + f.orf + f.cat
    if side == "sell":
        total += f.taf_sell_only
        if session >= f.sec_s31_from and np.isfinite(proceeds_usd):
            total += max(proceeds_usd, 0.0) * f.sec_s31_rate
    elif side != "buy":
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return float(total)


# --------------------------------------------------------------------------
# Assumed width table -- the DOCUMENTED FALLBACK only
# --------------------------------------------------------------------------

#: Quoted width in dollars, (low, high). Midpoints are used.
ASSUMED_WIDTH_TABLE: Dict[str, Tuple[float, float]] = {
    "index_atm_30_45": (0.03, 0.12),
    "index_0dte_atm": (0.01, 0.02),
    "index_low_delta": (0.02, 0.06),
    "single_name_atm": (0.05, 0.15),
    "rank_50_100_atm": (0.10, 0.30),
    "leaps": (0.30, 1.00),
}

#: Event windows quote 1.5-3x normal width; midpoint used on ASSUMED widths only
#: (a measured census/quote width already embeds the widening).
EVENT_WIDTH_MULTIPLIER = 2.25

INDEX_ROOTS = frozenset({"SPY", "QQQ", "IWM", "DIA", "SPX"})

LEAPS_DTE_FLOOR = 365


class Tier(str, Enum):
    NORMAL = "normal"
    STRESSED = "stressed"


# --------------------------------------------------------------------------
# Census bucketing -- must match `scripts/data/vbattery/_sweep_lib.py` exactly.
# Cross-checked against that module in the test suite; if the census builder
# ever moves an edge, that test fails rather than the lookup going silently wrong.
# --------------------------------------------------------------------------

MONEYNESS_EDGES = (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10)
MONEYNESS_LABELS = (
    "m_le_-0.10", "m_-0.10_-0.05", "m_-0.05_-0.02", "m_atm_-0.02_0.02",
    "m_0.02_0.05", "m_0.05_0.10", "m_gt_0.10", "m_undefined",
)
ABS_DELTA_LABELS = (
    "d_0_0.05", "d_0.05_0.15", "d_0.15_0.35", "d_0.35_0.65",
    "d_0.65_1", "d_null_or_oob",
)
DTE_LABELS = (
    "dte_0", "dte_1_7", "dte_8_30", "dte_31_60", "dte_61_90",
    "dte_91_180", "dte_181p", "dte_negative_or_null",
)
VOL_STATE_LABELS = ("vol_low", "vol_mid", "vol_high", "vol_unknown")

#: The census's vol proxy is a trailing-20-session RV percentile split into
#: TERCILES. That is a different question from the stressed-tier trigger
#: (percentile > 0.80); the two thresholds are deliberately not merged.
VOL_TERCILES = (0.33, 0.67)

#: A census cell thinner than this is not trusted; the assumed table is used.
MIN_CENSUS_SAMPLE = 100


def abs_delta_bucket(delta: Optional[float]) -> str:
    if delta is None:
        return ABS_DELTA_LABELS[5]
    ad = abs(float(delta))
    if not np.isfinite(ad) or ad > 1.0:
        return ABS_DELTA_LABELS[5]
    if ad < 0.05:
        return ABS_DELTA_LABELS[0]
    if ad < 0.15:
        return ABS_DELTA_LABELS[1]
    if ad < 0.35:
        return ABS_DELTA_LABELS[2]
    if ad <= 0.65:
        return ABS_DELTA_LABELS[3]
    return ABS_DELTA_LABELS[4]


def dte_bucket(dte: Optional[int]) -> str:
    if dte is None or not np.isfinite(dte) or dte < 0:
        return DTE_LABELS[7]
    d = int(dte)
    if d == 0:
        return DTE_LABELS[0]
    if d <= 7:
        return DTE_LABELS[1]
    if d <= 30:
        return DTE_LABELS[2]
    if d <= 60:
        return DTE_LABELS[3]
    if d <= 90:
        return DTE_LABELS[4]
    if d <= 180:
        return DTE_LABELS[5]
    return DTE_LABELS[6]


def moneyness_bucket(strike: float, underlying_px: float) -> str:
    if not (np.isfinite(strike) and np.isfinite(underlying_px)) or (
        strike <= 0 or underlying_px <= 0
    ):
        return MONEYNESS_LABELS[7]
    m = float(np.log(strike / underlying_px))
    if not np.isfinite(m):
        return MONEYNESS_LABELS[7]
    return MONEYNESS_LABELS[int(np.searchsorted(np.array(MONEYNESS_EDGES), m, side="right"))]


def vol_state_bucket(rv_percentile: Optional[float]) -> str:
    if rv_percentile is None or not np.isfinite(rv_percentile):
        return "vol_unknown"
    if rv_percentile < VOL_TERCILES[0]:
        return "vol_low"
    if rv_percentile < VOL_TERCILES[1]:
        return "vol_mid"
    return "vol_high"


_CENSUS_CACHE: Dict[str, pd.DataFrame] = {}


def load_spread_census() -> pd.DataFrame:
    """Load the materialized V11 spread census (31 roots)."""
    cached = _CENSUS_CACHE.get("census")
    if cached is None:
        from src.settings import get_local_storage_dir

        path = (
            get_local_storage_dir()
            / "options" / "derived" / "spread_census" / "spread_census.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"[-] no spread census at {path}")
        cached = pd.read_parquet(path)
        _CENSUS_CACHE["census"] = cached
        logger.info(f"[+] spread_census loaded: {len(cached):,} cells")
    return cached


# --------------------------------------------------------------------------
# Legs and the tier / width resolution
# --------------------------------------------------------------------------


@dataclass
class Leg:
    """One option leg of a transit. Quote fields may be NaN -- the census then
    supplies the width, which is exactly why the census exists."""

    root: str
    session_date: _date
    expiry: _date
    strike: float
    right: str
    side: Literal["buy", "sell"]
    qty: float
    bid: float = float("nan")
    ask: float = float("nan")
    dte: Optional[int] = None
    delta: Optional[float] = None
    underlying_px: float = float("nan")

    @property
    def quoted_width(self) -> float:
        if np.isfinite(self.bid) and np.isfinite(self.ask) and self.ask >= self.bid:
            return float(self.ask - self.bid)
        return float("nan")

    @property
    def mid(self) -> float:
        if np.isfinite(self.bid) and np.isfinite(self.ask) and self.ask >= self.bid:
            return float((self.bid + self.ask) / 2.0)
        return float("nan")


def alpha_from_width_fraction(width_fraction: float) -> float:
    """Convert a FULL-width fraction into the half-spread `alpha` the existing
    options cost module speaks. 25% of width == alpha 0.50."""
    return 2.0 * float(width_fraction)


def width_fraction_from_alpha(alpha: float) -> float:
    """Inverse of `alpha_from_width_fraction`."""
    return float(alpha) / 2.0


def resolve_tier(
    regime_state: Optional[str],
    rv_percentile: Optional[float] = None,
    event_window: bool = False,
) -> Tier:
    """Registered stressed triggers: event windows, regime in {UNPREDICTABLE,
    BEAR}, or realized-vol percentile > 0.80."""
    if regime_state is not None and regime_state not in KNOWN_REGIMES:
        raise ValueError(
            f"[-] unknown regime_state {regime_state!r}; expected one of "
            f"{sorted(KNOWN_REGIMES)}"
        )
    if event_window:
        return Tier.STRESSED
    if regime_state in STRESSED_REGIMES:
        return Tier.STRESSED
    if rv_percentile is not None and np.isfinite(rv_percentile):
        if rv_percentile > STRESSED_RV_PERCENTILE:
            return Tier.STRESSED
    return Tier.NORMAL


def width_fraction_for(n_legs: int, tier: Tier, dte: Optional[int]) -> float:
    """Fraction of quoted width paid beyond mid, per leg, per transit."""
    if dte is not None and np.isfinite(dte) and int(dte) == 0:
        return SINGLE_LEG_WIDTH_FRAC          # 0DTE is always single-leg grade
    if n_legs == 1 or n_legs > NATIVE_COMBO_MAX_LEGS:
        return SINGLE_LEG_WIDTH_FRAC
    if tier is Tier.STRESSED:
        return float(np.mean(STRESSED_COMBO_WIDTH_FRAC))
    return float(np.mean(COMBO_WIDTH_FRAC))


def assumed_width_class(leg: Leg) -> str:
    """Which row of the assumed width table a leg falls in."""
    is_index = leg.root.upper() in INDEX_ROOTS
    dte = leg.dte
    if dte is not None and np.isfinite(dte) and int(dte) >= LEAPS_DTE_FLOOR:
        return "leaps"
    if is_index:
        if dte is not None and np.isfinite(dte) and int(dte) == 0:
            return "index_0dte_atm"
        if leg.delta is not None and np.isfinite(leg.delta) and abs(leg.delta) <= 0.10:
            return "index_low_delta"
        return "index_atm_30_45"
    return "single_name_atm"


def assumed_width(leg: Leg, event_window: bool = False) -> float:
    lo, hi = ASSUMED_WIDTH_TABLE[assumed_width_class(leg)]
    w = (lo + hi) / 2.0
    return w * (EVENT_WIDTH_MULTIPLIER if event_window else 1.0)


def census_width(
    leg: Leg,
    census: pd.DataFrame,
    vol_state: str,
    min_sample: int = MIN_CENSUS_SAMPLE,
) -> Optional[float]:
    """Median quoted width for the leg's census cell, or None if uncovered."""
    hit = census[
        (census["root"] == leg.root)
        & (census["year"] == leg.session_date.year)
        & (census["moneyness_bucket"] == moneyness_bucket(leg.strike, leg.underlying_px))
        & (census["dte_bucket"] == dte_bucket(leg.dte))
        & (census["abs_delta_bucket"] == abs_delta_bucket(leg.delta))
        & (census["vol_state_proxy"] == vol_state)
    ]
    if hit.empty:
        return None
    row = hit.iloc[0]
    if int(row.get("n_sampled", 0)) < min_sample:
        return None
    w = float(row["spread_abs_p50"])
    return w if np.isfinite(w) and w > 0 else None


# --------------------------------------------------------------------------
# The registered entry point
# --------------------------------------------------------------------------


@dataclass
class CostBreakdown:
    """Unpacks as the registered 3-tuple `(entry_cost, exit_cost, fees)`, with
    the provenance every call is required to record hanging off it."""

    entry_cost: float
    exit_cost: float
    fees: float
    tier: Tier
    width_fraction: float
    alpha_equivalent: float
    width_by_leg: List[float] = field(default_factory=list)
    width_source_by_leg: List[str] = field(default_factory=list)
    cost_multiplier: float = 1.0

    def __iter__(self) -> Iterator[float]:
        yield self.entry_cost
        yield self.exit_cost
        yield self.fees

    @property
    def total(self) -> float:
        return self.entry_cost + self.exit_cost + self.fees


def cost(
    legs: Sequence[Leg],
    regime_state: Optional[str] = None,
    width_source: Literal["census", "quote", "assumed"] = "census",
    tier: Optional[Tier] = None,
    rv_percentile: Optional[float] = None,
    event_window: bool = False,
    cost_multiplier: float = 1.0,
    fees: Optional[FeeSchedule] = None,
    census: Optional[pd.DataFrame] = None,
    vol_state: Optional[str] = None,
    min_census_sample: int = MIN_CENSUS_SAMPLE,
) -> CostBreakdown:
    """The registered cost entry point.

    Args:
        legs: the structure's legs, one transit's worth.
        regime_state: point-in-time classifier state, for the stressed tier.
        width_source: 'census' (registered default, falls back to the assumed
            table per leg and records it), 'quote' (use each leg's own quoted
            width), or 'assumed' (the documented fallback table only).
        tier: override the resolved tier. Normally left None.
        cost_multiplier: 1.0 / 1.5 / 0.5 -- the mandatory sensitivity band.

    Returns:
        CostBreakdown, which unpacks as (entry_cost, exit_cost, fees) in USD.
    """
    if not legs:
        raise ValueError("[-] cost() requires at least one leg")
    if cost_multiplier <= 0:
        raise ValueError(f"[-] cost_multiplier must be > 0, got {cost_multiplier}")

    resolved_tier = tier or resolve_tier(regime_state, rv_percentile, event_window)
    dtes = {l.dte for l in legs if l.dte is not None}
    structure_dte = min(dtes) if dtes else None
    frac = width_fraction_for(len(legs), resolved_tier, structure_dte)
    vstate = vol_state or vol_state_bucket(rv_percentile)

    if width_source == "census" and census is None:
        census = load_spread_census()

    widths: List[float] = []
    sources: List[str] = []
    entry = 0.0
    fee_total = 0.0

    for leg in legs:
        w: Optional[float] = None
        src = ""
        if width_source == "quote":
            w = leg.quoted_width
            src = "quote"
            if not np.isfinite(w):
                w = None
        elif width_source == "census":
            w = census_width(leg, census, vstate, min_census_sample)
            src = "census"
        if w is None:
            w = assumed_width(leg, event_window=event_window)
            src = "assumed"

        widths.append(float(w))
        sources.append(src)
        entry += frac * w * CONTRACT_MULTIPLIER * leg.qty

        exit_side = "sell" if leg.side == "buy" else "buy"
        proceeds = leg.mid * CONTRACT_MULTIPLIER
        proceeds = proceeds if np.isfinite(proceeds) else 0.0
        fee_total += leg.qty * per_contract_fees(
            leg.side, leg.session_date, proceeds, fees
        )
        fee_total += leg.qty * per_contract_fees(
            exit_side, leg.session_date, proceeds, fees
        )

    exit_cost = entry  # one transit in, one transit out, same convention
    return CostBreakdown(
        entry_cost=entry * cost_multiplier,
        exit_cost=exit_cost * cost_multiplier,
        fees=fee_total * cost_multiplier,
        tier=resolved_tier,
        width_fraction=frac,
        alpha_equivalent=alpha_from_width_fraction(frac),
        width_by_leg=widths,
        width_source_by_leg=sources,
        cost_multiplier=cost_multiplier,
    )


def cost_sensitivity_band(
    legs: Sequence[Leg], **kwargs
) -> Dict[str, CostBreakdown]:
    """The mandatory 1.0x / +/-50% report. Never optional."""
    kwargs.pop("cost_multiplier", None)
    return {
        "0.5x": cost(legs, cost_multiplier=0.5, **kwargs),
        "1.0x": cost(legs, cost_multiplier=1.0, **kwargs),
        "1.5x": cost(legs, cost_multiplier=1.5, **kwargs),
    }


def slippage_via_alpha_module(
    bid: float, ask: float, side: Literal["buy", "sell"], width_fraction: float
) -> Tuple[float, float]:
    """Route a width-fraction through the pre-existing alpha module.

    Exists so the two conventions have exactly one bridge and any drift shows
    up as a test failure rather than a silently doubled cost.
    """
    return options_slippage_per_contract(
        bid, ask, side=side, override_alpha=alpha_from_width_fraction(width_fraction)
    )
