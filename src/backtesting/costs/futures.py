"""Futures cost model per docs/methodology/backtesting.md Section 4.2.

Per-contract costs: IBKR commission + exchange/reg fees + slippage.
Half-tick slippage for ES/NQ during regular hours; one tick off-hours.

Tick values come from ``src.data.futures.contract_specs.SPECS`` (the single
source of truth for contract multipliers/ticks). Per-side commission has a
SINGLE source of truth, ``PER_SIDE_COMMISSION_USD``: a measured,
methodology-midpoint value for the original 9 roots, and an approximate IBKR
all-in per-side tier estimate (execution + exchange + reg, in USD) for the
other 44 roots (53 total).
"""
from __future__ import annotations

from typing import Optional

from src.data.futures.contract_specs import SPECS

# Reference only, not consulted at runtime. Superseded by
# PER_SIDE_COMMISSION_USD below (the single source of truth for per-side
# commission) -- kept here only because src.backtesting.costs.__init__
# still re-exports it. The (commission_usd, exchange_fee_usd) breakdown for
# the original 9 roots was folded into PER_SIDE_COMMISSION_USD as
# commission + exch, e.g. ES: 1.17 + 1.75 = 2.92.
FUTURES_PER_SIDE_USD: dict[str, tuple[float, float]] = {
    "ES":  (1.17, 1.75),
    "NQ":  (1.17, 1.75),
    "YM":  (1.17, 1.50),
    "RTY": (1.17, 1.50),
    "CL":  (0.85, 1.75),
    "GC":  (1.50, 2.50),
    "ZN":  (0.85, 1.00),
    "6E":  (0.85, 1.00),
    "ZC":  (1.00, 1.50),
}

# Single source of truth for per-side (execution + exchange + reg) commission
# in USD, for all 53 SPECS roots. The original 9 roots use the measured
# methodology-midpoint total (commission + exchange fee, see
# FUTURES_PER_SIDE_USD above for the breakdown); the other 44 use an
# approximate IBKR all-in tier estimate: micros ~0.85, minis ~1.25, full
# index/energy/metal ~2.25, rates/FX ~2.50, ag ~2.50, crypto micro ~0.90.
_TIER_DEFAULT = 2.50
PER_SIDE_COMMISSION_USD: dict[str, float] = {
    # original 9 roots -- measured methodology-midpoint total (commission + exch)
    "ES": 2.92, "NQ": 2.92, "YM": 2.67, "RTY": 2.67, "CL": 2.60,
    "GC": 4.00, "ZN": 1.85, "6E": 1.85, "ZC": 2.50,
    # micros
    "MES": 0.85, "MNQ": 0.85, "M2K": 0.85, "MYM": 0.85, "MCL": 0.85, "MNG": 0.85,
    "MGC": 0.85, "SIL": 0.85, "MBT": 0.90, "MET": 0.90,
    # minis / index (non-overlapping with the 9 roots above)
    "BTC": 5.00, "ETH": 5.00,
    # energy / metals full (non-overlapping with the 9 roots above)
    "NG": 2.25, "HO": 2.25, "RB": 2.25, "BZ": 2.25,
    "SI": 2.25, "HG": 2.25, "PL": 2.25,
    # rates (non-overlapping with the 9 roots above)
    "ZT": 2.50, "ZF": 2.50, "TN": 2.50, "ZB": 2.50, "UB": 2.50,
    "SR3": 2.50, "SR1": 2.50, "10Y": 1.25, "30Y": 1.25, "5YY": 1.25, "2YY": 1.25,
    # FX (non-overlapping with the 9 roots above)
    "6J": 2.50, "6B": 2.50, "6A": 2.50, "6C": 2.50, "6S": 2.50, "6N": 2.50, "6M": 2.50,
    # ag (non-overlapping with the 9 roots above)
    "ZS": 2.50, "ZW": 2.50, "KE": 2.50, "ZL": 2.50, "ZM": 2.50, "LE": 2.50, "HE": 2.50,
}


def futures_round_trip_usd(
    contract: str,
    regular_hours: bool = True,
    n_contracts: int = 1,
    override_per_side_usd: Optional[float] = None,
) -> float:
    """Total round-trip cost in USD for ``n_contracts``.

    Args:
        contract: contract root symbol, must exist in
            ``src.data.futures.contract_specs.SPECS``
        regular_hours: half-tick slippage if True, one tick otherwise
        n_contracts: number of contracts
        override_per_side_usd: if you have a measured value, pass it directly

    Returns:
        Round-trip cost in USD. Add roll-spread cost when the strategy
        holds across a contract roll (one-half extra tick per roll).

    Raises:
        KeyError: if ``contract`` is not a known root in ``SPECS``.
    """
    if contract not in SPECS:
        raise KeyError(
            f"Unknown contract {contract!r}. Not in contract_specs.SPECS."
        )

    if override_per_side_usd is not None:
        per_side = float(override_per_side_usd)
    else:
        per_side = PER_SIDE_COMMISSION_USD.get(contract, _TIER_DEFAULT)

    tick = SPECS[contract].tick_value
    slippage_per_side = tick * (0.5 if regular_hours else 1.0)
    return n_contracts * 2 * (per_side + slippage_per_side)
