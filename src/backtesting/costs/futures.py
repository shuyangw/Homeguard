"""Futures cost model per docs/methodology/backtesting.md Section 4.2.

Per-contract costs: IBKR commission + exchange/reg fees + slippage.
Half-tick slippage for ES/NQ during regular hours; one tick off-hours.
"""
from __future__ import annotations

from typing import Literal, Optional

FuturesContract = Literal["ES", "NQ", "YM", "RTY", "CL", "GC", "ZN", "6E", "ZC"]

# (commission_usd, exchange_fee_usd) per side. Methodology midpoints.
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

# Minimum tick size in USD-per-contract.
FUTURES_TICK_USD: dict[str, float] = {
    "ES":  12.50,   # 0.25 index point x $50 multiplier
    "NQ":  5.00,    # 0.25 x $20
    "YM":  5.00,    # 1 x $5
    "RTY": 5.00,    # 0.10 x $50
    "CL":  10.00,
    "GC":  10.00,
    "ZN":  15.625,
    "6E":  6.25,
    "ZC":  12.50,
}


def futures_round_trip_usd(
    contract: FuturesContract,
    regular_hours: bool = True,
    n_contracts: int = 1,
    override_per_side_usd: Optional[float] = None,
) -> float:
    """Total round-trip cost in USD for ``n_contracts``.

    Args:
        contract: contract symbol
        regular_hours: half-tick slippage if True, one tick otherwise
        n_contracts: number of contracts
        override_per_side_usd: if you have a measured value, pass it directly

    Returns:
        Round-trip cost in USD. Add roll-spread cost when the strategy
        holds across a contract roll (one-half extra tick per roll).
    """
    if contract not in FUTURES_PER_SIDE_USD:
        raise ValueError(
            f"Unknown contract {contract!r}. Choices: {list(FUTURES_PER_SIDE_USD)}"
        )

    if override_per_side_usd is not None:
        per_side = float(override_per_side_usd)
    else:
        commission, exch = FUTURES_PER_SIDE_USD[contract]
        per_side = commission + exch

    tick = FUTURES_TICK_USD[contract]
    slippage_per_side = tick * (0.5 if regular_hours else 1.0)
    return n_contracts * 2 * (per_side + slippage_per_side)
