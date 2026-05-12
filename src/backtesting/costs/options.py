"""Options cost model per docs/methodology/backtesting.md Section 4.5.

Fill price relative to NBBO midpoint:

    fill_buy  = mid + alpha * (1/2)(ask - bid)
    fill_sell = mid - alpha * (1/2)(ask - bid)

alpha is the FRACTION OF THE HALF-SPREAD crossed. alpha=1.0 means you
cross the full half-spread (you pay/receive the ask/bid).
"""
from __future__ import annotations

from typing import Literal, Optional

OptionsLiquidity = Literal["very_liquid", "liquid_etf", "single_stock_atm", "wings_illiquid"]

# Methodology midpoints from Section 4.5 Table.
OPTIONS_ALPHA_TABLE: dict[str, float] = {
    "very_liquid": 0.4,       # SPY weekly ATM
    "liquid_etf": 0.6,        # QQQ, IWM ATM
    "single_stock_atm": 0.85, # AAPL, MSFT
    "wings_illiquid": 1.1,    # OTM, far-dated -- can exceed half-spread
}

# IBKR retail per-contract commission. Section 4.5: $0.50-$0.65.
DEFAULT_IBKR_PER_CONTRACT_USD = 0.58


def options_slippage_per_contract(
    bid: float,
    ask: float,
    side: Literal["buy", "sell"],
    liquidity: OptionsLiquidity = "single_stock_atm",
    override_alpha: Optional[float] = None,
) -> tuple[float, float]:
    """Per-contract fill price + slippage cost in dollars.

    Args:
        bid: NBBO bid (per contract, dollars)
        ask: NBBO ask
        side: "buy" or "sell"
        liquidity: tier per Section 4.5
        override_alpha: fraction of half-spread to cross; overrides table

    Returns:
        (fill_price, slippage_usd_per_contract)
        slippage_usd_per_contract is positive -- the cost paid relative
        to mid. Note: an options contract typically multiplies by 100
        for dollar P&L; that scaling is the caller's responsibility.
    """
    if ask < bid:
        raise ValueError(f"ask {ask} < bid {bid}")
    if override_alpha is not None:
        alpha = float(override_alpha)
    else:
        if liquidity not in OPTIONS_ALPHA_TABLE:
            raise ValueError(
                f"Unknown liquidity {liquidity!r}. Choices: {list(OPTIONS_ALPHA_TABLE)}"
            )
        alpha = OPTIONS_ALPHA_TABLE[liquidity]

    mid = (bid + ask) / 2.0
    half_spread = (ask - bid) / 2.0
    if side == "buy":
        fill = mid + alpha * half_spread
        slippage = fill - mid
    elif side == "sell":
        fill = mid - alpha * half_spread
        slippage = mid - fill
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    return float(fill), float(slippage)
