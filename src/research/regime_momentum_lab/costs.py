"""Transaction cost models for Phase B harness.

Currently flat-bps only. Abstraction exists so Phase C can add models
(spread proxy, market impact) without touching engine.py.
"""

from __future__ import annotations

from typing import List, Mapping


def flat_bps_cost(trades: List[Mapping], bps: float) -> float:
    """Total transaction cost = sum(|trade_value_usd|) * bps / 1e4.

    Args:
        trades: list of dicts with key 'trade_value_usd' (signed: + buy, - sell).
        bps: cost per side, in basis points.

    Returns:
        Total cost in USD (non-negative).
    """
    if not trades:
        return 0.0
    return sum(abs(t['trade_value_usd']) for t in trades) * bps / 1e4
