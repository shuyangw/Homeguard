"""Beta-weighted spread sizing: convert a book of 2-leg spreads to net
per-instrument target notionals, each spread vol-targeted on its own spread vol.
"""
from __future__ import annotations

import math
from typing import NamedTuple


class Spread(NamedTuple):
    leg_a: str
    leg_b: str
    hedge_ratio: float
    strength: float  # Carver scale; 10 = 1x vol-target spread, sign = direction


_ANN = math.sqrt(252)


def spread_leg_targets(spreads, sigma_s, close_row, quote_usd_row,
                       equity: float, vol_target: float, idm: float) -> dict:
    targets: dict[str, float] = {}
    for sp in spreads:
        key = (sp.leg_a, sp.leg_b)
        sig = sigma_s.get(key)
        if sig is None or sig <= 0 or not math.isfinite(sig):
            continue
        pa, pb = close_row.get(sp.leg_a), close_row.get(sp.leg_b)
        qa, qb = quote_usd_row.get(sp.leg_a), quote_usd_row.get(sp.leg_b)
        if None in (pa, pb, qa, qb) or pa <= 0 or pb <= 0:
            continue
        # notional_A (USD) so spread annualized vol == vol_target, scaled by strength/10 and idm.
        scale = (sp.strength / 10.0) * idm
        notional_a_usd = scale * vol_target * equity / (sig * _ANN)
        notional_b_usd = -sp.hedge_ratio * notional_a_usd
        units_a = notional_a_usd / (pa * qa)
        units_b = notional_b_usd / (pb * qb)
        targets[sp.leg_a] = targets.get(sp.leg_a, 0.0) + units_a
        targets[sp.leg_b] = targets.get(sp.leg_b, 0.0) + units_b
    return targets
