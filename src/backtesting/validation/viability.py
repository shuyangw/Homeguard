"""Statistical-viability screen: can this spec clear the bar even if it is right?

The FX campaign's sharpest lesson. The deflated bar sat at roughly 1.05-1.14
annualized for most of the campaign, while the realistic literature ceiling for
a single daily G10 factor is about 0.3-0.6 net Sharpe. Most of those 141 trials
were arithmetically incapable of passing even if their thesis was entirely
correct. Each one still raised the bar for everything tested after it.

So before a spec consumes a trial, state its best-case-if-true Sharpe and
compare it against the CURRENT SR_zero. A spec that cannot reach the bar belongs
in the forward-paper queue or inside a portfolio-combination spec, not in a
standalone historical trial.

    if_true_sharpe = sqrt(trades_per_year) * (gross_edge - cost) / per_trade_vol

`gross_edge_bps` and `per_trade_vol_bps` are the author's honest estimates from
the literature or a measurement, and must be stated in the pre-registration.
`cost_bps` is NOT an estimate: it is computed from the measured hour-of-week
spread surface for the pairs and hours the spec actually trades. That matters
because the same signal can be viable on EURUSD in the London hours and hopeless
on USDNOK at the rollover, and a spec-level average would hide it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.backtesting.costs.fx import fx_round_trip_bps_at


@dataclass(frozen=True)
class ViabilityResult:
    name: str
    if_true_sharpe: float
    sr_zero: float
    cost_bps: float
    margin: float

    @property
    def viable(self) -> bool:
        return self.if_true_sharpe > self.sr_zero

    def summary(self) -> str:
        verdict = "VIABLE" if self.viable else "NOT VIABLE"
        return (f"{self.name}: if-true Sharpe {self.if_true_sharpe:.2f} vs bar "
                f"{self.sr_zero:.2f} (cost {self.cost_bps:.2f} bps RT) -> {verdict}")


def if_true_sharpe(trades_per_year: float, gross_edge_bps: float,
                   cost_bps: float, per_trade_vol_bps: float) -> float:
    """Annualized Sharpe the spec would achieve if its thesis were exactly right."""
    if trades_per_year <= 0:
        raise ValueError(f"trades_per_year must be positive, got {trades_per_year}")
    if per_trade_vol_bps <= 0:
        raise ValueError(f"per_trade_vol_bps must be positive, got {per_trade_vol_bps}")
    return math.sqrt(trades_per_year) * (gross_edge_bps - cost_bps) / per_trade_vol_bps


def expected_cost_bps(pairs: Sequence[str], hours_of_week: Sequence[int],
                      commission_bps_per_side: float | None = None) -> float:
    """Mean measured round-trip cost over the pairs and hours a spec trades."""
    if not pairs or not hours_of_week:
        raise ValueError("expected_cost_bps needs at least one pair and one hour")
    costs = [fx_round_trip_bps_at(p, h, commission_bps_per_side)
             for p in pairs for h in hours_of_week]
    return sum(costs) / len(costs)


def screen_spec(*, name: str, trades_per_year: float, gross_edge_bps: float,
                per_trade_vol_bps: float, pairs: Sequence[str],
                hours_of_week: Sequence[int], sr_zero: float,
                commission_bps_per_side: float | None = None) -> ViabilityResult:
    """Screen one spec against the current deflated bar."""
    cost = expected_cost_bps(pairs, hours_of_week, commission_bps_per_side)
    sharpe = if_true_sharpe(trades_per_year, gross_edge_bps, cost, per_trade_vol_bps)
    return ViabilityResult(name=name, if_true_sharpe=sharpe, sr_zero=sr_zero,
                           cost_bps=cost, margin=sharpe - sr_zero)
