"""Turn a set of weighted futures legs into a spread signal + unit-return series.

Two return modes:
- additive: for price/yield-difference spreads (yield DV01 steepener, crack,
  crush, calendar). Level = sum(weight * close); unit_return = level.diff() /
  reference_scale (a fixed positive scale captured from the first valid level's
  magnitude, so the return series is dimensionless and Sharpe is scale-stable).
- multiplicative: for ratio spreads (gold/silver, inter-market RV). Signal =
  log(long_close / short_close); unit_return = sum(weight * close.pct_change()).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtesting.costs.futures import futures_round_trip_usd


@dataclass(frozen=True)
class SpreadLeg:
    root: str
    weight: float  # > 0 long, < 0 short


@dataclass(frozen=True)
class SpreadSeries:
    signal: pd.Series
    unit_return: pd.Series


def _weighted_level(legs: list[SpreadLeg], closes: pd.DataFrame) -> pd.Series:
    level = pd.Series(0.0, index=closes.index)
    for leg in legs:
        level = level + leg.weight * closes[leg.root]
    return level


def build_spread(legs: list[SpreadLeg], closes: pd.DataFrame, mode: str) -> SpreadSeries:
    if mode == "additive":
        level = _weighted_level(legs, closes)
        valid = level.dropna()
        scale = float(np.abs(valid.iloc[0])) if not valid.empty else 1.0
        scale = scale if scale > 1e-9 else 1.0
        unit_return = level.diff() / scale
        return SpreadSeries(signal=level, unit_return=unit_return)

    if mode == "multiplicative":
        longs = [lg for lg in legs if lg.weight > 0]
        shorts = [lg for lg in legs if lg.weight < 0]
        if len(longs) != 1 or len(shorts) != 1:
            raise ValueError("multiplicative mode requires exactly one long and one short leg")
        signal = np.log(closes[longs[0].root] / closes[shorts[0].root])
        unit_return = pd.Series(0.0, index=closes.index)
        for leg in legs:
            unit_return = unit_return + leg.weight * closes[leg.root].pct_change()
        return SpreadSeries(signal=signal.rename("signal"), unit_return=unit_return)

    raise ValueError(f"unknown mode: {mode!r}")


def round_trip_cost_usd(legs: list[SpreadLeg]) -> float:
    total = 0.0
    for leg in legs:
        n = max(1, round(abs(leg.weight)))
        total += futures_round_trip_usd(leg.root, n_contracts=n)
    return total
