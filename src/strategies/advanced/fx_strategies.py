"""Spot-FX reference strategies.

Both are price-only forecast_panel strategies, so they reuse the futures
forecast logic unchanged: FX trend = Carver multi-speed EWMAC; FX value =
Asness nominal 5yr-to-1yr reversal. Thin subclasses keep the FX names distinct
in the registry and leave room to diverge (e.g. a future PPP value signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy
from src.strategies.advanced.futures_value_strategy import FuturesValueStrategy


class FxTrendStrategy(CarverMomentumStrategy):
    pass


class FxValueStrategy(FuturesValueStrategy):
    pass


class FxTSMOMStrategy:
    """Time-series momentum (#3, Moskowitz-Ooi-Pedersen).

    Forecast = scale * mean(sign(ret_short), sign(ret_long)): long when both the
    short and long trailing returns are positive, short when both negative, flat
    when they disagree. Vol-scaling comes from the engine's per-instrument
    vol-target sizing. Forecast is on the Carver scale (10 = full 1x position).
    """

    def __init__(self, universe, lookback_short: int = 63, lookback_long: int = 252,
                 scale: float = 10.0, **params):
        self.universe = list(universe)
        self.lookback_short = int(lookback_short)
        self.lookback_long = int(lookback_long)
        self.scale = float(scale)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for root in self.universe:
            if root not in close_panel.columns:
                continue
            c = close_panel[root].astype(float)
            s = np.sign(c.pct_change(self.lookback_short, fill_method=None))
            l = np.sign(c.pct_change(self.lookback_long, fill_method=None))
            out[root] = (self.scale * (s + l) / 2.0).fillna(0.0)
        cols = [r for r in self.universe if r in out]
        return pd.DataFrame(out)[cols]
