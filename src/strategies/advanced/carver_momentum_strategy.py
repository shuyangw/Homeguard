"""Carver multi-speed TSMOM (parameter-free) across a futures basket."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting.base.strategy import MultiSymbolStrategy
from src.features.volatility import close_to_close_rv
from src.strategies.advanced.carver_indicators import combined_forecast

_SPEEDS = [(4, 16), (16, 64), (64, 256)]


class CarverMomentumStrategy(MultiSymbolStrategy):
    def __init__(self, universe, speeds=None, forecast_cap: float = 20.0, **params):
        self.universe = list(universe)
        self.speeds = speeds or _SPEEDS
        self.forecast_cap = forecast_cap
        super().__init__(universe=self.universe, speeds=self.speeds,
                         forecast_cap=forecast_cap, **params)

    def get_required_symbols(self):
        return self.universe

    def generate_multi_signals(self, data_dict):  # not used by the futures harness path
        raise NotImplementedError("Use forecast_panel via the futures backtest runner.")

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for root in self.universe:
            if root not in close_panel.columns:
                continue
            close = close_panel[root].astype(float)
            rets = close.pct_change(fill_method=None)
            daily_ret_std = close_to_close_rv(rets, 25, annualization_factor=1)  # daily stdev (no annualization)
            price_vol = (close * daily_ret_std).replace(0, np.nan)
            out[root] = combined_forecast(close, price_vol, self.speeds, self.forecast_cap)
        return pd.DataFrame(out)[self.universe]
