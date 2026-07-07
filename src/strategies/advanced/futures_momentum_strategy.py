"""Cross-sectional and time-series momentum/reversal futures strategies (SP-A).

#3 XS commodity momentum: rank trailing 12-1 return across the commodity block.
#23 short-horizon reversal: forecast ~ -z(5-day return) on liquid index roots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.advanced.futures_signal_base import CrossSectionalRankStrategy

_SKIP_DAYS = 21     # skip the most recent month (short-term reversal contamination)
_LOOKBACK_DAYS = 252  # ~12 months
_COMMODITY_GROUP = "commodity"


class FuturesXSMomentumStrategy(CrossSectionalRankStrategy):
    """12-1 cross-sectional momentum ranked across one commodity bucket."""

    def __init__(self, universe, cap: float = 20.0, **params):
        super().__init__(universe, group_fn=lambda r: _COMMODITY_GROUP, cap=cap)

    def _raw_signal_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        cols = [r for r in self.universe if r in close_panel.columns]
        px = close_panel[cols].astype(float)
        # 12-1 return: price at t-21 relative to price at t-252 (skip last month)
        mom = px.shift(_SKIP_DAYS) / px.shift(_LOOKBACK_DAYS) - 1.0
        return mom.reindex(columns=self.universe)


_REV_HORIZON = 5      # days
_REV_WINDOW = 60      # trailing window for the z-score
_REV_SCALAR = 10.0    # maps a unit z to forecast units


class FuturesReversalStrategy:
    """Short-horizon (5-day) reversal on liquid index roots.

    forecast ~ -z(5-day return) vs a trailing 60-day window, scaled and clipped.
    Continuous per-root signal; causal (z uses only strictly-prior stats)."""

    def __init__(self, universe, cap: float = 20.0, **params):
        self.universe = list(universe)
        self.cap = float(cap)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for root in self.universe:
            if root not in close_panel.columns:
                out[root] = pd.Series(np.nan, index=close_panel.index)
                continue
            close = close_panel[root].astype(float)
            r5 = close.pct_change(_REV_HORIZON, fill_method=None)
            mean = r5.rolling(_REV_WINDOW).mean().shift(1)
            std = r5.rolling(_REV_WINDOW).std().shift(1)
            z = (r5 - mean) / std.replace(0.0, np.nan)
            fc = (-z * _REV_SCALAR).clip(-self.cap, self.cap)
            out[root] = fc.fillna(0.0)
        return pd.DataFrame(out).reindex(columns=self.universe)
