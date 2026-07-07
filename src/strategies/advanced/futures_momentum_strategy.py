"""Cross-sectional and time-series momentum/reversal futures strategies (SP-A).

#3 XS commodity momentum: rank trailing 12-1 return across the commodity block.
#23 short-horizon reversal: forecast ~ -z(5-day return) on liquid index roots."""
from __future__ import annotations

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
