"""Conditioning-overlay futures strategies (SP-A).

#13 carry-trend agreement gate: pass the trend forecast only where carry agrees
in sign; otherwise flat. A fundamentally-supported trend (carry and trend agree)
has better continuation odds than a flow-driven one."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.advanced.futures_signal_base import ConditioningOverlayStrategy


class FuturesCarryTrendStrategy(ConditioningOverlayStrategy):
    """Trend forecast gated by carry-sign agreement."""

    def __init__(self, universe, cap: float = 20.0, **params):
        super().__init__(universe, base_name="CarverMomentum", cond_name="FuturesCarry",
                         cap=cap)

    def _combine(self, base_fc: pd.DataFrame, cond_fc: pd.DataFrame) -> pd.DataFrame:
        agree = np.sign(base_fc) == np.sign(cond_fc)
        return base_fc.where(agree, 0.0)
