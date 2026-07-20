"""Base for spread strategies: produce a per-date active-spread book plus the
trailing spread-vol map the simulator needs. Subclasses implement the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class SpreadStrategy:
    vol_window = 60

    def spread_book(self, close_panel: pd.DataFrame):
        raise NotImplementedError

    def _spread_sigma(self, close_panel, leg_a, leg_b, hedge_ratio, upto_idx):
        # trailing daily std of r_a - beta*r_b over vol_window, causal (<= upto_idx)
        ra = close_panel[leg_a].pct_change(fill_method=None)
        rb = close_panel[leg_b].pct_change(fill_method=None)
        s = (ra - hedge_ratio * rb).iloc[max(0, upto_idx - self.vol_window):upto_idx + 1]
        v = float(s.std())
        return v if np.isfinite(v) and v > 0 else None
