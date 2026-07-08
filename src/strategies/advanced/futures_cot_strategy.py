"""#37 CoT positioning tilt (Legacy report).

Net-speculator (non-commercial) positioning treated momentum-like: normalize net
position to its trailing 3-year range and tilt the root by it. The CFTC Tuesday
snapshot is PUBLISHED the following Friday, so the signal is lagged +3 days to
its publication date (causal). Self-loads alt_data/cot/<root>/legacy_weekly.parquet."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from src.settings import get_local_storage_dir

_RANGE_YEARS = 3
_COT_SCALAR = 20.0   # doctrine: maps a [-1,1]-range normalized position to forecast units


def _publication_lag(snapshot: pd.Timestamp) -> pd.Timestamp:
    return snapshot + pd.Timedelta(days=3)  # Tuesday snapshot -> Friday publication


class FuturesCoTTiltStrategy:
    def __init__(self, universe, cap: float = 20.0, **params):
        self.universe = list(universe)
        self.cap = float(cap)

    def _load_cot(self, root: str):
        fp = get_local_storage_dir() / "alt_data" / "cot" / root / "legacy_weekly.parquet"
        if not fp.exists():
            return None
        return pl.read_parquet(fp).to_pandas()

    def _root_forecast(self, close: pd.Series, cot: pd.DataFrame) -> pd.Series:
        c = cot.copy()
        c["report_date"] = pd.to_datetime(c["report_date"])
        net = (c["noncommercial_long"] - c["noncommercial_short"]).astype(float)
        net.index = c["report_date"].map(_publication_lag)  # lag to publication
        net = net.sort_index()
        # normalize to trailing 3-year range, centered to [-1, 1]
        win = _RANGE_YEARS * 52
        lo = net.rolling(win, min_periods=win // 2).min()
        hi = net.rolling(win, min_periods=win // 2).max()
        norm = (2.0 * (net - lo) / (hi - lo).replace(0.0, np.nan)) - 1.0
        daily = norm.reindex(close.index, method="ffill")
        return (daily * _COT_SCALAR).clip(-self.cap, self.cap).fillna(0.0)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for root in self.universe:
            if root not in close_panel.columns:
                out[root] = pd.Series(0.0, index=close_panel.index)
                continue
            cot = self._load_cot(root)
            if cot is None or "noncommercial_long" not in cot.columns:
                out[root] = pd.Series(0.0, index=close_panel.index)
                continue
            out[root] = self._root_forecast(close_panel[root].astype(float), cot)
        return pd.DataFrame(out).reindex(columns=self.universe)
