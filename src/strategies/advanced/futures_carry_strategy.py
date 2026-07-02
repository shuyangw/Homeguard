"""Absolute (Carver-style) futures carry strategy.

forecast = clip(EWMA(carry) / annualized_price_vol * carry_scalar, -cap, cap),
per instrument, sourced from the per-root carry cache (build_carry_cache.py).
carry_scalar and ewma_span are FIXED doctrine constants -- never optimized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from src.data.futures.paths import carry_dir
from src.features.volatility import close_to_close_rv

_SQRT252 = np.sqrt(252.0)


class FuturesCarryStrategy:
    def __init__(self, universe, carry_scalar: float = 30.0, ewma_span: int = 10,
                 cap: float = 20.0, **params):
        self.universe = list(universe)
        self.carry_scalar = float(carry_scalar)
        self.ewma_span = int(ewma_span)
        self.cap = float(cap)

    def _load_carry(self, root: str):
        fp = carry_dir() / f"{root}.parquet"
        if not fp.exists():
            return None
        df = pl.read_parquet(fp).to_pandas()
        return pd.Series(df["carry"].to_numpy(),
                         index=pd.to_datetime(df["date"]))

    def _forecast_root(self, close: pd.Series, carry: pd.Series) -> pd.Series:
        carry = carry.reindex(close.index).ffill()
        carry_sm = carry.ewm(span=self.ewma_span, adjust=False).mean()
        rets = close.pct_change(fill_method=None)
        ann_vol = close_to_close_rv(rets, 25, annualization_factor=1) * _SQRT252
        fc = (carry_sm / ann_vol) * self.carry_scalar
        return fc.clip(-self.cap, self.cap)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for root in self.universe:
            if root not in close_panel.columns:
                out[root] = pd.Series(np.nan, index=close_panel.index)
                continue
            carry = self._load_carry(root)
            if carry is None:
                out[root] = pd.Series(np.nan, index=close_panel.index)
                continue
            out[root] = self._forecast_root(close_panel[root], carry)
        return pd.DataFrame(out).reindex(columns=self.universe)
