"""Absolute (Carver-style) futures carry strategy.

forecast = clip(EWMA(carry) / annualized_price_vol * carry_scalar, -cap, cap),
per instrument, sourced from the per-root carry cache (build_carry_cache.py).
carry_scalar and ewma_span are FIXED doctrine constants -- never optimized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from src.data.futures.asset_class import asset_class_for
from src.data.futures.paths import carry_dir
from src.features.volatility import close_to_close_rv

_SQRT252 = np.sqrt(252.0)
_XS_SCALE = 10.0  # doctrine: maps a same-day cross-sectional z-score to forecast units


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


class FuturesCarryXSStrategy(FuturesCarryStrategy):
    """Cross-sectional carry: absolute carry forecast demeaned WITHIN asset-class
    each day (removes the common directional carry bet), z-scored by the same-day
    within-class dispersion, scaled to forecast units, clipped. Same-day stats only
    -> causal. Singleton/empty classes contribute 0 (no relative-value bet)."""
    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        raw = super().forecast_panel(close_panel)  # per-root absolute carry forecasts
        groups: dict[str, list[str]] = {}
        for r in self.universe:
            groups.setdefault(asset_class_for(r), []).append(r)
        out = pd.DataFrame(0.0, index=raw.index, columns=self.universe)
        for _, roots in groups.items():
            if len(roots) < 2:
                continue  # singleton class: no relative-value bet, stays 0.0
            block = raw[roots]                          # dates x class-roots
            mean = block.mean(axis=1)                   # same-day within-class mean
            std = block.std(axis=1)                     # same-day within-class dispersion
            valid = block.notna().all(axis=1)            # exclude warmup/missing-data rows
            z = block.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
            zero_dispersion = valid & std.eq(0.0)        # real data, no cross-sectional spread
            z = z.where(~zero_dispersion, 0.0)           # -> no relative-value bet, not NaN
            out[roots] = (z * _XS_SCALE).clip(-self.cap, self.cap)
        return out.reindex(columns=self.universe)
