"""#49 Perp-funding crypto carry.

Rich positive perpetual funding (longs pay shorts) signals a paid crypto-carry
premium; tilt the CME BTC/ETH roots by the smoothed annualized funding scaled by
inverse price vol. Self-loads the funding cache (alt_data/funding). Funding is
realized (past), so the signal is causal for a next-day position."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from src.features.volatility import close_to_close_rv
from src.settings import get_local_storage_dir

_SQRT252 = np.sqrt(252.0)
_FUNDING_SCALAR = 2.0   # doctrine: maps annualized funding units to forecast units
_EWMA_SPAN = 10


class FuturesFundingCarryStrategy:
    def __init__(self, universe, cap: float = 20.0, **params):
        self.universe = list(universe)
        self.cap = float(cap)

    def _load_funding(self, root: str):
        fp = get_local_storage_dir() / "alt_data" / "funding" / root / "funding.parquet"
        if not fp.exists():
            return None
        df = pl.read_parquet(fp).to_pandas()
        return pd.Series(df["funding_annualized"].to_numpy(), index=pd.to_datetime(df["date"]))

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for root in self.universe:
            if root not in close_panel.columns:
                out[root] = pd.Series(0.0, index=close_panel.index)
                continue
            funding = self._load_funding(root)
            if funding is None:
                out[root] = pd.Series(0.0, index=close_panel.index)
                continue
            close = close_panel[root].astype(float)
            f = funding.reindex(close.index).ffill()
            f_sm = f.ewm(span=_EWMA_SPAN, adjust=False).mean()
            rets = close.pct_change(fill_method=None)
            ann_vol = close_to_close_rv(rets, 25, annualization_factor=1) * _SQRT252
            fc = (f_sm / ann_vol.replace(0.0, np.nan)) * _FUNDING_SCALAR
            out[root] = fc.clip(-self.cap, self.cap).fillna(0.0)
        return pd.DataFrame(out).reindex(columns=self.universe)
