from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import numpy as np
import pandas as pd
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.utils import logger


def dollar_factor(returns_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Extract PC1 (the "dollar factor") from a date x pair returns panel.

    Returns (pc1, residuals) where residuals are the standardized returns
    with the PC1 projection removed. `residuals` shares `returns_df`'s shape
    and column order; rows are NaN wherever any pair had a missing return.
    """
    X = returns_df.dropna(how="any")
    Z = (X - X.mean()) / X.std(ddof=0)
    _, _, vt = np.linalg.svd(Z.values, full_matrices=False)
    w = vt[0]
    pc1_values = Z.values @ w
    pc1 = pd.Series(pc1_values, index=X.index, name="pc1").reindex(returns_df.index)
    proj = np.outer(pc1_values, w)
    residuals = pd.DataFrame(Z.values - proj, index=X.index, columns=X.columns)
    residuals = residuals.reindex(returns_df.index)
    return pc1, residuals


class PcaDollar(ArtifactBuilder):
    name = "pca_dollar"
    output_subdir = "pca_dollar"

    def inputs(self) -> list[str]:
        return ["daily_ohlc_cache"]

    def build(self, start: date, end: date) -> Path:
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        pairs = [p for p in DailyOhlcCache().target_pairs()
                 if p.endswith("USD") or p.startswith("USD")]
        panel = load_fx_daily_panel(pairs, start, end)
        rets = panel.xs("ret", axis=1, level=1)
        pc1, resid = dollar_factor(rets)
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, obj in [("factor", pc1.reset_index()), ("residuals", resid.reset_index())]:
            tmp = out_dir / f"{name}.parquet.tmp"
            pl.from_pandas(obj).write_parquet(tmp)
            os.replace(tmp, out_dir / f"{name}.parquet")
        logger.info(f"[pca_dollar] wrote factor + residuals ({len(pc1)} obs, {len(pairs)} pairs)")
        return out_dir
