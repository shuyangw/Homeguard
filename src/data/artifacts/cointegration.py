from __future__ import annotations
from datetime import date
from itertools import combinations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.stattools import coint
from src.data.artifacts.base import ArtifactBuilder
from src.utils import logger


def ou_half_life(spread: pd.Series) -> float:
    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lag = lag.loc[delta.index]
    beta = np.polyfit(lag.values, delta.values, 1)[0]
    if beta >= 0:
        return float("inf")
    return float(-np.log(2) / np.log(1 + beta))


def test_pair(a: pd.Series, b: pd.Series) -> dict:
    df = pd.concat([a, b], axis=1).dropna()
    _, pval, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
    hedge = np.polyfit(df.iloc[:, 1].values, df.iloc[:, 0].values, 1)[0]
    spread = df.iloc[:, 0] - hedge * df.iloc[:, 1]
    return {"adf_pvalue": float(pval), "hedge_ratio": float(hedge),
            "half_life": ou_half_life(spread)}


test_pair.__test__ = False  # not a pytest test despite the name -- public API function


class Cointegration(ArtifactBuilder):
    name = "cointegration"
    output_subdir = "cointegration"

    def inputs(self) -> list[str]:
        return ["daily_ohlc_cache"]

    def _shares_one_currency(self, a: str, b: str) -> bool:
        return len({a[:3], a[3:]} & {b[:3], b[3:]}) <= 1

    def build(self, start: date, end: date) -> Path:
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        pairs = DailyOhlcCache().target_pairs()
        panel = load_fx_daily_panel(pairs, start, end)
        close = panel.xs("close", axis=1, level=1)
        rows = []
        for a, b in combinations(close.columns, 2):
            if not self._shares_one_currency(a, b):
                continue
            try:
                res = test_pair(np.log(close[a]), np.log(close[b]))
            except Exception as e:
                logger.warning(f"[cointegration] {a}/{b} failed: {e}")
                continue
            if res["adf_pvalue"] < 0.05 and 5 <= res["half_life"] <= 25:
                rows.append({"pair_a": a, "pair_b": b, **res})
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        table = pl.DataFrame(rows) if rows else pl.DataFrame(
            {"pair_a": [], "pair_b": [], "adf_pvalue": [], "hedge_ratio": [], "half_life": []})
        tmp = out_dir / "pairs.parquet.tmp"
        table.write_parquet(tmp)
        os.replace(tmp, out_dir / "pairs.parquet")
        logger.info(f"[cointegration] {len(rows)} tradeable pairs")
        return out_dir
