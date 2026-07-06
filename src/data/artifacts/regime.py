from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import numpy as np
import pandas as pd
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.utils import logger

TREND_HI = 1.2
MR_LO = 0.8


def classify_atr_regime(atr_fast: float, atr_slow: float) -> str:
    if atr_slow <= 0:
        return "NEUTRAL"
    ratio = atr_fast / atr_slow
    if ratio > TREND_HI:
        return "TREND"
    if ratio < MR_LO:
        return "MR"
    return "NEUTRAL"


def _true_range(high, low, close_prev):
    return np.maximum(high - low, np.maximum((high - close_prev).abs(), (low - close_prev).abs()))


class Regime(ArtifactBuilder):
    name = "regime"
    output_subdir = "regime"

    def inputs(self) -> list[str]:
        return ["daily_ohlc_cache", "vol_surface"]

    def build(self, start: date, end: date) -> Path:
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        panel = load_fx_daily_panel(DailyOhlcCache().target_pairs(), start, end)
        rows = []
        for pair in {c[0] for c in panel.columns}:
            sub = panel[pair]
            tr = _true_range(sub["high"], sub["low"], sub["close"].shift(1))
            atr_fast = tr.rolling(14).mean()
            atr_slow = tr.rolling(100).mean()
            for d in sub.index:
                af, aslow = atr_fast.get(d, np.nan), atr_slow.get(d, np.nan)
                if pd.isna(af) or pd.isna(aslow):
                    continue
                rows.append({"date": d, "pair": pair, "atr_ratio": float(af / aslow),
                             "state": classify_atr_regime(float(af), float(aslow))})
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "regime.parquet.tmp"
        pl.from_pandas(pd.DataFrame(rows)).write_parquet(tmp)
        os.replace(tmp, out_dir / "regime.parquet")
        logger.info(f"[regime] wrote {len(rows)} rows")
        return out_dir
