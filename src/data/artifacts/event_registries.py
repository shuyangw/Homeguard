from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import pandas as pd
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.utils import logger


def label_vol_spikes(returns: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    rows = []
    for pair in returns.columns:
        r = returns[pair].dropna()
        roll_std = r.rolling(60, min_periods=20).std()
        zscore = r / roll_std
        hits = zscore[zscore.abs() > z]
        for d, val in hits.items():
            rows.append({"date": d, "pair": pair, "z": float(val)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "pair", "z"])


class EventRegistries(ArtifactBuilder):
    name = "event_registries"
    output_subdir = "event_registries"

    def inputs(self) -> list[str]:
        return ["daily_ohlc_cache"]

    def build(self, start: date, end: date) -> Path:
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        panel = load_fx_daily_panel(DailyOhlcCache().target_pairs(), start, end)
        rets = panel.xs("ret", axis=1, level=1)
        spikes = label_vol_spikes(rets)
        _corr = rets.rolling(20).corr().dropna()  # scaffolded for strategy #40, not persisted
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "vol_spikes.parquet.tmp"
        pl.from_pandas(spikes).write_parquet(tmp)
        os.replace(tmp, out_dir / "vol_spikes.parquet")
        logger.info(f"[event_registries] {len(spikes)} vol spikes")
        return out_dir
