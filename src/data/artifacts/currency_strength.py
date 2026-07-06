from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import pandas as pd
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.utils import logger


def aggregate_currency_returns(rets: pd.DataFrame) -> pd.DataFrame:
    contrib: dict[str, list[pd.Series]] = {}
    for pair in rets.columns:
        base, quote = pair[:3], pair[3:]
        r = rets[pair]
        contrib.setdefault(base, []).append(r)
        contrib.setdefault(quote, []).append(-r)
    out = {ccy: pd.concat(series, axis=1).mean(axis=1) for ccy, series in contrib.items()}
    return pd.DataFrame(out)


def currency_returns(close_panel: pd.DataFrame) -> pd.DataFrame:
    return aggregate_currency_returns(close_panel.pct_change(fill_method=None))


class CurrencyStrength(ArtifactBuilder):
    name = "currency_strength"
    output_subdir = "currency_strength"

    def inputs(self) -> list[str]:
        return ["daily_ohlc_cache"]

    def build(self, start: date, end: date) -> Path:
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        panel = load_fx_daily_panel(DailyOhlcCache().target_pairs(), start, end)
        rets = panel.xs("ret", axis=1, level=1)
        cr = aggregate_currency_returns(rets)
        strength = cr.cumsum()
        long = strength.reset_index().melt(id_vars=strength.index.name or "index",
                                           var_name="currency", value_name="strength")
        long.columns = ["date", "currency", "strength"]
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "strength.parquet.tmp"
        pl.from_pandas(long).write_parquet(tmp)
        os.replace(tmp, out_dir / "strength.parquet")
        logger.info(f"[currency_strength] wrote {len(long)} rows")
        return out_dir
