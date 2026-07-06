"""Daily spot-FX panel + per-currency USD-conversion panel.

load_fx_daily_panel builds a (pair, {open,high,low,close,ret}) MultiIndex daily
panel from the fx_daily/ cache. build_quote_usd_panel derives, for each pair, the daily rate
that converts its QUOTE currency into USD -- USD legs are read directly
(EURUSD), inverted (USDJPY -> 1/rate), or sourced from another pair's USD leg
for true crosses (EURGBP -> GBPUSD). A missing USD leg is a hard error: silent
mis-conversion is never acceptable.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from src.settings import get_local_storage_dir
from src.utils import logger


def load_fx_daily_panel(pairs: list[str], start: date, end: date) -> pd.DataFrame:
    """Load the daily OHLC+ret panel for `pairs` in [start, end].

    `ret` is each pair's close.pct_change() on its OWN native index, computed
    BEFORE cross-pair alignment. This means a date gap in one pair's calendar
    does not NaN-contaminate the following row of another pair's `ret`.
    """
    base = Path(get_local_storage_dir()) / "fx_daily"
    frames: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        sym_dir = base / f"symbol={pair}"
        if not sym_dir.exists() or not any(sym_dir.glob("**/*.parquet")):
            logger.warning(f"[load_fx_daily_panel] no fx_daily data for {pair}")
            continue
        pdf = pl.scan_parquet(sym_dir / "**/*.parquet").collect().to_pandas()
        pdf["fx_date"] = pd.to_datetime(pdf["fx_date"]).dt.date
        pdf = pdf[(pdf["fx_date"] >= start) & (pdf["fx_date"] <= end)]
        if pdf.empty:
            continue
        pdf = pdf.set_index("fx_date").sort_index()
        frames[pair] = pdf[["open", "high", "low", "close"]].astype(float)
    if not frames:
        raise FileNotFoundError(f"no fx_daily data for pairs {pairs} in {start}..{end}")
    fields = ("open", "high", "low", "close", "ret")
    per_pair = {}
    for p, df in frames.items():
        d = df.copy()
        d["ret"] = d["close"].pct_change(fill_method=None)
        per_pair[p] = d[list(fields)]
    panel = pd.concat(per_pair, axis=1).sort_index()
    panel.columns = pd.MultiIndex.from_tuples(
        [(p, f) for p in per_pair for f in fields])
    return panel


def _currency_to_usd(currency: str, close_panel: pd.DataFrame) -> pd.Series:
    if currency == "USD":
        idx = close_panel.index
        return pd.Series(1.0, index=idx)
    pairs = {c[0] for c in close_panel.columns}
    direct = f"{currency}USD"
    inverse = f"USD{currency}"
    if direct in pairs:
        return close_panel[(direct, "close")].astype(float)
    if inverse in pairs:
        return 1.0 / close_panel[(inverse, "close")].astype(float)
    raise ValueError(
        f"cannot convert {currency} to USD: neither {direct} nor {inverse} in panel")


def build_quote_usd_panel(close_panel: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    for pair in pairs:
        quote = pair[3:]
        out[pair] = _currency_to_usd(quote, close_panel)
    return pd.DataFrame(out)
