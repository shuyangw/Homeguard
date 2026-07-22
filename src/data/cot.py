"""CFTC Commitments-of-Traders speculative-positioning panel for FX.

Loads the cached weekly net non-commercial positioning (`alt_data/cot/cot_fx.parquet`,
signed bullish-the-pair as net%OI), applies the publication lag, and exposes it as a
weekly panel indexed by the date the reading becomes ACTIVE. COT is a Tuesday snapshot
(date D) published the following Friday (D+3); a reading is made active at D+7 calendar
days (a conservative buffer past publication) so no consumer can see it before it was
public. Strategies compute their forecast on the weekly panel and forward-fill to daily.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.settings import get_local_storage_dir
from src.utils import logger

_PUBLICATION_LAG_DAYS = 7


def cot_path() -> Path:
    return Path(get_local_storage_dir()) / "alt_data" / "cot" / "cot_fx.parquet"


def load_cot_weekly_panel(pairs: list[str]) -> pd.DataFrame:
    """Weekly net%OI per pair, indexed by active date (snapshot + publication lag).

    Columns are the subset of `pairs` present in the cache; index is the active
    (publication-lagged) date, sorted ascending. Empty frame if the cache is absent.
    """
    fp = cot_path()
    if not fp.exists():
        logger.warning(f"[cot] cache missing: {fp}")
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    df["active"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=_PUBLICATION_LAG_DAYS)
    out: dict[str, pd.Series] = {}
    for pair in pairs:
        g = df[df["pair"] == pair].sort_values("active")
        if g.empty:
            continue
        out[pair] = pd.Series(g["noncomm_net_pct_oi"].to_numpy(), index=pd.DatetimeIndex(g["active"]))
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


def to_daily(weekly_forecast: pd.DataFrame, daily_index: pd.Index) -> pd.DataFrame:
    """Forward-fill a weekly (active-date-indexed) forecast onto a daily index.

    Each daily bar gets the most recent weekly value whose active date <= the bar,
    so the D+7 lag is preserved end-to-end (no lookahead)."""
    di = pd.to_datetime(pd.Index(daily_index))
    ff = weekly_forecast.sort_index().reindex(di.union(weekly_forecast.index)).ffill().reindex(di)
    ff.index = daily_index
    return ff
