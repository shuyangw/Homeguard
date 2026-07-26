"""Commodity price series aligned to the FX trading calendar.

Terms-of-trade strategies need the price of the commodity a country EXPORTS
(Brent for CAD/NOK, gold for AUD/NZD). Brent comes from the keyless yfinance
cache (`alt_data/oil/BRENT`); gold is XAUUSD from the validated FX daily cache.

Alignment is causal: a commodity series is reindexed onto the FX dates and
FORWARD-filled only. Commodity and FX holidays do not coincide, so a missing
commodity day carries the last PUBLISHED price rather than interpolating (which
would blend a future observation into the gap).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.settings import get_local_storage_dir
from src.utils import logger

# spot pair -> (commodity key, sign of the pair's response to that commodity)
# Signs are FIXED by the pre-registration (2026-07-25 Tier B) and must not be
# flipped after seeing results: oil up -> CAD/NOK strengthen -> USDxxx DOWN;
# gold up -> AUD/NZD strengthen -> AUDUSD/NZDUSD UP.
COMMODITY_LEGS: dict[str, tuple[str, int]] = {
    "USDCAD": ("oil", -1),
    "USDNOK": ("oil", -1),
    "AUDUSD": ("gold", +1),
    "NZDUSD": ("gold", +1),
}


def _brent_path() -> Path:
    return Path(get_local_storage_dir()) / "alt_data" / "oil" / "BRENT" / "daily.parquet"


def load_commodity_series(name: str, index: pd.Index) -> pd.Series:
    """One commodity's close, causally aligned to `index` (forward-fill only)."""
    idx_dt = pd.to_datetime(pd.Index(index))
    if name == "oil":
        fp = _brent_path()
        if not fp.exists():
            logger.warning(f"[commodities] Brent cache missing: {fp}")
            return pd.Series(float("nan"), index=index)
        raw = pd.read_parquet(fp)
        s = pd.Series(raw["close"].to_numpy(dtype=float),
                      index=pd.to_datetime(raw["date"]))
    elif name == "gold":
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        d0, d1 = idx_dt.min().date(), idx_dt.max().date()
        panel = load_fx_daily_panel(["XAUUSD"], d0, d1)
        s = panel[("XAUUSD", "close")].astype(float)
        s.index = pd.to_datetime(s.index)
    else:
        raise ValueError(f"unknown commodity {name!r}")

    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    out = s.reindex(idx_dt.union(s.index)).ffill().reindex(idx_dt)
    out.index = index
    return out


def load_commodity_panel(pairs: list[str], index: pd.Index) -> pd.DataFrame:
    """{pair: its commodity's price} for the pairs that have a mapped commodity."""
    needed = {COMMODITY_LEGS[p][0] for p in pairs if p in COMMODITY_LEGS}
    series = {c: load_commodity_series(c, index) for c in needed}
    return pd.DataFrame(
        {p: series[COMMODITY_LEGS[p][0]] for p in pairs if p in COMMODITY_LEGS},
        index=index)
