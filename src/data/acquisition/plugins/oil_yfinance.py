from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import polars as pl
import yfinance as yf
from src.settings import get_local_storage_dir
from src.utils import logger

REQUIRES_KEY = None
_TICKER = "BZ=F"


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end, progress=False)


def fetch_brent(start: str, end: str, write: bool = True) -> pd.DataFrame:
    raw = _download(_TICKER, start, end)
    if raw.empty:
        raise ValueError("Brent download returned empty")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = pd.DataFrame({"date": pd.to_datetime(close.index).date,
                        "close": close.astype(float).values})
    if write:
        d = get_local_storage_dir() / "alt_data" / "oil" / "BRENT"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "daily.parquet.tmp"
        pl.from_pandas(out).write_parquet(tmp)
        os.replace(tmp, d / "daily.parquet")
        logger.info(f"[oil] wrote {len(out)} Brent rows")
    return out
