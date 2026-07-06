from __future__ import annotations
import os
import pandas as pd
import polars as pl
import yfinance as yf
from src.settings import get_local_storage_dir
from src.utils import logger

REQUIRES_KEY = None
INDICES = {"SPX": "^GSPC", "STOXX50E": "^STOXX50E", "N225": "^N225"}


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end, progress=False)


def fetch_index(name: str, start: str, end: str, write: bool = True) -> pd.DataFrame:
    raw = _download(INDICES[name], start, end)
    if raw.empty:
        raise ValueError(f"{name} download returned empty")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = pd.DataFrame({"date": pd.to_datetime(close.index).date,
                        "close": close.astype(float).values})
    if write:
        d = get_local_storage_dir() / "alt_data" / "equity_index" / name
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "daily.parquet.tmp"
        pl.from_pandas(out).write_parquet(tmp)
        os.replace(tmp, d / "daily.parquet")
        logger.info(f"[equity_index] wrote {len(out)} {name} rows")
    return out
