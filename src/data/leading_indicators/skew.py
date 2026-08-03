"""CBOE SKEW index: tail-risk options pricing.

Higher SKEW = options pricing greater tail risk; precedes drawdowns
historically. Data from yfinance: ^SKEW. Daily close.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def CACHE_PATH() -> Path:
    return get_local_storage_dir() / 'alt_data' / 'leading_indicators' / 'skew.parquet'


def load_skew(
    start: datetime,
    end: datetime,
    cache: bool = True,
) -> pd.DataFrame:
    """Load CBOE SKEW daily closes.

    Returns DataFrame indexed by date with column:
    - skew_close
    """
    path = CACHE_PATH()
    if cache and path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
            logger.info(f'[+] skew: serving from cache {path}')
            return cached.loc[pd.Timestamp(start):pd.Timestamp(end)]

    logger.info(f'[+] skew: downloading from yfinance {start.date()} to {end.date()}')
    skew = yf.Ticker('^SKEW').history(start=start, end=end, auto_adjust=False)
    if skew.empty:
        raise RuntimeError(f'yfinance returned empty for ^SKEW ({start.date()}..{end.date()})')

    df = pd.DataFrame({'skew_close': skew['Close']})
    df.index = pd.to_datetime(df.index.date)
    df = df.dropna()

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f'[+] skew: cached {len(df)} rows to {path}')

    return df
