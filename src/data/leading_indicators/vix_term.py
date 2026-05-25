"""VIX term structure: VIX / VIX3M daily ratio.

Term-structure inversion (ratio > 1) precedes drawdowns historically.
Data from yfinance: ^VIX (spot) and ^VIX3M (3-month). Daily close-to-close.
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
    return get_local_storage_dir() / 'alt_data' / 'leading_indicators' / 'vix_term.parquet'


def load_vix_term(
    start: datetime,
    end: datetime,
    cache: bool = True,
) -> pd.DataFrame:
    """Load VIX/VIX3M daily ratio.

    Returns a DataFrame indexed by date (datetime64) with columns:
    - vix_close
    - vix3m_close
    - vix_term_ratio (vix / vix3m)

    If cache is True and the cache file exists with the requested range,
    serve from cache. Otherwise download from yfinance and update cache.
    """
    path = CACHE_PATH()
    cached: Optional[pd.DataFrame] = None
    if cache and path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
            logger.info(f'[+] vix_term: serving from cache {path}')
            return cached.loc[pd.Timestamp(start):pd.Timestamp(end)]

    logger.info(f'[+] vix_term: downloading from yfinance {start.date()} to {end.date()}')
    vix = yf.Ticker('^VIX').history(start=start, end=end, auto_adjust=False)
    vix3m = yf.Ticker('^VIX3M').history(start=start, end=end, auto_adjust=False)
    if vix.empty or vix3m.empty:
        raise RuntimeError(f'yfinance returned empty for VIX or VIX3M ({start.date()}..{end.date()})')

    # yfinance returns tz-aware indexes that may differ across tickers
    # (^VIX is sometimes Chicago, ^VIX3M is New York). Normalize both to
    # tz-naive midnight dates before joining so closes align on trade date.
    vix_close = vix['Close'].copy()
    vix_close.index = pd.to_datetime(pd.DatetimeIndex(vix_close.index).date)
    vix3m_close = vix3m['Close'].copy()
    vix3m_close.index = pd.to_datetime(pd.DatetimeIndex(vix3m_close.index).date)

    df = pd.DataFrame({
        'vix_close': vix_close,
        'vix3m_close': vix3m_close,
    })
    df['vix_term_ratio'] = df['vix_close'] / df['vix3m_close']
    df = df.dropna()

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f'[+] vix_term: cached {len(df)} rows to {path}')

    return df
