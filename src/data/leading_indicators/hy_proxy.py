"""HY credit-spread proxy: HYG / IEF ratio.

HYG = iShares iBoxx US High Yield Corporate Bond ETF
IEF = iShares 7-10 Year Treasury Bond ETF

The ratio (HYG / IEF) inversely tracks HY credit spreads: when credit
markets stress, HYG falls relative to IEF, the ratio compresses.
Historical correlation with FRED's HY OAS (BAMLH0A0HYM2) is high
(>0.9 over typical post-2007 windows), but this is a PRICE-based
proxy not a yield-based measure.

Substituted for FRED BAMLH0A0HYM2 in 2026-05 after ICE Data Indices
licensing truncated the FRED series to a rolling 3-year window. See
docs/superpowers/specs/2026-05-25-ws3d-detector-replacement-design.md
Amendment 5.

Data from yfinance: HYG (2007-04-present), IEF (2002-07-present).
Daily close-to-close.
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
    return get_local_storage_dir() / 'alt_data' / 'leading_indicators' / 'hy_proxy.parquet'


def load_hy_proxy(
    start: datetime,
    end: datetime,
    cache: bool = True,
) -> pd.DataFrame:
    """Load HYG/IEF daily ratio.

    Returns DataFrame indexed by date (datetime64) with columns:
    - hyg_close
    - ief_close
    - hy_proxy_ratio (hyg_close / ief_close)

    If cache is True and the cache file exists with the requested range,
    serve from cache. Otherwise download from yfinance and update cache.
    """
    path = CACHE_PATH()
    cached: Optional[pd.DataFrame] = None
    if cache and path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
            logger.info(f'[+] hy_proxy: serving from cache {path}')
            return cached.loc[pd.Timestamp(start):pd.Timestamp(end)]

    logger.info(f'[+] hy_proxy: downloading HYG, IEF from yfinance {start.date()} to {end.date()}')
    hyg = yf.Ticker('HYG').history(start=start, end=end, auto_adjust=False)
    ief = yf.Ticker('IEF').history(start=start, end=end, auto_adjust=False)
    if hyg.empty or ief.empty:
        raise RuntimeError(f'yfinance returned empty for HYG or IEF ({start.date()}..{end.date()})')

    hyg_close = hyg['Close'].copy()
    hyg_close.index = pd.to_datetime(pd.DatetimeIndex(hyg_close.index).date)
    ief_close = ief['Close'].copy()
    ief_close.index = pd.to_datetime(pd.DatetimeIndex(ief_close.index).date)

    df = pd.DataFrame({
        'hyg_close': hyg_close,
        'ief_close': ief_close,
    })
    df['hy_proxy_ratio'] = df['hyg_close'] / df['ief_close']
    df = df.dropna()

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f'[+] hy_proxy: cached {len(df)} rows to {path}')

    return df
