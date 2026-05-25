"""High-yield credit spread: FRED series BAMLH0A0HYM2 (HY OAS).

Bond market leads equity vol; HY OAS widens before drawdowns.
Data from FRED. Daily values.

History:
  Full history 1996-12-31 to present is available via the FRED REST API
  (https://api.stlouisfed.org/fred/series/observations) which requires a
  free API key. Set `FRED_API_KEY` in the environment to enable.

  Without an API key, this module falls back to `pandas_datareader`'s
  FRED reader which serves only the trailing ~3 years (unauthenticated
  rolling window). The fallback is sufficient for short-window smoke
  tests but NOT for the WS-3d 2017-present training window.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas_datareader.data import DataReader

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

SERIES_ID = 'BAMLH0A0HYM2'
FRED_API_URL = 'https://api.stlouisfed.org/fred/series/observations'


def CACHE_PATH() -> Path:
    return get_local_storage_dir() / 'alt_data' / 'leading_indicators' / 'hy_oas.parquet'


def _fetch_via_api(start: datetime, end: datetime, api_key: str) -> pd.DataFrame:
    """Fetch full HY OAS history via the FRED REST API (requires API key)."""
    import requests

    params = {
        'series_id': SERIES_ID,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start.strftime('%Y-%m-%d'),
        'observation_end': end.strftime('%Y-%m-%d'),
    }
    resp = requests.get(FRED_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    obs = resp.json().get('observations', [])
    rows = [
        (pd.Timestamp(o['date']), float(o['value']))
        for o in obs
        if o.get('value') not in (None, '.', '')
    ]
    if not rows:
        raise RuntimeError(f'FRED API returned no observations for {SERIES_ID}')
    df = pd.DataFrame(rows, columns=['date', 'hy_oas']).set_index('date')
    df.index = pd.to_datetime(df.index)
    return df


def _fetch_via_pdr(start: datetime, end: datetime) -> pd.DataFrame:
    """Fallback: pandas-datareader FRED reader (unauthenticated ~3y window)."""
    series = DataReader(SERIES_ID, 'fred', start, end)
    if series.empty:
        raise RuntimeError(f'FRED returned empty for {SERIES_ID} ({start.date()}..{end.date()})')
    df = pd.DataFrame({'hy_oas': series[SERIES_ID]})
    df.index = pd.to_datetime(df.index)
    return df


def load_hy_oas(
    start: datetime,
    end: datetime,
    cache: bool = True,
) -> pd.DataFrame:
    """Load HY OAS daily series.

    Returns DataFrame indexed by date with column:
    - hy_oas (BAMLH0A0HYM2 value, percent)
    """
    path = CACHE_PATH()
    if cache and path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
            logger.info(f'[+] hy_oas: serving from cache {path}')
            return cached.loc[pd.Timestamp(start):pd.Timestamp(end)]

    api_key = os.environ.get('FRED_API_KEY')
    if api_key:
        logger.info(f'[+] hy_oas: downloading via FRED REST API {start.date()} to {end.date()}')
        df = _fetch_via_api(start, end, api_key)
    else:
        logger.warning(
            '[!] hy_oas: FRED_API_KEY not set; falling back to pandas-datareader '
            '(rolling ~3y window). Set FRED_API_KEY for full 1996-present history.'
        )
        df = _fetch_via_pdr(start, end)

    df = df.dropna()

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f'[+] hy_oas: cached {len(df)} rows to {path}')

    return df
