"""NYSE A-D breadth: % of S&P 500 constituents above their 50-day MA.

Market breadth deterioration precedes drawdowns. Computed from
historical S&P 500 constituent closes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

SP500_UNIVERSE_CSV = Path('config/universes/sp500-2025.csv')
MA_WINDOW = 50


def CACHE_PATH() -> Path:
    return get_local_storage_dir() / 'alt_data' / 'leading_indicators' / 'breadth.parquet'


def load_breadth(
    start: datetime,
    end: datetime,
    cache: bool = True,
) -> pd.DataFrame:
    """Load NYSE A-D breadth: % S&P 500 above 50-day MA.

    Returns DataFrame indexed by date with columns:
    - breadth_pct (% of constituents above 50-day MA, 0.0-1.0)
    - n_constituents (count of symbols with valid data on the day)
    """
    path = CACHE_PATH()
    if cache and path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
            logger.info(f'[+] breadth: serving from cache {path}')
            return cached.loc[pd.Timestamp(start):pd.Timestamp(end)]

    logger.info(f'[+] breadth: computing from {SP500_UNIVERSE_CSV} for {start.date()}..{end.date()}')
    if not SP500_UNIVERSE_CSV.exists():
        raise FileNotFoundError(SP500_UNIVERSE_CSV)

    # Need MA_WINDOW + buffer extra trading days before `start` so the first
    # output day has a valid 50-day MA. ~75 calendar days covers 50 trading days.
    panel_start = start - timedelta(days=MA_WINDOW + 30)
    from src.research.ramp_phase4.data import load_universe_panel
    panel = load_universe_panel(SP500_UNIVERSE_CSV, panel_start, end)
    sym_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    closes = panel[sym_cols]
    # Panel may carry pd.NA / object dtype for symbols missing on a given day.
    # Coerce to float64 so rolling() can aggregate.
    closes = closes.apply(pd.to_numeric, errors='coerce').astype('float64')

    ma = closes.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    above_ma = (closes > ma)
    valid_mask = closes.notna() & ma.notna()
    n_const = valid_mask.sum(axis=1)
    # Avoid divide-by-zero on early rows where no symbol has a valid MA yet.
    pct_above = (above_ma & valid_mask).sum(axis=1) / n_const.where(n_const > 0)

    df = pd.DataFrame({
        'breadth_pct': pct_above,
        'n_constituents': n_const,
    })
    df = df.dropna(subset=['breadth_pct'])
    # Trim back to requested range.
    df = df.loc[pd.Timestamp(start):pd.Timestamp(end)]

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f'[+] breadth: cached {len(df)} rows to {path}')

    return df
