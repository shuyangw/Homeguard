"""Unified leading-indicator loader.

Joins vix_term, hy_oas, breadth, skew on date. Forward-fills gaps up to
2 trading days (FRED weekly publishing schedules; CBOE holiday gaps).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.leading_indicators.vix_term import load_vix_term
from src.data.leading_indicators.fred_hy_oas import load_hy_oas
from src.data.leading_indicators.breadth import load_breadth
from src.data.leading_indicators.skew import load_skew
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_FFILL_DAYS = 2


def load_leading_indicators(
    start: datetime,
    end: datetime,
    cache: bool = True,
) -> pd.DataFrame:
    """Load all 4 leading indicators joined on date.

    Returns DataFrame indexed by date with columns:
    - vix_close, vix3m_close, vix_term_ratio (from vix_term)
    - hy_oas (from fred_hy_oas)
    - breadth_pct, n_constituents (from breadth)
    - skew_close (from skew)

    Forward-fills gaps up to MAX_FFILL_DAYS (handles FRED weekly schedule
    and CBOE holiday closures). Drops rows where any indicator has > 2
    consecutive missing days.
    """
    vix = load_vix_term(start, end, cache=cache)
    oas = load_hy_oas(start, end, cache=cache)
    brd = load_breadth(start, end, cache=cache)
    skw = load_skew(start, end, cache=cache)

    df = vix.join(oas, how='outer').join(brd, how='outer').join(skw, how='outer')
    df = df.sort_index()
    df = df.ffill(limit=MAX_FFILL_DAYS)
    df = df.dropna()

    logger.info(f'[+] leading_indicators: joined {len(df)} rows over {start.date()}..{end.date()}')
    return df
