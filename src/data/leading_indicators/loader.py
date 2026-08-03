"""Unified leading-indicator loader.

Joins vix_term, hy_proxy, breadth, skew on date. Forward-fills gaps up to
2 trading days (CBOE/NYSE holiday gaps).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.leading_indicators.vix_term import load_vix_term
from src.data.leading_indicators.hy_proxy import load_hy_proxy
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
    - hyg_close, ief_close, hy_proxy_ratio (from hy_proxy)
    - breadth_pct, n_constituents (from breadth)
    - skew_close (from skew)

    HY proxy (HYG/IEF ratio) substituted for FRED HY OAS in 2026-05
    after ICE licensing change truncated the FRED series. See
    Amendment 5 in the WS-3d spec.
    """
    vix = load_vix_term(start, end, cache=cache)
    hy = load_hy_proxy(start, end, cache=cache)
    brd = load_breadth(start, end, cache=cache)
    skw = load_skew(start, end, cache=cache)

    df = vix.join(hy, how='outer').join(brd, how='outer').join(skw, how='outer')
    df = df.sort_index()
    df = df.ffill(limit=MAX_FFILL_DAYS)
    df = df.dropna()

    logger.info(f'[+] leading_indicators: joined {len(df)} rows over {start.date()}..{end.date()}')
    return df
