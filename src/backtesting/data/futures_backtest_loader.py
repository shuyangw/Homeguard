"""Daily multi-instrument panel for futures backtests.

Ratio-adjusted continuous daily closes per root, joined on date, with daily
pct returns. The .v.0 volume-roll already removes roll discontinuities, so
pct_change on the ratio-adjusted close is a clean return series.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.data.continuous_contract_loader import ContinuousContractDataLoader
from src.utils import logger


def load_daily_panel(roots: list[str], start: date, end: date) -> pd.DataFrame:
    """Load a ratio-adjusted daily close/return panel for `roots` in [start, end].

    A root is silently EXCLUDED from the returned panel (not fatal) if it has
    no data in this window (e.g. a micro contract that had not yet been
    listed -- CME Micro E-mini S&P/Nasdaq/Russell/Dow launched 2019-05) or if
    `ContinuousContractDataLoader` raises while building it (e.g. a
    roll-calendar data-quality issue for one specific root/window). Each
    exclusion is logged as a WARNING. Raises only if NO requested root
    produced usable data for the window.
    """
    loader = ContinuousContractDataLoader()
    frames = {}
    for root in roots:
        try:
            d = loader.aggregate_to_daily(root, method="ratio_adjusted", start=start, end=end)
        except Exception as e:
            logger.warning(f"[load_daily_panel] skipping {root} in {start}..{end}: {type(e).__name__}: {e}")
            continue
        if d.is_empty():
            continue
        pdf = d.select(["timestamp", "close"]).to_pandas()
        pdf["date"] = pd.to_datetime(pdf["timestamp"]).dt.date
        pdf = pdf.set_index("date")["close"]
        frames[root] = pdf
    if not frames:
        raise FileNotFoundError(f"no continuous daily data for roots {roots} in {start}..{end}")
    close = pd.DataFrame(frames).sort_index()
    ret = close.pct_change()
    panel = pd.concat({r: pd.DataFrame({"close": close[r], "ret": ret[r]}) for r in close.columns}, axis=1)
    panel.columns = pd.MultiIndex.from_tuples([(r, f) for r in close.columns for f in ("close", "ret")])
    return panel
