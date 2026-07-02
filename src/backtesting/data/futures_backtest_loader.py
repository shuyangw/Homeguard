"""Daily multi-instrument panel for futures backtests.

Ratio-adjusted continuous daily closes per root, joined on date, with daily
pct returns. The .v.0 volume-roll already removes roll discontinuities, so
pct_change on the ratio-adjusted close is a clean return series.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.data.continuous_contract_loader import ContinuousContractDataLoader


def load_daily_panel(roots: list[str], start: date, end: date) -> pd.DataFrame:
    loader = ContinuousContractDataLoader()
    frames = {}
    for root in roots:
        d = loader.aggregate_to_daily(root, method="ratio_adjusted", start=start, end=end)
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
