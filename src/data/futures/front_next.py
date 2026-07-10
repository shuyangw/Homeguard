"""Per-root daily front (F1) / next (F2) settle series for calendar spreads.

Reuses CarryCalculator._find_front_second_close (the two most-active outright
contracts by daily volume) so the calendar spread trades the liquid F1/F2 pair,
matching the carry sleeve's contract selection.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.data.carry_calculator import CarryCalculator
from src.data.futures.paths import front_next_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

_COLUMNS = ["date", "front_symbol", "f1", "second_symbol", "f2", "months"]


def front_next_history(root: str, start: date, end: date) -> pd.DataFrame:
    cache_fp = front_next_dir() / f"{root}.parquet"
    if cache_fp.exists():
        cached = pd.read_parquet(cache_fp)
        mask = (cached["date"] >= pd.Timestamp(start)) & (cached["date"] <= pd.Timestamp(end))
        window = cached.loc[mask]
        if not window.empty:
            return window.reset_index(drop=True)

    calc = CarryCalculator()
    rows: list[dict] = []
    d = start
    while d <= end:
        try:
            fsym, fc, ssym, sc = calc._find_front_second_close(root, d)
            months = calc._months_between(fsym, ssym, root)
            if months != 0:
                rows.append({"date": pd.Timestamp(d), "front_symbol": fsym, "f1": fc,
                             "second_symbol": ssym, "f2": sc, "months": months})
        except ValueError:
            pass  # weekend / holiday / missing data -- skip
        d += timedelta(days=1)

    df = pd.DataFrame(rows, columns=_COLUMNS)
    if not df.empty:
        front_next_dir().mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_fp, index=False)
        logger.info(f"[front_next] {root}: {len(df)} rows -> {cache_fp}")
    return df
