"""Treasury yield reads from Micro Yield futures.

Micro Yield contracts (2YY, 5YY, 10Y, 30Y) are priced directly in yield space:
a close of 4.228 represents a 4.228% yield on the on-the-run Treasury at that
tenor. Listing date: 2022-08-15.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from src.settings import get_local_storage_dir

MICRO_YIELD_LISTING_DATE = date(2022, 8, 15)

TENOR_TO_SYMBOL: dict[str, str] = {
    "2Y": "2YY",
    "5Y": "5YY",
    "10Y": "10Y",
    "30Y": "30Y",
}


def _storage_root() -> Path:
    return get_local_storage_dir()


def get_treasury_yield(tenor: str, d: date) -> float:
    """Read Treasury yield from Micro Yield futures close.

    Args:
        tenor: One of "2Y", "5Y", "10Y", "30Y".
        d: Date for which to read the yield.

    Returns:
        Yield as a decimal percentage (e.g., 4.228 for 4.228%).

    Raises:
        KeyError: If tenor not in TENOR_TO_SYMBOL.
        ValueError: If d is before listing or no data exists.
    """
    if d < MICRO_YIELD_LISTING_DATE:
        raise ValueError(
            f"Micro Yield futures listing date is {MICRO_YIELD_LISTING_DATE}; "
            f"cannot read yield before that."
        )

    sym = TENOR_TO_SYMBOL[tenor]
    path = (
        _storage_root()
        / "futures_1min"
        / f"symbol={sym}"
        / f"year={d.year}"
        / f"month={d.month}"
        / "data.parquet"
    )
    if not path.exists():
        raise ValueError(f"no {sym} data for {d}: missing {path}")

    df = (
        pl.scan_parquet(path)
        .filter(pl.col("timestamp").dt.date() == d)
        .sort("timestamp")
        .collect()
    )
    if df.is_empty():
        raise ValueError(f"no {sym} data for {d}")

    return float(df["close"][-1])
