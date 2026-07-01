"""SOFR derivation from SR1 front-month price.

SR1 quote convention: 100 - average_SOFR_for_the_month.
SR1 listing date: 2018-05-07.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from src.settings import get_local_storage_dir
from src.data.futures.paths import per_contract_1min_dir

SR1_LISTING_DATE = date(2018, 5, 7)

# CME month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun,
#                  N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
_MONTH_CODES = "FGHJKMNQUVXZ"


def _storage_root() -> Path:
    return get_local_storage_dir()


def sr1_front_month_symbol(d: date) -> str:
    """Resolve SR1 front-month contract symbol for a given date.

    SR1 expires monthly; the "front-month" on a given date is the contract
    expiring that calendar month.
    """
    code = _MONTH_CODES[d.month - 1]
    year_digit = d.year % 10
    return f"SR1{code}{year_digit}"


def derive_sofr(d: date) -> float:
    """Derive overnight SOFR from SR1 front-month close.

    Returns implied SOFR rate (e.g. 4.50 for 4.50%).
    Raises ValueError if d is before SR1 listing or no data exists.
    """
    if d < SR1_LISTING_DATE:
        raise ValueError(
            f"SR1 listing date is {SR1_LISTING_DATE}; cannot derive SOFR before that."
        )

    front = sr1_front_month_symbol(d)
    pcm = (
        per_contract_1min_dir()
        / f"year={d.year}"
        / f"month={d.month}"
        / "data.parquet"
    )
    if not pcm.exists():
        raise ValueError(f"no SR1 data for {d}: missing {pcm}")

    df = (
        pl.scan_parquet(pcm)
        .filter(pl.col("symbol") == front)
        .filter(pl.col("timestamp").dt.date() == d)
        .sort("timestamp")
        .collect()
    )
    if df.is_empty():
        raise ValueError(f"no SR1 data for {front} on {d}")

    last_close = df["close"][-1]
    return float(100.0 - last_close)
