"""Per-asset-class carry calculator.

Front-month and second-month identified by volume ranking from
futures_per_contract_1min/ on the target date. Outright contracts only
(spreads filtered via month-code regex).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from src.settings import get_local_storage_dir


_MONTH_CODES = "FGHJKMNQUVXZ"


def _storage_root() -> Path:
    return get_local_storage_dir()


def _is_outright(symbol: str, root: str) -> bool:
    if "-" in symbol or " " in symbol:
        return False
    if not symbol.startswith(root):
        return False
    suffix = symbol[len(root):]
    if len(suffix) < 2:
        return False
    return suffix[0] in _MONTH_CODES and suffix[1:].isdigit()


class CarryCalculator:
    """Computes per-asset-class carry signals."""

    def _find_front_second_close(
        self, root: str, d: date,
    ) -> tuple[str, float, str, float]:
        """Return (front_symbol, front_close, second_symbol, second_close)
        ranked by daily volume on date d.

        Raises ValueError if fewer than 2 outright contracts have data on d.
        """
        pcm = (
            _storage_root() / "futures_per_contract_1min"
            / f"year={d.year}" / f"month={d.month}" / "data.parquet"
        )
        if not pcm.exists():
            raise ValueError(f"no per-contract data for {d}: missing {pcm}")
        df = pl.read_parquet(pcm)
        df = df.filter(pl.col("symbol").map_elements(
            lambda s: _is_outright(s, root), return_dtype=pl.Boolean,
        ))
        df = df.filter(pl.col("timestamp").dt.date() == d)
        if df.is_empty():
            raise ValueError(f"no outright {root} data on {d}")
        daily = df.group_by("symbol").agg([
            pl.col("close").last().alias("c"),
            pl.col("volume").sum().alias("v"),
        ]).sort("v", descending=True)
        if daily.shape[0] < 2:
            raise ValueError(
                f"need 2 outright contracts on {d}, found {daily.shape[0]}"
            )
        rows = daily.head(2).to_dicts()
        return (rows[0]["symbol"], rows[0]["c"],
                rows[1]["symbol"], rows[1]["c"])
