"""Continuous futures contract loader with 3 adjustment methods.

Reads raw .v.0 continuous bars from futures_1min/ and per-contract bars
from futures_per_contract_1min/. Provides:
- load(symbol, method) -> pl.DataFrame for raw | ratio_adjusted | panama_adjusted
- detect_roll_dates(symbol) -> list[date]
- aggregate_to_daily(symbol, method) -> pl.DataFrame
- aggregate_to_hourly(symbol, method) -> pl.DataFrame

Roll detection: per-day highest-volume outright contract. Spreads
(symbols containing "-") are excluded from active-contract candidates.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from src.settings import get_local_storage_dir


def _storage_root() -> Path:
    return get_local_storage_dir()


# CME month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
_MONTH_CODES = "FGHJKMNQUVXZ"


def _is_outright(symbol: str, root: str) -> bool:
    """True if `symbol` is an outright contract of `root` (not a spread)."""
    if "-" in symbol or " " in symbol:
        return False
    if not symbol.startswith(root):
        return False
    suffix = symbol[len(root):]
    if len(suffix) < 2:
        return False
    # suffix should be one month-code letter + digits (year)
    return suffix[0] in _MONTH_CODES and suffix[1:].isdigit()


class ContinuousContractDataLoader:
    def _active_contract_per_day(
        self, root: str, start: date, end: date
    ) -> pl.DataFrame:
        """Return DataFrame with columns [date, active] for each trading day
        in [start, end] where `active` is the highest-volume outright contract
        of `root` on that day."""
        pcm_root = _storage_root() / "futures_per_contract_1min"
        if not pcm_root.exists():
            return pl.DataFrame(schema={"date": pl.Date, "active": pl.String})

        all_files: list[Path] = []
        for y in range(start.year, end.year + 1):
            for m in range(1, 13):
                f = pcm_root / f"year={y}" / f"month={m}" / "data.parquet"
                if f.exists():
                    all_files.append(f)
        if not all_files:
            return pl.DataFrame(schema={"date": pl.Date, "active": pl.String})

        df = pl.concat([pl.read_parquet(f) for f in all_files])
        df = df.filter(pl.col("symbol").map_elements(
            lambda s: _is_outright(s, root), return_dtype=pl.Boolean,
        ))
        df = df.filter(
            (pl.col("timestamp").dt.date() >= start)
            & (pl.col("timestamp").dt.date() <= end)
        )
        if df.is_empty():
            return pl.DataFrame(schema={"date": pl.Date, "active": pl.String})

        # Daily total volume per (date, symbol); pick the symbol with max volume per date
        daily = df.group_by([
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("symbol"),
        ]).agg(pl.col("volume").sum().alias("vol"))
        # For each date, take row with max vol
        ranked = daily.sort(["date", "vol"], descending=[False, True])
        active = ranked.group_by("date").agg(pl.col("symbol").first().alias("active")).sort("date")
        return active
