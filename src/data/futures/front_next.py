"""Per-root daily front (F1) / next (F2) settle series for calendar spreads.

Reuses CarryCalculator's "two most-active outright contracts by daily volume"
selection rule (matching `_find_front_second_close`) so the calendar spread
trades the liquid F1/F2 pair, matching the carry sleeve's contract selection.

The build loop is MONTH-BATCHED: each monthly per-contract parquet is read
once and ranked for every date within it, rather than re-reading the whole
month's parquet once per calendar day (the naive per-day loop cost ~1s/day,
~1.6 hours per root at #31 scale; batching cuts that to seconds).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import polars as pl

from src.data.carry_calculator import CarryCalculator, _is_outright
from src.data.futures.paths import front_next_dir, per_contract_1min_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

_COLUMNS = ["date", "front_symbol", "f1", "second_symbol", "f2", "months"]


def _month_range(start: date, end: date) -> list[tuple[int, int]]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def _build_month_rows(
    root: str, year: int, month: int, start: date, end: date, calc: CarryCalculator,
) -> list[dict]:
    pcm = per_contract_1min_dir() / f"year={year}" / f"month={month}" / "data.parquet"
    if not pcm.exists():
        logger.debug(f"[front_next] {root}: no per-contract data for {year}-{month:02d}, skipping")
        return []

    df = pl.read_parquet(pcm)
    df = df.filter(pl.col("symbol").map_elements(
        lambda s: _is_outright(s, root), return_dtype=pl.Boolean,
    ))
    if df.is_empty():
        return []

    df = df.with_columns(pl.col("timestamp").dt.date().alias("d"))
    df = df.filter((pl.col("d") >= start) & (pl.col("d") <= end))
    if df.is_empty():
        return []

    daily = df.group_by(["d", "symbol"]).agg([
        pl.col("close").last().alias("c"),
        pl.col("volume").sum().alias("v"),
    ])

    rows: list[dict] = []
    daily_pd = daily.to_pandas()
    for d_val, group in daily_pd.groupby("d"):
        group = group.sort_values("v", ascending=False)
        if len(group) < 2:
            continue
        top2 = group.iloc[:2]
        front_symbol, f1 = top2.iloc[0]["symbol"], top2.iloc[0]["c"]
        second_symbol, f2 = top2.iloc[1]["symbol"], top2.iloc[1]["c"]
        months = calc._months_between(front_symbol, second_symbol, root)
        if months == 0:
            continue
        rows.append({
            "date": pd.Timestamp(d_val), "front_symbol": front_symbol, "f1": float(f1),
            "second_symbol": second_symbol, "f2": float(f2), "months": months,
        })
    return rows


def front_next_history(root: str, start: date, end: date) -> pd.DataFrame:
    cache_fp = front_next_dir() / f"{root}.parquet"
    cached: pd.DataFrame | None = None
    if cache_fp.exists():
        cached = pd.read_parquet(cache_fp)
        if not cached.empty and cached["date"].min() <= pd.Timestamp(start) and cached["date"].max() >= pd.Timestamp(end):
            mask = (cached["date"] >= pd.Timestamp(start)) & (cached["date"] <= pd.Timestamp(end))
            return cached.loc[mask].reset_index(drop=True)

    calc = CarryCalculator()
    rows: list[dict] = []
    for year, month in _month_range(start, end):
        rows.extend(_build_month_rows(root, year, month, start, end, calc))

    df = pd.DataFrame(rows, columns=_COLUMNS)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, df], ignore_index=True)
            merged = merged.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
        else:
            merged = df
        front_next_dir().mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_fp, index=False)
        logger.info(f"[front_next] {root}: {len(merged)} rows -> {cache_fp}")
        mask = (merged["date"] >= pd.Timestamp(start)) & (merged["date"] <= pd.Timestamp(end))
        return merged.loc[mask].reset_index(drop=True)
    return df
