"""Aggregate open interest per symbol root from Databento futures_statistics.

Databento's futures_statistics dataset publishes per-contract daily statistics
including end-of-session open interest (`stat_type == 9`). This module sums
open interest across all outright contracts of a given symbol root on a given
date, producing a single aggregate-OI value per (root, date).

Aggregate OI is a partial substitute for CFTC's Commitment of Traders (CoT)
report -- it tells you total speculative + commercial positioning in a
contract family, though it does NOT decompose into the CoT trader categories.
Use it as a regime feature ("trend strength via OI growth") or as a sanity
check on liquidity assumptions.

Statistics dataset stat_type codes (Databento spec):
    1 opening, 2 indicative opening, 3 settlement, 4 high, 5 low,
    6 cleared volume, 7 lowest offer, 8 highest bid, 9 open interest,
    10 fixing price, 11 close price
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import polars as pl

from src.settings import get_local_storage_dir
from src.data.futures.paths import statistics_dir

STAT_TYPE_OPEN_INTEREST = 9
_MONTH_CODES = "FGHJKMNQUVXZ"
_OUTRIGHT_RE = re.compile(rf"^([A-Z0-9]+?)([{_MONTH_CODES}])(\d+)$")


def _storage_root() -> Path:
    return get_local_storage_dir()


def _is_outright(symbol: str, root: str) -> bool:
    """True iff `symbol` is an outright contract of `root` (not a spread)."""
    if "-" in symbol or " " in symbol:
        return False
    m = _OUTRIGHT_RE.match(symbol)
    if m is None:
        return False
    return m.group(1) == root


def aggregate_open_interest(symbol_root: str, d: date) -> int:
    """Sum end-of-day open interest across all outright contracts of `root`.

    Args:
        symbol_root: Symbol root (e.g., "ES", "CL", "ZN"). Spreads excluded.
        d: Date for which to read OI.

    Returns:
        Aggregate open interest as an int. Returns 0 if no OI rows exist for
        the (root, date) pair.

    Raises:
        FileNotFoundError: If the partition for the date is missing.
    """
    path = (
        statistics_dir()
        / f"year={d.year}"
        / f"month={d.month}"
        / "data.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"futures_statistics partition not found: {path}")

    df = (
        pl.scan_parquet(path)
        .filter(pl.col("stat_type") == STAT_TYPE_OPEN_INTEREST)
        .filter(pl.col("timestamp").dt.date() == d)
        .select("symbol", "timestamp", "quantity")
        .collect()
    )

    if df.is_empty():
        return 0

    df = df.filter(
        pl.col("symbol").map_elements(
            lambda s: _is_outright(s, symbol_root), return_dtype=pl.Boolean,
        )
    )
    if df.is_empty():
        return 0

    # Within a day, keep latest OI snapshot per outright contract, then sum.
    latest = (
        df.sort("timestamp")
        .group_by("symbol")
        .agg(pl.col("quantity").last().alias("oi"))
    )
    return int(latest["oi"].sum())


def per_contract_open_interest(symbol_root: str, d: date) -> dict[str, int]:
    """Return {contract_symbol: end-of-day OI} for each outright of `root` on `d`.

    Unlike aggregate_open_interest (which sums), this preserves per-contract OI
    so the roll calendar can detect the front-to-back OI crossover.

    Raises:
        FileNotFoundError: If the statistics partition for the date is missing.
    """
    path = (
        statistics_dir()
        / f"year={d.year}"
        / f"month={d.month}"
        / "data.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"futures_statistics partition not found: {path}")

    df = (
        pl.scan_parquet(path)
        .filter(pl.col("stat_type") == STAT_TYPE_OPEN_INTEREST)
        .filter(pl.col("timestamp").dt.date() == d)
        .select("symbol", "timestamp", "quantity")
        .collect()
    )
    if df.is_empty():
        return {}

    df = df.filter(
        pl.col("symbol").map_elements(
            lambda s: _is_outright(s, symbol_root), return_dtype=pl.Boolean,
        )
    )
    if df.is_empty():
        return {}

    latest = (
        df.sort("timestamp")
        .group_by("symbol")
        .agg(pl.col("quantity").last().alias("oi"))
    )
    return {row["symbol"]: int(row["oi"]) for row in latest.iter_rows(named=True)}
