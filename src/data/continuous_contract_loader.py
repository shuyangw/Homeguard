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


class ContinuousContractDataLoader:
    pass
