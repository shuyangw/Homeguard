"""Loader for per-contract futures definitions from Databento.

Reads from H:/Stock_Data/futures_definitions/year=Y/month=M/data.parquet
and exposes a clean ContractDefinition view for the trading layer. Used by
FuturesSymbolResolver to attach a real expiration_date to ResolvedOrder so
the ExpirationGuard sees ground truth instead of placeholders.

Database schema is broad (65 columns); we extract the operationally critical
fields: expiration, activation, tick size, tick value. Multiplier is not
exposed by this loader because the upstream contract_multiplier field is
unreliable (i32 sentinel values for ES) -- callers should source multiplier
from a separate hardcoded table per symbol root if they need it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import polars as pl

from src.settings import get_local_storage_dir

_MONTH_CODES = "FGHJKMNQUVXZ"


def _storage_root() -> Path:
    return get_local_storage_dir()


@dataclass(frozen=True)
class ContractDefinition:
    """Operationally relevant subset of a Databento futures definition row."""
    raw_symbol: str           # e.g. "ESM4"
    symbol_root: str          # e.g. "ES"
    contract_month: str       # YYYYMM
    expiration: date          # last trade date
    activation: date          # first listing date
    tick_size: float          # min_price_increment
    tick_value: float         # min_price_increment_amount


class DefinitionNotFoundError(LookupError):
    """No futures definition row matched the requested (raw_symbol, as_of)."""


def _parse_contract_month(raw_symbol: str, symbol_root: str) -> str:
    """Decode contract month from raw symbol relative to as-of year.

    Single-digit CME year codes (e.g. ESM4) decode via the current decade's
    tens digit; if that yields a year >5 in the past, roll forward one decade.
    """
    suffix = raw_symbol[len(symbol_root):]
    if len(suffix) < 2:
        raise ValueError(f"cannot parse contract month from {raw_symbol}")
    month_letter = suffix[0]
    year_digit_str = suffix[1:]
    if not year_digit_str.isdigit():
        raise ValueError(f"cannot parse year digits from {raw_symbol}")
    year_digit = int(year_digit_str)
    month_num = _MONTH_CODES.index(month_letter) + 1
    today = date.today()
    if len(year_digit_str) >= 2:
        # Two-digit year (e.g. ESM24)
        century_base = (today.year // 100) * 100
        candidate = century_base + year_digit
    else:
        # Single-digit year decoded via current decade
        decade_base = (today.year // 10) * 10
        candidate = decade_base + year_digit
        if candidate < today.year - 5:
            candidate += 10
    return f"{candidate:04d}{month_num:02d}"


class FuturesDefinitionsLoader:
    """Reads per-contract definitions from futures_definitions partitions."""

    def __init__(self, storage_root: Path | None = None) -> None:
        self._root = storage_root if storage_root is not None else _storage_root()
        # Per-partition cache to avoid re-reading the same month repeatedly.
        # Keyed by (year, month) -> DataFrame.
        self._partition_cache: dict[tuple[int, int], pl.DataFrame] = {}

    def _load_partition(self, year: int, month: int) -> pl.DataFrame:
        key = (year, month)
        if key in self._partition_cache:
            return self._partition_cache[key]
        path = (
            self._root
            / "futures"
            / "definitions"
            / f"year={year}"
            / f"month={month}"
            / "data.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"futures_definitions partition not found: {path}"
            )
        df = (
            pl.scan_parquet(path)
            .filter(pl.col("instrument_class") == "F")
            .select(
                "symbol",
                "expiration",
                "activation",
                "min_price_increment",
                "min_price_increment_amount",
                "timestamp",
            )
            .collect()
        )
        self._partition_cache[key] = df
        return df

    def get_definition(
        self, raw_symbol: str, symbol_root: str, as_of: date,
    ) -> ContractDefinition:
        """Look up the latest definition row for a contract in the as_of partition.

        Args:
            raw_symbol: Contract symbol (e.g. "ESM4").
            symbol_root: Symbol root (e.g. "ES") -- used to compute contract_month.
            as_of: Date determining which monthly partition to read.

        Raises:
            DefinitionNotFoundError: If no row matches raw_symbol in the partition.
            FileNotFoundError: If the partition file is missing.
            ValueError: If raw_symbol cannot be parsed.
        """
        df = self._load_partition(as_of.year, as_of.month)
        matched = df.filter(pl.col("symbol") == raw_symbol)
        if matched.is_empty():
            raise DefinitionNotFoundError(
                f"no definition for {raw_symbol} in "
                f"year={as_of.year}/month={as_of.month}"
            )
        # Multiple rows per symbol are possible (security_update_action events);
        # take latest by timestamp.
        latest = matched.sort("timestamp").tail(1)
        row = latest.row(0, named=True)
        return ContractDefinition(
            raw_symbol=raw_symbol,
            symbol_root=symbol_root,
            contract_month=_parse_contract_month(raw_symbol, symbol_root),
            expiration=row["expiration"].date(),
            activation=row["activation"].date(),
            tick_size=float(row["min_price_increment"]),
            tick_value=float(row["min_price_increment_amount"]),
        )

    def get_expiration(
        self, raw_symbol: str, symbol_root: str, as_of: date,
    ) -> date:
        """Convenience wrapper returning only the expiration date."""
        return self.get_definition(raw_symbol, symbol_root, as_of).expiration
