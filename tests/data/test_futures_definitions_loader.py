"""Tests for FuturesDefinitionsLoader."""
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.data.futures_definitions_loader import (
    ContractDefinition,
    DefinitionNotFoundError,
    FuturesDefinitionsLoader,
    _parse_contract_month,
)


def _write_definitions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def _make_def_row(
    symbol: str, expiration: date, activation: date,
    instrument_class: str = "F", timestamp: datetime | None = None,
    tick_size: float = 0.25, tick_value: float = 12.50,
) -> dict:
    return {
        "symbol": symbol,
        "instrument_class": instrument_class,
        "expiration": datetime(expiration.year, expiration.month, expiration.day, tzinfo=timezone.utc),
        "activation": datetime(activation.year, activation.month, activation.day, tzinfo=timezone.utc),
        "min_price_increment": tick_size,
        "min_price_increment_amount": tick_value,
        "timestamp": timestamp or datetime(2024, 6, 1, tzinfo=timezone.utc),
    }


def test_parse_single_digit_year():
    assert _parse_contract_month("ESM4", "ES") == "202406"
    assert _parse_contract_month("ESZ5", "ES") == "202512"


def test_parse_two_digit_year():
    assert _parse_contract_month("ESM24", "ES") == "202406"
    assert _parse_contract_month("ESH28", "ES") == "202803"


def test_parse_invalid_symbol_raises():
    with pytest.raises(ValueError, match="cannot parse"):
        _parse_contract_month("ES", "ES")
    with pytest.raises(ValueError, match="cannot parse"):
        _parse_contract_month("ESMX", "ES")


def test_get_definition_returns_correct_fields(tmp_path: Path):
    path = tmp_path / "futures" / "definitions" / "year=2024" / "month=6" / "data.parquet"
    _write_definitions(path, [
        _make_def_row("ESM4", date(2024, 6, 21), date(2019, 6, 1)),
    ])
    loader = FuturesDefinitionsLoader(storage_root=tmp_path)
    d = loader.get_definition("ESM4", "ES", date(2024, 6, 15))
    assert d.raw_symbol == "ESM4"
    assert d.symbol_root == "ES"
    assert d.contract_month == "202406"
    assert d.expiration == date(2024, 6, 21)
    assert d.activation == date(2019, 6, 1)
    assert d.tick_size == 0.25
    assert d.tick_value == 12.50


def test_filters_to_futures_only(tmp_path: Path):
    """Options on futures (instrument_class != 'F') must be excluded."""
    path = tmp_path / "futures" / "definitions" / "year=2024" / "month=6" / "data.parquet"
    _write_definitions(path, [
        _make_def_row("ESM4", date(2024, 6, 21), date(2019, 6, 1),
                      instrument_class="C"),  # Call option, NOT a future
    ])
    loader = FuturesDefinitionsLoader(storage_root=tmp_path)
    with pytest.raises(DefinitionNotFoundError):
        loader.get_definition("ESM4", "ES", date(2024, 6, 15))


def test_takes_latest_row_for_symbol(tmp_path: Path):
    """When multiple definition snapshots exist for a symbol, use the latest."""
    path = tmp_path / "futures" / "definitions" / "year=2024" / "month=6" / "data.parquet"
    _write_definitions(path, [
        _make_def_row("ESM4", date(2024, 6, 14), date(2019, 6, 1),
                      timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc)),
        _make_def_row("ESM4", date(2024, 6, 21), date(2019, 6, 1),
                      timestamp=datetime(2024, 6, 5, tzinfo=timezone.utc)),
    ])
    loader = FuturesDefinitionsLoader(storage_root=tmp_path)
    d = loader.get_definition("ESM4", "ES", date(2024, 6, 15))
    assert d.expiration == date(2024, 6, 21)


def test_missing_partition_raises(tmp_path: Path):
    loader = FuturesDefinitionsLoader(storage_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="futures_definitions partition"):
        loader.get_definition("ESM4", "ES", date(2099, 1, 1))


def test_missing_symbol_raises(tmp_path: Path):
    path = tmp_path / "futures" / "definitions" / "year=2024" / "month=6" / "data.parquet"
    _write_definitions(path, [
        _make_def_row("CLM4", date(2024, 5, 20), date(2019, 5, 1)),
    ])
    loader = FuturesDefinitionsLoader(storage_root=tmp_path)
    with pytest.raises(DefinitionNotFoundError, match="no definition for ESM4"):
        loader.get_definition("ESM4", "ES", date(2024, 6, 15))


def test_get_expiration_convenience(tmp_path: Path):
    path = tmp_path / "futures" / "definitions" / "year=2024" / "month=6" / "data.parquet"
    _write_definitions(path, [
        _make_def_row("ESM4", date(2024, 6, 21), date(2019, 6, 1)),
    ])
    loader = FuturesDefinitionsLoader(storage_root=tmp_path)
    assert loader.get_expiration("ESM4", "ES", date(2024, 6, 15)) == date(2024, 6, 21)


def test_partition_cache_avoids_reread(tmp_path: Path):
    """Two calls into the same partition shouldn't both touch disk."""
    path = tmp_path / "futures" / "definitions" / "year=2024" / "month=6" / "data.parquet"
    _write_definitions(path, [
        _make_def_row("ESM4", date(2024, 6, 21), date(2019, 6, 1)),
        _make_def_row("ESU4", date(2024, 9, 20), date(2019, 9, 1)),
    ])
    loader = FuturesDefinitionsLoader(storage_root=tmp_path)
    loader.get_definition("ESM4", "ES", date(2024, 6, 15))
    # Delete the file; cache should serve the second call
    path.unlink()
    d2 = loader.get_definition("ESU4", "ES", date(2024, 6, 15))
    assert d2.expiration == date(2024, 9, 20)
