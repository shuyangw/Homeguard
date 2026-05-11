"""Tests for FuturesSymbolResolver."""
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.trading.futures.symbol_resolver import (
    FuturesSymbolResolver, ResolvedOrder, InvalidIntentError,
)
from src.trading.brokers.interfaces.base import OrderSide


def _write_pcm_fixture(root: Path, year: int, month: int, rows: list[dict]) -> None:
    d = root / "futures_per_contract_1min" / f"year={year}" / f"month={month}"
    d.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    )
    df.write_parquet(d / "data.parquet")


def test_resolve_active_contract_returns_volume_leader(tmp_path, monkeypatch):
    """Highest-volume outright on the requested date is the active contract."""
    rows = [
        {"timestamp": datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
         "open": 5300.0, "high": 5300.0, "low": 5300.0, "close": 5300.0,
         "volume": 1_000_000, "symbol": "ESM4"},
        {"timestamp": datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
         "open": 5362.0, "high": 5362.0, "low": 5362.0, "close": 5362.0,
         "volume": 50_000, "symbol": "ESU4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    monkeypatch.setattr(
        "src.data.continuous_contract_loader._storage_root",
        lambda: tmp_path,
    )
    res = FuturesSymbolResolver().resolve_active_contract("ES", date(2024, 6, 3))
    assert res.raw_symbol == "ESM4"
    assert res.contract_month == "202406"
    assert res.symbol_root == "ES"


def test_resolve_for_order_accepts_continuous_intent(tmp_path, monkeypatch):
    """Strategy intent in continuous form (ES.v.0) is converted to a ResolvedOrder."""
    rows = [
        {"timestamp": datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
         "open": 5300.0, "high": 5300.0, "low": 5300.0, "close": 5300.0,
         "volume": 1_000_000, "symbol": "ESM4"},
        {"timestamp": datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
         "open": 5362.0, "high": 5362.0, "low": 5362.0, "close": 5362.0,
         "volume": 50_000, "symbol": "ESU4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    monkeypatch.setattr(
        "src.data.continuous_contract_loader._storage_root",
        lambda: tmp_path,
    )
    order = FuturesSymbolResolver().resolve_for_order(
        strategy_intent="ES.v.0",
        side=OrderSide.BUY,
        quantity=2,
        as_of=date(2024, 6, 3),
        strategy="adaptation_d",
    )
    assert order.strategy_intent == "ES.v.0"
    assert order.symbol_root == "ES"
    assert order.raw_symbol == "ESM4"
    assert order.contract_month == "202406"
    assert order.side == OrderSide.BUY
    assert order.quantity == 2
    assert order.strategy == "adaptation_d"


def test_resolve_for_order_rejects_raw_symbol():
    """Passing a raw symbol (e.g. ESM4) as strategy_intent must raise."""
    resolver = FuturesSymbolResolver()
    with pytest.raises(InvalidIntentError):
        resolver.resolve_for_order(
            strategy_intent="ESM4",  # NOT continuous form
            side=OrderSide.BUY, quantity=1,
            as_of=date(2024, 6, 3), strategy="x",
        )


def test_resolve_for_order_rejects_lowercase():
    """Continuous-form regex requires uppercase root."""
    resolver = FuturesSymbolResolver()
    with pytest.raises(InvalidIntentError):
        resolver.resolve_for_order(
            strategy_intent="es.v.0",
            side=OrderSide.BUY, quantity=1,
            as_of=date(2024, 6, 3), strategy="x",
        )


def test_resolve_for_order_caches_intra_session(tmp_path, monkeypatch):
    """Resolutions cached per (root, date) so repeat calls are cheap."""
    rows = [
        {"timestamp": datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
         "open": 5300.0, "high": 5300.0, "low": 5300.0, "close": 5300.0,
         "volume": 1_000_000, "symbol": "ESM4"},
        {"timestamp": datetime(2024, 6, 3, 14, 1, tzinfo=timezone.utc),
         "open": 5362.0, "high": 5362.0, "low": 5362.0, "close": 5362.0,
         "volume": 50_000, "symbol": "ESU4"},
    ]
    _write_pcm_fixture(tmp_path, 2024, 6, rows)
    monkeypatch.setattr(
        "src.data.continuous_contract_loader._storage_root",
        lambda: tmp_path,
    )
    resolver = FuturesSymbolResolver()
    r1 = resolver.resolve_active_contract("ES", date(2024, 6, 3))
    r2 = resolver.resolve_active_contract("ES", date(2024, 6, 3))
    # Same object identity = cache hit
    assert r1 is r2


def test_contract_month_format():
    """contract_month is YYYYMM. ESM4 -> 202406."""
    from src.trading.futures.symbol_resolver import _raw_symbol_to_contract_month
    assert _raw_symbol_to_contract_month("ESM4", "ES") == "202406"
    assert _raw_symbol_to_contract_month("GCQ5", "GC") == "202508"
    assert _raw_symbol_to_contract_month("ZNZ4", "ZN") == "202412"
