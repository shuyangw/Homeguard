"""Tests for FuturesPosition dataclass."""
from datetime import date, datetime, timezone

import pytest

from src.trading.futures.position import FuturesPosition


def _mk_position(**overrides) -> FuturesPosition:
    """Build a FuturesPosition with sensible defaults; override per test."""
    defaults = dict(
        symbol_root="MES",
        contract_month="202603",
        raw_symbol="MESH6",
        quantity=2,
        avg_entry_price=5300.0,
        multiplier=5.0,
        tick_size=0.25,
        tick_value=1.25,
        expiration_date=date(2026, 3, 20),
        broker="ibkr_futures",
        strategy="adaptation_d",
        opened_at=datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return FuturesPosition(**defaults)


def test_position_construction():
    pos = _mk_position()
    assert pos.symbol_root == "MES"
    assert pos.contract_month == "202603"
    assert pos.raw_symbol == "MESH6"
    assert pos.quantity == 2
    assert pos.multiplier == 5.0


def test_position_key_tuple():
    """position_key is (symbol_root, contract_month) -- the reconciliation key."""
    pos = _mk_position()
    assert pos.position_key == ("MES", "202603")


def test_days_to_expiration_future(monkeypatch):
    """When expiration is in the future, days_to_expiration is positive."""
    import src.trading.futures.position as pos_mod
    # Freeze "today" to a known date
    fixed_today = date(2026, 3, 1)
    monkeypatch.setattr(pos_mod, "_today", lambda: fixed_today)

    pos = _mk_position(expiration_date=date(2026, 3, 20))
    assert pos.days_to_expiration == 19


def test_days_to_expiration_past(monkeypatch):
    """Past expiration produces negative days."""
    import src.trading.futures.position as pos_mod
    monkeypatch.setattr(pos_mod, "_today", lambda: date(2026, 4, 1))

    pos = _mk_position(expiration_date=date(2026, 3, 20))
    assert pos.days_to_expiration == -12


def test_position_short_quantity():
    """Negative quantity represents a short position."""
    pos = _mk_position(quantity=-3)
    assert pos.quantity == -3
    # position_key is still per-contract-month, not signed
    assert pos.position_key == ("MES", "202603")
