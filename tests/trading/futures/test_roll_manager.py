"""Tests for the futures-broker FuturesRollManager."""
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.trading.futures.position import FuturesPosition
from src.trading.futures.roll_manager import (
    FuturesRollManager, RollEvent,
)


def _mk_position(
    symbol_root="ES",
    contract_month="202406",
    raw_symbol="ESM4",
    quantity=2,
    expiration=date(2024, 6, 21),
) -> FuturesPosition:
    return FuturesPosition(
        symbol_root=symbol_root, contract_month=contract_month, raw_symbol=raw_symbol,
        quantity=quantity, avg_entry_price=5300.0,
        multiplier=5.0, tick_size=0.25, tick_value=1.25,
        expiration_date=expiration,
        broker="ibkr_futures", strategy="adaptation_d",
        opened_at=datetime(2024, 5, 1, 14, 0, tzinfo=timezone.utc),
    )


def test_no_positions_returns_empty():
    broker = MagicMock()
    broker.get_futures_positions.return_value = []
    resolver = MagicMock()
    mgr = FuturesRollManager(broker=broker, resolver=resolver)
    assert mgr.get_upcoming_rolls(lookahead_days=14, today=date(2024, 6, 1)) == []


def test_position_far_from_expiry_not_returned():
    """Expiration 30 days out, lookahead 14 -> empty."""
    pos = _mk_position(expiration=date(2024, 7, 1))
    broker = MagicMock()
    broker.get_futures_positions.return_value = [pos]
    resolver = MagicMock()
    mgr = FuturesRollManager(broker=broker, resolver=resolver)
    rolls = mgr.get_upcoming_rolls(lookahead_days=14, today=date(2024, 6, 1))
    assert rolls == []


def test_position_within_lookahead_returns_roll_event():
    """Expiration 5 days out, lookahead 14 -> one RollEvent."""
    pos = _mk_position(expiration=date(2024, 6, 6))
    broker = MagicMock()
    broker.get_futures_positions.return_value = [pos]
    # Resolver returns the next active contract
    resolver = MagicMock()
    resolver.resolve_active_contract.return_value = MagicMock(
        symbol_root="ES", contract_month="202409", raw_symbol="ESU4",
    )
    mgr = FuturesRollManager(broker=broker, resolver=resolver)
    rolls = mgr.get_upcoming_rolls(lookahead_days=14, today=date(2024, 6, 1))
    assert len(rolls) == 1
    ev = rolls[0]
    assert isinstance(ev, RollEvent)
    assert ev.position.symbol_root == "ES"
    assert ev.position.contract_month == "202406"
    assert ev.suggested_new_month == "202409"


def test_position_at_expiration_returns_zero_days_until_required():
    """Position 1 day out should have days_until_required = 0."""
    pos = _mk_position(expiration=date(2024, 6, 2))
    broker = MagicMock()
    broker.get_futures_positions.return_value = [pos]
    resolver = MagicMock()
    resolver.resolve_active_contract.return_value = MagicMock(
        symbol_root="ES", contract_month="202409", raw_symbol="ESU4",
    )
    mgr = FuturesRollManager(broker=broker, resolver=resolver)
    rolls = mgr.get_upcoming_rolls(lookahead_days=14, today=date(2024, 6, 1))
    # ES threshold is 5; 1 day expiry minus 5 threshold capped at max(0, -4) = 0
    assert rolls[0].days_until_required == 0


def test_resolver_failure_propagates():
    """If resolver can't find a new contract (no data on that date), surface clearly."""
    pos = _mk_position(expiration=date(2024, 6, 6))
    broker = MagicMock()
    broker.get_futures_positions.return_value = [pos]
    resolver = MagicMock()
    resolver.resolve_active_contract.side_effect = ValueError("no data")
    mgr = FuturesRollManager(broker=broker, resolver=resolver)
    rolls = mgr.get_upcoming_rolls(lookahead_days=14, today=date(2024, 6, 1))
    # No swallow; either skip with logging or raise. For this impl: skip silently
    # and surface in the RollEvent.suggested_new_month=None.
    assert len(rolls) == 1
    assert rolls[0].suggested_new_month is None


def test_multiple_positions_only_rolling_ones_returned():
    near_expiry = _mk_position(symbol_root="ES", contract_month="202406", expiration=date(2024, 6, 6))
    far_expiry = _mk_position(symbol_root="GC", contract_month="202412", expiration=date(2024, 12, 27))
    broker = MagicMock()
    broker.get_futures_positions.return_value = [near_expiry, far_expiry]
    resolver = MagicMock()
    resolver.resolve_active_contract.return_value = MagicMock(
        symbol_root="ES", contract_month="202409", raw_symbol="ESU4",
    )
    mgr = FuturesRollManager(broker=broker, resolver=resolver)
    rolls = mgr.get_upcoming_rolls(lookahead_days=14, today=date(2024, 6, 1))
    assert len(rolls) == 1
    assert rolls[0].position.symbol_root == "ES"
