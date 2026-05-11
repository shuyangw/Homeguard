"""Tests for PerCycleReconciler."""
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.trading.futures.position import FuturesPosition
from src.trading.futures.reconciliation import (
    PerCycleReconciler, PositionDiff, ReconciliationResult, ReconciliationVerdict,
)


def _mk_position(symbol_root="MES", contract_month="202603", quantity=2) -> FuturesPosition:
    return FuturesPosition(
        symbol_root=symbol_root,
        contract_month=contract_month,
        raw_symbol=f"{symbol_root}{contract_month[-2:]}",
        quantity=quantity,
        avg_entry_price=5300.0,
        multiplier=5.0,
        tick_size=0.25,
        tick_value=1.25,
        expiration_date=date(2026, 3, 20),
        broker="ibkr_futures",
        strategy="adaptation_d",
        opened_at=datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc),
    )


def test_match_when_state_equals_broker():
    state = [_mk_position()]
    broker = [_mk_position()]
    reconciler = PerCycleReconciler(
        state_loader=lambda strategy: state,
        broker_positions=lambda: broker,
    )
    result = reconciler.reconcile("adaptation_d")
    assert result.verdict == ReconciliationVerdict.MATCH
    assert result.diffs == []


def test_drift_quantity():
    """Same (root, month) but different quantity."""
    state = [_mk_position(quantity=2)]
    broker = [_mk_position(quantity=1)]
    reconciler = PerCycleReconciler(
        state_loader=lambda strategy: state,
        broker_positions=lambda: broker,
    )
    result = reconciler.reconcile("adaptation_d")
    assert result.verdict == ReconciliationVerdict.DRIFT_QUANTITY
    assert len(result.diffs) == 1
    assert result.diffs[0].key == ("MES", "202603")


def test_missing_on_broker():
    """State has it; broker doesn't."""
    state = [_mk_position()]
    broker: list = []
    reconciler = PerCycleReconciler(
        state_loader=lambda strategy: state,
        broker_positions=lambda: broker,
    )
    result = reconciler.reconcile("adaptation_d")
    assert result.verdict == ReconciliationVerdict.MISSING_ON_BROKER


def test_missing_in_state():
    """Broker has it; state doesn't."""
    state: list = []
    broker = [_mk_position()]
    reconciler = PerCycleReconciler(
        state_loader=lambda strategy: state,
        broker_positions=lambda: broker,
    )
    result = reconciler.reconcile("adaptation_d")
    assert result.verdict == ReconciliationVerdict.MISSING_IN_STATE


def test_reconcile_and_gate_returns_true_on_match():
    state = [_mk_position()]
    broker = [_mk_position()]
    reconciler = PerCycleReconciler(
        state_loader=lambda strategy: state,
        broker_positions=lambda: broker,
    )
    assert reconciler.reconcile_and_gate("adaptation_d") is True


def test_reconcile_and_gate_returns_false_on_drift():
    state = [_mk_position(quantity=2)]
    broker = [_mk_position(quantity=1)]
    notifier = MagicMock()
    reconciler = PerCycleReconciler(
        state_loader=lambda strategy: state,
        broker_positions=lambda: broker,
        notifier=notifier,
    )
    assert reconciler.reconcile_and_gate("adaptation_d") is False
    # Drift should trigger notification
    notifier.assert_called_once()
    call_kwargs = notifier.call_args.kwargs
    assert "drift" in call_kwargs["message"].lower() or "drift" in call_kwargs.get("severity", "").lower()


def test_multiple_positions_some_match_some_drift():
    """Multi-position case: some match, one drifts."""
    state = [
        _mk_position(symbol_root="MES", contract_month="202603", quantity=2),
        _mk_position(symbol_root="MNQ", contract_month="202606", quantity=-1),
    ]
    broker = [
        _mk_position(symbol_root="MES", contract_month="202603", quantity=2),  # match
        _mk_position(symbol_root="MNQ", contract_month="202606", quantity=3),  # drift
    ]
    reconciler = PerCycleReconciler(
        state_loader=lambda strategy: state,
        broker_positions=lambda: broker,
    )
    result = reconciler.reconcile("adaptation_d")
    assert result.verdict in (ReconciliationVerdict.DRIFT_QUANTITY,)
    assert len(result.diffs) == 1
    assert result.diffs[0].key == ("MNQ", "202606")
