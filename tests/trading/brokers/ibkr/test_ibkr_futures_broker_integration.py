"""Integration tests for the IBKRFuturesBroker safeguard chain (mock IBKR)."""
from datetime import date
from unittest.mock import MagicMock

import pytest

from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.ibkr.ibkr_futures_broker import (
    IBKRFuturesBroker,
    OrderRejectedError,
)
from src.trading.brokers.interfaces.base import OrderSide, OrderType, TimeInForce
from src.trading.futures.audit_log import AuditLog
from src.trading.futures.expiration_guard import ExpirationGuard, ExpirationVerdict
from src.trading.futures.margin_guard import MarginGuard, MarginVerdict, MarginCheckResult
from src.trading.futures.symbol_resolver import ResolvedOrder


def _mk_resolved_order() -> ResolvedOrder:
    return ResolvedOrder(
        strategy_intent="ES.v.0",
        symbol_root="ES",
        contract_month="202406",
        raw_symbol="ESM4",
        side=OrderSide.BUY,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=5300.0,
        stop_price=None,
        time_in_force=TimeInForce.DAY,
        strategy="adaptation_d",
        as_of=date(2024, 5, 1),
    )


def _mk_broker_with_passing_guards(tmp_path) -> IBKRFuturesBroker:
    """Build a broker whose guards are tuned to accept the test order."""
    audit = AuditLog(log_dir=tmp_path / "audit")
    exp_guard = MagicMock()
    exp_guard.check_new_entry_with_expiration.return_value = ExpirationVerdict.OK
    margin_guard = MagicMock()
    margin_guard.pre_trade_check.return_value = MarginCheckResult(verdict=MarginVerdict.OK)
    return IBKRFuturesBroker(
        config=IBKRConfig(port=4002),
        audit_log=audit,
        expiration_guard=exp_guard,
        margin_guard=margin_guard,
    )


def test_submit_resolved_order_happy_path(tmp_path):
    """All guards pass -> returns ibkr_response dict + audit log records 'submit'."""
    broker = _mk_broker_with_passing_guards(tmp_path)
    response = broker.submit_resolved_order(
        _mk_resolved_order(),
        expiration_date=date(2024, 6, 21),
        hold_overnight=False,
    )
    assert response["status"] == "stub_accepted"
    assert "orderId" in response
    # Audit log has one 'submit' entry
    import json
    audit_files = list((tmp_path / "audit").glob("audit_*.jsonl"))
    assert len(audit_files) == 1
    entries = [json.loads(line) for line in audit_files[0].read_text().strip().split("\n")]
    assert len(entries) == 1
    assert entries[0]["event_type"] == "submit"
    assert entries[0]["raw_symbol"] == "ESM4"


def test_submit_resolved_order_rejected_on_expiration(tmp_path):
    """ExpirationGuard returns WARN -> OrderRejectedError + audit log records 'reject'."""
    audit = AuditLog(log_dir=tmp_path / "audit")
    exp_guard = MagicMock()
    exp_guard.check_new_entry_with_expiration.return_value = ExpirationVerdict.WARN
    margin_guard = MagicMock()
    broker = IBKRFuturesBroker(
        config=IBKRConfig(port=4002),
        audit_log=audit,
        expiration_guard=exp_guard,
        margin_guard=margin_guard,
    )
    with pytest.raises(OrderRejectedError, match="expiration"):
        broker.submit_resolved_order(
            _mk_resolved_order(),
            expiration_date=date(2024, 5, 5),
            hold_overnight=False,
        )
    # Margin guard NOT called (short-circuit on first failed guard)
    margin_guard.pre_trade_check.assert_not_called()
    # Audit log has 'reject' entry
    import json
    audit_files = list((tmp_path / "audit").glob("audit_*.jsonl"))
    entries = [json.loads(line) for line in audit_files[0].read_text().strip().split("\n")]
    assert entries[0]["event_type"] == "reject"
    assert "expiration" in entries[0]["error_message"]


def test_submit_resolved_order_rejected_on_margin(tmp_path):
    """MarginGuard returns REJECT -> OrderRejectedError + audit log records 'reject'."""
    audit = AuditLog(log_dir=tmp_path / "audit")
    exp_guard = MagicMock()
    exp_guard.check_new_entry_with_expiration.return_value = ExpirationVerdict.OK
    margin_guard = MagicMock()
    margin_guard.pre_trade_check.return_value = MarginCheckResult(
        verdict=MarginVerdict.REJECT, reason="post-trade margin too high",
    )
    broker = IBKRFuturesBroker(
        config=IBKRConfig(port=4002),
        audit_log=audit,
        expiration_guard=exp_guard,
        margin_guard=margin_guard,
    )
    with pytest.raises(OrderRejectedError, match="margin"):
        broker.submit_resolved_order(
            _mk_resolved_order(),
            expiration_date=date(2024, 6, 21),
            hold_overnight=False,
        )
    import json
    audit_files = list((tmp_path / "audit").glob("audit_*.jsonl"))
    entries = [json.loads(line) for line in audit_files[0].read_text().strip().split("\n")]
    assert entries[0]["event_type"] == "reject"
    assert "margin" in entries[0]["error_message"].lower()


def test_safeguards_lazy_init():
    """Broker created without explicit safeguards lazily constructs them."""
    broker = IBKRFuturesBroker(IBKRConfig(port=4002))
    broker._ensure_safeguards()
    from src.trading.futures.audit_log import AuditLog as AL
    from src.trading.futures.expiration_guard import ExpirationGuard as EG
    from src.trading.futures.margin_guard import MarginGuard as MG
    assert isinstance(broker._audit_log, AL)
    assert isinstance(broker._expiration_guard, EG)
    assert isinstance(broker._margin_guard, MG)
