"""ExecutionEngine must never re-submit an order the broker already accepted.

On 2026-09-07 RAMP's market sells could not fill (market closed); after each 30 s fill
timeout the retry loop placed a NEW order without cancelling the first, so every sell
went out three times.
"""

import pytest

from src.trading.brokers.broker_interface import (
    BrokerConnectionError,
    BrokerError,
    OrderSide,
    OrderType,
)
from src.trading.core.execution_engine import ExecutionEngine


class FakeBroker:
    """Accepts orders, fails the first `place_failures` placements, and fills only on request."""

    name = "fake"

    def __init__(self, place_failures=0, fill_status="pending", filled_qty=0, fill_on_cancel=False,
                 cancel_error=None, status_error=None):
        self.place_failures = place_failures
        self.fill_status = fill_status
        self.filled_qty = filled_qty
        self.fill_on_cancel = fill_on_cancel
        self.cancel_error = cancel_error
        self.status_error = status_error
        self.place_calls = 0
        self.cancelled = []

    def place_stock_order(self, symbol, quantity, side, order_type, limit_price=None, stop_price=None,
                          time_in_force=None, **kwargs):
        self.place_calls += 1
        if self.place_calls <= self.place_failures:
            raise BrokerConnectionError("gateway hiccup")
        return self._order(symbol, quantity)

    def get_order(self, order_id):
        if self.status_error is not None:
            raise self.status_error
        return self._order("IP", 54, order_id)

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        if self.cancel_error is not None:
            raise self.cancel_error
        if self.fill_on_cancel:
            # The fill raced the cancel: IBKR rejects the cancel and the order ends filled.
            self.fill_status, self.filled_qty = "filled", 54
        else:
            self.fill_status = "cancelled"
        return True

    def _order(self, symbol, quantity, order_id=None):
        return {
            "order_id": order_id or f"order-{self.place_calls}",
            "symbol": symbol,
            "quantity": quantity,
            "status": self.fill_status,
            "filled_qty": self.filled_qty,
            "filled_avg_price": 36.55 if self.filled_qty else None,
        }


def make_engine(broker):
    engine = ExecutionEngine(broker, max_retries=3, retry_delay=0.0, fill_timeout=0.2)
    engine.settle_timeout = 0.2
    return engine


def test_unfilled_accepted_order_is_cancelled_and_never_resubmitted():
    broker = FakeBroker(fill_status="pending")

    with pytest.raises(BrokerError, match="did not fill"):
        make_engine(broker).execute_order("IP", 54, OrderSide.SELL, OrderType.MARKET)

    assert broker.place_calls == 1
    assert broker.cancelled == ["order-1"]


def test_partially_filled_order_reports_the_fill_and_is_not_resubmitted():
    broker = FakeBroker(fill_status="pending", filled_qty=37)

    with pytest.raises(BrokerError, match="37"):
        make_engine(broker).execute_order("IP", 54, OrderSide.SELL, OrderType.MARKET)

    assert broker.place_calls == 1
    assert broker.cancelled == ["order-1"]


def test_order_rejected_after_acceptance_is_not_resubmitted():
    broker = FakeBroker(fill_status="rejected")

    with pytest.raises(BrokerError):
        make_engine(broker).execute_order("IP", 54, OrderSide.SELL, OrderType.MARKET)

    assert broker.place_calls == 1


def test_placement_failures_are_still_retried():
    broker = FakeBroker(place_failures=2, fill_status="filled", filled_qty=54)

    execution = make_engine(broker).execute_order("IP", 54, OrderSide.SELL, OrderType.MARKET)

    assert broker.place_calls == 3
    assert execution["order"]["status"] == "filled"
    assert broker.cancelled == []


def test_placement_failing_every_time_gives_up_after_max_retries():
    broker = FakeBroker(place_failures=10)

    with pytest.raises(BrokerError, match="after 3 attempts"):
        make_engine(broker).execute_order("IP", 54, OrderSide.SELL, OrderType.MARKET)

    assert broker.place_calls == 3
    assert broker.cancelled == []


def test_fill_landing_during_the_cancel_is_reported_as_success():
    broker = FakeBroker(fill_status="pending", fill_on_cancel=True)

    execution = make_engine(broker).execute_order("IP", 54, OrderSide.SELL, OrderType.MARKET)

    assert execution["order"]["status"] == "filled"
    assert execution["order"]["filled_qty"] == 54
    assert broker.place_calls == 1


@pytest.mark.parametrize("cancel_error", [BrokerConnectionError("gateway gone"), RuntimeError("ib torn down")])
def test_failed_cancel_still_never_resubmits(cancel_error):
    broker = FakeBroker(fill_status="pending", cancel_error=cancel_error)

    with pytest.raises(BrokerError, match="did not fill"):
        make_engine(broker).execute_order("IP", 54, OrderSide.SELL, OrderType.MARKET)

    assert broker.place_calls == 1


def test_unreadable_status_still_never_resubmits():
    broker = FakeBroker(fill_status="pending", status_error=BrokerConnectionError("status unavailable"))

    with pytest.raises(BrokerError):
        make_engine(broker).execute_order("IP", 54, OrderSide.SELL, OrderType.MARKET)

    assert broker.place_calls == 1
    assert broker.cancelled == ["order-1"]
