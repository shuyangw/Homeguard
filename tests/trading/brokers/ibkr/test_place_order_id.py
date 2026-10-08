"""A placement that fails after placeOrder ran must tell the caller which order IBKR may now hold.

Otherwise ExecutionEngine treats it as "no order" and places it again (review #1 of the
2026-10-07 duplicate-order fixes).
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.trading.brokers.ibkr.ibkr_broker import IBKRBroker
from src.trading.brokers.interfaces import BrokerConnectionError, InsufficientFundsError, OrderSide

ASSIGNED_ORDER_ID = 4242


class FakeIB:
    def __init__(self, trade=None):
        self.client = SimpleNamespace(getReqId=lambda: ASSIGNED_ORDER_ID)
        self.trade = trade
        self.placed = []

    def placeOrder(self, contract, order):
        self.placed.append(order)
        return self.trade


class FakeResolver:
    def __init__(self, error=None):
        self.error = error

    def resolve_stock(self, symbol):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(symbol=symbol)


def make_broker(ib, run_sync, resolver=None):
    broker = IBKRBroker()
    broker._conn = SimpleNamespace(ib=ib, run_sync=run_sync)
    broker._resolver = resolver or FakeResolver()
    return broker


def run_with_timeout(coro, timeout=None):
    return asyncio.run(asyncio.wait_for(coro, 0.01))


def run_to_completion(coro, timeout=None):
    return asyncio.run(coro)


def test_timeout_after_place_order_reports_the_assigned_order_id():
    ib = FakeIB()
    broker = make_broker(ib, run_with_timeout)

    with pytest.raises(BrokerConnectionError) as raised:
        broker.place_stock_order("IP", 54, OrderSide.SELL)

    assert len(ib.placed) == 1
    assert ib.placed[0].orderId == ASSIGNED_ORDER_ID
    assert raised.value.order_id == str(ASSIGNED_ORDER_ID)


def test_untranslatable_trade_reports_the_assigned_order_id():
    ib = FakeIB(trade=SimpleNamespace())
    broker = make_broker(ib, run_to_completion)

    with pytest.raises(BrokerConnectionError) as raised:
        broker.place_stock_order("IP", 54, OrderSide.SELL)

    assert raised.value.order_id == str(ASSIGNED_ORDER_ID)


def test_insufficient_funds_after_placement_reports_the_assigned_order_id():
    ib = FakeIB()

    def run_then_fail(coro, timeout=None):
        result = asyncio.run(coro)
        if ib.placed:
            raise RuntimeError("insufficient margin")
        return result

    broker = make_broker(ib, run_then_fail)

    with pytest.raises(InsufficientFundsError) as raised:
        broker.place_stock_order("IP", 54, OrderSide.SELL)

    assert raised.value.order_id == str(ASSIGNED_ORDER_ID)


def test_order_id_is_drawn_on_the_event_loop():
    ib = FakeIB(trade=SimpleNamespace())
    on_loop = []

    def take_id():
        assert on_loop, "getReqId called outside run_sync"
        return ASSIGNED_ORDER_ID

    def run_on_loop(coro, timeout=None):
        on_loop.append(True)
        try:
            return asyncio.run(coro)
        finally:
            on_loop.pop()

    ib.client = SimpleNamespace(getReqId=take_id)
    broker = make_broker(ib, run_on_loop)

    with pytest.raises(BrokerConnectionError):
        broker.place_stock_order("IP", 54, OrderSide.SELL)

    assert ib.placed[0].orderId == ASSIGNED_ORDER_ID


def test_failure_before_placement_reports_no_order_id():
    ib = FakeIB()
    broker = make_broker(ib, run_to_completion, resolver=FakeResolver(error=RuntimeError("no such contract")))

    with pytest.raises(BrokerConnectionError) as raised:
        broker.place_stock_order("IP", 54, OrderSide.SELL)

    assert ib.placed == []
    assert getattr(raised.value, "order_id", None) is None
