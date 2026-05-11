"""Tests for MarginGuard."""
from unittest.mock import MagicMock

import pytest

from src.trading.futures.margin_guard import (
    MarginCheckResult, MarginGuard, MarginVerdict,
)
from src.trading.futures.symbol_resolver import ResolvedOrder
from src.trading.brokers.interfaces.base import OrderSide, OrderType, TimeInForce
from datetime import date


def _mk_order() -> ResolvedOrder:
    return ResolvedOrder(
        strategy_intent="ES.v.0",
        symbol_root="ES",
        contract_month="202606",
        raw_symbol="ESM6",
        side=OrderSide.BUY,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=5300.0,
        stop_price=None,
        time_in_force=TimeInForce.DAY,
        strategy="adaptation_d",
        as_of=date(2026, 5, 11),
    )


def _mk_broker(
    net_liquidation: float = 50_000.0,
    initial_after: float = 5_000.0,
    maintenance_after: float = 4_000.0,
) -> MagicMock:
    broker = MagicMock()
    broker.get_margin_status.return_value = {
        "net_liquidation": net_liquidation,
        "initial_margin": 1_000.0,
        "maintenance_margin": 800.0,
        "free_cash": net_liquidation - 800.0,
        "buying_power": net_liquidation * 4,
    }
    broker.what_if_order.return_value = {
        "initial_margin_after": initial_after,
        "maintenance_margin_after": maintenance_after,
        "buying_power_after": net_liquidation * 4 - maintenance_after,
    }
    return broker


def test_pre_trade_check_ok():
    """Maintenance $4k on $50k equity leaves $46k free (92%); well above 30% buffer."""
    broker = _mk_broker()
    guard = MarginGuard(broker=broker)
    result = guard.pre_trade_check(_mk_order(), hold_overnight=False)
    assert result.verdict == MarginVerdict.OK


def test_pre_trade_check_rejects_when_maintenance_exceeds_buffer():
    """Maintenance $40k on $50k equity leaves only $10k free (20%); below 30%."""
    broker = _mk_broker(net_liquidation=50_000, maintenance_after=40_000.0)
    guard = MarginGuard(broker=broker)
    result = guard.pre_trade_check(_mk_order(), hold_overnight=False)
    assert result.verdict == MarginVerdict.REJECT
    assert "buffer" in result.reason.lower() or "30%" in result.reason


def test_pre_trade_check_rejects_overnight():
    """Maintenance $10k looks fine intraday, but overnight 2x initial pushes us over."""
    broker = _mk_broker(net_liquidation=50_000, initial_after=20_000.0, maintenance_after=10_000.0)
    # overnight margin = 20_000 * 2 = 40_000. Free cash = 50_000 - 40_000 = 10_000 (20%) < 30%
    guard = MarginGuard(broker=broker)
    result = guard.pre_trade_check(_mk_order(), hold_overnight=True)
    assert result.verdict == MarginVerdict.REJECT_OVERNIGHT


def test_pre_trade_check_overnight_passes_when_within_buffer():
    """Maintenance $4k AND overnight 2x initial $10k = $10k both leave plenty of buffer."""
    broker = _mk_broker(net_liquidation=50_000, initial_after=5_000.0, maintenance_after=4_000.0)
    # overnight margin = 5_000 * 2 = 10_000. Free cash = 50_000 - 10_000 = 40_000 (80%)
    guard = MarginGuard(broker=broker)
    result = guard.pre_trade_check(_mk_order(), hold_overnight=True)
    assert result.verdict == MarginVerdict.OK


def test_pre_trade_check_calls_what_if_order():
    """Guard must invoke broker.what_if_order with the order's parameters."""
    broker = _mk_broker()
    guard = MarginGuard(broker=broker)
    order = _mk_order()
    guard.pre_trade_check(order, hold_overnight=False)
    broker.what_if_order.assert_called_once()
    kwargs = broker.what_if_order.call_args.kwargs
    assert kwargs["symbol_root"] == "ES"
    assert kwargs["contract_month"] == "202606"
    assert kwargs["side"] == OrderSide.BUY
    assert kwargs["quantity"] == 2


def test_cash_buffer_constant():
    """30% buffer is non-configurable -- it's a Homeguard CLAUDE.md rule."""
    assert MarginGuard.CASH_BUFFER_PCT == 0.30
