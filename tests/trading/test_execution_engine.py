"""
ExecutionEngine Unit Tests

Tests for broker-agnostic order execution engine.
"""

import pytest
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.core.execution_engine import ExecutionEngine, ExecutionStatus
from src.trading.brokers.broker_interface import (
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    BrokerError,
    InvalidOrderError,
)
from tests.trading.mock_broker import MockBroker


class TestExecutionEngine:
    """Test suite for ExecutionEngine."""

    @pytest.fixture
    def broker(self):
        """Create mock broker."""
        return MockBroker(initial_cash=100000.0)

    @pytest.fixture
    def engine(self, broker):
        """Create execution engine."""
        return ExecutionEngine(
            broker=broker,
            max_retries=3,
            retry_delay=0.1,  # Fast retries for testing
            fill_timeout=5.0
        )

    # ==================== Initialization Tests ====================

    def test_initialization(self, engine):
        """Test ExecutionEngine initialization."""
        assert engine.max_retries == 3
        assert engine.retry_delay == 0.1
        assert engine.fill_timeout == 5.0
        assert len(engine.orders) == 0
        assert len(engine.execution_history) == 0
        assert engine.total_orders == 0

    # ==================== Order Execution Tests ====================

    def test_execute_market_order_success(self, broker, engine):
        """Test successful market order execution."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)

        result = engine.execute_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            wait_for_fill=False
        )

        assert result['status'] == ExecutionStatus.SUCCESS
        assert result['symbol'] == 'AAPL'
        assert result['quantity'] == 10
        assert result['attempts'] == 1
        assert 'order' in result
        assert result['order']['status'] == OrderStatus.FILLED.value

    def test_execute_limit_order(self, broker, engine):
        """Test limit order execution."""
        result = engine.execute_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
            wait_for_fill=False
        )

        assert result['status'] == ExecutionStatus.SUCCESS
        assert result['order']['order_type'] == OrderType.LIMIT.value
        assert result['order']['limit_price'] == 150.0

    def test_execute_stop_order(self, broker, engine):
        """Test stop order execution."""
        result = engine.execute_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=155.0,
            wait_for_fill=False
        )

        assert result['status'] == ExecutionStatus.SUCCESS
        assert result['order']['order_type'] == OrderType.STOP.value
        assert result['order']['stop_price'] == 155.0

    def test_execute_invalid_order_no_retry(self, broker, engine):
        """Test that invalid orders are not retried."""
        with pytest.raises(InvalidOrderError):
            engine.execute_order(
                symbol='AAPL',
                quantity=-10,  # Invalid quantity
                side=OrderSide.BUY,
                wait_for_fill=False
            )

        # Should fail immediately without retries
        assert engine.total_orders == 1
        assert engine.failed_orders == 1
        assert engine.retry_count == 0

    def test_execution_tracking(self, broker, engine):
        """Test that executions are tracked."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)

        engine.execute_order('AAPL', 10, OrderSide.BUY, wait_for_fill=False)
        engine.execute_order('MSFT', 5, OrderSide.BUY, wait_for_fill=False)

        assert engine.total_orders == 2
        assert engine.successful_orders == 2
        assert len(engine.execution_history) == 2
        assert len(engine.orders) == 2

    # ==================== Batch Execution Tests ====================

    def test_execute_batch_sequential(self, broker, engine):
        """Test sequential batch execution."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        broker.set_quote('MSFT', bid=300.0, ask=300.05)

        orders = [
            {'symbol': 'AAPL', 'quantity': 10, 'side': OrderSide.BUY, 'wait_for_fill': False},
            {'symbol': 'MSFT', 'quantity': 5, 'side': OrderSide.BUY, 'wait_for_fill': False},
        ]

        results = engine.execute_batch(orders, sequential=True)

        assert len(results) == 2
        assert all(r['status'] == ExecutionStatus.SUCCESS for r in results)
        assert engine.total_orders == 2

    def test_execute_batch_parallel(self, broker, engine):
        """Test parallel batch execution."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        broker.set_quote('MSFT', bid=300.0, ask=300.05)
        broker.set_quote('TSLA', bid=200.0, ask=200.05)

        orders = [
            {'symbol': 'AAPL', 'quantity': 10, 'side': OrderSide.BUY, 'wait_for_fill': False},
            {'symbol': 'MSFT', 'quantity': 5, 'side': OrderSide.BUY, 'wait_for_fill': False},
            {'symbol': 'TSLA', 'quantity': 8, 'side': OrderSide.BUY, 'wait_for_fill': False},
        ]

        results = engine.execute_batch(orders, sequential=False)

        assert len(results) == 3
        assert engine.total_orders == 3

    # ==================== Position Management Tests ====================

    def test_close_position(self, broker, engine):
        """Test closing a position."""
        # First open a position
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        engine.execute_order('AAPL', 10, OrderSide.BUY, wait_for_fill=False)

        # Then close it
        broker.set_quote('AAPL', bid=155.0, ask=155.05)
        result = engine.close_position('AAPL', wait_for_fill=False)

        assert result['status'] == ExecutionStatus.SUCCESS
        assert result['order']['side'] == OrderSide.SELL.value
        assert result['order']['quantity'] == 10

    def test_close_nonexistent_position(self, broker, engine):
        """Test closing non-existent position."""
        result = engine.close_position('AAPL', wait_for_fill=False)

        assert result['status'] == ExecutionStatus.FAILED
        assert 'error' in result

    def test_close_all_positions(self, broker, engine):
        """Test closing all positions."""
        # Open multiple positions
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        broker.set_quote('MSFT', bid=300.0, ask=300.05)

        engine.execute_order('AAPL', 10, OrderSide.BUY, wait_for_fill=False)
        engine.execute_order('MSFT', 5, OrderSide.BUY, wait_for_fill=False)

        # Close all
        results = engine.close_all_positions(wait_for_fill=False)

        assert len(results) == 2
        assert all(r['status'] == ExecutionStatus.SUCCESS for r in results)

    # ==================== Order Management Tests ====================

    def test_cancel_order(self, broker, engine):
        """Test order cancellation."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        result = engine.execute_order('AAPL', 10, OrderSide.BUY, wait_for_fill=False)

        order_id = result['order']['order_id']
        cancelled = engine.cancel_order(order_id)

        assert cancelled['status'] == OrderStatus.CANCELLED.value

    def test_get_order_status(self, broker, engine):
        """Test getting order status."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)
        result = engine.execute_order('AAPL', 10, OrderSide.BUY, wait_for_fill=False)

        order_id = result['order']['order_id']
        status = engine.get_order_status(order_id)

        assert status == OrderStatus.FILLED.value

    # ==================== Validation Tests ====================

    def test_validate_limit_order_requires_price(self, broker, engine):
        """Test that limit orders require limit price."""
        with pytest.raises(InvalidOrderError, match="Limit price required"):
            engine.execute_order(
                symbol='AAPL',
                quantity=10,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                wait_for_fill=False
            )

    def test_validate_stop_order_requires_price(self, broker, engine):
        """Test that stop orders require stop price."""
        with pytest.raises(InvalidOrderError, match="Stop price required"):
            engine.execute_order(
                symbol='AAPL',
                quantity=10,
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
                wait_for_fill=False
            )

    def test_validate_invalid_quantity(self, broker, engine):
        """Test that invalid quantities are rejected."""
        with pytest.raises(InvalidOrderError, match="Invalid quantity"):
            engine.execute_order(
                symbol='AAPL',
                quantity=0,
                side=OrderSide.BUY,
                wait_for_fill=False
            )

    # ==================== Analytics Tests ====================

    def test_get_execution_metrics_empty(self, engine):
        """Test metrics with no executions."""
        metrics = engine.get_execution_metrics()

        assert metrics['total_orders'] == 0
        assert metrics['successful_orders'] == 0
        assert metrics['failed_orders'] == 0
        assert metrics['success_rate'] == 0.0
        assert metrics['retry_count'] == 0

    def test_get_execution_metrics(self, broker, engine):
        """Test execution metrics calculation."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)

        # Execute successful orders
        engine.execute_order('AAPL', 10, OrderSide.BUY, wait_for_fill=False)
        engine.execute_order('AAPL', 5, OrderSide.BUY, wait_for_fill=False)

        # Try invalid order
        try:
            engine.execute_order('AAPL', -10, OrderSide.BUY, wait_for_fill=False)
        except:
            pass

        metrics = engine.get_execution_metrics()

        assert metrics['total_orders'] == 3
        assert metrics['successful_orders'] == 2
        assert metrics['failed_orders'] == 1
        assert metrics['success_rate'] == pytest.approx(2/3, rel=1e-2)

    def test_get_execution_history(self, broker, engine):
        """Test getting execution history."""
        broker.set_quote('AAPL', bid=150.0, ask=150.05)

        engine.execute_order('AAPL', 10, OrderSide.BUY, wait_for_fill=False)
        engine.execute_order('AAPL', 5, OrderSide.BUY, wait_for_fill=False)

        history = engine.get_execution_history()

        assert len(history) == 2
        assert all('start_time' in h for h in history)
        assert all('status' in h for h in history)

        # Test limit
        history = engine.get_execution_history(limit=1)
        assert len(history) == 1

    # ==================== Integration Tests ====================

    def test_complete_trade_workflow(self, broker, engine):
        """Test complete buy and sell workflow."""
        # Set initial quote
        broker.set_quote('AAPL', bid=150.0, ask=150.05)

        # Buy
        buy_result = engine.execute_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            wait_for_fill=False
        )

        assert buy_result['status'] == ExecutionStatus.SUCCESS
        assert len(broker.get_positions()) == 1

        # Update quote
        broker.set_quote('AAPL', bid=155.0, ask=155.05)

        # Sell
        sell_result = engine.execute_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.SELL,
            wait_for_fill=False
        )

        assert sell_result['status'] == ExecutionStatus.SUCCESS
        assert len(broker.get_positions()) == 0

    def test_multiple_symbols_workflow(self, broker, engine):
        """Test trading multiple symbols."""
        symbols = ['AAPL', 'MSFT', 'TSLA']

        for symbol in symbols:
            broker.set_quote(symbol, bid=100.0, ask=100.05)

        # Open positions
        for symbol in symbols:
            result = engine.execute_order(
                symbol=symbol,
                quantity=10,
                side=OrderSide.BUY,
                wait_for_fill=False
            )
            assert result['status'] == ExecutionStatus.SUCCESS

        assert len(broker.get_positions()) == 3

        # Close all
        results = engine.close_all_positions(wait_for_fill=False)
        assert len(results) == 3
        assert len(broker.get_positions()) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
