"""
Execution Engine - Broker-Agnostic Order Execution and Management

Handles order execution with retry logic, status tracking, and execution analytics.
This component is completely broker-agnostic and works with any BrokerInterface implementation.

Design Principles:
- Single Responsibility: Only handles order execution and tracking
- Broker-Agnostic: No dependencies on specific broker implementations
- Retry Logic: Handles transient failures gracefully
- Analytics: Tracks execution performance metrics
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import time

from src.trading.brokers.broker_interface import (
    BrokerInterface,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    BrokerError,
    InvalidOrderError,
)
from src.utils.logger import get_logger

logger = get_logger()  # Use global logger (no file creation)


class ExecutionStatus(Enum):
    """Execution attempt status."""
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"


class ExecutionEngine:
    """
    Broker-agnostic execution engine.

    Handles order execution with retry logic, status tracking, and analytics.
    Works with any broker implementation via BrokerInterface.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        fill_timeout: float = 30.0
    ):
        """
        Initialize execution engine.

        Args:
            broker: Broker interface implementation
            max_retries: Maximum order retry attempts
            retry_delay: Delay between retries (seconds)
            fill_timeout: Maximum time to wait for fill (seconds)
        """
        self.broker = broker
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fill_timeout = fill_timeout

        # Order tracking
        self.orders: Dict[str, Dict] = {}
        self.execution_history: List[Dict] = []

        # Performance metrics
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.retry_count = 0

        logger.info(
            f"Initialized ExecutionEngine (max_retries={max_retries}, "
            f"retry_delay={retry_delay}s, fill_timeout={fill_timeout}s)"
        )

    # ==================== Order Execution ====================

    def execute_order(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        wait_for_fill: bool = True,
        **kwargs
    ) -> Dict:
        """
        Execute order with retry logic and status tracking.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/STOP/STOP_LIMIT)
            limit_price: Limit price (for LIMIT/STOP_LIMIT orders)
            stop_price: Stop price (for STOP/STOP_LIMIT orders)
            time_in_force: Time in force (DAY/GTC/IOC/FOK)
            wait_for_fill: Whether to wait for order to fill
            **kwargs: Additional broker-specific parameters

        Returns:
            Dict with execution results including order details and status

        Raises:
            BrokerError: If order execution fails after all retries
        """
        self.total_orders += 1
        execution_start = datetime.now()

        logger.info(
            f"Executing order: {side.value} {quantity} {symbol} @ {order_type.value}"
        )

        # Track execution attempt
        execution = {
            'symbol': symbol,
            'quantity': quantity,
            'side': side.value,
            'order_type': order_type.value,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'time_in_force': time_in_force.value,
            'start_time': execution_start,
            'attempts': 0,
            'status': ExecutionStatus.RETRY,
        }

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            execution['attempts'] += 1

            try:
                # Place order
                order = self._place_order_with_validation(
                    symbol=symbol,
                    quantity=quantity,
                    side=side,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    time_in_force=time_in_force,
                    **kwargs
                )

                # Store order
                self.orders[order['order_id']] = order

                # Wait for fill if requested
                if wait_for_fill and order['status'] != OrderStatus.FILLED.value:
                    order = self._wait_for_fill(order['order_id'])

                # Mark execution as successful
                execution['status'] = ExecutionStatus.SUCCESS
                execution['order'] = order
                execution['end_time'] = datetime.now()
                execution['duration'] = (execution['end_time'] - execution_start).total_seconds()

                self.execution_history.append(execution)
                self.successful_orders += 1

                logger.success(
                    f"Order executed successfully: {order['order_id']} | "
                    f"Filled {order.get('filled_qty', 0)} @ ${order.get('filled_avg_price', 0):.2f} | "
                    f"Duration: {execution['duration']:.2f}s"
                )

                return execution

            except InvalidOrderError as e:
                # Don't retry invalid orders
                logger.error(f"Invalid order (no retry): {e}")
                execution['status'] = ExecutionStatus.FAILED
                execution['error'] = str(e)
                execution['end_time'] = datetime.now()
                self.execution_history.append(execution)
                self.failed_orders += 1
                raise

            except BrokerError as e:
                last_error = e
                logger.warning(f"Order attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay}s...")
                    self.retry_count += 1
                    time.sleep(self.retry_delay)
                else:
                    # All retries exhausted
                    logger.error(f"Order execution failed after {self.max_retries} attempts")
                    execution['status'] = ExecutionStatus.FAILED
                    execution['error'] = str(last_error)
                    execution['end_time'] = datetime.now()
                    self.execution_history.append(execution)
                    self.failed_orders += 1
                    raise BrokerError(f"Order execution failed after {self.max_retries} attempts: {last_error}")

        # Should not reach here
        raise BrokerError("Unexpected execution path")

    def execute_batch(
        self,
        orders: List[Dict],
        sequential: bool = False
    ) -> List[Dict]:
        """
        Execute multiple orders.

        Args:
            orders: List of order dicts with execution parameters
            sequential: If True, execute orders sequentially; if False, execute in parallel

        Returns:
            List of execution results
        """
        logger.info(f"Executing batch of {len(orders)} orders ({'sequential' if sequential else 'parallel'})")

        results = []

        if sequential:
            # Execute orders one by one
            for order_params in orders:
                try:
                    result = self.execute_order(**order_params)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch order failed: {e}")
                    results.append({
                        'status': ExecutionStatus.FAILED,
                        'error': str(e),
                        'params': order_params
                    })
        else:
            # Execute orders in parallel (simplified - can use threading/asyncio for true parallelism)
            for order_params in orders:
                try:
                    result = self.execute_order(**order_params)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch order failed: {e}")
                    results.append({
                        'status': ExecutionStatus.FAILED,
                        'error': str(e),
                        'params': order_params
                    })

        successful = sum(1 for r in results if r.get('status') == ExecutionStatus.SUCCESS)
        logger.info(f"Batch execution complete: {successful}/{len(orders)} successful")

        return results

    # ==================== Order Management ====================

    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel order with retry logic.

        Args:
            order_id: Order ID to cancel

        Returns:
            Updated order dict
        """
        logger.info(f"Cancelling order: {order_id}")

        for attempt in range(self.max_retries):
            try:
                order = self.broker.cancel_order(order_id)
                self.orders[order_id] = order
                logger.success(f"Order cancelled: {order_id}")
                return order

            except BrokerError as e:
                logger.warning(f"Cancel attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise

        raise BrokerError(f"Failed to cancel order {order_id}")

    def get_order_status(self, order_id: str) -> str:
        """
        Get current order status.

        Args:
            order_id: Order ID

        Returns:
            Order status string
        """
        try:
            order = self.broker.get_order(order_id)
            self.orders[order_id] = order
            return order['status']
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return "unknown"

    def close_position(
        self,
        symbol: str,
        wait_for_fill: bool = True
    ) -> Dict:
        """
        Close position for symbol.

        Args:
            symbol: Symbol to close
            wait_for_fill: Whether to wait for fill confirmation

        Returns:
            Execution result dict
        """
        logger.info(f"Closing position: {symbol}")

        try:
            # Get current position
            position = self.broker.get_position(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}")
                return {'status': ExecutionStatus.FAILED, 'error': 'No position'}

            quantity = abs(position['quantity'])
            side = OrderSide.SELL if position['quantity'] > 0 else OrderSide.BUY

            # Execute closing order
            return self.execute_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type=OrderType.MARKET,
                wait_for_fill=wait_for_fill
            )

        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            raise

    def close_all_positions(self, wait_for_fill: bool = True) -> List[Dict]:
        """
        Close all open positions.

        Args:
            wait_for_fill: Whether to wait for fill confirmations

        Returns:
            List of execution results
        """
        positions = self.broker.get_positions()
        logger.info(f"Closing {len(positions)} positions")

        results = []
        for position in positions:
            try:
                result = self.close_position(position['symbol'], wait_for_fill=wait_for_fill)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to close {position['symbol']}: {e}")
                results.append({
                    'status': ExecutionStatus.FAILED,
                    'error': str(e),
                    'symbol': position['symbol']
                })

        return results

    # ==================== Helper Methods ====================

    def _place_order_with_validation(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        order_type: OrderType,
        limit_price: Optional[float],
        stop_price: Optional[float],
        time_in_force: TimeInForce,
        **kwargs
    ) -> Dict:
        """Place order with parameter validation."""
        # Validate parameters
        if quantity <= 0:
            raise InvalidOrderError(f"Invalid quantity: {quantity}")

        if order_type == OrderType.LIMIT and limit_price is None:
            raise InvalidOrderError("Limit price required for LIMIT orders")

        if order_type == OrderType.STOP and stop_price is None:
            raise InvalidOrderError("Stop price required for STOP orders")

        if order_type == OrderType.STOP_LIMIT and (limit_price is None or stop_price is None):
            raise InvalidOrderError("Both limit and stop prices required for STOP_LIMIT orders")

        # Place order via broker
        return self.broker.place_order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            **kwargs
        )

    def _wait_for_fill(self, order_id: str) -> Dict:
        """
        Wait for order to fill.

        Args:
            order_id: Order ID to monitor

        Returns:
            Updated order dict with fill information

        Raises:
            BrokerError: If order doesn't fill within timeout
        """
        start_time = datetime.now()
        timeout = timedelta(seconds=self.fill_timeout)

        logger.info(f"Waiting for fill: {order_id} (timeout: {self.fill_timeout}s)")

        while datetime.now() - start_time < timeout:
            order = self.broker.get_order(order_id)
            status = order['status']

            if status == OrderStatus.FILLED.value:
                logger.success(f"Order filled: {order_id}")
                return order

            if status in [OrderStatus.CANCELLED.value, OrderStatus.REJECTED.value]:
                raise BrokerError(f"Order {order_id} failed with status: {status}")

            time.sleep(0.5)  # Check every 0.5 seconds

        raise BrokerError(f"Order {order_id} did not fill within {self.fill_timeout}s")

    # ==================== Analytics ====================

    def get_execution_metrics(self) -> Dict:
        """
        Get execution performance metrics.

        Returns:
            Dict with execution statistics
        """
        if self.total_orders == 0:
            return {
                'total_orders': 0,
                'successful_orders': 0,
                'failed_orders': 0,
                'success_rate': 0.0,
                'retry_count': 0,
                'avg_retries_per_order': 0.0,
            }

        success_rate = self.successful_orders / self.total_orders if self.total_orders > 0 else 0.0
        avg_retries = self.retry_count / self.total_orders if self.total_orders > 0 else 0.0

        # Calculate average execution time
        successful_executions = [e for e in self.execution_history if e['status'] == ExecutionStatus.SUCCESS]
        avg_duration = (
            sum(e['duration'] for e in successful_executions) / len(successful_executions)
            if successful_executions else 0.0
        )

        return {
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'failed_orders': self.failed_orders,
            'success_rate': success_rate,
            'retry_count': self.retry_count,
            'avg_retries_per_order': avg_retries,
            'avg_execution_time': avg_duration,
        }

    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get execution history.

        Args:
            limit: Optional limit on number of executions to return

        Returns:
            List of execution records (most recent first)
        """
        history = sorted(
            self.execution_history,
            key=lambda x: x['start_time'],
            reverse=True
        )
        return history[:limit] if limit else history
