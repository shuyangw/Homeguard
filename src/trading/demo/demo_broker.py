"""
Demo Broker - Custom paper trading with Binance streaming.

Implements CryptoTradingInterface with simulated execution, providing
realistic paper trading for crypto strategies without Alpaca's limitations.

Key Features:
- Fractional quantities (0.00001 BTC)
- Real-time Binance streaming
- Simulated slippage and fees
- Persistent portfolio state
- Complete trade logging
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from src.data.providers.binance import BinanceDataProvider
from src.streaming.binance_stream import BinanceStreamManager
from src.streaming.crypto_buffer import CryptoBarBuffer
from src.trading.brokers.interfaces.base import (
    InvalidOrderError,
    InsufficientFundsError,
    NoPositionError,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from src.trading.brokers.interfaces.crypto_trading import CryptoTradingInterface
from src.trading.demo.execution_simulator import ExecutionSimulator
from src.trading.demo.portfolio_state import DemoPortfolioState
from src.trading.demo.trade_logger import DemoTradeLogger
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger(__name__)


class DemoBroker(CryptoTradingInterface):
    """
    Demo broker with Binance streaming and simulated execution.

    Replaces Alpaca paper trading for crypto, providing:
    - Fractional quantity support
    - Real-time Binance price data
    - Simulated slippage and fees
    - Persistent portfolio state
    - Trade logging with portfolio snapshots

    Example:
        broker = DemoBroker(initial_cash=10000.0)
        broker.start_streaming(['BTC/USD', 'ETH/USD', 'MKR/USD'])

        # Buy 0.001 BTC
        order = broker.place_crypto_order(
            symbol='BTC/USD',
            quantity=Decimal('0.001'),
            side=OrderSide.BUY
        )

        # Check positions
        positions = broker.get_crypto_positions()

        # Get current price
        quote = broker.get_crypto_quote('BTC/USD')

        # Close when done
        broker.stop_streaming()
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        slippage_bps: float = 5.0,
        fee_bps: float = 10.0,
        auto_start_streaming: bool = False,
        symbols: Optional[List[str]] = None,
    ):
        """
        Initialize demo broker.

        Args:
            initial_cash: Starting cash balance (used if no saved state)
            slippage_bps: Slippage in basis points (default: 5)
            fee_bps: Trading fee in basis points (default: 10)
            auto_start_streaming: Whether to start streaming immediately
            symbols: Symbols to stream (required if auto_start_streaming=True)
        """
        # Core components
        self._portfolio = DemoPortfolioState(initial_cash=initial_cash)
        self._execution_sim = ExecutionSimulator(
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
        )
        self._trade_logger = DemoTradeLogger()
        self._bar_buffer = CryptoBarBuffer()

        # Streaming
        self._stream_manager: Optional[BinanceStreamManager] = None
        self._symbols: List[str] = symbols or []
        self._streaming = False

        # REST API fallback for quotes when streaming isn't available
        self._rest_provider = BinanceDataProvider()

        # Order tracking
        self._orders: Dict[str, Dict] = {}
        self._order_counter = 0

        logger.info(
            f"[DemoBroker] Initialized with ${initial_cash:.2f} cash, "
            f"slippage={slippage_bps}bps, fees={fee_bps}bps"
        )

        if auto_start_streaming and symbols:
            self.start_streaming(symbols)

    # ==================== Streaming ====================

    def start_streaming(self, symbols: List[str]) -> None:
        """
        Start streaming real-time data for symbols.

        Args:
            symbols: List of symbols in 'BTC/USD' format
        """
        if self._streaming:
            logger.warning("[DemoBroker] Already streaming, stopping first")
            self.stop_streaming()

        self._symbols = symbols
        self._stream_manager = BinanceStreamManager(symbols)
        self._stream_manager.subscribe_klines(self._on_bar)
        self._stream_manager.start()
        self._streaming = True

        logger.info(f"[DemoBroker] Started streaming {len(symbols)} symbols")

    def stop_streaming(self) -> None:
        """Stop streaming data."""
        if self._stream_manager:
            self._stream_manager.stop()
            self._stream_manager = None
        self._streaming = False
        logger.info("[DemoBroker] Stopped streaming")

    def is_streaming(self) -> bool:
        """Check if streaming is active."""
        return self._streaming and (
            self._stream_manager is not None
            and self._stream_manager.is_connected()
        )

    def _on_bar(self, bar) -> None:
        """Handle incoming bar from stream."""
        self._bar_buffer.add_bar(bar)

    # ==================== CryptoTradingInterface ====================

    def get_crypto_positions(self) -> List[Dict]:
        """Get all current crypto positions."""
        positions = []
        prices = self._get_prices()

        for pos in self._portfolio.get_all_positions():
            current_price = prices.get(pos.symbol, pos.avg_entry_price)
            positions.append({
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_entry_price": pos.avg_entry_price,
                "current_price": current_price,
                "market_value": pos.market_value(current_price),
                "unrealized_pnl": pos.unrealized_pnl(current_price),
                "unrealized_pnl_pct": pos.unrealized_pnl_pct(current_price),
                "cost_basis": pos.cost_basis,
            })

        return positions

    def get_crypto_position(self, symbol: str) -> Optional[Dict]:
        """Get specific crypto position by symbol."""
        symbol = self.normalize_symbol(symbol)
        pos = self._portfolio.get_position(symbol)

        if pos is None:
            return None

        # Get price from streaming or REST fallback
        prices = self._get_prices([symbol])
        current_price = prices.get(symbol, pos.avg_entry_price)

        return {
            "symbol": pos.symbol,
            "quantity": pos.quantity,
            "avg_entry_price": pos.avg_entry_price,
            "current_price": current_price,
            "market_value": pos.market_value(current_price),
            "unrealized_pnl": pos.unrealized_pnl(current_price),
            "unrealized_pnl_pct": pos.unrealized_pnl_pct(current_price),
            "cost_basis": pos.cost_basis,
        }

    def place_crypto_order(
        self,
        symbol: str,
        quantity: Decimal,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        **kwargs
    ) -> Dict:
        """
        Place a crypto order with simulated execution.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            quantity: Amount to buy/sell (Decimal for precision)
            side: OrderSide.BUY or OrderSide.SELL
            order_type: Order type (MARKET only for now)
            limit_price: Not used (market orders only)
            time_in_force: Not used
            **kwargs: Additional parameters

        Returns:
            Dict with order details
        """
        symbol = self.normalize_symbol(symbol)
        quantity = Decimal(str(quantity))

        # Validate
        if quantity <= 0:
            raise InvalidOrderError(f"Quantity must be positive, got {quantity}")

        if order_type != OrderType.MARKET:
            raise InvalidOrderError("Only MARKET orders supported in demo mode")

        # Get current price
        current_price = self._bar_buffer.get_latest_price(symbol)
        if current_price is None:
            raise InvalidOrderError(
                f"No price available for {symbol}. "
                f"Ensure streaming is active and symbol is subscribed."
            )

        notional = float(quantity) * current_price

        # Validate funds/position
        if side == OrderSide.BUY:
            if notional > self._portfolio.cash:
                raise InsufficientFundsError(
                    f"Insufficient funds: need ${notional:.2f}, "
                    f"have ${self._portfolio.cash:.2f}"
                )
        else:  # SELL
            pos = self._portfolio.get_position(symbol)
            if pos is None:
                raise NoPositionError(f"No position in {symbol} to sell")
            if quantity > pos.quantity:
                raise InvalidOrderError(
                    f"Cannot sell {quantity} {symbol}, only have {pos.quantity}"
                )

        # Simulate execution
        execution = self._execution_sim.simulate_execution(
            side=side,
            quantity=quantity,
            market_price=current_price,
            symbol=symbol,
        )

        # Update portfolio
        if side == OrderSide.BUY:
            # Deduct cash (including fees)
            total_cost = float(quantity) * execution.fill_price + execution.fees
            self._portfolio.adjust_cash(-total_cost)
            self._portfolio.add_or_update_position(
                symbol=symbol,
                quantity_delta=quantity,
                price=execution.fill_price,
            )
        else:  # SELL
            # Add cash (minus fees)
            proceeds = float(quantity) * execution.fill_price - execution.fees
            self._portfolio.adjust_cash(proceeds)
            self._portfolio.add_or_update_position(
                symbol=symbol,
                quantity_delta=-quantity,
                price=execution.fill_price,
            )

        # Create order record
        order_id = self._generate_order_id()
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side.value,
            "order_type": order_type.value,
            "status": OrderStatus.FILLED.value,
            "limit_price": None,
            "created_at": execution.timestamp,
            "filled_at": execution.timestamp,
            "filled_qty": quantity,
            "avg_fill_price": execution.fill_price,
            "fees": execution.fees,
            "slippage_cost": execution.slippage_cost,
            "latency_ms": execution.latency_ms,
        }
        self._orders[order_id] = order

        # Log trade with portfolio snapshot
        prices = self._bar_buffer.get_all_prices()
        snapshot = self._portfolio.get_portfolio_snapshot(prices)

        self._trade_logger.log_trade(
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            execution=execution,
            portfolio_snapshot=snapshot,
            order_id=order_id,
        )

        logger.info(
            f"[DemoBroker] {side.value.upper()} {quantity} {symbol} @ ${execution.fill_price:.2f} "
            f"(fees: ${execution.fees:.4f}, slippage: ${execution.slippage_cost:.4f})"
        )

        return order

    def close_crypto_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None
    ) -> Dict:
        """
        Close a crypto position (or partial position).

        Args:
            symbol: Crypto symbol
            quantity: Amount to close (None = close all)

        Returns:
            Dict with order details
        """
        symbol = self.normalize_symbol(symbol)
        pos = self._portfolio.get_position(symbol)

        if pos is None:
            raise NoPositionError(f"No position in {symbol}")

        close_qty = quantity if quantity is not None else pos.quantity

        return self.place_crypto_order(
            symbol=symbol,
            quantity=Decimal(str(close_qty)),
            side=OrderSide.SELL,
        )

    def close_all_crypto_positions(self) -> List[Dict]:
        """Close all open crypto positions."""
        results = []
        positions = self._portfolio.get_all_positions()

        for pos in positions:
            try:
                result = self.close_crypto_position(pos.symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"[DemoBroker] Error closing {pos.symbol}: {e}")

        return results

    def get_crypto_quote(self, symbol: str) -> Dict:
        """
        Get current quote for a crypto symbol.

        Uses streaming data if available, falls back to REST API.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')

        Returns:
            Dict with quote data (bid, ask, last, etc.)
        """
        symbol = self.normalize_symbol(symbol)
        bar = self._bar_buffer.get_latest(symbol)

        # Try streaming data first
        if bar is not None:
            # Simulate bid/ask spread (0.1%)
            spread_pct = 0.001
            mid = bar.close
            bid = mid * (1 - spread_pct / 2)
            ask = mid * (1 + spread_pct / 2)

            return {
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "last": bar.close,
                "volume_24h": bar.volume,
                "timestamp": bar.timestamp,
                "available": True,
                "source": "streaming",
            }

        # Fall back to REST API
        try:
            price = self._rest_provider.get_current_price(symbol)
            if price is not None:
                spread_pct = 0.001
                bid = price * (1 - spread_pct / 2)
                ask = price * (1 + spread_pct / 2)

                return {
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "last": price,
                    "volume_24h": 0.0,  # Not available from ticker endpoint
                    "timestamp": tz.now(),
                    "available": True,
                    "source": "rest",
                }
        except Exception as e:
            logger.debug(f"[DemoBroker] REST quote failed for {symbol}: {e}")

        # No data available
        return {
            "symbol": symbol,
            "bid": 0.0,
            "ask": 0.0,
            "last": 0.0,
            "volume_24h": 0.0,
            "timestamp": tz.now(),
            "available": False,
            "source": None,
        }

    def get_crypto_balance(self, currency: str = 'USD') -> Dict:
        """
        Get crypto account balance.

        Args:
            currency: Base currency (default 'USD')

        Returns:
            Dict with balance data
        """
        prices = self._bar_buffer.get_all_prices()
        total_value = self._portfolio.get_total_value(prices)

        return {
            "total": total_value,
            "available": self._portfolio.cash,
            "held": total_value - self._portfolio.cash,
            "currency": currency,
        }

    # ==================== OrderManagementInterface ====================

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order (no-op for market orders)."""
        # Market orders fill immediately, nothing to cancel
        return False

    def get_order(self, order_id: str) -> Dict:
        """Get order details by ID."""
        if order_id not in self._orders:
            raise InvalidOrderError(f"Order {order_id} not found")
        return self._orders[order_id]

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Get orders with optional filters."""
        orders = list(self._orders.values())

        if status:
            orders = [o for o in orders if o["status"] == status.value]

        if start_date:
            orders = [o for o in orders if o["created_at"] >= start_date]

        if end_date:
            orders = [o for o in orders if o["created_at"] <= end_date]

        return orders

    # ==================== Additional Methods ====================

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"DEMO-{datetime.now().strftime('%Y%m%d')}-{self._order_counter:06d}"

    def _get_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get current prices from streaming buffer or REST fallback.

        Args:
            symbols: Symbols to get prices for (defaults to all positions)

        Returns:
            Dict mapping symbol -> price
        """
        # First try streaming prices
        prices = self._bar_buffer.get_all_prices()

        # Get symbols we need prices for
        if symbols is None:
            symbols = list(self._portfolio._positions.keys())

        # Check which symbols are missing
        missing = [s for s in symbols if s not in prices or prices.get(s, 0) == 0]

        # Fetch missing from REST
        if missing:
            try:
                rest_prices = self._rest_provider.get_current_prices(missing)
                prices.update(rest_prices)
            except Exception as e:
                logger.debug(f"[DemoBroker] REST price fetch failed: {e}")

        return prices

    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        prices = self._get_prices()
        return self._portfolio.get_total_value(prices)

    def get_portfolio_snapshot(self) -> Dict:
        """Get complete portfolio snapshot."""
        prices = self._get_prices()
        return self._portfolio.get_portfolio_snapshot(prices)

    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary."""
        prices = self._get_prices()
        snapshot = self._portfolio.get_portfolio_snapshot(prices)

        return {
            **snapshot,
            "streaming": self.is_streaming(),
            "symbols_streaming": self._symbols,
            "bars_buffered": self._bar_buffer.total_bars(),
            "orders_placed": len(self._orders),
            "execution_config": self._execution_sim.get_config(),
        }

    def reset_portfolio(self, initial_cash: float = 10000.0) -> None:
        """Reset portfolio to initial state."""
        self._portfolio.reset(initial_cash=initial_cash)
        self._orders.clear()
        self._order_counter = 0
        logger.info(f"[DemoBroker] Reset portfolio with ${initial_cash:.2f}")

    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self._portfolio.cash

    @property
    def realized_pnl(self) -> float:
        """Get realized P&L."""
        return self._portfolio.realized_pnl

    @property
    def symbols(self) -> List[str]:
        """Get list of streaming symbols."""
        return self._symbols.copy()
