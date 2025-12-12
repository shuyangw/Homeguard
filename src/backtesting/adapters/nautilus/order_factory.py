"""
Order Factory for NautilusTrader Adapters.

Converts Homeguard signals to NautilusTrader orders.

This module provides two modes:
1. Mock mode: Works without NautilusTrader for testing
2. Real mode: Creates actual NautilusTrader order objects when available
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Protocol, TYPE_CHECKING

import pandas as pd

from src.strategies.core.signal import Signal

# Check if NautilusTrader is available
try:
    from nautilus_trader.trading.strategy import Strategy as NautilusStrategy
    from nautilus_trader.model.orders import MarketOrder
    from nautilus_trader.model.enums import OrderSide, TimeInForce
    from nautilus_trader.model.identifiers import InstrumentId, ClientOrderId
    from nautilus_trader.model.objects import Quantity
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False


class StrategyProtocol(Protocol):
    """Protocol for NautilusTrader-like strategy (works with mocks)."""
    def generate_order_id(self) -> Any: ...


@dataclass
class MockOrder:
    """
    Mock order for testing without NautilusTrader.

    Mimics the essential properties of a NautilusTrader order.
    """
    instrument_id: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    client_order_id: str
    time_in_force: str = 'GTC'

    def __str__(self) -> str:
        return f"MockOrder({self.side} {self.quantity} {self.instrument_id})"


class HomeguardOrderFactory:
    """
    Factory for converting Homeguard signals to NautilusTrader orders.

    Works in two modes:
    1. Mock mode (default when NautilusTrader not installed)
    2. Real mode (when NautilusTrader is available and strategy is provided)

    Attributes:
        strategy: NautilusTrader strategy instance (optional)
        default_quantity: Default order quantity when not calculable
        venue: Trading venue identifier (e.g., "XNAS")
    """

    def __init__(
        self,
        strategy: Optional[Any] = None,
        default_quantity: float = 100,
        venue: str = "XNAS"
    ) -> None:
        """
        Initialize order factory.

        Args:
            strategy: NautilusTrader strategy instance (None for mock mode)
            default_quantity: Default quantity for orders
            venue: Trading venue identifier
        """
        self.strategy = strategy
        self.default_quantity = default_quantity
        self.venue = venue
        self._order_counter = 0

    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        self._order_counter += 1
        return f"HG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._order_counter:04d}"

    def signal_to_order(
        self,
        signal: Signal,
        instrument_id: Optional[Any] = None,
        quantity: Optional[float] = None
    ) -> Optional[Any]:
        """
        Convert a Homeguard Signal to an order.

        Args:
            signal: Homeguard Signal object
            instrument_id: NautilusTrader InstrumentId (or string for mock)
            quantity: Order quantity (uses default if not provided)

        Returns:
            NautilusTrader MarketOrder if available, MockOrder otherwise.
            Returns None if signal is HOLD.
        """
        # HOLD signals don't generate orders
        if signal.direction == 'HOLD':
            return None

        qty = quantity if quantity is not None else self.default_quantity

        # Use real NautilusTrader orders if available
        if NAUTILUS_AVAILABLE and self.strategy is not None:
            return self._create_nautilus_order(signal, instrument_id, qty)

        # Fall back to mock orders
        return self._create_mock_order(signal, instrument_id, qty)

    def _create_nautilus_order(
        self,
        signal: Signal,
        instrument_id: Any,
        quantity: float
    ) -> Any:
        """Create a real NautilusTrader order."""
        if instrument_id is None:
            # Create InstrumentId from signal symbol
            instrument_id = InstrumentId.from_str(f"{signal.symbol}.{self.venue}")

        side = OrderSide.BUY if signal.direction == 'BUY' else OrderSide.SELL

        order = MarketOrder(
            trader_id=self.strategy.trader_id,
            strategy_id=self.strategy.id,
            instrument_id=instrument_id,
            client_order_id=self.strategy.generate_order_id(),
            order_side=side,
            quantity=Quantity.from_str(str(int(quantity))),
            time_in_force=TimeInForce.GTC,
            init_id=self.strategy.generate_uuid(),
            ts_init=self.strategy.clock.timestamp_ns(),
        )
        return order

    def _create_mock_order(
        self,
        signal: Signal,
        instrument_id: Optional[str],
        quantity: float
    ) -> MockOrder:
        """Create a mock order for testing."""
        if instrument_id is None:
            instrument_id = f"{signal.symbol}.{self.venue}"
        elif hasattr(instrument_id, 'value'):
            # Handle InstrumentId-like objects
            instrument_id = str(instrument_id)

        return MockOrder(
            instrument_id=str(instrument_id),
            side=signal.direction,
            quantity=quantity,
            client_order_id=self._generate_order_id(),
        )

    def boolean_series_to_orders(
        self,
        symbol: str,
        entries: pd.Series,
        exits: pd.Series,
        prices: pd.Series,
        bar_idx: int,
        quantity: Optional[float] = None
    ) -> List[Any]:
        """
        Convert boolean entry/exit Series to orders at a specific bar index.

        Used for BaseStrategy-style strategies that return boolean Series
        instead of Signal objects.

        Args:
            symbol: Stock symbol
            entries: Boolean Series where True = entry signal
            exits: Boolean Series where True = exit signal
            prices: Price Series for order pricing
            bar_idx: Current bar index to check
            quantity: Order quantity (uses default if not provided)

        Returns:
            List of orders (may be empty, or contain 1-2 orders)
        """
        orders = []
        qty = quantity if quantity is not None else self.default_quantity

        # Check bounds
        if bar_idx < 0 or bar_idx >= len(entries):
            return orders

        timestamp = entries.index[bar_idx] if hasattr(entries, 'index') else datetime.now()
        price = float(prices.iloc[bar_idx]) if bar_idx < len(prices) else 0.0

        # Check for entry signal (BUY)
        if entries.iloc[bar_idx]:
            signal = Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction='BUY',
                confidence=1.0,
                price=price,
            )
            order = self.signal_to_order(signal, quantity=qty)
            if order is not None:
                orders.append(order)

        # Check for exit signal (SELL)
        if exits.iloc[bar_idx]:
            signal = Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction='SELL',
                confidence=1.0,
                price=price,
            )
            order = self.signal_to_order(signal, quantity=qty)
            if order is not None:
                orders.append(order)

        return orders

    def signals_to_orders(
        self,
        signals: List[Signal],
        quantities: Optional[dict] = None
    ) -> List[Any]:
        """
        Convert multiple signals to orders.

        Args:
            signals: List of Homeguard Signal objects
            quantities: Optional dict mapping symbol -> quantity

        Returns:
            List of orders (HOLD signals are filtered out)
        """
        orders = []
        for signal in signals:
            qty = None
            if quantities and signal.symbol in quantities:
                qty = quantities[signal.symbol]

            order = self.signal_to_order(signal, quantity=qty)
            if order is not None:
                orders.append(order)

        return orders
