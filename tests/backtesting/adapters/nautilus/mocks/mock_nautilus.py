"""
Mock NautilusTrader components for testing.

These mocks mimic the essential interfaces of NautilusTrader components
without requiring the actual library to be installed.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class MockBar:
    """
    Mock NautilusTrader Bar object.

    Mimics the essential properties used by the adapter layer.
    """
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    ts_event: int  # Nanoseconds since epoch

    @classmethod
    def from_ohlcv(
        cls,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        timestamp: datetime
    ) -> "MockBar":
        """Create MockBar from OHLCV data."""
        # Convert datetime to nanoseconds
        ts_ns = int(timestamp.timestamp() * 1_000_000_000)
        return cls(
            symbol=symbol,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            ts_event=ts_ns,
        )


@dataclass
class MockInstrument:
    """Mock NautilusTrader Instrument object."""
    instrument_id: str
    symbol: str
    venue: str = "XNAS"
    currency: str = "USD"
    price_precision: int = 2
    size_precision: int = 0

    @classmethod
    def from_symbol(cls, symbol: str, venue: str = "XNAS") -> "MockInstrument":
        """Create MockInstrument from symbol."""
        return cls(
            instrument_id=f"{symbol}.{venue}",
            symbol=symbol,
            venue=venue,
        )


@dataclass
class MockOrder:
    """Mock NautilusTrader Order object."""
    client_order_id: str
    instrument_id: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: str = "MARKET"
    time_in_force: str = "GTC"
    status: str = "SUBMITTED"

    @classmethod
    def create(
        cls,
        instrument_id: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET"
    ) -> "MockOrder":
        """Create a new mock order."""
        return cls(
            client_order_id=f"ORDER-{uuid.uuid4().hex[:8].upper()}",
            instrument_id=instrument_id,
            side=side,
            quantity=quantity,
            order_type=order_type,
        )


@dataclass
class MockPosition:
    """Mock NautilusTrader Position object."""
    instrument_id: str
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0


class MockCache:
    """
    Mock NautilusTrader Cache object.

    Provides instrument and position lookup.
    """

    def __init__(self) -> None:
        self._instruments: Dict[str, MockInstrument] = {}
        self._positions: Dict[str, MockPosition] = {}
        self._orders: List[MockOrder] = []

    def add_instrument(self, instrument: MockInstrument) -> None:
        """Add an instrument to the cache."""
        self._instruments[instrument.instrument_id] = instrument

    def instrument(self, instrument_id: Any) -> Optional[MockInstrument]:
        """Get instrument by ID."""
        id_str = str(instrument_id)
        return self._instruments.get(id_str)

    def instruments(self) -> List[MockInstrument]:
        """Get all instruments."""
        return list(self._instruments.values())

    def add_position(self, position: MockPosition) -> None:
        """Add a position to the cache."""
        self._positions[position.instrument_id] = position

    def positions(self) -> List[MockPosition]:
        """Get all positions."""
        return list(self._positions.values())

    def position(self, instrument_id: str) -> Optional[MockPosition]:
        """Get position by instrument ID."""
        return self._positions.get(instrument_id)

    def add_order(self, order: MockOrder) -> None:
        """Add an order to the cache."""
        self._orders.append(order)

    def orders(self) -> List[MockOrder]:
        """Get all orders."""
        return self._orders.copy()


@dataclass
class MockAccount:
    """Mock trading account."""
    balance: float = 100000.0
    equity: float = 100000.0
    margin_available: float = 100000.0
    currency: str = "USD"


class MockPortfolio:
    """
    Mock NautilusTrader Portfolio object.

    Provides account and balance information.
    """

    def __init__(self, initial_balance: float = 100000.0) -> None:
        self._account = MockAccount(
            balance=initial_balance,
            equity=initial_balance,
            margin_available=initial_balance,
        )

    def account(self) -> MockAccount:
        """Get the account."""
        return self._account

    def balances(self) -> Dict[str, float]:
        """Get account balances."""
        return {
            self._account.currency: self._account.balance
        }

    def update_balance(self, amount: float) -> None:
        """Update account balance."""
        self._account.balance += amount
        self._account.equity += amount
        self._account.margin_available += amount


class MockClock:
    """Mock NautilusTrader Clock object."""

    def __init__(self, start_time: Optional[datetime] = None) -> None:
        self._current_time = start_time or datetime.now()

    def timestamp_ns(self) -> int:
        """Get current time in nanoseconds."""
        return int(self._current_time.timestamp() * 1_000_000_000)

    def utc_now(self) -> datetime:
        """Get current UTC time."""
        return self._current_time

    def advance(self, seconds: float) -> None:
        """Advance the clock by seconds."""
        from datetime import timedelta
        self._current_time += timedelta(seconds=seconds)


class MockStrategy:
    """
    Mock NautilusTrader Strategy base class.

    Provides the essential interface for testing adapters.
    """

    def __init__(self, strategy_id: str = "TEST") -> None:
        self.id = strategy_id
        self.trader_id = "TRADER-001"
        self.cache = MockCache()
        self.portfolio = MockPortfolio()
        self.clock = MockClock()

        self._submitted_orders: List[MockOrder] = []
        self._order_counter = 0

    def generate_order_id(self) -> str:
        """Generate a unique order ID."""
        self._order_counter += 1
        return f"O-{self.id}-{self._order_counter:06d}"

    def generate_uuid(self) -> str:
        """Generate a UUID."""
        return uuid.uuid4().hex

    def submit_order(self, order: MockOrder) -> None:
        """Submit an order."""
        self._submitted_orders.append(order)
        self.cache.add_order(order)

    def get_submitted_orders(self) -> List[MockOrder]:
        """Get all submitted orders."""
        return self._submitted_orders.copy()

    def subscribe_bars(self, instrument_id: Any) -> None:
        """Subscribe to bar data (no-op in mock)."""
        pass

    def on_start(self) -> None:
        """Called when strategy starts."""
        pass

    def on_bar(self, bar: MockBar) -> None:
        """Called when a bar is received."""
        pass

    def on_stop(self) -> None:
        """Called when strategy stops."""
        pass

    def on_reset(self) -> None:
        """Called when strategy resets."""
        pass

    def on_dispose(self) -> None:
        """Called when strategy is disposed."""
        pass
