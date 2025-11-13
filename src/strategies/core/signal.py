"""
Signal data structure - pure data representation.

Used by all strategies to represent trading signals.
No dependencies on backtesting or live trading infrastructure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class Signal:
    """
    Pure signal data structure.

    Represents a trading signal with all necessary information
    for execution. Can be used by both backtesting and live trading
    through adapters.

    Attributes:
        timestamp: When signal was generated
        symbol: Trading symbol (e.g., 'AAPL', 'TQQQ')
        direction: Signal direction ('BUY', 'SELL', 'HOLD')
        confidence: Signal confidence (0.0 to 1.0)
        price: Signal price (close price when signal generated)
        metadata: Strategy-specific data (indicators, reasoning, etc.)
    """
    timestamp: datetime
    symbol: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    price: float  # Signal price
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal data."""
        if self.direction not in ['BUY', 'SELL', 'HOLD']:
            raise ValueError(f"Invalid direction: {self.direction}. Must be 'BUY', 'SELL', or 'HOLD'")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}. Must be between 0.0 and 1.0")

        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}. Must be positive")

    def __str__(self):
        """String representation."""
        return (f"Signal({self.symbol} {self.direction} @ ${self.price:.2f}, "
                f"confidence={self.confidence:.1%}, time={self.timestamp})")

    def __repr__(self):
        """Detailed representation."""
        return (f"Signal(timestamp={self.timestamp}, symbol={self.symbol}, "
                f"direction={self.direction}, confidence={self.confidence:.2f}, "
                f"price={self.price:.2f}, metadata={self.metadata})")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'confidence': self.confidence,
            'price': self.price,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Signal':
        """Create Signal from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=data['symbol'],
            direction=data['direction'],
            confidence=data['confidence'],
            price=data['price'],
            metadata=data.get('metadata', {})
        )


@dataclass
class SignalBatch:
    """
    Collection of signals generated at the same time.

    Useful for strategies that generate multiple signals simultaneously.
    """
    timestamp: datetime
    signals: list[Signal] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_signal(self, signal: Signal):
        """Add a signal to the batch."""
        if signal.timestamp != self.timestamp:
            raise ValueError(f"Signal timestamp {signal.timestamp} doesn't match batch timestamp {self.timestamp}")
        self.signals.append(signal)

    def get_buy_signals(self) -> list[Signal]:
        """Get only BUY signals."""
        return [s for s in self.signals if s.direction == 'BUY']

    def get_sell_signals(self) -> list[Signal]:
        """Get only SELL signals."""
        return [s for s in self.signals if s.direction == 'SELL']

    def get_signals_for_symbol(self, symbol: str) -> list[Signal]:
        """Get signals for a specific symbol."""
        return [s for s in self.signals if s.symbol == symbol]

    def __len__(self):
        """Number of signals in batch."""
        return len(self.signals)

    def __iter__(self):
        """Iterate over signals."""
        return iter(self.signals)

    def __str__(self):
        """String representation."""
        buy_count = len(self.get_buy_signals())
        sell_count = len(self.get_sell_signals())
        return f"SignalBatch({len(self.signals)} signals: {buy_count} BUY, {sell_count} SELL, time={self.timestamp})"
