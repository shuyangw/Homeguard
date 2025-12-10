"""
Streaming data types for real-time market data.

These dataclasses match the Alpaca SDK model schemas and provide
conversion methods for seamless integration.

Reference: https://alpaca.markets/sdks/python/api_reference/data/models.html
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Any


@dataclass
class Bar:
    """
    OHLCV bar from streaming or historical data.

    Matches alpaca.data.models.bars.Bar schema.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: Optional[float] = None
    vwap: Optional[float] = None
    exchange: Optional[str] = None

    @classmethod
    def from_alpaca(cls, alpaca_bar: Any) -> "Bar":
        """Convert Alpaca Bar model to our Bar dataclass."""
        return cls(
            symbol=alpaca_bar.symbol,
            timestamp=alpaca_bar.timestamp,
            open=alpaca_bar.open,
            high=alpaca_bar.high,
            low=alpaca_bar.low,
            close=alpaca_bar.close,
            volume=alpaca_bar.volume,
            trade_count=getattr(alpaca_bar, "trade_count", None),
            vwap=getattr(alpaca_bar, "vwap", None),
            exchange=getattr(alpaca_bar, "exchange", None),
        )


@dataclass
class Quote:
    """
    Bid/ask quote from streaming or historical data.

    Matches alpaca.data.models.quotes.Quote schema.
    """

    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    bid_exchange: Optional[str] = None
    ask_exchange: Optional[str] = None
    conditions: Optional[List[str]] = None
    tape: Optional[str] = None

    @property
    def mid(self) -> float:
        """Mid price between bid and ask."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask_price - self.bid_price

    @classmethod
    def from_alpaca(cls, alpaca_quote: Any) -> "Quote":
        """Convert Alpaca Quote model to our Quote dataclass."""
        return cls(
            symbol=alpaca_quote.symbol,
            timestamp=alpaca_quote.timestamp,
            bid_price=alpaca_quote.bid_price,
            bid_size=alpaca_quote.bid_size,
            ask_price=alpaca_quote.ask_price,
            ask_size=alpaca_quote.ask_size,
            bid_exchange=getattr(alpaca_quote, "bid_exchange", None),
            ask_exchange=getattr(alpaca_quote, "ask_exchange", None),
            conditions=getattr(alpaca_quote, "conditions", None),
            tape=getattr(alpaca_quote, "tape", None),
        )


@dataclass
class Trade:
    """
    Single trade tick from streaming or historical data.

    Matches alpaca.data.models.trades.Trade schema.
    """

    symbol: str
    timestamp: datetime
    price: float
    size: float
    exchange: Optional[str] = None
    id: Optional[int] = None
    conditions: Optional[List[str]] = None
    tape: Optional[str] = None

    @classmethod
    def from_alpaca(cls, alpaca_trade: Any) -> "Trade":
        """Convert Alpaca Trade model to our Trade dataclass."""
        return cls(
            symbol=alpaca_trade.symbol,
            timestamp=alpaca_trade.timestamp,
            price=alpaca_trade.price,
            size=alpaca_trade.size,
            exchange=getattr(alpaca_trade, "exchange", None),
            id=getattr(alpaca_trade, "id", None),
            conditions=getattr(alpaca_trade, "conditions", None),
            tape=getattr(alpaca_trade, "tape", None),
        )
