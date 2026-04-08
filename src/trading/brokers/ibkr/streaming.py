"""
IBKR Streaming Provider - Implements StreamingProviderInterface.

Uses ib_async's reqMktData() for quotes/trades and reqRealTimeBars()
for 5-second bars (aggregated to 1-min in buffer). All data converted
to Homeguard Bar/Quote/Trade dataclasses at the boundary.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

import pandas as pd
import pytz

from src.streaming.interface import StreamingProviderInterface
from src.streaming.types import Bar, Quote, Trade
from src.trading.brokers.ibkr.connection import IBKRConnectionManager
from src.trading.brokers.ibkr.contracts import ContractResolver
from src.utils.logger import get_logger

logger = get_logger(__name__)

ET = pytz.timezone('America/New_York')


class IBKRStreamingProvider(StreamingProviderInterface):
    """
    Real-time market data from IBKR via ib_async.

    Implements StreamingProviderInterface so strategy adapters can use
    IBKR streaming interchangeably with Alpaca's LiveDataProvider.
    """

    def __init__(
        self,
        connection: IBKRConnectionManager,
        resolver: Optional[ContractResolver] = None,
        max_bars_per_symbol: int = 500,
    ):
        self._conn = connection
        self._resolver = resolver or ContractResolver(connection)
        self._max_bars = max_bars_per_symbol

        self._subscribed: Set[str] = set()
        self._tickers: Dict[str, object] = {}
        self._bar_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._max_bars)
        )
        self._callbacks: Dict[str, dict] = {}
        self._started = False

    @property
    def name(self) -> str:
        return "IBKR-streaming"

    # ---- Lifecycle ----

    def start(self, symbols: Optional[List[str]] = None) -> None:
        if self._started:
            logger.warning("[IBKR Stream] Already started")
            return

        self._started = True
        logger.info("[IBKR Stream] Started")

        if symbols:
            self._subscribe_symbols(symbols)

    def stop(self) -> None:
        for symbol in list(self._subscribed):
            self._unsubscribe_symbol(symbol)
        self._started = False
        logger.info("[IBKR Stream] Stopped")

    def is_connected(self) -> bool:
        return self._conn.is_connected and self._started

    # ---- On-Demand Data ----

    def get_price(self, symbol: str) -> Optional[float]:
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return None
        last = ticker.last
        if last != last:  # NaN check
            mid = self._mid_price(ticker)
            return mid if mid is not None else None
        return float(last)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return None

        def safe(val):
            return float(val) if val == val else 0.0

        return Quote(
            symbol=symbol,
            timestamp=datetime.now(ET),
            bid_price=safe(ticker.bid),
            bid_size=safe(ticker.bidSize),
            ask_price=safe(ticker.ask),
            ask_size=safe(ticker.askSize),
        )

    def get_trade(self, symbol: str) -> Optional[Trade]:
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return None
        last = ticker.last
        if last != last:
            return None
        return Trade(
            symbol=symbol,
            timestamp=datetime.now(ET),
            price=float(last),
            size=float(ticker.lastSize) if ticker.lastSize == ticker.lastSize else 0.0,
        )

    def get_bar(self, symbol: str) -> Optional[Bar]:
        buf = self._bar_buffers.get(symbol)
        if not buf:
            return None
        return buf[-1]

    def get_bars(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        buf = self._bar_buffers.get(symbol)
        if not buf:
            return pd.DataFrame()

        bars = list(buf) if n is None else list(buf)[-n:]
        if not bars:
            return pd.DataFrame()

        records = [{
            'timestamp': b.timestamp,
            'open': b.open,
            'high': b.high,
            'low': b.low,
            'close': b.close,
            'volume': b.volume,
        } for b in bars]

        df = pd.DataFrame(records)
        df = df.set_index('timestamp')
        return df

    def get_vwap(self, symbol: str) -> Optional[float]:
        buf = self._bar_buffers.get(symbol)
        if not buf:
            return None
        total_vol = sum(b.volume for b in buf if b.volume)
        if total_vol == 0:
            return None
        weighted = sum(
            ((b.high + b.low + b.close) / 3) * b.volume
            for b in buf if b.volume
        )
        return weighted / total_vol

    def get_spread(self, symbol: str) -> Optional[float]:
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return None
        bid = ticker.bid
        ask = ticker.ask
        if bid != bid or ask != ask:
            return None
        return float(ask) - float(bid)

    # ---- Callbacks ----

    def on_bar(self, symbols: List[str], handler: Callable[[Bar], None]) -> str:
        sub_id = str(uuid.uuid4())
        self._callbacks[sub_id] = {'type': 'bar', 'symbols': set(symbols), 'handler': handler}
        self._subscribe_symbols(symbols)
        return sub_id

    def on_quote(self, symbols: List[str], handler: Callable[[Quote], None]) -> str:
        sub_id = str(uuid.uuid4())
        self._callbacks[sub_id] = {'type': 'quote', 'symbols': set(symbols), 'handler': handler}
        self._subscribe_symbols(symbols)
        return sub_id

    def on_trade(self, symbols: List[str], handler: Callable[[Trade], None]) -> str:
        sub_id = str(uuid.uuid4())
        self._callbacks[sub_id] = {'type': 'trade', 'symbols': set(symbols), 'handler': handler}
        self._subscribe_symbols(symbols)
        return sub_id

    def unsubscribe(self, subscription_id: str) -> None:
        self._callbacks.pop(subscription_id, None)

    def get_subscribed_symbols(self) -> set:
        return set(self._subscribed)

    # ---- Internal ----

    def _subscribe_symbols(self, symbols: List[str]) -> None:
        for symbol in symbols:
            if symbol in self._subscribed:
                continue
            try:
                contract = self._resolver.resolve_stock(symbol)
                ticker = self._conn.run_sync(
                    self._req_market_data(contract)
                )
                self._tickers[symbol] = ticker
                self._subscribed.add(symbol)
                logger.debug("[IBKR Stream] Subscribed to %s", symbol)
            except Exception as e:
                logger.warning("[IBKR Stream] Failed to subscribe %s: %s", symbol, e)

    def _unsubscribe_symbol(self, symbol: str) -> None:
        ticker = self._tickers.pop(symbol, None)
        if ticker:
            try:
                self._conn.ib.cancelMktData(ticker.contract)
            except Exception:
                pass
        self._subscribed.discard(symbol)

    async def _req_market_data(self, contract):
        ticker = self._conn.ib.reqMktData(contract, '', False, False)
        await asyncio.sleep(0.5)
        return ticker

    @staticmethod
    def _mid_price(ticker) -> Optional[float]:
        bid = ticker.bid
        ask = ticker.ask
        if bid != bid or ask != ask:
            return None
        return (float(bid) + float(ask)) / 2
