"""
Binance WebSocket streaming for real-time crypto klines.

Provides real-time 1-minute kline data for multiple symbols via
Binance's combined stream WebSocket endpoint.

Reference: https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams
"""

import json
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BinanceStreamManager:
    """
    Binance WebSocket manager for real-time klines.

    Connects to Binance's combined stream endpoint and delivers
    1-minute klines for multiple symbols simultaneously.

    Features:
    - Multi-symbol subscription via combined streams
    - Automatic reconnection with exponential backoff
    - Thread-safe operation
    - Graceful shutdown
    - Connection statistics tracking

    Example:
        manager = BinanceStreamManager(['BTC/USD', 'ETH/USD', 'MKR/USD'])
        manager.subscribe_klines(lambda bar: print(f"New bar: {bar}"))
        manager.start()
        # ... later ...
        manager.stop()
    """

    WS_BASE_URL = "wss://stream.binance.com:9443/stream"

    # Reconnection settings
    INITIAL_RECONNECT_DELAY = 1.0  # seconds
    MAX_RECONNECT_DELAY = 60.0  # seconds
    RECONNECT_BACKOFF_FACTOR = 2.0

    def __init__(self, symbols: List[str]):
        """
        Initialize Binance stream manager.

        Args:
            symbols: List of symbols in 'BTC/USD' format.
                    Automatically converted to Binance format (BTCUSDT).
        """
        self._symbols = symbols
        self._binance_symbols = [self._to_binance_symbol(s) for s in symbols]
        self._symbol_map = {
            self._to_binance_symbol(s): s for s in symbols
        }  # BTCUSDT -> BTC/USD

        # WebSocket state
        self._ws = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._lock = threading.Lock()

        # Callbacks
        self._kline_handlers: List[Callable] = []

        # Statistics
        self._stats = {
            "messages_received": 0,
            "bars_processed": 0,
            "reconnect_count": 0,
            "connected_since": None,
            "last_message_time": None,
        }

        # Reconnection state
        self._reconnect_delay = self.INITIAL_RECONNECT_DELAY

    @staticmethod
    def _to_binance_symbol(symbol: str) -> str:
        """
        Convert symbol from our format to Binance format.

        BTC/USD -> btcusdt
        ETH/USD -> ethusdt
        """
        # Remove /USD suffix and add usdt, lowercase
        base = symbol.replace("/USD", "").replace("/", "")
        return f"{base.lower()}usdt"

    @staticmethod
    def _from_binance_symbol(binance_symbol: str) -> str:
        """
        Convert symbol from Binance format to our format.

        BTCUSDT -> BTC/USD
        btcusdt -> BTC/USD
        """
        upper = binance_symbol.upper()
        if upper.endswith("USDT"):
            base = upper[:-4]
            return f"{base}/USD"
        return binance_symbol

    def _build_stream_url(self) -> str:
        """Build the combined stream URL for all symbols."""
        # Format: btcusdt@kline_1m/ethusdt@kline_1m/...
        streams = [f"{s}@kline_1m" for s in self._binance_symbols]
        stream_param = "/".join(streams)
        return f"{self.WS_BASE_URL}?streams={stream_param}"

    def subscribe_klines(self, handler: Callable) -> None:
        """
        Subscribe to kline (bar) updates.

        Args:
            handler: Callback function that receives Bar objects.
                    Signature: handler(bar: Bar) -> None
        """
        from src.streaming.types import Bar

        self._kline_handlers.append(handler)
        logger.info(f"[BinanceStream] Registered kline handler: {handler.__name__}")

    def start(self) -> None:
        """
        Start WebSocket connection in background thread.

        The connection runs as a daemon thread, allowing the main
        program to exit cleanly.
        """
        if self._running:
            logger.warning("[BinanceStream] Already running")
            return

        self._running = True
        self._ws_thread = threading.Thread(
            target=self._run_websocket, name="BinanceStreamThread", daemon=True
        )
        self._ws_thread.start()
        logger.info(
            f"[BinanceStream] Started streaming for {len(self._symbols)} symbols"
        )

    def stop(self) -> None:
        """
        Stop WebSocket connection gracefully.

        Waits for the background thread to terminate.
        """
        if not self._running:
            return

        logger.info("[BinanceStream] Stopping...")
        self._running = False

        # Close WebSocket if connected
        if self._ws:
            try:
                self._ws.close()
            except Exception as e:
                logger.warning(f"[BinanceStream] Error closing WebSocket: {e}")

        # Wait for thread to finish
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)

        self._connected = False
        logger.info("[BinanceStream] Stopped")

    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        with self._lock:
            return self._connected

    def get_stats(self) -> Dict:
        """Get connection statistics."""
        with self._lock:
            stats = self._stats.copy()
            if stats["connected_since"]:
                stats["uptime_seconds"] = (
                    datetime.now() - stats["connected_since"]
                ).total_seconds()
            else:
                stats["uptime_seconds"] = 0
            return stats

    def _run_websocket(self) -> None:
        """Main WebSocket loop with reconnection logic."""
        try:
            import websocket
        except ImportError:
            logger.error(
                "[BinanceStream] websocket-client not installed. "
                "Run: pip install websocket-client"
            )
            self._running = False
            return

        while self._running:
            try:
                url = self._build_stream_url()
                logger.info(
                    f"[BinanceStream] Connecting to {url[:50]}... "
                    f"({len(self._symbols)} symbols)"
                )

                self._ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )

                # Run WebSocket (blocking)
                self._ws.run_forever(ping_interval=30, ping_timeout=10)

            except Exception as e:
                logger.error(f"[BinanceStream] WebSocket error: {e}")

            # Handle reconnection
            if self._running:
                self._connected = False
                logger.info(
                    f"[BinanceStream] Reconnecting in {self._reconnect_delay:.1f}s..."
                )
                time.sleep(self._reconnect_delay)

                # Exponential backoff
                self._reconnect_delay = min(
                    self._reconnect_delay * self.RECONNECT_BACKOFF_FACTOR,
                    self.MAX_RECONNECT_DELAY,
                )
                with self._lock:
                    self._stats["reconnect_count"] += 1

    def _on_open(self, ws) -> None:
        """Handle WebSocket connection opened."""
        with self._lock:
            self._connected = True
            self._stats["connected_since"] = datetime.now()
            self._reconnect_delay = self.INITIAL_RECONNECT_DELAY  # Reset backoff

        logger.info(
            f"[BinanceStream] Connected! Streaming {len(self._symbols)} symbols: "
            f"{', '.join(self._symbols[:5])}{'...' if len(self._symbols) > 5 else ''}"
        )

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Handle WebSocket connection closed."""
        with self._lock:
            self._connected = False

        if close_status_code:
            logger.warning(
                f"[BinanceStream] Connection closed: {close_status_code} - {close_msg}"
            )
        else:
            logger.info("[BinanceStream] Connection closed")

    def _on_error(self, ws, error) -> None:
        """Handle WebSocket error."""
        logger.error(f"[BinanceStream] WebSocket error: {error}")

    def _on_message(self, ws, message: str) -> None:
        """Handle incoming WebSocket message."""
        from src.streaming.types import Bar

        with self._lock:
            self._stats["messages_received"] += 1
            self._stats["last_message_time"] = datetime.now()

        try:
            data = json.loads(message)

            # Combined stream format: {"stream": "btcusdt@kline_1m", "data": {...}}
            if "stream" in data and "data" in data:
                stream_name = data["stream"]
                kline_data = data["data"]

                # Only process kline messages
                if "@kline_" in stream_name and kline_data.get("e") == "kline":
                    self._process_kline(kline_data)

        except json.JSONDecodeError as e:
            logger.warning(f"[BinanceStream] Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"[BinanceStream] Error processing message: {e}")

    def _process_kline(self, kline_data: dict) -> None:
        """Process a kline message and notify handlers."""
        from src.streaming.types import Bar

        # Get our symbol format from the map
        binance_symbol = kline_data.get("s", "").upper()
        our_symbol = self._symbol_map.get(binance_symbol.lower() + "usdt")

        if not our_symbol:
            # Try without extra usdt (symbol already in proper format)
            our_symbol = self._symbol_map.get(binance_symbol.lower())

        if not our_symbol:
            our_symbol = self._from_binance_symbol(binance_symbol)

        # Convert to Bar
        bar = Bar.from_binance(kline_data, symbol_override=our_symbol)

        with self._lock:
            self._stats["bars_processed"] += 1

        # Notify handlers
        for handler in self._kline_handlers:
            try:
                handler(bar)
            except Exception as e:
                logger.error(f"[BinanceStream] Handler error: {e}")

    @property
    def symbols(self) -> List[str]:
        """Get list of subscribed symbols in our format."""
        return self._symbols.copy()
