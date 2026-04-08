"""
IBKR Connection Manager -- Singleton that owns the ib_async.IB instance.

The key challenge: ib_async is asyncio-native, but Homeguard strategies
are synchronous.  We solve this by running the ib_async event loop on a
dedicated daemon thread and bridging sync<->async via
asyncio.run_coroutine_threadsafe().

Handles:
    - Event loop lifecycle on background thread
    - Automatic reconnection with exponential backoff
    - Connection health monitoring
    - Thread-safe sync->async call bridge
    - Graceful shutdown
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, ClassVar, Coroutine, Optional

from ib_async import IB, Contract

from src.trading.brokers.interfaces.base import BrokerConnectionError
from src.trading.brokers.ibkr.config import IBKRConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IBKRConnectionManager:
    """
    Singleton managing a single ib_async.IB connection on a background
    event loop thread.

    Usage:
        mgr = IBKRConnectionManager(config)
        mgr.start()                         # connect + start loop thread
        result = mgr.run_sync(some_coro())   # bridge sync -> async
        mgr.stop()                           # graceful teardown

    Thread safety:
        All ib_async calls are funneled through the single event-loop
        thread via run_sync().  Do NOT access mgr.ib directly from
        arbitrary threads; always go through run_sync().
    """

    _instance: ClassVar[Optional["IBKRConnectionManager"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # -- Construction ------------------------------------------------------

    def __init__(self, config: IBKRConfig):
        self._config = config
        self._ib = IB()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._shutting_down = False
        self._reconnect_count = 0

        # Wire IBKR events
        self._ib.disconnectedEvent += self._on_disconnect
        self._ib.errorEvent += self._on_error

        # External callbacks (set by IBKRBroker, streaming, etc.)
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_error: Optional[Callable[[int, int, str], None]] = None

    @classmethod
    def get_instance(cls, config: Optional[IBKRConfig] = None) -> "IBKRConnectionManager":
        """Get or create the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                if config is None:
                    config = IBKRConfig()
                cls._instance = cls(config)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Tear down the singleton (for tests)."""
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance.stop()
                except Exception:
                    pass
                cls._instance = None

    # -- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """
        Spin up the background event loop thread and connect to Gateway/TWS.
        Blocks until the connection is established or timeout is reached.
        """
        if self._loop is not None and self._loop.is_running():
            logger.warning("[IBKR] Event loop already running")
            return

        self._shutting_down = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="ibkr-event-loop",
        )
        self._thread.start()

        # Connect (blocks until done or timeout).
        # Use 2x timeout because connectAsync performs order syncs that
        # may time out independently (especially in read-only mode).
        try:
            self.run_sync(self._connect(), timeout=self._config.timeout_seconds * 3)
        except Exception as e:
            self.stop()
            raise BrokerConnectionError(f"Failed to connect: {e}") from e

        logger.info(
            f"[IBKR] Connected to {self._config.gateway_type} "
            f"at {self._config.host}:{self._config.port} "
            f"(clientId={self._config.client_id})"
        )

    def stop(self) -> None:
        """Disconnect and shut down the event loop thread."""
        self._shutting_down = True
        self._connected.clear()

        if self._loop and self._loop.is_running():
            try:
                self.run_sync(self._disconnect(), timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        self._loop = None
        self._thread = None
        logger.info("[IBKR] Disconnected and event loop stopped")

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and self._ib.isConnected()

    @property
    def ib(self) -> IB:
        """
        Direct access to the underlying IB instance.
        ONLY for use within this package, and ONLY via run_sync().
        """
        return self._ib

    @property
    def config(self) -> IBKRConfig:
        return self._config

    # -- Sync <-> Async Bridge ---------------------------------------------

    def run_sync(self, coro: Coroutine, timeout: Optional[float] = None) -> Any:
        """
        Submit a coroutine to the background event loop and block
        the calling thread until it completes.

        This is THE bridge that every public method in the IBKR module
        uses.  It is thread-safe: multiple threads can call run_sync()
        concurrently and they will be serialized on the event loop.

        Raises:
            BrokerConnectionError: If event loop is not running.
            TimeoutError: If the coroutine doesn't complete in time.
            Exception: Whatever the coroutine raises.
        """
        if self._loop is None or not self._loop.is_running():
            raise BrokerConnectionError("IBKR event loop is not running. Call start() first.")

        t = timeout or self._config.timeout_seconds
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=t)
        except asyncio.TimeoutError:
            future.cancel()
            raise TimeoutError(f"IBKR request timed out after {t}s")

    # -- Internal: Event Loop Thread ---------------------------------------

    def _run_loop(self) -> None:
        """Target for the background thread.  Runs forever until stopped."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        """Async connection with market data type configuration."""
        await self._ib.connectAsync(
            host=self._config.host,
            port=self._config.port,
            clientId=self._config.client_id,
            timeout=self._config.timeout_seconds,
            readonly=self._config.readonly,
            account=self._config.account or "",
        )
        # Set market data type (3 = delayed for paper, 1 = live)
        self._ib.reqMarketDataType(self._config.market_data_type)
        self._connected.set()
        self._reconnect_count = 0

        if self.on_connected:
            self.on_connected()

    async def _disconnect(self) -> None:
        self._ib.disconnect()
        self._connected.clear()

    # -- Internal: Reconnection --------------------------------------------

    def _on_disconnect(self) -> None:
        """Called by ib_async when the socket disconnects."""
        self._connected.clear()
        logger.warning("[IBKR] Disconnected from Gateway/TWS")

        if self.on_disconnected:
            self.on_disconnected()

        if self._shutting_down:
            return

        # Schedule reconnection on the event loop
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._reconnect_loop(), self._loop)

    async def _reconnect_loop(self) -> None:
        """
        Exponential backoff reconnection.

        Schedule:  2s, 4s, 8s, 16s, 30s, 30s, 30s, ...
        Max attempts: config.max_reconnect_attempts
        """
        base = self._config.reconnect_base_delay
        cap = self._config.reconnect_max_delay
        max_attempts = self._config.max_reconnect_attempts

        for attempt in range(1, max_attempts + 1):
            if self._shutting_down:
                return

            delay = min(base * (2 ** (attempt - 1)), cap)
            logger.info(f"[IBKR] Reconnect attempt {attempt}/{max_attempts} in {delay:.1f}s")
            await asyncio.sleep(delay)

            try:
                await self._connect()
                logger.info(f"[IBKR] Reconnected after {attempt} attempt(s)")
                self._reconnect_count = attempt
                return
            except Exception as e:
                logger.warning(f"[IBKR] Reconnect attempt {attempt} failed: {e}")

        logger.critical(
            f"[IBKR] Failed to reconnect after {max_attempts} attempts. "
            f"Manual intervention required."
        )

    # -- Internal: Error Handling ------------------------------------------

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract) -> None:
        """
        Central IBKR error handler.

        Classifies errors as informational, pacing-related, or fatal,
        and dispatches to the appropriate handler.
        """
        # Informational codes (2100-2110): data farm connections, etc.
        if 2100 <= errorCode <= 2110 or errorCode == 2158:
            logger.debug(f"[IBKR Info] {errorCode}: {errorString}")
            return

        # Pacing violation
        if errorCode in (162, 366):
            logger.warning(f"[IBKR Pacing] {errorCode}: {errorString}")
            # The PacingManager will handle backoff
            return

        # Connection-related (handled by _on_disconnect)
        if errorCode in (1100, 1101, 1102):
            logger.warning(f"[IBKR Connection] {errorCode}: {errorString}")
            return

        # Order-related or other errors
        if errorCode >= 100:
            logger.error(f"[IBKR Error] reqId={reqId} code={errorCode}: {errorString}")

        # Forward to external handler if registered
        if self.on_error:
            self.on_error(reqId, errorCode, errorString)
