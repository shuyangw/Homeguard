"""
Crypto Broker Router - Failover between Coinbase and Alpaca.

DEPRECATED for paper trading: Alpaca crypto has significant limitations
(integer-only quantities, limited universe). Use DemoBroker instead.

This module is retained for live trading with real money.

Provides automatic failover between crypto brokers:
1. Coinbase (primary) - Advanced Trade API
2. Alpaca (secondary) - Crypto trading

If primary fails, automatically routes to secondary.

Usage (live trading only):
    router = CryptoBrokerRouter()  # Auto-configures from env vars
    order = router.place_crypto_order(symbol='BTC/USD', ...)

For paper trading, use:
    from src.trading.demo import DemoBroker
    broker = DemoBroker(initial_cash=10000.0)
"""

from decimal import Decimal
from typing import Dict, List, Optional

from src.trading.brokers.interfaces import (
    CryptoTradingInterface,
    AccountInterface,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    BrokerConnectionError,
    NoPositionError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CryptoBrokerRouter(CryptoTradingInterface, AccountInterface):
    """
    Crypto broker router with automatic failover.

    Routes orders to primary broker (Coinbase) first, then falls back
    to secondary (Alpaca) if primary fails.

    Features:
    - Automatic failover on connection errors
    - Manual broker selection
    - Unified interface across brokers
    - Connection health monitoring

    Usage:
        # Auto-configure from environment
        router = CryptoBrokerRouter()

        # Or provide specific brokers
        from src.trading.brokers.coinbase_broker import CoinbaseBroker
        from src.trading.brokers.alpaca_crypto_broker import AlpacaCryptoBroker

        router = CryptoBrokerRouter(
            primary=CoinbaseBroker(),
            secondary=AlpacaCryptoBroker(paper=False)
        )

        # Place order (auto-failover)
        order = router.place_crypto_order('BTC/USD', Decimal('0.01'), OrderSide.BUY)
    """

    def __init__(
        self,
        primary: Optional[CryptoTradingInterface] = None,
        secondary: Optional[CryptoTradingInterface] = None,
        auto_failover: bool = True
    ):
        """
        Initialize crypto broker router.

        Args:
            primary: Primary broker (Coinbase). Auto-created if None.
            secondary: Secondary broker (Alpaca). Auto-created if None.
            auto_failover: Enable automatic failover on errors.
        """
        self._primary = primary
        self._secondary = secondary
        self._auto_failover = auto_failover
        self._active_broker_name = 'primary'
        self._failover_count = 0

        # Lazy initialization - brokers created on first use
        self._brokers_initialized = False

    def _ensure_brokers(self) -> None:
        """Ensure brokers are initialized."""
        if self._brokers_initialized:
            return

        # Create primary (Coinbase) if not provided
        if self._primary is None:
            try:
                from src.trading.brokers.coinbase_broker import CoinbaseBroker
                self._primary = CoinbaseBroker()
                logger.info("[Router] Initialized Coinbase as primary broker")
            except Exception as e:
                logger.warning(f"[Router] Failed to initialize Coinbase: {e}")

        # Create secondary (Alpaca) if not provided
        if self._secondary is None:
            try:
                from src.trading.brokers.alpaca_crypto_broker import AlpacaCryptoBroker
                self._secondary = AlpacaCryptoBroker(paper=False)
                logger.info("[Router] Initialized Alpaca as secondary broker")
            except Exception as e:
                logger.warning(f"[Router] Failed to initialize Alpaca: {e}")

        if self._primary is None and self._secondary is None:
            raise BrokerConnectionError("No crypto brokers available")

        self._brokers_initialized = True

    @property
    def _active_broker(self) -> CryptoTradingInterface:
        """Get the currently active broker."""
        self._ensure_brokers()

        if self._active_broker_name == 'primary' and self._primary:
            return self._primary
        elif self._secondary:
            return self._secondary
        elif self._primary:
            return self._primary
        else:
            raise BrokerConnectionError("No active broker available")

    def set_active_broker(self, name: str) -> None:
        """
        Manually set the active broker.

        Args:
            name: 'primary', 'secondary', 'coinbase', or 'alpaca'
        """
        self._ensure_brokers()

        name = name.lower()
        if name in ('primary', 'coinbase'):
            if self._primary is None:
                raise ValueError("Primary broker (Coinbase) not available")
            self._active_broker_name = 'primary'
            logger.info("[Router] Switched to primary broker (Coinbase)")
        elif name in ('secondary', 'alpaca'):
            if self._secondary is None:
                raise ValueError("Secondary broker (Alpaca) not available")
            self._active_broker_name = 'secondary'
            logger.info("[Router] Switched to secondary broker (Alpaca)")
        else:
            raise ValueError(f"Unknown broker: {name}")

    def _try_with_failover(self, operation_name: str, *args, **kwargs):
        """
        Execute operation with automatic failover.

        Args:
            operation_name: Method name to call on broker
            *args, **kwargs: Arguments to pass to the method

        Returns:
            Operation result

        Raises:
            Last exception if all brokers fail
        """
        self._ensure_brokers()
        brokers = []

        # Build broker list based on active preference
        if self._active_broker_name == 'primary':
            if self._primary:
                brokers.append(('primary', self._primary))
            if self._secondary and self._auto_failover:
                brokers.append(('secondary', self._secondary))
        else:
            if self._secondary:
                brokers.append(('secondary', self._secondary))
            if self._primary and self._auto_failover:
                brokers.append(('primary', self._primary))

        last_error = None
        for name, broker in brokers:
            try:
                # Get the method from the broker
                method = getattr(broker, operation_name)
                result = method(*args, **kwargs)

                # If we failed over, log it
                if name != self._active_broker_name:
                    self._failover_count += 1
                    logger.warning(
                        f"[Router] Failover to {name} broker "
                        f"(total failovers: {self._failover_count})"
                    )

                return result

            except BrokerConnectionError as e:
                last_error = e
                logger.warning(f"[Router] {name} broker failed: {e}")
                continue
            except Exception as e:
                # Non-connection errors should propagate
                raise

        # All brokers failed
        raise BrokerConnectionError(
            f"All brokers failed for {operation_name}: {last_error}"
        )

    # ==================== Account Interface ====================

    def get_account(self) -> Dict:
        """Get account information from active broker."""
        return self._try_with_failover('get_account')

    def test_connection(self) -> bool:
        """Test connection to any available broker."""
        self._ensure_brokers()

        if self._primary and self._primary.test_connection():
            return True
        if self._secondary and self._secondary.test_connection():
            return True
        return False

    # ==================== Crypto Trading Interface ====================

    def get_crypto_positions(self) -> List[Dict]:
        """Get all crypto positions."""
        return self._try_with_failover('get_crypto_positions')

    def get_crypto_position(self, symbol: str) -> Optional[Dict]:
        """Get specific crypto position."""
        return self._try_with_failover('get_crypto_position', symbol)

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
        """Place a crypto order."""
        return self._try_with_failover(
            'place_crypto_order',
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            **kwargs
        )

    def close_crypto_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None
    ) -> Dict:
        """Close a crypto position."""
        return self._try_with_failover(
            'close_crypto_position',
            symbol=symbol,
            quantity=quantity
        )

    def close_all_crypto_positions(self) -> List[Dict]:
        """Close all crypto positions."""
        return self._try_with_failover('close_all_crypto_positions')

    def get_crypto_quote(self, symbol: str) -> Dict:
        """Get crypto quote."""
        return self._try_with_failover('get_crypto_quote', symbol)

    def get_crypto_balance(self, currency: str = 'USD') -> Dict:
        """Get crypto balance."""
        return self._try_with_failover('get_crypto_balance', currency)

    # ==================== Order Management ====================

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order by ID."""
        return self._try_with_failover('get_order', order_id)

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """Get orders."""
        return self._try_with_failover('get_orders', status=status, symbol=symbol)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        return self._try_with_failover('cancel_order', order_id)

    def cancel_all_orders(self) -> int:
        """Cancel all orders."""
        return self._try_with_failover('cancel_all_orders')

    # ==================== Router-specific Methods ====================

    def get_broker_status(self) -> Dict:
        """Get status of all brokers."""
        self._ensure_brokers()

        status = {
            'active_broker': self._active_broker_name,
            'auto_failover': self._auto_failover,
            'failover_count': self._failover_count,
            'brokers': {}
        }

        if self._primary:
            if self._primary.test_connection():
                status['brokers']['primary'] = {'name': 'Coinbase', 'status': 'connected'}
            else:
                status['brokers']['primary'] = {'name': 'Coinbase', 'status': 'disconnected'}

        if self._secondary:
            if self._secondary.test_connection():
                status['brokers']['secondary'] = {'name': 'Alpaca', 'status': 'connected'}
            else:
                status['brokers']['secondary'] = {'name': 'Alpaca', 'status': 'disconnected'}

        return status

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format."""
        symbol = symbol.upper()
        symbol = symbol.replace('-', '/')
        symbol = symbol.replace('_', '/')
        if '/' not in symbol and not symbol.endswith('USD'):
            symbol = f"{symbol}/USD"
        return symbol
