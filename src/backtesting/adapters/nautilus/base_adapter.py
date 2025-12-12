"""
Base NautilusTrader Strategy Adapter.

Provides the base adapter class that connects Homeguard strategies
to NautilusTrader's backtesting engine.

This module requires NautilusTrader to be installed for the adapter
classes, but the base protocols work without it for testing.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from src.backtesting.adapters.nautilus.config import NautilusStrategyConfig
from src.backtesting.adapters.nautilus.bar_accumulator import BarAccumulator
from src.backtesting.adapters.nautilus.order_factory import HomeguardOrderFactory
from src.strategies.core.signal import Signal
from src.utils.logger import logger

# Check if NautilusTrader is available
try:
    from nautilus_trader.trading.strategy import Strategy as NautilusStrategy
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.identifiers import InstrumentId
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    # Create stub for type hints
    NautilusStrategy = object  # type: ignore


class NautilusStrategyAdapter(NautilusStrategy if NAUTILUS_AVAILABLE else object):  # type: ignore
    """
    Base adapter for running Homeguard strategies in NautilusTrader.

    This adapter:
    1. Receives bars from NautilusTrader's event system
    2. Accumulates them into DataFrames
    3. Calls the Homeguard strategy to generate signals
    4. Converts signals to NautilusTrader orders

    Subclasses should implement:
    - _create_homeguard_strategy(): Create the wrapped strategy
    - _get_lookback(): Return required lookback periods

    Attributes:
        config: NautilusStrategyConfig with strategy parameters
        bar_accumulator: Accumulates bars into DataFrames
        order_factory: Converts signals to orders
        homeguard_strategy: The wrapped Homeguard strategy instance
    """

    def __init__(self, config: NautilusStrategyConfig) -> None:
        """
        Initialize the adapter.

        Args:
            config: Strategy configuration

        Note:
            Call super().__init__() with NautilusTrader config if needed.
        """
        if NAUTILUS_AVAILABLE:
            # Initialize NautilusTrader Strategy base
            from nautilus_trader.config import StrategyConfig
            super().__init__(StrategyConfig(strategy_id=config.strategy_class))

        self.config = config
        self.bar_accumulator = BarAccumulator(config.lookback_periods)
        self.order_factory = HomeguardOrderFactory(
            strategy=self if NAUTILUS_AVAILABLE else None,
            venue=config.venue,
        )
        self.homeguard_strategy: Optional[Any] = None

        # Track instruments
        self._instruments: Dict[str, Any] = {}

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"  Strategy class: {config.strategy_class}")
        logger.info(f"  Symbols: {len(config.symbols)}")
        logger.info(f"  Lookback: {config.lookback_periods}")

    @abstractmethod
    def _create_homeguard_strategy(self) -> Any:
        """
        Create the Homeguard strategy instance.

        Subclasses must implement this to instantiate their specific strategy.

        Returns:
            Homeguard strategy instance (StrategySignals or BaseStrategy)
        """
        pass

    @abstractmethod
    def _get_lookback(self) -> int:
        """
        Get required lookback periods for the strategy.

        Returns:
            Number of periods needed for indicator calculation
        """
        pass

    def on_start(self) -> None:
        """
        Called when strategy starts.

        Creates the Homeguard strategy and subscribes to instruments.
        """
        logger.info(f"Starting {self.__class__.__name__}...")

        # Create the Homeguard strategy
        self.homeguard_strategy = self._create_homeguard_strategy()

        if NAUTILUS_AVAILABLE:
            # Subscribe to bar data for each symbol
            for symbol in self.config.symbols:
                try:
                    instrument_id = InstrumentId.from_str(f"{symbol}.{self.config.venue}")
                    instrument = self.cache.instrument(instrument_id)
                    if instrument:
                        self._instruments[symbol] = instrument
                        # Subscribe to bars (specific bar type depends on config)
                        self.subscribe_bars(instrument_id)
                        logger.debug(f"Subscribed to {symbol}")
                    else:
                        logger.warning(f"Instrument not found: {symbol}")
                except Exception as e:
                    logger.error(f"Error subscribing to {symbol}: {e}")

        logger.info(f"Strategy started with {len(self._instruments)} instruments")

    def on_bar(self, bar: Any) -> None:
        """
        Called when a new bar is received.

        Args:
            bar: NautilusTrader Bar object
        """
        # Extract symbol from bar
        symbol = self._extract_symbol(bar)
        if symbol is None:
            return

        # Accumulate bar
        self.bar_accumulator.add_bar(symbol, bar)

        # Check if we have enough data
        lookback = self._get_lookback()
        if not self.bar_accumulator.has_sufficient_data(symbol, lookback):
            return

        # Generate signals and execute
        self._process_signals(bar)

    def _extract_symbol(self, bar: Any) -> Optional[str]:
        """
        Extract symbol string from a bar object.

        Args:
            bar: NautilusTrader Bar or mock bar

        Returns:
            Symbol string or None
        """
        if NAUTILUS_AVAILABLE and hasattr(bar, 'bar_type'):
            # Real NautilusTrader bar
            return bar.bar_type.instrument_id.symbol.value
        elif hasattr(bar, 'symbol'):
            # Mock bar
            return str(bar.symbol)
        return None

    def _process_signals(self, bar: Any) -> None:
        """
        Generate signals and submit orders.

        Args:
            bar: The triggering bar
        """
        try:
            # Get market data as DataFrames
            market_data = self.bar_accumulator.get_all_dataframes()

            # Get current timestamp
            if NAUTILUS_AVAILABLE and hasattr(bar, 'ts_event'):
                timestamp = pd.Timestamp(bar.ts_event, unit='ns', tz='UTC').to_pydatetime()
            else:
                timestamp = datetime.now()

            # Generate signals using Homeguard strategy
            signals = self._generate_strategy_signals(market_data, timestamp)

            # Convert signals to orders and submit
            for signal in signals:
                if signal.direction == 'HOLD':
                    continue

                order = self.order_factory.signal_to_order(signal)
                if order is not None:
                    self._submit_order(order)

        except Exception as e:
            logger.error(f"Error processing signals: {e}")

    def _generate_strategy_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """
        Generate signals from the Homeguard strategy.

        Override this in subclasses for strategies with different interfaces.

        Args:
            market_data: Dict of symbol -> DataFrame
            timestamp: Current timestamp

        Returns:
            List of Signal objects
        """
        if self.homeguard_strategy is None:
            return []

        # Standard StrategySignals interface
        if hasattr(self.homeguard_strategy, 'generate_signals'):
            return self.homeguard_strategy.generate_signals(market_data, timestamp)

        return []

    def _submit_order(self, order: Any) -> None:
        """
        Submit an order to the trading system.

        Args:
            order: NautilusTrader order or mock order
        """
        if NAUTILUS_AVAILABLE and hasattr(self, 'submit_order'):
            self.submit_order(order)
            logger.info(f"Submitted order: {order}")
        else:
            # Mock mode - just log
            logger.info(f"[MOCK] Would submit order: {order}")

    def on_stop(self) -> None:
        """Called when strategy stops."""
        logger.info(f"Stopping {self.__class__.__name__}...")

    def on_reset(self) -> None:
        """Called when strategy resets."""
        self.bar_accumulator.clear()
        logger.info(f"Reset {self.__class__.__name__}")

    def on_dispose(self) -> None:
        """Called when strategy is disposed."""
        logger.info(f"Disposing {self.__class__.__name__}")
