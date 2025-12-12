"""
Strategy-Specific NautilusTrader Adapters.

Provides concrete adapters for each Homeguard strategy:
- StrategySignalsAdapter: For strategies returning List[Signal]
- BaseStrategyAdapter: For strategies returning (entries, exits) boolean Series
- RAMPAdapter: RAMP with internal state management
- OMRAdapter: Overnight Mean Reversion with time-based entry/exit
- ORBAdapter: Opening Range Breakout intraday strategy
"""

from datetime import datetime, time
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtesting.adapters.nautilus.config import NautilusStrategyConfig
from src.backtesting.adapters.nautilus.base_adapter import NautilusStrategyAdapter
from src.strategies.core.signal import Signal
from src.utils.logger import logger

# Check if NautilusTrader is available
try:
    from nautilus_trader.trading.strategy import Strategy as NautilusStrategy
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False


class StrategySignalsAdapter(NautilusStrategyAdapter):
    """
    Adapter for strategies that implement StrategySignals interface.

    Use this for strategies that return List[Signal] from generate_signals().

    Example strategies:
    - MACrossoverSignals
    - Any custom strategy implementing StrategySignals
    """

    def _create_homeguard_strategy(self) -> Any:
        """Create the Homeguard strategy instance from config."""
        from src.strategies import get_strategy_class

        strategy_cls = get_strategy_class(self.config.strategy_class)
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy class: {self.config.strategy_class}")

        return strategy_cls(**self.config.strategy_params)

    def _get_lookback(self) -> int:
        """Get required lookback from the strategy."""
        if self.homeguard_strategy and hasattr(self.homeguard_strategy, 'get_required_lookback'):
            return self.homeguard_strategy.get_required_lookback()
        return self.config.lookback_periods


class BaseStrategyAdapter(NautilusStrategyAdapter):
    """
    Adapter for strategies that extend BaseStrategy.

    Use this for strategies that return (entries, exits) boolean Series
    from generate_signals().

    The boolean Series format:
    - entries: pd.Series with True where entry signal occurs
    - exits: pd.Series with True where exit signal occurs
    """

    def _create_homeguard_strategy(self) -> Any:
        """Create the Homeguard strategy instance."""
        from src.strategies import get_strategy_class

        strategy_cls = get_strategy_class(self.config.strategy_class)
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy class: {self.config.strategy_class}")

        return strategy_cls(self.config.strategy_params)

    def _get_lookback(self) -> int:
        """Get required lookback from config."""
        return self.config.lookback_periods

    def _generate_strategy_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """
        Convert boolean Series signals to Signal objects.

        BaseStrategy.generate_signals() returns Tuple[pd.Series, pd.Series]
        where the Series are boolean entry/exit signals.
        """
        if self.homeguard_strategy is None:
            return []

        signals = []

        for symbol, df in market_data.items():
            if df.empty:
                continue

            try:
                # Generate boolean signals
                entries, exits = self.homeguard_strategy.generate_signals(df)

                # Get latest bar index
                bar_idx = len(df) - 1
                if bar_idx < 0:
                    continue

                price = float(df['close'].iloc[bar_idx])

                # Check for entry signal
                if entries.iloc[bar_idx]:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction='BUY',
                        confidence=1.0,
                        price=price,
                        metadata={'source': 'BaseStrategy', 'signal_type': 'entry'}
                    ))

                # Check for exit signal
                if exits.iloc[bar_idx]:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction='SELL',
                        confidence=1.0,
                        price=price,
                        metadata={'source': 'BaseStrategy', 'signal_type': 'exit'}
                    ))

            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")

        return signals


class RAMPAdapter(NautilusStrategyAdapter):
    """
    Adapter for RAMP (Regime-Aware Momentum Protection) strategy.

    RAMP has internal state management including:
    - Market regime detection
    - Price/return caches
    - Regime-specific parameters

    Rebalances daily at 3:55 PM ET (near close).
    """

    def __init__(self, config: NautilusStrategyConfig) -> None:
        """Initialize RAMP adapter with additional state tracking."""
        super().__init__(config)

        # RAMP-specific state
        self._price_cache: Dict[str, pd.Series] = {}
        self._current_regime: Optional[str] = None
        self._last_rebalance_date: Optional[datetime] = None

        # Rebalance time (3:55 PM ET)
        self._rebalance_time = time(15, 55)

    def _create_homeguard_strategy(self) -> Any:
        """Create RAMP strategy instance."""
        from src.strategies.advanced.ramp_strategy import RAMPStrategy

        params = self.config.strategy_params.copy()
        params['symbols'] = self.config.symbols

        return RAMPStrategy(params)

    def _get_lookback(self) -> int:
        """RAMP needs at least 42 periods for long-term momentum."""
        return max(42, self.config.lookback_periods)

    def _is_rebalance_time(self, timestamp: datetime) -> bool:
        """
        Check if it's time to rebalance.

        Args:
            timestamp: Current timestamp

        Returns:
            True if should rebalance (3:55 PM and not already rebalanced today)
        """
        current_time = timestamp.time()
        current_date = timestamp.date()

        # Check if within rebalance window (3:54-3:56 PM)
        if not (time(15, 54) <= current_time <= time(15, 56)):
            return False

        # Check if already rebalanced today
        if self._last_rebalance_date == current_date:
            return False

        return True

    def _update_ramp_caches(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Update RAMP's internal price cache.

        Args:
            symbol: Stock symbol
            df: Latest market data
        """
        if 'close' in df.columns:
            self._price_cache[symbol] = df['close'].copy()

    def _process_signals(self, bar: Any) -> None:
        """
        Override to add RAMP-specific rebalance logic.

        Only generates signals at rebalance time (3:55 PM).
        """
        # Extract timestamp
        if NAUTILUS_AVAILABLE and hasattr(bar, 'ts_event'):
            timestamp = pd.Timestamp(bar.ts_event, unit='ns', tz='UTC').to_pydatetime()
        else:
            timestamp = datetime.now()

        # Only rebalance at the right time
        if not self._is_rebalance_time(timestamp):
            return

        # Update caches
        market_data = self.bar_accumulator.get_all_dataframes()
        for symbol, df in market_data.items():
            self._update_ramp_caches(symbol, df)

        # Generate signals
        signals = self._generate_strategy_signals(market_data, timestamp)

        # Mark rebalance complete
        self._last_rebalance_date = timestamp.date()

        # Submit orders
        for signal in signals:
            if signal.direction == 'HOLD':
                continue
            order = self.order_factory.signal_to_order(signal)
            if order is not None:
                self._submit_order(order)


class OMRAdapter(NautilusStrategyAdapter):
    """
    Adapter for OMR (Overnight Mean Reversion) strategy.

    OMR has specific entry/exit timing:
    - Entry: 3:50 PM ET (buy leveraged ETFs)
    - Exit: 9:31 AM ET next day (sell at open)

    Trades overnight holding period only.
    """

    def __init__(self, config: NautilusStrategyConfig) -> None:
        """Initialize OMR adapter with timing state."""
        super().__init__(config)

        # OMR-specific timing
        self._entry_time = time(15, 50)  # 3:50 PM ET
        self._exit_time = time(9, 31)    # 9:31 AM ET

        # Position tracking
        self._overnight_positions: Dict[str, float] = {}

    def _create_homeguard_strategy(self) -> Any:
        """Create OMR strategy instance."""
        from src.strategies.advanced.overnight_mean_reversion import OvernightMeanReversionStrategy

        return OvernightMeanReversionStrategy(self.config.strategy_params)

    def _get_lookback(self) -> int:
        """OMR needs historical data for Bayesian model."""
        return max(252, self.config.lookback_periods)  # ~1 year

    def _is_entry_time(self, timestamp: datetime) -> bool:
        """Check if it's entry time (3:50 PM)."""
        current_time = timestamp.time()
        return time(15, 49) <= current_time <= time(15, 51)

    def _is_exit_time(self, timestamp: datetime) -> bool:
        """Check if it's exit time (9:31 AM)."""
        current_time = timestamp.time()
        return time(9, 30) <= current_time <= time(9, 32)

    def _process_signals(self, bar: Any) -> None:
        """
        Process signals with OMR-specific timing.

        - At 3:50 PM: Generate entry signals
        - At 9:31 AM: Generate exit signals for overnight positions
        """
        if NAUTILUS_AVAILABLE and hasattr(bar, 'ts_event'):
            timestamp = pd.Timestamp(bar.ts_event, unit='ns', tz='UTC').to_pydatetime()
        else:
            timestamp = datetime.now()

        market_data = self.bar_accumulator.get_all_dataframes()

        # Entry time - generate buy signals
        if self._is_entry_time(timestamp):
            signals = self._generate_strategy_signals(market_data, timestamp)
            for signal in signals:
                if signal.direction == 'BUY':
                    order = self.order_factory.signal_to_order(signal)
                    if order is not None:
                        self._submit_order(order)
                        self._overnight_positions[signal.symbol] = signal.price

        # Exit time - close overnight positions
        elif self._is_exit_time(timestamp) and self._overnight_positions:
            for symbol, entry_price in list(self._overnight_positions.items()):
                if symbol in market_data:
                    df = market_data[symbol]
                    current_price = float(df['close'].iloc[-1]) if not df.empty else entry_price

                    signal = Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction='SELL',
                        confidence=1.0,
                        price=current_price,
                        metadata={'source': 'OMR', 'signal_type': 'exit'}
                    )
                    order = self.order_factory.signal_to_order(signal)
                    if order is not None:
                        self._submit_order(order)

            # Clear overnight positions
            self._overnight_positions.clear()


class ORBAdapter(NautilusStrategyAdapter):
    """
    Adapter for ORB (Opening Range Breakout) strategy.

    ORB is an intraday strategy that:
    1. Establishes opening range in first N minutes
    2. Trades breakouts/breakdowns of the range
    3. Closes all positions before market close

    Requires minute-level bar data.
    """

    def __init__(self, config: NautilusStrategyConfig) -> None:
        """Initialize ORB adapter with intraday state."""
        super().__init__(config)

        # ORB-specific parameters
        self._opening_range_minutes = config.strategy_params.get('opening_range_minutes', 15)
        self._market_open = time(9, 30)
        self._market_close = time(16, 0)
        self._close_positions_time = time(15, 45)

        # Daily state (reset each day)
        self._current_date: Optional[datetime] = None
        self._opening_highs: Dict[str, float] = {}
        self._opening_lows: Dict[str, float] = {}
        self._range_established: Dict[str, bool] = {}
        self._intraday_positions: Dict[str, str] = {}  # symbol -> 'long' or 'short'

    def _create_homeguard_strategy(self) -> Any:
        """Create ORB strategy instance."""
        from src.strategies.advanced.orb_strategy import ORBStrategy

        return ORBStrategy(self.config.strategy_params)

    def _get_lookback(self) -> int:
        """ORB needs only recent bars for range calculation."""
        return max(self._opening_range_minutes + 5, self.config.lookback_periods)

    def _reset_daily_state(self) -> None:
        """Reset state for new trading day."""
        self._opening_highs.clear()
        self._opening_lows.clear()
        self._range_established.clear()
        self._intraday_positions.clear()

    def _is_opening_range(self, timestamp: datetime) -> bool:
        """Check if within opening range period."""
        current_time = timestamp.time()
        range_end = time(
            9 + (30 + self._opening_range_minutes) // 60,
            (30 + self._opening_range_minutes) % 60
        )
        return self._market_open <= current_time <= range_end

    def _should_close_positions(self, timestamp: datetime) -> bool:
        """Check if should close all positions (near market close)."""
        return timestamp.time() >= self._close_positions_time

    def _process_signals(self, bar: Any) -> None:
        """
        Process ORB signals with intraday logic.

        1. During opening range: Track high/low
        2. After range established: Trade breakouts
        3. Near close: Close all positions
        """
        if NAUTILUS_AVAILABLE and hasattr(bar, 'ts_event'):
            timestamp = pd.Timestamp(bar.ts_event, unit='ns', tz='UTC').to_pydatetime()
        else:
            timestamp = datetime.now()

        # Check for new day
        current_date = timestamp.date()
        if self._current_date != current_date:
            self._reset_daily_state()
            self._current_date = current_date

        symbol = self._extract_symbol(bar)
        if symbol is None:
            return

        market_data = self.bar_accumulator.get_all_dataframes()
        if symbol not in market_data or market_data[symbol].empty:
            return

        df = market_data[symbol]
        current_high = float(bar.high)
        current_low = float(bar.low)
        current_close = float(bar.close)

        # During opening range - track high/low
        if self._is_opening_range(timestamp):
            if symbol not in self._opening_highs:
                self._opening_highs[symbol] = current_high
                self._opening_lows[symbol] = current_low
            else:
                self._opening_highs[symbol] = max(self._opening_highs[symbol], current_high)
                self._opening_lows[symbol] = min(self._opening_lows[symbol], current_low)
            return

        # Mark range as established after opening period
        if symbol in self._opening_highs and symbol not in self._range_established:
            self._range_established[symbol] = True
            logger.debug(
                f"ORB range for {symbol}: "
                f"High={self._opening_highs[symbol]:.2f}, Low={self._opening_lows[symbol]:.2f}"
            )

        # Close positions near end of day
        if self._should_close_positions(timestamp):
            if symbol in self._intraday_positions:
                signal = Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction='SELL' if self._intraday_positions[symbol] == 'long' else 'BUY',
                    confidence=1.0,
                    price=current_close,
                    metadata={'source': 'ORB', 'signal_type': 'close'}
                )
                order = self.order_factory.signal_to_order(signal)
                if order is not None:
                    self._submit_order(order)
                del self._intraday_positions[symbol]
            return

        # Trade breakouts (only if range established and no position)
        if symbol in self._range_established and symbol not in self._intraday_positions:
            opening_high = self._opening_highs[symbol]
            opening_low = self._opening_lows[symbol]

            # Breakout long
            if current_close > opening_high:
                signal = Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction='BUY',
                    confidence=0.8,
                    price=current_close,
                    metadata={
                        'source': 'ORB',
                        'signal_type': 'breakout_long',
                        'opening_high': opening_high
                    }
                )
                order = self.order_factory.signal_to_order(signal)
                if order is not None:
                    self._submit_order(order)
                    self._intraday_positions[symbol] = 'long'

            # Breakdown short
            elif current_close < opening_low:
                signal = Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction='SELL',
                    confidence=0.8,
                    price=current_close,
                    metadata={
                        'source': 'ORB',
                        'signal_type': 'breakdown_short',
                        'opening_low': opening_low
                    }
                )
                order = self.order_factory.signal_to_order(signal)
                if order is not None:
                    self._submit_order(order)
                    self._intraday_positions[symbol] = 'short'
