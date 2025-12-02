"""
Base Strategy Adapter for Live Trading.

Connects pure strategy implementations to live trading infrastructure.
Handles data fetching, signal generation, and order execution.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import pytz

from src.strategies.core import StrategySignals, Signal
from src.trading.brokers.broker_interface import BrokerInterface, OrderSide, OrderType
from src.trading.core.execution_engine import ExecutionEngine
from src.trading.core.position_manager import PositionManager
from src.utils.logger import logger
from src.utils.timezone import now as get_now_et


class StrategyAdapter(ABC):
    """
    Base adapter for connecting pure strategies to live trading.

    Responsibilities:
    - Fetch market data from broker
    - Call pure strategy to generate signals
    - Convert signals to orders via ExecutionEngine
    - Manage positions and risk
    - Handle scheduling and lifecycle
    """

    def __init__(
        self,
        strategy: StrategySignals,
        broker: BrokerInterface,
        symbols: List[str],
        position_size: float = 0.1,
        max_positions: int = 5,
        data_lookback_days: int = 365
    ):
        """
        Initialize strategy adapter.

        Args:
            strategy: Pure strategy implementation
            broker: Broker interface for data and orders
            symbols: List of symbols to trade
            position_size: Position size as fraction of capital (default: 0.1 = 10%)
            max_positions: Maximum concurrent positions (default: 5)
            data_lookback_days: Days of historical data to fetch (default: 365)
        """
        self.strategy = strategy
        self.broker = broker
        self.symbols = symbols
        self.position_size = position_size
        self.max_positions = max_positions
        self.data_lookback_days = data_lookback_days

        # Initialize components
        self.execution_engine = ExecutionEngine(broker)

        # Create position manager config
        position_config = {
            'max_position_size_pct': position_size,
            'max_concurrent_positions': max_positions,
            'max_total_exposure_pct': position_size * max_positions,
            'stop_loss_pct': -0.02
        }
        self.position_manager = PositionManager(position_config)

        # Data caching for performance optimization
        self._data_cache: Optional[Dict[str, pd.DataFrame]] = None
        self._cache_date: Optional[datetime] = None

        # Intraday data caching for reliability (pre-fetch at 3:45 PM)
        self._intraday_cache: Optional[Dict[str, pd.DataFrame]] = None
        self._intraday_cache_time: Optional[datetime] = None

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"  Strategy: {strategy.__class__.__name__}")
        logger.info(f"  Symbols: {len(symbols)}")
        logger.info(f"  Position size: {position_size:.1%}")
        logger.info(f"  Max positions: {max_positions}")

    def preload_historical_data(self) -> None:
        """
        Pre-load historical data for all symbols.

        This should be called once at market open to cache historical data,
        avoiding the need to fetch a full year of data at execution time.
        """
        try:
            logger.info("Pre-loading historical data...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.data_lookback_days)

            self._data_cache = {}

            for symbol in self.symbols:
                try:
                    df = self.broker.get_historical_bars(
                        symbol=symbol,
                        start=start_date,
                        end=end_date,
                        timeframe='1D'
                    )

                    if df is not None and not df.empty:
                        self._data_cache[symbol] = df
                    else:
                        logger.warning(f"No data returned for {symbol}")

                except Exception as e:
                    logger.error(f"Error pre-loading data for {symbol}: {e}")
                    continue

            self._cache_date = datetime.now().date()
            logger.success(
                f"Pre-loaded {len(self._data_cache)}/{len(self.symbols)} symbols "
                f"({self.data_lookback_days} days)"
            )

        except Exception as e:
            logger.error(f"Error in preload_historical_data: {e}")
            self._data_cache = None

    def prefetch_intraday_data(self) -> None:
        """
        Pre-fetch today's intraday data for all symbols.

        This should be called at 3:45 PM to cache today's intraday data,
        avoiding network issues at the critical 3:50 PM execution time.
        Provides a 5-minute buffer while keeping data fresh.
        """
        try:
            logger.info("Pre-fetching today's intraday data...")
            # Use Eastern Time for market hours (market opens 9:30 AM ET)
            now_et = get_now_et()
            market_open_today = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

            # Convert to UTC for API calls (Alpaca expects UTC)
            now_utc = now_et.astimezone(pytz.UTC)
            market_open_utc = market_open_today.astimezone(pytz.UTC)

            logger.info(f"Fetching intraday data from {market_open_today.strftime('%H:%M')} ET to {now_et.strftime('%H:%M')} ET")

            self._intraday_cache = {}

            for symbol in self.symbols:
                try:
                    # Fetch intraday bars from market open to now (using UTC for API)
                    df = self.broker.get_historical_bars(
                        symbol=symbol,
                        start=market_open_utc,
                        end=now_utc,
                        timeframe='1Min'  # 1-minute bars for intraday
                    )

                    if df is not None and not df.empty:
                        self._intraday_cache[symbol] = df
                    else:
                        logger.warning(f"No intraday data returned for {symbol}")

                except Exception as e:
                    logger.error(f"Error pre-fetching intraday data for {symbol}: {e}")
                    continue

            self._intraday_cache_time = now_et
            logger.success(
                f"Pre-fetched intraday data for {len(self._intraday_cache)}/{len(self.symbols)} symbols "
                f"({now_et.strftime('%H:%M')} ET update)"
            )

        except Exception as e:
            logger.error(f"Error in prefetch_intraday_data: {e}")
            self._intraday_cache = None

    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for all symbols.

        Uses cached historical data if available (pre-loaded), and only fetches
        recent data to append. Falls back to full fetch if cache is unavailable.

        Returns:
            Dict of symbol -> DataFrame with OHLCV data
        """
        try:
            # Check if cache is available and current
            today = datetime.now().date()
            cache_is_valid = (
                self._data_cache is not None and
                self._cache_date is not None and
                self._cache_date == today
            )

            if cache_is_valid:
                # Use cached data + fetch only today's data
                logger.info("Using cached data + fetching today's update")
                market_data = {}

                for symbol in self.symbols:
                    try:
                        if symbol in self._data_cache:
                            # Get cached data
                            cached_df = self._data_cache[symbol].copy()

                            # Fetch only today's data
                            today_df = self.broker.get_historical_bars(
                                symbol=symbol,
                                start=datetime.now().replace(hour=0, minute=0, second=0),
                                end=datetime.now(),
                                timeframe='1D'
                            )

                            # Append today's data if available
                            if today_df is not None and not today_df.empty:
                                # Combine cached + today's data, removing duplicates
                                combined = pd.concat([cached_df, today_df])
                                combined = combined[~combined.index.duplicated(keep='last')]
                                market_data[symbol] = combined
                            else:
                                # No new data, use cached
                                market_data[symbol] = cached_df

                        else:
                            # Symbol not in cache, full fetch
                            logger.warning(f"{symbol} not in cache, performing full fetch")
                            df = self.broker.get_historical_bars(
                                symbol=symbol,
                                start=datetime.now() - timedelta(days=self.data_lookback_days),
                                end=datetime.now(),
                                timeframe='1D'
                            )
                            if df is not None and not df.empty:
                                market_data[symbol] = df

                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {e}")
                        # Fall back to cached data if available
                        if symbol in self._data_cache:
                            market_data[symbol] = self._data_cache[symbol]
                        continue

                logger.info(
                    f"Fetched data for {len(market_data)}/{len(self.symbols)} symbols "
                    "(cached + today's update)"
                )
                return market_data

            else:
                # No cache available - full fetch
                logger.info("No cache available, performing full data fetch")
                market_data = {}
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.data_lookback_days)

                for symbol in self.symbols:
                    try:
                        df = self.broker.get_historical_bars(
                            symbol=symbol,
                            start=start_date,
                            end=end_date,
                            timeframe='1D'
                        )

                        if df is not None and not df.empty:
                            market_data[symbol] = df
                        else:
                            logger.warning(f"No data returned for {symbol}")

                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {e}")
                        continue

                logger.info(f"Fetched data for {len(market_data)}/{len(self.symbols)} symbols (full)")
                return market_data

        except Exception as e:
            logger.error(f"Error in fetch_market_data: {e}")
            return {}

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals using pure strategy.

        Args:
            market_data: Market data for all symbols

        Returns:
            List of trading signals
        """
        try:
            timestamp = datetime.now()
            signals = self.strategy.generate_signals(market_data, timestamp)

            logger.info(f"Generated {len(signals)} signals")
            for signal in signals:
                logger.info(
                    f"  {signal.symbol}: {signal.direction} @ ${signal.price:.2f} "
                    f"(confidence: {signal.confidence:.1%})"
                )

            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []

    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter signals based on risk management rules.

        Args:
            signals: Raw signals from strategy

        Returns:
            Filtered signals
        """
        # Get current positions
        current_positions = self.position_manager.get_open_positions()
        current_symbols = {pos['symbol'] for pos in current_positions}

        filtered = []

        for signal in signals:
            # Skip if already have position in this symbol
            if signal.symbol in current_symbols:
                logger.info(f"Skipping {signal.symbol}: Already have position")
                continue

            # Check max positions limit
            if len(current_positions) + len(filtered) >= self.max_positions:
                logger.info(f"Skipping {signal.symbol}: Max positions reached")
                continue

            filtered.append(signal)

        logger.info(f"Filtered {len(signals)} -> {len(filtered)} signals")
        return filtered

    def execute_signals(self, signals: List[Signal]) -> None:
        """
        Execute trading signals via execution engine.

        Args:
            signals: Filtered signals to execute
        """
        if not signals:
            logger.info("No signals to execute")
            return

        # Get account info for position sizing
        account = self.broker.get_account()
        if account is None:
            logger.error("Cannot get account info, skipping execution")
            return

        buying_power = float(account.buying_power)

        for signal in signals:
            try:
                # Calculate position size
                position_value = buying_power * self.position_size
                qty = int(position_value / signal.price)

                if qty <= 0:
                    logger.warning(
                        f"Calculated qty {qty} for {signal.symbol}, skipping"
                    )
                    continue

                # Execute order
                logger.info(
                    f"Executing {signal.direction} {qty} shares of {signal.symbol} "
                    f"@ ${signal.price:.2f}"
                )

                if signal.direction == 'BUY':
                    side = OrderSide.BUY
                elif signal.direction == 'SELL':
                    side = OrderSide.SELL
                else:
                    logger.warning(f"Unknown direction: {signal.direction}")
                    continue

                order = self.execution_engine.execute_order(
                    symbol=signal.symbol,
                    quantity=qty,
                    side=side,
                    order_type=OrderType.MARKET
                )

                if order:
                    logger.success(f"Order placed: {order.get('order_id', 'UNKNOWN')}")
                else:
                    logger.error(f"Failed to place order for {signal.symbol}")

            except Exception as e:
                logger.error(f"Error executing signal for {signal.symbol}: {e}")
                continue

    def update_positions(self) -> None:
        """Log current positions from broker and position manager."""
        try:
            # Get positions from broker
            broker_positions = self.broker.get_positions()

            # Get positions from position manager
            managed_positions = self.position_manager.get_open_positions()

            logger.info(f"Current positions: {len(broker_positions)} (broker), {len(managed_positions)} (managed)")

            # Log broker positions (positions are dicts, not objects)
            for pos in broker_positions:
                pnl_pct = (float(pos['current_price']) - float(pos['avg_entry_price'])) / float(pos['avg_entry_price']) * 100
                logger.info(
                    f"  {pos['symbol']}: {pos['quantity']} shares @ ${float(pos['avg_entry_price']):.2f} "
                    f"(current: ${float(pos['current_price']):.2f}, P&L: {pnl_pct:+.2f}%)"
                )

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    def run_once(self) -> None:
        """
        Run one iteration of the strategy.

        Workflow:
        1. Fetch market data
        2. Generate signals
        3. Filter signals (risk management)
        4. Execute signals
        5. Update positions
        """
        logger.info("=" * 60)
        logger.info(f"Running {self.__class__.__name__} at {datetime.now()}")
        logger.info("=" * 60)

        try:
            # 1. Fetch market data
            market_data = self.fetch_market_data()
            if not market_data:
                logger.warning("No market data, skipping iteration")
                return

            # 2. Generate signals
            signals = self.generate_signals(market_data)

            # 3. Filter signals
            filtered_signals = self.filter_signals(signals)

            # 4. Execute signals
            self.execute_signals(filtered_signals)

            # 5. Update positions
            self.update_positions()

            logger.info("Strategy iteration complete")

        except Exception as e:
            logger.error(f"Error in run_once: {e}")

    @abstractmethod
    def get_schedule(self) -> Dict[str, any]:
        """
        Get scheduling configuration for this strategy.

        Returns:
            Dict with scheduling info:
            - 'interval': Time between runs (e.g., '5min', '1h', '1d')
            - 'market_hours_only': Run only during market hours
            - 'specific_time': Run at specific time (e.g., '15:50' for OMR)
        """
        pass

    def should_run_now(self) -> bool:
        """
        Check if strategy should run now based on schedule.

        Returns:
            True if should run, False otherwise
        """
        schedule = self.get_schedule()
        now = datetime.now()

        # Check market hours if required
        if schedule.get('market_hours_only', True):
            if not self.broker.is_market_open():
                return False

        # Check specific time
        specific_time = schedule.get('specific_time')
        if specific_time:
            target_time = datetime.strptime(specific_time, '%H:%M').time()
            current_time = now.time()
            # Run if within 1 minute of target time
            return abs((datetime.combine(now.date(), current_time) -
                       datetime.combine(now.date(), target_time)).total_seconds()) < 60

        return True


if __name__ == "__main__":
    logger.info("Strategy Adapter Base Class")
    logger.info("=" * 60)
    logger.info("This is the base adapter for live trading strategies")
    logger.info("Extend this class to create adapters for specific strategies")
