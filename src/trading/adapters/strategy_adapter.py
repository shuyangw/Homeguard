"""
Base Strategy Adapter for Live Trading.

Connects pure strategy implementations to live trading infrastructure.
Handles data fetching, signal generation, and order execution.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

from src.strategies.core import StrategySignals, Signal
from src.trading.brokers.broker_interface import BrokerInterface
from src.trading.core.execution_engine import ExecutionEngine
from src.trading.core.position_manager import PositionManager
from src.utils.logger import logger


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

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"  Strategy: {strategy.__class__.__name__}")
        logger.info(f"  Symbols: {len(symbols)}")
        logger.info(f"  Position size: {position_size:.1%}")
        logger.info(f"  Max positions: {max_positions}")

    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for all symbols.

        Returns:
            Dict of symbol -> DataFrame with OHLCV data
        """
        try:
            market_data = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.data_lookback_days)

            for symbol in self.symbols:
                try:
                    # Fetch data from broker
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

            logger.info(f"Fetched data for {len(market_data)}/{len(self.symbols)} symbols")
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

        logger.info(f"Filtered {len(signals)} → {len(filtered)} signals")
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
                    order = self.execution_engine.place_market_order(
                        symbol=signal.symbol,
                        qty=qty,
                        side='buy'
                    )
                elif signal.direction == 'SELL':
                    order = self.execution_engine.place_market_order(
                        symbol=signal.symbol,
                        qty=qty,
                        side='sell'
                    )
                else:
                    logger.warning(f"Unknown direction: {signal.direction}")
                    continue

                if order:
                    logger.success(f"Order placed: {order.id}")
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

            # Log broker positions
            for pos in broker_positions:
                pnl_pct = (float(pos.current_price) - float(pos.avg_entry_price)) / float(pos.avg_entry_price) * 100
                logger.info(
                    f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f} "
                    f"(current: ${float(pos.current_price):.2f}, P&L: {pnl_pct:+.2f}%)"
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
