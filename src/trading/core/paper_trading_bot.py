"""
Paper Trading Bot - Main Orchestrator for Live Trading

Coordinates all components for automated paper trading:
- Broker interface
- Execution engine
- Position manager
- Trading strategy
- Scheduling
- Risk management

This is the entry point for running the live trading system.
"""

from typing import Dict, Optional
from datetime import datetime, time
from pathlib import Path
import time as time_module
import yaml

from src.trading.brokers.broker_factory import BrokerFactory
from src.trading.brokers.broker_interface import BrokerInterface
from src.trading.core.execution_engine import ExecutionEngine
from src.trading.core.position_manager import PositionManager
from src.trading.strategies.omr_live_strategy import OMRLiveStrategy
from src.utils.logger import get_logger

logger = get_logger()  # Use global logger (no file creation)


class PaperTradingBot:
    """
    Main orchestrator for paper trading system.

    Coordinates all components and manages the trading lifecycle:
    1. Initialize components from configuration
    2. Train strategy models
    3. Schedule entry/exit times
    4. Execute trades
    5. Monitor positions and risk
    6. Handle errors and state persistence
    """

    def __init__(self, broker_config_path: str, strategy_config_path: str):
        """
        Initialize paper trading bot.

        Args:
            broker_config_path: Path to broker configuration YAML
            strategy_config_path: Path to strategy configuration YAML
        """
        logger.header("=" * 70)
        logger.header("Initializing Paper Trading Bot")
        logger.header("=" * 70)

        # Load configurations
        self.broker_config = self._load_config(broker_config_path)
        self.strategy_config = self._load_config(strategy_config_path)

        # Initialize components
        self.broker: Optional[BrokerInterface] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.position_manager: Optional[PositionManager] = None
        self.strategy: Optional[OMRLiveStrategy] = None

        # State
        self.is_initialized = False
        self.is_trading = False
        self.last_entry_check = None
        self.last_exit_check = None

        # Extract timing parameters
        strategy_params = self.strategy_config.get('strategy', {})
        self.entry_time = self._parse_time(strategy_params.get('entry_time', '15:50:00'))
        self.exit_time = self._parse_time(strategy_params.get('exit_time', '09:31:00'))
        self.entry_window = strategy_params.get('entry_window_seconds', 30)
        self.exit_window = strategy_params.get('exit_window_seconds', 30)

        logger.info(f"Entry time: {self.entry_time} ± {self.entry_window}s")
        logger.info(f"Exit time: {self.exit_time} ± {self.exit_window}s")

    def initialize(self) -> None:
        """
        Initialize all components.

        This must be called before start_trading().
        """
        logger.info("\nInitializing components...")

        # 1. Initialize broker
        logger.info("1. Creating broker connection...")
        broker_config_path = self.broker_config.get('_config_path', 'config/trading/broker_alpaca.yaml')
        self.broker = BrokerFactory.create_from_yaml(broker_config_path)

        # Test connection
        if not self.broker.test_connection():
            raise RuntimeError("Failed to connect to broker")
        logger.success("Broker connected successfully")

        # 2. Initialize execution engine
        logger.info("2. Initializing execution engine...")
        self.execution_engine = ExecutionEngine(
            broker=self.broker,
            max_retries=3,
            retry_delay=1.0,
            fill_timeout=30.0
        )
        logger.success("Execution engine initialized")

        # 3. Initialize position manager
        logger.info("3. Initializing position manager...")
        position_config = self.strategy_config.get('position_manager', {})
        self.position_manager = PositionManager(position_config)

        # Load saved state if exists
        state_file = position_config.get('state_file', 'data/trading/position_state.json')
        if Path(state_file).exists():
            self.position_manager.load_state(state_file)
            logger.info(f"Loaded position state: {len(self.position_manager.positions)} open positions")

        logger.success("Position manager initialized")

        # 4. Initialize strategy
        logger.info("4. Initializing OMR strategy...")
        strategy_params = self.strategy_config.get('strategy', {})
        self.strategy = OMRLiveStrategy(strategy_params)
        logger.success("Strategy initialized")

        self.is_initialized = True
        logger.success("\nAll components initialized successfully!")

    def train_strategy(self, historical_data: Dict) -> None:
        """
        Train strategy models with historical data.

        Args:
            historical_data: Dict of symbol -> DataFrame with historical data
        """
        if not self.is_initialized:
            raise RuntimeError("Bot not initialized. Call initialize() first.")

        logger.header("\nTraining Strategy Models")
        logger.header("=" * 70)

        self.strategy.train(historical_data)

        logger.success("\nStrategy training complete!")

    def start_trading(self, run_once: bool = False) -> None:
        """
        Start the trading bot.

        Args:
            run_once: If True, check signals once and exit. If False, run continuously.
        """
        if not self.is_initialized:
            raise RuntimeError("Bot not initialized. Call initialize() first.")

        if not self.strategy.is_trained:
            raise RuntimeError("Strategy not trained. Call train_strategy() first.")

        logger.header("\n" + "=" * 70)
        logger.header("Starting Paper Trading Bot")
        logger.header("=" * 70)

        self.is_trading = True

        try:
            if run_once:
                # Run once for testing
                logger.info("Running in single-execution mode...")
                self._check_and_execute_signals()
            else:
                # Continuous operation
                logger.info("Running in continuous mode...")
                logger.info("Press Ctrl+C to stop")

                while self.is_trading:
                    self._check_and_execute_signals()
                    self._monitor_positions()
                    time_module.sleep(10)  # Check every 10 seconds

        except KeyboardInterrupt:
            logger.warning("\nReceived stop signal")
            self.stop_trading()

        except Exception as e:
            logger.error(f"\nError in trading loop: {e}")
            import traceback
            traceback.print_exc()
            self.stop_trading()

    def stop_trading(self) -> None:
        """Stop the trading bot gracefully."""
        logger.info("\nStopping trading bot...")

        self.is_trading = False

        # Save position state
        if self.position_manager:
            state_file = self.strategy_config.get('position_manager', {}).get(
                'state_file', 'data/trading/position_state.json'
            )
            self.position_manager.save_state(state_file)
            logger.info("Position state saved")

        # Print final metrics
        if self.execution_engine:
            metrics = self.execution_engine.get_execution_metrics()
            logger.info("\nExecution Metrics:")
            logger.info(f"  Total Orders: {metrics['total_orders']}")
            logger.info(f"  Successful: {metrics['successful_orders']}")
            logger.info(f"  Failed: {metrics['failed_orders']}")
            logger.info(f"  Success Rate: {metrics['success_rate']:.1%}")

        if self.position_manager:
            portfolio_metrics = self.position_manager.get_portfolio_metrics()
            logger.info("\nPortfolio Metrics:")
            logger.info(f"  Total Trades: {portfolio_metrics['total_trades']}")
            logger.info(f"  Win Rate: {portfolio_metrics['win_rate']:.1%}")
            logger.info(f"  Total P&L: ${portfolio_metrics['total_pnl']:,.2f}")

        logger.success("\nTrading bot stopped successfully")

    def _check_and_execute_signals(self) -> None:
        """Check if it's time to generate and execute signals."""
        current_time = datetime.now().time()

        # Check if market is open
        if not self.broker.is_market_open():
            return

        # Check for entry signals (3:50 PM ± 30s)
        if self._is_within_window(current_time, self.entry_time, self.entry_window):
            if self.last_entry_check != datetime.now().date():
                logger.info(f"\nEntry window active ({current_time})")
                self._execute_entry_signals()
                self.last_entry_check = datetime.now().date()

        # Check for exit signals (9:31 AM ± 30s)
        if self._is_within_window(current_time, self.exit_time, self.exit_window):
            if self.last_exit_check != datetime.now().date():
                logger.info(f"\nExit window active ({current_time})")
                self._execute_exit_signals()
                self.last_exit_check = datetime.now().date()

    def _execute_entry_signals(self) -> None:
        """Generate and execute entry signals."""
        logger.header("Generating Entry Signals")

        try:
            # Get current market data (simplified - would need real implementation)
            current_data = self._fetch_current_data()

            # Generate signals
            signals = self.strategy.generate_entry_signals(current_data, self.broker)

            if not signals:
                logger.info("No entry signals generated")
                return

            # Check risk limits
            account = self.broker.get_account()
            portfolio_value = account['portfolio_value']

            # Execute signals
            executed_count = 0
            for signal in signals:
                # Calculate position size
                position_size_pct = self.strategy_config['strategy']['position_size_pct']
                position_value = portfolio_value * position_size_pct
                quantity = int(position_value / signal['entry_price'])

                # Check risk limits
                is_valid, reason = self.position_manager.check_risk_limits(
                    new_position_value=position_value,
                    portfolio_value=portfolio_value
                )

                if not is_valid:
                    logger.warning(f"Risk limit check failed for {signal['symbol']}: {reason}")
                    continue

                # Execute order
                try:
                    result = self.execution_engine.execute_order(
                        symbol=signal['symbol'],
                        quantity=quantity,
                        side=signal['side'],
                        wait_for_fill=True
                    )

                    # Record position
                    if result['status'].value == 'success':
                        order = result['order']
                        self.position_manager.add_position(
                            symbol=signal['symbol'],
                            entry_price=order['filled_avg_price'],
                            qty=order['filled_qty'],
                            timestamp=datetime.now(),
                            order_id=order['order_id'],
                            signal=signal
                        )
                        executed_count += 1

                except Exception as e:
                    logger.error(f"Failed to execute entry for {signal['symbol']}: {e}")

            logger.success(f"Executed {executed_count}/{len(signals)} entry signals")

        except Exception as e:
            logger.error(f"Error executing entry signals: {e}")
            import traceback
            traceback.print_exc()

    def _execute_exit_signals(self) -> None:
        """Generate and execute exit signals."""
        logger.header("Generating Exit Signals")

        try:
            # Generate exit signals
            signals = self.strategy.generate_exit_signals(self.broker)

            if not signals:
                logger.info("No exit signals generated")
                return

            # Execute exits
            executed_count = 0
            for signal in signals:
                try:
                    result = self.execution_engine.execute_order(
                        symbol=signal['symbol'],
                        quantity=signal['quantity'],
                        side=signal['side'],
                        wait_for_fill=True
                    )

                    # Close position
                    if result['status'].value == 'success':
                        position = self.position_manager.get_position_by_symbol(signal['symbol'])
                        if position:
                            order = result['order']
                            self.position_manager.close_position(
                                position_id=position['position_id'],
                                exit_price=order['filled_avg_price'],
                                timestamp=datetime.now(),
                                reason='scheduled_exit'
                            )
                            executed_count += 1

                except Exception as e:
                    logger.error(f"Failed to execute exit for {signal['symbol']}: {e}")

            logger.success(f"Executed {executed_count}/{len(signals)} exit signals")

        except Exception as e:
            logger.error(f"Error executing exit signals: {e}")
            import traceback
            traceback.print_exc()

    def _monitor_positions(self) -> None:
        """Monitor open positions and check stop-losses."""
        if not self.position_manager.positions:
            return

        try:
            # Get current prices
            current_prices = {}
            for position in self.position_manager.get_open_positions():
                try:
                    quote = self.broker.get_latest_quote(position['symbol'])
                    current_prices[position['symbol']] = (quote['bid'] + quote['ask']) / 2
                except:
                    pass

            # Check stop-losses
            positions_to_stop = self.position_manager.check_stop_losses(current_prices)

            if positions_to_stop:
                logger.warning(f"\nStop-loss triggered for {len(positions_to_stop)} position(s)")

                for position in positions_to_stop:
                    try:
                        result = self.execution_engine.close_position(
                            symbol=position['symbol'],
                            wait_for_fill=True
                        )

                        if result['status'].value == 'success':
                            order = result['order']
                            self.position_manager.close_position(
                                position_id=position['position_id'],
                                exit_price=order['filled_avg_price'],
                                timestamp=datetime.now(),
                                reason='stop_loss'
                            )

                    except Exception as e:
                        logger.error(f"Failed to close position for {position['symbol']}: {e}")

        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")

    def _fetch_current_data(self) -> Dict:
        """
        Fetch current market data for strategy.

        For signal generation, we need:
        - SPY/VIX: Daily data for regime detection
        - Leveraged ETFs: Minute data for today to calculate intraday returns

        Returns:
            Dict of symbol -> DataFrame with price data
        """
        from datetime import timedelta
        import pandas as pd

        # Get list of symbols from strategy config
        strategy_params = self.strategy_config.get('strategy', {})
        symbols = strategy_params.get('symbols', [])

        end_date = datetime.now()
        market_data = {}

        try:
            # 1. Fetch SPY and VIX daily data for regime detection
            logger.debug("Fetching SPY/VIX daily data for regime detection...")

            for symbol in ['SPY', 'VIX']:
                try:
                    # VIX needs 252+ days for percentile, SPY needs 200+ for moving averages
                    days_needed = 365 if symbol == 'VIX' else 250
                    start_date = end_date - timedelta(days=days_needed)

                    bars = self.broker.get_bars(
                        symbols=[symbol],
                        timeframe='1Day',
                        start=start_date,
                        end=end_date
                    )

                    if bars is not None and not bars.empty:
                        # get_bars returns MultiIndex (symbol, timestamp) - extract for this symbol
                        if isinstance(bars.index, pd.MultiIndex):
                            symbol_bars = bars.xs(symbol, level=0, drop_level=True)
                        else:
                            symbol_bars = bars

                        market_data[symbol] = symbol_bars
                        logger.debug(f"  Fetched {len(symbol_bars)} days of {symbol} data")

                except Exception as e:
                    logger.error(f"  Failed to fetch {symbol} data: {e}")

            # Create mock VIX from SPY volatility if VIX not available
            if 'VIX' not in market_data and 'SPY' in market_data:
                logger.warning("  VIX data not available - creating mock VIX from SPY volatility")
                import numpy as np

                spy_df = market_data['SPY']

                # Calculate rolling volatility as proxy for VIX
                returns = spy_df['close'].pct_change()
                rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100  # Annualized vol as %

                # Create VIX-like dataframe
                vix_df = pd.DataFrame({
                    'open': rolling_vol,
                    'high': rolling_vol * 1.1,
                    'low': rolling_vol * 0.9,
                    'close': rolling_vol,
                    'volume': 0,
                    'trade_count': 0,
                    'vwap': rolling_vol
                }, index=spy_df.index)

                # Fill NaN with default value
                vix_df = vix_df.fillna(15.0)

                market_data['VIX'] = vix_df
                logger.debug(f"  Created mock VIX: {len(vix_df)} days")

            # 2. Fetch minute data for leveraged ETFs (today only)
            logger.debug("Fetching minute data for leveraged ETFs...")

            if symbols:
                try:
                    # Get today's market hours
                    market_open, market_close = self.broker.get_market_hours(end_date)

                    # Fetch from market open to now
                    start_time = market_open
                    end_time = min(end_date, market_close)

                    # Fetch all symbols at once
                    bars = self.broker.get_bars(
                        symbols=symbols,
                        timeframe='1Min',
                        start=start_time,
                        end=end_time
                    )

                    if bars is not None and not bars.empty:
                        # Extract each symbol's data
                        for symbol in symbols:
                            try:
                                if isinstance(bars.index, pd.MultiIndex):
                                    symbol_bars = bars.xs(symbol, level=0, drop_level=True)
                                else:
                                    # Single symbol - already flattened
                                    symbol_bars = bars

                                if not symbol_bars.empty:
                                    market_data[symbol] = symbol_bars
                                    logger.debug(f"  Fetched {len(symbol_bars)} minute bars for {symbol}")

                            except KeyError:
                                logger.debug(f"  No data for {symbol}")
                                continue

                except Exception as e:
                    logger.error(f"  Failed to fetch minute data: {e}")

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            import traceback
            traceback.print_exc()

        logger.info(f"Fetched data for {len(market_data)} symbols")
        return market_data

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['_config_path'] = config_path
        return config

    @staticmethod
    def _parse_time(time_str: str) -> time:
        """Parse time string (HH:MM:SS) to time object."""
        parts = time_str.split(':')
        return time(int(parts[0]), int(parts[1]), int(parts[2]))

    @staticmethod
    def _is_within_window(current_time: time, target_time: time, window_seconds: int) -> bool:
        """Check if current time is within window of target time."""
        current_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
        target_seconds = target_time.hour * 3600 + target_time.minute * 60 + target_time.second

        diff = abs(current_seconds - target_seconds)
        return diff <= window_seconds
