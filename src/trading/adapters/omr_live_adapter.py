"""
Overnight Mean Reversion (OMR) Live Trading Adapter.

Connects OMR strategy to live trading infrastructure.
Runs at 3:50 PM EST to generate overnight signals.
"""

from typing import List, Dict, Optional
from datetime import datetime, time, timedelta
import pandas as pd
import yfinance as yf

from src.trading.adapters.strategy_adapter import StrategyAdapter
from src.strategies.core import StrategySignals, Signal
from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
from src.strategies.universe import ETFUniverse
from src.trading.brokers.broker_interface import BrokerInterface, OrderSide, OrderType
from src.trading.utils.portfolio_health_check import PortfolioHealthChecker
from src.trading.state import StrategyStateManager
from src.utils.logger import logger
from src.utils.timezone import tz

# Strategy identifier for state tracking
STRATEGY_NAME = 'omr'


class OMRSignalWrapper(StrategySignals):
    """
    Wrapper to make OvernightReversionSignals compatible with StrategyAdapter.

    OvernightReversionSignals returns dicts, but StrategyAdapter expects Signal objects.
    This wrapper converts the dict-based signals to proper Signal objects.
    """

    def __init__(self, omr_signals: OvernightReversionSignals):
        self._omr_signals = omr_signals

    def get_required_lookback(self) -> int:
        """Return number of periods needed for signal generation."""
        return 1  # OMR only needs today's intraday data

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: Optional[datetime] = None
    ) -> List[Signal]:
        """
        Generate signals compatible with base StrategyAdapter.

        Args:
            market_data: Dict of symbol -> DataFrame with OHLCV data
            timestamp: Current timestamp

        Returns:
            List of Signal objects
        """
        now = timestamp or datetime.now()

        # Call the underlying OMR signal generator (returns list of dicts)
        raw_signals = self._omr_signals.generate_signals(market_data, now)

        # Convert dicts to Signal objects
        signals = []
        for raw in raw_signals:
            # Map 'SHORT' to 'SELL' for Signal compatibility
            direction = raw['direction']
            if direction == 'SHORT':
                direction = 'SELL'

            signals.append(Signal(
                timestamp=now,
                symbol=raw['symbol'],
                direction=direction,
                confidence=raw.get('signal_strength', raw.get('probability', 0.5)),
                price=raw.get('current_price', 0.01),  # Use current_price from signal
                metadata={
                    'regime': raw.get('regime'),
                    'intraday_return': raw.get('intraday_return'),
                    'probability': raw.get('probability'),
                    'expected_return': raw.get('expected_return'),
                    'sharpe': raw.get('sharpe'),
                    'sample_size': raw.get('sample_size'),
                    'entry_time': raw.get('entry_time'),
                    'exit_time': raw.get('exit_time')
                }
            ))

        return signals


class OMRLiveAdapter(StrategyAdapter):
    """
    Live trading adapter for Overnight Mean Reversion strategy.

    Generates signals at 3:50 PM EST based on:
    - Market regime
    - Intraday price movements
    - Bayesian reversion probabilities

    Positions are entered at 3:50 PM and exited next day at 9:31 AM.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbols: Optional[List[str]] = None,
        min_probability: float = 0.55,
        min_expected_return: float = 0.002,
        max_positions: int = 5,
        position_size: float = 0.1,
        regime_detector: Optional[MarketRegimeDetector] = None,
        bayesian_model: Optional[BayesianReversionModel] = None
    ):
        """
        Initialize OMR live adapter.

        Args:
            broker: Broker interface
            symbols: List of symbols to trade (default: leveraged 3x ETFs)
            min_probability: Min win rate threshold (default: 0.55)
            min_expected_return: Min expected return threshold (default: 0.002)
            max_positions: Max concurrent positions (default: 5)
            position_size: Position size as fraction (default: 0.1)
            regime_detector: Trained regime detector (optional)
            bayesian_model: Trained Bayesian model (optional)
        """
        # Use default symbols if not specified
        if symbols is None:
            symbols = ETFUniverse.LEVERAGED_3X
            logger.info(f"[OMR] Using default OMR universe: {len(symbols)} leveraged 3x ETFs")

        # Initialize regime detector if not provided
        if regime_detector is None:
            regime_detector = MarketRegimeDetector()
            logger.info("[OMR] Created new MarketRegimeDetector (untrained)")

        # Initialize Bayesian model if not provided
        if bayesian_model is None:
            bayesian_model = BayesianReversionModel()
            # Try to load pre-trained model from disk
            try:
                bayesian_model.load_model()
                model_symbols = set(bayesian_model.regime_probabilities.keys())
                trading_symbols = set(symbols)
                covered = trading_symbols & model_symbols
                missing = trading_symbols - model_symbols

                logger.success(f"[OMR] Loaded pre-trained Bayesian model")
                logger.info(f"[OMR]   Model covers {len(covered)}/{len(trading_symbols)} trading symbols")

                if missing:
                    logger.warning(f"[OMR]   Missing from model ({len(missing)}): {sorted(missing)}")
                    logger.warning("[OMR]   These symbols will not generate signals until model is retrained")

            except FileNotFoundError:
                logger.warning("[OMR] No pre-trained Bayesian model found - will train at market open")
                logger.warning(f"[OMR] Expected model at: {bayesian_model.model_path}")
            except Exception as e:
                logger.error(f"[OMR] Failed to load Bayesian model: {e}")
                logger.warning("[OMR] Will train at market open")

        # Create pure OMR strategy with injected symbols
        omr_signals = OvernightReversionSignals(
            regime_detector=regime_detector,
            bayesian_model=bayesian_model,
            symbols=symbols,  # ✅ Inject symbols instead of using hardcoded list
            min_probability=min_probability,
            min_expected_return=min_expected_return,
            max_positions=max_positions
        )

        # Wrap for compatibility with base adapter (converts dicts to Signal objects)
        strategy = OMRSignalWrapper(omr_signals)

        # OMR needs 252+ trading days for regime detection (VIX percentile)
        # 400 calendar days ≈ 274 trading days, safely above 252 requirement
        data_lookback_days = 400

        # Initialize base adapter
        super().__init__(
            strategy=strategy,
            broker=broker,
            symbols=symbols,
            position_size=position_size,
            max_positions=max_positions,
            data_lookback_days=data_lookback_days
        )

        self.min_probability = min_probability
        self.min_expected_return = min_expected_return

        # Store references for training
        self._bayesian_model = bayesian_model
        self._regime_detector = regime_detector

        # Initialize state manager for multi-strategy coordination
        self.state_manager = StrategyStateManager()

        # Initialize portfolio health checker with state manager for multi-strategy support
        self.health_checker = PortfolioHealthChecker(
            broker=broker,
            min_buying_power=1000.0,
            min_portfolio_value=5000.0,
            max_positions=max_positions,
            max_position_age_hours=48,
            state_manager=self.state_manager
        )

        logger.info("[OMR] Strategy Configuration:")
        logger.info(f"[OMR]   Min probability: {min_probability:.1%}")
        logger.info(f"[OMR]   Min expected return: {min_expected_return:.2%}")
        logger.info(f"[OMR]   Signal time: 3:50 PM EST")
        logger.info(f"[OMR]   Entry: 3:50 PM | Exit: Next day 9:31 AM")
        logger.info(f"[OMR]   Portfolio health checks: ENABLED")

    def preload_historical_data(self) -> None:
        """
        Pre-load historical data and train Bayesian model if needed.

        Extends parent method to:
        1. Fetch historical data for all symbols + SPY + VIX
        2. Train Bayesian model if not already trained
        """
        # Call parent to fetch historical data
        super().preload_historical_data()

        # Also fetch SPY and VIX for training if not in cache
        if self._data_cache is not None:
            end_date = tz.now()
            start_date = end_date - timedelta(days=self.data_lookback_days)

            for market_symbol in ['SPY', 'VIX']:
                if market_symbol not in self._data_cache:
                    try:
                        if market_symbol == 'VIX':
                            df = self._fetch_vix_yfinance(start_date, end_date)
                        else:
                            df = self.broker.get_historical_bars(
                                symbol=market_symbol,
                                start=start_date,
                                end=end_date,
                                timeframe='1D'
                            )

                        if df is not None and not df.empty:
                            self._data_cache[market_symbol] = df
                            logger.info(f"[OMR] Fetched {market_symbol}: {len(df)} days")
                    except Exception as e:
                        logger.error(f"[OMR] Error fetching {market_symbol}: {e}")

        # Train Bayesian model if not already trained
        if not self._bayesian_model.trained:
            self._train_bayesian_model()

    def _train_bayesian_model(self) -> None:
        """Train the Bayesian model using cached historical data."""
        if self._data_cache is None or len(self._data_cache) == 0:
            logger.error("[OMR] Cannot train Bayesian model: no historical data available")
            return

        if 'SPY' not in self._data_cache or 'VIX' not in self._data_cache:
            logger.error("[OMR] Cannot train Bayesian model: missing SPY or VIX data")
            return

        try:
            logger.info("[OMR] Training Bayesian model with historical data...")

            # Prepare data for training (need daily OHLCV)
            spy_data = self._data_cache['SPY']
            vix_data = self._data_cache['VIX']

            # Normalize column names to lowercase for consistency
            historical_data = {}
            for symbol, df in self._data_cache.items():
                df_copy = df.copy()
                df_copy.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df_copy.columns]
                historical_data[symbol] = df_copy

            # Also normalize SPY and VIX
            spy_normalized = spy_data.copy()
            spy_normalized.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in spy_normalized.columns]

            vix_normalized = vix_data.copy()
            vix_normalized.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in vix_normalized.columns]

            # Train the model
            self._bayesian_model.train(
                historical_data=historical_data,
                regime_detector=self._regime_detector,
                spy_data=spy_normalized,
                vix_data=vix_normalized
            )

            logger.success(f"[OMR] Bayesian model trained on {len(historical_data)} symbols")

        except Exception as e:
            logger.error(f"[OMR] Failed to train Bayesian model: {e}")
            import traceback
            traceback.print_exc()

    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday market data for OMR strategy.

        OMR needs intraday bars to calculate intraday moves.
        Uses intraday cache refreshed at execution time (3:50 PM).
        """
        try:
            import pandas as pd
            from datetime import timedelta

            market_data = {}
            end_date = tz.now()
            start_date = end_date - timedelta(days=self.data_lookback_days)

            # Check if intraday cache is available
            intraday_cache_available = (
                self._intraday_cache is not None and
                len(self._intraday_cache) > 0
            )

            if intraday_cache_available:
                logger.info("[OMR] Using pre-fetched intraday data cache")

                # Use cached intraday data for symbols
                for symbol in self.symbols:
                    if symbol in self._intraday_cache:
                        market_data[symbol] = self._intraday_cache[symbol]
                    else:
                        logger.warning(f"[OMR] {symbol} not in intraday cache, fetching...")
                        # Fall back to live fetch
                        try:
                            df = self.broker.get_historical_bars(
                                symbol=symbol,
                                start=end_date.replace(hour=9, minute=30, second=0),
                                end=end_date,
                                timeframe='1Min'
                            )
                            if df is not None and not df.empty:
                                market_data[symbol] = df
                        except Exception as e:
                            logger.error(f"[OMR] Error fetching {symbol}: {e}")

            else:
                logger.info("[OMR] No intraday cache available, fetching data...")

                # For OMR, we need intraday data (1-minute bars)
                market_open_today = end_date.replace(hour=9, minute=30, second=0, microsecond=0)

                for symbol in self.symbols:
                    try:
                        # Fetch intraday data from market open to now
                        df = self.broker.get_historical_bars(
                            symbol=symbol,
                            start=market_open_today,
                            end=end_date,
                            timeframe='1Min'
                        )

                        if df is not None and not df.empty:
                            market_data[symbol] = df
                        else:
                            logger.warning(f"[OMR] No intraday data returned for {symbol}")

                    except Exception as e:
                        logger.error(f"[OMR] Error fetching data for {symbol}: {e}")
                        continue

            # Also need historical daily data for regime detection
            # Use base class cache if available
            if self._data_cache is not None:
                logger.info("[OMR] Using cached historical data for regime detection")
                for market_symbol in ['SPY', 'VIX']:
                    if market_symbol in self._data_cache:
                        market_data[market_symbol] = self._data_cache[market_symbol]
            else:
                # Fetch historical data for SPY and VIX
                for market_symbol in ['SPY', 'VIX']:
                    if market_symbol not in market_data:
                        try:
                            # VIX is not available from Alpaca - use yfinance directly
                            if market_symbol == 'VIX':
                                logger.info("[OMR] Fetching VIX data via yfinance (Alpaca does not provide VIX)")
                                df = self._fetch_vix_yfinance(start_date, end_date)
                                if df is not None and not df.empty:
                                    market_data[market_symbol] = df
                                    logger.info(f"[OMR] Fetched {len(df)} days of VIX data via yfinance")
                            else:
                                # Use Alpaca for other symbols (SPY, etc.)
                                df = self.broker.get_historical_bars(
                                    symbol=market_symbol,
                                    start=start_date,
                                    end=end_date,
                                    timeframe='1D'
                                )
                                if df is not None and not df.empty:
                                    market_data[market_symbol] = df
                        except Exception as e:
                            logger.error(f"[OMR] Error fetching {market_symbol}: {e}")
                            # VIX errors are already handled in _fetch_vix_yfinance

            cache_status = "cached intraday" if intraday_cache_available else "live fetch"
            logger.info(
                f"[OMR] Fetched data for {len(market_data)} symbols ({cache_status})"
            )

            # Normalize column names to lowercase for consistency
            # (yfinance returns 'Close', Alpaca returns 'close', etc.)
            normalized_data = {}
            for symbol, df in market_data.items():
                df_copy = df.copy()
                # Handle both single-level and multi-level column names
                if hasattr(df_copy.columns, 'levels'):
                    # Multi-level columns (e.g., from yfinance with multiple tickers)
                    df_copy.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_copy.columns]
                else:
                    df_copy.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df_copy.columns]
                normalized_data[symbol] = df_copy

            return normalized_data

        except Exception as e:
            logger.error(f"[OMR] Error in fetch_market_data: {e}")
            return {}

    def _fetch_vix_yfinance(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch VIX data via yfinance.

        Alpaca doesn't provide VIX data, so we use yfinance as a fallback.
        The VIX ticker on Yahoo Finance is ^VIX.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with VIX data in OHLCV format, or None if fetch fails
        """
        try:
            # Convert string dates to pandas Timestamps and add buffer
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date) + timedelta(days=1)

            # Fetch VIX data using Yahoo Finance ticker ^VIX
            vix_data = yf.download(
                '^VIX',
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True  # Suppress FutureWarning
            )

            if vix_data is None or vix_data.empty:
                logger.error("[OMR] yfinance returned empty VIX data")
                return None

            # Ensure timezone-aware index for consistency with broker data
            if vix_data.index.tz is None:
                vix_data.index = vix_data.index.tz_localize('America/New_York')
            else:
                vix_data.index = vix_data.index.tz_convert('America/New_York')

            return vix_data

        except Exception as e:
            logger.error(f"[OMR] Failed to fetch VIX via yfinance: {e}")
            return None

    def execute_signals(self, signals: List[Signal]) -> None:
        """
        Execute trading signals with position tracking.

        Overrides base class to add state manager position tracking
        for multi-strategy coordination.

        Args:
            signals: Filtered signals to execute
        """
        if not signals:
            logger.info("[OMR] No signals to execute")
            return

        # Get account info for position sizing
        account = self.broker.get_account()
        if account is None:
            logger.error("[OMR] Cannot get account info, skipping execution")
            return

        buying_power = float(account['buying_power'])

        for signal in signals:
            try:
                # Calculate position size
                position_value = buying_power * self.position_size
                qty = int(position_value / signal.price)

                if qty <= 0:
                    logger.warning(
                        f"[OMR] Calculated qty {qty} for {signal.symbol}, skipping"
                    )
                    continue

                # Execute order
                logger.info(
                    f"[OMR] Executing {signal.direction} {qty} shares of {signal.symbol} "
                    f"@ ${signal.price:.2f}"
                )

                if signal.direction == 'BUY':
                    side = OrderSide.BUY
                elif signal.direction == 'SELL':
                    side = OrderSide.SELL
                else:
                    logger.warning(f"[OMR] Unknown direction: {signal.direction}")
                    continue

                order = self.execution_engine.execute_order(
                    symbol=signal.symbol,
                    quantity=qty,
                    side=side,
                    order_type=OrderType.MARKET
                )

                if order:
                    logger.success(f"[OMR] Order placed: {order.get('order_id', 'UNKNOWN')}")
                    # Track position in state manager for multi-strategy coordination
                    # Use add_or_update_position to safely handle any edge cases
                    order_id = order.get('order_id')
                    self.state_manager.add_or_update_position(
                        STRATEGY_NAME, signal.symbol, qty, signal.price, order_id
                    )
                else:
                    logger.error(f"[OMR] Failed to place order for {signal.symbol}")

            except Exception as e:
                logger.error(f"[OMR] Error executing signal for {signal.symbol}: {e}")
                continue

    def run_once(self) -> None:
        """
        Run one iteration of the strategy with portfolio health checks.

        Overrides base class to add:
        - Fresh intraday data fetch at execution time (3:50 PM)
        - Pre-entry health validation
        - Execution lock for multi-strategy coordination
        - Position tracking per strategy
        """
        logger.info("[OMR] " + "=" * 60)
        logger.info(f"[OMR] Running {self.__class__.__name__} at {tz.now()}")
        logger.info("[OMR] " + "=" * 60)

        try:
            # Check if strategy is enabled
            if not self.state_manager.is_enabled(STRATEGY_NAME):
                logger.warning("[OMR] Strategy is DISABLED - skipping execution")
                return

            # Check if shutdown requested
            if self.state_manager.is_shutdown_requested(STRATEGY_NAME):
                logger.warning("[OMR] Shutdown requested - skipping new entries")
                return

            # Refresh intraday data at 3:50 PM execution time
            logger.info("[OMR] Refreshing intraday data for 3:50 PM execution...")
            self.prefetch_intraday_data()

            # Acquire execution lock (blocks if another strategy is executing)
            if not self.state_manager.acquire_execution_lock(STRATEGY_NAME):
                logger.error("[OMR] Failed to acquire execution lock - another strategy is running")
                return

            try:
                # Sync state with broker (detect external position changes)
                broker_positions = {p['symbol']: int(p['quantity']) for p in self.broker.get_positions()}
                changes = self.state_manager.sync_with_broker(broker_positions)
                if changes['removed']:
                    logger.info(f"[OMR] Detected closed positions: {changes['removed']}")

                # CRITICAL: Portfolio health check before entry
                # Use strategy_name='omr' to only count OMR positions for max_positions check
                logger.info("[OMR] Running pre-entry portfolio health check...")
                health_result = self.health_checker.check_before_entry(
                    required_capital=None,
                    allow_existing_positions=True,
                    strategy_name='omr'
                )

                if not health_result.passed:
                    logger.error("[OMR] Portfolio health check FAILED - BLOCKING ENTRY")
                    for error in health_result.errors:
                        logger.error(f"[OMR]   - {error}")
                    return

                if health_result.warnings:
                    logger.warning("[OMR] Portfolio health check passed with warnings:")
                    for warning in health_result.warnings:
                        logger.warning(f"[OMR]   - {warning}")

                logger.success("[OMR] Portfolio health check PASSED - proceeding with entry")
                logger.info("")

                # Call parent's run_once() for normal strategy execution
                super().run_once()

                # Update last execution timestamp
                self.state_manager.update_last_execution(STRATEGY_NAME)

            finally:
                # Always release execution lock
                self.state_manager.release_execution_lock(STRATEGY_NAME)

        except Exception as e:
            logger.error(f"[OMR] Error in run_once: {e}")
            import traceback
            traceback.print_exc()

    def get_schedule(self) -> Dict[str, any]:
        """
        Get scheduling configuration.

        OMR requires TWO execution times:
        - 3:50 PM EST: Generate signals and enter positions
        - 9:31 AM EST: Close overnight positions

        Returns:
            Schedule dict with entry and exit times
        """
        return {
            'execution_times': [
                {'time': '15:50', 'action': 'entry'},   # 3:50 PM - Enter positions
                {'time': '09:31', 'action': 'exit'}     # 9:31 AM - Exit positions
            ],
            'market_hours_only': True,
            'strategy_type': 'overnight'  # Indicates overnight holding
        }

    def close_overnight_positions(self) -> None:
        """
        Close overnight positions at market open (9:31 AM).

        Should be called at 9:31 AM to exit positions entered at 3:50 PM.
        Uses execution lock for multi-strategy coordination.
        """
        try:
            now = tz.now()
            if now.time() < time(9, 30) or now.time() > time(9, 35):
                logger.warning(
                    f"[OMR] close_overnight_positions called at {now.time()}, "
                    "expected 9:31 AM"
                )

            # Acquire execution lock for closing
            if not self.state_manager.acquire_execution_lock(STRATEGY_NAME):
                logger.error("[OMR] Failed to acquire execution lock for closing positions")
                # Still attempt to close - safety is more important than coordination
                logger.warning("[OMR] Proceeding with close despite lock failure (safety priority)")

            try:
                # CRITICAL: Portfolio health check before exit
                logger.info("[OMR] Running pre-exit portfolio health check...")
                health_result = self.health_checker.check_before_exit()

                if not health_result.passed:
                    logger.error("[OMR] Portfolio health check FAILED - CRITICAL ERRORS DETECTED")
                    for error in health_result.errors:
                        logger.error(f"[OMR]   - {error}")
                    logger.warning("[OMR] Attempting to close positions despite errors (safety measure)")

                if health_result.warnings:
                    logger.warning("[OMR] Portfolio health check warnings:")
                    for warning in health_result.warnings:
                        logger.warning(f"[OMR]   - {warning}")

                # Get OMR's tracked positions from state manager
                omr_positions = self.state_manager.get_positions(STRATEGY_NAME)

                # Get all broker positions
                broker_positions = self.broker.get_positions()

                if not broker_positions:
                    logger.info("[OMR] No overnight positions to close")
                    return

                # Filter to only close OMR's positions (not MP's)
                positions_to_close = []
                for pos in broker_positions:
                    symbol = pos['symbol']
                    # Close if it's tracked by OMR OR if it's a leveraged ETF (OMR's universe)
                    if symbol in omr_positions or ETFUniverse.is_leveraged(symbol):
                        positions_to_close.append(pos)
                    else:
                        # Check if another strategy owns it
                        owner = self.state_manager.symbol_owned_by_other(STRATEGY_NAME, symbol)
                        if owner:
                            logger.info(f"[OMR] Skipping {symbol} - owned by {owner}")

                logger.info(f"[OMR] Closing {len(positions_to_close)} overnight positions at market open")

                for position in positions_to_close:
                    try:
                        symbol = position['symbol']
                        entry_price = float(position['avg_entry_price'])
                        current_price = float(position['current_price'])
                        qty = int(position['quantity'])

                        pnl = (current_price - entry_price) * qty
                        pnl_pct = (current_price - entry_price) / entry_price * 100

                        logger.info(
                            f"[OMR] Closing {symbol}: {qty} shares "
                            f"@ ${entry_price:.2f} -> ${current_price:.2f} "
                            f"(P&L: ${pnl:+.2f}, {pnl_pct:+.2f}%)"
                        )

                        side = OrderSide.SELL if qty > 0 else OrderSide.BUY
                        order = self.execution_engine.execute_order(
                            symbol=symbol,
                            quantity=abs(qty),
                            side=side,
                            order_type=OrderType.MARKET
                        )

                        if order:
                            logger.success(f"[OMR] Close order placed: {order.get('order_id', 'UNKNOWN')}")
                            # Remove from state tracking
                            self.state_manager.remove_position(STRATEGY_NAME, symbol)
                        else:
                            logger.error(f"[OMR] Failed to close {symbol}")

                    except Exception as e:
                        logger.error(f"[OMR] Error closing {position.get('symbol', 'UNKNOWN')}: {e}")
                        continue

                logger.info("[OMR] Overnight position closing complete")

            finally:
                self.state_manager.release_execution_lock(STRATEGY_NAME)

        except Exception as e:
            logger.error(f"[OMR] Error in close_overnight_positions: {e}")


if __name__ == "__main__":
    logger.info("[OMR] Overnight Mean Reversion Live Trading Adapter")
    logger.info("[OMR] " + "=" * 60)
    logger.info("[OMR] Generates signals at 3:50 PM EST based on:")
    logger.info("[OMR]   - Market regime (bull/bear/choppy)")
    logger.info("[OMR]   - Intraday price movements")
    logger.info("[OMR]   - Bayesian reversion probabilities")
    logger.info("")
    logger.info("[OMR] Entry: 3:50 PM EST")
    logger.info("[OMR] Exit: Next day 9:31 AM EST")
    logger.info("")
    logger.info("[OMR] Default universe: Leveraged 3x ETFs")
    logger.info("[OMR]   (TQQQ, SQQQ, UPRO, SPXU, TMF, TMV, etc.)")
