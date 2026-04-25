"""
Overnight Mean Reversion (OMR) Live Trading Adapter.

Connects OMR strategy to live trading infrastructure.
Runs at 3:50 PM EST to generate overnight signals.
"""

from typing import List, Dict, Optional, TYPE_CHECKING, Union
from datetime import datetime, time, timedelta
import pandas as pd

from src.trading.adapters.strategy_adapter import StrategyAdapter

if TYPE_CHECKING:
    from src.data.providers.base import DataProviderInterface
    from src.streaming.live_data_provider import LiveDataProvider
from src.streaming.interface import StreamingProviderInterface
from src.strategies.core import StrategySignals, Signal
from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
from src.strategies.universe import ETFUniverse
from src.trading.brokers.broker_interface import BrokerInterface, OrderSide, OrderType
from src.trading.utils.portfolio_health_check import PortfolioHealthChecker
from src.trading.state import StrategyStateManager
from src.utils.vix_provider import get_vix_provider
from src.utils.logger import logger
from src.utils.timezone import tz
from src.utils.trading_logger import get_trade_log_writer

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

    # Class-level constants used by StrategyAdapter base helpers
    STRATEGY_NAME: str = 'omr'
    STRATEGY_VERSION: int = 1

    # Persists across entry -> exit within one session so the exit record
    # can reference its corresponding entry via parent_decision_id.
    _last_entry_decision_id: Optional[str] = None

    def __init__(
        self,
        broker: BrokerInterface,
        symbols: Optional[List[str]] = None,
        min_probability: float = 0.55,
        min_expected_return: float = 0.002,
        max_positions: int = 5,
        position_size: float = 0.1,
        regime_detector: Optional[MarketRegimeDetector] = None,
        bayesian_model: Optional[BayesianReversionModel] = None,
        data_provider: Optional[Union["DataProviderInterface", "LiveDataProvider"]] = None,
        max_capital_usd: Optional[float] = None,
        *,
        broker_name: str,
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
            data_provider: Data provider - supports both DataProviderInterface (polling)
                          and LiveDataProvider (streaming). If LiveDataProvider, uses
                          real-time WebSocket data. If not provided, falls back to broker.
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
        self.max_capital_usd = max_capital_usd
        if max_capital_usd is not None:
            logger.info(f"[OMR] Capital cap: ${max_capital_usd:,.0f} per strategy")

        # Store references for training
        self._bayesian_model = bayesian_model
        self._regime_detector = regime_detector

        # Store data provider for fetching with fallback
        self._data_provider = data_provider
        if data_provider is not None:
            logger.info(f"[OMR] Using data provider: {data_provider.name}")

        # Broker identifier used when tagging positions in the shared state file
        self._broker_name = broker_name

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

    @property
    def broker_name(self) -> str:
        return self._broker_name

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
                            # Use VIX provider with fallback chain
                            df = self._fetch_vix_data(lookback_days=self.data_lookback_days)
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

    def _fetch_from_cache(
        self,
        market_open_today: datetime,
        end_date: datetime
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Try to fetch intraday data from the pre-fetched intraday cache.

        Args:
            market_open_today: Market open time today
            end_date: Current timestamp

        Returns:
            Dict of symbol -> DataFrame if cache is available, None otherwise
        """
        if self._intraday_cache is None or len(self._intraday_cache) == 0:
            return None

        logger.info("[OMR] Using pre-fetched intraday data cache")
        market_data = {}

        for symbol in self.symbols:
            if symbol in self._intraday_cache:
                market_data[symbol] = self._intraday_cache[symbol]
            else:
                logger.warning(f"[OMR] {symbol} not in intraday cache, fetching...")
                df = self._fetch_intraday_symbol(symbol, market_open_today, end_date)
                if df is not None and not df.empty:
                    market_data[symbol] = df

        return market_data

    def _fetch_from_streaming(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Try to fetch intraday data from the LiveDataProvider streaming buffer.

        Returns:
            Dict of symbol -> DataFrame if streaming is available, None otherwise
        """
        if self._data_provider is None or not isinstance(self._data_provider, StreamingProviderInterface):
            return None

        logger.info(f"[OMR] Fetching intraday data from LiveDataProvider (streaming)...")
        market_data = {}

        for symbol in self.symbols:
            try:
                bars_df = self._data_provider.get_bars(symbol, n=390)

                if bars_df is not None and not bars_df.empty:
                    bars_count = len(bars_df)
                    expected_bars = 390
                    data_quality = bars_count / expected_bars if expected_bars > 0 else 0

                    if data_quality < 0.9:
                        logger.warning(
                            f"[OMR] {symbol} has {bars_count}/{expected_bars} bars ({data_quality:.1%}). "
                            f"Streaming buffer may be incomplete (recent restart?)."
                        )
                    elif data_quality < 1.0:
                        logger.info(
                            f"[OMR] {symbol} has {bars_count}/{expected_bars} bars ({data_quality:.1%})"
                        )
                    else:
                        logger.debug(f"[OMR] {symbol} has {bars_count} bars (complete)")

                    market_data[symbol] = bars_df
                else:
                    logger.warning(f"[OMR] No bars in buffer for {symbol}")
            except Exception as e:
                logger.error(f"[OMR] Error getting bars from provider for {symbol}: {e}")
                continue

        logger.info(f"[OMR] Retrieved {len(market_data)} symbols from streaming buffer")
        return market_data

    def _fetch_from_polling(
        self,
        market_open_today: datetime,
        end_date: datetime
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Try to fetch intraday data via DataProviderInterface polling with fallback.

        Args:
            market_open_today: Market open time today
            end_date: Current timestamp

        Returns:
            Dict of symbol -> DataFrame if polling provider is available, None otherwise
        """
        if self._data_provider is None or not hasattr(self._data_provider, 'get_historical_bars_batch'):
            return None

        logger.info(f"[OMR] Fetching intraday data via {self._data_provider.name} provider...")
        market_data = self._data_provider.get_historical_bars_batch(
            self.symbols, market_open_today, end_date, timeframe='1Min',
            force_refresh=True
        )
        return market_data

    def _fetch_from_broker(
        self,
        market_open_today: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday data directly from broker (last-resort fallback).

        Args:
            market_open_today: Market open time today
            end_date: Current timestamp

        Returns:
            Dict of symbol -> DataFrame
        """
        logger.info("[OMR] No data provider, fetching from broker...")
        market_data = {}

        for symbol in self.symbols:
            try:
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

        return market_data

    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday market data for OMR strategy.

        OMR needs intraday bars to calculate intraday moves.

        Data source priority:
        1. Pre-fetched intraday cache (if available)
        2. LiveDataProvider streaming (instant from buffer)
        3. DataProviderInterface polling (with fallback chain)
        4. Broker direct (original behavior)
        """
        try:
            market_data: Dict[str, pd.DataFrame] = {}
            end_date = tz.now()
            start_date = end_date - timedelta(days=self.data_lookback_days)
            market_open_today = end_date.replace(hour=9, minute=30, second=0, microsecond=0)

            intraday_cache_used = (
                self._intraday_cache is not None and
                len(self._intraday_cache) > 0
            )

            # Try each data source in priority order
            result = self._fetch_from_cache(market_open_today, end_date)
            if result is None:
                result = self._fetch_from_streaming()
            if result is None:
                result = self._fetch_from_polling(market_open_today, end_date)
            if result is None:
                result = self._fetch_from_broker(market_open_today, end_date)

            market_data = result if result is not None else {}

            # Also need historical daily data for regime detection
            if self._data_cache is not None:
                logger.info("[OMR] Using cached historical data for regime detection")
                for market_symbol in ['SPY', 'VIX']:
                    if market_symbol in self._data_cache:
                        market_data[market_symbol] = self._data_cache[market_symbol]
            else:
                for market_symbol in ['SPY', 'VIX']:
                    if market_symbol not in market_data:
                        try:
                            if market_symbol == 'VIX':
                                logger.info("[OMR] Fetching VIX data with fallback chain...")
                                df = self._fetch_vix_data(lookback_days=self.data_lookback_days)
                                if df is not None and not df.empty:
                                    market_data[market_symbol] = df
                            else:
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

            cache_status = "cached intraday" if intraday_cache_used else "live fetch"
            logger.info(
                f"[OMR] Fetched data for {len(market_data)} symbols ({cache_status})"
            )

            # Normalize column names to lowercase for consistency
            normalized_data = {}
            for symbol, df in market_data.items():
                df_copy = df.copy()
                if hasattr(df_copy.columns, 'levels'):
                    df_copy.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_copy.columns]
                else:
                    df_copy.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df_copy.columns]
                normalized_data[symbol] = df_copy

            return normalized_data

        except Exception as e:
            logger.error(f"[OMR] Error in fetch_market_data: {e}")
            return {}

    def _fetch_vix_data(self, lookback_days: int = 400) -> Optional[pd.DataFrame]:
        """
        Fetch VIX data with multi-source fallback chain.

        Uses VIXProvider which tries:
        1. yfinance (primary) - Yahoo Finance ^VIX
        2. FRED API (fallback) - Federal Reserve VIXCLS series
        3. Persisted cache (last resort) - Last known good VIX value

        Args:
            lookback_days: Number of days of history needed

        Returns:
            DataFrame with VIX data ('close' column), or None if all sources fail
        """
        try:
            vix_provider = get_vix_provider()
            vix_data = vix_provider.get_vix_data(lookback_days=lookback_days)

            if vix_data is not None:
                source, fetch_time = vix_provider.get_source_info()
                logger.info(f"[OMR] VIX data from {source}: {len(vix_data)} days")

                # Log warning if using cached data
                if source == "cache":
                    logger.warning(f"[OMR] Using cached VIX data (may be stale)")

                return vix_data
            else:
                logger.error("[OMR] All VIX data sources failed!")
                return None

        except Exception as e:
            logger.error(f"[OMR] Failed to fetch VIX data: {e}")
            return None

    # Keep old method name as alias for backward compatibility
    def _fetch_vix_yfinance(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Deprecated: Use _fetch_vix_data() instead. This is kept for compatibility."""
        logger.warning("[OMR] _fetch_vix_yfinance is deprecated, using _fetch_vix_data with fallback chain")
        return self._fetch_vix_data(lookback_days=self.data_lookback_days)

    def _fetch_intraday_symbol(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday data for a single symbol using provider or broker.

        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with intraday bars, or None on failure
        """
        try:
            if self._data_provider is not None:
                df = self._data_provider.get_historical_bars(symbol, start, end, '1Min')
                if df is not None and not df.empty:
                    return df

            # Fall back to broker
            df = self.broker.get_historical_bars(
                symbol=symbol,
                start=start,
                end=end,
                timeframe='1Min'
            )
            return df

        except Exception as e:
            logger.error(f"[OMR] Error fetching intraday data for {symbol}: {e}")
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
        sizing_base = min(buying_power, self.max_capital_usd) if self.max_capital_usd else buying_power

        for signal in signals:
            try:
                # Calculate position size
                position_value = sizing_base * self.position_size
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
                        STRATEGY_NAME, signal.symbol, qty, signal.price, order_id,
                        broker=self._broker_name,
                    )

                    # Log trade entry to persistent trade log
                    # Error handling ensures logging failures don't block trading
                    try:
                        trade_logger = get_trade_log_writer()
                        fill_price = order.get('filled_avg_price', signal.price)
                        trade_logger.log_entry(
                            strategy=STRATEGY_NAME,
                            symbol=signal.symbol,
                            qty=qty,
                            price=float(fill_price) if fill_price else signal.price,
                            order_id=order_id,
                            metadata={
                                'probability': signal.metadata.get('probability') if signal.metadata else None,
                                'expected_return': signal.metadata.get('expected_return') if signal.metadata else None
                            },
                            account_snapshot=self.broker.get_account()
                        )
                    except Exception as log_err:
                        logger.error(f"[OMR] Trade logging failed (non-blocking): {log_err}")
                else:
                    logger.error(f"[OMR] Failed to place order for {signal.symbol}")

            except Exception as e:
                logger.error(f"[OMR] Error executing signal for {signal.symbol}: {e}")
                continue

    def run_once(self, action: str = "entry") -> None:
        """Run one iteration: entry (15:50 ET) or exit (09:31 ET).

        Both paths emit a DecisionRecord.  The exit record references the
        preceding entry via parent_decision_id.

        Args:
            action: "entry" to enter overnight positions (default),
                    "exit" to close overnight positions at market open.
        """
        if action == "exit":
            self._run_exit()
        else:
            self._run_entry()

    def _run_entry(self) -> None:
        """Entry path: generate signals and enter overnight positions (15:50 ET).

        Emits a DecisionRecord with trigger.kind="scheduled_entry".
        Stores the decision_id so the subsequent exit record can link back.
        """
        import traceback as _tb

        logger.info("[OMR] " + "=" * 60)
        logger.info(f"[OMR] Running entry at {tz.now()}")
        logger.info("[OMR] " + "=" * 60)

        rec = self._begin_decision("scheduled_entry", schedule_time="15:50")
        had_lock = False
        try:
            # ---- Preconditions ----
            with self._stage(rec, "preconditions"):
                from src.trading.decision_log.record import GateResult as _GR

                # Check enabled/shutdown BEFORE acquiring lock (preserves original OMR behavior)
                rec.preconditions.strategy_enabled = _GR(
                    passed=self.state_manager.is_enabled(STRATEGY_NAME), details={}
                )
                if not rec.preconditions.strategy_enabled.passed:
                    logger.warning("[OMR] Strategy is DISABLED - skipping execution")
                    return

                rec.preconditions.shutdown_requested = _GR(
                    passed=not self.state_manager.is_shutdown_requested(STRATEGY_NAME), details={}
                )
                if not rec.preconditions.shutdown_requested.passed:
                    logger.warning("[OMR] Shutdown requested - skipping new entries")
                    return

                # Now acquire the execution lock
                rec.preconditions.execution_lock_acquired = _GR(
                    passed=self.state_manager.acquire_execution_lock(STRATEGY_NAME), details={}
                )
                if not rec.preconditions.execution_lock_acquired.passed:
                    logger.error("[OMR] Failed to acquire execution lock - another strategy is running")
                    return
                had_lock = True

                # Sync state with broker
                broker_positions = {p['symbol']: int(p['quantity']) for p in self.broker.get_positions()}
                changes = self.state_manager.sync_with_broker(self._broker_name, broker_positions)
                if changes['removed']:
                    logger.info(f"[OMR] Detected closed positions: {changes['removed']}")

                # Health check gate
                from src.trading.decision_log.record import GateResult
                logger.info("[OMR] Running pre-entry portfolio health check...")
                health_result = self.health_checker.check_before_entry(
                    required_capital=None,
                    allow_existing_positions=True,
                    strategy_name=STRATEGY_NAME,
                )
                rec.preconditions.health_check = GateResult(
                    passed=health_result.passed,
                    details={"warnings_count": len(health_result.warnings)},
                    error="; ".join(health_result.errors) if health_result.errors else None,
                )
                if not health_result.passed:
                    logger.error("[OMR] Portfolio health check FAILED - BLOCKING ENTRY")
                    for err in health_result.errors:
                        logger.error(f"[OMR]   - {err}")
                    return

                if health_result.warnings:
                    logger.warning("[OMR] Portfolio health check passed with warnings:")
                    for warning in health_result.warnings:
                        logger.warning(f"[OMR]   - {warning}")

                logger.success("[OMR] Portfolio health check PASSED - proceeding with entry")

                # Data freshness gate (intraday cache presence is a proxy)
                from src.trading.decision_log.record import GateResult as _GR
                intraday_ready = (
                    self._intraday_cache is not None and len(self._intraday_cache) > 0
                )
                rec.preconditions.data_freshness = _GR(
                    passed=True,  # not blocking; OMR refreshes inside inputs stage
                    details={"intraday_cache_populated": intraday_ready},
                )

            # ---- Inputs ----
            with self._stage(rec, "inputs"):
                logger.info("[OMR] Refreshing intraday data for 3:50 PM execution...")
                self.prefetch_intraday_data()
                rec.inputs = self._build_decision_inputs()

            # ---- Logic ----
            with self._stage(rec, "logic"):
                market_data = self.fetch_market_data()
                signals = self.strategy.generate_signals(market_data, tz.now())
                rec.logic_decisions = self._build_decision_logic(signals)

            # ---- Execution ----
            with self._stage(rec, "execution"):
                self.execute_signals(signals)

            # ---- Post-state ----
            with self._stage(rec, "post_state"):
                rec.post_state = self._snapshot_post_state()
                self.state_manager.update_last_execution(STRATEGY_NAME)
                # Store for the subsequent exit record to link back
                self._last_entry_decision_id = rec.decision_id

        except Exception as e:
            from src.trading.decision_log.record import ErrorInfo
            rec.error = ErrorInfo(
                type=type(e).__name__,
                message=str(e),
                traceback=_tb.format_exc(),
                stage=getattr(rec, "_current_stage", "unknown"),
            )
            logger.error(f"[OMR] Error in _run_entry: {e}")
            _tb.print_exc()
            raise
        finally:
            try:
                if had_lock:
                    self.state_manager.release_execution_lock(STRATEGY_NAME)
            finally:
                self._write_decision(rec)

    def _run_exit(self) -> None:
        """Exit path: close overnight positions at market open (09:31 ET).

        Emits a DecisionRecord with trigger.kind="scheduled_exit".
        Links back to the preceding entry via parent_decision_id.
        """
        import traceback as _tb

        logger.info("[OMR] " + "=" * 60)
        logger.info(f"[OMR] Running exit at {tz.now()}")
        logger.info("[OMR] " + "=" * 60)

        rec = self._begin_decision("scheduled_exit", schedule_time="09:31")
        # Link to the prior entry if we have one
        if self._last_entry_decision_id is not None:
            rec.parent_decision_id = self._last_entry_decision_id

        had_lock = False
        try:
            now = tz.now()
            if now.time() < time(9, 30) or now.time() > time(9, 35):
                logger.warning(
                    f"[OMR] _run_exit called at {now.time()}, expected 09:31"
                )

            # ---- Preconditions ----
            with self._stage(rec, "preconditions"):
                from src.trading.decision_log.record import GateResult as _GR

                # Check enabled/shutdown BEFORE acquiring lock (safety: still close if enabled)
                rec.preconditions.strategy_enabled = _GR(
                    passed=self.state_manager.is_enabled(STRATEGY_NAME), details={}
                )
                # Note: for exit we do NOT block on disabled - safety requires closing positions
                # We still record the gate but proceed regardless

                rec.preconditions.shutdown_requested = _GR(
                    passed=not self.state_manager.is_shutdown_requested(STRATEGY_NAME), details={}
                )

                # Acquire lock - proceed even if it fails (safety priority for exit)
                lock_acquired = self.state_manager.acquire_execution_lock(STRATEGY_NAME)
                rec.preconditions.execution_lock_acquired = _GR(
                    passed=lock_acquired, details={}
                )
                if not lock_acquired:
                    logger.error("[OMR] Failed to acquire execution lock for closing positions")
                    logger.warning("[OMR] Proceeding with close despite lock failure (safety priority)")
                else:
                    had_lock = True

                # Health check (non-blocking for exit - safety first)
                from src.trading.decision_log.record import GateResult
                logger.info("[OMR] Running pre-exit portfolio health check...")
                health_result = self.health_checker.check_before_exit()
                rec.preconditions.health_check = GateResult(
                    passed=health_result.passed,
                    details={"warnings_count": len(health_result.warnings)},
                    error="; ".join(health_result.errors) if health_result.errors else None,
                )
                if not health_result.passed:
                    logger.error("[OMR] Portfolio health check FAILED - CRITICAL ERRORS DETECTED")
                    for err in health_result.errors:
                        logger.error(f"[OMR]   - {err}")
                    logger.warning("[OMR] Attempting to close positions despite errors (safety measure)")

                if health_result.warnings:
                    logger.warning("[OMR] Portfolio health check warnings:")
                    for warning in health_result.warnings:
                        logger.warning(f"[OMR]   - {warning}")

                from src.trading.decision_log.record import GateResult as _GR
                rec.preconditions.data_freshness = _GR(passed=True, details={})

            # ---- Inputs ----
            with self._stage(rec, "inputs"):
                rec.inputs = self._build_decision_inputs()

            # ---- Logic / Execution ----
            with self._stage(rec, "logic"):
                omr_positions = self.state_manager.get_positions(STRATEGY_NAME)
                broker_positions = self.broker.get_positions()

                positions_to_close = []
                if broker_positions:
                    for pos in broker_positions:
                        symbol = pos['symbol']
                        if symbol in omr_positions or ETFUniverse.is_leveraged(symbol):
                            positions_to_close.append(pos)
                        else:
                            owner = self.state_manager.symbol_owned_by_other(STRATEGY_NAME, symbol)
                            if owner:
                                logger.info(f"[OMR] Skipping {symbol} - owned by {owner}")

                close_symbols = [p['symbol'] for p in positions_to_close]
                from src.trading.decision_log.record import LogicDecisions
                rec.logic_decisions = LogicDecisions(
                    top_n=0,
                    target_symbols=[],
                    target_weights={},
                    target_value_usd={},
                    reduce_exposure=False,
                    exposure_pct=0.0,
                    exit_signals=close_symbols,
                    hold_signals=[],
                    skip_reasons={},
                )

            with self._stage(rec, "execution"):
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

                            # Log trade exit (non-blocking)
                            try:
                                position_info = self.state_manager.get_positions(STRATEGY_NAME).get(symbol, {})
                                trade_logger = get_trade_log_writer()
                                fill_price = order.get('filled_avg_price', current_price)
                                trade_logger.log_exit(
                                    strategy=STRATEGY_NAME,
                                    symbol=symbol,
                                    qty=abs(qty),
                                    exit_price=float(fill_price) if fill_price else current_price,
                                    order_id=order.get('order_id'),
                                    entry_price=position_info.get('entry_price', entry_price),
                                    entry_time=position_info.get('entry_time'),
                                    account_snapshot=self.broker.get_account()
                                )
                            except Exception as log_err:
                                logger.error(f"[OMR] Trade logging failed (non-blocking): {log_err}")

                            self.state_manager.remove_position(STRATEGY_NAME, symbol)
                        else:
                            logger.error(f"[OMR] Failed to close {symbol}")

                    except Exception as e:
                        logger.error(f"[OMR] Error closing {position.get('symbol', 'UNKNOWN')}: {e}")
                        continue

                logger.info("[OMR] Overnight position closing complete")

            # ---- Post-state ----
            with self._stage(rec, "post_state"):
                rec.post_state = self._snapshot_post_state()
                # Note: update_last_execution is intentionally omitted for exit;
                # the exit is a close-positions operation, not a new decision cycle.

        except Exception as e:
            from src.trading.decision_log.record import ErrorInfo
            rec.error = ErrorInfo(
                type=type(e).__name__,
                message=str(e),
                traceback=_tb.format_exc(),
                stage=getattr(rec, "_current_stage", "unknown"),
            )
            logger.error(f"[OMR] Error in _run_exit: {e}")
            _tb.print_exc()
            raise
        finally:
            try:
                if had_lock:
                    self.state_manager.release_execution_lock(STRATEGY_NAME)
            finally:
                self._write_decision(rec)

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
        """Close overnight positions at market open (09:31 AM).

        Backward-compatible wrapper - delegates to run_once(action="exit").
        """
        self.run_once(action="exit")

    # ------------------------------------------------------------------
    # Decision-log helpers (mirrors RAMP's _build_decision_inputs etc.)
    # ------------------------------------------------------------------

    def _build_decision_inputs(self):
        """Build StrategyInputs for the decision record."""
        from src.trading.decision_log.record import StrategyInputs

        universe_size = len(self.symbols) if self.symbols else 0
        intraday_ready = (
            self._intraday_cache is not None and len(self._intraday_cache) > 0
        )
        return StrategyInputs(
            regime=None,
            regime_confidence=None,
            regime_params=None,
            vix=None,
            spy_drawdown_pct=None,
            universe_size=universe_size,
            data_completeness_pct=100.0 if intraday_ready else 0.0,
            cache_source="intraday_cache" if intraday_ready else "none",
            momentum_scores={},
            extra={},
        )

    def _build_decision_logic(self, signals):
        """Build LogicDecisions from a list of signals."""
        from src.trading.decision_log.record import LogicDecisions

        buy = [s for s in signals if getattr(s, "direction", None) == "BUY"]
        sell = [s for s in signals if getattr(s, "direction", None) in ("SELL", "SHORT")]
        hold = [s for s in signals if getattr(s, "direction", None) == "HOLD"]

        return LogicDecisions(
            top_n=0,
            target_symbols=[s.symbol for s in buy],
            target_weights={
                s.symbol: float(getattr(s, "confidence", 0.0))
                for s in buy
            },
            target_value_usd={},
            reduce_exposure=False,
            exposure_pct=1.0,
            exit_signals=[s.symbol for s in sell],
            hold_signals=[s.symbol for s in hold],
            skip_reasons={},
        )

    def _snapshot_post_state(self):
        """Build PostState reflecting positions and equity after execution."""
        from src.trading.decision_log.record import PostState, PositionSnapshot

        omr_positions = self.state_manager.get_positions(STRATEGY_NAME)
        broker_pos_list = self.broker.get_positions() or []
        broker_positions = {p.get("symbol"): p for p in broker_pos_list}

        positions_after: dict = {}
        for sym in omr_positions.keys():
            bp = broker_positions.get(sym)
            if bp:
                positions_after[sym] = PositionSnapshot(
                    qty=float(bp.get("quantity", 0)),
                    avg_price=float(bp.get("avg_entry_price", 0)),
                    unrealized_pnl=float(bp.get("unrealized_pnl", 0) or 0),
                )

        try:
            account = self.broker.get_account()
            cash_after = float(account.get("cash", 0) or 0)
        except Exception:
            cash_after = 0.0

        initial = float(getattr(self, "initial_capital", 0.0) or 0.0)
        equity_after = initial + sum(
            p.unrealized_pnl for p in positions_after.values()
        )

        return PostState(
            positions_after=positions_after,
            cash_after=cash_after,
            strategy_equity_after=equity_after,
            state_writes=[],
        )


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
