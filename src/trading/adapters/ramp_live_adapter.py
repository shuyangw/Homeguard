"""
RAMP (Regime-Aware Momentum Protection) Live Trading Adapter.

Extends momentum protection with market regime detection.
Adapts parameters based on whether market is in bull, bear, or sideways state.
Rebalances daily at 3:55 PM EST based on regime-adjusted momentum rankings.
"""

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np

from src.trading.adapters.strategy_adapter import StrategyAdapter
from src.utils.vix_provider import get_vix_provider

if TYPE_CHECKING:
    from src.data.providers.base import DataProviderInterface
from src.strategies.advanced.ramp_strategy import (
    RAMPSignals,
    RAMPSignal,
    RAMPRiskSignals
)
from src.strategies.core import StrategySignals, Signal
from src.strategies.universe import ETFUniverse
from src.trading.brokers.broker_interface import BrokerInterface, OrderSide, OrderType
from src.trading.utils.portfolio_health_check import PortfolioHealthChecker
from src.trading.state import StrategyStateManager
from src.utils.logger import logger
from src.utils.timezone import tz
from src.utils.trading_logger import get_trade_log_writer

# Strategy identifier for state tracking
STRATEGY_NAME = 'ramp'


class RAMPSignalWrapper(StrategySignals):
    """
    Wrapper to make RAMPSignals compatible with StrategyAdapter.
    """

    def __init__(self, ramp_signals: RAMPSignals):
        self._ramp_signals = ramp_signals
        self._current_positions: Dict[str, float] = {}

    def get_required_lookback(self) -> int:
        """Return number of periods needed for momentum calculation (252 days + buffer)."""
        return 300

    def set_current_positions(self, positions: Dict[str, float]):
        """Update current positions for signal generation."""
        self._current_positions = positions

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
        # Extract prices from market data
        prices_dict = {}
        for symbol, df in market_data.items():
            if symbol not in ('SPY', 'VIX') and 'close' in df.columns:
                prices_dict[symbol] = df['close']

        if not prices_dict:
            logger.warning("[RAMP] No price data available for signal generation")
            return []

        # Create prices DataFrame
        prices_df = pd.DataFrame(prices_dict)

        # Get SPY and VIX
        spy_prices = None
        vix_prices = None

        if 'SPY' in market_data and 'close' in market_data['SPY'].columns:
            spy_prices = market_data['SPY']['close']

        if 'VIX' in market_data and 'close' in market_data['VIX'].columns:
            vix_prices = market_data['VIX']['close']

        # Update historical data cache
        self._ramp_signals.update_historical_data(prices_df, spy_prices, vix_prices)

        # Generate RAMP signals
        ramp_signals, risk_signals = self._ramp_signals.generate_signals(
            current_positions=self._current_positions,
            prices_df=prices_df,
            spy_prices=spy_prices,
            vix_prices=vix_prices
        )

        # Convert to base Signal objects
        signals = []
        now = timestamp or datetime.now()

        for rs in ramp_signals:
            # Get latest price for this symbol
            price = 0.0
            if rs.symbol in market_data and 'close' in market_data[rs.symbol].columns:
                price = float(market_data[rs.symbol]['close'].iloc[-1])

            if rs.action == 'buy':
                signals.append(Signal(
                    timestamp=now,
                    symbol=rs.symbol,
                    direction='BUY',
                    confidence=rs.weight,  # Use weight as confidence
                    price=price,
                    metadata={
                        'momentum_score': rs.momentum_score,
                        'rank': rs.rank,
                        'regime': rs.regime,
                        'risk_exposure': risk_signals.exposure_pct,
                        'weight': rs.weight  # Keep original weight for position sizing
                    }
                ))
            elif rs.action == 'sell':
                signals.append(Signal(
                    timestamp=now,
                    symbol=rs.symbol,
                    direction='SELL',
                    confidence=1.0,
                    price=price,
                    metadata={'action': 'sell', 'regime': rs.regime}
                ))

        return signals


class RAMPLiveAdapter(StrategyAdapter):
    """
    Live trading adapter for RAMP (Regime-Aware Momentum Protection) strategy.

    Key differences from standard Momentum Protection:
    1. Uses MarketRegimeDetector to classify current market state
    2. Adapts momentum parameters based on detected regime
    3. Uses optimized formula: long_w * long_ret - pen_w * short_ret

    Regime-Specific Parameters (Walk-Forward Validated 2017-2024):
        STRONG_BULL: L21/S5, LW=0.3, PW=5.0, TopN=20
        WEAK_BULL:   L21/S5, LW=0.3, PW=5.0, TopN=10
        SIDEWAYS:    L21/S5, LW=0.5, PW=2.0, TopN=5
        UNPREDICTABLE: L42/S21, LW=0.5, PW=4.0, TopN=10
        BEAR:        L21/S5, LW=0.3, PW=3.0, TopN=10

    Rebalances at 3:55 PM EST based on:
    - Regime-adjusted momentum rankings
    - Rule-based crash protection signals
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbols: Optional[List[str]] = None,
        reduced_exposure: float = 0.5,
        vix_threshold: float = 25.0,
        spy_dd_threshold: float = -0.05,
        slippage_per_share: float = 0.01,
        data_provider: Optional["DataProviderInterface"] = None
    ):
        """
        Initialize RAMP live adapter.

        Args:
            broker: Broker interface
            symbols: List of symbols to trade (default: S&P 500)
            reduced_exposure: Exposure when risk signals trigger (0-1)
            vix_threshold: VIX level that triggers protection
            spy_dd_threshold: SPY drawdown threshold (negative)
            slippage_per_share: Expected slippage in dollars
            data_provider: Optional data provider with fallback (uses broker if not provided)
        """
        # Use default S&P 500 symbols if not specified
        if symbols is None:
            symbols = self._load_sp500_symbols()
            logger.info(f"[RAMP] Using default momentum universe: {len(symbols)} S&P 500 stocks")

        # Create RAMP signal generator
        ramp_signals = RAMPSignals(
            symbols=symbols,
            reduced_exposure=reduced_exposure,
            vix_threshold=vix_threshold,
            spy_dd_threshold=spy_dd_threshold
        )

        # Wrap for compatibility with base adapter
        strategy = RAMPSignalWrapper(ramp_signals)

        # RAMP needs 1+ years of daily data for momentum calculation
        data_lookback_days = 400

        # Get initial top_n from RAMP (will adapt based on regime)
        initial_top_n = ramp_signals.current_top_n

        # Initialize base adapter
        super().__init__(
            strategy=strategy,
            broker=broker,
            symbols=symbols,
            position_size=0.10,  # Base position size, adjusted by regime
            max_positions=20,  # Maximum across all regimes (STRONG_BULL uses 20)
            data_lookback_days=data_lookback_days,
            data_provider=data_provider
        )

        # Store configuration
        self.reduced_exposure = reduced_exposure
        self.vix_threshold = vix_threshold
        self.spy_dd_threshold = spy_dd_threshold
        self.slippage_per_share = slippage_per_share

        # Store reference to RAMP signals
        self._ramp_signals = ramp_signals

        # Initialize state manager for multi-strategy coordination
        self.state_manager = StrategyStateManager()

        # Initialize portfolio health checker with state manager for multi-strategy support
        self.health_checker = PortfolioHealthChecker(
            broker=broker,
            min_buying_power=5000.0,
            min_portfolio_value=10000.0,
            max_positions=25,  # Allow buffer above max 20
            max_position_age_hours=48,
            state_manager=self.state_manager
        )

        # Track last risk signals
        self._last_risk_signals: Optional[RAMPRiskSignals] = None

        logger.info("[RAMP] Regime-Aware Momentum Protection Configuration:")
        logger.info(f"[RAMP]   Universe: {len(symbols)} S&P 500 stocks")
        logger.info(f"[RAMP]   Reduced exposure: {reduced_exposure:.0%}")
        logger.info(f"[RAMP]   VIX threshold: {vix_threshold}")
        logger.info(f"[RAMP]   Rebalance time: 3:55 PM EST")
        logger.info(f"[RAMP]   Portfolio health checks: ENABLED")
        logger.info("[RAMP]   Regime-specific parameters loaded")

    def _load_sp500_symbols(self) -> List[str]:
        """Load S&P 500 symbols from CSV, excluding leveraged ETFs."""
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        csv_path = project_root / 'backtest_lists' / 'sp500-2025.csv'

        try:
            import pandas as pd
            symbols_df = pd.read_csv(csv_path)
            symbols = symbols_df['Symbol'].tolist()

            # Filter out any leveraged ETFs (to avoid conflict with OMR)
            original_count = len(symbols)
            symbols = [s for s in symbols if not ETFUniverse.is_leveraged(s)]
            filtered_count = original_count - len(symbols)

            if filtered_count > 0:
                logger.info(f"[RAMP] Filtered out {filtered_count} leveraged ETFs from universe")

            return symbols
        except Exception as e:
            logger.error(f"[RAMP] Failed to load S&P 500 symbols: {e}")
            # Return a minimal default list (no leveraged ETFs)
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

    def preload_historical_data(self) -> None:
        """
        Pre-load historical data for momentum calculation.

        Fetches via Alpaca:
        1. Daily prices for all symbols (252+ days for momentum)
        2. SPY prices for drawdown calculation and regime detection
        3. VIX prices for fear signals and regime detection
        """
        logger.info("[RAMP] Pre-loading historical data for regime-aware momentum strategy...")

        try:
            end_date = tz.now()
            start_date = end_date - timedelta(days=self.data_lookback_days)

            # Fetch historical data for all symbols via Alpaca
            logger.info(f"[RAMP] Fetching {len(self.symbols)} symbols from {start_date.date()} to {end_date.date()}")

            prices_dict = {}
            failed_symbols = []

            # Batch fetch from Alpaca (in chunks to avoid API limits)
            batch_size = 50
            for i in range(0, len(self.symbols), batch_size):
                batch = self.symbols[i:i + batch_size]
                for symbol in batch:
                    try:
                        df = self.broker.get_historical_bars(
                            symbol=symbol,
                            start=start_date,
                            end=end_date,
                            timeframe='1D'
                        )
                        if df is not None and not df.empty:
                            # Normalize column names
                            df.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]
                            if 'close' in df.columns:
                                prices_dict[symbol] = df['close']
                    except Exception as e:
                        failed_symbols.append(symbol)

                # Log progress every batch
                if (i + batch_size) % 100 == 0:
                    logger.info(f"[RAMP] Fetched {min(i + batch_size, len(self.symbols))}/{len(self.symbols)} symbols...")

            if not prices_dict:
                logger.error("[RAMP] Failed to download historical price data from Alpaca")
                return

            # Create prices DataFrame
            prices_df = pd.DataFrame(prices_dict)

            if failed_symbols:
                logger.warning(f"[RAMP] Failed to fetch {len(failed_symbols)} symbols: {failed_symbols[:10]}...")

            logger.info(f"[RAMP] Downloaded {len(prices_df.columns)} symbols, {len(prices_df)} days via Alpaca")

            # Fetch SPY via Alpaca
            spy_data = self.broker.get_historical_bars(
                symbol='SPY',
                start=start_date,
                end=end_date,
                timeframe='1D'
            )

            # Fetch VIX via VIXProvider with fallback chain (yfinance -> FRED -> cache)
            logger.info("[RAMP] Fetching VIX via VIXProvider (not available on Alpaca)")
            vix_data = self._fetch_vix_data(lookback_days=self.data_lookback_days)

            # Extract close prices
            if spy_data is not None and not spy_data.empty:
                spy_data.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in spy_data.columns]
                spy_prices = spy_data['close'] if 'close' in spy_data.columns else pd.Series()
            else:
                spy_prices = pd.Series()
                logger.warning("[RAMP] Failed to fetch SPY data")

            if vix_data is not None and not vix_data.empty:
                # yfinance returns 'Close' (capitalized) after MultiIndex flattening
                if 'Close' in vix_data.columns:
                    vix_prices = vix_data['Close']
                elif 'close' in vix_data.columns:
                    vix_prices = vix_data['close']
                else:
                    vix_prices = pd.Series()
                    logger.warning(f"[RAMP] VIX data has unexpected columns: {list(vix_data.columns)}")
            else:
                vix_prices = pd.Series()
                logger.warning("[RAMP] Failed to fetch VIX data")

            # Update cache in RAMP signals
            self._ramp_signals.update_historical_data(prices_df, spy_prices, vix_prices)

            # Detect initial regime
            regime, confidence = self._ramp_signals.detect_regime(spy_prices, vix_prices)
            logger.info(f"[RAMP] Initial regime: {regime} (confidence: {confidence:.1%})")

            # Cache for base adapter
            self._data_cache = {
                'prices': prices_df,
                'SPY': spy_data,
                'VIX': vix_data
            }
            self._cache_date = end_date

            logger.success(f"[RAMP] Historical data pre-loaded: {len(prices_df.columns)} symbols")
            logger.info(f"[RAMP]   SPY data: {len(spy_prices)} days")
            logger.info(f"[RAMP]   VIX data: {len(vix_prices)} days")

        except Exception as e:
            logger.error(f"[RAMP] Failed to pre-load historical data: {e}")
            import traceback
            traceback.print_exc()

    def _fetch_vix_data(self, lookback_days: int = 400) -> Optional[pd.DataFrame]:
        """
        Fetch VIX data via VIXProvider with multi-source fallback.

        Uses VIXProvider which tries:
        1. yfinance (primary) - Yahoo Finance ^VIX
        2. FRED API (fallback) - Federal Reserve VIXCLS series
        3. Persisted cache (last resort) - Last known good VIX value

        Args:
            lookback_days: Number of days of history needed

        Returns:
            DataFrame with VIX data, or None if all sources fail
        """
        try:
            vix_provider = get_vix_provider()
            vix_data = vix_provider.get_vix_data(lookback_days=lookback_days)

            if vix_data is not None:
                source, fetch_time = vix_provider.get_source_info()
                logger.info(f"[RAMP] VIX data from {source}: {len(vix_data)} days")

                # Log warning if using cached data
                if source == "cache":
                    logger.warning("[RAMP] Using cached VIX data (may be stale)")

                # VIXProvider returns 'close' column, convert to DataFrame with 'Close' for compatibility
                result = pd.DataFrame({'Close': vix_data['close']})
                return result
            else:
                logger.error("[RAMP] All VIX data sources failed!")
                return None

        except Exception as e:
            logger.error(f"[RAMP] Failed to fetch VIX data: {e}")
            return None

    def fetch_todays_closes(self) -> bool:
        """
        Fetch only today's close prices and append to historical cache.

        This is a lightweight fetch at 3:55 PM that only gets today's data
        for the universe symbols, rather than re-fetching all historical data.

        Returns:
            True if successful, False otherwise
        """
        logger.info("[RAMP] " + "=" * 60)
        logger.info("[RAMP] FETCHING TODAY'S CLOSES (3:55 PM)")
        logger.info("[RAMP] " + "=" * 60)

        try:
            # Check if we have historical data to append to
            if self._data_cache is None or 'prices' not in self._data_cache:
                logger.warning("[RAMP] No historical cache - falling back to full fetch")
                self.preload_historical_data()
                return True

            prices_df = self._data_cache['prices']
            today = tz.now().date()

            # Check if we already have today's data
            if len(prices_df) > 0:
                last_date = prices_df.index[-1]
                if hasattr(last_date, 'date'):
                    last_date = last_date.date()
                if last_date == today:
                    logger.info("[RAMP] Already have today's data in cache")
                    return True

            logger.info(f"[RAMP] Fetching today's closes for {len(self.symbols)} symbols...")

            # Fetch today's data only (just 1 day)
            today_start = tz.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = tz.now()

            todays_prices = {}
            failed = 0

            for symbol in self.symbols:
                try:
                    df = self.broker.get_historical_bars(
                        symbol=symbol,
                        start=today_start,
                        end=today_end,
                        timeframe='1D'
                    )
                    if df is not None and not df.empty:
                        df.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]
                        if 'close' in df.columns:
                            # Get the last (most recent) close price
                            todays_prices[symbol] = df['close'].iloc[-1]
                except Exception:
                    failed += 1
                    continue

            if not todays_prices:
                logger.error("[RAMP] Failed to fetch any today's prices")
                return False

            logger.info(f"[RAMP] Fetched {len(todays_prices)}/{len(self.symbols)} symbols ({failed} failed)")

            # Create today's row and append to historical data
            today_row = pd.Series(todays_prices, name=pd.Timestamp(today))

            # Append to existing prices DataFrame
            new_prices_df = pd.concat([prices_df, today_row.to_frame().T])

            # Fetch today's SPY
            spy_close = None
            try:
                spy_df = self.broker.get_historical_bars(
                    symbol='SPY',
                    start=today_start,
                    end=today_end,
                    timeframe='1D'
                )
                if spy_df is not None and not spy_df.empty:
                    spy_df.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in spy_df.columns]
                    spy_close = spy_df['close'].iloc[-1]
            except Exception as e:
                logger.warning(f"[RAMP] Failed to fetch today's SPY: {e}")

            # Fetch today's VIX via VIXProvider
            vix_close = None
            try:
                vix_provider = get_vix_provider()
                current_vix = vix_provider.get_current_vix()
                if current_vix is not None:
                    vix_close = current_vix
                    logger.info(f"[RAMP] Today's VIX: {vix_close:.2f}")
            except Exception as e:
                logger.warning(f"[RAMP] Failed to fetch today's VIX: {e}")

            # Append SPY and VIX to their caches
            spy_cache = self._data_cache.get('SPY')
            vix_cache = self._data_cache.get('VIX')

            if spy_close is not None and spy_cache is not None:
                if hasattr(spy_cache, 'columns'):
                    spy_cache.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in spy_cache.columns]
                    if 'close' in spy_cache.columns:
                        new_spy_row = pd.DataFrame({'close': [spy_close]}, index=[pd.Timestamp(today)])
                        spy_cache = pd.concat([spy_cache, new_spy_row])
                        self._data_cache['SPY'] = spy_cache

            if vix_close is not None and vix_cache is not None:
                if hasattr(vix_cache, 'columns'):
                    col_name = 'Close' if 'Close' in vix_cache.columns else 'close'
                    new_vix_row = pd.DataFrame({col_name: [vix_close]}, index=[pd.Timestamp(today)])
                    vix_cache = pd.concat([vix_cache, new_vix_row])
                    self._data_cache['VIX'] = vix_cache

            # Update cache
            self._data_cache['prices'] = new_prices_df

            # Update RAMP signals cache
            spy_prices = self._data_cache['SPY']['close'] if 'SPY' in self._data_cache and self._data_cache['SPY'] is not None else pd.Series()
            vix_col = 'Close' if 'VIX' in self._data_cache and self._data_cache['VIX'] is not None and 'Close' in self._data_cache['VIX'].columns else 'close'
            vix_prices = self._data_cache['VIX'][vix_col] if 'VIX' in self._data_cache and self._data_cache['VIX'] is not None else pd.Series()

            self._ramp_signals.update_historical_data(new_prices_df, spy_prices, vix_prices)

            # Re-detect regime with updated data
            regime, confidence = self._ramp_signals.detect_regime(spy_prices, vix_prices)
            logger.info(f"[RAMP] Current regime: {regime} (confidence: {confidence:.1%})")

            logger.success(f"[RAMP] Appended today's data - cache now has {len(new_prices_df)} days")
            return True

        except Exception as e:
            logger.error(f"[RAMP] Failed to fetch today's closes: {e}")
            import traceback
            traceback.print_exc()
            return False

    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch current market data for signal generation.

        Uses cached historical data (refreshed at execution time to include today's prices).
        """
        try:
            market_data = {}

            # Use cached data if available
            if self._data_cache is not None:
                logger.info("[RAMP] Using cached historical data")

                prices_df = self._data_cache.get('prices')
                if prices_df is not None:
                    for symbol in prices_df.columns:
                        market_data[symbol] = pd.DataFrame({'close': prices_df[symbol]})

                if 'SPY' in self._data_cache:
                    spy_df = self._data_cache['SPY']
                    if 'Close' in spy_df.columns:
                        market_data['SPY'] = pd.DataFrame({'close': spy_df['Close']})
                    elif 'close' in spy_df.columns:
                        market_data['SPY'] = pd.DataFrame({'close': spy_df['close']})

                if 'VIX' in self._data_cache:
                    vix_df = self._data_cache['VIX']
                    if 'Close' in vix_df.columns:
                        market_data['VIX'] = pd.DataFrame({'close': vix_df['Close']})
                    elif 'close' in vix_df.columns:
                        market_data['VIX'] = pd.DataFrame({'close': vix_df['close']})

            else:
                logger.warning("[RAMP] No cached data available, fetching...")
                self.preload_historical_data()
                return self.fetch_market_data()

            logger.info(f"[RAMP] Market data prepared: {len(market_data)} symbols")
            return market_data

        except Exception as e:
            logger.error(f"[RAMP] Error in fetch_market_data: {e}")
            return {}

    def run_once(self) -> None:
        """
        Run one iteration of the strategy with portfolio health checks.

        Includes:
        - Fresh data fetch at execution time (3:55 PM)
        - Toggle/shutdown checks
        - Execution lock for multi-strategy coordination
        - Position tracking per strategy
        """
        logger.info("[RAMP] " + "=" * 60)
        logger.info(f"[RAMP] Running {self.__class__.__name__} at {tz.now()}")
        logger.info("[RAMP] " + "=" * 60)

        try:
            # Check if strategy is enabled
            if not self.state_manager.is_enabled(STRATEGY_NAME):
                logger.warning("[RAMP] Strategy is DISABLED - skipping execution")
                return

            # Check if shutdown requested
            if self.state_manager.is_shutdown_requested(STRATEGY_NAME):
                logger.warning("[RAMP] Shutdown requested - skipping new entries")
                return

            # Fetch today's closes only (lightweight, ~30s vs ~2min for full fetch)
            # This appends today's prices to the 9:30 AM historical cache
            logger.info("[RAMP] Fetching today's closes for 3:55 PM execution...")
            self.fetch_todays_closes()

            # Acquire execution lock (blocks if another strategy is executing)
            if not self.state_manager.acquire_execution_lock(STRATEGY_NAME):
                logger.error("[RAMP] Failed to acquire execution lock - another strategy is running")
                return

            try:
                # Sync state with broker (detect external position changes)
                broker_positions = {p['symbol']: int(p['quantity']) for p in self.broker.get_positions()}
                changes = self.state_manager.sync_with_broker(broker_positions)
                if changes['removed']:
                    logger.info(f"[RAMP] Detected closed positions: {changes['removed']}")

                # Portfolio health check before entry
                # Use strategy_name='ramp' to only count RAMP positions for max_positions check
                logger.info("[RAMP] Running pre-entry portfolio health check...")
                health_result = self.health_checker.check_before_entry(
                    required_capital=None,
                    allow_existing_positions=True,
                    strategy_name=STRATEGY_NAME
                )

                if not health_result.passed:
                    logger.error("[RAMP] Portfolio health check FAILED - BLOCKING ENTRY")
                    for error in health_result.errors:
                        logger.error(f"[RAMP]   - {error}")
                    return

                if health_result.warnings:
                    logger.warning("[RAMP] Portfolio health check passed with warnings:")
                    for warning in health_result.warnings:
                        logger.warning(f"[RAMP]   - {warning}")

                logger.success("[RAMP] Portfolio health check PASSED - proceeding with rebalance")

                # Get RAMP's own positions from state manager
                ramp_positions = self.state_manager.get_positions(STRATEGY_NAME)

                # Update current positions in signal generator (using broker positions for value)
                positions = self.broker.get_positions()
                current_positions = {}
                for pos in positions:
                    symbol = pos.get('symbol')
                    # Only include if tracked by RAMP or not owned by another strategy
                    owner = self.state_manager.symbol_owned_by_other(STRATEGY_NAME, symbol)
                    if symbol in ramp_positions or owner is None:
                        value = float(pos.get('market_value', 0))
                        current_positions[symbol] = value

                self.strategy.set_current_positions(current_positions)

                # Fetch market data
                market_data = self.fetch_market_data()

                if not market_data:
                    logger.error("[RAMP] No market data available")
                    return

                # Generate signals
                signals = self.strategy.generate_signals(market_data, tz.now())

                # Log regime and risk signals
                if signals:
                    first_signal = signals[0]
                    regime = first_signal.metadata.get('regime', 'UNKNOWN') if first_signal.metadata else 'UNKNOWN'
                    risk_exposure = first_signal.metadata.get('risk_exposure', 1.0) if first_signal.metadata else 1.0
                    logger.info(f"[RAMP] Current regime: {regime}")
                    if risk_exposure < 1.0:
                        logger.warning(f"[RAMP] Risk signals active - exposure reduced to {risk_exposure:.0%}")

                # Execute trades
                self._execute_rebalance(signals, current_positions)

                # Update last execution timestamp
                self.state_manager.update_last_execution(STRATEGY_NAME)

            finally:
                # Always release execution lock
                self.state_manager.release_execution_lock(STRATEGY_NAME)

        except Exception as e:
            logger.error(f"[RAMP] Error in run_once: {e}")
            import traceback
            traceback.print_exc()

    def _execute_rebalance(
        self,
        signals: List[Signal],
        current_positions: Dict[str, float]
    ) -> None:
        """
        Execute rebalance based on signals.

        Args:
            signals: List of trading signals
            current_positions: Current position values by symbol
        """
        try:
            account = self.broker.get_account()
            portfolio_value = float(account.get('portfolio_value', 0))

            if portfolio_value <= 0:
                logger.error("[RAMP] Portfolio value is zero or negative")
                return

            logger.info(f"[RAMP] Portfolio value: ${portfolio_value:,.2f}")

            # Separate buy and sell signals
            buy_signals = [s for s in signals if s.direction == 'BUY']
            sell_signals = [s for s in signals if s.direction == 'SELL']

            # Get current top_n from RAMP (regime-dependent)
            current_top_n = self._ramp_signals.current_top_n

            # Execute sells first
            for signal in sell_signals:
                symbol = signal.symbol
                if symbol in current_positions:
                    try:
                        pos = next((p for p in self.broker.get_positions() if p['symbol'] == symbol), None)
                        if pos:
                            qty = int(pos['quantity'])
                            logger.info(f"[RAMP] Selling {symbol}: {qty} shares (exiting position)")

                            order = self.execution_engine.execute_order(
                                symbol=symbol,
                                quantity=abs(qty),
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET
                            )

                            if order:
                                logger.success(f"[RAMP] Sell order placed: {symbol}")

                                # Log trade exit to persistent trade log BEFORE removing position
                                # Get entry info from state manager while it still exists
                                try:
                                    position_info = self.state_manager.get_positions(STRATEGY_NAME).get(symbol, {})
                                    trade_logger = get_trade_log_writer()
                                    fill_price = order.get('avg_fill_price', pos.get('current_price', 0))
                                    trade_logger.log_exit(
                                        strategy=STRATEGY_NAME,
                                        symbol=symbol,
                                        qty=abs(qty),
                                        exit_price=float(fill_price) if fill_price else float(pos.get('current_price', 0)),
                                        order_id=order.get('order_id'),
                                        entry_price=position_info.get('entry_price', float(pos.get('avg_entry_price', 0))),
                                        entry_time=position_info.get('entry_time')
                                    )
                                except Exception as log_err:
                                    logger.error(f"[RAMP] Trade logging failed (non-blocking): {log_err}")

                                # Remove from state tracking
                                self.state_manager.remove_position(STRATEGY_NAME, symbol)
                    except Exception as e:
                        logger.error(f"[RAMP] Error selling {symbol}: {e}")

            # Execute buys
            for signal in buy_signals:
                symbol = signal.symbol
                weight = signal.metadata.get('weight', signal.confidence) if signal.metadata else signal.confidence
                target_value = portfolio_value * self.position_size * weight * current_top_n

                # Skip if already at target
                current_value = current_positions.get(symbol, 0)
                if abs(target_value - current_value) < 100:  # $100 threshold
                    continue

                try:
                    # Get current price
                    quote = self.broker.get_latest_quote(symbol)
                    if not quote:
                        logger.warning(f"[RAMP] No quote available for {symbol}")
                        continue

                    current_price = float(quote.get('ask', quote.get('bid', 0)))
                    if current_price <= 0:
                        continue

                    # Calculate shares to buy
                    target_shares = int(target_value / current_price)
                    current_shares = int(current_value / current_price) if current_value > 0 else 0
                    shares_to_buy = target_shares - current_shares

                    if shares_to_buy > 0:
                        # Check if symbol is owned by another strategy
                        owner = self.state_manager.symbol_owned_by_other(STRATEGY_NAME, symbol)
                        if owner:
                            logger.warning(f"[RAMP] Skipping {symbol} - owned by {owner}")
                            continue

                        regime = signal.metadata.get('regime', 'UNKNOWN') if signal.metadata else 'UNKNOWN'
                        logger.info(
                            f"[RAMP] Buying {symbol}: {shares_to_buy} shares @ ${current_price:.2f} "
                            f"(rank #{signal.metadata.get('rank', '?')}, regime={regime})"
                        )

                        order = self.execution_engine.execute_order(
                            symbol=symbol,
                            quantity=shares_to_buy,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET
                        )

                        if order:
                            logger.success(f"[RAMP] Buy order placed: {symbol}")
                            # Use add_or_update_position to handle both new positions AND top-ups
                            # CRITICAL: This prevents state drift when topping up existing positions
                            order_id = order.get('order_id')
                            self.state_manager.add_or_update_position(
                                STRATEGY_NAME, symbol, shares_to_buy, current_price, order_id
                            )

                            # Log trade entry to persistent trade log
                            # Error handling ensures logging failures don't block trading
                            try:
                                trade_logger = get_trade_log_writer()
                                fill_price = order.get('avg_fill_price', current_price)
                                trade_logger.log_entry(
                                    strategy=STRATEGY_NAME,
                                    symbol=symbol,
                                    qty=shares_to_buy,
                                    price=float(fill_price) if fill_price else current_price,
                                    order_id=order_id,
                                    metadata={
                                        'rank': signal.metadata.get('rank') if signal.metadata else None,
                                        'momentum_score': signal.metadata.get('momentum_score') if signal.metadata else None,
                                        'regime': regime
                                    }
                                )
                            except Exception as log_err:
                                logger.error(f"[RAMP] Trade logging failed (non-blocking): {log_err}")

                except Exception as e:
                    logger.error(f"[RAMP] Error buying {symbol}: {e}")

            logger.info("[RAMP] Rebalance execution complete")

        except Exception as e:
            logger.error(f"[RAMP] Error in _execute_rebalance: {e}")

    def get_schedule(self) -> Dict[str, any]:
        """
        Get scheduling configuration.

        RAMP rebalances once daily at 3:55 PM EST.
        Uses today's momentum (known at 3:55 PM) with regime-specific parameters.
        """
        return {
            'execution_times': [
                {'time': '15:55', 'action': 'rebalance'}  # 3:55 PM EST - Rebalance near close
            ],
            'market_hours_only': True,
            'strategy_type': 'daily'  # Indicates daily rebalancing
        }

    def show_current_signals(self) -> None:
        """Display current RAMP signals and risk status."""
        try:
            # Get current positions
            positions = self.broker.get_positions()
            current_positions = {p['symbol']: float(p['market_value']) for p in positions}

            # Generate signals
            signals, risk_signals = self._ramp_signals.generate_signals(
                current_positions=current_positions
            )

            logger.info("\n" + "=" * 60)
            logger.info("[RAMP] CURRENT REGIME-AWARE MOMENTUM SIGNALS")
            logger.info("=" * 60)

            # Regime status
            logger.info("\n[RAMP] Regime Status:")
            logger.info(f"[RAMP]   Current Regime: {risk_signals.regime}")
            logger.info(f"[RAMP]   Confidence: {risk_signals.regime_confidence:.1%}")
            logger.info(f"[RAMP]   Parameters: {risk_signals.params}")

            # Risk status
            logger.info("\n[RAMP] Risk Signals:")
            logger.info(f"[RAMP]   VIX > {self.vix_threshold}: {'YES' if risk_signals.high_vix else 'NO'}")
            logger.info(f"[RAMP]   SPY Drawdown: {'YES' if risk_signals.spy_drawdown else 'NO'}")
            logger.info(f"[RAMP]   Reduce Exposure: {'YES' if risk_signals.reduce_exposure else 'NO'}")
            logger.info(f"[RAMP]   Exposure: {risk_signals.exposure_pct:.0%}")

            # Top momentum stocks
            top_n = risk_signals.params.get('top_n', 10)
            logger.info(f"\n[RAMP] Top {top_n} Momentum Stocks ({risk_signals.regime}):")
            buy_signals = [s for s in signals if s.action in ('buy', 'hold')]
            for s in sorted(buy_signals, key=lambda x: x.rank):
                logger.info(f"[RAMP]   #{s.rank}: {s.symbol} (score: {s.momentum_score:.2%})")

            # Sell signals
            sell_signals = [s for s in signals if s.action == 'sell']
            if sell_signals:
                logger.info("\n[RAMP] Positions to Exit:")
                for s in sell_signals:
                    logger.info(f"[RAMP]   {s.symbol}")

        except Exception as e:
            logger.error(f"[RAMP] Error showing signals: {e}")


if __name__ == "__main__":
    logger.info("[RAMP] Regime-Aware Momentum Protection Live Trading Adapter")
    logger.info("[RAMP] " + "=" * 60)
    logger.info("[RAMP] Extends Momentum Protection with regime detection")
    logger.info("[RAMP] Adapts parameters based on market state")
    logger.info("")
    logger.info("[RAMP] Regime-Specific Parameters:")
    logger.info("[RAMP]   STRONG_BULL: L21/S5, LW=0.3, PW=5.0, TopN=20")
    logger.info("[RAMP]   WEAK_BULL:   L21/S5, LW=0.3, PW=5.0, TopN=10")
    logger.info("[RAMP]   SIDEWAYS:    L21/S5, LW=0.5, PW=2.0, TopN=5")
    logger.info("[RAMP]   UNPREDICTABLE: L42/S21, LW=0.5, PW=4.0, TopN=10")
    logger.info("[RAMP]   BEAR:        L21/S5, LW=0.3, PW=3.0, TopN=10")
    logger.info("")
    logger.info("[RAMP] Rebalances at 3:55 PM EST")
    logger.info("[RAMP] Walk-Forward Validation: Sharpe 1.859 (vs 1.271 static)")
