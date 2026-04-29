"""
Momentum Protection Live Trading Adapter.

Connects momentum strategy with crash protection to live trading infrastructure.
Rebalances daily at 3:55 PM EST based on momentum rankings and risk signals.
"""

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np

from src.trading.adapters.strategy_adapter import StrategyAdapter
from src.utils.vix_provider import get_vix_provider

if TYPE_CHECKING:
    from src.data.providers.base import DataProviderInterface
from src.strategies.advanced.momentum_protection_strategy import (
    MomentumProtectionSignals,
    MomentumSignal,
    RiskSignals
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
STRATEGY_NAME = 'mp'


class MomentumSignalWrapper(StrategySignals):
    """
    Wrapper to make MomentumProtectionSignals compatible with StrategyAdapter.
    """

    def __init__(self, momentum_signals: MomentumProtectionSignals):
        self._momentum_signals = momentum_signals
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
            logger.warning("[MP] No price data available for signal generation")
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
        self._momentum_signals.update_historical_data(prices_df, spy_prices, vix_prices)

        # Generate momentum signals
        momentum_signals, risk_signals = self._momentum_signals.generate_signals(
            current_positions=self._current_positions,
            prices_df=prices_df,
            spy_prices=spy_prices,
            vix_prices=vix_prices
        )

        # Convert to base Signal objects
        signals = []
        now = timestamp or datetime.now()

        for ms in momentum_signals:
            # Get latest price for this symbol
            price = 0.0
            if ms.symbol in market_data and 'close' in market_data[ms.symbol].columns:
                price = float(market_data[ms.symbol]['close'].iloc[-1])

            if ms.action == 'buy':
                signals.append(Signal(
                    timestamp=now,
                    symbol=ms.symbol,
                    direction='BUY',
                    confidence=ms.weight,  # Use weight as confidence
                    price=price,
                    metadata={
                        'momentum_score': ms.momentum_score,
                        'rank': ms.rank,
                        'risk_exposure': risk_signals.exposure_pct,
                        'weight': ms.weight  # Keep original weight for position sizing
                    }
                ))
            elif ms.action == 'sell':
                signals.append(Signal(
                    timestamp=now,
                    symbol=ms.symbol,
                    direction='SELL',
                    confidence=1.0,
                    price=price,
                    metadata={'action': 'sell'}
                ))

        return signals


class MomentumLiveAdapter(StrategyAdapter):
    """
    Live trading adapter for Momentum Protection strategy.

    Rebalances at 3:55 PM EST based on:
    - 1m-1w momentum rankings (using today's near-close prices)
    - Rule-based crash protection signals

    Positions are held until next day's 3:55 PM rebalance.
    """

    STRATEGY_NAME: str = STRATEGY_NAME  # class attribute for StrategyAdapter base helpers

    def __init__(
        self,
        broker: BrokerInterface,
        symbols: Optional[List[str]] = None,
        top_n: int = 10,
        position_size: float = 0.10,
        reduced_exposure: float = 0.5,
        vix_threshold: float = 25.0,
        vix_spike_threshold: float = 0.20,
        spy_dd_threshold: float = -0.05,
        mom_vol_percentile: float = 0.90,
        slippage_per_share: float = 0.01,
        data_provider: Optional["DataProviderInterface"] = None,
        *,
        broker_name: str,
    ):
        """
        Initialize Momentum live adapter.

        Args:
            broker: Broker interface
            symbols: List of symbols to trade (default: S&P 500)
            top_n: Number of top momentum stocks to hold
            position_size: Position size per stock as fraction
            reduced_exposure: Exposure when risk signals trigger (0-1)
            vix_threshold: VIX level that triggers protection
            vix_spike_threshold: VIX 5-day change threshold
            spy_dd_threshold: SPY drawdown threshold (negative)
            mom_vol_percentile: Momentum volatility percentile threshold
            slippage_per_share: Expected slippage in dollars
            data_provider: Optional data provider with fallback (uses broker if not provided)
        """
        # Use default S&P 500 symbols if not specified
        if symbols is None:
            symbols = self._load_sp500_symbols()
            logger.info(f"[MP] Using default momentum universe: {len(symbols)} S&P 500 stocks")

        # Create momentum signal generator
        momentum_signals = MomentumProtectionSignals(
            symbols=symbols,
            top_n=top_n,
            reduced_exposure=reduced_exposure,
            vix_threshold=vix_threshold,
            vix_spike_threshold=vix_spike_threshold,
            spy_dd_threshold=spy_dd_threshold,
            mom_vol_percentile=mom_vol_percentile
        )

        # Wrap for compatibility with base adapter
        strategy = MomentumSignalWrapper(momentum_signals)

        # Momentum needs 1+ years of daily data for momentum calculation
        data_lookback_days = 400

        # Initialize base adapter
        super().__init__(
            strategy=strategy,
            broker=broker,
            symbols=symbols,
            position_size=position_size,
            max_positions=top_n,
            data_lookback_days=data_lookback_days,
            data_provider=data_provider
        )

        # Store configuration
        self.top_n = top_n
        self.reduced_exposure = reduced_exposure
        self.vix_threshold = vix_threshold
        self.vix_spike_threshold = vix_spike_threshold
        self.spy_dd_threshold = spy_dd_threshold
        self.mom_vol_percentile = mom_vol_percentile
        self.slippage_per_share = slippage_per_share

        # Store reference to momentum signals
        self._momentum_signals = momentum_signals

        # Broker identity for multi-broker state tracking
        self._broker_name = broker_name

        # Initialize state manager for multi-strategy coordination
        self.state_manager = StrategyStateManager()

        # Initialize portfolio health checker with state manager for multi-strategy support
        self.health_checker = PortfolioHealthChecker(
            broker=broker,
            min_buying_power=5000.0,
            min_portfolio_value=10000.0,
            max_positions=top_n + 5,  # Allow some buffer
            max_position_age_hours=48,
            state_manager=self.state_manager
        )

        # Track last risk signals
        self._last_risk_signals: Optional[RiskSignals] = None

        logger.info("[MP] Momentum Strategy Configuration:")
        logger.info(f"[MP]   Top N stocks: {top_n}")
        logger.info(f"[MP]   Position size: {position_size:.0%}")
        logger.info(f"[MP]   Reduced exposure: {reduced_exposure:.0%}")
        logger.info(f"[MP]   VIX threshold: {vix_threshold}")
        logger.info(f"[MP]   Rebalance time: 3:55 PM EST")
        logger.info(f"[MP]   Portfolio health checks: ENABLED")

    @property
    def broker_name(self) -> str:
        return self._broker_name

    def _load_sp500_symbols(self) -> List[str]:
        """Load S&P 500 symbols from CSV, excluding leveraged ETFs."""
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        csv_path = project_root / 'config/universes' / 'sp500-2025.csv'

        try:
            import pandas as pd
            symbols_df = pd.read_csv(csv_path)
            symbols = symbols_df['Symbol'].tolist()

            # Filter out any leveraged ETFs (to avoid conflict with OMR)
            original_count = len(symbols)
            symbols = [s for s in symbols if not ETFUniverse.is_leveraged(s)]
            filtered_count = original_count - len(symbols)

            if filtered_count > 0:
                logger.info(f"[MP] Filtered out {filtered_count} leveraged ETFs from universe")

            return symbols
        except Exception as e:
            logger.error(f"[MP] Failed to load S&P 500 symbols: {e}")
            # Return a minimal default list (no leveraged ETFs)
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

    def preload_historical_data(self) -> None:
        """
        Pre-load historical data for momentum calculation.

        Fetches via Alpaca:
        1. Daily prices for all symbols (252+ days for momentum)
        2. SPY prices for drawdown calculation
        3. VIX prices for fear signals (via yfinance - Alpaca doesn't provide VIX)
        """
        logger.info("[MP] Pre-loading historical data for momentum strategy...")

        try:
            end_date = tz.now()
            start_date = end_date - timedelta(days=self.data_lookback_days)

            # Fetch historical data for all symbols via Alpaca
            logger.info(f"[MP] Fetching {len(self.symbols)} symbols from {start_date.date()} to {end_date.date()}")

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
                    logger.info(f"[MP] Fetched {min(i + batch_size, len(self.symbols))}/{len(self.symbols)} symbols...")

            if not prices_dict:
                logger.error("[MP] Failed to download historical price data from Alpaca")
                return

            # Create prices DataFrame
            prices_df = pd.DataFrame(prices_dict)

            if failed_symbols:
                logger.warning(f"[MP] Failed to fetch {len(failed_symbols)} symbols: {failed_symbols[:10]}...")

            logger.info(f"[MP] Downloaded {len(prices_df.columns)} symbols, {len(prices_df)} days via Alpaca")

            # Fetch SPY via Alpaca
            spy_data = self.broker.get_historical_bars(
                symbol='SPY',
                start=start_date,
                end=end_date,
                timeframe='1D'
            )

            # Fetch VIX via VIXProvider with fallback chain (yfinance -> FRED -> cache)
            logger.info("[MP] Fetching VIX via VIXProvider (not available on Alpaca)")
            vix_data = self._fetch_vix_data(lookback_days=self.data_lookback_days)

            # Extract close prices
            if spy_data is not None and not spy_data.empty:
                spy_data.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in spy_data.columns]
                spy_prices = spy_data['close'] if 'close' in spy_data.columns else pd.Series()
            else:
                spy_prices = pd.Series()
                logger.warning("[MP] Failed to fetch SPY data")

            if vix_data is not None and not vix_data.empty:
                # yfinance returns 'Close' (capitalized) after MultiIndex flattening
                if 'Close' in vix_data.columns:
                    vix_prices = vix_data['Close']
                elif 'close' in vix_data.columns:
                    vix_prices = vix_data['close']
                else:
                    vix_prices = pd.Series()
                    logger.warning(f"[MP] VIX data has unexpected columns: {list(vix_data.columns)}")
            else:
                vix_prices = pd.Series()
                logger.warning("[MP] Failed to fetch VIX data")

            # Update cache in momentum signals
            self._momentum_signals.update_historical_data(prices_df, spy_prices, vix_prices)

            # Cache for base adapter
            self._data_cache = {
                'prices': prices_df,
                'SPY': spy_data,
                'VIX': vix_data
            }
            self._cache_date = end_date

            logger.success(f"[MP] Historical data pre-loaded: {len(prices_df.columns)} symbols")
            logger.info(f"[MP]   SPY data: {len(spy_prices)} days")
            logger.info(f"[MP]   VIX data: {len(vix_prices)} days")

        except Exception as e:
            logger.error(f"[MP] Failed to pre-load historical data: {e}")
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
                logger.info(f"[MP] VIX data from {source}: {len(vix_data)} days")

                # Log warning if using cached data
                if source == "cache":
                    logger.warning("[MP] Using cached VIX data (may be stale)")

                # VIXProvider returns 'close' column, convert to DataFrame with 'Close' for compatibility
                result = pd.DataFrame({'Close': vix_data['close']})
                return result
            else:
                logger.error("[MP] All VIX data sources failed!")
                return None

        except Exception as e:
            logger.error(f"[MP] Failed to fetch VIX data: {e}")
            return None

    def fetch_todays_closes(self) -> bool:
        """
        Fetch only today's close prices and append to historical cache.

        This is a lightweight fetch at 3:55 PM that only gets today's data
        for the universe symbols, rather than re-fetching all historical data.

        Returns:
            True if successful, False otherwise
        """
        logger.info("[MP] " + "=" * 60)
        logger.info("[MP] FETCHING TODAY'S CLOSES (3:55 PM)")
        logger.info("[MP] " + "=" * 60)

        try:
            # Check if we have historical data to append to
            if self._data_cache is None or 'prices' not in self._data_cache:
                logger.warning("[MP] No historical cache - falling back to full fetch")
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
                    logger.info("[MP] Already have today's data in cache")
                    return True

            logger.info(f"[MP] Fetching today's closes for {len(self.symbols)} symbols...")

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
                except (ValueError, IndexError, KeyError):
                    failed += 1
                    continue

            if not todays_prices:
                logger.error("[MP] Failed to fetch any today's prices")
                return False

            logger.info(f"[MP] Fetched {len(todays_prices)}/{len(self.symbols)} symbols ({failed} failed)")

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
                logger.warning(f"[MP] Failed to fetch today's SPY: {e}")

            # Fetch today's VIX via VIXProvider
            vix_close = None
            try:
                vix_provider = get_vix_provider()
                current_vix = vix_provider.get_current_vix()
                if current_vix is not None:
                    vix_close = current_vix
                    logger.info(f"[MP] Today's VIX: {vix_close:.2f}")
            except Exception as e:
                logger.warning(f"[MP] Failed to fetch today's VIX: {e}")

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

            # Update momentum signals cache
            spy_prices = self._data_cache['SPY']['close'] if 'SPY' in self._data_cache and self._data_cache['SPY'] is not None else pd.Series()
            vix_col = 'Close' if 'VIX' in self._data_cache and self._data_cache['VIX'] is not None and 'Close' in self._data_cache['VIX'].columns else 'close'
            vix_prices = self._data_cache['VIX'][vix_col] if 'VIX' in self._data_cache and self._data_cache['VIX'] is not None else pd.Series()

            self._momentum_signals.update_historical_data(new_prices_df, spy_prices, vix_prices)

            logger.success(f"[MP] Appended today's data - cache now has {len(new_prices_df)} days")
            return True

        except Exception as e:
            logger.error(f"[MP] Failed to fetch today's closes: {e}")
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
                logger.info("[MP] Using cached historical data")

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
                logger.warning("[MP] No cached data available, fetching from yfinance...")
                self.preload_historical_data()
                return self.fetch_market_data()

            logger.info(f"[MP] Market data prepared: {len(market_data)} symbols")
            return market_data

        except Exception as e:
            logger.error(f"[MP] Error in fetch_market_data: {e}")
            return {}

    def run_once(self) -> None:
        """Run one iteration of the strategy with portfolio health checks.

        Emits a DecisionRecord at end (clean run, blocked, or errored).
        """
        import traceback

        logger.info("[MP] " + "=" * 60)
        logger.info(f"[MP] Running {self.__class__.__name__} at {tz.now()}")
        logger.info("[MP] " + "=" * 60)

        # Fetch today's closes before lock acquisition - preserves original run_once ordering
        # (original code fetched data before acquiring the execution lock)
        logger.info("[MP] Fetching today's closes for 3:55 PM execution...")
        self.fetch_todays_closes()

        rec = self._begin_decision("scheduled_rebalance", schedule_time="15:55")
        had_lock = False
        try:
            # ---- Preconditions ----
            with self._stage(rec, "preconditions"):
                if not self._check_common_preconditions(rec):
                    if not rec.preconditions.strategy_enabled.passed:
                        logger.warning("[MP] Strategy is DISABLED - skipping execution")
                    elif not rec.preconditions.shutdown_requested.passed:
                        logger.warning("[MP] Shutdown requested - skipping new entries")
                    elif not rec.preconditions.execution_lock_acquired.passed:
                        logger.error("[MP] Failed to acquire execution lock - another strategy is running")
                    return
                had_lock = True

                # Sync state with broker (detect external position changes)
                broker_positions = {p['symbol']: int(p['quantity']) for p in self.broker.get_positions()}
                changes = self.state_manager.sync_with_broker(self._broker_name, broker_positions)
                if changes['removed']:
                    logger.info(f"[MP] Detected closed positions: {changes['removed']}")

                # Strategy-specific health check
                from src.trading.decision_log.record import GateResult
                logger.info("[MP] Running pre-entry portfolio health check...")
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
                    logger.error("[MP] Portfolio health check FAILED - BLOCKING ENTRY")
                    for err in health_result.errors:
                        logger.error(f"[MP]   - {err}")
                    return

                if health_result.warnings:
                    logger.warning("[MP] Portfolio health check passed with warnings:")
                    for warning in health_result.warnings:
                        logger.warning(f"[MP]   - {warning}")

                logger.success("[MP] Portfolio health check PASSED - proceeding with rebalance")

                # Data freshness gate
                cache_complete = self._estimate_cache_freshness()
                rec.preconditions.data_freshness = GateResult(
                    passed=cache_complete >= 0.80,
                    details={"valid_pct": cache_complete * 100},
                )

            # ---- Inputs ----
            with self._stage(rec, "inputs"):
                rec.inputs = self._build_decision_inputs()

            # ---- Logic ----
            with self._stage(rec, "logic"):
                mp_positions = self.state_manager.get_positions(STRATEGY_NAME)
                positions = self.broker.get_positions()
                current_positions = {}
                for pos in positions:
                    symbol = pos.get('symbol')
                    owner = self.state_manager.symbol_owned_by_other(STRATEGY_NAME, symbol)
                    if symbol in mp_positions or owner is None:
                        value = float(pos.get('market_value', 0))
                        current_positions[symbol] = value
                self.strategy.set_current_positions(current_positions)

                market_data = self.fetch_market_data()
                if not market_data:
                    logger.error("[MP] No market data available")
                    return

                signals = self.strategy.generate_signals(market_data, tz.now())

                if signals:
                    risk_exposure = signals[0].metadata.get('risk_exposure', 1.0) if signals[0].metadata else 1.0
                    if risk_exposure < 1.0:
                        logger.warning(f"[MP] Risk signals active - exposure reduced to {risk_exposure:.0%}")

                rec.logic_decisions = self._build_decision_logic(signals)

            # ---- Execution ----
            with self._stage(rec, "execution"):
                self._execute_rebalance(signals, current_positions)

            # ---- Post-state ----
            with self._stage(rec, "post_state"):
                rec.post_state = self._snapshot_post_state()
                self.state_manager.update_last_execution(STRATEGY_NAME)

        except Exception as e:
            from src.trading.decision_log.record import ErrorInfo
            rec.error = ErrorInfo(
                type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc(),
                stage=getattr(rec, "_current_stage", "unknown"),
            )
            logger.error(f"[MP] Error in run_once: {e}")
            traceback.print_exc()
            raise
        finally:
            try:
                if had_lock:
                    self.state_manager.release_execution_lock(STRATEGY_NAME)
            finally:
                self._write_decision(rec)

    def _estimate_cache_freshness(self) -> float:
        """Fraction of universe symbols with valid data on the cache's last row."""
        if self._data_cache is None or "prices" not in self._data_cache:
            return 0.0
        prices = self._data_cache["prices"]
        if len(prices) == 0:
            return 0.0
        last_row = prices.iloc[-1]
        if len(last_row) == 0:
            return 0.0
        return float(last_row.notna().sum()) / float(len(last_row))

    def _build_decision_inputs(self):
        """Build StrategyInputs for the decision record."""
        from src.trading.decision_log.record import StrategyInputs

        cache_completeness = self._estimate_cache_freshness() * 100

        momentum_scores: dict = {}
        try:
            scores = self._momentum_signals.calculate_momentum_scores(
                self._data_cache["prices"] if self._data_cache else None
            )
            if scores is not None:
                momentum_scores = {
                    s: (None if pd.isna(v) else float(v)) for s, v in scores.items()
                }
        except Exception:
            pass

        return StrategyInputs(
            regime=None,
            regime_confidence=None,
            regime_params=None,
            vix=None,
            spy_drawdown_pct=None,
            universe_size=len(self.symbols) if self.symbols else 0,
            data_completeness_pct=cache_completeness,
            cache_source="memory_cache",
            momentum_scores=momentum_scores,
            extra={},
        )

    def _build_decision_logic(self, signals):
        """Build LogicDecisions from a list of signals."""
        from src.trading.decision_log.record import LogicDecisions

        buy = [s for s in signals if getattr(s, "direction", None) == "BUY"]
        sell = [s for s in signals if getattr(s, "direction", None) == "SELL"]
        hold = [s for s in signals if getattr(s, "direction", None) == "HOLD"]

        target_symbols = [s.symbol for s in buy]
        target_weights = {
            s.symbol: float(s.metadata.get("weight", 0.0)) if s.metadata else 0.0
            for s in buy
        }
        target_value_usd = {
            sym: w * self.initial_capital for sym, w in target_weights.items()
        }
        return LogicDecisions(
            top_n=self.top_n if hasattr(self, "top_n") else 10,
            target_symbols=target_symbols,
            target_weights=target_weights,
            target_value_usd=target_value_usd,
            reduce_exposure=False,
            exposure_pct=1.0,
            exit_signals=[s.symbol for s in sell],
            hold_signals=[s.symbol for s in hold],
            skip_reasons={},
        )

    def _snapshot_post_state(self):
        """Build PostState reflecting positions and equity after execution.

        Enriches via PriceOracle for live unrealized_pnl. See
        ramp_live_adapter._snapshot_post_state for rationale.
        """
        from src.trading.decision_log.record import PostState, PositionSnapshot

        mp_positions = self.state_manager.get_positions(STRATEGY_NAME)
        raw_positions = self.broker.get_positions() or []
        oracle = getattr(self, '_price_oracle', None)
        if oracle is not None:
            raw_positions = oracle.enrich_positions(raw_positions)
        broker_positions = {p.get("symbol"): p for p in raw_positions}
        positions_after: dict = {}
        for sym in mp_positions.keys():
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
        equity_after = self.initial_capital + sum(
            p.unrealized_pnl for p in positions_after.values()
        )
        return PostState(
            positions_after=positions_after,
            cash_after=cash_after,
            strategy_equity_after=equity_after,
            state_writes=[],
        )

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
                logger.error("[MP] Portfolio value is zero or negative")
                return

            logger.info(f"[MP] Portfolio value: ${portfolio_value:,.2f}")

            # Separate buy and sell signals
            buy_signals = [s for s in signals if s.direction == 'BUY']
            sell_signals = [s for s in signals if s.direction == 'SELL']

            # Execute sells first
            for signal in sell_signals:
                symbol = signal.symbol
                if symbol in current_positions:
                    try:
                        pos = next((p for p in self.broker.get_positions() if p['symbol'] == symbol), None)
                        if pos:
                            qty = int(pos['quantity'])
                            logger.info(f"[MP] Selling {symbol}: {qty} shares (exiting position)")

                            order = self.execution_engine.execute_order(
                                symbol=symbol,
                                quantity=abs(qty),
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET
                            )

                            if order:
                                logger.success(f"[MP] Sell order placed: {symbol}")

                                # Log trade exit to persistent trade log BEFORE removing position
                                # Get entry info from state manager while it still exists
                                try:
                                    position_info = self.state_manager.get_positions(STRATEGY_NAME).get(symbol, {})
                                    trade_logger = get_trade_log_writer()
                                    fill_price = order.get('filled_avg_price', pos.get('current_price', 0))
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
                                    logger.error(f"[MP] Trade logging failed (non-blocking): {log_err}")

                                # Remove from state tracking
                                self.state_manager.remove_position(STRATEGY_NAME, symbol)
                    except Exception as e:
                        logger.error(f"[MP] Error selling {symbol}: {e}")

            # Execute buys
            for signal in buy_signals:
                symbol = signal.symbol
                weight = signal.metadata.get('weight', signal.confidence) if signal.metadata else signal.confidence
                target_value = portfolio_value * self.position_size * weight * self.top_n

                # Skip if already at target
                current_value = current_positions.get(symbol, 0)
                if abs(target_value - current_value) < 100:  # $100 threshold
                    continue

                try:
                    # Get current price
                    quote = self.broker.get_latest_quote(symbol)
                    if not quote:
                        logger.warning(f"[MP] No quote available for {symbol}")
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
                            logger.warning(f"[MP] Skipping {symbol} - owned by {owner}")
                            continue

                        logger.info(
                            f"[MP] Buying {symbol}: {shares_to_buy} shares @ ${current_price:.2f} "
                            f"(rank #{signal.metadata.get('rank', '?')})"
                        )

                        order = self.execution_engine.execute_order(
                            symbol=symbol,
                            quantity=shares_to_buy,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET
                        )

                        if order:
                            logger.success(f"[MP] Buy order placed: {symbol}")
                            # Use add_or_update_position to handle both new positions AND top-ups
                            # CRITICAL: This prevents state drift when topping up existing positions
                            order_id = order.get('order_id')
                            self.state_manager.add_or_update_position(
                                STRATEGY_NAME, symbol, shares_to_buy, current_price, order_id,
                                broker=self._broker_name
                            )

                            # Log trade entry to persistent trade log
                            # Error handling ensures logging failures don't block trading
                            try:
                                trade_logger = get_trade_log_writer()
                                fill_price = order.get('filled_avg_price', current_price)
                                trade_logger.log_entry(
                                    strategy=STRATEGY_NAME,
                                    symbol=symbol,
                                    qty=shares_to_buy,
                                    price=float(fill_price) if fill_price else current_price,
                                    order_id=order_id,
                                    metadata={
                                        'rank': signal.metadata.get('rank') if signal.metadata else None,
                                        'momentum_score': signal.metadata.get('momentum_score') if signal.metadata else None
                                    }
                                )
                            except Exception as log_err:
                                logger.error(f"[MP] Trade logging failed (non-blocking): {log_err}")

                except Exception as e:
                    logger.error(f"[MP] Error buying {symbol}: {e}")

            logger.info("[MP] Rebalance execution complete")

        except Exception as e:
            logger.error(f"[MP] Error in _execute_rebalance: {e}")

    def get_schedule(self) -> Dict[str, any]:
        """
        Get scheduling configuration.

        Momentum rebalances once daily at 3:55 PM EST.
        Uses today's momentum (known at 3:55 PM) to select stocks,
        then holds overnight until next day's 3:55 PM rebalance.
        """
        return {
            'execution_times': [
                {'time': '15:55', 'action': 'rebalance'}  # 3:55 PM EST - Rebalance near close
            ],
            'market_hours_only': True,
            'strategy_type': 'daily'  # Indicates daily rebalancing
        }

    def show_current_signals(self) -> None:
        """Display current momentum signals and risk status."""
        try:
            # Get current positions
            positions = self.broker.get_positions()
            current_positions = {p['symbol']: float(p['market_value']) for p in positions}

            # Generate signals
            signals, risk_signals = self._momentum_signals.generate_signals(
                current_positions=current_positions
            )

            logger.info("\n" + "=" * 60)
            logger.info("[MP] CURRENT MOMENTUM SIGNALS")
            logger.info("=" * 60)

            # Risk status
            logger.info("\n[MP] Risk Signals:")
            logger.info(f"[MP]   VIX > {self.vix_threshold}: {'YES' if risk_signals.high_vix else 'NO'}")
            logger.info(f"[MP]   VIX Spike: {'YES' if risk_signals.vix_spike else 'NO'}")
            logger.info(f"[MP]   SPY Drawdown: {'YES' if risk_signals.spy_drawdown else 'NO'}")
            logger.info(f"[MP]   High Mom Vol: {'YES' if risk_signals.high_mom_vol else 'NO'}")
            logger.info(f"[MP]   Exposure: {risk_signals.exposure_pct:.0%}")

            # Top momentum stocks
            logger.info(f"\n[MP] Top {self.top_n} Momentum Stocks:")
            buy_signals = [s for s in signals if s.action in ('buy', 'hold')]
            for s in sorted(buy_signals, key=lambda x: x.rank):
                logger.info(f"[MP]   #{s.rank}: {s.symbol} (score: {s.momentum_score:.2%})")

            # Sell signals
            sell_signals = [s for s in signals if s.action == 'sell']
            if sell_signals:
                logger.info("\n[MP] Positions to Exit:")
                for s in sell_signals:
                    logger.info(f"[MP]   {s.symbol}")

        except Exception as e:
            logger.error(f"[MP] Error showing signals: {e}")


if __name__ == "__main__":
    logger.info("[MP] Momentum Protection Live Trading Adapter")
    logger.info("[MP] " + "=" * 60)
    logger.info("[MP] Rebalances at 3:55 PM EST based on:")
    logger.info("[MP]   - 1m-1w momentum rankings")
    logger.info("[MP]   - Rule-based crash protection")
    logger.info("")
    logger.info("[MP] Risk signals that reduce exposure:")
    logger.info("[MP]   - VIX > 25")
    logger.info("[MP]   - VIX 5-day spike > 20%")
    logger.info("[MP]   - SPY drawdown > 5%")
    logger.info("[MP]   - Momentum volatility > 90th percentile")
