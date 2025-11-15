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
from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
from src.strategies.universe import ETFUniverse
from src.trading.brokers.broker_interface import BrokerInterface
from src.trading.utils.portfolio_health_check import PortfolioHealthChecker
from src.utils.logger import logger


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
            logger.info(f"Using default OMR universe: {len(symbols)} leveraged 3x ETFs")

        # Initialize regime detector if not provided
        if regime_detector is None:
            regime_detector = MarketRegimeDetector()
            logger.info("Created new MarketRegimeDetector (untrained)")

        # Initialize Bayesian model if not provided
        if bayesian_model is None:
            bayesian_model = BayesianReversionModel()
            logger.info("Created new BayesianReversionModel (untrained)")

        # Create pure OMR strategy with injected symbols
        strategy = OvernightReversionSignals(
            regime_detector=regime_detector,
            bayesian_model=bayesian_model,
            symbols=symbols,  # ✅ Inject symbols instead of using hardcoded list
            min_probability=min_probability,
            min_expected_return=min_expected_return,
            max_positions=max_positions
        )

        # OMR needs intraday data, so need more lookback
        data_lookback_days = 365

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

        # Initialize portfolio health checker
        self.health_checker = PortfolioHealthChecker(
            broker=broker,
            min_buying_power=1000.0,
            min_portfolio_value=5000.0,
            max_positions=max_positions,
            max_position_age_hours=48
        )

        logger.info("OMR Strategy Configuration:")
        logger.info(f"  Min probability: {min_probability:.1%}")
        logger.info(f"  Min expected return: {min_expected_return:.2%}")
        logger.info(f"  Signal time: 3:50 PM EST")
        logger.info(f"  Entry: 3:50 PM | Exit: Next day 9:31 AM")
        logger.info(f"  Portfolio health checks: ENABLED")

    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday market data for OMR strategy.

        OMR needs intraday bars to calculate intraday moves.
        Uses pre-fetched intraday cache if available (3:45 PM pre-fetch).
        """
        try:
            import pandas as pd
            from datetime import timedelta

            market_data = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.data_lookback_days)

            # Check if intraday cache is available
            intraday_cache_available = (
                self._intraday_cache is not None and
                len(self._intraday_cache) > 0
            )

            if intraday_cache_available:
                logger.info("Using pre-fetched intraday data cache")

                # Use cached intraday data for symbols
                for symbol in self.symbols:
                    if symbol in self._intraday_cache:
                        market_data[symbol] = self._intraday_cache[symbol]
                    else:
                        logger.warning(f"{symbol} not in intraday cache, fetching...")
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
                            logger.error(f"Error fetching {symbol}: {e}")

            else:
                logger.info("No intraday cache available, fetching data...")

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
                            logger.warning(f"No intraday data returned for {symbol}")

                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {e}")
                        continue

            # Also need historical daily data for regime detection
            # Use base class cache if available
            if self._data_cache is not None:
                logger.info("Using cached historical data for regime detection")
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
                                logger.info("Fetching VIX data via yfinance (Alpaca does not provide VIX)")
                                df = self._fetch_vix_yfinance(start_date, end_date)
                                if df is not None and not df.empty:
                                    market_data[market_symbol] = df
                                    logger.info(f"[OK] Fetched {len(df)} days of VIX data via yfinance")
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
                            logger.error(f"Error fetching {market_symbol}: {e}")
                            # VIX errors are already handled in _fetch_vix_yfinance

            cache_status = "cached intraday" if intraday_cache_available else "live fetch"
            logger.info(
                f"Fetched data for {len(market_data)} symbols ({cache_status})"
            )
            return market_data

        except Exception as e:
            logger.error(f"Error in fetch_market_data: {e}")
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
                logger.error("yfinance returned empty VIX data")
                return None

            # Ensure timezone-aware index for consistency with broker data
            if vix_data.index.tz is None:
                vix_data.index = vix_data.index.tz_localize('America/New_York')
            else:
                vix_data.index = vix_data.index.tz_convert('America/New_York')

            return vix_data

        except Exception as e:
            logger.error(f"Failed to fetch VIX via yfinance: {e}")
            return None

    def run_once(self) -> None:
        """
        Run one iteration of the strategy with portfolio health checks.

        Overrides base class to add pre-entry health validation.
        """
        logger.info("=" * 60)
        logger.info(f"Running {self.__class__.__name__} at {datetime.now()}")
        logger.info("=" * 60)

        try:
            # CRITICAL: Portfolio health check before entry
            logger.info("Running pre-entry portfolio health check...")
            health_result = self.health_checker.check_before_entry(
                required_capital=None,  # Will be calculated after signals generated
                allow_existing_positions=True  # OMR can have multiple concurrent positions
            )

            # Check for critical errors
            if not health_result.passed:
                logger.error("Portfolio health check FAILED - BLOCKING ENTRY")
                logger.error("Critical errors detected:")
                for error in health_result.errors:
                    logger.error(f"  - {error}")
                logger.info("Skipping signal generation and order execution")
                return

            # Log warnings if any
            if health_result.warnings:
                logger.warning("Portfolio health check passed with warnings:")
                for warning in health_result.warnings:
                    logger.warning(f"  - {warning}")

            # Health check passed - proceed with normal strategy execution
            logger.success("Portfolio health check PASSED - proceeding with entry")
            logger.info("")

            # Call parent's run_once() for normal strategy execution
            super().run_once()

        except Exception as e:
            logger.error(f"Error in run_once: {e}")
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
        """
        try:
            now = datetime.now()
            if now.time() < time(9, 30) or now.time() > time(9, 35):
                logger.warning(
                    f"close_overnight_positions called at {now.time()}, "
                    "expected 9:31 AM"
                )

            # CRITICAL: Portfolio health check before exit
            logger.info("Running pre-exit portfolio health check...")
            health_result = self.health_checker.check_before_exit()

            # Check for critical errors
            if not health_result.passed:
                logger.error("Portfolio health check FAILED - CRITICAL ERRORS DETECTED")
                logger.error("Critical errors:")
                for error in health_result.errors:
                    logger.error(f"  - {error}")
                logger.warning("Attempting to close positions despite errors (safety measure)")

            # Log warnings if any
            if health_result.warnings:
                logger.warning("Portfolio health check warnings:")
                for warning in health_result.warnings:
                    logger.warning(f"  - {warning}")

            # Get all open positions
            positions = self.broker.get_positions()

            if not positions:
                logger.info("No overnight positions to close")
                return

            logger.info(f"Closing {len(positions)} overnight positions at market open")

            for position in positions:
                try:
                    # Calculate P&L
                    entry_price = float(position.avg_entry_price)
                    current_price = float(position.current_price)
                    qty = int(position.qty)

                    pnl = (current_price - entry_price) * qty
                    pnl_pct = (current_price - entry_price) / entry_price * 100

                    logger.info(
                        f"Closing {position.symbol}: {qty} shares "
                        f"@ ${entry_price:.2f} → ${current_price:.2f} "
                        f"(P&L: ${pnl:+.2f}, {pnl_pct:+.2f}%)"
                    )

                    # Place market order to close
                    side = 'sell' if qty > 0 else 'buy'
                    order = self.execution_engine.place_market_order(
                        symbol=position.symbol,
                        qty=abs(qty),
                        side=side
                    )

                    if order:
                        logger.success(f"Close order placed: {order.id}")
                    else:
                        logger.error(f"Failed to close {position.symbol}")

                except Exception as e:
                    logger.error(f"Error closing {position.symbol}: {e}")
                    continue

            logger.info("Overnight position closing complete")

        except Exception as e:
            logger.error(f"Error in close_overnight_positions: {e}")


if __name__ == "__main__":
    logger.info("OMR (Overnight Mean Reversion) Live Trading Adapter")
    logger.info("=" * 60)
    logger.info("Generates signals at 3:50 PM EST based on:")
    logger.info("  - Market regime (bull/bear/choppy)")
    logger.info("  - Intraday price movements")
    logger.info("  - Bayesian reversion probabilities")
    logger.info("")
    logger.info("Entry: 3:50 PM EST")
    logger.info("Exit: Next day 9:31 AM EST")
    logger.info("")
    logger.info("Default universe: Leveraged 3x ETFs")
    logger.info("  (TQQQ, SQQQ, UPRO, SPXU, TMF, TMV, etc.)")
