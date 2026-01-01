"""
CSCM Paper Trading Adapter.

Paper trading variant of the CSCM strategy for testing and validation.

Key Differences from CSCMLiveAdapter:
- Uses Alpaca for live price data (with Binance fallback)
- Uses Alpaca paper trading for order execution
- Rebalances on Monday (not Sunday)
- Integrates with PortfolioTracker for value monitoring

Usage:
    adapter = CSCMPaperAdapter()
    adapter.run()  # Main loop with hourly checks

    # Or run once for testing
    adapter.run_once()

    # Get current status
    status = adapter.get_status()
"""

from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional
import pickle

import pandas as pd

from src.data.providers.binance import CryptoDataProviderWithFallback
from src.strategies.advanced.cscm_signals import CSCMSignals
from src.trading.adapters.cscm_live_adapter import CSCMLiveAdapter
from src.trading.brokers.alpaca_crypto_broker import AlpacaCryptoBroker
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger(__name__)

# Strategy identifier
STRATEGY_NAME = 'cscm-paper'

# Cache configuration
CACHE_DIR = Path.home() / '.homeguard' / 'cache'
PAPER_CACHE_FILE = CACHE_DIR / 'cscm_paper_state.pkl'


class CSCMPaperAdapter(CSCMLiveAdapter):
    """
    Paper trading adapter for CSCM strategy.

    Extends CSCMLiveAdapter with:
    - Alpaca data provider for live prices (Binance fallback)
    - Alpaca paper trading for execution
    - Monday rebalancing schedule
    - Portfolio tracking during equity hours

    The adapter uses Alpaca for momentum signal calculation and price data,
    with Binance as a fallback. Executes trades through Alpaca's paper trading.

    Note on Price Sources:
    - Primary: Alpaca crypto API (BTCUSD)
    - Fallback: Binance (BTCUSDT converted to USD)
    - Small price differences (0.1-1%) may occur between sources

    Usage:
        adapter = CSCMPaperAdapter(
            top_n=5,
            rebalance_day='monday'
        )

        # Check status
        status = adapter.get_status()
        print(f"Regime: {status['regime']}")

        # Run rebalance if needed
        if adapter._should_rebalance():
            adapter.rebalance()
    """

    # Override default rebalance day and time
    DEFAULT_REBALANCE_DAY = 'monday'
    DEFAULT_REBALANCE_HOUR_UTC = 21  # 21:30 UTC
    DEFAULT_REBALANCE_MINUTE_UTC = 30

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        top_n: int = CSCMLiveAdapter.DEFAULT_TOP_N,
        momentum_period: int = CSCMLiveAdapter.DEFAULT_MOMENTUM_PERIOD,
        btc_sma_period: int = CSCMLiveAdapter.DEFAULT_BTC_SMA_PERIOD,
        trailing_stop_pct: float = CSCMLiveAdapter.DEFAULT_TRAILING_STOP,
        rebalance_day: str = DEFAULT_REBALANCE_DAY,
        rebalance_hour_utc: int = DEFAULT_REBALANCE_HOUR_UTC,
        rebalance_minute_utc: int = DEFAULT_REBALANCE_MINUTE_UTC,
        go_to_cash_in_bear: bool = True,
        **kwargs
    ):
        """
        Initialize CSCM paper trading adapter.

        Args:
            universe: Crypto symbols to trade (defaults to CSCM standard universe)
            top_n: Number of top coins to hold (default 7)
            momentum_period: Days for momentum calculation (default 28)
            btc_sma_period: BTC SMA period for regime (default 40)
            trailing_stop_pct: Trailing stop percentage (default 0.25 = 25%)
            rebalance_day: Day of week to rebalance (default 'monday')
            rebalance_hour_utc: Hour in UTC to rebalance (default 21)
            rebalance_minute_utc: Minute in UTC to rebalance (default 30)
            go_to_cash_in_bear: Exit all in bearish regime (default True)
            **kwargs: Additional arguments passed to parent
        """
        # Set paper-specific cache file BEFORE parent init (parent calls _load_state)
        self._cache_file = PAPER_CACHE_FILE

        # Create Alpaca paper trading broker
        broker = AlpacaCryptoBroker(paper=True)

        # Initialize parent with paper=True and our broker
        super().__init__(
            universe=universe,
            top_n=top_n,
            momentum_period=momentum_period,
            btc_sma_period=btc_sma_period,
            trailing_stop_pct=trailing_stop_pct,
            rebalance_day=rebalance_day,
            go_to_cash_in_bear=go_to_cash_in_bear,
            broker=broker,
            paper=True
        )

        # Store rebalance time (UTC)
        self.rebalance_hour_utc = rebalance_hour_utc
        self.rebalance_minute_utc = rebalance_minute_utc
        self.rebalance_time_utc = time(rebalance_hour_utc, rebalance_minute_utc)

        # Override data provider with Binance+fallback
        self._data_provider = CryptoDataProviderWithFallback()

        # Portfolio tracker (lazy init)
        self._portfolio_tracker = None

        logger.info(f"[CSCM Paper] Initialized paper trading adapter")
        logger.info(f"[CSCM Paper]   Data Source: Alpaca (Binance fallback)")
        logger.info(f"[CSCM Paper]   Execution: Alpaca Paper Trading")
        logger.info(f"[CSCM Paper]   Rebalance: {rebalance_day} at {rebalance_hour_utc:02d}:{rebalance_minute_utc:02d} UTC")

    def _fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data using Alpaca with Binance fallback.

        Overrides parent to use our composite data provider.
        """
        end = datetime.now()
        start = end - timedelta(days=60)  # Need enough for momentum + SMA

        try:
            data_dict = self._data_provider.get_historical_bars_batch(
                self.universe,
                start,
                end,
                '1D'
            )
            logger.info(f"[CSCM Paper] Fetched data for {len(data_dict)} symbols")
            return data_dict

        except Exception as e:
            logger.error(f"[CSCM Paper] Failed to fetch data: {e}")
            return {}

    def _get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices from Alpaca with Binance fallback.

        Returns:
            Dict mapping symbol -> current price
        """
        return self._data_provider.get_current_prices(self.universe)

    def _get_alpaca_quote(self, symbol: str) -> float:
        """
        Get current quote from Alpaca for execution.

        We use Alpaca quotes at execution time to ensure paper fills
        match what would happen in real trading.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')

        Returns:
            Current mid price from Alpaca
        """
        try:
            quote = self.broker.get_crypto_quote(symbol)
            return quote['last']
        except Exception as e:
            logger.warning(f"[CSCM Paper] Failed to get Alpaca quote for {symbol}: {e}")
            # Fall back to Binance price
            return self._data_provider.get_current_price(symbol) or 0.0

    def _should_rebalance(self) -> bool:
        """
        Check if we should rebalance now.

        Overrides parent to check specific time (21:30 UTC on Monday).
        """
        # Get current time in UTC
        now_utc = datetime.now(timezone.utc)

        # Check if it's the right day
        day_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        target_day = day_map.get(self.rebalance_day.lower(), 0)

        if now_utc.weekday() != target_day:
            return False

        # Check if we're at or past the rebalance time
        current_time = now_utc.time()
        if current_time < self.rebalance_time_utc:
            logger.debug(
                f"[CSCM Paper] Waiting for rebalance time "
                f"(current: {current_time.strftime('%H:%M')} UTC, "
                f"target: {self.rebalance_time_utc.strftime('%H:%M')} UTC)"
            )
            return False

        # Check if we already rebalanced today
        if self._last_rebalance:
            last_date = self._last_rebalance.date() if hasattr(self._last_rebalance, 'date') else self._last_rebalance
            today = now_utc.date()
            if last_date == today:
                logger.debug("[CSCM Paper] Already rebalanced today")
                return False

        logger.info(
            f"[CSCM Paper] Rebalance window: {self.rebalance_day} "
            f"{self.rebalance_hour_utc:02d}:{self.rebalance_minute_utc:02d} UTC"
        )
        return True

    def _load_state(self) -> None:
        """Load saved state from paper-specific cache file."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, 'rb') as f:
                    state = pickle.load(f)
                    self._last_rebalance = state.get('last_rebalance')
                    self._current_positions = state.get('positions', {})
                    # Load trailing stop state
                    peak_value = state.get('peak_value', 0.0)
                    if peak_value > 0:
                        self.signals._peak_portfolio_value = peak_value
                    self.signals._trailing_stop_triggered = state.get('trailing_stop_triggered', False)
                    logger.info(f"[CSCM Paper] Loaded state: last rebalance {self._last_rebalance}")
        except Exception as e:
            logger.warning(f"[CSCM Paper] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to paper-specific cache file."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, 'wb') as f:
                pickle.dump({
                    'last_rebalance': self._last_rebalance,
                    'positions': self._current_positions,
                    'saved_at': tz.now(),
                    'peak_value': self.signals._peak_portfolio_value,
                    'trailing_stop_triggered': self.signals._trailing_stop_triggered,
                }, f)
            logger.debug("[CSCM Paper] Saved state to disk")
        except Exception as e:
            logger.warning(f"[CSCM Paper] Failed to save state: {e}")

    def get_portfolio_value(self, force_refresh: bool = False) -> float:
        """
        Get current portfolio value.

        Args:
            force_refresh: If True, refresh even outside equity hours

        Returns:
            Total portfolio value in USD
        """
        return self._get_total_portfolio_value()

    def get_portfolio_tracker(self):
        """
        Get or create portfolio tracker.

        Lazy initialization to avoid import cycles.
        """
        if self._portfolio_tracker is None:
            try:
                from src.trading.portfolio.tracker import PortfolioTracker
                self._portfolio_tracker = PortfolioTracker(self)
            except ImportError:
                logger.warning("[CSCM Paper] PortfolioTracker not available")
                return None
        return self._portfolio_tracker

    def get_status(self) -> Dict:
        """
        Get current strategy status with paper trading info.

        Extends parent status with paper-specific data.
        """
        status = super().get_status()

        # Add paper trading info
        status['mode'] = 'paper'
        status['data_source'] = 'Alpaca (Binance fallback)'
        status['execution'] = 'Alpaca Paper'

        # Add price comparison (for debugging)
        try:
            if 'BTC/USD' in self.universe:
                binance_price = self._data_provider.get_current_price('BTC/USD')
                alpaca_price = self._get_alpaca_quote('BTC/USD')
                if binance_price and alpaca_price:
                    spread_pct = abs(binance_price - alpaca_price) / alpaca_price * 100
                    status['btc_price_spread_pct'] = round(spread_pct, 4)
        except Exception:
            pass

        return status

    def run_once(self) -> None:
        """
        Run single iteration of the paper trading strategy.

        Extended to update portfolio tracker if available.
        """
        # Run parent logic
        super().run_once()

        # Update portfolio tracker if available
        tracker = self.get_portfolio_tracker()
        if tracker:
            try:
                tracker.update_value()
            except Exception as e:
                logger.debug(f"[CSCM Paper] Portfolio tracker update failed: {e}")


def create_paper_adapter(**kwargs) -> CSCMPaperAdapter:
    """
    Factory function to create CSCM paper adapter with defaults.

    Args:
        **kwargs: Override any default parameters

    Returns:
        Configured CSCMPaperAdapter instance
    """
    defaults = {
        'top_n': 5,
        'rebalance_day': 'monday',
        'trailing_stop_pct': 0.25,
    }
    defaults.update(kwargs)
    return CSCMPaperAdapter(**defaults)
