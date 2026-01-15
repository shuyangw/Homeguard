"""
CSCM Demo Trading Adapter.

Uses DemoBroker with Binance streaming for simulated execution.
Self-contained paper trading without external platform dependency.

Key Features:
- Real-time Binance WebSocket streaming
- Simulated execution (slippage, fees, latency)
- Fractional crypto quantities
- 24/7 operation (optional equity hours mode)
- Weekly rebalance based on momentum

Usage:
    adapter = CSCMDemoAdapter(
        universe=['BTC/USD', 'ETH/USD', ...],
        top_n=7,
        initial_cash=10000.0,
    )

    # In main loop
    if adapter._should_rebalance():
        adapter.rebalance()
    else:
        adapter.run_once()
"""

from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pickle

import pandas as pd

from src.data.providers.binance import BinanceDataProvider
from src.strategies.advanced.cscm_signals import CSCMSignals
from src.trading.demo import DemoBroker
from src.trading.brokers.interfaces import OrderSide
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger(__name__)

# Strategy identifier
STRATEGY_NAME = 'cscm-demo'

# Cache configuration
CACHE_DIR = Path.home() / '.homeguard' / 'cache'
CACHE_FILE = CACHE_DIR / 'cscm_demo_state.pkl'

# Optimal universe (14 coins) - based on backtesting results
# BTC used for regime detection only, not traded
DEFAULT_DEMO_UNIVERSE = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'LINK/USD',
    'DOGE/USD', 'DOT/USD', 'LTC/USD', 'BCH/USD', 'UNI/USD',
    'AAVE/USD', 'SUSHI/USD', 'XRP/USD', 'CRV/USD'
]


class CSCMDemoAdapter:
    """
    Demo trading adapter for CSCM strategy.

    Uses DemoBroker with Binance streaming for self-contained
    paper trading. Supports fractional quantities and 24/7 operation.

    Optimal Configuration (from backtesting):
    - Top N: 5 positions (concentration improves returns)
    - Allocation: 40% of capital (35% CAGR with agreement filter)
    - Trailing Stop: 8% (balances protection vs whipsaw)
    - Profit Target: 20% (locks in gains)
    - Market Hours: Equity hours only (9:30 AM - 4:00 PM ET)

    Attributes:
        broker: DemoBroker instance with Binance streaming
        signals: CSCMSignals for momentum calculation
        universe: List of symbols to trade
        top_n: Number of positions to hold
    """

    # Optimal defaults from backtesting (35% CAGR with agreement filter, 16.6% max DD)
    DEFAULT_TOP_N = 5
    DEFAULT_MOMENTUM_PERIOD = 28
    DEFAULT_BTC_SMA_PERIOD = 40
    DEFAULT_TRAILING_STOP = 0.08  # 8% trailing stop
    DEFAULT_ALLOCATION = 0.40  # 40% of capital in positions
    DEFAULT_PROFIT_TARGET = 0.20  # 20% profit target
    DEFAULT_INITIAL_CASH = 100000.0  # $100k starting capital

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        top_n: int = DEFAULT_TOP_N,
        momentum_period: int = DEFAULT_MOMENTUM_PERIOD,
        btc_sma_period: int = DEFAULT_BTC_SMA_PERIOD,
        trailing_stop_pct: float = DEFAULT_TRAILING_STOP,
        allocation_pct: float = DEFAULT_ALLOCATION,
        profit_target_pct: float = DEFAULT_PROFIT_TARGET,
        rebalance_day: str = 'sunday',
        go_to_cash_in_bear: bool = True,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        slippage_bps: float = 5.0,
        fee_bps: float = 10.0,
        market_hours_only: bool = True,  # Default to equity hours
        broker: Optional[DemoBroker] = None,
    ):
        """
        Initialize CSCM demo adapter.

        Args:
            universe: Crypto symbols to trade
            top_n: Number of top coins to hold
            momentum_period: Days for momentum calculation
            btc_sma_period: BTC SMA period for regime
            trailing_stop_pct: Trailing stop percentage (0.08 = 8%)
            allocation_pct: Percentage of capital to allocate (0.18 = 18%)
            profit_target_pct: Profit target to take gains (0.20 = 20%)
            rebalance_day: Day of week to rebalance
            go_to_cash_in_bear: Exit all in bearish regime
            initial_cash: Starting cash (if no saved state)
            slippage_bps: Slippage in basis points
            fee_bps: Fee in basis points
            market_hours_only: Restrict to equity market hours (default: True)
            broker: DemoBroker instance (auto-created if None)
        """
        self.universe = universe or DEFAULT_DEMO_UNIVERSE
        self.top_n = top_n
        self.momentum_period = momentum_period
        self.btc_sma_period = btc_sma_period
        self.trailing_stop_pct = trailing_stop_pct
        self.allocation_pct = allocation_pct
        self.profit_target_pct = profit_target_pct
        self.rebalance_day = rebalance_day
        self.go_to_cash_in_bear = go_to_cash_in_bear
        self.market_hours_only = market_hours_only
        self._cache_file = CACHE_FILE

        # Track entry prices for profit target
        self._entry_prices: Dict[str, float] = {}

        # Initialize signal generator
        self.signals = CSCMSignals(
            symbols=self.universe,
            top_n=top_n,
            momentum_period=momentum_period,
            btc_sma_period=btc_sma_period,
            rebalance_day=rebalance_day,
            go_to_cash_in_bear=go_to_cash_in_bear,
            trailing_stop_pct=trailing_stop_pct
        )

        # Initialize broker
        if broker is not None:
            self._broker = broker
        else:
            self._broker = DemoBroker(
                initial_cash=initial_cash,
                slippage_bps=slippage_bps,
                fee_bps=fee_bps,
            )

        # State tracking
        self._last_rebalance: Optional[datetime] = None
        self._current_positions: Dict[str, Decimal] = {}
        self._streaming_started = False

        # REST API provider for historical data (40-day SMA needs more than streaming buffer)
        self._data_provider = BinanceDataProvider()

        # Load saved state
        self._load_state()

        logger.info(f"[CSCMDemo] Initialized demo adapter (OPTIMAL CONFIG)")
        logger.info(f"[CSCMDemo]   Universe: {len(self.universe)} symbols")
        logger.info(f"[CSCMDemo]   Top N: {top_n}")
        logger.info(f"[CSCMDemo]   Allocation: {allocation_pct:.0%}")
        logger.info(f"[CSCMDemo]   Trailing Stop: {trailing_stop_pct:.0%}")
        logger.info(f"[CSCMDemo]   Profit Target: {profit_target_pct:.0%}")
        logger.info(f"[CSCMDemo]   BTC SMA Period: {btc_sma_period}")
        logger.info(f"[CSCMDemo]   Rebalance Day: {rebalance_day}")
        logger.info(f"[CSCMDemo]   Market Hours Only: {market_hours_only}")
        logger.info(f"[CSCMDemo]   Initial Cash: ${initial_cash:,.2f}")
        logger.info(f"[CSCMDemo]   Historical Data: Binance.US REST API")
        logger.info(f"[CSCMDemo]   Real-time Quotes: Binance.US WebSocket (optional)")

    @property
    def broker(self) -> DemoBroker:
        """Get the demo broker."""
        return self._broker

    def start_streaming(self) -> None:
        """Start Binance WebSocket streaming."""
        if not self._streaming_started:
            logger.info("[CSCMDemo] Starting Binance streaming...")
            self._broker.start_streaming(self.universe)
            self._streaming_started = True

    def stop_streaming(self) -> None:
        """Stop streaming."""
        if self._streaming_started:
            logger.info("[CSCMDemo] Stopping streaming...")
            self._broker.stop_streaming()
            self._streaming_started = False

    def _load_state(self) -> None:
        """Load saved state from disk."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, 'rb') as f:
                    state = pickle.load(f)
                    self._last_rebalance = state.get('last_rebalance')
                    self._current_positions = state.get('positions', {})
                    self._entry_prices = state.get('entry_prices', {})
                    # Load trailing stop state
                    peak_value = state.get('peak_value', 0.0)
                    if peak_value > 0:
                        self.signals._peak_portfolio_value = peak_value
                    self.signals._trailing_stop_triggered = state.get('trailing_stop_triggered', False)
                    logger.info(f"[CSCMDemo] Loaded state: last rebalance {self._last_rebalance}")
                    if self.signals._trailing_stop_triggered:
                        logger.warning("[CSCMDemo] Trailing stop was triggered in previous session")
        except Exception as e:
            logger.warning(f"[CSCMDemo] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, 'wb') as f:
                pickle.dump({
                    'last_rebalance': self._last_rebalance,
                    'positions': self._current_positions,
                    'entry_prices': self._entry_prices,
                    'saved_at': tz.now(),
                    'peak_value': self.signals._peak_portfolio_value,
                    'trailing_stop_triggered': self.signals._trailing_stop_triggered,
                }, f)
            logger.debug("[CSCMDemo] Saved state to disk")
        except Exception as e:
            logger.warning(f"[CSCMDemo] Failed to save state: {e}")

    def _fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical daily data from Binance REST API.

        Uses BinanceDataProvider to fetch 60 days of daily klines for
        momentum calculation (28 days) and BTC SMA (40 days).

        Returns:
            Dict mapping symbol to DataFrame with daily OHLCV data
        """
        end = datetime.now()
        start = end - timedelta(days=60)  # Need 60 days for momentum + SMA

        try:
            data_dict = self._data_provider.get_historical_bars_batch(
                self.universe,
                start,
                end,
                '1D'
            )

            if data_dict:
                logger.info(
                    f"[CSCMDemo] Fetched {len(data_dict)}/{len(self.universe)} "
                    f"symbols from Binance REST API"
                )
            else:
                logger.warning("[CSCMDemo] No data returned from Binance REST API")

            return data_dict

        except Exception as e:
            logger.error(f"[CSCMDemo] Failed to fetch historical data: {e}")
            return {}

    def _get_current_positions(self) -> Dict[str, Decimal]:
        """Get current positions from demo broker."""
        try:
            positions = self._broker.get_crypto_positions()
            pos_dict = {}
            for pos in positions:
                symbol = pos['symbol']
                if symbol in self.universe:
                    pos_dict[symbol] = pos['quantity']
            return pos_dict
        except Exception as e:
            logger.error(f"[CSCMDemo] Failed to get positions: {e}")
            return {}

    def _get_account_balance(self) -> float:
        """Get available cash balance."""
        return self._broker.cash

    def _get_total_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        return self._broker.get_portfolio_value()

    def _is_market_hours(self) -> bool:
        """Check if within equity market hours (if enabled)."""
        if not self.market_hours_only:
            return True  # 24/7 mode

        now = tz.now()
        # US equity hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        # Check weekday (Mon=0, Sun=6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        return market_open <= now <= market_close

    def _check_trailing_stop(self) -> bool:
        """
        Check and update trailing stop.

        Returns:
            True if stop was triggered and positions should be closed
        """
        portfolio_value = self._get_total_portfolio_value()
        if portfolio_value <= 0:
            return False

        # Update portfolio value in signals (this checks trailing stop)
        self.signals.update_portfolio_value(portfolio_value)

        # If stop just triggered, close all positions
        if self.signals.is_trailing_stop_active():
            current_positions = self._get_current_positions()
            if current_positions:
                logger.warning(f"[CSCMDemo] Trailing stop triggered - closing all positions")
                for symbol in list(current_positions.keys()):
                    try:
                        self._broker.close_crypto_position(symbol)
                        logger.info(f"[CSCMDemo] Closed {symbol} due to trailing stop")
                    except Exception as e:
                        logger.error(f"[CSCMDemo] Failed to close {symbol}: {e}")
                self._save_state()
                return True

        return False

    def _should_rebalance(self) -> bool:
        """Check if we should rebalance now."""
        now = tz.now()

        # Check market hours if enabled
        if self.market_hours_only and not self._is_market_hours():
            return False

        # Check if it's rebalance day
        if not self.signals.should_rebalance(now):
            return False

        # Check if we already rebalanced today
        if self._last_rebalance:
            last_date = self._last_rebalance.date() if hasattr(self._last_rebalance, 'date') else self._last_rebalance
            today = now.date()
            if last_date == today:
                logger.debug("[CSCMDemo] Already rebalanced today")
                return False

        return True

    def rebalance(self) -> List[Dict]:
        """
        Execute rebalancing.

        Returns:
            List of executed orders
        """
        logger.info("[CSCMDemo] Starting rebalance...")

        # Fetch latest data from buffer
        data_dict = self._fetch_historical_data()
        if not data_dict or 'BTC/USD' not in data_dict:
            logger.error("[CSCMDemo] Failed to fetch required data, skipping rebalance")
            return []

        # Update signals with latest data
        self.signals.update_historical_data(data_dict)

        # Get risk signals and target positions
        risk_signals = self.signals.get_risk_signals()
        target_positions = self.signals.get_target_positions()

        logger.info(f"[CSCMDemo] Regime: {risk_signals.regime}")
        logger.info(f"[CSCMDemo] BTC: ${risk_signals.btc_price:,.2f} (SMA: ${risk_signals.btc_sma:,.2f})")
        logger.info(f"[CSCMDemo] Top symbols: {risk_signals.top_symbols}")

        # Get current positions and balance
        current_positions = self._get_current_positions()
        available_balance = self._get_account_balance()

        logger.info(f"[CSCMDemo] Available balance: ${available_balance:,.2f}")
        logger.info(f"[CSCMDemo] Current positions: {list(current_positions.keys())}")

        # Calculate total portfolio value and allocated capital
        total_value = self._get_total_portfolio_value()
        allocated_capital = total_value * self.allocation_pct
        logger.info(f"[CSCMDemo] Total portfolio value: ${total_value:,.2f}")
        logger.info(f"[CSCMDemo] Allocated capital ({self.allocation_pct:.0%}): ${allocated_capital:,.2f}")

        orders = []

        # Close positions not in target
        for symbol in list(current_positions.keys()):
            if target_positions.get(symbol, 0) == 0:
                try:
                    logger.info(f"[CSCMDemo] Closing position: {symbol}")
                    order = self._broker.close_crypto_position(symbol)
                    orders.append(order)
                    # Clear entry price
                    self._entry_prices.pop(symbol, None)
                except Exception as e:
                    logger.error(f"[CSCMDemo] Failed to close {symbol}: {e}")

        # Open/adjust positions in target
        # Use allocated capital (18%) divided equally among top N
        per_position_value = allocated_capital / self.top_n

        for symbol, weight in target_positions.items():
            if weight == 0:
                continue

            target_value = per_position_value  # Equal weight within allocation
            current_qty = current_positions.get(symbol, Decimal('0'))

            try:
                quote = self._broker.get_crypto_quote(symbol)
                if quote is None or not quote.get('available'):
                    logger.warning(f"[CSCMDemo] No quote for {symbol}, skipping")
                    continue

                current_price = quote['last']
                target_qty = Decimal(str(target_value / current_price))

                # Calculate difference
                qty_diff = target_qty - current_qty

                # Skip small adjustments (< 5% of target)
                if abs(qty_diff) < target_qty * Decimal('0.05'):
                    continue

                if qty_diff > 0:
                    # Buy
                    logger.info(f"[CSCMDemo] Buying {qty_diff:.6f} {symbol}")
                    order = self._broker.place_crypto_order(
                        symbol=symbol,
                        quantity=qty_diff,
                        side=OrderSide.BUY,
                    )
                    orders.append(order)
                    # Track entry price for profit target
                    self._entry_prices[symbol] = current_price

                elif qty_diff < 0:
                    # Sell
                    sell_qty = abs(qty_diff)
                    logger.info(f"[CSCMDemo] Selling {sell_qty:.6f} {symbol}")
                    order = self._broker.place_crypto_order(
                        symbol=symbol,
                        quantity=sell_qty,
                        side=OrderSide.SELL,
                    )
                    orders.append(order)

            except Exception as e:
                logger.error(f"[CSCMDemo] Failed to rebalance {symbol}: {e}")

        # Mark rebalance complete
        self._last_rebalance = tz.now()
        self.signals.mark_rebalanced(self._last_rebalance)
        self._current_positions = self._get_current_positions()
        self._save_state()

        logger.info(f"[CSCMDemo] Rebalance complete: {len(orders)} orders executed")
        return orders

    def get_status(self) -> Dict:
        """Get current strategy status."""
        try:
            data_dict = self._fetch_historical_data()
            if data_dict:
                self.signals.update_historical_data(data_dict)

            risk_signals = self.signals.get_risk_signals()
            target_positions = self.signals.get_target_positions()
            current_positions = self._get_current_positions()
            balance = self._get_account_balance()
            total_value = self._get_total_portfolio_value()

            return {
                'strategy': STRATEGY_NAME,
                'regime': risk_signals.regime,
                'btc_price': risk_signals.btc_price,
                'btc_sma': risk_signals.btc_sma,
                'is_rebalance_day': risk_signals.is_rebalance_day,
                'top_symbols': risk_signals.top_symbols,
                'target_positions': target_positions,
                'current_positions': {k: float(v) for k, v in current_positions.items()},
                'available_balance': balance,
                'total_value': total_value,
                'last_rebalance': str(self._last_rebalance) if self._last_rebalance else None,
                'trailing_stop_triggered': risk_signals.trailing_stop_triggered,
                'current_drawdown': risk_signals.current_drawdown,
                'peak_value': risk_signals.peak_value,
                'trailing_stop_pct': self.trailing_stop_pct,
                'market_hours_only': self.market_hours_only,
                'streaming': self._streaming_started,
            }
        except Exception as e:
            logger.error(f"[CSCMDemo] Failed to get status: {e}")
            return {'error': str(e)}

    def _check_profit_targets(self) -> List[Dict]:
        """
        Check if any positions hit profit target (20%).

        Returns:
            List of orders from closing profitable positions
        """
        orders = []
        positions = self._broker.get_crypto_positions()

        for pos in positions:
            symbol = pos['symbol']
            entry_price = self._entry_prices.get(symbol)

            if entry_price is None:
                continue

            current_price = pos['current_price']
            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct >= self.profit_target_pct:
                logger.info(
                    f"[CSCMDemo] Profit target hit for {symbol}: "
                    f"{pnl_pct:.1%} >= {self.profit_target_pct:.0%}"
                )
                try:
                    order = self._broker.close_crypto_position(symbol)
                    orders.append(order)
                    self._entry_prices.pop(symbol, None)
                    logger.info(f"[CSCMDemo] Closed {symbol} at profit target")
                except Exception as e:
                    logger.error(f"[CSCMDemo] Failed to close {symbol}: {e}")

        if orders:
            self._save_state()

        return orders

    def run_once(self) -> None:
        """Run single iteration of the strategy."""
        now = tz.now()
        logger.debug(f"[CSCMDemo] Run at {now}")

        try:
            # Check market hours if enabled (but allow Sunday rebalance)
            is_rebalance_day = self.signals.should_rebalance(now)
            if self.market_hours_only and not self._is_market_hours() and not is_rebalance_day:
                logger.debug("[CSCMDemo] Outside market hours, skipping")
                return

            # Check trailing stop first (may close positions)
            stop_triggered = self._check_trailing_stop()
            if stop_triggered:
                logger.info("[CSCMDemo] Trailing stop active - waiting for next rebalance")
                return

            # Check profit targets (may close individual positions)
            self._check_profit_targets()

            # Check if we should rebalance
            if self._should_rebalance():
                # Reset trailing stop on rebalance day (allows re-entry)
                self.signals.reset_trailing_stop()
                self.rebalance()

        except Exception as e:
            logger.error(f"[CSCMDemo] Error in run_once: {e}")

    def reset(self, initial_cash: float = DEFAULT_INITIAL_CASH) -> None:
        """Reset adapter state and portfolio."""
        self._broker.reset_portfolio(initial_cash=initial_cash)
        self._last_rebalance = None
        self._current_positions = {}
        self._entry_prices = {}
        self.signals.reset_trailing_stop()

        # Delete cache file
        try:
            if self._cache_file.exists():
                self._cache_file.unlink()
        except (OSError, FileNotFoundError, PermissionError):
            pass

        logger.info(f"[CSCMDemo] Reset complete with ${initial_cash:,.2f}")
