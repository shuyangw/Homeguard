"""
Momentum Protection Live Trading Adapter.

Connects momentum strategy with crash protection to live trading infrastructure.
Rebalances daily at 3:55 PM EST based on momentum rankings and risk signals.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

from src.trading.adapters.strategy_adapter import StrategyAdapter
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
        for ms in momentum_signals:
            if ms.action == 'buy':
                signals.append(Signal(
                    symbol=ms.symbol,
                    direction='long',
                    strength=ms.weight,
                    metadata={
                        'momentum_score': ms.momentum_score,
                        'rank': ms.rank,
                        'risk_exposure': risk_signals.exposure_pct
                    }
                ))
            elif ms.action == 'sell':
                signals.append(Signal(
                    symbol=ms.symbol,
                    direction='exit',
                    strength=1.0,
                    metadata={'action': 'sell'}
                ))

        return signals


class MomentumLiveAdapter(StrategyAdapter):
    """
    Live trading adapter for Momentum Protection strategy.

    Rebalances at 3:55 PM EST based on:
    - 12-1 month momentum rankings
    - Rule-based crash protection signals

    Positions are held until next day's rebalance.
    """

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
        slippage_per_share: float = 0.01
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
            data_lookback_days=data_lookback_days
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

        # Initialize portfolio health checker
        self.health_checker = PortfolioHealthChecker(
            broker=broker,
            min_buying_power=5000.0,
            min_portfolio_value=10000.0,
            max_positions=top_n + 5,  # Allow some buffer
            max_position_age_hours=48
        )

        # Initialize state manager for multi-strategy coordination
        self.state_manager = StrategyStateManager()

        # Track last risk signals
        self._last_risk_signals: Optional[RiskSignals] = None

        logger.info("[MP] Momentum Strategy Configuration:")
        logger.info(f"[MP]   Top N stocks: {top_n}")
        logger.info(f"[MP]   Position size: {position_size:.0%}")
        logger.info(f"[MP]   Reduced exposure: {reduced_exposure:.0%}")
        logger.info(f"[MP]   VIX threshold: {vix_threshold}")
        logger.info(f"[MP]   Rebalance time: 3:55 PM EST")
        logger.info(f"[MP]   Portfolio health checks: ENABLED")

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

            # Fetch VIX via yfinance (Alpaca doesn't provide VIX)
            logger.info("[MP] Fetching VIX via yfinance (not available on Alpaca)")
            vix_data = self._fetch_vix_yfinance(start_date, end_date)

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

    def _fetch_vix_yfinance(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch VIX data via yfinance (Alpaca doesn't provide VIX).

        Args:
            start_date: Start date (datetime or string)
            end_date: End date (datetime or string)

        Returns:
            DataFrame with VIX data, or None if fetch fails
        """
        try:
            # Convert dates to string format for yfinance
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)[:10]

            if hasattr(end_date, 'strftime'):
                # Add 1 day to end_date to include today's data
                end_dt = end_date + timedelta(days=1)
                end_str = end_dt.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)[:10]

            vix_data = yf.download(
                '^VIX',
                start=start_str,
                end=end_str,
                progress=False,
                auto_adjust=True
            )

            if vix_data is None or vix_data.empty:
                logger.error("[MP] yfinance returned empty VIX data")
                return None

            logger.info(f"[MP] Fetched {len(vix_data)} days of VIX data via yfinance")

            # Handle MultiIndex columns from yfinance (e.g., ('Close', '^VIX'))
            if isinstance(vix_data.columns, pd.MultiIndex):
                # Flatten MultiIndex by taking first level
                vix_data.columns = vix_data.columns.get_level_values(0)
                logger.info("[MP] Flattened MultiIndex columns from yfinance")

            # Ensure timezone-aware index
            if vix_data.index.tz is None:
                vix_data.index = vix_data.index.tz_localize('America/New_York')
            else:
                vix_data.index = vix_data.index.tz_convert('America/New_York')

            return vix_data

        except Exception as e:
            logger.error(f"[MP] Failed to fetch VIX via yfinance: {e}")
            import traceback
            traceback.print_exc()
            return None

    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch current market data for signal generation.

        Uses cached historical data + today's data from broker.
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
        """
        Run one iteration of the strategy with portfolio health checks.

        Includes:
        - Toggle/shutdown checks
        - Execution lock for multi-strategy coordination
        - Position tracking per strategy
        """
        logger.info("[MP] " + "=" * 60)
        logger.info(f"[MP] Running {self.__class__.__name__} at {tz.now()}")
        logger.info("[MP] " + "=" * 60)

        try:
            # Check if strategy is enabled
            if not self.state_manager.is_enabled(STRATEGY_NAME):
                logger.warning("[MP] Strategy is DISABLED - skipping execution")
                return

            # Check if shutdown requested
            if self.state_manager.is_shutdown_requested(STRATEGY_NAME):
                logger.warning("[MP] Shutdown requested - skipping new entries")
                return

            # Acquire execution lock (blocks if another strategy is executing)
            if not self.state_manager.acquire_execution_lock(STRATEGY_NAME):
                logger.error("[MP] Failed to acquire execution lock - another strategy is running")
                return

            try:
                # Sync state with broker (detect external position changes)
                broker_positions = {p['symbol']: int(p['quantity']) for p in self.broker.get_positions()}
                changes = self.state_manager.sync_with_broker(broker_positions)
                if changes['removed']:
                    logger.info(f"[MP] Detected closed positions: {changes['removed']}")

                # Portfolio health check before entry
                logger.info("[MP] Running pre-entry portfolio health check...")
                health_result = self.health_checker.check_before_entry(
                    required_capital=None,
                    allow_existing_positions=True
                )

                if not health_result.passed:
                    logger.error("[MP] Portfolio health check FAILED - BLOCKING ENTRY")
                    for error in health_result.errors:
                        logger.error(f"[MP]   - {error}")
                    return

                if health_result.warnings:
                    logger.warning("[MP] Portfolio health check passed with warnings:")
                    for warning in health_result.warnings:
                        logger.warning(f"[MP]   - {warning}")

                logger.success("[MP] Portfolio health check PASSED - proceeding with rebalance")

                # Get MP's own positions from state manager
                mp_positions = self.state_manager.get_positions(STRATEGY_NAME)

                # Update current positions in signal generator (using broker positions for value)
                positions = self.broker.get_positions()
                current_positions = {}
                for pos in positions:
                    symbol = pos.get('symbol')
                    # Only include if tracked by MP or not owned by another strategy
                    owner = self.state_manager.symbol_owned_by_other(STRATEGY_NAME, symbol)
                    if symbol in mp_positions or owner is None:
                        value = float(pos.get('market_value', 0))
                        current_positions[symbol] = value

                self.strategy.set_current_positions(current_positions)

                # Fetch market data
                market_data = self.fetch_market_data()

                if not market_data:
                    logger.error("[MP] No market data available")
                    return

                # Generate signals
                signals = self.strategy.generate_signals(market_data, tz.now())

                # Log risk signals
                if signals:
                    risk_exposure = signals[0].metadata.get('risk_exposure', 1.0) if signals[0].metadata else 1.0
                    if risk_exposure < 1.0:
                        logger.warning(f"[MP] Risk signals active - exposure reduced to {risk_exposure:.0%}")

                # Execute trades
                self._execute_rebalance(signals, current_positions)

                # Update last execution timestamp
                self.state_manager.update_last_execution(STRATEGY_NAME)

            finally:
                # Always release execution lock
                self.state_manager.release_execution_lock(STRATEGY_NAME)

        except Exception as e:
            logger.error(f"[MP] Error in run_once: {e}")
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
                logger.error("[MP] Portfolio value is zero or negative")
                return

            logger.info(f"[MP] Portfolio value: ${portfolio_value:,.2f}")

            # Separate buy and sell signals
            buy_signals = [s for s in signals if s.direction == 'long']
            sell_signals = [s for s in signals if s.direction == 'exit']

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
                                # Remove from state tracking
                                self.state_manager.remove_position(STRATEGY_NAME, symbol)
                    except Exception as e:
                        logger.error(f"[MP] Error selling {symbol}: {e}")

            # Execute buys
            for signal in buy_signals:
                symbol = signal.symbol
                target_value = portfolio_value * self.position_size * signal.strength * self.top_n

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
                            # Add to state tracking
                            order_id = order.get('order_id')
                            self.state_manager.add_position(
                                STRATEGY_NAME, symbol, shares_to_buy, current_price, order_id
                            )

                except Exception as e:
                    logger.error(f"[MP] Error buying {symbol}: {e}")

            logger.info("[MP] Rebalance execution complete")

        except Exception as e:
            logger.error(f"[MP] Error in _execute_rebalance: {e}")

    def get_schedule(self) -> Dict[str, any]:
        """
        Get scheduling configuration.

        Momentum rebalances once daily at 3:55 PM EST.
        """
        return {
            'execution_times': [
                {'time': '15:55', 'action': 'rebalance'}  # 3:55 PM - Rebalance
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
    logger.info("[MP]   - 12-1 month momentum rankings")
    logger.info("[MP]   - Rule-based crash protection")
    logger.info("")
    logger.info("[MP] Risk signals that reduce exposure:")
    logger.info("[MP]   - VIX > 25")
    logger.info("[MP]   - VIX 5-day spike > 20%")
    logger.info("[MP]   - SPY drawdown > 5%")
    logger.info("[MP]   - Momentum volatility > 90th percentile")
