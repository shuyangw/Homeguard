"""
Multi-asset portfolio simulator for holding multiple positions simultaneously.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pytz
from datetime import time

from backtesting.utils.risk_config import RiskConfig
from backtesting.utils.position_sizer import FixedPercentageSizer
from backtesting.utils.portfolio_construction import (
    EqualWeightSizer,
    RiskParitySizer,
    FixedCountSizer,
    RankedSizer,
    AdaptiveWeightSizer
)
from backtesting.utils.rebalancing import (
    get_rebalancing_trigger,
    RebalancingTrigger,
    RebalancingExecutor,
    RebalancingEvent
)


@dataclass
class Position:
    """Represents an open position in the portfolio."""
    symbol: str
    shares: float
    entry_price: float
    entry_bar: int
    entry_timestamp: pd.Timestamp
    current_price: float = 0.0
    highest_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        """Original cost of position."""
        return self.shares * self.entry_price

    @property
    def pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis

    @property
    def pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.pnl / self.cost_basis) * 100


class MultiAssetPortfolio:
    """
    Portfolio that can hold multiple asset positions simultaneously.

    Features:
    - Track multiple open positions (dict keyed by symbol)
    - Cash management across positions
    - Position sizing (equal weight, risk parity, etc.)
    - Rebalancing (periodic, threshold-based, signal-based)
    - Portfolio-level metrics (Sharpe, drawdown, etc.)
    """

    def __init__(
        self,
        symbols: List[str],
        prices: pd.DataFrame,  # MultiIndex (timestamp, symbol) or wide format
        entries: pd.DataFrame,  # Columns = symbols
        exits: pd.DataFrame,    # Columns = symbols
        init_cash: float,
        fees: float,
        slippage: float = 0.0,
        freq: str = '1min',
        market_hours_only: bool = True,
        risk_config: Optional[RiskConfig] = None,
        position_sizing_method: str = 'equal_weight',
        max_positions: int = 10,
        rebalancing_frequency: str = 'never',
        price_data: Optional[pd.DataFrame] = None  # Full OHLCV data
    ):
        """
        Initialize multi-asset portfolio.

        Args:
            symbols: List of symbols to trade
            prices: Price data (close prices) for all symbols
            entries: Entry signals (boolean DataFrame, columns = symbols)
            exits: Exit signals (boolean DataFrame, columns = symbols)
            init_cash: Initial capital
            fees: Trading fees as decimal
            slippage: Slippage as decimal
            freq: Data frequency
            market_hours_only: Only trade during market hours
            risk_config: Risk management configuration
            position_sizing_method: 'equal_weight', 'risk_parity', 'fixed_count', 'ranked'
            max_positions: Maximum concurrent positions
            rebalancing_frequency: 'never', 'monthly', 'quarterly', 'on_signal'
            price_data: Full OHLCV data for all symbols
        """
        self.symbols = symbols
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        self.market_hours_only = market_hours_only
        self.risk_config = risk_config or RiskConfig.moderate()
        self.position_sizing_method = position_sizing_method
        self.max_positions = max_positions
        self.rebalancing_frequency = rebalancing_frequency
        self.price_data = price_data

        # Market hours
        self.market_open = time(9, 35)
        self.market_close = time(15, 55)
        self.eastern_tz = pytz.timezone('US/Eastern')

        # Align data across all symbols
        self.prices, self.entries, self.exits, self.unified_index = self._align_data(
            prices, entries, exits
        )

        # Portfolio state
        self.cash = init_cash
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []
        self.equity_curve: List[float] = []
        self.equity_timestamps: List[pd.Timestamp] = []
        self.rebalancing_events: List[Dict] = []

        # Time-series tracking for analytics
        self.position_count_history: List[Tuple[pd.Timestamp, int]] = []
        self.symbol_weights_history: List[Tuple[pd.Timestamp, Dict[str, float]]] = []
        self.cash_history: List[Tuple[pd.Timestamp, float]] = []

        # Initialize position sizer
        self._init_position_sizer()

        # Initialize rebalancing trigger
        self._init_rebalancing_trigger()

        # Run simulation
        self._simulate()

    def _align_data(
        self,
        prices: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        """
        Align data across all symbols to unified timestamp index.

        Handles:
        - Different trading times per symbol
        - Data gaps
        - Missing values

        Returns:
            Tuple of (aligned_prices, aligned_entries, aligned_exits, unified_index)
        """
        # Get union of all timestamps
        unified_index = prices.index.unique()

        # Reindex all DataFrames to unified index
        prices_aligned = prices.reindex(unified_index)
        entries_aligned = entries.reindex(unified_index)
        exits_aligned = exits.reindex(unified_index)

        # Forward-fill prices to handle gaps
        prices_aligned = prices_aligned.ffill()

        # Fill signal NaNs with False (no signal if data missing)
        entries_aligned = entries_aligned.fillna(False).astype(bool)
        exits_aligned = exits_aligned.fillna(False).astype(bool)

        return prices_aligned, entries_aligned, exits_aligned, unified_index

    def _init_position_sizer(self):
        """Initialize portfolio position sizer based on risk_config."""
        # Use portfolio_sizing_method from risk_config if available
        method = getattr(self.risk_config, 'portfolio_sizing_method', 'equal_weight')

        if method == 'equal_weight':
            self.portfolio_sizer = EqualWeightSizer(target_positions=self.max_positions)

        elif method == 'risk_parity':
            lookback = getattr(self.risk_config, 'risk_parity_lookback', 60)
            self.portfolio_sizer = RiskParitySizer(lookback=lookback)

        elif method == 'fixed_count':
            max_single_weight = getattr(self.risk_config, 'max_single_position_pct', 0.20)
            self.portfolio_sizer = FixedCountSizer(
                max_positions=self.max_positions,
                max_single_weight=max_single_weight
            )

        elif method == 'ranked':
            self.portfolio_sizer = RankedSizer(top_n=self.max_positions)

        elif method == 'adaptive':
            self.portfolio_sizer = AdaptiveWeightSizer(max_positions=self.max_positions)

        else:
            # Default to equal weight
            self.portfolio_sizer = EqualWeightSizer(target_positions=self.max_positions)

    def _init_rebalancing_trigger(self):
        """Initialize rebalancing trigger based on risk_config."""
        # Get rebalancing frequency from risk_config or use instance variable
        frequency = getattr(self.risk_config, 'rebalancing_frequency', self.rebalancing_frequency)
        threshold_pct = getattr(self.risk_config, 'rebalancing_threshold_pct', 0.05)

        self.rebalancing_trigger = get_rebalancing_trigger(
            frequency=frequency,
            threshold_pct=threshold_pct
        )

    def _is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within market hours."""
        if not self.market_hours_only:
            return True

        if timestamp.tz is None:
            eastern_time = timestamp
        else:
            eastern_time = timestamp.tz_convert(self.eastern_tz)

        if eastern_time.weekday() >= 5:
            return False

        current_time = eastern_time.time()
        return self.market_open <= current_time <= self.market_close

    def _calculate_portfolio_value(self, timestamp: pd.Timestamp) -> float:
        """Calculate total portfolio value (cash + positions)."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    def _get_position_weights(self) -> Dict[str, float]:
        """Get current position weights as % of portfolio."""
        portfolio_value = self._calculate_portfolio_value(
            self.equity_timestamps[-1] if self.equity_timestamps else pd.Timestamp.now()
        )

        weights = {}
        for symbol, position in self.positions.items():
            weights[symbol] = position.market_value / portfolio_value if portfolio_value > 0 else 0.0

        return weights

    def _update_position_prices(self, timestamp: pd.Timestamp):
        """Update current prices for all open positions."""
        for symbol, position in self.positions.items():
            if symbol in self.prices.columns:
                current_price = self.prices.loc[timestamp, symbol]
                if pd.notna(current_price):
                    position.current_price = current_price
                    position.highest_price = max(position.highest_price, current_price)

    def _check_exit_signals(
        self,
        timestamp: pd.Timestamp,
        bar_index: int
    ) -> Dict[str, str]:
        """
        Check exit signals for all open positions.

        Returns:
            Dict mapping symbol -> exit_reason
        """
        exit_symbols = {}

        for symbol in list(self.positions.keys()):
            # Check strategy exit signal
            if symbol in self.exits.columns:
                if self.exits.loc[timestamp, symbol]:
                    exit_symbols[symbol] = 'strategy_signal'

        # TODO: Check stop losses via RiskManager

        return exit_symbols

    def _execute_exit(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        exit_reason: str,
        bar_index: int
    ):
        """Execute exit for a position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        price = position.current_price

        # Apply slippage (worse fill on exit)
        slippage_adj = price * (1 - self.slippage)

        # Calculate proceeds
        proceeds = position.shares * slippage_adj
        fee = proceeds * self.fees
        net_proceeds = proceeds - fee

        # Update cash
        self.cash += net_proceeds

        # Calculate hold duration
        hold_duration = (timestamp - position.entry_timestamp).total_seconds() / 86400  # days

        # Record closed position
        self.closed_positions.append({
            'symbol': symbol,
            'entry_timestamp': position.entry_timestamp,
            'exit_timestamp': timestamp,
            'entry_price': position.entry_price,
            'exit_price': price,
            'shares': position.shares,
            'pnl': position.pnl,
            'pnl_pct': position.pnl_pct,
            'exit_reason': exit_reason,
            'hold_duration_days': hold_duration
        })

        # Remove position
        del self.positions[symbol]

    def _check_entry_signals(
        self,
        timestamp: pd.Timestamp,
        bar_index: int
    ) -> Dict[str, bool]:
        """
        Check entry signals and return dict of entry signals.

        Returns:
            Dict of {symbol: should_enter}
        """
        entry_signals = {}

        for symbol in self.symbols:
            # Skip if already in position
            if symbol in self.positions:
                entry_signals[symbol] = False
                continue

            # Check entry signal
            if symbol in self.entries.columns:
                entry_signals[symbol] = bool(self.entries.loc[timestamp, symbol])
            else:
                entry_signals[symbol] = False

        return entry_signals

    def _execute_batch_entries(
        self,
        entry_signals: Dict[str, bool],
        timestamp: pd.Timestamp,
        bar_index: int
    ):
        """
        Execute batch entries using portfolio sizer.

        Args:
            entry_signals: Dict of {symbol: should_enter}
            timestamp: Current timestamp
            bar_index: Current bar index
        """
        # Get current prices for all symbols
        prices_dict = {}
        for symbol in self.symbols:
            if symbol in self.prices.columns:
                price = self.prices.loc[timestamp, symbol]
                if pd.notna(price) and price > 0:
                    prices_dict[symbol] = price

        if not prices_dict:
            return

        # Calculate portfolio value
        portfolio_value = self._calculate_portfolio_value(timestamp)

        # Use portfolio sizer to calculate positions for all entry signals
        # TODO: Add support for volatilities, signal_strengths, etc. for advanced sizers
        kwargs = {'current_positions': len(self.positions)}

        positions_to_enter = self.portfolio_sizer.calculate_positions(
            portfolio_value=portfolio_value,
            entry_signals=entry_signals,
            prices=prices_dict,
            **kwargs
        )

        if not positions_to_enter:
            return

        # Execute entries (with cash management)
        for symbol, shares in positions_to_enter.items():
            if shares <= 0:
                continue

            if symbol not in prices_dict:
                continue

            price = prices_dict[symbol]

            # Apply slippage (worse fill on entry)
            slippage_adj = price * (1 + self.slippage)

            # Calculate cost
            cost = shares * slippage_adj
            fee = cost * self.fees
            total_cost = cost + fee

            # Check if we have enough cash
            if total_cost > self.cash:
                # Reduce shares to fit available cash
                if self.cash <= 0:
                    continue

                shares = (self.cash / (1 + self.fees)) / slippage_adj
                shares = int(shares)

                if shares <= 0:
                    continue

                cost = shares * slippage_adj
                fee = cost * self.fees
                total_cost = cost + fee

            # Create position
            position = Position(
                symbol=symbol,
                shares=shares,
                entry_price=price,
                entry_bar=bar_index,
                entry_timestamp=timestamp,
                current_price=price,
                highest_price=price
            )

            # Update cash and positions
            self.cash -= total_cost
            self.positions[symbol] = position

    def _simulate(self):
        """Run portfolio simulation."""
        bar_index = 0

        for timestamp in self.unified_index:
            bar_index += 1

            # Check market hours
            if not self._is_market_hours(timestamp):
                # Still update equity curve even when market closed
                portfolio_value = self._calculate_portfolio_value(timestamp)
                self.equity_curve.append(portfolio_value)
                self.equity_timestamps.append(timestamp)

                # Track time-series data for analytics (market closed)
                self.position_count_history.append((timestamp, len(self.positions)))
                self.cash_history.append((timestamp, self.cash))
                if self.positions:
                    weights = self._get_position_weights()
                    self.symbol_weights_history.append((timestamp, weights.copy()))
                continue

            # Update position prices
            self._update_position_prices(timestamp)

            # Check exits
            exit_signals = self._check_exit_signals(timestamp, bar_index)
            for symbol, exit_reason in exit_signals.items():
                self._execute_exit(symbol, timestamp, exit_reason, bar_index)

            # Check entries (batch execution using portfolio sizer)
            entry_signals = self._check_entry_signals(timestamp, bar_index)
            self._execute_batch_entries(entry_signals, timestamp, bar_index)

            # Check if rebalancing needed
            self._check_and_execute_rebalancing(timestamp, entry_signals)

            # Record portfolio value
            portfolio_value = self._calculate_portfolio_value(timestamp)
            self.equity_curve.append(portfolio_value)
            self.equity_timestamps.append(timestamp)

            # Track time-series data for analytics
            self.position_count_history.append((timestamp, len(self.positions)))
            self.cash_history.append((timestamp, self.cash))
            if self.positions:
                weights = self._get_position_weights()
                self.symbol_weights_history.append((timestamp, weights.copy()))

    def _check_and_execute_rebalancing(
        self,
        timestamp: pd.Timestamp,
        entry_signals: Dict[str, bool]
    ):
        """
        Check if rebalancing is needed and execute if triggered.

        Args:
            timestamp: Current timestamp
            entry_signals: Current entry signals (for signal-based rebalancing)
        """
        if self.rebalancing_trigger is None:
            return

        if not self.positions:
            return  # No positions to rebalance

        # Calculate current weights
        current_weights = self._get_position_weights()

        # Calculate target weights (equal weight across all positions for now)
        # TODO: Support other target weight schemes
        target_symbols = list(self.positions.keys())
        target_weight = 1.0 / len(target_symbols) if target_symbols else 0.0
        target_weights = {symbol: target_weight for symbol in target_symbols}

        # Check if should rebalance
        should_rebalance = self.rebalancing_trigger.should_rebalance(
            timestamp=timestamp,
            current_weights=current_weights,
            target_weights=target_weights,
            current_signals=entry_signals
        )

        if not should_rebalance:
            return

        # Execute rebalancing
        portfolio_value_before = self._calculate_portfolio_value(timestamp)

        # Get current positions and prices
        current_positions = {
            symbol: int(pos.shares)
            for symbol, pos in self.positions.items()
        }

        current_prices = {
            symbol: pos.current_price
            for symbol, pos in self.positions.items()
        }

        # Execute rebalancing
        new_positions, new_cash, trades, total_cost = RebalancingExecutor.execute_rebalancing(
            current_positions=current_positions,
            current_prices=current_prices,
            target_weights=target_weights,
            portfolio_value=portfolio_value_before,
            cash=self.cash,
            fees=self.fees,
            slippage=self.slippage
        )

        # Update positions
        for symbol, shares in new_positions.items():
            if symbol in self.positions:
                self.positions[symbol].shares = shares
            # Note: We don't create new positions here, only adjust existing ones

        # Remove positions with 0 shares
        symbols_to_remove = [
            symbol for symbol, pos in self.positions.items()
            if pos.shares <= 0
        ]
        for symbol in symbols_to_remove:
            del self.positions[symbol]

        # Update cash
        self.cash = new_cash

        # Record rebalancing event
        portfolio_value_after = self._calculate_portfolio_value(timestamp)

        rebalancing_event = {
            'timestamp': timestamp,
            'trigger': getattr(self.rebalancing_trigger, '__class__', type(self.rebalancing_trigger)).__name__,
            'trades': trades,
            'total_cost': total_cost,
            'portfolio_value_before': portfolio_value_before,
            'portfolio_value_after': portfolio_value_after
        }

        self.rebalancing_events.append(rebalancing_event)

    def stats(self) -> Optional[pd.Series]:
        """
        Calculate portfolio statistics.

        Returns:
            pd.Series with statistics (VectorBT-compatible format)
        """
        if not self.equity_curve or len(self.equity_curve) == 0:
            return None

        equity_series = pd.Series(self.equity_curve, index=self.equity_timestamps)

        # Total return
        total_return_pct = ((equity_series.iloc[-1] - self.init_cash) / self.init_cash) * 100

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        if len(returns) == 0:
            stats_dict = {
                'Total Return [%]': total_return_pct,
                'Start Value': self.init_cash,
                'End Value': equity_series.iloc[-1],
                'Total Trades': len(self.closed_positions)
            }
            return pd.Series(stats_dict)

        # Annualized return
        mean_return = returns.mean()
        std_return = returns.std()

        periods_per_year = self._get_periods_per_year()
        annual_return_pct = mean_return * periods_per_year * 100

        # Sharpe ratio
        if std_return > 0:
            sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0

        # Max drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown_pct = drawdown.min()

        # Win rate
        winning_trades = [t for t in self.closed_positions if t.get('pnl', 0) > 0]
        total_trades = len(self.closed_positions)
        win_rate_pct = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        stats_dict = {
            'Total Return [%]': total_return_pct,
            'Annual Return [%]': annual_return_pct,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown [%]': max_drawdown_pct,
            'Win Rate [%]': win_rate_pct,
            'Total Trades': total_trades,
            'Start Value': self.init_cash,
            'End Value': equity_series.iloc[-1],
            'Avg Positions': np.mean([len(self.positions)]),  # TODO: Track over time
        }

        return pd.Series(stats_dict)

    def _get_periods_per_year(self) -> int:
        """Get number of periods per year based on frequency."""
        freq_map = {
            '1min': 252 * 6.5 * 60,
            '1h': 252 * 6.5,
            '1d': 252,
            'D': 252,
            'daily': 252
        }
        return freq_map.get(self.freq, 252)

    @property
    def equity_curve_series(self) -> pd.Series:
        """Get equity curve as pandas Series."""
        return pd.Series(self.equity_curve, index=self.equity_timestamps)

    # VectorBT-Compatible Methods

    def value(self) -> pd.Series:
        """
        Get portfolio value over time (VectorBT-compatible).

        Returns:
            pd.Series with portfolio value indexed by timestamp
        """
        return pd.Series(self.equity_curve, index=self.equity_timestamps)

    def returns(self, freq: Optional[str] = None) -> pd.Series:
        """
        Get returns series (VectorBT-compatible).

        Args:
            freq: Frequency for resampling (e.g., '1H' for hourly, 'D' for daily)
                  Use this to downsample for performance with large datasets

        Returns:
            pd.Series with returns indexed by timestamp
        """
        # PERFORMANCE FIX: Support downsampling for large datasets
        if freq and len(self.equity_curve) > 1000:
            # Resample equity first, then calculate returns
            equity = self.value()
            resampled_equity = equity.resample(freq).last()
            return resampled_equity.pct_change().fillna(0)
        else:
            return self.value().pct_change().fillna(0)

    @property
    def trades(self) -> pd.DataFrame:
        """
        Get trades as DataFrame (VectorBT-compatible).

        Returns:
            pd.DataFrame with closed positions/trades
        """
        if not self.closed_positions:
            return pd.DataFrame()
        return pd.DataFrame(self.closed_positions)

    @property
    def wrapper(self):
        """
        Minimal wrapper for VectorBT compatibility.

        Provides access to index for tools that need it.
        """
        class MinimalWrapper:
            def __init__(self, index):
                self.index = pd.DatetimeIndex(index)

        return MinimalWrapper(self.equity_timestamps)

    def plot(self, **kwargs):
        """Plot equity curve (placeholder for compatibility)."""
        if self.equity_curve:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(self.equity_timestamps, self.equity_curve)
            plt.title('Multi-Asset Portfolio Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.tight_layout()
            return plt.gcf()
        return None

    def get_position_weights_at_timestamp(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        Get position weights at a specific timestamp.

        Args:
            timestamp: Timestamp to get weights for

        Returns:
            Dictionary of {symbol: weight} where weight is % of portfolio value
        """
        # Find the closest timestamp in our history
        for ts, weights in reversed(self.symbol_weights_history):
            if ts <= timestamp:
                return weights.copy()
        return {}

    def get_symbol_equity_curves(self, resample: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate per-symbol equity curves by reconstructing symbol-level performance.

        Args:
            resample: Optional resampling frequency (e.g., '1H' for hourly, 'D' for daily)
                     Use this to downsample for visualization performance

        Returns:
            Dictionary of {symbol: pd.Series} with equity curves per symbol
        """
        if not self.closed_positions:
            return {}

        # Group closed positions by symbol
        symbol_trades = {}
        for trade in self.closed_positions:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)

        # Calculate equity curve for each symbol
        symbol_equity = {}
        initial_capital_per_symbol = self.init_cash / len(self.symbols)

        # Use downsampled timestamps if requested (for performance)
        if resample and len(self.equity_timestamps) > 1000:
            # Create a downsampled version of the equity curve for visualization
            equity_series = pd.Series(self.equity_curve, index=self.equity_timestamps)
            downsampled = equity_series.resample(resample).last()
            target_timestamps = downsampled.index
        else:
            target_timestamps = self.equity_timestamps

        for symbol in self.symbols:
            trades = symbol_trades.get(symbol, [])
            if not trades:
                # No trades for this symbol - flat equity at initial capital
                symbol_equity[symbol] = pd.Series(
                    [initial_capital_per_symbol] * len(target_timestamps),
                    index=target_timestamps
                )
                continue

            # Build equity curve from trade P&L (only at trade timestamps)
            equity = [initial_capital_per_symbol]
            equity_times = [self.equity_timestamps[0]]

            for trade in sorted(trades, key=lambda t: t['entry_timestamp']):
                # Add equity at entry and exit
                entry_equity = equity[-1]
                exit_equity = entry_equity + trade['pnl']

                equity.append(entry_equity)
                equity_times.append(trade['entry_timestamp'])
                equity.append(exit_equity)
                equity_times.append(trade['exit_timestamp'])

            # Create sparse series and interpolate only to target timestamps
            trade_series = pd.Series(equity, index=equity_times)
            symbol_equity[symbol] = trade_series.reindex(
                target_timestamps,
                method='ffill'
            ).fillna(initial_capital_per_symbol)

        return symbol_equity

    def get_correlation_matrix(self, resample: Optional[str] = '1H') -> pd.DataFrame:
        """
        Calculate correlation matrix between symbol returns.

        Args:
            resample: Optional resampling frequency for performance (default: '1H')

        Returns:
            DataFrame with pairwise correlations between symbols
        """
        symbol_equity = self.get_symbol_equity_curves(resample=resample)
        if not symbol_equity:
            return pd.DataFrame()

        # Calculate returns for each symbol
        returns_data = {}
        for symbol, equity in symbol_equity.items():
            returns_data[symbol] = equity.pct_change().dropna()

        # Create returns DataFrame and calculate correlation
        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr()
