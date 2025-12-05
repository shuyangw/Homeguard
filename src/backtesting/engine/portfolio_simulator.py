"""
Custom portfolio simulator to replace vectorbt dependency.

This module provides the Portfolio class for backtesting trading strategies.
For performance-critical applications, it uses Numba JIT compilation when possible,
falling back to pure Python for advanced features.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import pytz
from datetime import time

from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting.utils.position_sizer import FixedPercentageSizer, FixedDollarSizer, VolatilityBasedSizer, KellyCriterionSizer
from src.backtesting.utils.risk_manager import RiskManager, Position

# Try to import Numba simulation module
try:
    from src.backtesting.engine.numba_sim import (
        simulate_portfolio_numba,
        TRADE_ENTRY, TRADE_EXIT, TRADE_SHORT_ENTRY, TRADE_COVER_SHORT,
        EXIT_SIGNAL, EXIT_STOP_LOSS, EXIT_TIME_STOP, EXIT_PROFIT_TARGET,
        TRADE_TYPE_NAMES, EXIT_REASON_NAMES
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class Portfolio:
    """
    Portfolio object that tracks trades and performance metrics.
    """

    def __init__(
        self,
        price: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        init_cash: float,
        fees: float,
        slippage: float,
        freq: str = '1min',
        market_hours_only: bool = True,
        risk_config: Optional[RiskConfig] = None,
        price_data: Optional[pd.DataFrame] = None,
        allow_shorts: bool = False,
        use_numba: bool = True
    ):
        """
        Initialize portfolio.

        Args:
            price: Price series for backtesting
            entries: Boolean series for entry signals
            exits: Boolean series for exit signals
            init_cash: Initial cash
            fees: Trading fees as decimal (e.g., 0.001 = 0.1%)
            slippage: Slippage as decimal
            freq: Data frequency
            market_hours_only: If True, only trade during market hours
            risk_config: Risk management configuration
            price_data: Full OHLCV data for indicators
            allow_shorts: If True, enable short selling (default: False)
            use_numba: If True, use Numba JIT compilation for performance (default: True)
        """
        self.price = price
        self.entries = entries.astype(bool)
        self.exits = exits.astype(bool)
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        self.market_hours_only = market_hours_only
        self.risk_config = risk_config or RiskConfig.moderate()
        self.price_data = price_data
        self.allow_shorts = allow_shorts
        self.use_numba = use_numba
        self.borrow_cost = 0.0030  # 30 bps/year for short positions

        # Define market hours in EST/EDT
        self.market_open = time(9, 35)  # 9:35 AM EST
        self.market_close = time(15, 55)  # 3:55 PM EST
        self.eastern_tz = pytz.timezone('US/Eastern')

        self.trades: List[Dict[str, Any]] = []
        self.equity_curve = None
        self._stats = None

        # Initialize position sizer based on risk config
        self._init_position_sizer()

        # Initialize risk manager if using stop losses or portfolio constraints
        self._init_risk_manager()

        # Choose simulation method based on feature requirements
        if self._can_use_numba():
            self._simulate_fast()
        else:
            self._simulate()

    def _is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if the given timestamp is within market trading hours.

        Args:
            timestamp: Timestamp to check

        Returns:
            True if within market hours (9:35 AM - 3:55 PM EST), False otherwise
        """
        if not self.market_hours_only:
            return True

        # Convert timestamp to Eastern time if it has timezone info
        if timestamp.tz is None:
            # Assume the timestamp is already in Eastern time if no timezone
            eastern_time = timestamp
        else:
            # Convert to Eastern timezone
            eastern_time = timestamp.tz_convert(self.eastern_tz)

        # Check if it's a weekday (Monday=0, Sunday=6)
        if eastern_time.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check if within trading hours
        current_time = eastern_time.time()
        return self.market_open <= current_time <= self.market_close

    def _init_position_sizer(self):
        """Initialize position sizer based on risk config."""
        method = self.risk_config.position_sizing_method

        if method == 'fixed_percentage':
            self.position_sizer = FixedPercentageSizer(
                position_pct=self.risk_config.position_size_pct
            )
        elif method == 'fixed_dollar':
            # Default to 10% of initial capital as fixed dollar amount
            fixed_amount = self.init_cash * self.risk_config.position_size_pct
            self.position_sizer = FixedPercentageSizer(
                position_pct=self.risk_config.position_size_pct
            )
        elif method == 'volatility':
            if self.price_data is not None:
                self.position_sizer = VolatilityBasedSizer(
                    risk_pct=self.risk_config.position_size_pct / 10,
                    atr_multiplier=self.risk_config.atr_multiplier,
                    atr_lookback=self.risk_config.atr_lookback
                )
            else:
                # Fallback to fixed percentage if no price data available
                self.position_sizer = FixedPercentageSizer(
                    position_pct=self.risk_config.position_size_pct
                )
        elif method == 'kelly':
            # Kelly requires strategy stats - use fixed % as fallback for now
            # TODO: Implement Kelly when we have trade history
            self.position_sizer = FixedPercentageSizer(
                position_pct=self.risk_config.position_size_pct
            )
        else:
            # Default to fixed percentage
            self.position_sizer = FixedPercentageSizer(
                position_pct=self.risk_config.position_size_pct
            )

    def _init_risk_manager(self):
        """Initialize risk manager if needed."""
        if self.risk_config.use_stop_loss or self.risk_config.max_positions is not None:
            self.risk_manager = RiskManager(self.risk_config)
        else:
            self.risk_manager = None

    def _can_use_numba(self) -> bool:
        """
        Check if Numba simulation can be used for current configuration.

        Numba is used when:
        - use_numba=True (explicitly enabled)
        - Numba module is available
        - Position sizing method is 'fixed_percentage'
        - Stop loss type is not 'atr' (ATR requires price history)

        Returns:
            True if Numba simulation can be used, False otherwise
        """
        if not self.use_numba:
            return False

        if not NUMBA_AVAILABLE:
            return False

        # Check position sizing method - only fixed_percentage supported
        if self.risk_config.position_sizing_method not in ['fixed_percentage', 'fixed_dollar']:
            return False

        # Check stop loss type - ATR stops require price history
        if self.risk_config.stop_loss_type == 'atr':
            return False

        return True

    def _compute_market_hours_mask(self) -> np.ndarray:
        """
        Pre-compute market hours for all timestamps (vectorized).

        This is much faster than calling _is_market_hours() per bar,
        especially for large datasets.

        Returns:
            Boolean numpy array where True = within market hours
        """
        if not self.market_hours_only:
            return np.ones(len(self.price), dtype=np.bool_)

        index = self.price.index

        # Convert to Eastern timezone
        if index.tz is None:
            # Assume already in Eastern time
            eastern = index
        else:
            eastern = index.tz_convert('US/Eastern')

        # Vectorized weekday check (Monday=0, Sunday=6)
        is_weekday = eastern.weekday < 5

        # Vectorized time check
        # Note: Using pandas Series for time comparison
        times = pd.Series(eastern).dt.time
        is_after_open = times >= self.market_open
        is_before_close = times <= self.market_close
        is_market_hours = is_after_open & is_before_close

        return (is_weekday & is_market_hours.values).astype(np.bool_)

    def _convert_numba_trades(
        self,
        trade_bars: np.ndarray,
        trade_types: np.ndarray,
        trade_prices: np.ndarray,
        trade_shares: np.ndarray,
        trade_pnls: np.ndarray,
        trade_pnl_pcts: np.ndarray,
        trade_exit_reasons: np.ndarray,
        trade_costs: np.ndarray,
        trade_proceeds: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Convert Numba trade arrays to list of dicts (matching Python format).

        Args:
            trade_bars: Bar indices where trades occurred
            trade_types: Trade type codes (entry, exit, short_entry, cover_short)
            trade_prices: Execution prices
            trade_shares: Shares traded
            trade_pnls: P&L for exits
            trade_pnl_pcts: P&L percentages for exits
            trade_exit_reasons: Exit reason codes
            trade_costs: Entry costs
            trade_proceeds: Exit proceeds

        Returns:
            List of trade dictionaries compatible with Python simulation format
        """
        trades = []
        timestamps = self.price.index

        for i in range(len(trade_bars)):
            bar_idx = trade_bars[i]
            trade_type = trade_types[i]
            timestamp = timestamps[bar_idx]

            trade = {
                'timestamp': timestamp,
                'type': TRADE_TYPE_NAMES.get(trade_type, 'unknown'),
                'price': trade_prices[i],
                'shares': trade_shares[i]
            }

            # Add type-specific fields
            if trade_type in [TRADE_ENTRY, TRADE_SHORT_ENTRY]:
                # Entry trade
                if trade_costs[i] > 0:
                    trade['cost'] = trade_costs[i]
                if trade_type == TRADE_SHORT_ENTRY and trade_proceeds[i] > 0:
                    trade['proceeds'] = trade_proceeds[i]
            else:
                # Exit trade
                trade['pnl'] = trade_pnls[i]
                trade['pnl_pct'] = trade_pnl_pcts[i]
                if trade_proceeds[i] > 0:
                    trade['proceeds'] = trade_proceeds[i]
                if trade_costs[i] > 0:
                    trade['cost'] = trade_costs[i]

                # Add exit reason
                exit_reason = trade_exit_reasons[i]
                if exit_reason >= 0:
                    trade['exit_reason'] = EXIT_REASON_NAMES.get(exit_reason, 'unknown')

            trades.append(trade)

        return trades

    def _simulate_fast(self):
        """
        Fast portfolio simulation using Numba JIT compilation.

        This method achieves 10-100x speedup over pure Python by:
        1. Pre-computing market hours mask (vectorized)
        2. Running the simulation loop in Numba-compiled code
        3. Converting results back to Python objects

        Supports:
        - Long and short positions
        - Slippage and fees
        - Fixed percentage position sizing
        - Percentage stop loss
        - Time-based stop
        - Profit target
        """
        # Convert inputs to numpy arrays
        prices = self.price.values.astype(np.float64)
        entries = self.entries.values.astype(np.bool_)
        exits = self.exits.values.astype(np.bool_)

        # Pre-compute market hours mask (vectorized - much faster)
        market_hours = self._compute_market_hours_mask()

        # Extract risk config parameters
        # Match Python's RiskManager behavior: stop_loss_type determines which mechanism is active
        stop_loss_type = self.risk_config.stop_loss_type

        # Percentage stop loss: active for 'percentage' type
        # Also active for 'profit_target' type (which includes both stop loss AND take profit)
        use_stop_loss = (
            self.risk_config.use_stop_loss and
            stop_loss_type in ['percentage', 'profit_target']
        )
        stop_loss_pct = self.risk_config.stop_loss_pct

        # Profit target: only active when stop_loss_type is 'profit_target'
        use_profit_target = (
            self.risk_config.use_stop_loss and
            stop_loss_type == 'profit_target' and
            self.risk_config.take_profit_pct is not None
        )
        profit_target_pct = self.risk_config.take_profit_pct or 0.0

        # Time-based stop: only active when stop_loss_type is 'time'
        use_time_stop = (
            self.risk_config.use_stop_loss and
            stop_loss_type == 'time' and
            self.risk_config.max_holding_bars is not None
        )
        max_bars_in_position = self.risk_config.max_holding_bars or 99999999

        # Run Numba simulation
        result = simulate_portfolio_numba(
            prices=prices,
            entries=entries,
            exits=exits,
            market_hours=market_hours,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            position_size_pct=self.risk_config.position_size_pct,
            use_stop_loss=use_stop_loss,
            stop_loss_pct=stop_loss_pct,
            use_profit_target=use_profit_target,
            profit_target_pct=profit_target_pct,
            use_time_stop=use_time_stop,
            max_bars_in_position=max_bars_in_position,
            allow_shorts=self.allow_shorts
        )

        # Unpack results
        (equity, trade_bars, trade_types, trade_prices, trade_shares,
         trade_pnls, trade_pnl_pcts, trade_exit_reasons,
         trade_costs, trade_proceeds, trade_count) = result

        # Convert equity to pandas Series
        self.equity_curve = pd.Series(equity, index=self.price.index)

        # Convert trades to list of dicts
        self.trades = self._convert_numba_trades(
            trade_bars, trade_types, trade_prices, trade_shares,
            trade_pnls, trade_pnl_pcts, trade_exit_reasons,
            trade_costs, trade_proceeds
        )

    def _simulate(self):
        """
        Simulate portfolio trades based on entry and exit signals.
        """
        cash = self.init_cash
        position = 0.0
        position_price = 0.0
        entry_timestamp = None
        entry_bar = 0
        equity = []
        trades = []
        bars_in_position = 0
        bar_index = 0

        for timestamp, price in self.price.items():
            bar_index += 1
            entry_signal = self.entries.get(timestamp, False)
            exit_signal = self.exits.get(timestamp, False)

            # Check if within market hours before executing any trade
            in_market_hours = self._is_market_hours(timestamp)

            # Calculate current portfolio value
            # For long: value = cash + (shares * price)
            # For short: value = cash + (shares * (2*entry_price - price))
            # Simplified: position can be positive (long) or negative (short)
            if position > 0:
                # Long position
                portfolio_value = cash + (position * price)
            elif position < 0:
                # Short position: value = cash + short_liability
                # Short liability = proceeds_from_short - current_buyback_cost
                # = abs(position) * entry_price - abs(position) * price
                # = abs(position) * (entry_price - price)
                short_liability = abs(position) * (position_price - price)
                portfolio_value = cash + short_liability
            else:
                # Flat
                portfolio_value = cash

            # Check for stop loss exits if we have an open position
            if position != 0 and self.risk_manager is not None:
                # Track bars in position for time-based stop
                bars_in_position += 1

                # Check if stop loss should trigger
                current_position = Position(
                    symbol='ASSET',
                    entry_price=position_price,
                    shares=int(position),
                    entry_bar=entry_bar
                )

                # Build price history for ATR-based stops (last 20 bars)
                idx = self.price.index.get_loc(timestamp)
                start_idx = max(0, idx - 20)
                recent_prices = self.price.iloc[start_idx:idx+1]

                # Prepare price data for check_exits
                if self.price_data is not None:
                    price_df = self.price_data.iloc[start_idx:idx+1]
                else:
                    price_df = pd.DataFrame({
                        'close': recent_prices,
                        'high': recent_prices,
                        'low': recent_prices
                    })

                # Check for exits
                exit_signals = self.risk_manager.check_exits(
                    current_prices={'ASSET': price},
                    current_bar=bar_index,
                    price_data_dict={'ASSET': price_df} if price_df is not None else None
                )

                should_exit = 'ASSET' in exit_signals
                exit_reason = exit_signals.get('ASSET', None) if should_exit else None

                if should_exit and in_market_hours:
                    # Execute stop loss exit
                    if position > 0:
                        # Close long position
                        slippage_adj = price * (1 - self.slippage)
                        proceeds = position * slippage_adj
                        fee = proceeds * self.fees
                        net_proceeds = proceeds - fee

                        cash += net_proceeds

                        pnl = net_proceeds - (position * position_price)
                        pnl_pct = (pnl / (position * position_price)) * 100 if position_price > 0 else 0

                        trades.append({
                            'timestamp': timestamp,
                            'type': 'exit',
                            'exit_reason': exit_reason,
                            'price': price,
                            'shares': position,
                            'proceeds': net_proceeds,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct
                        })
                    else:
                        # Close short position (buy to cover)
                        slippage_adj = price * (1 + self.slippage)
                        cost_to_cover = abs(position) * slippage_adj
                        fee = cost_to_cover * self.fees
                        total_cost = cost_to_cover + fee

                        proceeds_from_short = abs(position) * position_price
                        pnl = proceeds_from_short - cost_to_cover - fee
                        pnl_pct = (pnl / proceeds_from_short) * 100 if proceeds_from_short > 0 else 0

                        cash -= total_cost

                        trades.append({
                            'timestamp': timestamp,
                            'type': 'cover_short',
                            'exit_reason': exit_reason,
                            'price': price,
                            'shares': abs(position),
                            'cost': total_cost,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct
                        })

                    if self.risk_manager is not None:
                        self.risk_manager.remove_position('ASSET')

                    position = 0
                    position_price = 0
                    entry_timestamp = None
                    entry_bar = 0
                    bars_in_position = 0

            # Handle strategy signals
            # Entry signal logic:
            # - If flat (position==0): Open long
            # - If short (position<0): Close short and open long
            # Exit signal logic (if allow_shorts):
            # - If long (position>0): Close long and open short
            # - If flat (position==0): Open short
            # Exit signal logic (if not allow_shorts):
            # - If long (position>0): Close long (go flat)

            # Handle ENTRY signals (want to be LONG)
            if entry_signal and in_market_hours:
                # Close short position if we have one
                if position < 0:
                    # Buy to cover short
                    slippage_adj = price * (1 + self.slippage)  # Buy at higher price
                    cost_to_cover = abs(position) * slippage_adj
                    fee = cost_to_cover * self.fees
                    total_cost = cost_to_cover + fee

                    # P&L for short: profit when price drops
                    # Proceeds from short = abs(position) * position_price
                    # Cost to cover = abs(position) * price
                    # P&L = proceeds - cost - fees
                    proceeds_from_short = abs(position) * position_price
                    pnl = proceeds_from_short - cost_to_cover - fee
                    pnl_pct = (pnl / proceeds_from_short) * 100 if proceeds_from_short > 0 else 0

                    cash -= total_cost

                    trades.append({
                        'timestamp': timestamp,
                        'type': 'cover_short',
                        'price': price,
                        'shares': abs(position),
                        'cost': total_cost,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })

                    if self.risk_manager is not None:
                        self.risk_manager.remove_position('ASSET')

                    position = 0
                    position_price = 0
                    entry_timestamp = None
                    entry_bar = 0
                    bars_in_position = 0
                    # Recalculate portfolio value after closing position
                    portfolio_value = cash

                # Open long position if we have cash and no position
                if position == 0 and cash > 0:
                    # Check portfolio constraints
                    can_enter = True
                    if self.risk_manager is not None:
                        can_enter = self.risk_manager.can_open_position('ASSET', price, portfolio_value)

                    if can_enter:
                        # Calculate position size using position sizer
                        if self.risk_config.position_sizing_method == 'volatility' and self.price_data is not None:
                            # For volatility-based sizing, we need OHLC data
                            shares = self.position_sizer.calculate_shares(
                                portfolio_value=portfolio_value,
                                price=price,
                                price_data=self.price_data
                            )
                        else:
                            # For other methods, just portfolio value and price
                            shares = self.position_sizer.calculate_shares(
                                portfolio_value=portfolio_value,
                                price=price
                            )

                        # Apply slippage and fees
                        slippage_adj = price * (1 + self.slippage)
                        cost = shares * slippage_adj
                        fee = cost * self.fees
                        total_cost = cost + fee

                        # Ensure we have enough cash
                        if total_cost <= cash and shares > 0:
                            position = shares
                            position_price = price
                            entry_timestamp = timestamp
                            entry_bar = bar_index
                            cash -= total_cost
                            bars_in_position = 0

                            trades.append({
                                'timestamp': timestamp,
                                'type': 'entry',
                                'price': price,
                                'shares': shares,
                                'cost': total_cost
                            })

                            # Register position with risk manager
                            if self.risk_manager is not None:
                                # Build price data DataFrame for ATR stops (last 20 bars)
                                idx = self.price.index.get_loc(timestamp)
                                start_idx = max(0, idx - 20)
                                recent_price_series = self.price.iloc[start_idx:idx+1]

                                # If we have full OHLCV data, use it; otherwise create from close
                                if self.price_data is not None:
                                    price_df = self.price_data.iloc[start_idx:idx+1]
                                else:
                                    # Create minimal DataFrame from close prices
                                    price_df = pd.DataFrame({
                                        'close': recent_price_series,
                                        'high': recent_price_series,
                                        'low': recent_price_series
                                    })

                                self.risk_manager.add_position(
                                    symbol='ASSET',
                                    entry_price=position_price,
                                    shares=int(position),
                                    entry_bar=entry_bar,
                                    price_data=price_df
                                )

            # Handle EXIT signals (want to be SHORT or FLAT)
            elif exit_signal and in_market_hours:
                # Close long position if we have one
                if position > 0:
                    slippage_adj = price * (1 - self.slippage)
                    proceeds = position * slippage_adj
                    fee = proceeds * self.fees
                    net_proceeds = proceeds - fee

                    cash += net_proceeds

                    pnl = net_proceeds - (position * position_price)
                    pnl_pct = (pnl / (position * position_price)) * 100 if position_price > 0 else 0

                    trades.append({
                        'timestamp': timestamp,
                        'type': 'exit',
                        'exit_reason': 'strategy_signal',
                        'price': price,
                        'shares': position,
                        'proceeds': net_proceeds,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })

                    if self.risk_manager is not None:
                        self.risk_manager.remove_position('ASSET')

                    position = 0
                    position_price = 0
                    entry_timestamp = None
                    entry_bar = 0
                    bars_in_position = 0
                    # Recalculate portfolio value after closing position
                    portfolio_value = cash

                # Open short position if allow_shorts and we're flat
                if position == 0 and self.allow_shorts and cash > 0:
                    # Check portfolio constraints
                    can_enter = True
                    if self.risk_manager is not None:
                        can_enter = self.risk_manager.can_open_position('ASSET', price, portfolio_value)

                    if can_enter:
                        # Calculate position size using position sizer
                        if self.risk_config.position_sizing_method == 'volatility' and self.price_data is not None:
                            shares = self.position_sizer.calculate_shares(
                                portfolio_value=portfolio_value,
                                price=price,
                                price_data=self.price_data
                            )
                        else:
                            shares = self.position_sizer.calculate_shares(
                                portfolio_value=portfolio_value,
                                price=price
                            )

                        # Apply slippage and fees for short entry
                        # Short: Sell at lower price due to slippage
                        slippage_adj = price * (1 - self.slippage)
                        proceeds_from_short = shares * slippage_adj
                        fee = proceeds_from_short * self.fees
                        net_proceeds = proceeds_from_short - fee

                        # For short, we receive cash (proceeds)
                        # But we don't need to have the full amount upfront (margin)
                        # We'll keep it simple: proceeds added to cash
                        if shares > 0:
                            position = -shares  # Negative position for short
                            position_price = price  # Entry price for short
                            entry_timestamp = timestamp
                            entry_bar = bar_index
                            cash += net_proceeds  # Receive proceeds from short sale
                            bars_in_position = 0

                            trades.append({
                                'timestamp': timestamp,
                                'type': 'short_entry',
                                'price': price,
                                'shares': shares,
                                'proceeds': net_proceeds
                            })

                            # Register short position with risk manager
                            if self.risk_manager is not None:
                                idx = self.price.index.get_loc(timestamp)
                                start_idx = max(0, idx - 20)
                                recent_price_series = self.price.iloc[start_idx:idx+1]

                                if self.price_data is not None:
                                    price_df = self.price_data.iloc[start_idx:idx+1]
                                else:
                                    price_df = pd.DataFrame({
                                        'close': recent_price_series,
                                        'high': recent_price_series,
                                        'low': recent_price_series
                                    })

                                self.risk_manager.add_position(
                                    symbol='ASSET',
                                    entry_price=position_price,
                                    shares=-int(shares),  # Negative for short
                                    entry_bar=entry_bar,
                                    price_data=price_df
                                )

            # Recalculate portfolio value after trades
            if position > 0:
                # Long position
                portfolio_value = cash + (position * price)
            elif position < 0:
                # Short position
                short_liability = abs(position) * (position_price - price)
                portfolio_value = cash + short_liability
            else:
                # Flat
                portfolio_value = cash
            equity.append(portfolio_value)

        self.trades = trades
        self.equity_curve = pd.Series(equity, index=self.price.index)

    def stats(self) -> Optional[Dict[str, Any]]:
        """
        Calculate portfolio statistics.
        """
        if self._stats is not None:
            return self._stats

        if self.equity_curve is None or len(self.equity_curve) == 0:
            return None

        total_return_pct = ((self.equity_curve.iloc[-1] - self.init_cash) / self.init_cash) * 100

        # Use fill_method=None to avoid FutureWarning (we drop NaN anyway)
        returns = self.equity_curve.pct_change(fill_method=None).dropna()

        # Calculate annualized return using CAGR (Compound Annual Growth Rate)
        # This is more accurate than simple arithmetic annualization,
        # especially for high-frequency data (e.g., 1-minute bars)
        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        total_days = (end_date - start_date).days
        years = max(total_days / 365.25, 1/252)  # At least 1 trading day

        if self.equity_curve.iloc[-1] > 0 and self.init_cash > 0 and years > 0:
            # CAGR = (End Value / Start Value)^(1/years) - 1
            cagr = ((self.equity_curve.iloc[-1] / self.init_cash) ** (1/years) - 1) * 100
            annual_return_pct = cagr
        else:
            annual_return_pct = total_return_pct

        if len(returns) == 0:
            sharpe_ratio = 0
        else:
            mean_return = returns.mean()
            std_return = returns.std()

            periods_per_year = self._get_periods_per_year()

            if std_return > 0:
                sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year)
            else:
                sharpe_ratio = 0

        cummax = self.equity_curve.cummax()
        # Prevent division by zero (theoretical edge case if init_cash=0)
        cummax = cummax.replace(0, 1)
        drawdown = (self.equity_curve - cummax) / cummax * 100
        max_drawdown_pct = drawdown.min()

        # Count both long exits and short covers as completed trades
        exit_trades = [t for t in self.trades if t.get('type') in ['exit', 'cover_short']]
        winning_trades = [t for t in exit_trades if t.get('pnl', 0) > 0]
        total_trades = len(exit_trades)
        win_rate_pct = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        # Count short positions if shorts are enabled
        short_trades = [t for t in self.trades if t.get('type') in ['short_entry', 'cover_short']]
        num_short_trades = len([t for t in short_trades if t.get('type') == 'short_entry'])

        self._stats = {
            'Total Return [%]': total_return_pct,
            'Annual Return [%]': annual_return_pct,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown [%]': max_drawdown_pct,
            'Win Rate [%]': win_rate_pct,
            'Total Trades': total_trades,
            'Start Value': self.init_cash,
            'End Value': self.equity_curve.iloc[-1],
            'Short Selling': 'Enabled' if self.allow_shorts else 'Disabled',
            'Short Trades': num_short_trades if self.allow_shorts else 0
        }

        return self._stats

    def _get_periods_per_year(self) -> int:
        """
        Get number of periods per year based on frequency.
        """
        freq_map = {
            '1min': 252 * 6.5 * 60,
            '1h': 252 * 6.5,
            '1d': 252,
            'D': 252,
            'daily': 252
        }
        return freq_map.get(self.freq, 252)

    def returns(self, freq: Optional[str] = None) -> pd.Series:
        """
        Get returns series (compatible with QuantStats and MultiAssetPortfolio).

        Args:
            freq: Frequency for resampling (e.g., 'D' for daily, 'H' for hourly)
                  Use this to downsample for performance with large datasets

        Returns:
            pd.Series with returns indexed by timestamp
        """
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return pd.Series(dtype=float)

        # PERFORMANCE FIX: Support downsampling for large datasets
        if freq and len(self.equity_curve) > 1000:
            # Resample equity first, then calculate returns
            resampled_equity = self.equity_curve.resample(freq).last()
            return resampled_equity.pct_change().fillna(0)
        else:
            return self.equity_curve.pct_change().fillna(0)

    def plot(self, **kwargs):
        """
        Plot equity curve (placeholder for compatibility).
        """
        if self.equity_curve is not None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(self.equity_curve.index, self.equity_curve.values)
            plt.title('Portfolio Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.tight_layout()
            return plt.gcf()
        return None


def from_signals(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float,
    fees: float,
    slippage: float = 0.0,
    freq: str = '1min',
    market_hours_only: bool = True,
    risk_config: Optional[RiskConfig] = None,
    price_data: Optional[pd.DataFrame] = None,
    allow_shorts: bool = False,
    use_numba: bool = True,
    **kwargs
) -> Portfolio:
    """
    Create a portfolio from entry/exit signals.

    Args:
        close: Price series
        entries: Entry signals
        exits: Exit signals
        init_cash: Initial capital
        fees: Trading fees
        slippage: Slippage
        freq: Data frequency
        market_hours_only: If True, only execute trades during market hours (9:35 AM - 3:55 PM EST)
        risk_config: Risk management configuration (defaults to RiskConfig.moderate())
        price_data: Historical OHLC data for ATR-based position sizing (optional)
        allow_shorts: If True, enable short selling (default: False)
        use_numba: If True, use Numba JIT compilation for performance (default: True)
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Portfolio object
    """
    return Portfolio(
        price=close,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
        market_hours_only=market_hours_only,
        risk_config=risk_config,
        price_data=price_data,
        allow_shorts=allow_shorts,
        use_numba=use_numba
    )
