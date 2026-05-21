"""
Custom portfolio simulator to replace vectorbt dependency.

This module provides the Portfolio class for backtesting trading strategies.
For performance-critical applications, it uses Numba JIT compilation when possible,
falling back to pure Python for advanced features.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

from src.backtesting.engine.base_portfolio import BasePortfolio
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

# V2 is imported lazily in from_signals() to avoid circular imports
# (V2 subclasses V1's Portfolio class)


class Portfolio(BasePortfolio):
    """
    Portfolio object that tracks trades and performance metrics.

    Inherits common functionality from BasePortfolio:
    - Market hours checking (_is_market_hours)
    - Period annualization (_get_periods_per_year)
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
        use_numba: bool = True,
        fractional_shares: bool = False,
        stop_slippage_multiplier: float = 1.0,
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
            fractional_shares: If True, allow fractional share quantities (for crypto)
            stop_slippage_multiplier: Multiplier applied to slippage on stop-loss
                exits per methodology Section 11.5. Default 1.0 (no multiplier).
        """
        # Initialize base class (handles init_cash, fees, slippage, freq, market_hours_only,
        # market hours constants, and equity_curve/_stats)
        super().__init__(
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            freq=freq,
            market_hours_only=market_hours_only
        )

        # Portfolio-specific attributes
        self.price = price
        self.entries = entries.astype(bool)
        self.exits = exits.astype(bool)
        self.risk_config = risk_config or RiskConfig.moderate()
        self.price_data = price_data
        self.allow_shorts = allow_shorts
        self.use_numba = use_numba
        self.fractional_shares = fractional_shares
        self.stop_slippage_multiplier = stop_slippage_multiplier
        self.borrow_cost = 0.0030  # 30 bps/year for short positions

        self.trades: List[Dict[str, Any]] = []

        # Initialize position sizer based on risk config
        self._init_position_sizer()

        # Initialize risk manager if using stop losses or portfolio constraints
        self._init_risk_manager()

        # Choose simulation method based on feature requirements
        if self._can_use_numba():
            self._simulate_fast()
        else:
            self._simulate()

    # NOTE: _is_market_hours is inherited from BasePortfolio

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
        trade_proceeds: np.ndarray,
        trade_mae_pcts: Optional[np.ndarray] = None,
        trade_mfe_pcts: Optional[np.ndarray] = None,
        trade_mae_bars: Optional[np.ndarray] = None,
        trade_mfe_bars: Optional[np.ndarray] = None,
        stop_loss_pct_for_hit: Optional[float] = None,
        profit_target_pct_for_hit: Optional[float] = None,
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
            trade_mae_pcts: Per-trade signed MAE (long-convention). Exit rows only.
            trade_mfe_pcts: Per-trade signed MFE (long-convention). Exit rows only.
            trade_mae_bars: Bar index where MAE occurred (-1 for entries).
            trade_mfe_bars: Bar index where MFE occurred (-1 for entries).
            stop_loss_pct_for_hit: Configured stop threshold used to derive
                `hit_stop` on each exit. None means hit_stop=False always.
            profit_target_pct_for_hit: Configured target threshold used to derive
                `hit_target` on each exit. None means hit_target=False always.

        Returns:
            List of trade dictionaries compatible with Python simulation format
        """
        trades = []
        timestamps = self.price.index
        have_mae_mfe = trade_mae_pcts is not None and trade_mfe_pcts is not None

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

                # Section 11.6 MAE/MFE diagnostics.
                if have_mae_mfe:
                    mae_pct = float(trade_mae_pcts[i])
                    mfe_pct = float(trade_mfe_pcts[i])
                    trade['mae_pct'] = mae_pct
                    trade['mfe_pct'] = mfe_pct
                    mae_bar = int(trade_mae_bars[i]) if trade_mae_bars is not None else -1
                    mfe_bar = int(trade_mfe_bars[i]) if trade_mfe_bars is not None else -1
                    trade['mae_time'] = timestamps[mae_bar] if mae_bar >= 0 else None
                    trade['mfe_time'] = timestamps[mfe_bar] if mfe_bar >= 0 else None
                    # hit_stop / hit_target compare excursion against configured
                    # thresholds. If thresholds are not configured the flags are
                    # vacuously False (nothing to compare against).
                    trade['hit_stop'] = (
                        stop_loss_pct_for_hit is not None
                        and mae_pct <= -float(stop_loss_pct_for_hit)
                    )
                    trade['hit_target'] = (
                        profit_target_pct_for_hit is not None
                        and mfe_pct >= float(profit_target_pct_for_hit)
                    )

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
            allow_shorts=self.allow_shorts,
            fractional_shares=self.fractional_shares,
            stop_slippage_multiplier=self.stop_slippage_multiplier,
        )

        # Unpack results
        (equity, trade_bars, trade_types, trade_prices, trade_shares,
         trade_pnls, trade_pnl_pcts, trade_exit_reasons,
         trade_costs, trade_proceeds,
         trade_mae_pcts, trade_mfe_pcts, trade_mae_bars, trade_mfe_bars,
         trade_count) = result

        # Convert equity to pandas Series
        self.equity_curve = pd.Series(equity, index=self.price.index)

        # Convert trades to list of dicts (per methodology Section 11.6,
        # exit rows are annotated with MAE/MFE and hit_stop/hit_target flags
        # derived from the configured stop/target thresholds).
        self.trades = self._convert_numba_trades(
            trade_bars, trade_types, trade_prices, trade_shares,
            trade_pnls, trade_pnl_pcts, trade_exit_reasons,
            trade_costs, trade_proceeds,
            trade_mae_pcts=trade_mae_pcts,
            trade_mfe_pcts=trade_mfe_pcts,
            trade_mae_bars=trade_mae_bars,
            trade_mfe_bars=trade_mfe_bars,
            stop_loss_pct_for_hit=stop_loss_pct if use_stop_loss else None,
            profit_target_pct_for_hit=profit_target_pct if use_profit_target else None,
        )

    def _on_bar_start(self, i: int, price: float, cash: float,
                     position: float, position_price: float):
        """
        Hook called at start of each bar in _simulate().

        Subclasses can override to record per-bar state. No-op in base class.

        Args:
            i: Bar index (0-based)
            price: Current bar price
            cash: Current cash balance
            position: Current position size (negative for short)
            position_price: Entry price of current position (0 if flat)
        """
        pass

    def _on_trade(self, trade_record: dict):
        """
        Hook called after each trade execution in _simulate().

        Subclasses can override to process trades. No-op in base class.

        Args:
            trade_record: Dictionary with trade details (timestamp, type, price, etc.)
        """
        pass

    def _on_simulation_end(self):
        """
        Hook called after simulation completes in _simulate().

        Subclasses can override for post-simulation processing. No-op in base class.
        """
        pass

    @staticmethod
    def _calc_portfolio_value(cash, position, position_price, price):
        """Calculate current portfolio value given position state.

        Args:
            cash: Current cash balance
            position: Current position size (positive=long, negative=short)
            position_price: Entry price of current position
            price: Current market price

        Returns:
            Portfolio value as float
        """
        if position > 0:
            return cash + (position * price)
        elif position < 0:
            short_liability = abs(position) * (position_price - price)
            return cash + short_liability
        return cash

    def _get_recent_price_data(self, timestamp):
        """Get recent price data (last 20 bars) for ATR-based stops.

        Args:
            timestamp: Current bar timestamp

        Returns:
            DataFrame with close/high/low columns for recent bars
        """
        idx = self.price.index.get_loc(timestamp)
        start_idx = max(0, idx - 20)
        recent_prices = self.price.iloc[start_idx:idx+1]

        if self.price_data is not None:
            return self.price_data.iloc[start_idx:idx+1]
        return pd.DataFrame({
            'close': recent_prices,
            'high': recent_prices,
            'low': recent_prices
        })

    def _calculate_position_shares(self, portfolio_value, price):
        """Calculate number of shares for a new position.

        Uses the configured position sizer. For volatility-based sizing,
        includes full OHLC price data.

        Args:
            portfolio_value: Current portfolio value
            price: Current market price

        Returns:
            Number of shares to trade
        """
        if self.risk_config.position_sizing_method == 'volatility' and self.price_data is not None:
            return self.position_sizer.calculate_shares(
                portfolio_value=portfolio_value,
                price=price,
                price_data=self.price_data
            )
        return self.position_sizer.calculate_shares(
            portfolio_value=portfolio_value,
            price=price
        )

    def _execute_long_exit(self, timestamp, price, cash, position, position_price,
                           exit_reason, trades):
        """Execute a long position exit (sell shares).

        Args:
            timestamp: Current bar timestamp
            price: Current market price
            cash: Current cash balance
            position: Current position size (must be > 0)
            position_price: Entry price of long position
            exit_reason: Reason for exit (e.g. 'strategy_signal', stop loss reason)
            trades: Trade list to append to

        Returns:
            Tuple of (new_cash, trade_record)
        """
        slippage_adj = price * (1 - self.slippage)
        proceeds = position * slippage_adj
        fee = proceeds * self.fees
        net_proceeds = proceeds - fee

        new_cash = cash + net_proceeds

        pnl = net_proceeds - (position * position_price)
        pnl_pct = (pnl / (position * position_price)) * 100 if position_price > 0 else 0

        trade_record = {
            'timestamp': timestamp,
            'type': 'exit',
            'exit_reason': exit_reason,
            'price': price,
            'shares': position,
            'proceeds': net_proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
        trades.append(trade_record)
        self._on_trade(trade_record)

        return new_cash, trade_record

    def _execute_short_cover(self, timestamp, price, cash, position, position_price,
                             exit_reason, trades):
        """Execute a short cover (buy to cover).

        Args:
            timestamp: Current bar timestamp
            price: Current market price
            cash: Current cash balance
            position: Current position size (must be < 0)
            position_price: Entry price of short position
            exit_reason: Reason for cover (optional, None for signal-based)
            trades: Trade list to append to

        Returns:
            Tuple of (new_cash, trade_record)
        """
        slippage_adj = price * (1 + self.slippage)
        cost_to_cover = abs(position) * slippage_adj
        fee = cost_to_cover * self.fees
        total_cost = cost_to_cover + fee

        proceeds_from_short = abs(position) * position_price
        pnl = proceeds_from_short - cost_to_cover - fee
        pnl_pct = (pnl / proceeds_from_short) * 100 if proceeds_from_short > 0 else 0

        new_cash = cash - total_cost

        trade_record = {
            'timestamp': timestamp,
            'type': 'cover_short',
            'price': price,
            'shares': abs(position),
            'cost': total_cost,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
        if exit_reason is not None:
            trade_record['exit_reason'] = exit_reason
        trades.append(trade_record)
        self._on_trade(trade_record)

        return new_cash, trade_record

    def _execute_long_entry(self, timestamp, price, cash, portfolio_value, bar_index, trades):
        """Execute a long entry (buy shares).

        Checks portfolio constraints via risk manager, calculates position
        size, applies slippage/fees, and registers position if risk manager
        is active.

        Args:
            timestamp: Current bar timestamp
            price: Current market price
            cash: Current cash balance
            portfolio_value: Current portfolio value
            bar_index: Current bar index (1-based)
            trades: Trade list to append to

        Returns:
            Tuple of (new_cash, new_position, new_position_price, entry_bar)
            or (cash, 0, 0, 0) if entry was not executed
        """
        can_enter = True
        if self.risk_manager is not None:
            can_enter = self.risk_manager.can_open_position('ASSET', price, portfolio_value)

        if not can_enter:
            return cash, 0, 0, 0

        shares = self._calculate_position_shares(portfolio_value, price)

        slippage_adj = price * (1 + self.slippage)
        cost = shares * slippage_adj
        fee = cost * self.fees
        total_cost = cost + fee

        if total_cost > cash or shares <= 0:
            return cash, 0, 0, 0

        new_cash = cash - total_cost

        trade_record = {
            'timestamp': timestamp,
            'type': 'entry',
            'price': price,
            'shares': shares,
            'cost': total_cost
        }
        trades.append(trade_record)
        self._on_trade(trade_record)

        if self.risk_manager is not None:
            price_df = self._get_recent_price_data(timestamp)
            self.risk_manager.add_position(
                symbol='ASSET',
                entry_price=price,
                shares=int(shares),
                entry_bar=bar_index,
                price_data=price_df
            )

        return new_cash, shares, price, bar_index

    def _execute_short_entry(self, timestamp, price, cash, portfolio_value, bar_index, trades):
        """Execute a short entry (sell short).

        Checks portfolio constraints via risk manager, calculates position
        size, applies slippage/fees, and registers position if risk manager
        is active.

        Args:
            timestamp: Current bar timestamp
            price: Current market price
            cash: Current cash balance
            portfolio_value: Current portfolio value
            bar_index: Current bar index (1-based)
            trades: Trade list to append to

        Returns:
            Tuple of (new_cash, new_position, new_position_price, entry_bar)
            or (cash, 0, 0, 0) if entry was not executed
        """
        can_enter = True
        if self.risk_manager is not None:
            can_enter = self.risk_manager.can_open_position('ASSET', price, portfolio_value)

        if not can_enter:
            return cash, 0, 0, 0

        shares = self._calculate_position_shares(portfolio_value, price)

        slippage_adj = price * (1 - self.slippage)
        proceeds_from_short = shares * slippage_adj
        fee = proceeds_from_short * self.fees
        net_proceeds = proceeds_from_short - fee

        if shares <= 0:
            return cash, 0, 0, 0

        new_cash = cash + net_proceeds

        trade_record = {
            'timestamp': timestamp,
            'type': 'short_entry',
            'price': price,
            'shares': shares,
            'proceeds': net_proceeds
        }
        trades.append(trade_record)
        self._on_trade(trade_record)

        if self.risk_manager is not None:
            price_df = self._get_recent_price_data(timestamp)
            self.risk_manager.add_position(
                symbol='ASSET',
                entry_price=price,
                shares=-int(shares),
                entry_bar=bar_index,
                price_data=price_df
            )

        return new_cash, -shares, price, bar_index

    def _check_stop_loss(self, timestamp, price, cash, position, position_price,
                         entry_bar, bar_index, in_market_hours, trades):
        """Check and execute stop-loss exits for open positions.

        Evaluates whether the risk manager triggers a stop-loss exit
        (percentage, ATR, time-based, or profit target) and executes
        the exit if conditions are met.

        Args:
            timestamp: Current bar timestamp
            price: Current market price
            cash: Current cash balance
            position: Current position size (positive=long, negative=short)
            position_price: Entry price of current position
            entry_bar: Bar index when position was entered
            bar_index: Current bar index (1-based)
            in_market_hours: Whether current bar is within market hours
            trades: Trade list to append to

        Returns:
            Tuple of (new_cash, new_position, new_position_price, was_stopped)
            If no stop triggered, returns original values with was_stopped=False.
        """
        price_df = self._get_recent_price_data(timestamp)

        exit_signals = self.risk_manager.check_exits(
            current_prices={'ASSET': price},
            current_bar=bar_index,
            price_data_dict={'ASSET': price_df} if price_df is not None else None
        )

        should_exit = 'ASSET' in exit_signals
        exit_reason = exit_signals.get('ASSET', None) if should_exit else None

        if not (should_exit and in_market_hours):
            return cash, position, position_price, False

        if position > 0:
            cash, _ = self._execute_long_exit(
                timestamp, price, cash, position, position_price,
                exit_reason, trades
            )
        else:
            cash, _ = self._execute_short_cover(
                timestamp, price, cash, position, position_price,
                exit_reason, trades
            )

        if self.risk_manager is not None:
            self.risk_manager.remove_position('ASSET')

        return cash, 0, 0, True

    def _simulate(self):
        """
        Simulate portfolio trades based on entry and exit signals.

        This is the main simulation loop that orchestrates position management
        by delegating to focused helper methods for each trade type.
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

        for i, (timestamp, price) in enumerate(self.price.items()):
            bar_index += 1
            entry_signal = self.entries.get(timestamp, False)
            exit_signal = self.exits.get(timestamp, False)

            in_market_hours = self._is_market_hours(timestamp)

            portfolio_value = self._calc_portfolio_value(
                cash, position, position_price, price
            )

            # Hook: record state at start of bar
            self._on_bar_start(i, price, cash, position, position_price)

            # Check for stop loss exits if we have an open position
            if position != 0 and self.risk_manager is not None:
                bars_in_position += 1

                cash, position, position_price, was_stopped = self._check_stop_loss(
                    timestamp, price, cash, position, position_price,
                    entry_bar, bar_index, in_market_hours, trades
                )
                if was_stopped:
                    entry_timestamp = None
                    entry_bar = 0
                    bars_in_position = 0

            # Handle strategy signals
            # Entry signal -> want to be LONG
            # Exit signal -> want to be SHORT (if allow_shorts) or FLAT

            # Handle ENTRY signals (want to be LONG)
            if entry_signal and in_market_hours:
                # Close short position if we have one
                if position < 0:
                    cash, _ = self._execute_short_cover(
                        timestamp, price, cash, position, position_price,
                        None, trades
                    )

                    if self.risk_manager is not None:
                        self.risk_manager.remove_position('ASSET')

                    position = 0
                    position_price = 0
                    entry_timestamp = None
                    entry_bar = 0
                    bars_in_position = 0
                    portfolio_value = cash

                # Open long position if we have cash and no position
                if position == 0 and cash > 0:
                    result = self._execute_long_entry(
                        timestamp, price, cash, portfolio_value, bar_index, trades
                    )
                    new_cash, new_pos, new_price, new_bar = result
                    if new_pos != 0:
                        cash = new_cash
                        position = new_pos
                        position_price = new_price
                        entry_timestamp = timestamp
                        entry_bar = new_bar
                        bars_in_position = 0

            # Handle EXIT signals (want to be SHORT or FLAT)
            elif exit_signal and in_market_hours:
                # Close long position if we have one
                if position > 0:
                    cash, _ = self._execute_long_exit(
                        timestamp, price, cash, position, position_price,
                        'strategy_signal', trades
                    )

                    if self.risk_manager is not None:
                        self.risk_manager.remove_position('ASSET')

                    position = 0
                    position_price = 0
                    entry_timestamp = None
                    entry_bar = 0
                    bars_in_position = 0
                    portfolio_value = cash

                # Open short position if allow_shorts and we're flat
                if position == 0 and self.allow_shorts and cash > 0:
                    result = self._execute_short_entry(
                        timestamp, price, cash, portfolio_value, bar_index, trades
                    )
                    new_cash, new_pos, new_price, new_bar = result
                    if new_pos != 0:
                        cash = new_cash
                        position = new_pos
                        position_price = new_price
                        entry_timestamp = timestamp
                        entry_bar = new_bar
                        bars_in_position = 0

            # Recalculate portfolio value after trades
            portfolio_value = self._calc_portfolio_value(
                cash, position, position_price, price
            )
            equity.append(portfolio_value)

        self.trades = trades
        self.equity_curve = pd.Series(equity, index=self.price.index)
        self._on_simulation_end()

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

    # NOTE: _get_periods_per_year is inherited from BasePortfolio

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
    fractional_shares: bool = False,
    track_state: bool = False,
    stop_slippage_multiplier: float = 1.0,
    **kwargs
):
    """
    Create a portfolio from entry/exit signals.

    This function delegates to the V2 portfolio simulator which provides:
    - Same performance as V1 (uses same Numba simulation)
    - Optional per-bar state tracking via track_state parameter
    - Complete parity with V1 (validated with $0.00 difference)

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
        fractional_shares: If True, allow fractional share quantities (for crypto)
        track_state: If True, enable per-bar portfolio state tracking (default: False)
        **kwargs: Additional arguments (for compatibility)

    Returns:
        PortfolioV2 object (compatible with Portfolio interface)
    """
    # Lazy import V2 to avoid circular imports (V2 subclasses V1's Portfolio)
    try:
        from src.backtesting_v2.engine.portfolio_simulator import (
            from_signals_v2 as _from_signals_v2,
        )
        return _from_signals_v2(
            close=close,
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
            use_numba=use_numba,
            fractional_shares=fractional_shares,
            track_state=track_state,
            stop_slippage_multiplier=stop_slippage_multiplier,
            **kwargs
        )
    except ImportError:
        pass

    # Fallback to V1 implementation
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
        use_numba=use_numba,
        fractional_shares=fractional_shares,
        stop_slippage_multiplier=stop_slippage_multiplier
    )
