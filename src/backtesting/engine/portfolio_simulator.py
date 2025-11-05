"""
Custom portfolio simulator to replace vectorbt dependency.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import pytz
from datetime import time

from backtesting.utils.risk_config import RiskConfig
from backtesting.utils.position_sizer import FixedPercentageSizer, FixedDollarSizer, VolatilityBasedSizer, KellyCriterionSizer
from backtesting.utils.risk_manager import RiskManager, Position


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
        price_data: Optional[pd.DataFrame] = None
    ):
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

        # Define market hours in EST/EDT
        self.market_open = time(9, 35)  # 9:35 AM EST
        self.market_close = time(15, 55)  # 3:55 PM EST
        self.eastern_tz = pytz.timezone('US/Eastern')

        self.trades = []
        self.equity_curve = None
        self._stats = None

        # Initialize position sizer based on risk config
        self._init_position_sizer()

        # Initialize risk manager if using stop losses or portfolio constraints
        self._init_risk_manager()

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
            portfolio_value = cash + (position * price if position > 0 else 0)

            # Check for stop loss exits if we have an open position
            if position > 0 and self.risk_manager is not None:
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

                    if self.risk_manager is not None:
                        self.risk_manager.remove_position('ASSET')

                    position = 0
                    position_price = 0
                    entry_timestamp = None
                    entry_bar = 0
                    bars_in_position = 0

            # Handle strategy entry signals
            if entry_signal and position == 0 and cash > 0 and in_market_hours:
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

            # Handle strategy exit signals
            elif exit_signal and position > 0 and in_market_hours:
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

            # Recalculate portfolio value after trades
            portfolio_value = cash + (position * price if position > 0 else 0)
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

        if len(returns) == 0:
            annual_return_pct = 0
            sharpe_ratio = 0
        else:
            mean_return = returns.mean()
            std_return = returns.std()

            periods_per_year = self._get_periods_per_year()
            annual_return_pct = mean_return * periods_per_year * 100

            if std_return > 0:
                sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year)
            else:
                sharpe_ratio = 0

        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax * 100
        max_drawdown_pct = drawdown.min()

        winning_trades = [t for t in self.trades if t.get('type') == 'exit' and t.get('pnl', 0) > 0]
        total_trades = len([t for t in self.trades if t.get('type') == 'exit'])
        win_rate_pct = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        self._stats = {
            'Total Return [%]': total_return_pct,
            'Annual Return [%]': annual_return_pct,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown [%]': max_drawdown_pct,
            'Win Rate [%]': win_rate_pct,
            'Total Trades': total_trades,
            'Start Value': self.init_cash,
            'End Value': self.equity_curve.iloc[-1]
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
        price_data=price_data
    )
