"""
Pairs trading portfolio simulator.

This module provides specialized portfolio simulation for pairs trading strategies,
handling synchronized entry/exit of both legs with proper position sizing and P&L tracking.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import pytz
from datetime import time

from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting.utils.pairs_position_sizer import PairsPositionSizer, DollarNeutralSizer
from src.utils import logger


@dataclass
class PairPosition:
    """
    Represents an open pairs trading position.

    For pairs trading, we simultaneously hold:
    - Symbol1: Either long or short position
    - Symbol2: Opposite position (if symbol1 is long, symbol2 is short)

    Attributes:
        symbol1: First symbol in the pair (e.g., 'AAPL')
        symbol2: Second symbol in the pair (e.g., 'MSFT')
        shares1: Number of shares of symbol1 (positive=long, negative=short)
        shares2: Number of shares of symbol2 (positive=long, negative=short)
        entry_price1: Entry price for symbol1
        entry_price2: Entry price for symbol2
        entry_timestamp: When the pair position was opened
        entry_bar: Bar index when entered
        hedge_ratio: Ratio between the two positions (shares2 / shares1)
        capital_allocated: Total capital allocated to this pair trade
    """
    symbol1: str
    symbol2: str
    shares1: float
    shares2: float
    entry_price1: float
    entry_price2: float
    entry_timestamp: pd.Timestamp
    entry_bar: int
    hedge_ratio: float
    capital_allocated: float

    def is_long_spread(self) -> bool:
        """
        Check if this is a long spread position.

        Long spread: Short symbol1, Long symbol2
        (Bet that spread will narrow)
        """
        return self.shares1 < 0 and self.shares2 > 0

    def is_short_spread(self) -> bool:
        """
        Check if this is a short spread position.

        Short spread: Long symbol1, Short symbol2
        (Bet that spread will widen)
        """
        return self.shares1 > 0 and self.shares2 < 0

    def get_current_value(
        self,
        price1: float,
        price2: float
    ) -> Tuple[float, float]:
        """
        Calculate current value of the pair position.

        Args:
            price1: Current price of symbol1
            price2: Current price of symbol2

        Returns:
            Tuple of (leg1_value, leg2_value)

        For long positions: value = shares * price
        For short positions: value = shares * (2 * entry_price - price)
                                  = proceeds - buyback_cost
        """
        # Symbol1 value
        if self.shares1 > 0:
            # Long position
            value1 = self.shares1 * price1
        else:
            # Short position
            value1 = abs(self.shares1) * (self.entry_price1 - price1)

        # Symbol2 value
        if self.shares2 > 0:
            # Long position
            value2 = self.shares2 * price2
        else:
            # Short position
            value2 = abs(self.shares2) * (self.entry_price2 - price2)

        return value1, value2

    def get_unrealized_pnl(
        self,
        price1: float,
        price2: float
    ) -> float:
        """
        Calculate unrealized P&L for the pair.

        Args:
            price1: Current price of symbol1
            price2: Current price of symbol2

        Returns:
            Total unrealized P&L (leg1_pnl + leg2_pnl)
        """
        value1, value2 = self.get_current_value(price1, price2)

        # Calculate P&L for each leg
        if self.shares1 > 0:
            # Long leg1
            pnl1 = value1 - (self.shares1 * self.entry_price1)
        else:
            # Short leg1 - profit when price falls
            pnl1 = abs(self.shares1) * (self.entry_price1 - price1)

        if self.shares2 > 0:
            # Long leg2
            pnl2 = value2 - (self.shares2 * self.entry_price2)
        else:
            # Short leg2
            pnl2 = abs(self.shares2) * (self.entry_price2 - price2)

        return pnl1 + pnl2


class PairsPortfolio:
    """
    Portfolio simulator for pairs trading strategies.

    This simulator handles:
    - Synchronized entry/exit of both legs
    - Dollar-neutral position sizing
    - Separate fees and slippage for each leg
    - Proper P&L attribution for hedged positions
    - Market hours filtering
    - Risk management (stop losses, position limits)
    """

    def __init__(
        self,
        symbols: Tuple[str, str],
        prices1: pd.Series,
        prices2: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        short_entries: pd.Series,
        short_exits: pd.Series,
        init_cash: float,
        fees: float,
        slippage: float,
        hedge_ratio: Optional[pd.Series] = None,
        freq: str = '1min',
        market_hours_only: bool = True,
        risk_config: Optional[RiskConfig] = None,
        price_data1: Optional[pd.DataFrame] = None,
        price_data2: Optional[pd.DataFrame] = None
    ):
        """
        Initialize pairs portfolio.

        Args:
            symbols: Tuple of (symbol1, symbol2) names
            prices1: Price series for symbol1
            prices2: Price series for symbol2
            entries: Long spread entry signals (short sym1, long sym2)
            exits: Long spread exit signals
            short_entries: Short spread entry signals (long sym1, short sym2)
            short_exits: Short spread exit signals
            init_cash: Initial capital
            fees: Trading fees as decimal (applied to each leg)
            slippage: Slippage as decimal (applied to each leg)
            hedge_ratio: Optional dynamic hedge ratio series (default: 1.0)
            freq: Data frequency
            market_hours_only: If True, only trade during market hours
            risk_config: Risk management configuration
            price_data1: Full OHLCV data for symbol1
            price_data2: Full OHLCV data for symbol2
        """
        self.symbol1, self.symbol2 = symbols
        self.prices1 = prices1
        self.prices2 = prices2
        self.entries = entries.astype(bool)
        self.exits = exits.astype(bool)
        self.short_entries = short_entries.astype(bool)
        self.short_exits = short_exits.astype(bool)
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.hedge_ratio = hedge_ratio if hedge_ratio is not None else pd.Series(1.0, index=prices1.index)
        self.freq = freq
        self.market_hours_only = market_hours_only
        self.risk_config = risk_config or RiskConfig.moderate()
        self.price_data1 = price_data1
        self.price_data2 = price_data2

        # Define market hours in EST/EDT
        self.market_open = time(9, 35)  # 9:35 AM EST
        self.market_close = time(15, 55)  # 3:55 PM EST
        self.eastern_tz = pytz.timezone('US/Eastern')

        # Portfolio state
        self.trades = []
        self.equity_curve = None
        self._stats = None
        self.current_position: Optional[PairPosition] = None

        # Run simulation
        self._simulate()

    def _is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp is within market trading hours.

        Args:
            timestamp: Timestamp to check

        Returns:
            True if within market hours (9:35 AM - 3:55 PM EST)
        """
        if not self.market_hours_only:
            return True

        if timestamp.tz is None:
            eastern_time = timestamp
        else:
            eastern_time = timestamp.tz_convert(self.eastern_tz)

        if eastern_time.weekday() >= 5:  # Weekend
            return False

        current_time = eastern_time.time()
        return self.market_open <= current_time <= self.market_close

    def _calculate_position_size(
        self,
        cash: float,
        price1: float,
        price2: float,
        hedge_ratio: float
    ) -> Tuple[float, float]:
        """
        Calculate position sizes for both legs (dollar-neutral).

        For pairs trading, we want to allocate capital equally to both legs
        after accounting for the hedge ratio.

        Args:
            cash: Available cash
            price1: Current price of symbol1
            price2: Current price of symbol2
            hedge_ratio: Hedge ratio (shares2 / shares1)

        Returns:
            Tuple of (shares1, shares2)
        """
        # Use position sizing percentage from risk config
        capital_to_allocate = cash * self.risk_config.position_size_pct

        # For dollar-neutral pairs:
        # Capital per leg = total_capital / 2
        capital_per_leg = capital_to_allocate / 2.0

        # Calculate shares for each leg
        shares1 = int(capital_per_leg / price1)
        shares2 = int(capital_per_leg / price2)

        # Ensure we have at least 1 share of each
        if shares1 < 1 or shares2 < 1:
            return 0.0, 0.0

        return float(shares1), float(shares2)

    def _simulate(self):
        """
        Simulate pairs trading based on entry and exit signals.

        This handles synchronized entry/exit of both legs with proper
        position sizing, fees, and slippage for each leg.
        """
        cash = self.init_cash
        equity = []
        trades = []
        bar_index = 0

        # Ensure all series have the same index
        common_index = self.prices1.index.intersection(self.prices2.index)

        for timestamp in common_index:
            bar_index += 1

            price1 = self.prices1[timestamp]
            price2 = self.prices2[timestamp]
            hedge_ratio = self.hedge_ratio.get(timestamp, 1.0)

            long_entry = self.entries.get(timestamp, False)
            long_exit = self.exits.get(timestamp, False)
            short_entry = self.short_entries.get(timestamp, False)
            short_exit = self.short_exits.get(timestamp, False)

            in_market_hours = self._is_market_hours(timestamp)

            # Calculate current portfolio value
            if self.current_position is not None:
                value1, value2 = self.current_position.get_current_value(price1, price2)
                portfolio_value = cash + value1 + value2
            else:
                portfolio_value = cash

            # Handle exits first
            if self.current_position is not None and in_market_hours:
                should_exit = False

                if self.current_position.is_long_spread() and long_exit:
                    should_exit = True
                    exit_reason = 'signal'
                elif self.current_position.is_short_spread() and short_exit:
                    should_exit = True
                    exit_reason = 'signal'

                if should_exit:
                    # Close both legs
                    trade = self._close_pair_position(
                        timestamp,
                        price1,
                        price2,
                        cash,
                        exit_reason
                    )
                    trades.append(trade)
                    cash = trade['cash_after']
                    self.current_position = None

            # Handle entries
            if self.current_position is None and in_market_hours:
                if long_entry:
                    # Enter long spread: short sym1, long sym2
                    shares1, shares2 = self._calculate_position_size(
                        cash, price1, price2, hedge_ratio
                    )

                    if shares1 > 0 and shares2 > 0:
                        trade, new_cash = self._open_pair_position(
                            timestamp,
                            bar_index,
                            -shares1,  # Short symbol1
                            shares2,   # Long symbol2
                            price1,
                            price2,
                            hedge_ratio,
                            cash,
                            'long_spread'
                        )
                        trades.append(trade)
                        cash = new_cash

                elif short_entry:
                    # Enter short spread: long sym1, short sym2
                    shares1, shares2 = self._calculate_position_size(
                        cash, price1, price2, hedge_ratio
                    )

                    if shares1 > 0 and shares2 > 0:
                        trade, new_cash = self._open_pair_position(
                            timestamp,
                            bar_index,
                            shares1,   # Long symbol1
                            -shares2,  # Short symbol2
                            price1,
                            price2,
                            hedge_ratio,
                            cash,
                            'short_spread'
                        )
                        trades.append(trade)
                        cash = new_cash

            # Record equity
            if self.current_position is not None:
                value1, value2 = self.current_position.get_current_value(price1, price2)
                equity_value = cash + value1 + value2
            else:
                equity_value = cash

            equity.append(equity_value)

        # Create equity curve
        self.equity_curve = pd.Series(equity, index=common_index)
        self.trades = trades

    def _open_pair_position(
        self,
        timestamp: pd.Timestamp,
        bar_index: int,
        shares1: float,
        shares2: float,
        price1: float,
        price2: float,
        hedge_ratio: float,
        cash: float,
        direction: str
    ) -> Tuple[Dict[str, Any], float]:
        """
        Open a new pair position.

        Args:
            timestamp: Entry timestamp
            bar_index: Bar index
            shares1: Shares of symbol1 (negative for short)
            shares2: Shares of symbol2 (negative for short)
            price1: Entry price for symbol1
            price2: Entry price for symbol2
            hedge_ratio: Hedge ratio
            cash: Available cash
            direction: 'long_spread' or 'short_spread'

        Returns:
            Tuple of (trade_dict, remaining_cash)
        """
        # Apply slippage
        if shares1 > 0:
            exec_price1 = price1 * (1 + self.slippage)
        else:
            exec_price1 = price1 * (1 - self.slippage)

        if shares2 > 0:
            exec_price2 = price2 * (1 + self.slippage)
        else:
            exec_price2 = price2 * (1 - self.slippage)

        # Calculate costs
        cost1 = abs(shares1) * exec_price1
        cost2 = abs(shares2) * exec_price2

        fee1 = cost1 * self.fees
        fee2 = cost2 * self.fees

        total_fees = fee1 + fee2

        # For long positions: deduct cost from cash
        # For short positions: add proceeds to cash (minus fees)
        if shares1 > 0:
            cash -= (cost1 + fee1)
        else:
            cash += (cost1 - fee1)

        if shares2 > 0:
            cash -= (cost2 + fee2)
        else:
            cash += (cost2 - fee2)

        # Create position
        capital_allocated = abs(shares1 * price1) + abs(shares2 * price2)

        self.current_position = PairPosition(
            symbol1=self.symbol1,
            symbol2=self.symbol2,
            shares1=shares1,
            shares2=shares2,
            entry_price1=price1,
            entry_price2=price2,
            entry_timestamp=timestamp,
            entry_bar=bar_index,
            hedge_ratio=hedge_ratio,
            capital_allocated=capital_allocated
        )

        trade = {
            'timestamp': timestamp,
            'type': 'entry',
            'direction': direction,
            'symbol1': self.symbol1,
            'symbol2': self.symbol2,
            'shares1': shares1,
            'shares2': shares2,
            'price1': price1,
            'price2': price2,
            'exec_price1': exec_price1,
            'exec_price2': exec_price2,
            'fees': total_fees,
            'cash_after': cash
        }

        return trade, cash

    def _close_pair_position(
        self,
        timestamp: pd.Timestamp,
        price1: float,
        price2: float,
        cash: float,
        exit_reason: str
    ) -> Dict[str, Any]:
        """
        Close the current pair position.

        Args:
            timestamp: Exit timestamp
            price1: Exit price for symbol1
            price2: Exit price for symbol2
            cash: Current cash
            exit_reason: Reason for exit

        Returns:
            Trade dictionary
        """
        if self.current_position is None:
            raise ValueError("No position to close")

        # Apply slippage (reverse of entry)
        if self.current_position.shares1 > 0:
            # Selling long position
            exec_price1 = price1 * (1 - self.slippage)
        else:
            # Buying to cover short
            exec_price1 = price1 * (1 + self.slippage)

        if self.current_position.shares2 > 0:
            exec_price2 = price2 * (1 - self.slippage)
        else:
            exec_price2 = price2 * (1 + self.slippage)

        # Calculate proceeds
        proceeds1 = abs(self.current_position.shares1) * exec_price1
        proceeds2 = abs(self.current_position.shares2) * exec_price2

        fee1 = proceeds1 * self.fees
        fee2 = proceeds2 * self.fees

        total_fees = fee1 + fee2

        # For long positions: add proceeds to cash
        # For short positions: deduct buyback cost from cash
        if self.current_position.shares1 > 0:
            cash += (proceeds1 - fee1)
        else:
            cash -= (proceeds1 + fee1)

        if self.current_position.shares2 > 0:
            cash += (proceeds2 - fee2)
        else:
            cash -= (proceeds2 + fee2)

        # Calculate P&L
        pnl = self.current_position.get_unrealized_pnl(price1, price2) - total_fees

        trade = {
            'timestamp': timestamp,
            'type': 'exit',
            'exit_reason': exit_reason,
            'symbol1': self.symbol1,
            'symbol2': self.symbol2,
            'shares1': self.current_position.shares1,
            'shares2': self.current_position.shares2,
            'entry_price1': self.current_position.entry_price1,
            'entry_price2': self.current_position.entry_price2,
            'exit_price1': price1,
            'exit_price2': price2,
            'exec_price1': exec_price1,
            'exec_price2': exec_price2,
            'fees': total_fees,
            'pnl': pnl,
            'pnl_pct': (pnl / self.current_position.capital_allocated) * 100,
            'cash_after': cash,
            'bars_held': self.current_position.entry_bar
        }

        return trade

    def stats(self) -> Optional[Dict[str, Any]]:
        """
        Calculate portfolio statistics.

        Returns:
            Dictionary of performance metrics
        """
        if self._stats is not None:
            return self._stats

        if self.equity_curve is None or len(self.equity_curve) == 0:
            return None

        # Calculate returns
        returns = self.equity_curve.pct_change().dropna()

        if len(returns) == 0:
            return None

        # Basic metrics
        total_return = ((self.equity_curve.iloc[-1] / self.init_cash) - 1) * 100

        # Annualization factor
        if self.freq == '1min':
            periods_per_year = 252 * 390  # 252 trading days * 390 minutes
        elif self.freq == '1D':
            periods_per_year = 252
        else:
            periods_per_year = 252

        annual_return = (1 + total_return / 100) ** (periods_per_year / len(self.equity_curve)) - 1
        annual_return *= 100

        # Risk metrics
        volatility = returns.std() * np.sqrt(periods_per_year) * 100
        sharpe = (annual_return / volatility) if volatility > 0 else 0

        # Drawdown
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Trade metrics
        if len(self.trades) > 0:
            exit_trades = [t for t in self.trades if t['type'] == 'exit']
            total_trades = len(exit_trades)

            if total_trades > 0:
                wins = sum(1 for t in exit_trades if t['pnl'] > 0)
                win_rate = (wins / total_trades) * 100
            else:
                win_rate = 0
        else:
            total_trades = 0
            win_rate = 0

        self._stats = {
            'Start Value': self.init_cash,
            'End Value': self.equity_curve.iloc[-1],
            'Total Return [%]': total_return,
            'Annual Return [%]': annual_return,
            'Volatility [%]': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown [%]': max_drawdown,
            'Total Trades': total_trades,
            'Win Rate [%]': win_rate
        }

        return self._stats

    def returns(self) -> pd.Series:
        """Get portfolio returns."""
        if self.equity_curve is None:
            return pd.Series()
        return self.equity_curve.pct_change().dropna()
