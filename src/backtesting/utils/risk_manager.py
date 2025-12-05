"""
Risk management and stop loss logic for backtesting.

This module provides risk management functionality including:
- Portfolio-level constraints (max positions, max heat)
- Stop loss monitoring (percentage, ATR, time, profit target)
- Position exit signals based on risk rules

Classes:
    StopLoss: Base class for stop loss logic
    PercentageStopLoss: Fixed percentage stop loss
    ATRStopLoss: ATR-based trailing stop loss
    TimeStopLoss: Time-based exit
    ProfitTargetStopLoss: Take profit + stop loss
    RiskManager: Portfolio-level risk management
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Position:
    """
    Represents an open position.

    Attributes:
        symbol: Stock symbol
        entry_price: Price at entry
        shares: Number of shares
        entry_bar: Bar index at entry
        stop_price: Current stop loss price (None = no stop)
        highest_price: Highest price since entry (for trailing stops)
    """
    symbol: str
    entry_price: float
    shares: int
    entry_bar: int
    stop_price: Optional[float] = None
    highest_price: Optional[float] = None

    def __post_init__(self):
        """Initialize highest price to entry price."""
        if self.highest_price is None:
            self.highest_price = self.entry_price

    @property
    def is_short(self) -> bool:
        """True if this is a short position (negative shares)."""
        return self.shares < 0

    @property
    def position_value(self) -> float:
        """Current position value at entry price."""
        return self.entry_price * abs(self.shares)

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.is_short:
            # Short: profit when price drops
            return (self.entry_price - current_price) * abs(self.shares)
        else:
            # Long: profit when price rises
            return (current_price - self.entry_price) * self.shares

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.is_short:
            # Short: profit when price drops
            return (self.entry_price - current_price) / self.entry_price
        else:
            # Long: profit when price rises
            return (current_price - self.entry_price) / self.entry_price


class StopLoss(ABC):
    """
    Abstract base class for stop loss logic.

    Subclasses must implement:
    - should_exit: Whether position should be exited
    - update: Update stop loss based on new price
    """

    @abstractmethod
    def should_exit(
        self,
        position: Position,
        current_price: float,
        current_bar: int
    ) -> bool:
        """
        Check if position should be exited.

        Args:
            position: Current position
            current_price: Current price
            current_bar: Current bar index

        Returns:
            True if position should be exited
        """
        pass

    @abstractmethod
    def update(
        self,
        position: Position,
        current_price: float,
        price_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Update stop loss price based on new information.

        Args:
            position: Current position
            current_price: Current price
            price_data: Historical price data (for ATR calculation)
        """
        pass


class PercentageStopLoss(StopLoss):
    """
    Fixed percentage stop loss.

    Exits when price drops by a fixed percentage from entry.

    Example:
        stop = PercentageStopLoss(stop_loss_pct=0.02)  # 2% stop
        position = Position('AAPL', entry_price=150, shares=100, entry_bar=0)

        # Price drops to $147 (-2%)
        should_exit = stop.should_exit(position, current_price=147, current_bar=1)
        # Returns: True
    """

    def __init__(self, stop_loss_pct: float = 0.02):
        """
        Args:
            stop_loss_pct: Stop loss percentage (0.02 = 2%)
        """
        if not 0.0 < stop_loss_pct <= 1.0:
            raise ValueError(
                f"stop_loss_pct must be between 0 and 1, got {stop_loss_pct}"
            )

        self.stop_loss_pct = stop_loss_pct

    def should_exit(
        self,
        position: Position,
        current_price: float,
        current_bar: int
    ) -> bool:
        """Check if stop loss triggered."""
        if position.stop_price is None:
            return False

        if position.is_short:
            # Short: exit when price rises above stop
            return current_price >= position.stop_price
        else:
            # Long: exit when price drops below stop
            return current_price <= position.stop_price

    def update(
        self,
        position: Position,
        current_price: float,
        price_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Set stop price based on entry price."""
        if position.is_short:
            # Short: stop is ABOVE entry (exit when price rises)
            position.stop_price = position.entry_price * (1.0 + self.stop_loss_pct)
        else:
            # Long: stop is BELOW entry (exit when price drops)
            position.stop_price = position.entry_price * (1.0 - self.stop_loss_pct)


class ATRStopLoss(StopLoss):
    """
    ATR-based trailing stop loss.

    Stop distance is based on ATR (Average True Range), which adjusts
    to market volatility. Stop trails the highest price achieved.

    Example:
        stop = ATRStopLoss(atr_multiplier=2.0, atr_lookback=14)
        position = Position('AAPL', entry_price=150, shares=100, entry_bar=0)

        # Update stop as price moves
        stop.update(position, current_price=155, price_data=df)
        # Stop price trails 2× ATR below highest price
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        atr_lookback: int = 14
    ):
        """
        Args:
            atr_multiplier: Number of ATRs for stop distance
            atr_lookback: Number of periods for ATR calculation
        """
        if atr_multiplier <= 0:
            raise ValueError(
                f"atr_multiplier must be positive, got {atr_multiplier}"
            )

        if atr_lookback < 1:
            raise ValueError(
                f"atr_lookback must be >= 1, got {atr_lookback}"
            )

        self.atr_multiplier = atr_multiplier
        self.atr_lookback = atr_lookback

    def calculate_atr(self, price_data: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        if len(price_data) < self.atr_lookback:
            raise ValueError(
                f"Need at least {self.atr_lookback} bars for ATR"
            )

        high = price_data['high']
        low = price_data['low']
        close = price_data['close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = moving average of true range
        atr = true_range.rolling(window=self.atr_lookback).mean().iloc[-1]

        return atr

    def should_exit(
        self,
        position: Position,
        current_price: float,
        current_bar: int
    ) -> bool:
        """Check if stop loss triggered."""
        if position.stop_price is None:
            return False

        return current_price <= position.stop_price

    def update(
        self,
        position: Position,
        current_price: float,
        price_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Update trailing stop based on ATR."""
        # Update highest price
        if position.highest_price is None or current_price > position.highest_price:
            position.highest_price = current_price

        # Calculate ATR
        if price_data is None:
            # Fallback to simple percentage if no price data
            position.stop_price = position.highest_price * 0.98
            return

        try:
            atr = self.calculate_atr(price_data)
        except (ValueError, KeyError):
            # Fallback to simple percentage
            position.stop_price = position.highest_price * 0.98
            return

        # Stop price = highest price - (ATR × multiplier)
        stop_distance = atr * self.atr_multiplier
        new_stop = position.highest_price - stop_distance

        # Only move stop up (trailing), never down
        if position.stop_price is None or new_stop > position.stop_price:
            position.stop_price = new_stop


class TimeStopLoss(StopLoss):
    """
    Time-based exit.

    Exits position after a fixed number of bars, regardless of P&L.
    Useful for mean-reversion strategies or preventing stale positions.

    Example:
        stop = TimeStopLoss(max_holding_bars=20)
        position = Position('AAPL', entry_price=150, shares=100, entry_bar=0)

        # After 20 bars
        should_exit = stop.should_exit(position, current_price=155, current_bar=20)
        # Returns: True (exit regardless of profit/loss)
    """

    def __init__(self, max_holding_bars: int = 20):
        """
        Args:
            max_holding_bars: Maximum holding period in bars
        """
        if max_holding_bars < 1:
            raise ValueError(
                f"max_holding_bars must be >= 1, got {max_holding_bars}"
            )

        self.max_holding_bars = max_holding_bars

    def should_exit(
        self,
        position: Position,
        current_price: float,
        current_bar: int
    ) -> bool:
        """Check if time limit exceeded."""
        bars_held = current_bar - position.entry_bar
        return bars_held >= self.max_holding_bars

    def update(
        self,
        position: Position,
        current_price: float,
        price_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Time stop doesn't need updating."""
        pass


class ProfitTargetStopLoss(StopLoss):
    """
    Combined profit target and stop loss.

    Exits on either:
    - Take profit: Price rises by take_profit_pct
    - Stop loss: Price drops by stop_loss_pct

    Example:
        stop = ProfitTargetStopLoss(
            stop_loss_pct=0.02,     # 2% stop loss
            take_profit_pct=0.05    # 5% take profit
        )
        position = Position('AAPL', entry_price=150, shares=100, entry_bar=0)

        # Price rises to $157.50 (+5%)
        should_exit = stop.should_exit(position, current_price=157.50, current_bar=1)
        # Returns: True (take profit triggered)
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.05
    ):
        """
        Args:
            stop_loss_pct: Stop loss percentage (0.02 = 2%)
            take_profit_pct: Take profit percentage (0.05 = 5%)
        """
        if not 0.0 < stop_loss_pct <= 1.0:
            raise ValueError(
                f"stop_loss_pct must be between 0 and 1, got {stop_loss_pct}"
            )

        if not 0.0 < take_profit_pct <= 10.0:
            raise ValueError(
                f"take_profit_pct must be between 0 and 10, got {take_profit_pct}"
            )

        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.stop_price = None
        self.target_price = None

    def should_exit(
        self,
        position: Position,
        current_price: float,
        current_bar: int
    ) -> bool:
        """Check if stop or target triggered."""
        if self.stop_price is None or self.target_price is None:
            return False

        if position.is_short:
            # Short: stop is above entry, target is below entry
            return current_price >= self.stop_price or current_price <= self.target_price
        else:
            # Long: stop is below entry, target is above entry
            return current_price <= self.stop_price or current_price >= self.target_price

    def update(
        self,
        position: Position,
        current_price: float,
        price_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Set stop and target prices."""
        if position.is_short:
            # Short: stop ABOVE entry (exit on rise), target BELOW entry (profit on drop)
            self.stop_price = position.entry_price * (1.0 + self.stop_loss_pct)
            self.target_price = position.entry_price * (1.0 - self.take_profit_pct)
        else:
            # Long: stop BELOW entry (exit on drop), target ABOVE entry (profit on rise)
            self.stop_price = position.entry_price * (1.0 - self.stop_loss_pct)
            self.target_price = position.entry_price * (1.0 + self.take_profit_pct)

        position.stop_price = self.stop_price


class RiskManager:
    """
    Portfolio-level risk management.

    Enforces portfolio-level constraints:
    - Maximum number of concurrent positions
    - Maximum single position size
    - Maximum portfolio heat (total capital at risk)
    - Stop loss management for all positions

    Example:
        from backtesting.utils.risk_config import RiskConfig

        config = RiskConfig.moderate()
        risk_mgr = RiskManager(config)

        # Check if new position allowed
        can_open = risk_mgr.can_open_position(
            portfolio_value=100000,
            position_value=10000,
            current_positions=5
        )

        # Monitor positions for stop loss exits
        exits = risk_mgr.check_exits(
            positions=open_positions,
            current_prices={'AAPL': 148, 'MSFT': 348},
            current_bar=10,
            price_data_dict={'AAPL': aapl_df, 'MSFT': msft_df}
        )
    """

    def __init__(self, config):
        """
        Args:
            config: RiskConfig instance
        """
        from .risk_config import RiskConfig  # Import here to avoid circular import

        if not isinstance(config, RiskConfig):
            raise TypeError(f"config must be RiskConfig, got {type(config)}")

        self.config = config

        # Create stop loss instance based on config
        self.stop_loss = self._create_stop_loss()

        # Track open positions
        self.positions: Dict[str, Position] = {}

    def _create_stop_loss(self) -> Optional[StopLoss]:
        """Create stop loss instance from config."""
        if not self.config.use_stop_loss:
            return None

        if self.config.stop_loss_type == 'percentage':
            return PercentageStopLoss(
                stop_loss_pct=self.config.stop_loss_pct
            )

        elif self.config.stop_loss_type == 'atr':
            return ATRStopLoss(
                atr_multiplier=self.config.atr_multiplier,
                atr_lookback=self.config.atr_lookback
            )

        elif self.config.stop_loss_type == 'time':
            if self.config.max_holding_bars is None:
                raise ValueError(
                    "max_holding_bars required for time-based stop loss"
                )
            return TimeStopLoss(
                max_holding_bars=self.config.max_holding_bars
            )

        elif self.config.stop_loss_type == 'profit_target':
            if self.config.take_profit_pct is None:
                raise ValueError(
                    "take_profit_pct required for profit_target stop loss"
                )
            return ProfitTargetStopLoss(
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct
            )

        else:
            raise ValueError(
                f"Unknown stop_loss_type: {self.config.stop_loss_type}"
            )

    def can_open_position(
        self,
        portfolio_value: float,
        position_value: float,
        current_positions: int = 0
    ) -> Tuple[bool, str]:
        """
        Check if new position can be opened.

        Args:
            portfolio_value: Current portfolio value
            position_value: Value of proposed position
            current_positions: Number of current open positions

        Returns:
            (can_open, reason) tuple
        """
        # Check if risk management is disabled
        if not self.config.enabled:
            return True, "Risk management disabled"

        # Check max positions
        if current_positions >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"

        # Check max single position size
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        if position_pct > self.config.max_single_position_pct:
            return False, (
                f"Position size ({position_pct*100:.1f}%) exceeds "
                f"max ({self.config.max_single_position_pct*100:.1f}%)"
            )

        # Check portfolio heat (total risk)
        # Note: This is a simplified check. Full implementation would
        # calculate actual risk based on stop loss distances.
        max_new_position_value = portfolio_value * self.config.max_portfolio_heat
        if position_value > max_new_position_value:
            return False, (
                f"Position would exceed max portfolio heat "
                f"({self.config.max_portfolio_heat*100:.1f}%)"
            )

        return True, "OK"

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        entry_bar: int,
        price_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Add new position and initialize stop loss.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            shares: Number of shares
            entry_bar: Bar index at entry
            price_data: Historical price data (for ATR stop)
        """
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            shares=shares,
            entry_bar=entry_bar
        )

        # Initialize stop loss
        if self.stop_loss is not None:
            self.stop_loss.update(position, entry_price, price_data)

        self.positions[symbol] = position

    def remove_position(self, symbol: str) -> None:
        """Remove position from tracking."""
        if symbol in self.positions:
            del self.positions[symbol]

    def check_exits(
        self,
        current_prices: Dict[str, float],
        current_bar: int,
        price_data_dict: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, str]:
        """
        Check all positions for exit signals.

        Args:
            current_prices: Dict mapping symbol -> current price
            current_bar: Current bar index
            price_data_dict: Dict mapping symbol -> price DataFrame (for ATR)

        Returns:
            Dict mapping symbol -> exit reason for positions to exit
        """
        if self.stop_loss is None:
            return {}

        exits = {}

        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            price_data = price_data_dict.get(symbol) if price_data_dict else None

            # Update stop loss (for trailing stops)
            self.stop_loss.update(position, current_price, price_data)

            # Check if should exit
            if self.stop_loss.should_exit(position, current_price, current_bar):
                # Determine exit reason
                pnl_pct = position.unrealized_pnl_pct(current_price)

                if isinstance(self.stop_loss, TimeStopLoss):
                    reason = f"Time stop (held {current_bar - position.entry_bar} bars)"
                elif isinstance(self.stop_loss, ProfitTargetStopLoss):
                    if pnl_pct > 0:
                        reason = f"Profit target ({pnl_pct*100:.1f}%)"
                    else:
                        reason = f"Stop loss ({pnl_pct*100:.1f}%)"
                else:
                    reason = f"Stop loss ({pnl_pct*100:.1f}%)"

                exits[symbol] = reason

        return exits

    def get_position_info(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """
        Get detailed information about all positions.

        Args:
            current_prices: Dict mapping symbol -> current price

        Returns:
            DataFrame with position details
        """
        info = []

        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            info.append({
                'symbol': symbol,
                'shares': position.shares,
                'entry_price': position.entry_price,
                'current_price': current_price,
                'stop_price': position.stop_price,
                'unrealized_pnl': position.unrealized_pnl(current_price),
                'unrealized_pnl_pct': position.unrealized_pnl_pct(current_price) * 100,
                'bars_held': 0  # Would need current_bar to calculate
            })

        return pd.DataFrame(info)
