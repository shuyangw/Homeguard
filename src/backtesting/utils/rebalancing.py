"""
Portfolio rebalancing utilities.

Provides triggers and execution logic for portfolio rebalancing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class RebalancingEvent:
    """Record of a rebalancing event."""
    timestamp: pd.Timestamp
    trigger: str  # 'periodic', 'drift', 'signal'
    trades: List[Dict]  # List of {symbol, action, shares, price}
    total_cost: float  # Total fees incurred
    portfolio_value_before: float
    portfolio_value_after: float


class RebalancingTrigger(ABC):
    """Base class for rebalancing triggers."""

    @abstractmethod
    def should_rebalance(
        self,
        timestamp: pd.Timestamp,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        **kwargs
    ) -> bool:
        """
        Determine if portfolio should be rebalanced.

        Args:
            timestamp: Current timestamp
            current_weights: Current position weights {symbol: weight}
            target_weights: Target position weights {symbol: weight}
            **kwargs: Additional context

        Returns:
            True if should rebalance, False otherwise
        """
        pass


class PeriodicRebalancing(RebalancingTrigger):
    """
    Rebalance on calendar schedule (monthly, quarterly, annually).

    Example:
    - Monthly: First trading day of each month
    - Quarterly: Jan 1, Apr 1, Jul 1, Oct 1
    """

    def __init__(
        self,
        frequency: Literal['monthly', 'quarterly', 'annually'] = 'quarterly'
    ):
        """
        Initialize periodic rebalancing.

        Args:
            frequency: Rebalancing frequency
        """
        self.frequency = frequency
        self.last_rebalance: Optional[pd.Timestamp] = None

    def should_rebalance(
        self,
        timestamp: pd.Timestamp,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        **kwargs
    ) -> bool:
        """Check if it's time for periodic rebalance."""
        # First rebalance
        if self.last_rebalance is None:
            self.last_rebalance = timestamp
            return False  # Don't rebalance on first bar

        if self.frequency == 'monthly':
            # Rebalance on first day of each month
            if timestamp.month != self.last_rebalance.month:
                if timestamp.day <= 5:  # Allow first 5 days of month
                    self.last_rebalance = timestamp
                    return True

        elif self.frequency == 'quarterly':
            # Rebalance on Jan 1, Apr 1, Jul 1, Oct 1
            if timestamp.month in [1, 4, 7, 10]:
                if self.last_rebalance.month not in [1, 4, 7, 10] or \
                   timestamp.year != self.last_rebalance.year:
                    if timestamp.day <= 5:  # Allow first 5 days of quarter
                        self.last_rebalance = timestamp
                        return True

        elif self.frequency == 'annually':
            # Rebalance on January 1
            if timestamp.month == 1 and self.last_rebalance.year != timestamp.year:
                if timestamp.day <= 5:  # Allow first 5 days of year
                    self.last_rebalance = timestamp
                    return True

        return False


class ThresholdRebalancing(RebalancingTrigger):
    """
    Rebalance when position weights drift beyond threshold.

    Example:
    - Target: 10% per stock
    - Threshold: 5%
    - If any stock reaches 15% or 5%, trigger rebalance
    """

    def __init__(self, threshold_pct: float = 0.05):
        """
        Initialize threshold rebalancing.

        Args:
            threshold_pct: Drift threshold (0.05 = 5% drift triggers rebalance)
        """
        self.threshold_pct = threshold_pct

    def should_rebalance(
        self,
        timestamp: pd.Timestamp,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        **kwargs
    ) -> bool:
        """Check if any position has drifted beyond threshold."""
        for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            drift = abs(current - target)

            if drift > self.threshold_pct:
                return True

        return False


class SignalBasedRebalancing(RebalancingTrigger):
    """
    Rebalance when strategy signals change.

    Example:
    - Currently holding: AAPL, MSFT, GOOGL
    - New signals: AAPL, MSFT, NVDA (GOOGL -> NVDA)
    - Trigger rebalance to exit GOOGL, enter NVDA
    """

    def __init__(self):
        """Initialize signal-based rebalancing."""
        self.previous_signals: Optional[Dict[str, bool]] = None

    def should_rebalance(
        self,
        timestamp: pd.Timestamp,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        **kwargs
    ) -> bool:
        """Check if signals have changed."""
        current_signals = kwargs.get('current_signals', {})

        if self.previous_signals is None:
            self.previous_signals = current_signals
            return False

        # Check if set of symbols with signals has changed
        prev_symbols = {s for s, signal in self.previous_signals.items() if signal}
        curr_symbols = {s for s, signal in current_signals.items() if signal}

        signals_changed = prev_symbols != curr_symbols

        self.previous_signals = current_signals

        return signals_changed


class RebalancingExecutor:
    """
    Executes portfolio rebalancing.

    Calculates trades needed to move from current weights to target weights.
    """

    @staticmethod
    def calculate_rebalancing_trades(
        current_positions: Dict[str, int],  # {symbol: shares}
        current_prices: Dict[str, float],   # {symbol: price}
        target_weights: Dict[str, float],   # {symbol: target_weight}
        portfolio_value: float,
        cash_available: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Calculate trades needed to rebalance portfolio.

        Args:
            current_positions: Current holdings {symbol: shares}
            current_prices: Current prices {symbol: price}
            target_weights: Target weights {symbol: weight}
            portfolio_value: Total portfolio value
            cash_available: Available cash

        Returns:
            Tuple of (sells, buys) where each is a list of trade dicts
        """
        sells = []
        buys = []

        # Calculate current values
        current_values = {
            symbol: shares * current_prices.get(symbol, 0)
            for symbol, shares in current_positions.items()
            if symbol in current_prices
        }

        # Calculate target values
        target_values = {
            symbol: weight * portfolio_value
            for symbol, weight in target_weights.items()
        }

        # Determine trades
        all_symbols = set(list(current_positions.keys()) + list(target_weights.keys()))

        for symbol in all_symbols:
            if symbol not in current_prices:
                continue

            current_value = current_values.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            value_diff = target_value - current_value

            price = current_prices[symbol]

            if abs(value_diff) < 1.0:  # Ignore tiny differences
                continue

            if value_diff < 0:
                # Need to sell
                shares_to_sell = int(abs(value_diff) / price)
                if shares_to_sell > 0:
                    sells.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'shares': shares_to_sell,
                        'price': price,
                        'value': shares_to_sell * price
                    })

            else:
                # Need to buy
                shares_to_buy = int(value_diff / price)
                if shares_to_buy > 0:
                    buys.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': price,
                        'value': shares_to_buy * price
                    })

        return sells, buys

    @staticmethod
    def execute_rebalancing(
        current_positions: Dict[str, int],
        current_prices: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        cash: float,
        fees: float,
        slippage: float
    ) -> Tuple[Dict[str, int], float, List[Dict], float]:
        """
        Execute rebalancing trades.

        Args:
            current_positions: Current holdings
            current_prices: Current prices
            target_weights: Target weights
            portfolio_value: Total portfolio value
            cash: Available cash
            fees: Trading fees (decimal)
            slippage: Slippage (decimal)

        Returns:
            Tuple of (new_positions, new_cash, trades_executed, total_cost)
        """
        sells, buys = RebalancingExecutor.calculate_rebalancing_trades(
            current_positions=current_positions,
            current_prices=current_prices,
            target_weights=target_weights,
            portfolio_value=portfolio_value,
            cash_available=cash
        )

        new_positions = current_positions.copy()
        new_cash = cash
        trades_executed = []
        total_cost = 0.0

        # Execute sells first (to free up cash)
        for sell in sells:
            symbol = sell['symbol']
            shares = sell['shares']
            price = sell['price']

            if symbol not in new_positions:
                continue

            # Limit shares to what we actually have
            shares = min(shares, new_positions[symbol])

            if shares <= 0:
                continue

            # Apply slippage (worse fill on sell)
            slippage_adj = price * (1 - slippage)

            # Calculate proceeds
            proceeds = shares * slippage_adj
            fee = proceeds * fees
            net_proceeds = proceeds - fee

            # Update positions and cash
            new_positions[symbol] -= shares
            if new_positions[symbol] <= 0:
                del new_positions[symbol]

            new_cash += net_proceeds
            total_cost += fee

            trades_executed.append({
                **sell,
                'shares_executed': shares,
                'net_proceeds': net_proceeds,
                'fee': fee
            })

        # Execute buys (using freed cash)
        for buy in buys:
            symbol = buy['symbol']
            shares = buy['shares']
            price = buy['price']

            # Apply slippage (worse fill on buy)
            slippage_adj = price * (1 + slippage)

            # Calculate cost
            cost = shares * slippage_adj
            fee = cost * fees
            total_cost_buy = cost + fee

            # Check if we have enough cash
            if total_cost_buy > new_cash:
                # Reduce shares to fit available cash
                if new_cash <= 0:
                    continue

                shares = int((new_cash / (1 + fees)) / slippage_adj)

                if shares <= 0:
                    continue

                cost = shares * slippage_adj
                fee = cost * fees
                total_cost_buy = cost + fee

            # Execute buy
            if symbol not in new_positions:
                new_positions[symbol] = 0

            new_positions[symbol] += shares
            new_cash -= total_cost_buy
            total_cost += fee

            trades_executed.append({
                **buy,
                'shares_executed': shares,
                'cost': total_cost_buy,
                'fee': fee
            })

        return new_positions, new_cash, trades_executed, total_cost


def get_rebalancing_trigger(
    frequency: str,
    threshold_pct: float = 0.05
) -> Optional[RebalancingTrigger]:
    """
    Factory function to create rebalancing trigger.

    Args:
        frequency: 'never', 'monthly', 'quarterly', 'on_signal', 'drift'
        threshold_pct: Drift threshold for drift-based rebalancing

    Returns:
        RebalancingTrigger instance or None if frequency='never'
    """
    if frequency == 'never':
        return None

    elif frequency in ['monthly', 'quarterly', 'annually']:
        return PeriodicRebalancing(frequency=frequency)  # type: ignore[arg-type]

    elif frequency == 'drift':
        return ThresholdRebalancing(threshold_pct=threshold_pct)

    elif frequency == 'on_signal':
        return SignalBasedRebalancing()

    else:
        # Default to never
        return None
