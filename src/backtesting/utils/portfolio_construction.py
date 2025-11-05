"""
Portfolio construction utilities for multi-symbol portfolios.

Provides position sizing methods for allocating capital across multiple symbols.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class PortfolioSizer(ABC):
    """Base class for portfolio position sizers."""

    @abstractmethod
    def calculate_positions(
        self,
        portfolio_value: float,
        entry_signals: Dict[str, bool],
        prices: Dict[str, float],
        **kwargs
    ) -> Dict[str, int]:
        """
        Calculate position sizes (in shares) for each symbol.

        Args:
            portfolio_value: Current total portfolio value
            entry_signals: Dict of {symbol: should_enter}
            prices: Dict of {symbol: current_price}
            **kwargs: Additional data (volatilities, signal_strengths, etc.)

        Returns:
            Dict of {symbol: shares}
        """
        pass


class EqualWeightSizer(PortfolioSizer):
    """
    Allocate equal weight to each position.

    Example: 10 stocks, 10% each
    """

    def __init__(self, target_positions: int = 10):
        """
        Initialize equal weight sizer.

        Args:
            target_positions: Target number of positions (determines weight per position)
        """
        self.target_positions = target_positions
        self.weight_per_position = 1.0 / target_positions

    def calculate_positions(
        self,
        portfolio_value: float,
        entry_signals: Dict[str, bool],
        prices: Dict[str, float],
        **kwargs
    ) -> Dict[str, int]:
        """
        Calculate equal-weighted positions.

        Each signal gets equal allocation (portfolio_value / target_positions).
        """
        positions = {}

        # Count number of signals
        num_signals = sum(entry_signals.values())

        if num_signals == 0:
            return {}

        # Calculate allocation per signal
        # Use target_positions as denominator (not num_signals) for consistent sizing
        allocation_per_symbol = portfolio_value * self.weight_per_position

        for symbol, should_enter in entry_signals.items():
            if should_enter and symbol in prices:
                price = prices[symbol]
                if price > 0:
                    shares = int(allocation_per_symbol / price)
                    if shares > 0:
                        positions[symbol] = shares

        return positions


class RiskParitySizer(PortfolioSizer):
    """
    Size positions by inverse volatility (risk parity).

    High-volatility stocks get smaller positions to equalize risk contribution.

    Example:
    - Stock A: 1% daily vol → weight = 1/0.01 = 100
    - Stock B: 2% daily vol → weight = 1/0.02 = 50
    - Stock A gets 2x the allocation of Stock B
    """

    def __init__(self, lookback: int = 60, min_volatility: float = 0.001):
        """
        Initialize risk parity sizer.

        Args:
            lookback: Number of bars to use for volatility calculation
            min_volatility: Minimum volatility (prevents division by zero)
        """
        self.lookback = lookback
        self.min_volatility = min_volatility

    def calculate_positions(
        self,
        portfolio_value: float,
        entry_signals: Dict[str, bool],
        prices: Dict[str, float],
        **kwargs
    ) -> Dict[str, int]:
        """
        Calculate risk-parity weighted positions.

        Requires 'volatilities' in kwargs: Dict[str, float] mapping symbol to volatility.
        If not provided, falls back to equal weight.
        """
        volatilities = kwargs.get('volatilities', {})

        if not volatilities:
            # Fall back to equal weight if no volatility data
            equal_sizer = EqualWeightSizer(target_positions=len(entry_signals))
            return equal_sizer.calculate_positions(portfolio_value, entry_signals, prices)

        positions = {}

        # Calculate inverse volatility weights
        inv_vols = {}
        for symbol, should_enter in entry_signals.items():
            if should_enter and symbol in volatilities:
                vol = max(volatilities[symbol], self.min_volatility)
                inv_vols[symbol] = 1.0 / vol

        if not inv_vols:
            return {}

        # Normalize to weights
        total_inv_vol = sum(inv_vols.values())
        weights = {symbol: inv_vol / total_inv_vol for symbol, inv_vol in inv_vols.items()}

        # Allocate capital
        for symbol, weight in weights.items():
            if symbol in prices:
                price = prices[symbol]
                if price > 0:
                    allocation = portfolio_value * weight
                    shares = int(allocation / price)
                    if shares > 0:
                        positions[symbol] = shares

        return positions

    @staticmethod
    def calculate_volatilities(
        price_data: Dict[str, pd.Series],
        lookback: int = 60
    ) -> Dict[str, float]:
        """
        Calculate rolling volatility for each symbol.

        Args:
            price_data: Dict of {symbol: price_series}
            lookback: Number of bars for volatility calculation

        Returns:
            Dict of {symbol: volatility}
        """
        volatilities = {}

        for symbol, prices in price_data.items():
            if len(prices) < lookback:
                continue

            # Calculate returns
            returns = prices.pct_change().dropna()

            if len(returns) < 2:
                continue

            # Use last 'lookback' returns
            recent_returns = returns.iloc[-lookback:]

            # Calculate volatility (standard deviation of returns)
            vol = recent_returns.std()

            if pd.notna(vol) and vol > 0:
                volatilities[symbol] = vol

        return volatilities


class FixedCountSizer(PortfolioSizer):
    """
    Hold a fixed number of positions with risk constraints.

    Features:
    - Max N concurrent positions
    - Max single-stock weight constraint
    - Equal weight within constraints
    """

    def __init__(
        self,
        max_positions: int = 10,
        max_single_weight: float = 0.20
    ):
        """
        Initialize fixed count sizer.

        Args:
            max_positions: Maximum concurrent positions
            max_single_weight: Max % of portfolio in single stock (e.g., 0.20 = 20%)
        """
        self.max_positions = max_positions
        self.max_single_weight = max_single_weight
        self.target_weight = 1.0 / max_positions

    def calculate_positions(
        self,
        portfolio_value: float,
        entry_signals: Dict[str, bool],
        prices: Dict[str, float],
        **kwargs
    ) -> Dict[str, int]:
        """
        Calculate positions respecting count and weight constraints.

        Args:
            portfolio_value: Current portfolio value
            entry_signals: Entry signals
            prices: Current prices
            **kwargs: May include 'current_positions' (int) for available slots

        Returns:
            Dict of {symbol: shares}
        """
        current_positions_count = kwargs.get('current_positions', 0)

        # How many slots available?
        available_slots = self.max_positions - current_positions_count

        if available_slots <= 0:
            return {}

        # Get symbols to enter
        entry_symbols = [s for s, should_enter in entry_signals.items() if should_enter]

        # Limit to available slots
        # TODO: Rank by signal strength instead of taking first N
        symbols_to_enter = entry_symbols[:available_slots]

        if not symbols_to_enter:
            return {}

        positions = {}

        # Calculate weight per position (respecting max_single_weight)
        actual_weight = min(self.target_weight, self.max_single_weight)

        for symbol in symbols_to_enter:
            if symbol in prices:
                price = prices[symbol]
                if price > 0:
                    allocation = portfolio_value * actual_weight
                    shares = int(allocation / price)
                    if shares > 0:
                        positions[symbol] = shares

        return positions


class RankedSizer(PortfolioSizer):
    """
    Allocate capital proportional to signal strength.

    Stronger signals get more capital.

    Example:
    - Signal A strength: 0.8 → 40% allocation
    - Signal B strength: 0.6 → 30% allocation
    - Signal C strength: 0.6 → 30% allocation
    """

    def __init__(
        self,
        top_n: int = 10,
        min_strength: float = 0.0
    ):
        """
        Initialize ranked sizer.

        Args:
            top_n: Number of top signals to allocate to
            min_strength: Minimum signal strength threshold (0-1)
        """
        self.top_n = top_n
        self.min_strength = min_strength

    def calculate_positions(
        self,
        portfolio_value: float,
        entry_signals: Dict[str, bool],
        prices: Dict[str, float],
        **kwargs
    ) -> Dict[str, int]:
        """
        Calculate positions proportional to signal strength.

        Requires 'signal_strengths' in kwargs: Dict[str, float] mapping symbol to strength (0-1).
        If not provided, falls back to equal weight.
        """
        signal_strengths = kwargs.get('signal_strengths', {})

        if not signal_strengths:
            # Fall back to equal weight if no signal strengths
            equal_sizer = EqualWeightSizer(target_positions=self.top_n)
            return equal_sizer.calculate_positions(portfolio_value, entry_signals, prices)

        # Filter by entry signals and minimum strength
        valid_signals = {
            symbol: strength
            for symbol, strength in signal_strengths.items()
            if entry_signals.get(symbol, False) and strength >= self.min_strength
        }

        if not valid_signals:
            return {}

        # Sort by strength and take top N
        ranked = sorted(valid_signals.items(), key=lambda x: x[1], reverse=True)[:self.top_n]

        if not ranked:
            return {}

        # Calculate weights proportional to signal strength
        total_strength = sum(strength for _, strength in ranked)

        if total_strength == 0:
            return {}

        positions = {}

        for symbol, strength in ranked:
            if symbol in prices:
                price = prices[symbol]
                if price > 0:
                    weight = strength / total_strength
                    allocation = portfolio_value * weight
                    shares = int(allocation / price)
                    if shares > 0:
                        positions[symbol] = shares

        return positions


class AdaptiveWeightSizer(PortfolioSizer):
    """
    Adaptive weighting that combines multiple factors.

    Factors:
    - Signal strength
    - Volatility (risk parity component)
    - Momentum
    - Custom weights
    """

    def __init__(
        self,
        signal_strength_weight: float = 0.4,
        volatility_weight: float = 0.3,
        momentum_weight: float = 0.3,
        max_positions: int = 10
    ):
        """
        Initialize adaptive weight sizer.

        Args:
            signal_strength_weight: Weight for signal strength factor (0-1)
            volatility_weight: Weight for volatility factor (0-1)
            momentum_weight: Weight for momentum factor (0-1)
            max_positions: Maximum concurrent positions
        """
        self.signal_strength_weight = signal_strength_weight
        self.volatility_weight = volatility_weight
        self.momentum_weight = momentum_weight
        self.max_positions = max_positions

        # Normalize weights to sum to 1
        total = signal_strength_weight + volatility_weight + momentum_weight
        if total > 0:
            self.signal_strength_weight /= total
            self.volatility_weight /= total
            self.momentum_weight /= total

    def calculate_positions(
        self,
        portfolio_value: float,
        entry_signals: Dict[str, bool],
        prices: Dict[str, float],
        **kwargs
    ) -> Dict[str, int]:
        """
        Calculate adaptive-weighted positions.

        Kwargs may include:
        - signal_strengths: Dict[str, float]
        - volatilities: Dict[str, float]
        - momentums: Dict[str, float]
        """
        signal_strengths = kwargs.get('signal_strengths', {})
        volatilities = kwargs.get('volatilities', {})
        momentums = kwargs.get('momentums', {})

        # Get symbols to enter
        entry_symbols = [s for s, should_enter in entry_signals.items() if should_enter]

        if not entry_symbols:
            return {}

        # Calculate composite scores
        scores = {}

        for symbol in entry_symbols:
            score = 0.0
            weight_sum = 0.0

            # Signal strength component
            if symbol in signal_strengths:
                score += self.signal_strength_weight * signal_strengths[symbol]
                weight_sum += self.signal_strength_weight

            # Volatility component (inverse)
            if symbol in volatilities:
                vol = volatilities[symbol]
                if vol > 0:
                    # Normalize inverse volatility to 0-1 range
                    inv_vol = 1.0 / vol
                    score += self.volatility_weight * inv_vol
                    weight_sum += self.volatility_weight

            # Momentum component
            if symbol in momentums:
                score += self.momentum_weight * momentums[symbol]
                weight_sum += self.momentum_weight

            # Normalize by weights used
            if weight_sum > 0:
                scores[symbol] = score / weight_sum
            else:
                scores[symbol] = 0.0

        if not scores:
            # Fall back to equal weight
            equal_sizer = EqualWeightSizer(target_positions=self.max_positions)
            return equal_sizer.calculate_positions(portfolio_value, entry_signals, prices)

        # Rank and take top N
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.max_positions]

        # Calculate weights proportional to scores
        total_score = sum(score for _, score in ranked)

        if total_score == 0:
            # Equal weight if all scores are 0
            equal_sizer = EqualWeightSizer(target_positions=len(ranked))
            filtered_signals = {symbol: True for symbol, _ in ranked}
            return equal_sizer.calculate_positions(portfolio_value, filtered_signals, prices)

        positions = {}

        for symbol, score in ranked:
            if symbol in prices:
                price = prices[symbol]
                if price > 0:
                    weight = score / total_score
                    allocation = portfolio_value * weight
                    shares = int(allocation / price)
                    if shares > 0:
                        positions[symbol] = shares

        return positions
