"""
Position sizing strategies for pairs trading.

This module provides various position sizing methods for pairs trading,
allowing flexible capital allocation across both legs of the pair.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class PairsPositionSizer(ABC):
    """
    Abstract base class for pairs position sizing strategies.

    Pairs position sizers determine how much capital to allocate to each
    leg of a pair trade, considering factors like:
    - Total available capital
    - Risk tolerance
    - Price levels
    - Volatilities
    - Hedge ratios
    """

    @abstractmethod
    def calculate_position_size(
        self,
        cash: float,
        price1: float,
        price2: float,
        hedge_ratio: float,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Calculate position sizes for both legs of the pair.

        Args:
            cash: Available cash for allocation
            price1: Current price of symbol1
            price2: Current price of symbol2
            hedge_ratio: Hedge ratio (shares2 / shares1)
            **kwargs: Additional parameters specific to sizing method

        Returns:
            Tuple of (shares1, shares2) - number of shares for each leg
        """
        pass


class DollarNeutralSizer(PairsPositionSizer):
    """
    Dollar-neutral position sizing (equal capital to both legs).

    This is the standard approach for pairs trading. Capital is allocated
    equally to both legs after accounting for the hedge ratio.

    Example:
        $10,000 available, position_pct = 0.5 (50%)
        Allocate $5,000 to the pair
        Split equally: $2,500 to each leg

        If price1=$100, price2=$200:
        shares1 = $2,500 / $100 = 25 shares
        shares2 = $2,500 / $200 = 12.5 shares (round to 12)

    Args:
        position_pct: Percentage of cash to allocate to pair (default: 0.1 = 10%)
        min_shares: Minimum shares required per leg (default: 1)
    """

    def __init__(self, position_pct: float = 0.1, min_shares: int = 1):
        """
        Initialize dollar-neutral sizer.

        Args:
            position_pct: Percentage of capital to allocate (0.0 to 1.0)
            min_shares: Minimum shares per leg (typically 1)
        """
        if not 0.0 < position_pct <= 1.0:
            raise ValueError(f"position_pct must be between 0 and 1, got {position_pct}")
        if min_shares < 1:
            raise ValueError(f"min_shares must be >= 1, got {min_shares}")

        self.position_pct = position_pct
        self.min_shares = min_shares

    def calculate_position_size(
        self,
        cash: float,
        price1: float,
        price2: float,
        hedge_ratio: float,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Calculate dollar-neutral position sizes.

        Allocates capital equally to both legs, ensuring we can afford
        at least min_shares of each.
        """
        # Total capital to allocate to this pair
        capital_to_allocate = cash * self.position_pct

        # Split equally between both legs
        capital_per_leg = capital_to_allocate / 2.0

        # Calculate shares for each leg
        shares1 = int(capital_per_leg / price1)
        shares2 = int(capital_per_leg / price2)

        # Ensure minimum shares requirement
        if shares1 < self.min_shares or shares2 < self.min_shares:
            return 0.0, 0.0

        return float(shares1), float(shares2)


class VolatilityAdjustedSizer(PairsPositionSizer):
    """
    Volatility-adjusted position sizing for pairs.

    Allocates more capital to the less volatile leg, aiming for equal
    risk contribution from both legs.

    The idea: If symbol1 has 2x the volatility of symbol2, we should
    allocate 2x more capital to symbol2 to balance risk.

    Risk contribution per leg:
        risk1 = capital1 * volatility1
        risk2 = capital2 * volatility2

    For equal risk: capital1 * vol1 = capital2 * vol2
    Therefore: capital1 / capital2 = vol2 / vol1

    Args:
        position_pct: Percentage of cash to allocate to pair
        min_shares: Minimum shares per leg
        volatility_lookback: Days to calculate volatility (default: 20)
    """

    def __init__(
        self,
        position_pct: float = 0.1,
        min_shares: int = 1,
        volatility_lookback: int = 20
    ):
        """Initialize volatility-adjusted sizer."""
        if not 0.0 < position_pct <= 1.0:
            raise ValueError(f"position_pct must be between 0 and 1, got {position_pct}")
        if min_shares < 1:
            raise ValueError(f"min_shares must be >= 1, got {min_shares}")
        if volatility_lookback < 2:
            raise ValueError(f"volatility_lookback must be >= 2, got {volatility_lookback}")

        self.position_pct = position_pct
        self.min_shares = min_shares
        self.volatility_lookback = volatility_lookback

    def calculate_position_size(
        self,
        cash: float,
        price1: float,
        price2: float,
        hedge_ratio: float,
        volatility1: float = None,
        volatility2: float = None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Calculate volatility-adjusted position sizes.

        Requires volatility1 and volatility2 in kwargs.
        If not provided, falls back to dollar-neutral sizing.
        """
        if volatility1 is None or volatility2 is None:
            # Fall back to dollar-neutral if no volatility data
            return self._dollar_neutral_fallback(cash, price1, price2)

        if volatility1 <= 0 or volatility2 <= 0:
            # Invalid volatilities - fall back
            return self._dollar_neutral_fallback(cash, price1, price2)

        # Total capital to allocate
        capital_to_allocate = cash * self.position_pct

        # Calculate inverse volatility weights
        # More capital to less volatile asset
        weight1 = 1.0 / volatility1
        weight2 = 1.0 / volatility2
        total_weight = weight1 + weight2

        # Normalize to get capital allocation
        capital1 = capital_to_allocate * (weight1 / total_weight)
        capital2 = capital_to_allocate * (weight2 / total_weight)

        # Calculate shares
        shares1 = int(capital1 / price1)
        shares2 = int(capital2 / price2)

        # Ensure minimum
        if shares1 < self.min_shares or shares2 < self.min_shares:
            return 0.0, 0.0

        return float(shares1), float(shares2)

    def _dollar_neutral_fallback(
        self,
        cash: float,
        price1: float,
        price2: float
    ) -> Tuple[float, float]:
        """Fallback to dollar-neutral sizing."""
        sizer = DollarNeutralSizer(self.position_pct, self.min_shares)
        return sizer.calculate_position_size(cash, price1, price2, 1.0)


class RiskParitySizer(PairsPositionSizer):
    """
    Risk-parity position sizing for pairs.

    Similar to volatility-adjusted, but also considers correlation
    between the two assets. Aims for equal risk-adjusted returns.

    This is a simplified risk-parity approach. Full risk-parity would
    require optimization, but this provides a good heuristic.

    Args:
        position_pct: Percentage of cash to allocate
        min_shares: Minimum shares per leg
        target_risk: Target risk level as fraction of capital (default: 0.02 = 2%)
    """

    def __init__(
        self,
        position_pct: float = 0.1,
        min_shares: int = 1,
        target_risk: float = 0.02
    ):
        """Initialize risk-parity sizer."""
        if not 0.0 < position_pct <= 1.0:
            raise ValueError(f"position_pct must be between 0 and 1, got {position_pct}")
        if min_shares < 1:
            raise ValueError(f"min_shares must be >= 1, got {min_shares}")
        if not 0.0 < target_risk <= 0.1:
            raise ValueError(f"target_risk must be between 0 and 0.1, got {target_risk}")

        self.position_pct = position_pct
        self.min_shares = min_shares
        self.target_risk = target_risk

    def calculate_position_size(
        self,
        cash: float,
        price1: float,
        price2: float,
        hedge_ratio: float,
        volatility1: float = None,
        volatility2: float = None,
        correlation: float = None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Calculate risk-parity position sizes.

        Requires volatility1, volatility2, and optionally correlation.
        Falls back to volatility-adjusted if correlation not provided.
        """
        if volatility1 is None or volatility2 is None:
            # Fall back to dollar-neutral
            return self._dollar_neutral_fallback(cash, price1, price2)

        if volatility1 <= 0 or volatility2 <= 0:
            return self._dollar_neutral_fallback(cash, price1, price2)

        # If no correlation provided, assume moderate positive correlation (0.5)
        if correlation is None:
            correlation = 0.5

        # Clip correlation to valid range
        correlation = np.clip(correlation, -0.99, 0.99)

        # Calculate portfolio risk for equal weights
        # portfolio_var = w1^2 * var1 + w2^2 * var2 + 2 * w1 * w2 * cov12
        # For equal weights (w1=w2=0.5):
        # portfolio_var = 0.25 * (var1 + var2 + 2 * corr * std1 * std2)

        var1 = volatility1 ** 2
        var2 = volatility2 ** 2
        cov12 = correlation * volatility1 * volatility2

        portfolio_variance = 0.25 * (var1 + var2 + 2 * cov12)
        portfolio_volatility = np.sqrt(portfolio_variance)

        if portfolio_volatility <= 0:
            return self._dollar_neutral_fallback(cash, price1, price2)

        # Scale position size to target risk
        # target_risk = position_size * portfolio_volatility
        # position_size = target_risk / portfolio_volatility
        risk_scaled_allocation = cash * self.target_risk / portfolio_volatility

        # Cap at position_pct
        capital_to_allocate = min(risk_scaled_allocation, cash * self.position_pct)

        # Split equally (risk parity with hedging)
        capital_per_leg = capital_to_allocate / 2.0

        shares1 = int(capital_per_leg / price1)
        shares2 = int(capital_per_leg / price2)

        if shares1 < self.min_shares or shares2 < self.min_shares:
            return 0.0, 0.0

        return float(shares1), float(shares2)

    def _dollar_neutral_fallback(
        self,
        cash: float,
        price1: float,
        price2: float
    ) -> Tuple[float, float]:
        """Fallback to dollar-neutral sizing."""
        sizer = DollarNeutralSizer(self.position_pct, self.min_shares)
        return sizer.calculate_position_size(cash, price1, price2, 1.0)


def create_pairs_sizer(method: str = 'dollar_neutral', **kwargs) -> PairsPositionSizer:
    """
    Factory function to create pairs position sizers.

    Args:
        method: Sizing method ('dollar_neutral', 'volatility_adjusted', 'risk_parity')
        **kwargs: Parameters for the specific sizer

    Returns:
        PairsPositionSizer instance

    Example:
        >>> sizer = create_pairs_sizer('dollar_neutral', position_pct=0.2)
        >>> shares1, shares2 = sizer.calculate_position_size(10000, 100, 200, 1.0)
    """
    if method == 'dollar_neutral':
        return DollarNeutralSizer(**kwargs)
    elif method == 'volatility_adjusted':
        return VolatilityAdjustedSizer(**kwargs)
    elif method == 'risk_parity':
        return RiskParitySizer(**kwargs)
    else:
        raise ValueError(
            f"Unknown sizing method: {method}. "
            f"Choose from: 'dollar_neutral', 'volatility_adjusted', 'risk_parity'"
        )
