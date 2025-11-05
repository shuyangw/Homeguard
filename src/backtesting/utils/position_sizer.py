"""
Position sizing algorithms for risk management.

This module provides various position sizing methods:
- Fixed Percentage: Allocate fixed % of portfolio per trade
- Fixed Dollar: Allocate fixed dollar amount per trade
- Volatility-Based (ATR): Equalize risk based on volatility
- Kelly Criterion: Mathematically optimal sizing based on edge
- Risk Parity: Equal risk contribution across positions

Classes:
    FixedPercentageSizer: Fixed percentage position sizing
    FixedDollarSizer: Fixed dollar amount position sizing
    VolatilityBasedSizer: ATR-based volatility position sizing
    KellyCriterionSizer: Kelly Criterion optimal sizing
    RiskParitySizer: Risk parity multi-asset sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class FixedPercentageSizer:
    """
    Allocate a fixed percentage of portfolio to each position.

    This is the simplest and most common position sizing method.
    Position size automatically scales with portfolio value.

    Example:
        sizer = FixedPercentageSizer(position_pct=0.10)
        shares = sizer.calculate_shares(
            portfolio_value=100000,
            price=150.00
        )
        # Returns: 66 shares ($9,900 position = 9.9% of portfolio)

    Args:
        position_pct: Percentage of portfolio per trade (0.0 to 1.0)
                     Default: 0.10 (10%)

    Recommended:
        Conservative: 0.02-0.05 (2-5%)
        Moderate: 0.05-0.10 (5-10%)
        Aggressive: 0.10-0.20 (10-20%)
    """

    def __init__(self, position_pct: float = 0.10):
        if not 0.0 < position_pct <= 1.0:
            raise ValueError(
                f"position_pct must be between 0 and 1, got {position_pct}"
            )

        self.position_pct = position_pct

    def calculate_shares(
        self,
        portfolio_value: float,
        price: float
    ) -> int:
        """
        Calculate number of shares to buy.

        Args:
            portfolio_value: Current portfolio value
            price: Current share price

        Returns:
            Number of shares (integer, rounded down)
        """
        if portfolio_value <= 0 or price <= 0:
            return 0

        position_value = portfolio_value * self.position_pct
        shares = int(position_value / price)

        return shares


class FixedDollarSizer:
    """
    Allocate a fixed dollar amount to each position.

    Best for small accounts where percentage-based sizing rounds to zero.
    Requires manual adjustment as portfolio grows.

    Example:
        sizer = FixedDollarSizer(position_dollars=10000)
        shares = sizer.calculate_shares(price=150.00)
        # Returns: 66 shares ($9,900 position)

    Args:
        position_dollars: Dollar amount per trade
                         Default: $10,000

    Warning:
        Fixed dollar sizing doesn't scale with portfolio growth.
        Transition to percentage-based sizing when account is larger.
    """

    def __init__(self, position_dollars: float = 10000.0):
        if position_dollars <= 0:
            raise ValueError(
                f"position_dollars must be positive, got {position_dollars}"
            )

        self.position_dollars = position_dollars

    def calculate_shares(
        self,
        price: float,
        portfolio_value: Optional[float] = None
    ) -> int:
        """
        Calculate number of shares to buy.

        Args:
            price: Current share price
            portfolio_value: Current portfolio value (optional, for validation)

        Returns:
            Number of shares (integer, rounded down)
        """
        if price <= 0:
            return 0

        # Optional: Prevent position from exceeding portfolio
        if portfolio_value is not None:
            max_position = min(self.position_dollars, portfolio_value * 0.95)
        else:
            max_position = self.position_dollars

        shares = int(max_position / price)

        return shares


class VolatilityBasedSizer:
    """
    Size positions inversely to volatility using ATR (Average True Range).

    Keeps dollar risk constant across all trades by allocating less
    to volatile stocks and more to stable stocks.

    Example:
        sizer = VolatilityBasedSizer(
            risk_pct=0.01,
            atr_multiplier=2.0,
            atr_lookback=14
        )
        shares = sizer.calculate_shares(
            portfolio_value=100000,
            price=150.00,
            price_data=df  # DataFrame with 'high', 'low', 'close' columns
        )

    Args:
        risk_pct: Percentage of portfolio to risk per trade (0.0 to 1.0)
                 Default: 0.01 (1%)
        atr_multiplier: How many ATRs for stop distance
                       Default: 2.0
        atr_lookback: Number of periods for ATR calculation
                     Default: 14

    Formula:
        Risk per Trade ($) = Portfolio Value × Risk %
        Stop Distance ($) = ATR × ATR Multiplier
        Shares = Risk per Trade / Stop Distance
    """

    def __init__(
        self,
        risk_pct: float = 0.01,
        atr_multiplier: float = 2.0,
        atr_lookback: int = 14
    ):
        if not 0.0 < risk_pct <= 0.05:
            raise ValueError(
                f"risk_pct should be between 0 and 0.05 (5%), got {risk_pct}"
            )

        if atr_multiplier <= 0:
            raise ValueError(
                f"atr_multiplier must be positive, got {atr_multiplier}"
            )

        if atr_lookback < 1:
            raise ValueError(
                f"atr_lookback must be >= 1, got {atr_lookback}"
            )

        self.risk_pct = risk_pct
        self.atr_multiplier = atr_multiplier
        self.atr_lookback = atr_lookback

    def calculate_atr(self, price_data: pd.DataFrame) -> float:
        """
        Calculate Average True Range.

        Args:
            price_data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            ATR value (average true range)
        """
        if len(price_data) < self.atr_lookback:
            raise ValueError(
                f"Need at least {self.atr_lookback} bars for ATR, got {len(price_data)}"
            )

        high = price_data['high']
        low = price_data['low']
        close = price_data['close']

        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = moving average of true range
        atr = true_range.rolling(window=self.atr_lookback).mean().iloc[-1]

        return atr

    def calculate_shares(
        self,
        portfolio_value: float,
        price: float,
        price_data: pd.DataFrame
    ) -> int:
        """
        Calculate position size based on ATR.

        Args:
            portfolio_value: Current portfolio value
            price: Current share price
            price_data: Historical OHLC data for ATR calculation

        Returns:
            Number of shares (integer, rounded down)
        """
        if portfolio_value <= 0 or price <= 0:
            return 0

        # Calculate ATR
        try:
            atr = self.calculate_atr(price_data)
        except (ValueError, KeyError) as e:
            # Fallback to fixed percentage if ATR calculation fails
            import warnings
            warnings.warn(
                f"ATR calculation failed: {e}. Using fallback fixed percentage sizing.",
                UserWarning
            )
            return int((portfolio_value * 0.10) / price)

        if atr <= 0:
            return 0

        # Risk per trade in dollars
        risk_dollars = portfolio_value * self.risk_pct

        # Stop distance = ATR × multiplier
        stop_distance = atr * self.atr_multiplier

        # Shares = risk / stop distance
        shares = int(risk_dollars / stop_distance)

        # Ensure position doesn't exceed portfolio
        max_shares = int((portfolio_value * 0.95) / price)
        shares = min(shares, max_shares)

        return shares


class KellyCriterionSizer:
    """
    Calculate optimal position size using Kelly Criterion.

    The Kelly Criterion calculates the mathematically optimal bet size
    to maximize long-term compound growth based on win rate and
    average win/loss ratio.

    Example:
        sizer = KellyCriterionSizer(
            win_rate=0.55,
            avg_win=500,
            avg_loss=300,
            kelly_fraction=0.5  # Half Kelly for safety
        )
        shares = sizer.calculate_shares(
            portfolio_value=100000,
            price=150.00
        )

    Args:
        win_rate: Probability of winning (0.0 to 1.0)
        avg_win: Average profit on winning trades
        avg_loss: Average loss on losing trades (positive number)
        kelly_fraction: Fraction of Kelly to use (0.25, 0.5, or 1.0)
                       Default: 0.5 (Half Kelly)

    Formula:
        Kelly % = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win
        Position Size = Kelly % × Kelly Fraction × Portfolio Value

    Warning:
        NEVER use Full Kelly (kelly_fraction=1.0) in real trading.
        Use Half Kelly (0.5) or Quarter Kelly (0.25) for safety.
    """

    def __init__(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.5
    ):
        if not 0.0 <= win_rate <= 1.0:
            raise ValueError(
                f"win_rate must be between 0 and 1, got {win_rate}"
            )

        if avg_win <= 0 or avg_loss <= 0:
            raise ValueError(
                "avg_win and avg_loss must be positive"
            )

        if not 0.0 < kelly_fraction <= 1.0:
            raise ValueError(
                f"kelly_fraction must be between 0 and 1, got {kelly_fraction}"
            )

        self.win_rate = win_rate
        self.loss_rate = 1.0 - win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction

        # Calculate Kelly %
        self.kelly_pct = self._calculate_kelly()

    def _calculate_kelly(self) -> float:
        """Calculate raw Kelly percentage."""
        if self.avg_win == 0:
            return 0.0

        # Kelly % = (p × W - q × L) / W
        # Where: p = win rate, q = loss rate, W = avg win, L = avg loss
        kelly = (
            (self.win_rate * self.avg_win) - (self.loss_rate * self.avg_loss)
        ) / self.avg_win

        # Apply Kelly fraction (Half Kelly, Quarter Kelly, etc.)
        kelly = kelly * self.kelly_fraction

        # Ensure non-negative and cap at 100%
        kelly = max(0.0, min(kelly, 1.0))

        return kelly

    def calculate_shares(
        self,
        portfolio_value: float,
        price: float
    ) -> int:
        """
        Calculate position size based on Kelly Criterion.

        Args:
            portfolio_value: Current portfolio value
            price: Current share price

        Returns:
            Number of shares (integer, rounded down)
        """
        if portfolio_value <= 0 or price <= 0:
            return 0

        # Position value = Kelly % × portfolio
        position_value = portfolio_value * self.kelly_pct

        # Convert to shares
        shares = int(position_value / price)

        return shares

    def get_kelly_info(self) -> dict:
        """Get detailed Kelly calculation info."""
        return {
            'win_rate': self.win_rate,
            'loss_rate': self.loss_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'win_loss_ratio': self.avg_win / self.avg_loss if self.avg_loss > 0 else 0,
            'raw_kelly_pct': self.kelly_pct / self.kelly_fraction if self.kelly_fraction > 0 else 0,
            'kelly_fraction': self.kelly_fraction,
            'final_kelly_pct': self.kelly_pct
        }


class RiskParitySizer:
    """
    Size positions using Risk Parity - equal risk contribution.

    Allocates capital so that each position contributes equally to
    portfolio risk. Positions are weighted inversely to volatility.

    Example:
        sizer = RiskParitySizer(lookback=60)

        positions = sizer.calculate_positions(
            portfolio_value=100000,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            prices={'AAPL': 150, 'MSFT': 350, 'GOOGL': 140},
            returns_data=returns_df  # Historical returns for volatility
        )

    Args:
        lookback: Number of periods for volatility calculation
                 Default: 60 (60 days ≈ 3 months)
        min_weight: Minimum weight per position (prevents tiny positions)
                   Default: 0.05 (5%)
        max_weight: Maximum weight per position (prevents concentration)
                   Default: 0.50 (50%)

    Formula:
        Weight[i] = (1 / Volatility[i]) / Σ(1 / Volatility[j])
        Position Size[i] = Portfolio Value × Weight[i]

    Best for:
        Multi-asset portfolios with different volatility profiles
    """

    def __init__(
        self,
        lookback: int = 60,
        min_weight: float = 0.05,
        max_weight: float = 0.50
    ):
        if lookback < 1:
            raise ValueError(
                f"lookback must be >= 1, got {lookback}"
            )

        if not 0.0 < min_weight < max_weight <= 1.0:
            raise ValueError(
                f"Must have 0 < min_weight < max_weight <= 1.0"
            )

        self.lookback = lookback
        self.min_weight = min_weight
        self.max_weight = max_weight

    def calculate_volatilities(
        self,
        returns_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate annualized volatility for each symbol.

        Args:
            returns_data: DataFrame with columns = symbols, values = daily returns

        Returns:
            Series with symbol -> volatility mapping
        """
        # Calculate standard deviation of returns
        vol = returns_data.tail(self.lookback).std()

        # Annualize (daily -> annual)
        vol_annual = vol * np.sqrt(252)

        return vol_annual

    def calculate_weights(
        self,
        volatilities: pd.Series
    ) -> pd.Series:
        """
        Calculate risk parity weights.

        Args:
            volatilities: Series with symbol -> volatility mapping

        Returns:
            Series with symbol -> weight mapping (sums to 1.0)
        """
        # Inverse volatility
        inv_vol = 1.0 / volatilities

        # Normalize to sum to 1.0
        weights = inv_vol / inv_vol.sum()

        # Apply min/max constraints
        weights = weights.clip(lower=self.min_weight, upper=self.max_weight)

        # Renormalize after clipping
        weights = weights / weights.sum()

        return weights

    def calculate_positions(
        self,
        portfolio_value: float,
        symbols: list,
        prices: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Dict[str, int]:
        """
        Calculate position sizes for all symbols.

        Args:
            portfolio_value: Current portfolio value
            symbols: List of symbols to trade
            prices: Dict mapping symbol -> current price
            returns_data: DataFrame with historical returns (columns = symbols)

        Returns:
            Dict mapping symbol -> number of shares
        """
        # Calculate volatilities
        volatilities = self.calculate_volatilities(returns_data)

        # Calculate weights
        weights = self.calculate_weights(volatilities)

        # Calculate position sizes
        positions = {}
        for symbol in symbols:
            if symbol not in weights or symbol not in prices:
                positions[symbol] = 0
                continue

            # Dollar allocation
            allocation = portfolio_value * weights[symbol]

            # Convert to shares
            shares = int(allocation / prices[symbol])

            positions[symbol] = shares

        return positions

    def get_allocation_info(
        self,
        positions: Dict[str, int],
        prices: Dict[str, float],
        volatilities: pd.Series
    ) -> pd.DataFrame:
        """
        Get detailed allocation information.

        Returns:
            DataFrame with columns: symbol, shares, value, weight, volatility, risk_contribution
        """
        info = []
        total_value = 0

        for symbol, shares in positions.items():
            if symbol not in prices:
                continue

            value = shares * prices[symbol]
            total_value += value

            info.append({
                'symbol': symbol,
                'shares': shares,
                'value': value,
                'volatility': volatilities.get(symbol, 0)
            })

        df = pd.DataFrame(info)

        if total_value > 0:
            # Calculate weights and risk contributions
            df['weight'] = df['value'] / total_value
            df['risk_contribution'] = df['value'] * df['volatility']
        else:
            df['weight'] = 0
            df['risk_contribution'] = 0

        return df
