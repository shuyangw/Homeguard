"""
Risk configuration for backtesting.

This module provides the RiskConfig class for configuring position sizing,
stop losses, and portfolio-level constraints.

Classes:
    RiskConfig: Configuration for risk management parameters
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class RiskConfig:
    """
    Configuration for risk management in backtesting.

    This class encapsulates all risk management parameters including:
    - Position sizing (what % of capital per trade)
    - Stop losses (when to exit losing trades)
    - Portfolio constraints (max positions, max single position size)

    Example:
        # Use preset profiles
        config = RiskConfig.moderate()  # 10% per trade, 2% stop loss

        # Custom configuration
        config = RiskConfig(
            position_size_pct=0.15,
            use_stop_loss=True,
            stop_loss_pct=0.025,
            max_positions=8
        )

        # Disable risk management (not recommended)
        config = RiskConfig.disabled()
    """

    # Position Sizing
    position_size_pct: float = 0.10
    """Percentage of portfolio to allocate per trade (0.0 to 1.0)"""

    position_sizing_method: Literal[
        'fixed_percentage',
        'fixed_dollar',
        'volatility_based',
        'kelly_criterion',
        'risk_parity'
    ] = 'fixed_percentage'
    """Method for calculating position size"""

    # Stop Loss Configuration
    use_stop_loss: bool = True
    """Whether to use stop losses"""

    stop_loss_pct: float = 0.02
    """Stop loss percentage (0.02 = 2% loss triggers exit)"""

    stop_loss_type: Literal[
        'percentage',
        'atr',
        'time',
        'profit_target'
    ] = 'percentage'
    """Type of stop loss to use"""

    # ATR-Based Stop Loss (if stop_loss_type='atr')
    atr_multiplier: float = 2.0
    """Number of ATRs for stop distance (2.0 = 2× ATR)"""

    atr_lookback: int = 14
    """Number of periods for ATR calculation"""

    # Time-Based Stop Loss (if stop_loss_type='time')
    max_holding_bars: Optional[int] = None
    """Maximum holding period in bars (None = no time limit)"""

    # Profit Target (if stop_loss_type='profit_target')
    take_profit_pct: Optional[float] = None
    """Take profit percentage (0.05 = 5% gain triggers exit)"""

    # Portfolio-Level Constraints
    max_positions: int = 10
    """Maximum number of concurrent positions"""

    max_single_position_pct: float = 0.25
    """Maximum size of any single position (0.25 = 25% max)"""

    max_portfolio_heat: float = 0.20
    """Maximum total capital at risk (0.20 = 20% max total risk)"""

    # Volatility-Based Sizing Parameters (if method='volatility_based')
    risk_per_trade_pct: float = 0.01
    """Percentage of portfolio to risk per trade (for ATR sizing)"""

    # Kelly Criterion Parameters (if method='kelly_criterion')
    kelly_win_rate: Optional[float] = None
    """Win rate for Kelly Criterion (0.0 to 1.0)"""

    kelly_avg_win: Optional[float] = None
    """Average win amount for Kelly Criterion"""

    kelly_avg_loss: Optional[float] = None
    """Average loss amount for Kelly Criterion"""

    kelly_fraction: float = 0.5
    """Kelly fraction to use (0.5 = Half Kelly, 1.0 = Full Kelly)"""

    # Risk Parity Parameters (if method='risk_parity')
    risk_parity_lookback: int = 60
    """Lookback period for volatility calculation (Risk Parity)"""

    # Multi-Symbol Portfolio Construction Parameters
    portfolio_sizing_method: Literal[
        'equal_weight',
        'risk_parity',
        'fixed_count',
        'ranked',
        'adaptive'
    ] = 'equal_weight'
    """Method for allocating capital across multiple symbols in portfolio mode"""

    rebalancing_frequency: Literal[
        'never',
        'monthly',
        'quarterly',
        'on_signal',
        'drift'
    ] = 'never'
    """When to rebalance portfolio (multi-symbol mode only)"""

    rebalancing_threshold_pct: float = 0.05
    """Drift threshold for rebalancing (0.05 = 5% drift triggers rebalance)"""

    # System Flags
    enabled: bool = True
    """Whether risk management is enabled (disable at your own risk!)"""

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate percentages
        if not 0.0 < self.position_size_pct <= 1.0:
            raise ValueError(
                f"position_size_pct must be between 0 and 1, got {self.position_size_pct}"
            )

        if not 0.0 < self.stop_loss_pct <= 1.0:
            raise ValueError(
                f"stop_loss_pct must be between 0 and 1, got {self.stop_loss_pct}"
            )

        if not 0.0 < self.max_single_position_pct <= 1.0:
            raise ValueError(
                f"max_single_position_pct must be between 0 and 1, got {self.max_single_position_pct}"
            )

        if not 0.0 < self.max_portfolio_heat <= 1.0:
            raise ValueError(
                f"max_portfolio_heat must be between 0 and 1, got {self.max_portfolio_heat}"
            )

        # Validate max positions
        if self.max_positions < 1:
            raise ValueError(
                f"max_positions must be >= 1, got {self.max_positions}"
            )

        # Validate ATR parameters
        if self.atr_multiplier <= 0:
            raise ValueError(
                f"atr_multiplier must be positive, got {self.atr_multiplier}"
            )

        if self.atr_lookback < 1:
            raise ValueError(
                f"atr_lookback must be >= 1, got {self.atr_lookback}"
            )

        # Validate Kelly parameters if using Kelly
        if self.position_sizing_method == 'kelly_criterion':
            if self.kelly_win_rate is None or self.kelly_avg_win is None or self.kelly_avg_loss is None:
                raise ValueError(
                    "kelly_win_rate, kelly_avg_win, and kelly_avg_loss are required "
                    "when using kelly_criterion position sizing method"
                )

            if not 0.0 <= self.kelly_win_rate <= 1.0:
                raise ValueError(
                    f"kelly_win_rate must be between 0 and 1, got {self.kelly_win_rate}"
                )

            if self.kelly_avg_win <= 0 or self.kelly_avg_loss <= 0:
                raise ValueError(
                    "kelly_avg_win and kelly_avg_loss must be positive"
                )

        # Validate Kelly fraction
        if not 0.0 < self.kelly_fraction <= 1.0:
            raise ValueError(
                f"kelly_fraction must be between 0 and 1, got {self.kelly_fraction}"
            )

        # Warning for disabled risk management
        if not self.enabled:
            import warnings
            warnings.warn(
                "Risk management is DISABLED. Using 99% capital per trade. "
                "This is unrealistic and will produce misleading backtest results. "
                "Enable risk management with RiskConfig.moderate() or RiskConfig.conservative().",
                UserWarning,
                stacklevel=2
            )

        # Warning for very aggressive position sizing
        if self.position_size_pct > 0.25 and self.enabled:
            import warnings
            warnings.warn(
                f"Position size of {self.position_size_pct*100:.1f}% is very aggressive. "
                "Consider using 10-20% (RiskConfig.moderate() or RiskConfig.aggressive()).",
                UserWarning,
                stacklevel=2
            )

    @classmethod
    def conservative(cls) -> 'RiskConfig':
        """
        Conservative risk profile.

        Settings:
        - 5% per trade
        - 1% stop loss
        - Max 15 positions
        - Max 20% single position

        Best for:
        - Risk-averse traders
        - Volatile markets
        - Learning/testing new strategies

        Returns:
            RiskConfig with conservative settings
        """
        return cls(
            position_size_pct=0.05,
            stop_loss_pct=0.01,
            max_positions=15,
            max_single_position_pct=0.20,
            max_portfolio_heat=0.15,
            use_stop_loss=True,
            stop_loss_type='percentage'
        )

    @classmethod
    def moderate(cls) -> 'RiskConfig':
        """
        Moderate risk profile (DEFAULT).

        Settings:
        - 10% per trade
        - 2% stop loss
        - Max 10 positions
        - Max 25% single position

        Best for:
        - Most traders
        - Balanced risk/return
        - Long-term systematic trading

        Returns:
            RiskConfig with moderate settings
        """
        return cls(
            position_size_pct=0.10,
            stop_loss_pct=0.02,
            max_positions=10,
            max_single_position_pct=0.25,
            max_portfolio_heat=0.20,
            use_stop_loss=True,
            stop_loss_type='percentage'
        )

    @classmethod
    def aggressive(cls) -> 'RiskConfig':
        """
        Aggressive risk profile.

        Settings:
        - 20% per trade
        - 3% stop loss
        - Max 5 positions
        - Max 35% single position

        Best for:
        - Experienced traders
        - High-conviction strategies
        - Shorter holding periods

        Warning:
            This profile has higher risk. Only use if you understand
            the increased volatility and drawdown potential.

        Returns:
            RiskConfig with aggressive settings
        """
        return cls(
            position_size_pct=0.20,
            stop_loss_pct=0.03,
            max_positions=5,
            max_single_position_pct=0.35,
            max_portfolio_heat=0.25,
            use_stop_loss=True,
            stop_loss_type='percentage'
        )

    @classmethod
    def disabled(cls) -> 'RiskConfig':
        """
        Disable risk management (NOT RECOMMENDED).

        This uses 99% capital per trade with no stop losses.
        Only use for:
        - Testing legacy code
        - Academic exercises
        - Understanding why risk management matters

        Warning:
            This will produce unrealistic backtest results.
            Never use this configuration for actual trading decisions.

        Returns:
            RiskConfig with risk management disabled
        """
        return cls(
            position_size_pct=0.99,
            use_stop_loss=False,
            max_positions=1,
            max_single_position_pct=1.0,
            max_portfolio_heat=1.0,
            enabled=False
        )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary of all configuration parameters
        """
        return {
            'position_size_pct': self.position_size_pct,
            'position_sizing_method': self.position_sizing_method,
            'use_stop_loss': self.use_stop_loss,
            'stop_loss_pct': self.stop_loss_pct,
            'stop_loss_type': self.stop_loss_type,
            'atr_multiplier': self.atr_multiplier,
            'atr_lookback': self.atr_lookback,
            'max_holding_bars': self.max_holding_bars,
            'take_profit_pct': self.take_profit_pct,
            'max_positions': self.max_positions,
            'max_single_position_pct': self.max_single_position_pct,
            'max_portfolio_heat': self.max_portfolio_heat,
            'risk_per_trade_pct': self.risk_per_trade_pct,
            'kelly_win_rate': self.kelly_win_rate,
            'kelly_avg_win': self.kelly_avg_win,
            'kelly_avg_loss': self.kelly_avg_loss,
            'kelly_fraction': self.kelly_fraction,
            'risk_parity_lookback': self.risk_parity_lookback,
            'enabled': self.enabled
        }

    def __repr__(self) -> str:
        """String representation of config."""
        if not self.enabled:
            return "RiskConfig(DISABLED - 99% per trade, no stops)"

        return (
            f"RiskConfig({self.position_size_pct*100:.1f}% per trade, "
            f"{self.stop_loss_pct*100:.1f}% stop loss, "
            f"max {self.max_positions} positions)"
        )
