"""
Hurst Exponent Mean Reversion Strategy.

A mean reversion strategy using Hurst exponent as the primary regime filter.
Unlike the ML-based approach, this uses a simple threshold on Hurst exponent
to identify mean-reverting conditions.

Entry Logic:
- Long: Z-score < -threshold AND Hurst < hurst_threshold (mean-reverting)
- Short: Z-score > +threshold AND Hurst < hurst_threshold (mean-reverting)

Exit Logic:
- Mean reversion exit: Z-score crosses toward zero
- Stop loss: ATR-based or fixed percentage
- Take profit: ATR-based or fixed percentage (~3:1 R:R)
- Time stop: Exit after max_hold_bars

Hurst Interpretation:
- H < 0.5: Anti-persistent (mean-reverting) - GOOD for this strategy
- H = 0.5: Random walk
- H > 0.5: Persistent (trending) - AVOID trades

This strategy is simpler and more interpretable than the ML-based approach,
making it easier to understand and debug.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

from src.backtesting.base.strategy import LongShortStrategy
from src.strategies.advanced.ml_crypto_mr_indicators import MLCryptoMRIndicators
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HurstMRPosition:
    """Represents an active position in the Hurst MR strategy."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    entry_bar: int
    stop_loss: float
    take_profit: float
    hurst_at_entry: float


@dataclass
class HurstMRSignal:
    """Signal state for Hurst Mean Reversion strategy."""
    timestamp: pd.Timestamp
    close: float
    zscore: float
    hurst: float
    atr: float
    is_mean_reverting: bool
    entry_long: bool
    entry_short: bool
    stop_long: float
    stop_short: float
    target_long: float
    target_short: float


class HurstMRStrategy(LongShortStrategy):
    """
    Hurst Exponent-filtered Mean Reversion Strategy.

    A swing trading strategy that uses the Hurst exponent to identify
    mean-reverting market conditions before taking Z-score based trades.

    The Hurst exponent provides a direct measure of mean reversion:
    - H < 0.5: Market is mean-reverting (favorable for strategy)
    - H = 0.5: Market follows random walk
    - H > 0.5: Market is trending (unfavorable for strategy)

    Parameters:
        hurst_window: Hurst exponent calculation window (default 100)
        hurst_threshold: H < threshold to allow trades (default 0.45)
        zscore_window: Z-score calculation window (default 20)
        zscore_entry_threshold: Entry Z-score threshold (default 2.0)
        zscore_exit_threshold: Mean reversion exit threshold (default 0.5)
        atr_period: ATR period (default 14)
        atr_stop_multiplier: Stop = ATR x mult (default 1.5)
        atr_target_multiplier: Target = ATR x mult (default 4.5)
        use_fixed_pct_exits: Use fixed % stops instead of ATR (default False)
        fixed_stop_pct: Fixed stop loss percentage (default 0.10)
        fixed_target_pct: Fixed take profit percentage (default 0.318)
        long_only: Long-only mode (default False)
        max_hold_bars: Maximum holding period (default 10)
    """

    def __init__(
        self,
        # Hurst parameters
        hurst_window: int = 100,
        hurst_threshold: float = 0.45,
        # Z-score parameters
        zscore_window: int = 20,
        zscore_entry_threshold: float = 2.0,
        zscore_exit_threshold: float = 0.5,
        # ATR / Risk parameters
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        atr_target_multiplier: float = 4.5,
        # Fixed percentage exits (alternative to ATR-based)
        use_fixed_pct_exits: bool = False,
        fixed_stop_pct: float = 0.10,
        fixed_target_pct: float = 0.318,
        # Trade management
        long_only: bool = False,
        max_hold_bars: int = 10,
        **kwargs
    ):
        # Store parameters as instance attributes
        self.hurst_window = hurst_window
        self.hurst_threshold = hurst_threshold
        self.zscore_window = zscore_window
        self.zscore_entry_threshold = zscore_entry_threshold
        self.zscore_exit_threshold = zscore_exit_threshold
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_target_multiplier = atr_target_multiplier
        self.use_fixed_pct_exits = use_fixed_pct_exits
        self.fixed_stop_pct = fixed_stop_pct
        self.fixed_target_pct = fixed_target_pct
        self.long_only = long_only
        self.max_hold_bars = max_hold_bars

        # Position tracking
        self._current_position: Optional[HurstMRPosition] = None

        # Call parent init
        super().__init__(
            hurst_window=hurst_window,
            hurst_threshold=hurst_threshold,
            zscore_window=zscore_window,
            zscore_entry_threshold=zscore_entry_threshold,
            zscore_exit_threshold=zscore_exit_threshold,
            atr_period=atr_period,
            atr_stop_multiplier=atr_stop_multiplier,
            atr_target_multiplier=atr_target_multiplier,
            use_fixed_pct_exits=use_fixed_pct_exits,
            fixed_stop_pct=fixed_stop_pct,
            fixed_target_pct=fixed_target_pct,
            long_only=long_only,
            max_hold_bars=max_hold_bars,
            **kwargs
        )

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.hurst_window <= 20:
            raise ValueError("hurst_window must be > 20 for reliable estimates")
        if self.hurst_threshold <= 0 or self.hurst_threshold >= 1:
            raise ValueError("hurst_threshold must be between 0 and 1")
        if self.zscore_window <= 0:
            raise ValueError("zscore_window must be positive")
        if self.zscore_entry_threshold <= 0:
            raise ValueError("zscore_entry_threshold must be positive")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.atr_stop_multiplier <= 0:
            raise ValueError("atr_stop_multiplier must be positive")
        if self.atr_target_multiplier <= 0:
            raise ValueError("atr_target_multiplier must be positive")
        if self.max_hold_bars <= 0:
            raise ValueError("max_hold_bars must be positive")

    def reset_state(self) -> None:
        """Reset strategy state for new backtest run."""
        self._current_position = None

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators needed for signal generation."""
        close = data['close']

        indicators = pd.DataFrame(index=data.index)
        indicators['close'] = close

        # Z-score
        indicators['zscore'] = MLCryptoMRIndicators.calculate_zscore(
            close, self.zscore_window
        )

        # ATR
        indicators['atr'] = MLCryptoMRIndicators.calculate_atr(
            data, self.atr_period
        )

        # Hurst exponent
        indicators['hurst'] = MLCryptoMRIndicators.calculate_hurst(
            close, self.hurst_window
        )

        return indicators

    def generate_long_short_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate entry and exit signals for long and short positions.

        This method implements the core strategy logic:
        1. Calculate Z-score and Hurst exponent
        2. Generate entry signals when H < threshold and Z-score extreme
        3. Track positions for ATR-based exits

        Args:
            data: OHLCV DataFrame with DatetimeIndex

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        n = len(data)

        # Initialize output arrays
        long_entries = np.zeros(n, dtype=bool)
        long_exits = np.zeros(n, dtype=bool)
        short_entries = np.zeros(n, dtype=bool)
        short_exits = np.zeros(n, dtype=bool)

        # Check minimum data
        min_required = max(self.hurst_window, self.zscore_window) + 10
        if n < min_required:
            logger.warning(f"Insufficient data for strategy: {n} bars < {min_required}")
            return (
                pd.Series(long_entries, index=data.index),
                pd.Series(long_exits, index=data.index),
                pd.Series(short_entries, index=data.index),
                pd.Series(short_exits, index=data.index)
            )

        # Calculate indicators
        indicators = self._calculate_indicators(data)

        zscore = indicators['zscore']
        hurst = indicators['hurst']
        atr = indicators['atr']
        close = indicators['close']

        # Position tracking
        position_active = False
        position_direction = None
        position_entry_bar = 0
        position_stop = 0.0
        position_target = 0.0

        # Bar-by-bar processing
        for i in range(n):
            current_close = close.iloc[i]
            current_zscore = zscore.iloc[i]
            current_hurst = hurst.iloc[i]
            current_atr = atr.iloc[i]

            # === Check for exit if in position ===
            if position_active:
                bars_held = i - position_entry_bar
                time_exit = bars_held >= self.max_hold_bars

                if position_direction == 'long':
                    if not pd.isna(current_close) and not pd.isna(current_zscore):
                        hit_stop = current_close <= position_stop
                        hit_target = current_close >= position_target
                        mean_reversion = current_zscore >= -self.zscore_exit_threshold
                    else:
                        hit_stop = hit_target = mean_reversion = False

                    if hit_stop or hit_target or mean_reversion or time_exit:
                        long_exits[i] = True
                        position_active = False
                        position_direction = None

                elif position_direction == 'short':
                    if not pd.isna(current_close) and not pd.isna(current_zscore):
                        hit_stop = current_close >= position_stop
                        hit_target = current_close <= position_target
                        mean_reversion = current_zscore <= self.zscore_exit_threshold
                    else:
                        hit_stop = hit_target = mean_reversion = False

                    if hit_stop or hit_target or mean_reversion or time_exit:
                        short_exits[i] = True
                        position_active = False
                        position_direction = None

            # Skip entry if data is invalid
            if pd.isna(current_zscore) or pd.isna(current_atr) or pd.isna(current_hurst):
                continue

            # === Check for entry if flat ===
            if not position_active:
                # Hurst filter: only trade in mean-reverting conditions
                is_mean_reverting = current_hurst < self.hurst_threshold

                if not is_mean_reverting:
                    continue  # Skip if not mean-reverting

                # Long entry: oversold condition
                long_condition = current_zscore <= -self.zscore_entry_threshold

                # Short entry: overbought condition
                short_condition = current_zscore >= self.zscore_entry_threshold

                # Generate entry signals
                if long_condition:
                    long_entries[i] = True
                    position_active = True
                    position_direction = 'long'
                    position_entry_bar = i

                    if self.use_fixed_pct_exits:
                        position_stop = current_close * (1.0 - self.fixed_stop_pct)
                        position_target = current_close * (1.0 + self.fixed_target_pct)
                    else:
                        position_stop = current_close - (current_atr * self.atr_stop_multiplier)
                        position_target = current_close + (current_atr * self.atr_target_multiplier)

                elif short_condition and not self.long_only:
                    short_entries[i] = True
                    position_active = True
                    position_direction = 'short'
                    position_entry_bar = i

                    if self.use_fixed_pct_exits:
                        position_stop = current_close * (1.0 + self.fixed_stop_pct)
                        position_target = current_close * (1.0 - self.fixed_target_pct)
                    else:
                        position_stop = current_close + (current_atr * self.atr_stop_multiplier)
                        position_target = current_close - (current_atr * self.atr_target_multiplier)

        # Ensure time-based exits
        self._ensure_time_exits(long_entries, long_exits, n)
        self._ensure_time_exits(short_entries, short_exits, n)

        return (
            pd.Series(long_entries, index=data.index),
            pd.Series(long_exits, index=data.index),
            pd.Series(short_entries, index=data.index),
            pd.Series(short_exits, index=data.index)
        )

    def _ensure_time_exits(
        self,
        entries: np.ndarray,
        exits: np.ndarray,
        n: int
    ) -> None:
        """Ensure every entry has a corresponding exit within max_hold_bars."""
        entry_indices = np.where(entries)[0]

        for entry_idx in entry_indices:
            max_exit_idx = min(entry_idx + self.max_hold_bars, n - 1)
            exit_range = exits[entry_idx + 1:max_exit_idx + 1]

            if len(exit_range) > 0 and exit_range.any():
                continue

            exits[max_exit_idx] = True

    def get_current_signal(self, data: pd.DataFrame) -> HurstMRSignal:
        """
        Get current signal state for live trading.

        Args:
            data: Recent OHLCV data (enough for indicators)

        Returns:
            HurstMRSignal with current state
        """
        indicators = self._calculate_indicators(data)

        latest_idx = indicators.index[-1]
        latest = indicators.iloc[-1]

        # Determine regime
        is_mean_reverting = latest['hurst'] < self.hurst_threshold

        # Calculate stop/target levels
        if self.use_fixed_pct_exits:
            stop_dist = latest['close'] * self.fixed_stop_pct
            target_dist = latest['close'] * self.fixed_target_pct
        else:
            stop_dist = latest['atr'] * self.atr_stop_multiplier
            target_dist = latest['atr'] * self.atr_target_multiplier

        # Entry conditions
        zscore = latest['zscore']

        long_cond = (
            zscore <= -self.zscore_entry_threshold and
            is_mean_reverting
        )

        short_cond = (
            zscore >= self.zscore_entry_threshold and
            is_mean_reverting and
            not self.long_only
        )

        return HurstMRSignal(
            timestamp=latest_idx,
            close=latest['close'],
            zscore=zscore,
            hurst=latest['hurst'],
            atr=latest['atr'],
            is_mean_reverting=is_mean_reverting,
            entry_long=long_cond,
            entry_short=short_cond,
            stop_long=latest['close'] - stop_dist,
            stop_short=latest['close'] + stop_dist,
            target_long=latest['close'] + target_dist,
            target_short=latest['close'] - target_dist
        )

    def get_hurst_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get Hurst exponent statistics for analysis.

        Args:
            data: OHLCV DataFrame

        Returns:
            Dictionary with Hurst statistics
        """
        indicators = self._calculate_indicators(data)
        hurst = indicators['hurst'].dropna()

        is_mean_reverting = hurst < self.hurst_threshold
        is_random = (hurst >= self.hurst_threshold) & (hurst <= 0.55)
        is_trending = hurst > 0.55

        return {
            'mean_hurst': hurst.mean(),
            'std_hurst': hurst.std(),
            'min_hurst': hurst.min(),
            'max_hurst': hurst.max(),
            'mean_reverting_pct': is_mean_reverting.mean() * 100,
            'random_walk_pct': is_random.mean() * 100,
            'trending_pct': is_trending.mean() * 100,
            'regime_changes': (is_mean_reverting != is_mean_reverting.shift(1)).sum(),
        }
