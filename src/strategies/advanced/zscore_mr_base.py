"""
Z-Score Mean Reversion Base Strategy.

Abstract base class that captures the shared bar-by-bar signal generation loop
used by both HurstMRStrategy and MLCryptoMRStrategy. Subclasses override a
small number of hooks to supply their own regime filter and indicator set while
reusing the position tracking, exit logic, stop/target calculation, and
time-exit safety net.

Hook methods that subclasses MUST implement:
    _compute_indicators(data) -> pd.DataFrame
        Return a DataFrame that at minimum contains 'close', 'zscore', and 'atr'.
    _get_regime_filter_result(i, indicators) -> bool
        Return True when the regime allows new entries at bar i.
    _get_min_required_bars() -> int
        Return the minimum number of bars needed before signal generation.

Hook methods that subclasses MAY override:
    _check_entry_confirmation(i, indicators) -> Tuple[bool, bool]
        Additional confirmation beyond z-score for (long_ok, short_ok).
        Default returns (True, True).
    _check_indicator_nan_for_entry(i, indicators) -> bool
        Return True if entry should be skipped due to NaN indicators.
        Default checks zscore and atr only.
    _pre_loop_setup(data, indicators) -> None
        Called after indicator computation but before the bar-by-bar loop.
        Useful for ML training or precomputing regime predictions.
"""

from abc import abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np

from src.backtesting.base.strategy import LongShortStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ZScoreMeanReversionBase(LongShortStrategy):
    """
    Abstract base for z-score mean reversion strategies.

    Provides the shared bar-by-bar loop for position tracking, exit logic
    (stop loss, take profit, mean reversion, time stop), entry framework
    (z-score threshold + regime filter + optional confirmation), and the
    _ensure_time_exits safety net.

    Shared Parameters:
        zscore_window: Z-score calculation window
        zscore_entry_threshold: Entry Z-score threshold
        zscore_exit_threshold: Mean reversion exit threshold
        atr_period: ATR period
        atr_stop_multiplier: Stop = ATR x multiplier
        atr_target_multiplier: Target = ATR x multiplier
        use_fixed_pct_exits: Use fixed % stops instead of ATR
        fixed_stop_pct: Fixed stop loss percentage
        fixed_target_pct: Fixed take profit percentage
        long_only: Long-only mode
        max_hold_bars: Maximum holding period
    """

    def __init__(
        self,
        # Z-score parameters
        zscore_window: int = 20,
        zscore_entry_threshold: float = 2.0,
        zscore_exit_threshold: float = 0.5,
        # ATR / Risk parameters
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        atr_target_multiplier: float = 4.5,
        # Fixed percentage exits
        use_fixed_pct_exits: bool = False,
        fixed_stop_pct: float = 0.10,
        fixed_target_pct: float = 0.318,
        # Trade management
        long_only: bool = False,
        max_hold_bars: int = 10,
        **kwargs,
    ):
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

        super().__init__(
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
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Abstract hooks -- subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators needed for signal generation.

        Must return a DataFrame with at least 'close', 'zscore', and 'atr'
        columns, indexed identically to *data*.
        """

    @abstractmethod
    def _get_regime_filter_result(
        self, i: int, indicators: pd.DataFrame
    ) -> bool:
        """
        Return True when the regime at bar *i* allows new entries.

        Args:
            i: Current bar index (integer position).
            indicators: The full indicators DataFrame returned by
                _compute_indicators().

        Returns:
            True if conditions are favorable for mean-reversion entries.
        """

    @abstractmethod
    def _get_min_required_bars(self) -> int:
        """Return the minimum number of bars needed before the strategy can
        generate any signals."""

    # ------------------------------------------------------------------
    # Optional hooks -- subclasses MAY override
    # ------------------------------------------------------------------

    def _check_entry_confirmation(
        self, i: int, indicators: pd.DataFrame
    ) -> Tuple[bool, bool]:
        """
        Additional entry confirmation beyond z-score threshold.

        Returns:
            (long_ok, short_ok) -- True means the confirmation passes.
            Default implementation returns (True, True).
        """
        return True, True

    def _check_indicator_nan_for_entry(
        self, i: int, indicators: pd.DataFrame
    ) -> bool:
        """
        Return True if entry should be skipped at bar *i* due to NaN
        indicator values. Default checks zscore and atr.
        """
        return (
            pd.isna(indicators['zscore'].iloc[i])
            or pd.isna(indicators['atr'].iloc[i])
        )

    def _pre_loop_setup(
        self, data: pd.DataFrame, indicators: pd.DataFrame
    ) -> None:
        """
        Called after indicator computation, before the bar-by-bar loop.
        Override for ML model training, regime pre-computation, etc.
        """

    # ------------------------------------------------------------------
    # Core signal generation loop (shared)
    # ------------------------------------------------------------------

    def generate_long_short_signals(
        self,
        data: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate entry and exit signals for long and short positions.

        Implements the shared bar-by-bar loop:
        1. Compute indicators via _compute_indicators()
        2. Run _pre_loop_setup() (e.g. ML training)
        3. For each bar: check exits then entries
        4. Apply _ensure_time_exits() safety net

        Args:
            data: OHLCV DataFrame with DatetimeIndex.

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
        min_required = self._get_min_required_bars()
        if n < min_required:
            logger.warning(
                f"Insufficient data for strategy: {n} bars < {min_required}"
            )
            return (
                pd.Series(long_entries, index=data.index),
                pd.Series(long_exits, index=data.index),
                pd.Series(short_entries, index=data.index),
                pd.Series(short_exits, index=data.index),
            )

        # Compute indicators
        indicators = self._compute_indicators(data)

        # Pre-loop hook (e.g. ML training)
        self._pre_loop_setup(data, indicators)

        zscore = indicators['zscore']
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
            current_atr = atr.iloc[i]

            # === Check for exit if in position ===
            if position_active:
                bars_held = i - position_entry_bar
                time_exit = bars_held >= self.max_hold_bars

                if position_direction == 'long':
                    if not pd.isna(current_close) and not pd.isna(current_zscore):
                        hit_stop = current_close <= position_stop
                        hit_target = current_close >= position_target
                        mean_reversion = (
                            current_zscore >= -self.zscore_exit_threshold
                        )
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
                        mean_reversion = (
                            current_zscore <= self.zscore_exit_threshold
                        )
                    else:
                        hit_stop = hit_target = mean_reversion = False

                    if hit_stop or hit_target or mean_reversion or time_exit:
                        short_exits[i] = True
                        position_active = False
                        position_direction = None

            # Skip entry if indicators have NaN values
            if self._check_indicator_nan_for_entry(i, indicators):
                continue

            # === Check for entry if flat ===
            if not position_active:
                # Regime filter
                if not self._get_regime_filter_result(i, indicators):
                    continue

                # Additional entry confirmation (e.g. RSI)
                long_ok, short_ok = self._check_entry_confirmation(
                    i, indicators
                )

                # Long entry: oversold condition
                long_condition = (
                    current_zscore <= -self.zscore_entry_threshold and long_ok
                )

                # Short entry: overbought condition
                short_condition = (
                    current_zscore >= self.zscore_entry_threshold and short_ok
                )

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
                        position_stop = current_close - (
                            current_atr * self.atr_stop_multiplier
                        )
                        position_target = current_close + (
                            current_atr * self.atr_target_multiplier
                        )

                elif short_condition and not self.long_only:
                    short_entries[i] = True
                    position_active = True
                    position_direction = 'short'
                    position_entry_bar = i

                    if self.use_fixed_pct_exits:
                        position_stop = current_close * (1.0 + self.fixed_stop_pct)
                        position_target = current_close * (1.0 - self.fixed_target_pct)
                    else:
                        position_stop = current_close + (
                            current_atr * self.atr_stop_multiplier
                        )
                        position_target = current_close - (
                            current_atr * self.atr_target_multiplier
                        )

        # Ensure time-based exits
        self._ensure_time_exits(long_entries, long_exits, n)
        self._ensure_time_exits(short_entries, short_exits, n)

        return (
            pd.Series(long_entries, index=data.index),
            pd.Series(long_exits, index=data.index),
            pd.Series(short_entries, index=data.index),
            pd.Series(short_exits, index=data.index),
        )

    # ------------------------------------------------------------------
    # Shared utility
    # ------------------------------------------------------------------

    def _ensure_time_exits(
        self,
        entries: np.ndarray,
        exits: np.ndarray,
        n: int,
    ) -> None:
        """Ensure every entry has a corresponding exit within max_hold_bars."""
        entry_indices = np.where(entries)[0]

        for entry_idx in entry_indices:
            max_exit_idx = min(entry_idx + self.max_hold_bars, n - 1)
            exit_range = exits[entry_idx + 1 : max_exit_idx + 1]

            if len(exit_range) > 0 and exit_range.any():
                continue

            exits[max_exit_idx] = True
