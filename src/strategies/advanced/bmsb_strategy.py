"""
Bull Market Support Band (BMSB) Trading Strategy.

A macro trend-following strategy for crypto assets that uses the Bull Market
Support Band (20-week SMA and 21-week EMA) to identify long-term trend direction.

Entry Logic:
- Long: Weekly close crosses above the band (close > both SMA and EMA)
- Short: Weekly close crosses below the band (close < both SMA and EMA)

Exit Logic:
- Long Exit: Weekly close crosses below the band OR trailing stop
- Short Exit: Weekly close crosses above the band OR trailing stop

This is a position-trading strategy designed for weekly timeframe decisions,
making it suitable for crypto assets where the BMSB has historically identified
major bull/bear market transitions.

Original indicator by zkdev on TradingView.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np

from src.backtesting.base.strategy import LongShortStrategy
from src.strategies.advanced.bmsb_indicators import BMSBIndicators, BMSBSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BMSBPosition:
    """Represents an active BMSB position."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    entry_sma: float
    entry_ema: float
    # Trailing stop state
    trailing_stop: Optional[float] = None
    max_favorable_price: float = 0.0


@dataclass
class _BMSBState:
    """Mutable state for per-bar signal generation loop."""
    position: Optional[str] = None  # 'long', 'short', or None
    entry_price: float = 0.0
    trailing_stop: Optional[float] = None
    max_favorable: float = 0.0


class BMSBStrategy(LongShortStrategy):
    """
    Bull Market Support Band (BMSB) Trading Strategy.

    A macro trend-following strategy that uses weekly 20 SMA and 21 EMA
    to identify bull/bear market regimes in crypto assets.

    The band acts as dynamic support in bull markets and resistance in bear markets.
    Signals are generated when price crosses above or below the band.

    Parameters:
        sma_period: SMA period (default 20)
        ema_period: EMA period (default 21)
        timeframe: Band calculation timeframe - 'weekly', 'daily', or 'raw' (default 'weekly')
        long_only: Only take long trades (default False)
        use_trailing_stop: Enable trailing stop (default True)
        trailing_stop_pct: Trailing stop as % below high (default 0.15 = 15%)
        signal_on_close: Only signal on timeframe close (default True)
        require_both_above: Require close > both MAs for long (default True)
        require_both_below: Require close < both MAs for short (default True)

        # Filter parameters
        use_htf_filter: Use weekly trend filter for daily entries (default False)
        use_rsi_filter: Use RSI momentum confirmation (default False)
        rsi_period: RSI calculation period (default 14)
        rsi_long_threshold: RSI must be above this for longs (default 50)
        rsi_short_threshold: RSI must be below this for shorts (default 50)
        min_band_width_pct: Minimum band width % to trade (default 0.0 = disabled)
        use_atr_stop: Use ATR-based trailing stop (default False)
        atr_period: ATR calculation period (default 14)
        atr_stop_multiplier: ATR multiplier for stop distance (default 2.0)
    """

    def __init__(
        self,
        sma_period: int = 20,
        ema_period: int = 21,
        timeframe: str = 'weekly',
        long_only: bool = False,
        use_trailing_stop: bool = True,
        trailing_stop_pct: float = 0.15,
        signal_on_close: bool = True,
        require_both_above: bool = True,
        require_both_below: bool = True,
        # Filter parameters
        use_htf_filter: bool = False,
        use_rsi_filter: bool = False,
        rsi_period: int = 14,
        rsi_long_threshold: float = 50.0,
        rsi_short_threshold: float = 50.0,
        min_band_width_pct: float = 0.0,
        use_atr_stop: bool = False,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        # Deprecated parameter for backwards compatibility
        signal_on_weekly_close: bool = None,
        **kwargs
    ):
        # Set attributes BEFORE calling super().__init__()
        # because parent class calls validate_parameters()
        self.sma_period = sma_period
        self.ema_period = ema_period
        self.timeframe = timeframe
        self.long_only = long_only
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        # Handle deprecated parameter
        if signal_on_weekly_close is not None:
            self.signal_on_close = signal_on_weekly_close
        else:
            self.signal_on_close = signal_on_close
        self.require_both_above = require_both_above
        self.require_both_below = require_both_below

        # Filter parameters
        self.use_htf_filter = use_htf_filter
        self.use_rsi_filter = use_rsi_filter
        self.rsi_period = rsi_period
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold
        self.min_band_width_pct = min_band_width_pct
        self.use_atr_stop = use_atr_stop
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier

        # State tracking
        self._current_position: Optional[BMSBPosition] = None
        self._band_data: Optional[pd.DataFrame] = None

        super().__init__(
            sma_period=sma_period,
            ema_period=ema_period,
            timeframe=timeframe,
            long_only=long_only,
            use_trailing_stop=use_trailing_stop,
            trailing_stop_pct=trailing_stop_pct,
            signal_on_close=signal_on_close,
            require_both_above=require_both_above,
            require_both_below=require_both_below,
            use_htf_filter=use_htf_filter,
            use_rsi_filter=use_rsi_filter,
            rsi_period=rsi_period,
            rsi_long_threshold=rsi_long_threshold,
            rsi_short_threshold=rsi_short_threshold,
            min_band_width_pct=min_band_width_pct,
            use_atr_stop=use_atr_stop,
            atr_period=atr_period,
            atr_stop_multiplier=atr_stop_multiplier,
            **kwargs
        )

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.sma_period < 1:
            raise ValueError(f"sma_period must be >= 1, got {self.sma_period}")

        if self.ema_period < 1:
            raise ValueError(f"ema_period must be >= 1, got {self.ema_period}")

        if self.trailing_stop_pct < 0 or self.trailing_stop_pct > 1:
            raise ValueError(
                f"trailing_stop_pct must be between 0 and 1, got {self.trailing_stop_pct}"
            )

        if self.timeframe not in ['weekly', 'daily', 'raw']:
            raise ValueError(
                f"timeframe must be 'weekly', 'daily', or 'raw', got {self.timeframe}"
            )

    def reset_state(self) -> None:
        """Reset strategy state for new backtest run."""
        self._current_position = None
        self._band_data = None

    def generate_long_short_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Generate long and short entry/exit signals based on BMSB.

        Pre-computes band and filters, then iterates bars checking
        exit and entry conditions via focused helper methods.

        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        if data.empty:
            return (
                pd.Series(dtype=bool),
                pd.Series(dtype=bool),
                pd.Series(dtype=bool),
                pd.Series(dtype=bool)
            )

        n = len(data)
        long_entries_arr = np.zeros(n, dtype=bool)
        long_exits_arr = np.zeros(n, dtype=bool)
        short_entries_arr = np.zeros(n, dtype=bool)
        short_exits_arr = np.zeros(n, dtype=bool)

        ind = self._compute_band_and_filters(data)
        if ind is None:
            return (
                pd.Series(long_entries_arr, index=data.index),
                pd.Series(long_exits_arr, index=data.index),
                pd.Series(short_entries_arr, index=data.index),
                pd.Series(short_exits_arr, index=data.index)
            )

        state = _BMSBState()

        if self.timeframe == 'weekly':
            min_bars = max(self.sma_period, self.ema_period) * 7
        else:
            min_bars = max(self.sma_period, self.ema_period)

        for i in range(min_bars, n):
            if np.isnan(ind['sma'][i]) or np.isnan(ind['ema'][i]):
                continue

            is_above, is_below = self._band_position(i, ind)
            timestamp = data.index[i]
            is_tf_close = timestamp in ind['timeframe_closes']
            can_signal = (not self.signal_on_close) or is_tf_close

            # Exit logic (check before entries)
            if state.position == 'long':
                exited, reason = self._check_long_exit(
                    i, ind, state, is_below, can_signal
                )
                if exited:
                    long_exits_arr[i] = True
                    if not self.long_only and reason == 'band_cross':
                        short_entries_arr[i] = True
                        state.position = 'short'
                        state.entry_price = ind['close'][i]
                        state.max_favorable = ind['low'][i]

            elif state.position == 'short':
                exited, reason = self._check_short_exit(
                    i, ind, state, is_above, can_signal
                )
                if exited:
                    short_exits_arr[i] = True
                    if reason == 'band_cross':
                        long_entries_arr[i] = True
                        state.position = 'long'
                        state.entry_price = ind['close'][i]
                        state.max_favorable = ind['high'][i]

            # Entry logic (only if no position)
            if state.position is None and can_signal:
                entered = self._check_long_entry(i, ind, state, is_above)
                if entered:
                    long_entries_arr[i] = True
                else:
                    entered = self._check_short_entry(i, ind, state, is_below)
                    if entered:
                        short_entries_arr[i] = True

        return (
            pd.Series(long_entries_arr, index=data.index),
            pd.Series(long_exits_arr, index=data.index),
            pd.Series(short_entries_arr, index=data.index),
            pd.Series(short_exits_arr, index=data.index)
        )

    def _compute_band_and_filters(self, data: pd.DataFrame) -> Optional[Dict]:
        """Pre-compute BMSB band values and optional filter arrays.

        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex.

        Returns:
            Dict of pre-computed arrays, or None if insufficient data.
        """
        band_df = BMSBIndicators.calculate_band_on_original_timeframe(
            data, self.sma_period, self.ema_period, timeframe=self.timeframe
        )

        if band_df.empty:
            logger.warning("Insufficient data for BMSB calculation")
            return None

        if self.timeframe == 'weekly':
            tf_df = BMSBIndicators.resample_to_weekly(data)
        elif self.timeframe == 'daily':
            tf_df = BMSBIndicators.resample_to_daily(data)
        else:
            tf_df = data

        rsi = None
        if self.use_rsi_filter:
            rsi = BMSBIndicators.calculate_rsi(data['close'], self.rsi_period).values

        htf_trend = None
        if self.use_htf_filter and self.timeframe == 'daily':
            htf_trend = BMSBIndicators.get_higher_timeframe_trend(
                data, self.sma_period, self.ema_period
            ).values

        atr = None
        if self.use_atr_stop:
            atr = BMSBIndicators.calculate_atr(data, self.atr_period).values

        close = data['close'].values
        band_upper = band_df['band_upper'].reindex(data.index, method='ffill').values
        band_lower = band_df['band_lower'].reindex(data.index, method='ffill').values

        band_width = None
        if self.min_band_width_pct > 0:
            band_width = (band_upper - band_lower) / close

        return {
            'close': close,
            'high': data['high'].values,
            'low': data['low'].values,
            'sma': band_df['sma'].reindex(data.index, method='ffill').values,
            'ema': band_df['ema'].reindex(data.index, method='ffill').values,
            'band_upper': band_upper,
            'band_lower': band_lower,
            'timeframe_closes': set(tf_df.index),
            'rsi': rsi,
            'htf_trend': htf_trend,
            'atr': atr,
            'band_width': band_width,
        }

    def _band_position(self, i: int, ind: Dict) -> Tuple[bool, bool]:
        """Determine if close is above or below the band at bar i.

        Args:
            i: Bar index.
            ind: Pre-computed indicator arrays.

        Returns:
            Tuple of (is_above_band, is_below_band).
        """
        c = ind['close'][i]
        if self.require_both_above:
            is_above = c > ind['band_upper'][i]
        else:
            is_above = c > min(ind['sma'][i], ind['ema'][i])

        if self.require_both_below:
            is_below = c < ind['band_lower'][i]
        else:
            is_below = c < max(ind['sma'][i], ind['ema'][i])

        return is_above, is_below

    def _check_long_exit(
        self, i: int, ind: Dict, state: '_BMSBState',
        is_below_band: bool, can_signal: bool,
    ) -> Tuple[bool, str]:
        """Check if a long position should be exited.

        Updates trailing stop, then checks trailing stop hit
        and band cross conditions.

        Args:
            i: Bar index.
            ind: Pre-computed indicator arrays.
            state: Mutable position state (modified in-place on exit).
            is_below_band: Whether close is below the band.
            can_signal: Whether this bar is eligible for signals.

        Returns:
            Tuple of (exited, exit_reason).
        """
        if self.use_trailing_stop:
            state.max_favorable = max(state.max_favorable, ind['high'][i])
            atr = ind['atr']
            if self.use_atr_stop and atr is not None and not np.isnan(atr[i]):
                state.trailing_stop = state.max_favorable - (atr[i] * self.atr_stop_multiplier)
            else:
                state.trailing_stop = state.max_favorable * (1 - self.trailing_stop_pct)

        if state.trailing_stop and ind['low'][i] <= state.trailing_stop:
            state.position = None
            state.trailing_stop = None
            state.max_favorable = 0.0
            return True, 'trailing_stop'

        if is_below_band and can_signal:
            state.position = None
            state.trailing_stop = None
            state.max_favorable = 0.0
            return True, 'band_cross'

        return False, ''

    def _check_short_exit(
        self, i: int, ind: Dict, state: '_BMSBState',
        is_above_band: bool, can_signal: bool,
    ) -> Tuple[bool, str]:
        """Check if a short position should be exited.

        Updates trailing stop, then checks trailing stop hit
        and band cross conditions.

        Args:
            i: Bar index.
            ind: Pre-computed indicator arrays.
            state: Mutable position state (modified in-place on exit).
            is_above_band: Whether close is above the band.
            can_signal: Whether this bar is eligible for signals.

        Returns:
            Tuple of (exited, exit_reason).
        """
        if self.use_trailing_stop:
            if state.max_favorable > 0:
                state.max_favorable = min(state.max_favorable, ind['low'][i])
            else:
                state.max_favorable = ind['low'][i]
            atr = ind['atr']
            if self.use_atr_stop and atr is not None and not np.isnan(atr[i]):
                state.trailing_stop = state.max_favorable + (atr[i] * self.atr_stop_multiplier)
            else:
                state.trailing_stop = state.max_favorable * (1 + self.trailing_stop_pct)

        if state.trailing_stop and ind['high'][i] >= state.trailing_stop:
            state.position = None
            state.trailing_stop = None
            state.max_favorable = 0.0
            return True, 'trailing_stop'

        if is_above_band and can_signal:
            state.position = None
            state.trailing_stop = None
            state.max_favorable = 0.0
            return True, 'band_cross'

        return False, ''

    def _check_long_entry(
        self, i: int, ind: Dict, state: '_BMSBState', is_above_band: bool,
    ) -> bool:
        """Check if a long entry should be taken.

        Validates band width, RSI, HTF, and cross-detection filters.

        Args:
            i: Bar index.
            ind: Pre-computed indicator arrays.
            state: Mutable position state (modified in-place on entry).
            is_above_band: Whether close is above the band.

        Returns:
            True if entry was taken.
        """
        if not is_above_band:
            return False

        bw = ind['band_width']
        if bw is not None and bw[i] < self.min_band_width_pct:
            return False

        rsi = ind['rsi']
        if rsi is not None and not np.isnan(rsi[i]):
            if rsi[i] <= self.rsi_long_threshold:
                return False

        htf = ind['htf_trend']
        if htf is not None and htf[i] != 'bullish':
            return False

        if i > 0:
            prev_close = ind['close'][i - 1]
            prev_upper = ind['band_upper'][i - 1]
            if self.require_both_above:
                was_above = prev_close > prev_upper
            else:
                was_above = prev_close > min(ind['sma'][i - 1], ind['ema'][i - 1])
            if was_above and state.position is not None:
                return False

        state.position = 'long'
        state.entry_price = ind['close'][i]
        state.max_favorable = ind['high'][i]
        state.trailing_stop = None
        return True

    def _check_short_entry(
        self, i: int, ind: Dict, state: '_BMSBState', is_below_band: bool,
    ) -> bool:
        """Check if a short entry should be taken.

        Validates long-only, band width, RSI, HTF, and cross-detection filters.

        Args:
            i: Bar index.
            ind: Pre-computed indicator arrays.
            state: Mutable position state (modified in-place on entry).
            is_below_band: Whether close is below the band.

        Returns:
            True if entry was taken.
        """
        if not is_below_band or self.long_only:
            return False

        bw = ind['band_width']
        if bw is not None and bw[i] < self.min_band_width_pct:
            return False

        rsi = ind['rsi']
        if rsi is not None and not np.isnan(rsi[i]):
            if rsi[i] >= self.rsi_short_threshold:
                return False

        htf = ind['htf_trend']
        if htf is not None and htf[i] != 'bearish':
            return False

        if i > 0:
            prev_close = ind['close'][i - 1]
            prev_lower = ind['band_lower'][i - 1]
            if self.require_both_below:
                was_below = prev_close < prev_lower
            else:
                was_below = prev_close < max(ind['sma'][i - 1], ind['ema'][i - 1])
            if was_below and state.position is not None:
                return False

        state.position = 'short'
        state.entry_price = ind['close'][i]
        state.max_favorable = ind['low'][i]
        state.trailing_stop = None
        return True

    def get_current_signal(self, data: pd.DataFrame) -> BMSBSignal:
        """
        Get current signal state for live trading.

        Args:
            data: Recent OHLCV data

        Returns:
            BMSBSignal with current state
        """
        return BMSBIndicators.get_current_signal(
            data, self.sma_period, self.ema_period
        )
