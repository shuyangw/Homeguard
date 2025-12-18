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
from typing import Tuple, Optional
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
        """
        Generate long and short entry/exit signals based on BMSB.

        Signal Logic:
        - Long Entry: Close crosses above band (close > both SMA and EMA)
        - Long Exit: Close crosses below band OR trailing stop hit
        - Short Entry: Close crosses below band (close < both SMA and EMA)
        - Short Exit: Close crosses above band OR trailing stop hit

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

        # Initialize signal arrays
        long_entries_arr = np.zeros(n, dtype=bool)
        long_exits_arr = np.zeros(n, dtype=bool)
        short_entries_arr = np.zeros(n, dtype=bool)
        short_exits_arr = np.zeros(n, dtype=bool)

        # ==================================================================
        # PHASE 1: CALCULATE BMSB BAND
        # ==================================================================

        # Calculate band on specified timeframe
        band_df = BMSBIndicators.calculate_band_on_original_timeframe(
            data, self.sma_period, self.ema_period, timeframe=self.timeframe
        )

        if band_df.empty:
            logger.warning("Insufficient data for BMSB calculation")
            return (
                pd.Series(long_entries_arr, index=data.index),
                pd.Series(long_exits_arr, index=data.index),
                pd.Series(short_entries_arr, index=data.index),
                pd.Series(short_exits_arr, index=data.index)
            )

        # Get timeframe closes for signal timing
        if self.timeframe == 'weekly':
            tf_df = BMSBIndicators.resample_to_weekly(data)
        elif self.timeframe == 'daily':
            tf_df = BMSBIndicators.resample_to_daily(data)
        else:
            tf_df = data
        timeframe_closes = set(tf_df.index)

        # ==================================================================
        # PHASE 2: CALCULATE FILTERS
        # ==================================================================

        # Get aligned arrays
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # Get band values (forward-filled from timeframe)
        sma = band_df['sma'].reindex(data.index, method='ffill').values
        ema = band_df['ema'].reindex(data.index, method='ffill').values
        band_upper = band_df['band_upper'].reindex(data.index, method='ffill').values
        band_lower = band_df['band_lower'].reindex(data.index, method='ffill').values

        # Calculate optional filters
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

        band_width = None
        if self.min_band_width_pct > 0:
            band_width = ((band_upper - band_lower) / close)

        # ==================================================================
        # PHASE 3: GENERATE SIGNALS
        # ==================================================================

        # Track position state for signal generation
        position = None  # 'long', 'short', or None
        entry_price = 0.0
        trailing_stop = None
        max_favorable = 0.0

        # Minimum bars needed for signal (based on MA periods and timeframe)
        if self.timeframe == 'weekly':
            min_bars = max(self.sma_period, self.ema_period) * 7  # ~7 days per week
        elif self.timeframe == 'daily':
            min_bars = max(self.sma_period, self.ema_period)
        else:
            min_bars = max(self.sma_period, self.ema_period)

        for i in range(min_bars, n):
            current_close = close[i]
            current_high = high[i]
            current_low = low[i]
            current_sma = sma[i]
            current_ema = ema[i]
            current_upper = band_upper[i]
            current_lower = band_lower[i]

            # Skip if band values are NaN
            if np.isnan(current_sma) or np.isnan(current_ema):
                continue

            timestamp = data.index[i]

            # Determine current position relative to band
            if self.require_both_above:
                is_above_band = current_close > current_upper
            else:
                is_above_band = current_close > min(current_sma, current_ema)

            if self.require_both_below:
                is_below_band = current_close < current_lower
            else:
                is_below_band = current_close < max(current_sma, current_ema)

            # Only generate signals on timeframe close if configured
            is_timeframe_close = timestamp in timeframe_closes
            can_signal = (not self.signal_on_close) or is_timeframe_close

            # ============================================================
            # EXIT LOGIC (check first before entries)
            # ============================================================

            if position == 'long':
                # Update trailing stop
                if self.use_trailing_stop:
                    max_favorable = max(max_favorable, current_high)
                    if self.use_atr_stop and atr is not None and not np.isnan(atr[i]):
                        trailing_stop = max_favorable - (atr[i] * self.atr_stop_multiplier)
                    else:
                        trailing_stop = max_favorable * (1 - self.trailing_stop_pct)

                # Check exit conditions
                exit_signal = False
                exit_reason = ''

                # Trailing stop hit
                if trailing_stop and current_low <= trailing_stop:
                    exit_signal = True
                    exit_reason = 'trailing_stop'

                # Band exit (price crosses below)
                elif is_below_band and can_signal:
                    exit_signal = True
                    exit_reason = 'band_cross'

                if exit_signal:
                    long_exits_arr[i] = True
                    position = None
                    trailing_stop = None
                    max_favorable = 0.0

                    # Generate short entry if enabled and band exit
                    if not self.long_only and exit_reason == 'band_cross':
                        short_entries_arr[i] = True
                        position = 'short'
                        entry_price = current_close
                        max_favorable = current_low

            elif position == 'short':
                # Update trailing stop (inverted for short)
                if self.use_trailing_stop:
                    max_favorable = min(max_favorable, current_low) if max_favorable > 0 else current_low
                    if self.use_atr_stop and atr is not None and not np.isnan(atr[i]):
                        trailing_stop = max_favorable + (atr[i] * self.atr_stop_multiplier)
                    else:
                        trailing_stop = max_favorable * (1 + self.trailing_stop_pct)

                # Check exit conditions
                exit_signal = False
                exit_reason = ''

                # Trailing stop hit (price goes above)
                if trailing_stop and current_high >= trailing_stop:
                    exit_signal = True
                    exit_reason = 'trailing_stop'

                # Band exit (price crosses above)
                elif is_above_band and can_signal:
                    exit_signal = True
                    exit_reason = 'band_cross'

                if exit_signal:
                    short_exits_arr[i] = True
                    position = None
                    trailing_stop = None
                    max_favorable = 0.0

                    # Generate long entry on band cross
                    if exit_reason == 'band_cross':
                        long_entries_arr[i] = True
                        position = 'long'
                        entry_price = current_close
                        max_favorable = current_high

            # ============================================================
            # ENTRY LOGIC (only if no position)
            # ============================================================

            if position is None and can_signal:
                # Check band width filter
                passes_band_width = True
                if band_width is not None and band_width[i] < self.min_band_width_pct:
                    passes_band_width = False

                # Long entry: price crosses above band
                if is_above_band and passes_band_width:
                    # Check filters
                    passes_rsi = True
                    if rsi is not None and not np.isnan(rsi[i]):
                        passes_rsi = rsi[i] > self.rsi_long_threshold

                    passes_htf = True
                    if htf_trend is not None:
                        passes_htf = htf_trend[i] == 'bullish'

                    if passes_rsi and passes_htf:
                        # Check previous bar was not above (cross detection)
                        if i > 0:
                            prev_close = close[i - 1]
                            prev_upper = band_upper[i - 1]
                            was_above = prev_close > prev_upper if self.require_both_above else prev_close > min(sma[i-1], ema[i-1])
                            if not was_above or position is None:  # First signal or cross
                                long_entries_arr[i] = True
                                position = 'long'
                                entry_price = current_close
                                max_favorable = current_high
                                trailing_stop = None

                # Short entry: price crosses below band
                elif is_below_band and not self.long_only and passes_band_width:
                    # Check filters
                    passes_rsi = True
                    if rsi is not None and not np.isnan(rsi[i]):
                        passes_rsi = rsi[i] < self.rsi_short_threshold

                    passes_htf = True
                    if htf_trend is not None:
                        passes_htf = htf_trend[i] == 'bearish'

                    if passes_rsi and passes_htf:
                        # Check previous bar was not below (cross detection)
                        if i > 0:
                            prev_close = close[i - 1]
                            prev_lower = band_lower[i - 1]
                            was_below = prev_close < prev_lower if self.require_both_below else prev_close < max(sma[i-1], ema[i-1])
                            if not was_below or position is None:  # First signal or cross
                                short_entries_arr[i] = True
                                position = 'short'
                                entry_price = current_close
                                max_favorable = current_low
                                trailing_stop = None

        # Convert to pandas Series
        return (
            pd.Series(long_entries_arr, index=data.index),
            pd.Series(long_exits_arr, index=data.index),
            pd.Series(short_entries_arr, index=data.index),
            pd.Series(short_exits_arr, index=data.index)
        )

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
