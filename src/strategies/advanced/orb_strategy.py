"""
Opening Range Breakout (ORB) Strategy.

A filtered intraday breakout strategy that trades breakouts from the
first 15 minutes of trading (9:30-9:45 AM ET).

Entry Filters:
- RVOL > 1.5 (volume confirmation)
- VWAP alignment (longs above, shorts below)
- EMA trend filter (9 EMA vs 21 EMA)
- Regime filter (no longs in BEAR, no shorts in STRONG_BULL)

Exit Logic:
- Stop-loss at opposite side of opening range
- Target at entry + 1x opening range height (1:1 R:R)
- Time exit at 3:45 PM ET (no overnight exposure)

Best Instruments:
- Leveraged ETFs (TQQQ, SOXL, UPRO, TNA, TECL)
- High liquidity required for clean breakouts
"""

from datetime import datetime, time, timedelta
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.backtesting.base.strategy import LongShortStrategy
from src.backtesting.utils.indicators import Indicators
from src.strategies.advanced.orb_indicators import ORBIndicators
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ORBPosition:
    """Represents an active ORB position."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    stop_loss: float
    target: float
    or_high: float
    or_low: float
    or_height: float


class ORBStrategy(LongShortStrategy):
    """
    Opening Range Breakout Strategy.

    Trades breakouts from the 15-minute opening range (9:30-9:45 AM ET)
    with multiple confirmation filters.

    Parameters:
        opening_range_minutes: Minutes for opening range (default 15)
        rvol_threshold: Minimum RVOL for entry (default 1.5)
        rvol_lookback: Days for RVOL calculation (default 20)
        fast_ema: Fast EMA period (default 9)
        slow_ema: Slow EMA period (default 21)
        target_multiplier: Target as multiple of OR height (default 1.0)
        long_only: Only take long trades (default False)
        use_regime: Enable regime filtering (default True)
        use_gap_filter: Align trade with gap direction (default False)
        exit_time: Force exit time (default 15:45 = 3:45 PM)
    """

    # Market timing constants (Eastern Time)
    MARKET_OPEN = time(9, 30)
    OR_END_15MIN = time(9, 45)
    OR_END_30MIN = time(10, 0)
    ENTRY_CUTOFF = time(15, 30)  # No new entries after 3:30 PM
    EXIT_TIME = time(15, 45)     # Force exit at 3:45 PM
    MARKET_CLOSE = time(16, 0)

    def __init__(
        self,
        opening_range_minutes: int = 15,
        rvol_threshold: float = 1.5,
        rvol_lookback: int = 20,
        fast_ema: int = 9,
        slow_ema: int = 21,
        target_multiplier: float = 1.0,
        long_only: bool = False,
        use_regime: bool = True,
        use_gap_filter: bool = False,
        exit_time_hour: int = 15,
        exit_time_minute: int = 45,
        **kwargs
    ):
        # Set attributes BEFORE calling super().__init__()
        # because parent class calls validate_parameters()
        self.opening_range_minutes = opening_range_minutes
        self.rvol_threshold = rvol_threshold
        self.rvol_lookback = rvol_lookback
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.target_multiplier = target_multiplier
        self.long_only = long_only
        self.use_regime = use_regime
        self.use_gap_filter = use_gap_filter

        # Calculate OR end time based on minutes
        or_end_minutes = 30 + opening_range_minutes
        self.or_end_time = time(9, or_end_minutes % 60)
        if or_end_minutes >= 60:
            self.or_end_time = time(10, or_end_minutes - 60)

        self.exit_time = time(exit_time_hour, exit_time_minute)

        # Regime detector (optional)
        self.regime_detector = MarketRegimeDetector() if use_regime else None
        self.current_regime = 'SIDEWAYS'

        # Now call parent init (which calls validate_parameters)
        super().__init__(
            opening_range_minutes=opening_range_minutes,
            rvol_threshold=rvol_threshold,
            rvol_lookback=rvol_lookback,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            target_multiplier=target_multiplier,
            long_only=long_only,
            use_regime=use_regime,
            use_gap_filter=use_gap_filter,
            **kwargs
        )

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.opening_range_minutes not in [5, 10, 15, 30, 60]:
            raise ValueError(
                f"opening_range_minutes must be 5, 10, 15, 30, or 60. "
                f"Got {self.opening_range_minutes}"
            )

        if self.rvol_threshold < 0:
            raise ValueError(f"rvol_threshold must be >= 0. Got {self.rvol_threshold}")

        if self.fast_ema >= self.slow_ema:
            raise ValueError(
                f"fast_ema ({self.fast_ema}) must be < slow_ema ({self.slow_ema})"
            )

        if self.target_multiplier <= 0:
            raise ValueError(
                f"target_multiplier must be > 0. Got {self.target_multiplier}"
            )

    def generate_long_short_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long and short entry/exit signals for ORB strategy.

        Processes minute-level data day by day:
        1. Calculate opening range (9:30-9:45)
        2. Scan for breakout entries (9:45-3:30)
        3. Track positions for stop/target/time exits

        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex (minute bars)

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        # Initialize signal arrays
        n = len(data)
        long_entries = pd.Series(False, index=data.index)
        long_exits = pd.Series(False, index=data.index)
        short_entries = pd.Series(False, index=data.index)
        short_exits = pd.Series(False, index=data.index)

        if data.empty:
            return long_entries, long_exits, short_entries, short_exits

        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            logger.warning(f"Missing required columns. Have: {data.columns.tolist()}")
            return long_entries, long_exits, short_entries, short_exits

        # Pre-calculate indicators
        data = data.copy()
        data['fast_ema'] = Indicators.ema(data['close'], self.fast_ema)
        data['slow_ema'] = Indicators.ema(data['close'], self.slow_ema)
        data['rvol'] = ORBIndicators.relative_volume(data['volume'], self.rvol_lookback)

        # Use pre-calculated VWAP if available, otherwise use close as proxy
        if 'vwap' not in data.columns:
            # Calculate simple VWAP approximation (reset daily would be better)
            data['vwap'] = data['close']

        # Group by date and process each day
        data['date'] = data.index.date
        data['time'] = data.index.time

        # Track active position
        position: Optional[ORBPosition] = None
        prev_day_close: Optional[float] = None

        for date, day_data in data.groupby('date'):
            # Calculate opening range for this day
            or_data = ORBIndicators.opening_range(
                day_data,
                start_time=self.MARKET_OPEN,
                end_time=self.or_end_time
            )

            if not or_data:
                # Skip day if we can't calculate OR
                prev_day_close = day_data['close'].iloc[-1]
                continue

            or_high = or_data['or_high']
            or_low = or_data['or_low']
            or_height = or_data['or_height']

            # Skip if OR is too narrow (avoid whipsaws)
            if or_height < 0.001 * or_high:  # Less than 0.1% of price
                prev_day_close = day_data['close'].iloc[-1]
                continue

            # Get gap direction if enabled
            gap_direction = 'flat'
            if self.use_gap_filter and prev_day_close is not None:
                today_open = day_data['open'].iloc[0]
                gap_direction = ORBIndicators.gap_direction(today_open, prev_day_close)

            # Process each bar of the day
            prev_close_inside = True  # Track if previous bar was inside OR

            for i, (idx, bar) in enumerate(day_data.iterrows()):
                bar_time = bar['time']

                # Skip opening range formation period
                if bar_time < self.or_end_time:
                    prev_close_inside = or_low <= bar['close'] <= or_high
                    continue

                # Check for exits first (if we have a position)
                if position is not None:
                    exit_triggered, exit_reason = self._check_exit(
                        position, bar['close'], bar['low'], bar['high'], bar_time
                    )

                    if exit_triggered:
                        if position.direction == 'long':
                            long_exits.at[idx] = True
                        else:
                            short_exits.at[idx] = True
                        position = None

                # Skip new entries if too late in day
                if bar_time >= self.ENTRY_CUTOFF:
                    prev_close_inside = or_low <= bar['close'] <= or_high
                    continue

                # Skip if already in position (one trade per day)
                if position is not None:
                    prev_close_inside = or_low <= bar['close'] <= or_high
                    continue

                # Check for breakout
                current_close_inside = or_low <= bar['close'] <= or_high

                # Detect breakout (close breaks range, previous was inside)
                breakout_direction = None

                if bar['close'] > or_high and prev_close_inside:
                    breakout_direction = 'long'
                elif bar['close'] < or_low and prev_close_inside:
                    breakout_direction = 'short'

                if breakout_direction:
                    # Check all entry filters
                    should_enter = self._check_entry_filters(
                        bar,
                        or_high,
                        or_low,
                        breakout_direction,
                        gap_direction
                    )

                    if should_enter:
                        # Calculate stop and target
                        targets = ORBIndicators.calculate_targets(
                            bar['close'],
                            or_high,
                            or_low,
                            breakout_direction,
                            self.target_multiplier
                        )

                        # Create position
                        position = ORBPosition(
                            symbol='',  # Will be set by engine
                            direction=breakout_direction,
                            entry_price=bar['close'],
                            entry_time=idx,
                            stop_loss=targets['stop_loss'],
                            target=targets['target'],
                            or_high=or_high,
                            or_low=or_low,
                            or_height=or_height
                        )

                        # Signal entry
                        if breakout_direction == 'long':
                            long_entries.at[idx] = True
                        else:
                            short_entries.at[idx] = True

                prev_close_inside = current_close_inside

            # End of day: force exit if still in position
            if position is not None:
                last_idx = day_data.index[-1]
                if position.direction == 'long':
                    long_exits.at[last_idx] = True
                else:
                    short_exits.at[last_idx] = True
                position = None

            # Update previous day close
            prev_day_close = day_data['close'].iloc[-1]

        return long_entries, long_exits, short_entries, short_exits

    def _check_entry_filters(
        self,
        bar: pd.Series,
        or_high: float,
        or_low: float,
        direction: str,
        gap_direction: str
    ) -> bool:
        """
        Check if all entry filters pass for the given breakout.

        Args:
            bar: Current bar data (Series with close, vwap, rvol, emas)
            or_high: Opening range high
            or_low: Opening range low
            direction: 'long' or 'short'
            gap_direction: 'gap_up', 'gap_down', or 'flat'

        Returns:
            True if all filters pass
        """
        # Long-only mode check
        if self.long_only and direction == 'short':
            return False

        # RVOL filter
        if bar['rvol'] < self.rvol_threshold:
            return False

        # VWAP filter
        if direction == 'long' and bar['close'] <= bar['vwap']:
            return False
        if direction == 'short' and bar['close'] >= bar['vwap']:
            return False

        # EMA trend filter
        if direction == 'long' and bar['fast_ema'] <= bar['slow_ema']:
            return False
        if direction == 'short' and bar['fast_ema'] >= bar['slow_ema']:
            return False

        # Gap filter (optional)
        if self.use_gap_filter:
            if direction == 'long' and gap_direction == 'gap_down':
                return False
            if direction == 'short' and gap_direction == 'gap_up':
                return False

        # Regime filter (optional)
        if self.use_regime:
            if direction == 'long' and self.current_regime == 'BEAR':
                return False
            if direction == 'short' and self.current_regime == 'STRONG_BULL':
                return False

        return True

    def _check_exit(
        self,
        position: ORBPosition,
        current_close: float,
        current_low: float,
        current_high: float,
        current_time: time
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Args:
            position: Current position
            current_close: Current bar's close
            current_low: Current bar's low
            current_high: Current bar's high
            current_time: Current bar's time

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        # Time exit (force close before market close)
        if current_time >= self.exit_time:
            return True, 'time_exit'

        if position.direction == 'long':
            # Stop-loss hit (use low for more accurate stop check)
            if current_low <= position.stop_loss:
                return True, 'stop_loss'

            # Target hit (use high for more accurate target check)
            if current_high >= position.target:
                return True, 'target'

        else:  # short
            # Stop-loss hit
            if current_high >= position.stop_loss:
                return True, 'stop_loss'

            # Target hit
            if current_low <= position.target:
                return True, 'target'

        return False, ''

    def set_regime(self, regime: str) -> None:
        """
        Set the current market regime.

        Called externally when regime detector classifies market.

        Args:
            regime: One of 'STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS',
                   'UNPREDICTABLE', 'BEAR'
        """
        self.current_regime = regime

    def get_parameters(self) -> Dict:
        """Get strategy parameters."""
        return {
            'opening_range_minutes': self.opening_range_minutes,
            'rvol_threshold': self.rvol_threshold,
            'rvol_lookback': self.rvol_lookback,
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'target_multiplier': self.target_multiplier,
            'long_only': self.long_only,
            'use_regime': self.use_regime,
            'use_gap_filter': self.use_gap_filter
        }
