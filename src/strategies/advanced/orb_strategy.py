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
from typing import Tuple, Dict, Optional, List, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np

from src.backtesting.base.strategy import LongShortStrategy
from src.backtesting.utils.indicators import Indicators
from src.strategies.advanced.orb_indicators import ORBIndicators
from src.strategies.advanced.exit_checker import check_exit
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.orb_numba_core import (
    generate_orb_signals_numba,
    calculate_or_avg_volumes_numba
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ORBConfig:
    """Configuration for Opening Range Breakout Strategy parameters.

    Groups all constructor parameters into a single config object to reduce
    parameter explosion. Supports both config-based and kwargs-based
    construction for backward compatibility.
    """

    # Core parameters
    opening_range_minutes: int = 15
    rvol_threshold: float = 1.5
    rvol_lookback: int = 20
    fast_ema: int = 9
    slow_ema: int = 21
    target_multiplier: float = 1.0
    long_only: bool = False
    use_regime: bool = True
    use_gap_filter: bool = False
    exit_time_hour: int = 15
    exit_time_minute: int = 45
    one_trade_per_day: bool = False
    min_or_width_pct: float = 0.0

    # Phase 2: Entry Quality
    breakout_buffer_pct: float = 0.0
    entry_cutoff_hour: int = 15
    entry_cutoff_minute: int = 30

    # Phase 3: Risk Management
    use_atr_stops: bool = False
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    use_volume_confirmation: bool = False
    volume_breakout_mult: float = 1.5
    use_trailing_stop: bool = False

    # Phase 4: Earnings Filter
    use_earnings_filter: bool = False
    earnings_buffer_days: int = 2
    earnings_calendar_path: Optional[str] = None


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
        one_trade_per_day: If True, only take one trade per day (default False)
        min_or_width_pct: Minimum OR width as % of price (default 0.0 = no filter)
                          Recommended: 0.5% for 2x ETFs, 0.75% for 3x ETFs
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
        config: Optional[ORBConfig] = None,
        **kwargs
    ):
        # Build config from kwargs if not provided (backward compatibility)
        if config is None:
            config = ORBConfig(**kwargs)
        self._config = config

        # Set attributes BEFORE calling super().__init__()
        # because parent class calls validate_parameters()
        self.opening_range_minutes = config.opening_range_minutes
        self.rvol_threshold = config.rvol_threshold
        self.rvol_lookback = config.rvol_lookback
        self.fast_ema = config.fast_ema
        self.slow_ema = config.slow_ema
        self.target_multiplier = config.target_multiplier
        self.long_only = config.long_only
        self.use_regime = config.use_regime
        self.use_gap_filter = config.use_gap_filter
        self.one_trade_per_day = config.one_trade_per_day
        self.min_or_width_pct = config.min_or_width_pct

        # Phase 2: Entry Quality
        self.breakout_buffer_pct = config.breakout_buffer_pct
        self.entry_cutoff_hour = config.entry_cutoff_hour
        self.entry_cutoff_minute = config.entry_cutoff_minute

        # Phase 3: Risk Management
        self.use_atr_stops = config.use_atr_stops
        self.atr_period = config.atr_period
        self.atr_stop_multiplier = config.atr_stop_multiplier
        self.use_volume_confirmation = config.use_volume_confirmation
        self.volume_breakout_mult = config.volume_breakout_mult
        self.use_trailing_stop = config.use_trailing_stop

        # Phase 4: Earnings Filter
        self.use_earnings_filter = config.use_earnings_filter
        self.earnings_buffer_days = config.earnings_buffer_days
        self.earnings_calendar_path = config.earnings_calendar_path
        self.earnings_blackout_dates: Set[datetime] = set()

        # Load earnings calendar if filter is enabled
        if config.use_earnings_filter:
            self._load_earnings_calendar()

        # Calculate OR end time based on minutes
        or_end_minutes = 30 + config.opening_range_minutes
        self.or_end_time = time(9, or_end_minutes % 60)
        if or_end_minutes >= 60:
            self.or_end_time = time(10, or_end_minutes - 60)

        self.exit_time = time(config.exit_time_hour, config.exit_time_minute)

        # Regime detector (optional)
        self.regime_detector = MarketRegimeDetector() if config.use_regime else None
        self.current_regime = 'SIDEWAYS'

        # Now call parent init (which calls validate_parameters)
        super().__init__(**asdict(config))

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

        if self.min_or_width_pct < 0:
            raise ValueError(
                f"min_or_width_pct must be >= 0. Got {self.min_or_width_pct}"
            )

    def _load_earnings_calendar(self) -> None:
        """Load earnings calendar and build blackout date set."""
        if not self.earnings_calendar_path:
            # Use default path
            project_root = Path(__file__).parent.parent.parent.parent
            default_path = project_root / "config/universes" / "earnings_calendar_sp500_highvol.csv"
            if default_path.exists():
                self.earnings_calendar_path = str(default_path)
            else:
                logger.warning("Earnings calendar not found, disabling filter")
                self.use_earnings_filter = False
                return

        try:
            earnings_df = pd.read_csv(self.earnings_calendar_path)
            earnings_df['earnings_date'] = pd.to_datetime(earnings_df['earnings_date'])

            # Build blackout dates: earnings_date +/- buffer_days
            for _, row in earnings_df.iterrows():
                earnings_date = row['earnings_date'].date()
                for offset in range(-self.earnings_buffer_days, self.earnings_buffer_days + 1):
                    blackout_date = earnings_date + timedelta(days=offset)
                    self.earnings_blackout_dates.add(blackout_date)

            logger.info(
                f"Loaded earnings calendar: {len(earnings_df)} earnings dates, "
                f"{len(self.earnings_blackout_dates)} blackout dates "
                f"(+/- {self.earnings_buffer_days} days)"
            )
        except Exception as e:
            logger.error(f"Failed to load earnings calendar: {e}")
            self.use_earnings_filter = False

    def _is_earnings_blackout(self, date: datetime) -> bool:
        """Check if a date falls within earnings blackout period."""
        if not self.use_earnings_filter:
            return False
        return date.date() in self.earnings_blackout_dates

    def generate_long_short_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long and short entry/exit signals for ORB strategy.

        Uses Numba JIT-compiled core for 10-50x speedup on minute data.

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

        # Calculate ATR for ATR-based stops (Phase 3)
        data['atr'] = Indicators.atr(data['high'], data['low'], data['close'], self.atr_period)

        # Use pre-calculated VWAP if available, otherwise use close as proxy
        if 'vwap' not in data.columns:
            data['vwap'] = data['close']

        # Convert to numpy arrays for Numba
        # Extract time components
        hour_of_day = data.index.hour.values
        minute_of_day = data.index.minute.values
        dates = data.index.date
        # Convert dates to days since epoch
        dates_int = np.array([(d - pd.Timestamp('1970-01-01').date()).days for d in dates], dtype=np.int64)

        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        volumes = data['volume'].values
        vwaps = data['vwap'].values
        fast_emas = data['fast_ema'].values
        slow_emas = data['slow_ema'].values
        rvols = data['rvol'].values
        atrs = data['atr'].fillna(0).values

        # Calculate OR average volumes for volume confirmation (Phase 3)
        or_avg_volumes_arr, or_dates_arr = calculate_or_avg_volumes_numba(
            hour_of_day=hour_of_day,
            minute_of_day=minute_of_day,
            dates=dates_int,
            volumes=volumes,
            or_end_minutes=self.opening_range_minutes
        )

        # Create mapping from date to avg volume for indexing
        # The Numba function will use the or_idx to look up the avg volume
        # We pass the or_avg_volumes array directly - it's indexed by OR date order
        or_avg_volumes = or_avg_volumes_arr

        # Call Numba-optimized signal generation
        long_entry_arr, long_exit_arr, short_entry_arr, short_exit_arr = generate_orb_signals_numba(
            hour_of_day=hour_of_day,
            minute_of_day=minute_of_day,
            dates=dates_int,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            vwaps=vwaps,
            fast_emas=fast_emas,
            slow_emas=slow_emas,
            rvols=rvols,
            atrs=atrs,
            or_avg_volumes=or_avg_volumes,
            or_end_minutes=self.opening_range_minutes,
            rvol_threshold=self.rvol_threshold,
            target_multiplier=self.target_multiplier,
            long_only=self.long_only,
            use_vwap_filter=True,
            use_ema_filter=True,
            exit_hour=self.exit_time.hour,
            exit_minute=self.exit_time.minute,
            one_trade_per_day=self.one_trade_per_day,
            min_or_width_pct=self.min_or_width_pct,
            # Phase 2: Entry Quality
            breakout_buffer_pct=self.breakout_buffer_pct,
            entry_cutoff_hour=self.entry_cutoff_hour,
            entry_cutoff_minute=self.entry_cutoff_minute,
            # Phase 3: Risk Management
            use_atr_stops=self.use_atr_stops,
            atr_stop_multiplier=self.atr_stop_multiplier,
            use_volume_confirmation=self.use_volume_confirmation,
            volume_breakout_mult=self.volume_breakout_mult,
            use_trailing_stop=self.use_trailing_stop
        )

        # Convert back to Series
        long_entries = pd.Series(long_entry_arr, index=data.index)
        long_exits = pd.Series(long_exit_arr, index=data.index)
        short_entries = pd.Series(short_entry_arr, index=data.index)
        short_exits = pd.Series(short_exit_arr, index=data.index)

        # Phase 4: Apply earnings blackout filter (post-filter entry signals)
        if self.use_earnings_filter and len(self.earnings_blackout_dates) > 0:
            # Create mask for non-blackout dates
            blackout_mask = pd.Series(
                [d in self.earnings_blackout_dates for d in data.index.date],
                index=data.index
            )
            n_entries_before = long_entries.sum() + short_entries.sum()

            # Zero out entries on blackout dates (keep exits - we still want to exit)
            long_entries = long_entries & ~blackout_mask
            short_entries = short_entries & ~blackout_mask

            n_entries_after = long_entries.sum() + short_entries.sum()
            n_filtered = n_entries_before - n_entries_after
            if n_filtered > 0:
                logger.debug(
                    f"Earnings filter: blocked {n_filtered} entries "
                    f"({n_entries_after} remaining)"
                )

        return long_entries, long_exits, short_entries, short_exits

    def _generate_long_short_signals_python(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        LEGACY: Pure Python implementation (kept for reference/fallback).

        This method is replaced by Numba-optimized version but kept for:
        - Debugging and validation
        - Fallback if Numba unavailable
        - Reference implementation
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

        Delegates to shared exit_checker for time exit, stop-loss,
        and target checks.

        Args:
            position: Current position
            current_close: Current bar's close
            current_low: Current bar's low
            current_high: Current bar's high
            current_time: Current bar's time

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        pos_dir = 1 if position.direction == 'long' else -1

        exit_reason, reason_str = check_exit(
            position_dir=pos_dir,
            high=current_high,
            low=current_low,
            stop=position.stop_loss,
            target=position.target,
            current_time=current_time,
            exit_time=self.exit_time,
        )

        if exit_reason is not None:
            return True, reason_str

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
            'use_gap_filter': self.use_gap_filter,
            'one_trade_per_day': self.one_trade_per_day,
            'min_or_width_pct': self.min_or_width_pct,
            # Phase 2: Entry Quality
            'breakout_buffer_pct': self.breakout_buffer_pct,
            'entry_cutoff_hour': self.entry_cutoff_hour,
            'entry_cutoff_minute': self.entry_cutoff_minute,
            # Phase 3: Risk Management
            'use_atr_stops': self.use_atr_stops,
            'atr_period': self.atr_period,
            'atr_stop_multiplier': self.atr_stop_multiplier,
            'use_volume_confirmation': self.use_volume_confirmation,
            'volume_breakout_mult': self.volume_breakout_mult,
            'use_trailing_stop': self.use_trailing_stop,
            # Phase 4: Earnings Filter
            'use_earnings_filter': self.use_earnings_filter,
            'earnings_buffer_days': self.earnings_buffer_days,
            'earnings_calendar_path': self.earnings_calendar_path
        }
