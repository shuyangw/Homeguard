"""
High Volatility Opening Range Breakout (HV ORB) Strategy.

An advanced intraday breakout strategy that focuses on "Stocks in Play" -
stocks with abnormally high opening volume driven by news catalysts.

Key Features:
- Stocks in Play (SIP) scoring: Opening volume vs 14-day average
- Pre-market gap filtering: 2-15% gaps only
- ATR-based volatility bounds
- Tiered exits: Target 1 (50%), Target 2 (remaining), Trailing stop, EOD
- Daily risk limits: Max loss %, max trades, max concurrent positions
- Optional FinBERT sentiment filtering (Phase 2+)

Based on research by Zarattini, Barbon & Aziz (2024): General ORB on all
stocks underperforms. The edge concentrates in "Stocks in Play" with
abnormally high volume driven by news catalysts.

Best Instruments:
- Leveraged ETFs (TQQQ, SOXL, UPRO, TNA, TECL)
- High-volume stocks with 2-15% gaps
- Stocks with SIP score >= 2.0
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np

from src.backtesting.base.strategy import LongShortStrategy
from src.backtesting.utils.indicators import Indicators
from src.strategies.advanced.hv_orb_indicators import HVORBIndicators
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.backtesting.utils.tiered_exit_manager import (
    TieredExitManager,
    TieredExitConfig,
    TieredPosition,
    DailyRiskManager,
    ExitType
)
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HVORBPosition:
    """Represents an active ORB HV position."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_bar_idx: int
    entry_time: datetime
    shares: int

    # Exit levels
    stop_loss: float
    target1: float
    target2: float
    or_height: float

    # Entry quality metrics
    sip_score: float
    gap_pct: float
    confidence: float

    # State tracking
    target1_hit: bool = False
    remaining_pct: float = 1.0
    current_stop: float = 0.0
    trailing_active: bool = False
    max_favorable_price: float = 0.0


class HVORBStrategy(LongShortStrategy):
    """
    High Volatility Opening Range Breakout (HV ORB) Strategy.

    Trades breakouts from the 5-minute opening range (9:30-9:35 AM ET)
    with "Stocks in Play" scoring and tiered exit management.

    Parameters:
        opening_range_minutes: Minutes for opening range (default 5)
        sip_lookback_days: Days for SIP score average (default 14)
        sip_min_score: Minimum SIP score for entry (default 2.0)
        min_gap_pct: Minimum gap percentage (default 0.02 = 2%)
        max_gap_pct: Maximum gap percentage (default 0.15 = 15%)
        atr_period: ATR calculation period (default 14)
        min_atr_pct: Minimum ATR% for volatility (default 2.0%)
        max_atr_pct: Maximum ATR% for volatility (default 10.0%)
        target1_multiplier: Target 1 as multiple of OR height (default 1.0)
        target1_exit_pct: Percentage to exit at Target 1 (default 0.5)
        target2_multiplier: Target 2 as multiple of OR height (default 2.0)
        use_trailing_stop: Enable trailing stop after Target 1 (default True)
        trailing_offset_pct: Trailing stop offset (default 0.02 = 2%)
        eod_exit_hour: EOD exit hour (default 15 = 3 PM)
        eod_exit_minute: EOD exit minute (default 55 = 3:55 PM)
        daily_max_loss_pct: Maximum daily loss (default 0.03 = 3%)
        max_daily_trades: Maximum trades per day (default 10)
        max_concurrent_positions: Maximum concurrent positions (default 5)
        use_sentiment: Enable sentiment filtering (default False)
        min_sentiment_score: Minimum sentiment score (default 0.2)
        use_regime: Enable regime filtering (default True)
        long_only: Only take long trades (default False)
    """

    # Market timing constants (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    DEFAULT_ENTRY_CUTOFF = time(15, 30)

    def __init__(
        self,
        # Opening Range
        opening_range_minutes: int = 5,

        # SIP Score
        sip_lookback_days: int = 14,
        sip_min_score: float = 2.0,

        # Gap Filtering
        min_gap_pct: float = 0.02,
        max_gap_pct: float = 0.15,
        gap_direction_filter: bool = True,

        # Volatility (ATR)
        atr_period: int = 14,
        min_atr_pct: float = 2.0,
        max_atr_pct: float = 10.0,

        # Tiered Exits
        target1_multiplier: float = 1.0,
        target1_exit_pct: float = 0.5,
        target2_multiplier: float = 2.0,
        use_trailing_stop: bool = True,
        trailing_offset_pct: float = 0.02,
        eod_exit_hour: int = 15,
        eod_exit_minute: int = 55,

        # Risk Management
        daily_max_loss_pct: float = 0.03,
        max_daily_trades: int = 10,
        max_concurrent_positions: int = 5,

        # Entry filters
        entry_cutoff_hour: int = 15,
        entry_cutoff_minute: int = 30,
        rvol_threshold: float = 1.5,

        # Sentiment (Phase 2)
        use_sentiment: bool = False,
        min_sentiment_score: float = 0.2,
        min_sentiment_confidence: float = 0.0,  # Filter low-confidence sentiment
        sentiment_data_path: Optional[str] = None,
        skip_earnings_days: bool = True,  # Skip trading on earnings announcement days

        # Regime
        use_regime: bool = True,

        # Momentum filter (require price > MA for longs)
        use_momentum_filter: bool = False,
        momentum_ma_period: int = 20,

        # Pullback entry (wait for retest instead of immediate entry)
        use_pullback_entry: bool = False,
        pullback_threshold_pct: float = 0.5,  # How close to OR high for valid pullback
        pullback_timeout_bars: int = 30,  # Cancel setup after N bars without pullback

        # Time-based stop (exit if not profitable after N minutes)
        use_time_stop: bool = False,
        time_stop_minutes: int = 30,  # Exit if not profitable after this many minutes

        # Base
        long_only: bool = False,
        **kwargs
    ):
        # Set all attributes BEFORE calling super().__init__()
        self.opening_range_minutes = opening_range_minutes
        self.sip_lookback_days = sip_lookback_days
        self.sip_min_score = sip_min_score

        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.gap_direction_filter = gap_direction_filter

        self.atr_period = atr_period
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct

        self.target1_multiplier = target1_multiplier
        self.target1_exit_pct = target1_exit_pct
        self.target2_multiplier = target2_multiplier
        self.use_trailing_stop = use_trailing_stop
        self.trailing_offset_pct = trailing_offset_pct
        self.eod_exit_hour = eod_exit_hour
        self.eod_exit_minute = eod_exit_minute

        self.daily_max_loss_pct = daily_max_loss_pct
        self.max_daily_trades = max_daily_trades
        self.max_concurrent_positions = max_concurrent_positions

        self.entry_cutoff_hour = entry_cutoff_hour
        self.entry_cutoff_minute = entry_cutoff_minute
        self.rvol_threshold = rvol_threshold

        self.use_sentiment = use_sentiment
        self.min_sentiment_score = min_sentiment_score
        self.min_sentiment_confidence = min_sentiment_confidence
        self.sentiment_data_path = sentiment_data_path
        self.skip_earnings_days = skip_earnings_days

        self.use_regime = use_regime
        self.long_only = long_only

        self.use_momentum_filter = use_momentum_filter
        self.momentum_ma_period = momentum_ma_period
        self._daily_ma = {}  # Cache for daily MA values by symbol

        self.use_pullback_entry = use_pullback_entry
        self.pullback_threshold_pct = pullback_threshold_pct
        self.pullback_timeout_bars = pullback_timeout_bars
        self._pending_setups = {}  # Track breakouts waiting for pullback entry

        self.use_time_stop = use_time_stop
        self.time_stop_minutes = time_stop_minutes

        # Calculate OR end time
        or_end_minutes = 30 + opening_range_minutes
        self.or_end_time = time(9, or_end_minutes % 60)
        if or_end_minutes >= 60:
            self.or_end_time = time(10, or_end_minutes - 60)

        self.eod_exit_time = time(eod_exit_hour, eod_exit_minute)
        self.entry_cutoff_time = time(entry_cutoff_hour, entry_cutoff_minute)

        # Initialize tiered exit manager
        self.exit_config = TieredExitConfig(
            target1_multiplier=target1_multiplier,
            target2_multiplier=target2_multiplier,
            target1_exit_pct=target1_exit_pct,
            use_trailing=use_trailing_stop,
            trailing_offset_pct=trailing_offset_pct,
            eod_exit_time=self.eod_exit_time
        )
        self.exit_manager = TieredExitManager(self.exit_config)

        # Regime detector (optional)
        self.regime_detector = MarketRegimeDetector() if use_regime else None
        self.current_regime = 'SIDEWAYS'

        # Sentiment data (loaded on first use)
        self._sentiment_data = None

        # Call parent init
        super().__init__(
            opening_range_minutes=opening_range_minutes,
            sip_lookback_days=sip_lookback_days,
            sip_min_score=sip_min_score,
            min_gap_pct=min_gap_pct,
            max_gap_pct=max_gap_pct,
            gap_direction_filter=gap_direction_filter,
            atr_period=atr_period,
            min_atr_pct=min_atr_pct,
            max_atr_pct=max_atr_pct,
            target1_multiplier=target1_multiplier,
            target1_exit_pct=target1_exit_pct,
            target2_multiplier=target2_multiplier,
            use_trailing_stop=use_trailing_stop,
            trailing_offset_pct=trailing_offset_pct,
            eod_exit_hour=eod_exit_hour,
            eod_exit_minute=eod_exit_minute,
            daily_max_loss_pct=daily_max_loss_pct,
            max_daily_trades=max_daily_trades,
            max_concurrent_positions=max_concurrent_positions,
            entry_cutoff_hour=entry_cutoff_hour,
            entry_cutoff_minute=entry_cutoff_minute,
            rvol_threshold=rvol_threshold,
            use_sentiment=use_sentiment,
            min_sentiment_score=min_sentiment_score,
            use_regime=use_regime,
            long_only=long_only,
            **kwargs
        )

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.opening_range_minutes not in [3, 5, 10, 15, 30]:
            raise ValueError(
                f"opening_range_minutes must be 3, 5, 10, 15, or 30. "
                f"Got {self.opening_range_minutes}"
            )

        if self.sip_min_score < 1.0:
            raise ValueError(
                f"sip_min_score must be >= 1.0. Got {self.sip_min_score}"
            )

        if self.min_gap_pct < 0 or self.min_gap_pct >= self.max_gap_pct:
            raise ValueError(
                f"min_gap_pct ({self.min_gap_pct}) must be >= 0 and < "
                f"max_gap_pct ({self.max_gap_pct})"
            )

        if self.target1_multiplier >= self.target2_multiplier:
            raise ValueError(
                f"target1_multiplier ({self.target1_multiplier}) must be < "
                f"target2_multiplier ({self.target2_multiplier})"
            )

        if self.daily_max_loss_pct <= 0 or self.daily_max_loss_pct > 1.0:
            raise ValueError(
                f"daily_max_loss_pct must be between 0 and 1. "
                f"Got {self.daily_max_loss_pct}"
            )

    def generate_long_short_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long and short entry/exit signals for ORB HV strategy.

        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex (minute bars)

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        # Initialize signal arrays with original index (for proper return)
        original_index = data.index
        long_entries = pd.Series(False, index=original_index)
        long_exits = pd.Series(False, index=original_index)
        short_entries = pd.Series(False, index=original_index)
        short_exits = pd.Series(False, index=original_index)

        if data.empty:
            return long_entries, long_exits, short_entries, short_exits

        # Store data reference for momentum MA calculation
        self._current_data = data

        # Load sentiment data if enabled
        if self.use_sentiment:
            symbol = getattr(self, '_current_symbol', None)
            if symbol:
                # Track last symbol to detect changes
                last_symbol = getattr(self, '_last_sentiment_symbol', None)
                if symbol != last_symbol:
                    # New symbol - reload sentiment data
                    self._sentiment_data = None
                    self._last_sentiment_symbol = symbol
                    start_date = data.index.min().to_pydatetime()
                    end_date = data.index.max().to_pydatetime()
                    self.load_sentiment_for_symbols([symbol], start_date, end_date)

        # Filter to market hours (9:30 AM - 4:00 PM ET)
        # Convert index to ET for filtering
        if data.index.tz is not None:
            et_index = data.index.tz_convert('America/New_York')
        else:
            et_index = data.index.tz_localize('UTC').tz_convert('America/New_York')

        # Create mask for market hours
        market_open = time(9, 30)
        market_close = time(16, 0)
        et_times = et_index.time
        market_hours_mask = (et_times >= market_open) & (et_times <= market_close)

        # Filter data to market hours only
        data = data[market_hours_mask].copy()
        if data.empty:
            logger.warning("No data in market hours after filtering")
            return long_entries, long_exits, short_entries, short_exits

        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            logger.warning(f"Missing required columns. Have: {data.columns.tolist()}")
            return long_entries, long_exits, short_entries, short_exits

        # Pre-calculate indicators
        data = data.copy()

        # ATR for volatility filter
        data['atr'] = HVORBIndicators.atr(
            data['high'], data['low'], data['close'], self.atr_period
        )
        data['atr_pct'] = (data['atr'] / data['close']) * 100

        # RVOL for volume confirmation
        data['rvol'] = Indicators.sma(data['volume'], 20)
        data['rvol'] = data['volume'] / data['rvol'].replace(0, np.nan)
        data['rvol'] = data['rvol'].fillna(1.0)

        # VWAP if not available
        if 'vwap' not in data.columns:
            data['vwap'] = data['close']

        # Calculate SIP scores and gaps by day
        # Convert to Eastern Time for proper market hour comparisons
        if data.index.tz is not None:
            et_index = data.index.tz_convert('America/New_York')
        else:
            et_index = data.index.tz_localize('UTC').tz_convert('America/New_York')
        data['date'] = et_index.date
        data['time'] = et_index.time

        # Get daily gap and SIP data
        gap_df = HVORBIndicators.gap_by_day(data, self.min_gap_pct, self.max_gap_pct)
        sip_scores = HVORBIndicators.stocks_in_play_score(
            data,
            lookback_days=self.sip_lookback_days,
            opening_minutes=self.opening_range_minutes
        )

        # Generate signals using pure Python implementation
        filtered_long_entries, filtered_long_exits, filtered_short_entries, filtered_short_exits = self._generate_signals_python(
            data=data,
            gap_df=gap_df,
            sip_scores=sip_scores
        )

        # Map filtered signals back to original index
        # (signals only exist during market hours, rest are False)
        long_entries.loc[filtered_long_entries.index] = filtered_long_entries
        long_exits.loc[filtered_long_exits.index] = filtered_long_exits
        short_entries.loc[filtered_short_entries.index] = filtered_short_entries
        short_exits.loc[filtered_short_exits.index] = filtered_short_exits

        return long_entries, long_exits, short_entries, short_exits

    def _generate_signals_python(
        self,
        data: pd.DataFrame,
        gap_df: pd.DataFrame,
        sip_scores: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Pure Python signal generation.

        This implementation is clear and correct. Can be optimized with Numba later.
        """
        n = len(data)
        long_entries = pd.Series(False, index=data.index)
        long_exits = pd.Series(False, index=data.index)
        short_entries = pd.Series(False, index=data.index)
        short_exits = pd.Series(False, index=data.index)

        # State tracking
        position: Optional[HVORBPosition] = None
        current_date = None
        daily_or: Dict[str, float] = {}
        daily_trades = 0
        daily_pnl = 0.0

        # Pullback entry tracking: {date: {setup_info}}
        pending_pullback: Optional[Dict] = None

        # Process bar by bar
        for i, (idx, row) in enumerate(data.iterrows()):
            bar_date = row['date']
            bar_time = row['time']

            # Reset daily tracking on new day
            if bar_date != current_date:
                # Force exit any open position from previous day (missed EOD)
                if position is not None:
                    if position.direction == 'long':
                        long_exits.iloc[i] = True
                    else:
                        short_exits.iloc[i] = True
                    position = None

                current_date = bar_date
                daily_or = {}
                daily_trades = 0
                daily_pnl = 0.0
                pending_pullback = None  # Reset pending pullback setup on new day

            # Calculate opening range after OR period ends
            if bar_time >= self.or_end_time and not daily_or:
                day_data = data[data['date'] == bar_date]
                daily_or = HVORBIndicators.opening_range(
                    day_data,
                    start_time=self.MARKET_OPEN,
                    end_time=self.or_end_time
                )

                if not daily_or:
                    continue

            # Skip if no OR calculated yet
            if not daily_or:
                continue

            # Check exits first if in position
            if position is not None:
                should_exit, exit_type = self._check_exit_conditions(
                    position=position,
                    current_high=row['high'],
                    current_low=row['low'],
                    current_close=row['close'],
                    current_time=bar_time,
                    current_bar_idx=i
                )

                if should_exit:
                    if position.direction == 'long':
                        long_exits.iloc[i] = True
                    else:
                        short_exits.iloc[i] = True

                    # Track PnL
                    if position.direction == 'long':
                        pnl = (row['close'] - position.entry_price) * position.shares
                    else:
                        pnl = (position.entry_price - row['close']) * position.shares
                    daily_pnl += pnl

                    position = None
                    continue

            # Check entry conditions if no position
            if position is None and bar_time >= self.or_end_time:
                # Check daily risk limits
                if not self._check_risk_limits(daily_trades, daily_pnl):
                    continue

                # Check entry cutoff
                if bar_time >= self.entry_cutoff_time:
                    pending_pullback = None  # Cancel pending setup after cutoff
                    continue

                # Get gap info for today
                gap_info = {}
                if bar_date in gap_df.index:
                    gap_info = gap_df.loc[bar_date].to_dict()

                # Get SIP score for today
                sip_score = sip_scores.get(bar_date, 1.0)
                current_symbol = getattr(self, '_current_symbol', '')

                # === PULLBACK ENTRY MODE ===
                if self.use_pullback_entry:
                    # Check if we have a pending pullback setup
                    if pending_pullback is not None:
                        # Increment bar count
                        pending_pullback['bars_waited'] += 1

                        # Check for timeout
                        if pending_pullback['bars_waited'] > self.pullback_timeout_bars:
                            pending_pullback = None
                            continue

                        # Check for pullback entry condition
                        or_high = pending_pullback['or_high']
                        or_low = pending_pullback['or_low']
                        direction = pending_pullback['direction']

                        current_close = row['close']
                        current_low = row['low']
                        current_open = row['open']

                        if direction == 'long':
                            # For long: price pulled back near OR high and is bouncing
                            pullback_level = or_high * (1 + self.pullback_threshold_pct / 100)
                            touched_pullback = current_low <= pullback_level
                            is_bouncing = current_close > current_open  # Green candle
                            still_above_or = current_close > or_high

                            if touched_pullback and is_bouncing and still_above_or:
                                # Valid pullback entry!
                                position = self._create_position_from_setup(
                                    pending_pullback, row, current_symbol
                                )
                                if position is not None:
                                    daily_trades += 1
                                    long_entries.iloc[i] = True
                                    pending_pullback = None
                        else:
                            # For short: price pulled back near OR low and is dropping
                            pullback_level = or_low * (1 - self.pullback_threshold_pct / 100)
                            touched_pullback = row['high'] >= pullback_level
                            is_dropping = current_close < current_open  # Red candle
                            still_below_or = current_close < or_low

                            if touched_pullback and is_dropping and still_below_or:
                                position = self._create_position_from_setup(
                                    pending_pullback, row, current_symbol
                                )
                                if position is not None:
                                    daily_trades += 1
                                    short_entries.iloc[i] = True
                                    pending_pullback = None

                    # No pending setup - look for breakout to create one
                    elif pending_pullback is None:
                        entry_result = self._check_entry_conditions(
                            row=row,
                            prev_close=data['close'].iloc[i-1] if i > 0 else row['close'],
                            daily_or=daily_or,
                            gap_info=gap_info,
                            sip_score=sip_score,
                            bar_idx=i,
                            symbol=current_symbol
                        )

                        if entry_result is not None:
                            # Don't enter immediately - create pending setup
                            pending_pullback = {
                                'or_high': daily_or.get('or_high', 0),
                                'or_low': daily_or.get('or_low', 0),
                                'direction': entry_result.direction,
                                'breakout_price': row['close'],
                                'gap_info': gap_info,
                                'sip_score': sip_score,
                                'bars_waited': 0,
                                'entry_result': entry_result  # Store for position creation
                            }

                # === IMMEDIATE ENTRY MODE (original behavior) ===
                else:
                    entry_result = self._check_entry_conditions(
                        row=row,
                        prev_close=data['close'].iloc[i-1] if i > 0 else row['close'],
                        daily_or=daily_or,
                        gap_info=gap_info,
                        sip_score=sip_score,
                        bar_idx=i,
                        symbol=current_symbol
                    )

                    if entry_result is not None:
                        position = entry_result
                        daily_trades += 1

                        if position.direction == 'long':
                            long_entries.iloc[i] = True
                        else:
                            short_entries.iloc[i] = True

        return long_entries, long_exits, short_entries, short_exits

    def _create_position_from_setup(
        self,
        setup: Dict,
        row: pd.Series,
        symbol: str
    ) -> Optional[HVORBPosition]:
        """Create a position from a pending pullback setup at current price."""
        original_entry = setup.get('entry_result')
        if original_entry is None:
            return None

        # Update entry price to current price (pullback entry)
        entry_price = row['close']
        or_high = setup['or_high']
        or_low = setup['or_low']
        direction = setup['direction']

        # Recalculate targets based on new entry price
        targets = HVORBIndicators.calculate_tiered_targets(
            entry_price=entry_price,
            or_high=or_high,
            or_low=or_low,
            direction=direction,
            target1_multiplier=self.target1_multiplier,
            target2_multiplier=self.target2_multiplier
        )

        return HVORBPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_bar_idx=original_entry.entry_bar_idx,  # Keep original bar idx
            entry_time=row.name if hasattr(row.name, 'hour') else original_entry.entry_time,
            shares=original_entry.shares,
            stop_loss=targets['stop_loss'],
            target1=targets['target1'],
            target2=targets['target2'],
            or_height=original_entry.or_height,
            sip_score=original_entry.sip_score,
            gap_pct=original_entry.gap_pct,
            confidence=original_entry.confidence
        )

    def _check_entry_conditions(
        self,
        row: pd.Series,
        prev_close: float,
        daily_or: Dict[str, float],
        gap_info: Dict[str, any],
        sip_score: float,
        bar_idx: int,
        symbol: str = ''
    ) -> Optional[HVORBPosition]:
        """
        Check all entry conditions and return position if valid.

        Entry requires:
        1. SIP score >= sip_min_score
        2. Valid gap (min_gap_pct <= gap <= max_gap_pct)
        3. Volatility within bounds (min_atr_pct <= atr% <= max_atr_pct)
        4. Breakout from opening range
        5. Direction alignment (gap direction matches breakout)
        6. RVOL confirmation (optional)
        7. Not an earnings day (optional, when skip_earnings_days=True)
        8. Sentiment alignment (optional, when use_sentiment=True)
        9. Regime alignment (optional, when use_regime=True)
        """
        or_high = daily_or.get('or_high', 0)
        or_low = daily_or.get('or_low', 0)
        or_height = daily_or.get('or_height', 0)

        if or_height <= 0:
            return None

        current_close = row['close']
        current_high = row['high']
        current_low = row['low']
        atr_pct = row.get('atr_pct', 0)
        rvol = row.get('rvol', 1.0)

        # 1. Check SIP score
        if sip_score < self.sip_min_score:
            return None

        # 2. Check gap validity
        gap_pct = gap_info.get('gap_pct', 0)
        gap_direction = gap_info.get('gap_direction', 'flat')
        is_gap_valid = gap_info.get('is_valid', False)

        if not is_gap_valid:
            return None

        # 3. Check volatility bounds
        if atr_pct < self.min_atr_pct or atr_pct > self.max_atr_pct:
            return None

        # 4. Check for breakout
        prev_inside = or_low <= prev_close <= or_high
        breakout_direction = None

        if current_close > or_high and prev_inside:
            breakout_direction = 'long'
        elif current_close < or_low and prev_inside and not self.long_only:
            breakout_direction = 'short'

        if breakout_direction is None:
            return None

        # 5. Direction alignment (gap matches breakout)
        if self.gap_direction_filter:
            if breakout_direction == 'long' and gap_direction != 'up':
                return None
            if breakout_direction == 'short' and gap_direction != 'down':
                return None

        # 6. RVOL confirmation (optional)
        if self.rvol_threshold > 0 and rvol < self.rvol_threshold:
            return None

        # 7. Earnings day filter (optional)
        if self.skip_earnings_days and symbol:
            bar_date = row.name if isinstance(row.name, datetime) else row.get('date')
            if self._is_earnings_day(symbol, bar_date):
                return None

        # 8. Sentiment filter (optional)
        sentiment_score = 0.0
        if self.use_sentiment and symbol:
            bar_date = row.name if isinstance(row.name, datetime) else row.get('date')
            passes_sentiment, sentiment_score = self.check_sentiment_filter(
                symbol=symbol,
                date=bar_date,
                direction=breakout_direction
            )
            if not passes_sentiment:
                return None

        # 9. Regime filter (optional)
        if self.use_regime and self.regime_detector:
            if breakout_direction == 'long' and self.current_regime == 'BEAR':
                return None
            if breakout_direction == 'short' and self.current_regime == 'STRONG_BULL':
                return None

        # 10. Momentum filter (optional) - price must be above MA for longs
        if self.use_momentum_filter and symbol:
            ma_value = self._get_daily_ma(symbol, row)
            if ma_value is not None and ma_value > 0:
                if breakout_direction == 'long' and current_close < ma_value:
                    return None
                if breakout_direction == 'short' and current_close > ma_value:
                    return None

        # All filters passed - create position
        entry_price = current_close

        # Calculate exit levels
        targets = HVORBIndicators.calculate_tiered_targets(
            entry_price=entry_price,
            or_high=or_high,
            or_low=or_low,
            direction=breakout_direction,
            target1_multiplier=self.target1_multiplier,
            target2_multiplier=self.target2_multiplier
        )

        # Calculate confidence score with sentiment
        confidence = HVORBIndicators.confidence_score(
            sip_score=sip_score,
            sentiment_score=sentiment_score,
            gap_pct=gap_pct,
            volume_ratio=rvol,
            direction=breakout_direction
        )

        return HVORBPosition(
            symbol=symbol,
            direction=breakout_direction,
            entry_price=entry_price,
            entry_bar_idx=bar_idx,
            entry_time=row.name if isinstance(row.name, datetime) else datetime.now(),
            shares=1,  # Normalized, actual sizing done by position sizer
            stop_loss=targets['stop_loss'],
            target1=targets['target1'],
            target2=targets['target2'],
            or_height=or_height,
            sip_score=sip_score,
            gap_pct=gap_pct,
            confidence=confidence,
            current_stop=targets['stop_loss'],
            max_favorable_price=entry_price
        )

    def _check_exit_conditions(
        self,
        position: HVORBPosition,
        current_high: float,
        current_low: float,
        current_close: float,
        current_time: time,
        current_bar_idx: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check all exit conditions for current position.

        Exit priority:
        1. Stop loss (including trailing stop)
        2. Time-based stop (if not profitable after N minutes)
        3. Target 1 (partial exit - handled externally for actual partial)
        4. Target 2
        5. EOD exit
        """
        # Update max favorable price
        if position.direction == 'long':
            position.max_favorable_price = max(position.max_favorable_price, current_high)
        else:
            position.max_favorable_price = min(position.max_favorable_price, current_low)

        # Update trailing stop if active
        if position.trailing_active:
            if position.direction == 'long':
                new_stop = position.max_favorable_price * (1 - self.trailing_offset_pct)
                position.current_stop = max(position.current_stop, new_stop)
            else:
                new_stop = position.max_favorable_price * (1 + self.trailing_offset_pct)
                position.current_stop = min(position.current_stop, new_stop)

        # 1. Check stop loss
        if position.direction == 'long':
            if current_low <= position.current_stop:
                return True, 'stop_loss' if not position.trailing_active else 'trailing'
        else:
            if current_high >= position.current_stop:
                return True, 'stop_loss' if not position.trailing_active else 'trailing'

        # 2. Check time-based stop (exit if not profitable after N minutes)
        if self.use_time_stop:
            bars_held = current_bar_idx - position.entry_bar_idx
            # Each bar is 1 minute
            if bars_held >= self.time_stop_minutes:
                # Check if trade is profitable
                if position.direction == 'long':
                    is_profitable = current_close > position.entry_price
                else:
                    is_profitable = current_close < position.entry_price

                if not is_profitable:
                    return True, 'time_stop'

        # 3. Check target 1 (for tracking - full exit here, partial in real impl)
        if not position.target1_hit:
            if position.direction == 'long' and current_high >= position.target1:
                position.target1_hit = True
                # Move stop to breakeven + buffer
                buffer = position.or_height * 0.1
                position.current_stop = position.entry_price + buffer
                # In backtesting, we exit fully at T1 for simplicity
                # Real impl would do partial exit
                return True, 'target1'
            elif position.direction == 'short' and current_low <= position.target1:
                position.target1_hit = True
                buffer = position.or_height * 0.1
                position.current_stop = position.entry_price - buffer
                return True, 'target1'

        # 4. Check target 2
        if position.target1_hit:
            if position.direction == 'long' and current_high >= position.target2:
                return True, 'target2'
            elif position.direction == 'short' and current_low <= position.target2:
                return True, 'target2'

        # 5. Check EOD exit
        if current_time >= self.eod_exit_time:
            return True, 'eod'

        return False, None

    # Earnings-related keywords in headlines
    EARNINGS_KEYWORDS = [
        'earnings',
        'quarterly results',
        'q1 results', 'q2 results', 'q3 results', 'q4 results',
        'beat estimates',
        'miss estimates',
        'beats expectations',
        'misses expectations',
        'eps of',
        'revenue of',
        'reports profit',
        'reports loss',
        'fiscal quarter',
        'guidance',
    ]

    def _is_earnings_day(self, symbol: str, date: datetime) -> bool:
        """
        Detect if this is likely an earnings day based on news headlines.

        If 2+ headlines contain earnings-related keywords, skip the day.
        Earnings days have high volatility but are unpredictable for ORB.

        Args:
            symbol: Stock symbol
            date: Trading date

        Returns:
            True if this appears to be an earnings day
        """
        if self._sentiment_data is None:
            return False

        try:
            target_date = date.date() if hasattr(date, 'date') else date

            # Filter news for this symbol and date
            mask = (
                (self._sentiment_data['symbol'] == symbol) &
                (self._sentiment_data['timestamp'].dt.date == target_date)
            )
            day_news = self._sentiment_data[mask]

            if day_news.empty or 'headline' not in day_news.columns:
                return False

            # Count headlines with earnings keywords
            earnings_count = 0
            for headline in day_news['headline'].dropna():
                headline_lower = headline.lower()
                for keyword in self.EARNINGS_KEYWORDS:
                    if keyword in headline_lower:
                        earnings_count += 1
                        break  # Only count once per headline

            # If 2+ earnings-related headlines, it's likely earnings day
            return earnings_count >= 2

        except Exception as e:
            logger.debug(f"Error checking earnings day for {symbol}/{date}: {e}")
            return False

    def _check_risk_limits(
        self,
        daily_trades: int,
        daily_pnl: float,
        capital: float = 100000.0
    ) -> bool:
        """Check if we can take a new trade based on risk limits."""
        # Check trade count
        if daily_trades >= self.max_daily_trades:
            return False

        # Check daily loss
        max_loss = capital * self.daily_max_loss_pct
        if daily_pnl <= -max_loss:
            return False

        return True

    def load_sentiment_data(self, path: Optional[str] = None, symbol: Optional[str] = None) -> None:
        """
        Load pre-computed sentiment data for backtesting.

        Can load from:
        1. Explicit parquet file path
        2. SentimentCache for a specific symbol

        Args:
            path: Path to sentiment parquet file (optional)
            symbol: Symbol to load from SentimentCache (optional)
        """
        if path:
            try:
                self._sentiment_data = pd.read_parquet(path)
                logger.info(f"Loaded sentiment data from {path}")
            except Exception as e:
                logger.error(f"Failed to load sentiment data: {e}")
                self._sentiment_data = None
        elif symbol and self.use_sentiment:
            try:
                from src.data.news.sentiment_cache import SentimentCache

                cache = SentimentCache()
                # Load all available years for the symbol
                available = cache.list_available_news()
                symbol_data = [x for x in available if x['symbol'] == symbol and x['has_sentiment']]

                if symbol_data:
                    dfs = []
                    for item in symbol_data:
                        try:
                            df = cache.load(item['symbol'], item['year'])
                            dfs.append(df)
                        except Exception:
                            pass

                    if dfs:
                        self._sentiment_data = pd.concat(dfs, ignore_index=True)
                        logger.info(f"Loaded {len(self._sentiment_data)} sentiment records for {symbol}")
                    else:
                        logger.warning(f"No sentiment data found for {symbol}")
            except ImportError:
                logger.debug("SentimentCache not available")
            except Exception as e:
                logger.error(f"Failed to load sentiment from cache: {e}")

    def load_sentiment_for_symbols(self, symbols: List[str], start_date: datetime, end_date: datetime) -> None:
        """
        Load sentiment data for multiple symbols and a date range.

        Args:
            symbols: List of stock symbols
            start_date: Start date for sentiment data
            end_date: End date for sentiment data
        """
        if not self.use_sentiment:
            return

        try:
            from src.data.news.sentiment_cache import SentimentCache

            cache = SentimentCache()
            dfs = []

            for symbol in symbols:
                try:
                    df = cache.load_range(symbol, start_date, end_date)
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logger.debug(f"No sentiment data for {symbol}: {e}")

            if dfs:
                self._sentiment_data = pd.concat(dfs, ignore_index=True)
                logger.info(f"Loaded {len(self._sentiment_data)} sentiment records for {len(symbols)} symbols")
            else:
                logger.warning("No sentiment data found for any symbols")
                self._sentiment_data = None

        except ImportError:
            logger.warning("SentimentCache not available - install transformers and torch")
            self._sentiment_data = None
        except Exception as e:
            logger.error(f"Failed to load sentiment data: {e}")
            self._sentiment_data = None

    def get_sentiment_for_date(
        self,
        symbol: str,
        date: datetime
    ) -> float:
        """
        Get aggregated sentiment score for a symbol on a date.

        Args:
            symbol: Stock symbol
            date: Trading date

        Returns:
            Sentiment score from -1 (bearish) to +1 (bullish)
        """
        if self._sentiment_data is None or not self.use_sentiment:
            return 0.0

        try:
            # Handle both datetime objects and date objects
            target_date = date.date() if hasattr(date, 'date') else date

            # Filter sentiment data for symbol and date
            mask = (
                (self._sentiment_data['symbol'] == symbol) &
                (self._sentiment_data['timestamp'].dt.date == target_date)
            )
            day_sentiment = self._sentiment_data[mask]

            if day_sentiment.empty:
                return 0.0

            # Filter out spam headlines
            if 'headline' in day_sentiment.columns:
                spam_mask = day_sentiment['headline'].apply(self._is_spam_headline)
                day_sentiment = day_sentiment[~spam_mask]
                if day_sentiment.empty:
                    # Only spam news, treat as neutral
                    return 0.0

            # Filter by confidence if threshold set
            if self.min_sentiment_confidence > 0 and 'confidence' in day_sentiment.columns:
                day_sentiment = day_sentiment[day_sentiment['confidence'] >= self.min_sentiment_confidence]
                if day_sentiment.empty:
                    # No high-confidence sentiment, treat as neutral
                    return 0.0

            # Return mean sentiment score
            return float(day_sentiment['sentiment_score'].mean())
        except Exception as e:
            logger.debug(f"Error getting sentiment for {symbol}/{date}: {e}")
            return 0.0

    def check_sentiment_filter(
        self,
        symbol: str,
        date: datetime,
        direction: str
    ) -> Tuple[bool, float]:
        """
        Check if sentiment passes the filter for entry.

        Args:
            symbol: Stock symbol
            date: Trading date
            direction: 'long' or 'short'

        Returns:
            Tuple of (passes_filter, sentiment_score)
        """
        if not self.use_sentiment:
            return True, 0.0

        sentiment_score = self.get_sentiment_for_date(symbol, date)

        # For longs, we want positive sentiment (>= threshold)
        if direction == 'long':
            passes = sentiment_score >= self.min_sentiment_score
        # For shorts, we want negative sentiment (<= -threshold)
        else:
            passes = sentiment_score <= -self.min_sentiment_score

        return passes, sentiment_score

    def _get_daily_ma(self, symbol: str, row: pd.Series) -> Optional[float]:
        """
        Get the daily moving average for momentum filter.

        Calculates MA from daily close prices. Uses cached values when available.

        Args:
            symbol: Stock symbol
            row: Current bar data with 'close' price

        Returns:
            MA value or None if not enough data
        """
        # Get the date from the row
        if isinstance(row.name, datetime):
            current_date = row.name.date() if hasattr(row.name, 'date') else row.name
        else:
            current_date = row.get('date')

        if current_date is None:
            return None

        # Check cache
        cache_key = f"{symbol}_{current_date}"
        if cache_key in self._daily_ma:
            return self._daily_ma[cache_key]

        # Calculate from daily data if available
        # For intraday data, we use the close prices aggregated by day
        # The MA is calculated on previous days only (no lookahead)
        try:
            # Use SMA of recent daily closes from the data we have
            # This approximates a daily MA from intraday data
            data = getattr(self, '_current_data', None)
            if data is None or data.empty:
                return None

            # Get daily closes (last close of each day)
            if 'date' in data.columns:
                daily_closes = data.groupby('date')['close'].last()
            else:
                # Infer date from index
                if hasattr(data.index, 'date'):
                    data_copy = data.copy()
                    data_copy['_date'] = data.index.date
                    daily_closes = data_copy.groupby('_date')['close'].last()
                else:
                    return None

            # Filter to dates before current
            daily_closes = daily_closes[daily_closes.index < current_date]

            if len(daily_closes) < self.momentum_ma_period:
                return None

            # Calculate SMA
            ma_value = float(daily_closes.tail(self.momentum_ma_period).mean())

            # Cache it
            self._daily_ma[cache_key] = ma_value
            return ma_value

        except Exception:
            return None

    # Spam patterns to filter from sentiment analysis
    SPAM_HEADLINE_PATTERNS = [
        'whale alert',
        'wsb',
        'wallstreetbets',
        'trending on reddit',
        'trending stock',
        'most talked about',
        'unusual options',
        'dark pool',
        'insider buying',
        'insider selling',
        'short squeeze',
        'to the moon',
        'diamond hands',
        'rocket emoji',
        'meme stock',
    ]

    def _is_spam_headline(self, headline: str) -> bool:
        """
        Check if a headline is spam/low-quality.

        Filters out:
        - Whale alerts
        - WSB/Reddit trending
        - Generic aggregator content
        - Meme stock hype

        Args:
            headline: News headline text

        Returns:
            True if headline should be filtered out
        """
        if not headline:
            return True

        headline_lower = headline.lower()

        for pattern in self.SPAM_HEADLINE_PATTERNS:
            if pattern in headline_lower:
                return True

        return False
