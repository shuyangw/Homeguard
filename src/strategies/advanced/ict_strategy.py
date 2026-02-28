"""
ICT/SMC (Smart Money Concepts) Liquidity-Based Trading Strategy.

A filtered intraday strategy that trades liquidity sweeps at unmitigated
order blocks with switch candle confirmation for reversal/continuation entries.

Entry Filters:
- Market structure alignment (HH/HL for longs, LH/LL for shorts)
- Unmitigated order block at entry zone
- Liquidity sweep confirmation
- Switch candle pattern (hammer, engulfing)
- HTF bias alignment (optional)
- Regime filter (optional)

Exit Logic:
- Stop-loss at switch candle wick (with ATR buffer)
- Target at risk:reward ratio
- Time exit at 3:45 PM ET (no overnight exposure)

Two Variants:
- Reversal: External liquidity sweep at HTF zone, counter-trend entry
- Continuation: Pullback to internal liquidity, trend-following entry
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np

from src.backtesting.base.strategy import LongShortStrategy
from src.backtesting.utils.indicators import Indicators
from src.strategies.advanced.ict_indicators import (
    ICTIndicators,
    SwingPoint,
    OrderBlock,
    LiquiditySweep,
    SwitchCandle
)
from src.strategies.advanced.exit_checker import check_exit, ExitReason
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger(__name__)


@dataclass
class ICTPosition:
    """Represents an active ICT position."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    stop_loss: float
    target: float
    order_block: Optional[OrderBlock]
    trade_type: str  # 'reversal' or 'continuation'
    # Optional fields for vectorized mode (not always available)
    liquidity_sweep: Optional[LiquiditySweep] = None
    switch_candle: Optional[SwitchCandle] = None
    entry_bar_idx: int = 0  # Bar index at entry for time stop calculation
    # Trailing stop state
    initial_stop_loss: float = 0.0  # Original stop for R calculation
    trailing_active: bool = False  # Whether trailing has been triggered
    max_favorable_price: float = 0.0  # Best price achieved (for trailing)


class ICTStrategy(LongShortStrategy):
    """
    ICT/SMC Liquidity-Based Trading Strategy.

    Trades liquidity sweeps at unmitigated order blocks with
    switch candle confirmation for reversal/continuation entries.

    Two variants:
    - Reversal: External liquidity sweep at unmitigated zone
    - Continuation: Internal liquidity sweep with trend

    Multi-timeframe:
    - HTF (15m/1H): Trend direction and key levels
    - LTF (1m/5m): Entry timing and patterns

    Parameters:
        trade_type: 'reversal', 'continuation', or 'both'
        swing_lookback: Bars for swing point detection (default 5)
        min_swing_size_pct: Minimum swing size % (default 0.2%)
        min_impulse_move_pct: Minimum impulse for OB (default 1%)
        order_block_max_age: Max bars for OB validity (default 50)
        risk_reward_ratio: Target RR ratio (default 2.0)
        use_htf_filter: Require HTF alignment (default True)
        htf_timeframe: HTF timeframe string (default '15min')
        use_regime: Enable regime filtering (default True)
        long_only: Only take long trades (default False)
        exit_time_hour/minute: Force exit time (default 15:45)
        max_positions_per_day: Limit daily trades (default 3)
    """

    # Market timing constants (Eastern Time)
    MARKET_OPEN = time(9, 30)
    ENTRY_START = time(9, 45)  # Allow entries after first 15 minutes
    ENTRY_CUTOFF = time(15, 30)  # No new entries after 3:30 PM
    EXIT_TIME = time(15, 45)  # Force exit at 3:45 PM
    MARKET_CLOSE = time(16, 0)

    def __init__(
        self,
        trade_type: str = 'both',
        swing_lookback: int = 5,
        min_swing_size_pct: float = 0.002,
        min_impulse_move_pct: float = 0.01,
        order_block_max_age: int = 50,
        impulse_bars: int = 15,
        risk_reward_ratio: float = 2.0,
        use_htf_filter: bool = True,
        htf_lookback: int = 20,
        use_regime: bool = True,
        long_only: bool = False,
        max_positions_per_day: int = 3,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        exit_time_hour: int = 15,
        exit_time_minute: int = 45,
        min_wick_ratio: float = 0.5,
        min_body_ratio: float = 0.3,
        sweep_threshold_pct: float = 0.001,
        # New optional features (disabled by default to avoid overcomplication)
        use_atr_impulse: bool = False,
        atr_impulse_multiple: float = 1.5,
        use_volume_filter: bool = False,
        rvol_threshold: float = 1.5,
        session_filter: str = 'none',
        use_zone_quality: bool = False,
        min_zone_quality: float = 0.5,
        # Risk management: time stop and max loss cap
        max_hold_bars: int = 0,  # Max bars to hold position (0 = disabled). 3900 = ~10 trading days
        max_loss_pct: float = 0.0,  # Max loss % before forced exit (0 = disabled). 0.15 = 15%
        # NEW: Signal quality filters for improved win rate
        min_sweep_depth_pct: float = 0.0,  # Min sweep depth % (0.002 = 0.2%)
        use_momentum_filter: bool = False,  # Require momentum alignment (non-ICT)
        momentum_ema_period: int = 10,  # EMA period for momentum
        # ICT-aligned filters
        use_structure_filter: bool = False,  # Only trade with market structure (pure ICT)
        require_ob_confluence: bool = False,  # Require nearby unmitigated OB for entry
        # NEW: Trailing stop for locking in profits
        use_trailing_stop: bool = False,  # Enable trailing stop after hitting profit target
        trailing_trigger_r: float = 1.0,  # R multiple to trigger trailing (1.0 = 1R)
        trailing_offset_r: float = 0.5,  # Trail behind by this many R (0.5 = lock in 0.5R)
        **kwargs
    ):
        # Set attributes BEFORE calling super().__init__()
        # because parent class calls validate_parameters()
        self.trade_type = trade_type
        self.swing_lookback = swing_lookback
        self.min_swing_size_pct = min_swing_size_pct
        self.min_impulse_move_pct = min_impulse_move_pct
        self.order_block_max_age = order_block_max_age
        self.impulse_bars = impulse_bars
        self.risk_reward_ratio = risk_reward_ratio
        self.use_htf_filter = use_htf_filter
        self.htf_lookback = htf_lookback
        self.use_regime = use_regime
        self.long_only = long_only
        self.max_positions_per_day = max_positions_per_day
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.exit_time = time(exit_time_hour, exit_time_minute)
        self.min_wick_ratio = min_wick_ratio
        self.min_body_ratio = min_body_ratio
        self.sweep_threshold_pct = sweep_threshold_pct

        # New optional features (from original spec)
        self.use_atr_impulse = use_atr_impulse
        self.atr_impulse_multiple = atr_impulse_multiple
        self.use_volume_filter = use_volume_filter
        self.rvol_threshold = rvol_threshold
        self.session_filter = session_filter
        self.use_zone_quality = use_zone_quality
        self.min_zone_quality = min_zone_quality

        # Risk management: time stop and max loss cap
        self.max_hold_bars = max_hold_bars
        self.max_loss_pct = max_loss_pct

        # NEW: Signal quality filters
        self.min_sweep_depth_pct = min_sweep_depth_pct
        self.use_momentum_filter = use_momentum_filter
        self.momentum_ema_period = momentum_ema_period
        # ICT-aligned filters
        self.use_structure_filter = use_structure_filter
        self.require_ob_confluence = require_ob_confluence

        # NEW: Trailing stop
        self.use_trailing_stop = use_trailing_stop
        self.trailing_trigger_r = trailing_trigger_r
        self.trailing_offset_r = trailing_offset_r

        # Regime detector (optional)
        self.regime_detector = MarketRegimeDetector() if use_regime else None
        self.current_regime = 'SIDEWAYS'

        # State tracking (reset per backtest run)
        self._swing_points: List[SwingPoint] = []
        self._order_blocks: List[OrderBlock] = []
        self._liquidity_levels: Dict[str, List[float]] = {}
        self._positions: List[ICTPosition] = []
        self._daily_trades: Dict[str, int] = {}  # date -> trade count

        super().__init__(
            trade_type=trade_type,
            swing_lookback=swing_lookback,
            min_swing_size_pct=min_swing_size_pct,
            min_impulse_move_pct=min_impulse_move_pct,
            order_block_max_age=order_block_max_age,
            impulse_bars=impulse_bars,
            risk_reward_ratio=risk_reward_ratio,
            use_htf_filter=use_htf_filter,
            htf_lookback=htf_lookback,
            use_regime=use_regime,
            long_only=long_only,
            max_positions_per_day=max_positions_per_day,
            atr_period=atr_period,
            atr_stop_multiplier=atr_stop_multiplier,
            exit_time_hour=exit_time_hour,
            exit_time_minute=exit_time_minute,
            min_wick_ratio=min_wick_ratio,
            min_body_ratio=min_body_ratio,
            sweep_threshold_pct=sweep_threshold_pct,
            # New optional features
            use_atr_impulse=use_atr_impulse,
            atr_impulse_multiple=atr_impulse_multiple,
            use_volume_filter=use_volume_filter,
            rvol_threshold=rvol_threshold,
            session_filter=session_filter,
            use_zone_quality=use_zone_quality,
            min_zone_quality=min_zone_quality,
            # Risk management
            max_hold_bars=max_hold_bars,
            max_loss_pct=max_loss_pct,
            # NEW: Signal quality filters
            min_sweep_depth_pct=min_sweep_depth_pct,
            use_momentum_filter=use_momentum_filter,
            momentum_ema_period=momentum_ema_period,
            # NEW: Trailing stop
            use_trailing_stop=use_trailing_stop,
            trailing_trigger_r=trailing_trigger_r,
            trailing_offset_r=trailing_offset_r,
            **kwargs
        )

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.trade_type not in ['reversal', 'continuation', 'both']:
            raise ValueError(
                f"trade_type must be 'reversal', 'continuation', or 'both', "
                f"got '{self.trade_type}'"
            )

        if self.swing_lookback < 2:
            raise ValueError(f"swing_lookback must be >= 2, got {self.swing_lookback}")

        if self.risk_reward_ratio <= 0:
            raise ValueError(
                f"risk_reward_ratio must be > 0, got {self.risk_reward_ratio}"
            )

        if self.min_swing_size_pct < 0:
            raise ValueError(
                f"min_swing_size_pct must be >= 0, got {self.min_swing_size_pct}"
            )

    def generate_long_short_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long and short entry/exit signals.

        VECTORIZED IMPLEMENTATION:
        Pre-computes all indicators in bulk, then uses fast array lookups
        in the sequential signal generation loop.

        Signal generation logic:
        1. Detect swing points and classify market structure
        2. Identify unmitigated order blocks
        3. Map liquidity pools
        4. Detect liquidity sweeps (VECTORIZED)
        5. Confirm with switch candle (VECTORIZED)
        6. Check HTF alignment (optional)
        7. Apply regime filter (optional)

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

        # Initialize signal arrays (use numpy for speed)
        long_entries_arr = np.zeros(n, dtype=bool)
        long_exits_arr = np.zeros(n, dtype=bool)
        short_entries_arr = np.zeros(n, dtype=bool)
        short_exits_arr = np.zeros(n, dtype=bool)

        # =====================================================================
        # PHASE 1: PRE-COMPUTE ALL INDICATORS (VECTORIZED)
        # =====================================================================

        # Calculate ATR for stop sizing
        atr = ICTIndicators.calculate_atr(data, period=self.atr_period)
        atr_values = atr.values

        # Detect swing points for entire dataset
        self._swing_points = ICTIndicators.detect_swing_points(
            data,
            lookback=self.swing_lookback,
            min_swing_size_pct=self.min_swing_size_pct
        )

        # Identify order blocks (with optional ATR-based impulse detection)
        self._order_blocks = ICTIndicators.identify_order_blocks(
            data,
            self._swing_points,
            min_impulse_move_pct=self.min_impulse_move_pct,
            max_age_bars=self.order_block_max_age,
            impulse_bars=self.impulse_bars,
            use_atr_impulse=self.use_atr_impulse,
            atr_impulse_multiple=self.atr_impulse_multiple,
            atr_period=self.atr_period
        )

        # Map liquidity levels
        # NOTE: POTENTIAL LOOKAHEAD - liquidity levels are computed from ALL swing points
        # including future ones. A proper fix would filter by swing.available_index.
        # Impact is limited since sweep detection still uses real price action,
        # but the specific level reference may include not-yet-confirmed swings.
        self._liquidity_levels = ICTIndicators.identify_liquidity_levels(
            self._swing_points
        )

        # Get market structure
        # NOTE: POTENTIAL LOOKAHEAD - market structure uses ALL swing points.
        # A proper fix would compute per-bar using only available_index <= i.
        market_structure, _ = ICTIndicators.classify_market_structure(
            self._swing_points
        )

        # Get HTF bias if enabled
        # NOTE: POTENTIAL LOOKAHEAD - HTF bias uses ALL swing points.
        htf_bias = 'neutral'
        if self.use_htf_filter:
            htf_bias, _ = ICTIndicators.get_htf_bias(
                data,
                self._swing_points,
                lookback=self.htf_lookback
            )

        # PRE-COMPUTE: Liquidity sweeps (VECTORIZED - replaces per-bar calls)
        # Now includes sweep depth for quality filtering
        bullish_sweeps, bearish_sweeps, bullish_confirmed, bearish_confirmed, \
            bullish_sweep_depth, bearish_sweep_depth = \
            ICTIndicators.detect_liquidity_sweeps_vectorized(
                data,
                self._liquidity_levels,
                sweep_threshold_pct=self.sweep_threshold_pct,
                min_sweep_depth_pct=self.min_sweep_depth_pct
            )

        # PRE-COMPUTE: Switch candle patterns (VECTORIZED)
        switch_long, switch_short = ICTIndicators.detect_switch_candles_vectorized(
            data,
            min_wick_ratio=self.min_wick_ratio,
            min_body_ratio=self.min_body_ratio
        )

        # PRE-COMPUTE: Order block mitigation (fixes O(n^2) bug)
        mitigation_map = ICTIndicators.precompute_order_block_mitigation(
            self._order_blocks, data
        )

        # PRE-COMPUTE: Entry window mask
        entry_window_mask = ICTIndicators.compute_entry_window_mask(
            data, self.ENTRY_START, self.ENTRY_CUTOFF
        )

        # PRE-COMPUTE: Volume filter mask (if enabled)
        if self.use_volume_filter:
            volume_mask = ICTIndicators.compute_high_volume_mask(
                data, self.rvol_threshold
            )
        else:
            volume_mask = np.ones(n, dtype=bool)

        # PRE-COMPUTE: Session filter mask (if enabled)
        if self.session_filter != 'none':
            session_mask = np.array([
                ICTIndicators.is_optimal_session(t, self.session_filter)
                for t in data.index
            ])
        else:
            session_mask = np.ones(n, dtype=bool)

        # PRE-COMPUTE: Momentum alignment (if enabled)
        if self.use_momentum_filter:
            bullish_momentum, bearish_momentum = ICTIndicators.compute_momentum_alignment(
                data, ema_period=self.momentum_ema_period
            )
        else:
            bullish_momentum = np.ones(n, dtype=bool)
            bearish_momentum = np.ones(n, dtype=bool)

        # Extract numpy arrays for fast access in loop
        close_values = data['close'].values
        low_values = data['low'].values
        high_values = data['high'].values

        # =====================================================================
        # PHASE 2: SEQUENTIAL SIGNAL GENERATION (stateful, but now O(1) per bar)
        # =====================================================================

        active_position: Optional[ICTPosition] = None
        min_bars = max(self.swing_lookback * 2 + 1, 20)

        for i in range(min_bars, n):
            current_time = data.index[i]
            current_atr = atr_values[i] if i < len(atr_values) else 1.0

            # Get time of day for exit check - CONVERT FROM UTC TO ET
            # Data timestamps are stored in UTC, but exit_time is configured in ET
            if hasattr(current_time, 'time'):
                # Convert UTC timestamp to Eastern Time for proper comparison
                try:
                    et_time = tz.from_utc(current_time.to_pydatetime())
                    bar_time = et_time.time()
                except Exception:
                    # Fallback if conversion fails
                    bar_time = current_time.time()
            else:
                bar_time = time(12, 0)

            # Check for exits first
            just_exited = False
            if active_position is not None:
                should_exit, exit_reason = self._check_exit(
                    active_position,
                    close_values[i],
                    low_values[i],
                    high_values[i],
                    bar_time,
                    current_bar_idx=i
                )

                if should_exit:
                    if active_position.direction == 'long':
                        long_exits_arr[i] = True
                    else:
                        short_exits_arr[i] = True
                    active_position = None
                    just_exited = True  # Prevent re-entry on same bar

            # Skip entry logic if we have active position OR just exited
            # (prevents same-bar re-entry after time stop / max loss)
            if active_position is not None or just_exited:
                continue

            # Fast filter checks using pre-computed masks
            if not entry_window_mask[i]:
                continue

            if not session_mask[i]:
                continue

            # Check daily trade limit
            date_str = str(current_time.date()) if hasattr(current_time, 'date') else str(current_time)[:10]
            daily_count = self._daily_trades.get(date_str, 0)
            if daily_count >= self.max_positions_per_day:
                continue

            # Check for sweep + switch candle pattern using pre-computed arrays
            has_bullish_setup = bullish_confirmed[i] and switch_long[i]
            has_bearish_setup = bearish_confirmed[i] and switch_short[i]

            if not has_bullish_setup and not has_bearish_setup:
                continue

            # Determine trade direction based on setup type and market structure
            direction = None
            trade_type_found = None

            if self.trade_type in ['reversal', 'both']:
                # Reversal: bullish sweep in bearish market or vice versa
                if has_bullish_setup and market_structure != 'bullish':
                    direction = 'long'
                    trade_type_found = 'reversal'
                elif has_bearish_setup and market_structure != 'bearish':
                    direction = 'short'
                    trade_type_found = 'reversal'

            if direction is None and self.trade_type in ['continuation', 'both']:
                # Continuation: sweep in direction of trend
                if has_bullish_setup and market_structure == 'bullish':
                    direction = 'long'
                    trade_type_found = 'continuation'
                elif has_bearish_setup and market_structure == 'bearish':
                    direction = 'short'
                    trade_type_found = 'continuation'

            if direction is None:
                continue

            # Apply filters
            if self.use_regime and not self._passes_regime_filter(direction):
                continue

            if self.long_only and direction == 'short':
                continue

            if self.use_htf_filter and not self._passes_htf_filter(direction, htf_bias):
                continue

            if self.use_volume_filter and not volume_mask[i]:
                continue

            # NEW: Momentum alignment filter - trade with the short-term trend (non-ICT)
            if self.use_momentum_filter:
                if direction == 'long' and not bullish_momentum[i]:
                    continue
                if direction == 'short' and not bearish_momentum[i]:
                    continue

            # ICT-ALIGNED: Market structure filter - only trade with structure
            # This is pure ICT: longs only in bullish structure, shorts only in bearish
            if self.use_structure_filter:
                if direction == 'long' and market_structure != 'bullish':
                    continue
                if direction == 'short' and market_structure != 'bearish':
                    continue

            # Find nearby unmitigated order block for stop placement
            nearby_ob = None
            ob_direction = 'bullish' if direction == 'long' else 'bearish'
            for ob in self._order_blocks:
                if ob.direction != ob_direction:
                    continue
                # LOOKAHEAD FIX: Only use order blocks that are confirmed by current bar
                # OB is only available after impulse move completes (available_index)
                if ob.available_index > i:
                    continue  # OB not yet confirmed - would be lookahead bias
                # Check if mitigated by current bar
                mit_idx = mitigation_map.get(ob.index, -1)
                if mit_idx != -1 and mit_idx <= i:
                    continue  # Already mitigated
                # Check if price is near OB
                if direction == 'long' and low_values[i] <= ob.high * 1.01:
                    nearby_ob = ob
                    break
                elif direction == 'short' and high_values[i] >= ob.low * 0.99:
                    nearby_ob = ob
                    break

            # ICT-ALIGNED: Require order block confluence for entry
            # This is core ICT - sweeps should occur near valid order blocks
            if self.require_ob_confluence and nearby_ob is None:
                continue

            # Calculate stop loss
            if direction == 'long':
                stop_loss = low_values[i] - current_atr * self.atr_stop_multiplier
                if nearby_ob:
                    stop_loss = min(stop_loss, nearby_ob.low - current_atr * 0.25)
            else:
                stop_loss = high_values[i] + current_atr * self.atr_stop_multiplier
                if nearby_ob:
                    stop_loss = max(stop_loss, nearby_ob.high + current_atr * 0.25)

            # Calculate target
            entry_price = close_values[i]
            risk = abs(entry_price - stop_loss)
            if direction == 'long':
                target = entry_price + risk * self.risk_reward_ratio
            else:
                target = entry_price - risk * self.risk_reward_ratio

            # Create position
            position = ICTPosition(
                symbol='',
                direction=direction,
                entry_price=entry_price,
                entry_time=current_time,
                stop_loss=stop_loss,
                target=target,
                order_block=nearby_ob,
                trade_type=trade_type_found,
                entry_bar_idx=i,
                initial_stop_loss=stop_loss,  # Store for R calculation
                trailing_active=False,
                max_favorable_price=entry_price
            )

            # Generate signal
            if direction == 'long':
                long_entries_arr[i] = True
            else:
                short_entries_arr[i] = True

            active_position = position
            self._daily_trades[date_str] = daily_count + 1

        # Convert numpy arrays back to pandas Series
        long_entries = pd.Series(long_entries_arr, index=data.index)
        long_exits = pd.Series(long_exits_arr, index=data.index)
        short_entries = pd.Series(short_entries_arr, index=data.index)
        short_exits = pd.Series(short_exits_arr, index=data.index)

        return long_entries, long_exits, short_entries, short_exits

    def _check_reversal_setup(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        current_bar: pd.Series,
        sweep: Optional[LiquiditySweep],
        market_structure: str,
        htf_bias: str,
        current_atr: float
    ) -> Optional[Tuple[str, ICTPosition]]:
        """
        Check for reversal trade setup.

        Reversal criteria:
        1. Liquidity sweep detected
        2. Switch candle rejection pattern
        3. Unmitigated order block nearby (optional confluence)
        4. Counter to current trend (reversal)

        Returns:
            Tuple of (direction, position) or None
        """
        # Need a liquidity sweep for reversal
        if sweep is None or not sweep.confirmed:
            return None

        # Reversal should be against current trend
        # Bullish sweep in bearish trend = long reversal
        # Bearish sweep in bullish trend = short reversal
        if sweep.direction == 'bullish' and market_structure == 'bullish':
            return None  # Not a reversal, it's continuation
        if sweep.direction == 'bearish' and market_structure == 'bearish':
            return None

        # Look for switch candle confirmation
        # Get nearby order block for confluence
        direction = 'long' if sweep.direction == 'bullish' else 'short'
        ob_direction = 'bullish' if direction == 'long' else 'bearish'
        nearby_obs = ICTIndicators.get_unmitigated_order_blocks(
            self._order_blocks,
            current_bar['close'],
            ob_direction
        )
        order_block = nearby_obs[0] if nearby_obs else None

        switch = ICTIndicators.detect_switch_candle(
            df,
            sweep,
            order_block,
            bar_idx,
            min_wick_ratio=self.min_wick_ratio,
            min_body_ratio=self.min_body_ratio
        )

        if switch is None:
            return None

        # Calculate targets
        targets = ICTIndicators.calculate_entry_targets(
            switch,
            order_block,
            current_atr,
            risk_reward_ratio=self.risk_reward_ratio
        )

        position = ICTPosition(
            symbol='',  # Will be set by backtest engine
            direction=direction,
            entry_price=targets['entry'],
            entry_time=df.index[bar_idx],
            stop_loss=targets['stop_loss'],
            target=targets['target'],
            order_block=order_block,
            liquidity_sweep=sweep,
            switch_candle=switch,
            trade_type='reversal'
        )

        return (direction, position)

    def _check_continuation_setup(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        current_bar: pd.Series,
        sweep: Optional[LiquiditySweep],
        market_structure: str,
        htf_bias: str,
        current_atr: float
    ) -> Optional[Tuple[str, ICTPosition]]:
        """
        Check for continuation trade setup.

        Continuation criteria:
        1. Clear trend (HH/HL or LH/LL structure)
        2. Pullback to order block or internal liquidity
        3. Switch candle at support/resistance
        4. Aligned with current trend

        Returns:
            Tuple of (direction, position) or None
        """
        # Need a clear trend for continuation
        if market_structure == 'ranging':
            return None

        # Determine direction based on trend
        direction = 'long' if market_structure == 'bullish' else 'short'

        # For continuation, look for pullback to order block
        ob_direction = 'bullish' if direction == 'long' else 'bearish'
        nearby_obs = ICTIndicators.get_unmitigated_order_blocks(
            self._order_blocks,
            current_bar['close'],
            ob_direction
        )

        if not nearby_obs:
            return None

        order_block = nearby_obs[0]

        # Check if price is near order block
        if direction == 'long':
            # Price should be near or touching the OB zone
            if not (current_bar['low'] <= order_block.high * 1.002):
                return None
        else:
            if not (current_bar['high'] >= order_block.low * 0.998):
                return None

        # Look for switch candle confirmation
        switch = ICTIndicators.detect_switch_candle(
            df,
            sweep,
            order_block,
            bar_idx,
            min_wick_ratio=self.min_wick_ratio,
            min_body_ratio=self.min_body_ratio
        )

        if switch is None:
            return None

        # Verify switch direction matches expected
        if switch.direction != direction:
            return None

        # Calculate targets
        targets = ICTIndicators.calculate_entry_targets(
            switch,
            order_block,
            current_atr,
            risk_reward_ratio=self.risk_reward_ratio
        )

        position = ICTPosition(
            symbol='',
            direction=direction,
            entry_price=targets['entry'],
            entry_time=df.index[bar_idx],
            stop_loss=targets['stop_loss'],
            target=targets['target'],
            order_block=order_block,
            liquidity_sweep=sweep,
            switch_candle=switch,
            trade_type='continuation'
        )

        return (direction, position)

    def _check_exit(
        self,
        position: ICTPosition,
        current_close: float,
        current_low: float,
        current_high: float,
        current_time: time,
        current_bar_idx: int = 0
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Uses shared exit_checker for time exit, time stop, max loss, stop-loss,
        and target checks. Adds ICT-specific trailing stop logic on top.

        Exit conditions:
        - Stop loss hit (including trailing stop)
        - Target reached
        - Time exit (end of day)
        - Time stop (max hold bars exceeded)
        - Max loss cap exceeded

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        pos_dir = 1 if position.direction == 'long' else -1

        # Update trailing stop BEFORE checking exits so stop level is current
        if self.use_trailing_stop:
            self._update_trailing_stop(position, current_high, current_low)

        # Delegate core exit checks to shared utility
        exit_reason, reason_str = check_exit(
            position_dir=pos_dir,
            high=current_high,
            low=current_low,
            stop=position.stop_loss,
            target=position.target,
            current_time=current_time,
            exit_time=self.exit_time,
            entry_bar=position.entry_bar_idx,
            current_bar=current_bar_idx,
            max_bars=self.max_hold_bars,
            entry_price=position.entry_price,
            close=current_close,
            max_loss_pct=self.max_loss_pct,
        )

        if exit_reason is not None:
            # Override stop_loss reason with trailing_stop if trailing is active
            if exit_reason == ExitReason.STOP_LOSS and position.trailing_active:
                return True, 'trailing_stop'
            return True, reason_str

        return False, ''

    def _update_trailing_stop(
        self,
        position: ICTPosition,
        current_high: float,
        current_low: float
    ) -> None:
        """Update trailing stop level for ICT position.

        Calculates R (risk) from initial stop and adjusts stop level
        based on max favorable price movement.

        Args:
            position: Active ICT position (modified in-place).
            current_high: Current bar high price.
            current_low: Current bar low price.
        """
        # Calculate R (risk) for trailing stop
        if position.initial_stop_loss != 0:
            risk = abs(position.entry_price - position.initial_stop_loss)
        else:
            risk = abs(position.entry_price - position.stop_loss)

        if risk <= 0:
            return

        if position.direction == 'long':
            # Update max favorable price
            position.max_favorable_price = max(position.max_favorable_price, current_high)

            # Check if trailing should activate (hit trigger R)
            profit = position.max_favorable_price - position.entry_price
            profit_r = profit / risk
            if profit_r >= self.trailing_trigger_r and not position.trailing_active:
                position.trailing_active = True

            # Apply trailing stop
            if position.trailing_active:
                # Trail stop at offset_r below max favorable price
                new_stop = position.max_favorable_price - (self.trailing_offset_r * risk)
                # Only raise stop, never lower it
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
        else:  # short
            # Update max favorable price (lowest for shorts)
            if position.max_favorable_price == position.entry_price:
                position.max_favorable_price = current_low
            else:
                position.max_favorable_price = min(position.max_favorable_price, current_low)

            # Check if trailing should activate
            profit = position.entry_price - position.max_favorable_price
            profit_r = profit / risk
            if profit_r >= self.trailing_trigger_r and not position.trailing_active:
                position.trailing_active = True

            # Apply trailing stop
            if position.trailing_active:
                # Trail stop at offset_r above max favorable price
                new_stop = position.max_favorable_price + (self.trailing_offset_r * risk)
                # Only lower stop for shorts, never raise it
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop

    def _passes_regime_filter(self, direction: str) -> bool:
        """Check if trade passes regime filter."""
        if not self.use_regime:
            return True

        # In BEAR regime, avoid long reversals (risky)
        if self.current_regime == 'BEAR' and direction == 'long':
            return False

        # In STRONG_BULL regime, avoid short reversals
        if self.current_regime == 'STRONG_BULL' and direction == 'short':
            return False

        return True

    def _passes_htf_filter(self, direction: str, htf_bias: str) -> bool:
        """Check if trade aligns with HTF bias."""
        if not self.use_htf_filter:
            return True

        # Neutral HTF bias - allow any direction
        if htf_bias == 'neutral':
            return True

        # For reversals, we're counter-trend on LTF but should align with HTF
        # For continuation, should align with both LTF and HTF
        if direction == 'long' and htf_bias == 'bearish':
            return False
        if direction == 'short' and htf_bias == 'bullish':
            return False

        return True

    def set_regime(self, regime: str) -> None:
        """Set the current market regime."""
        self.current_regime = regime

    def get_parameters(self) -> Dict:
        """Get strategy parameters."""
        return {
            'trade_type': self.trade_type,
            'swing_lookback': self.swing_lookback,
            'min_swing_size_pct': self.min_swing_size_pct,
            'min_impulse_move_pct': self.min_impulse_move_pct,
            'order_block_max_age': self.order_block_max_age,
            'risk_reward_ratio': self.risk_reward_ratio,
            'use_htf_filter': self.use_htf_filter,
            'htf_lookback': self.htf_lookback,
            'use_regime': self.use_regime,
            'long_only': self.long_only,
            'max_positions_per_day': self.max_positions_per_day,
            'atr_period': self.atr_period,
            'atr_stop_multiplier': self.atr_stop_multiplier,
            'min_wick_ratio': self.min_wick_ratio,
            'min_body_ratio': self.min_body_ratio,
            'sweep_threshold_pct': self.sweep_threshold_pct,
            'max_hold_bars': self.max_hold_bars,
            'max_loss_pct': self.max_loss_pct,
            # NEW parameters
            'min_sweep_depth_pct': self.min_sweep_depth_pct,
            'use_momentum_filter': self.use_momentum_filter,
            'momentum_ema_period': self.momentum_ema_period,
            'use_trailing_stop': self.use_trailing_stop,
            'trailing_trigger_r': self.trailing_trigger_r,
            'trailing_offset_r': self.trailing_offset_r
        }

    def reset_state(self) -> None:
        """Reset internal state for new backtest run."""
        self._swing_points = []
        self._order_blocks = []
        self._liquidity_levels = {}
        self._positions = []
        self._daily_trades = {}
