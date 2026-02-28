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

from dataclasses import dataclass, field, asdict
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
class ICTConfig:
    """Configuration for ICT/SMC Strategy parameters.

    Groups all constructor parameters into a single config object to reduce
    parameter explosion. Supports both config-based and kwargs-based
    construction for backward compatibility.
    """

    # Core parameters
    trade_type: str = 'both'
    swing_lookback: int = 5
    min_swing_size_pct: float = 0.002
    min_impulse_move_pct: float = 0.01
    order_block_max_age: int = 50
    impulse_bars: int = 15
    risk_reward_ratio: float = 2.0
    use_htf_filter: bool = True
    htf_lookback: int = 20
    use_regime: bool = True
    long_only: bool = False
    max_positions_per_day: int = 3
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    exit_time_hour: int = 15
    exit_time_minute: int = 45
    min_wick_ratio: float = 0.5
    min_body_ratio: float = 0.3
    sweep_threshold_pct: float = 0.001

    # Optional features (disabled by default)
    use_atr_impulse: bool = False
    atr_impulse_multiple: float = 1.5
    use_volume_filter: bool = False
    rvol_threshold: float = 1.5
    session_filter: str = 'none'
    use_zone_quality: bool = False
    min_zone_quality: float = 0.5

    # Risk management: time stop and max loss cap
    max_hold_bars: int = 0
    max_loss_pct: float = 0.0

    # Signal quality filters
    min_sweep_depth_pct: float = 0.0
    use_momentum_filter: bool = False
    momentum_ema_period: int = 10
    use_structure_filter: bool = False
    require_ob_confluence: bool = False

    # Trailing stop
    use_trailing_stop: bool = False
    trailing_trigger_r: float = 1.0
    trailing_offset_r: float = 0.5


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
        config: Optional[ICTConfig] = None,
        **kwargs
    ):
        # Build config from kwargs if not provided (backward compatibility)
        if config is None:
            config = ICTConfig(**kwargs)
        self._config = config

        # Set attributes BEFORE calling super().__init__()
        # because parent class calls validate_parameters()
        self.trade_type = config.trade_type
        self.swing_lookback = config.swing_lookback
        self.min_swing_size_pct = config.min_swing_size_pct
        self.min_impulse_move_pct = config.min_impulse_move_pct
        self.order_block_max_age = config.order_block_max_age
        self.impulse_bars = config.impulse_bars
        self.risk_reward_ratio = config.risk_reward_ratio
        self.use_htf_filter = config.use_htf_filter
        self.htf_lookback = config.htf_lookback
        self.use_regime = config.use_regime
        self.long_only = config.long_only
        self.max_positions_per_day = config.max_positions_per_day
        self.atr_period = config.atr_period
        self.atr_stop_multiplier = config.atr_stop_multiplier
        self.exit_time = time(config.exit_time_hour, config.exit_time_minute)
        self.min_wick_ratio = config.min_wick_ratio
        self.min_body_ratio = config.min_body_ratio
        self.sweep_threshold_pct = config.sweep_threshold_pct

        # Optional features
        self.use_atr_impulse = config.use_atr_impulse
        self.atr_impulse_multiple = config.atr_impulse_multiple
        self.use_volume_filter = config.use_volume_filter
        self.rvol_threshold = config.rvol_threshold
        self.session_filter = config.session_filter
        self.use_zone_quality = config.use_zone_quality
        self.min_zone_quality = config.min_zone_quality

        # Risk management: time stop and max loss cap
        self.max_hold_bars = config.max_hold_bars
        self.max_loss_pct = config.max_loss_pct

        # Signal quality filters
        self.min_sweep_depth_pct = config.min_sweep_depth_pct
        self.use_momentum_filter = config.use_momentum_filter
        self.momentum_ema_period = config.momentum_ema_period
        self.use_structure_filter = config.use_structure_filter
        self.require_ob_confluence = config.require_ob_confluence

        # Trailing stop
        self.use_trailing_stop = config.use_trailing_stop
        self.trailing_trigger_r = config.trailing_trigger_r
        self.trailing_offset_r = config.trailing_offset_r

        # Regime detector (optional)
        self.regime_detector = MarketRegimeDetector() if config.use_regime else None
        self.current_regime = 'SIDEWAYS'

        # State tracking (reset per backtest run)
        self._swing_points: List[SwingPoint] = []
        self._order_blocks: List[OrderBlock] = []
        self._liquidity_levels: Dict[str, List[float]] = {}
        self._positions: List[ICTPosition] = []
        self._daily_trades: Dict[str, int] = {}  # date -> trade count

        super().__init__(**asdict(config))

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

        Pre-computes all indicators in bulk, then iterates bars
        calling _process_bar for per-bar signal logic.

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

        indicators = self._compute_indicators(data)

        close_values = data['close'].values
        low_values = data['low'].values
        high_values = data['high'].values

        active_position: Optional[ICTPosition] = None
        min_bars = max(self.swing_lookback * 2 + 1, 20)

        for i in range(min_bars, n):
            active_position, signal = self._process_bar(
                i, data, indicators, active_position,
                close_values, low_values, high_values
            )

            if signal == 'long_exit':
                long_exits_arr[i] = True
            elif signal == 'short_exit':
                short_exits_arr[i] = True
            elif signal == 'long_entry':
                long_entries_arr[i] = True
            elif signal == 'short_entry':
                short_entries_arr[i] = True

        return (
            pd.Series(long_entries_arr, index=data.index),
            pd.Series(long_exits_arr, index=data.index),
            pd.Series(short_entries_arr, index=data.index),
            pd.Series(short_exits_arr, index=data.index),
        )

    def _compute_indicators(self, data: pd.DataFrame) -> Dict:
        """Pre-compute all indicators for the entire dataset.

        Runs vectorized indicator calculations so the per-bar loop
        only needs O(1) lookups.

        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex.

        Returns:
            Dict of pre-computed numpy arrays and scalar values.
        """
        n = len(data)

        atr = ICTIndicators.calculate_atr(data, period=self.atr_period)

        self._swing_points = ICTIndicators.detect_swing_points(
            data,
            lookback=self.swing_lookback,
            min_swing_size_pct=self.min_swing_size_pct
        )

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

        # NOTE: POTENTIAL LOOKAHEAD - liquidity levels from ALL swing points
        self._liquidity_levels = ICTIndicators.identify_liquidity_levels(
            self._swing_points
        )

        # NOTE: POTENTIAL LOOKAHEAD - market structure uses ALL swing points
        market_structure, _ = ICTIndicators.classify_market_structure(
            self._swing_points
        )

        # NOTE: POTENTIAL LOOKAHEAD - HTF bias uses ALL swing points
        htf_bias = 'neutral'
        if self.use_htf_filter:
            htf_bias, _ = ICTIndicators.get_htf_bias(
                data, self._swing_points, lookback=self.htf_lookback
            )

        bullish_sweeps, bearish_sweeps, bullish_confirmed, bearish_confirmed, \
            bullish_sweep_depth, bearish_sweep_depth = \
            ICTIndicators.detect_liquidity_sweeps_vectorized(
                data, self._liquidity_levels,
                sweep_threshold_pct=self.sweep_threshold_pct,
                min_sweep_depth_pct=self.min_sweep_depth_pct
            )

        switch_long, switch_short = ICTIndicators.detect_switch_candles_vectorized(
            data,
            min_wick_ratio=self.min_wick_ratio,
            min_body_ratio=self.min_body_ratio
        )

        mitigation_map = ICTIndicators.precompute_order_block_mitigation(
            self._order_blocks, data
        )

        entry_window_mask = ICTIndicators.compute_entry_window_mask(
            data, self.ENTRY_START, self.ENTRY_CUTOFF
        )

        if self.use_volume_filter:
            volume_mask = ICTIndicators.compute_high_volume_mask(
                data, self.rvol_threshold
            )
        else:
            volume_mask = np.ones(n, dtype=bool)

        if self.session_filter != 'none':
            session_mask = np.array([
                ICTIndicators.is_optimal_session(t, self.session_filter)
                for t in data.index
            ])
        else:
            session_mask = np.ones(n, dtype=bool)

        if self.use_momentum_filter:
            bullish_momentum, bearish_momentum = \
                ICTIndicators.compute_momentum_alignment(
                    data, ema_period=self.momentum_ema_period
                )
        else:
            bullish_momentum = np.ones(n, dtype=bool)
            bearish_momentum = np.ones(n, dtype=bool)

        return {
            'atr_values': atr.values,
            'market_structure': market_structure,
            'htf_bias': htf_bias,
            'bullish_confirmed': bullish_confirmed,
            'bearish_confirmed': bearish_confirmed,
            'switch_long': switch_long,
            'switch_short': switch_short,
            'mitigation_map': mitigation_map,
            'entry_window_mask': entry_window_mask,
            'volume_mask': volume_mask,
            'session_mask': session_mask,
            'bullish_momentum': bullish_momentum,
            'bearish_momentum': bearish_momentum,
        }

    def _get_bar_time(self, timestamp) -> time:
        """Convert a bar timestamp to Eastern Time time-of-day.

        Args:
            timestamp: Bar timestamp (UTC DatetimeIndex element).

        Returns:
            time object in Eastern Time.
        """
        if hasattr(timestamp, 'time'):
            try:
                et_time = tz.from_utc(timestamp.to_pydatetime())
                return et_time.time()
            except Exception:
                return timestamp.time()
        return time(12, 0)

    def _process_bar(
        self,
        i: int,
        data: pd.DataFrame,
        indicators: Dict,
        active_position: Optional[ICTPosition],
        close_values: np.ndarray,
        low_values: np.ndarray,
        high_values: np.ndarray,
    ) -> Tuple[Optional[ICTPosition], Optional[str]]:
        """Process a single bar for entry/exit signals.

        Args:
            i: Bar index.
            data: Full OHLCV DataFrame.
            indicators: Pre-computed indicator arrays from _compute_indicators.
            active_position: Currently held position, or None.
            close_values: Close price array.
            low_values: Low price array.
            high_values: High price array.

        Returns:
            Tuple of (updated active_position, signal_type).
            signal_type is one of: 'long_entry', 'short_entry',
            'long_exit', 'short_exit', or None.
        """
        current_time = data.index[i]
        bar_time = self._get_bar_time(current_time)

        # Check exits first
        if active_position is not None:
            should_exit, _ = self._check_exit(
                active_position,
                close_values[i], low_values[i], high_values[i],
                bar_time, current_bar_idx=i
            )
            if should_exit:
                signal = 'long_exit' if active_position.direction == 'long' else 'short_exit'
                return None, signal
            # Still in position, no new signal
            return active_position, None

        # Entry logic
        if not self._passes_entry_filters(i, current_time, indicators):
            return None, None

        direction, trade_type_found = self._determine_direction(i, indicators)
        if direction is None:
            return None, None

        if not self._passes_directional_filters(
            direction, indicators['htf_bias'], i, indicators
        ):
            return None, None

        atr_values = indicators['atr_values']
        current_atr = atr_values[i] if i < len(atr_values) else 1.0
        nearby_ob = self._find_nearby_order_block(
            direction, i, low_values, high_values, indicators['mitigation_map']
        )

        if self.require_ob_confluence and nearby_ob is None:
            return None, None

        position = self._create_entry(
            direction, trade_type_found, i, current_time,
            nearby_ob, current_atr, close_values, low_values, high_values
        )

        signal = 'long_entry' if direction == 'long' else 'short_entry'
        date_str = str(current_time.date()) if hasattr(current_time, 'date') else str(current_time)[:10]
        self._daily_trades[date_str] = self._daily_trades.get(date_str, 0) + 1

        return position, signal

    def _passes_entry_filters(
        self, i: int, current_time, indicators: Dict,
    ) -> bool:
        """Check pre-directional entry filters (window, session, daily limit, setup).

        Args:
            i: Bar index.
            current_time: Timestamp for current bar.
            indicators: Pre-computed indicator arrays.

        Returns:
            True if all filters pass.
        """
        if not indicators['entry_window_mask'][i]:
            return False
        if not indicators['session_mask'][i]:
            return False

        date_str = str(current_time.date()) if hasattr(current_time, 'date') else str(current_time)[:10]
        if self._daily_trades.get(date_str, 0) >= self.max_positions_per_day:
            return False

        has_bullish = indicators['bullish_confirmed'][i] and indicators['switch_long'][i]
        has_bearish = indicators['bearish_confirmed'][i] and indicators['switch_short'][i]
        return has_bullish or has_bearish

    def _determine_direction(
        self, i: int, indicators: Dict,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Determine trade direction and type from setup signals.

        Args:
            i: Bar index.
            indicators: Pre-computed indicator arrays.

        Returns:
            Tuple of (direction, trade_type) or (None, None).
        """
        has_bullish = indicators['bullish_confirmed'][i] and indicators['switch_long'][i]
        has_bearish = indicators['bearish_confirmed'][i] and indicators['switch_short'][i]
        market_structure = indicators['market_structure']

        direction = None
        trade_type_found = None

        if self.trade_type in ['reversal', 'both']:
            if has_bullish and market_structure != 'bullish':
                direction = 'long'
                trade_type_found = 'reversal'
            elif has_bearish and market_structure != 'bearish':
                direction = 'short'
                trade_type_found = 'reversal'

        if direction is None and self.trade_type in ['continuation', 'both']:
            if has_bullish and market_structure == 'bullish':
                direction = 'long'
                trade_type_found = 'continuation'
            elif has_bearish and market_structure == 'bearish':
                direction = 'short'
                trade_type_found = 'continuation'

        return direction, trade_type_found

    def _passes_directional_filters(
        self, direction: str, htf_bias: str, i: int, indicators: Dict,
    ) -> bool:
        """Apply filters that depend on trade direction.

        Checks regime, long-only, HTF, volume, momentum, and structure filters.

        Args:
            direction: 'long' or 'short'.
            htf_bias: Higher-timeframe bias string.
            i: Bar index.
            indicators: Pre-computed indicator arrays.

        Returns:
            True if all directional filters pass.
        """
        if self.use_regime and not self._passes_regime_filter(direction):
            return False
        if self.long_only and direction == 'short':
            return False
        if self.use_htf_filter and not self._passes_htf_filter(direction, htf_bias):
            return False
        if self.use_volume_filter and not indicators['volume_mask'][i]:
            return False

        if self.use_momentum_filter:
            if direction == 'long' and not indicators['bullish_momentum'][i]:
                return False
            if direction == 'short' and not indicators['bearish_momentum'][i]:
                return False

        if self.use_structure_filter:
            ms = indicators['market_structure']
            if direction == 'long' and ms != 'bullish':
                return False
            if direction == 'short' and ms != 'bearish':
                return False

        return True

    def _find_nearby_order_block(
        self,
        direction: str,
        i: int,
        low_values: np.ndarray,
        high_values: np.ndarray,
        mitigation_map: Dict,
    ) -> Optional[OrderBlock]:
        """Find the nearest unmitigated order block for stop placement.

        Args:
            direction: 'long' or 'short'.
            i: Current bar index.
            low_values: Low price array.
            high_values: High price array.
            mitigation_map: Pre-computed OB mitigation bar indices.

        Returns:
            OrderBlock if found, else None.
        """
        ob_direction = 'bullish' if direction == 'long' else 'bearish'
        for ob in self._order_blocks:
            if ob.direction != ob_direction:
                continue
            if ob.available_index > i:
                continue
            mit_idx = mitigation_map.get(ob.index, -1)
            if mit_idx != -1 and mit_idx <= i:
                continue
            if direction == 'long' and low_values[i] <= ob.high * 1.01:
                return ob
            elif direction == 'short' and high_values[i] >= ob.low * 0.99:
                return ob
        return None

    def _create_entry(
        self,
        direction: str,
        trade_type_found: str,
        i: int,
        current_time,
        nearby_ob: Optional[OrderBlock],
        current_atr: float,
        close_values: np.ndarray,
        low_values: np.ndarray,
        high_values: np.ndarray,
    ) -> ICTPosition:
        """Calculate stop/target and create an ICTPosition for entry.

        Args:
            direction: 'long' or 'short'.
            trade_type_found: 'reversal' or 'continuation'.
            i: Bar index.
            current_time: Bar timestamp.
            nearby_ob: Nearby order block (may be None).
            current_atr: ATR value at bar i.
            close_values: Close price array.
            low_values: Low price array.
            high_values: High price array.

        Returns:
            ICTPosition ready for tracking.
        """
        if direction == 'long':
            stop_loss = low_values[i] - current_atr * self.atr_stop_multiplier
            if nearby_ob:
                stop_loss = min(stop_loss, nearby_ob.low - current_atr * 0.25)
        else:
            stop_loss = high_values[i] + current_atr * self.atr_stop_multiplier
            if nearby_ob:
                stop_loss = max(stop_loss, nearby_ob.high + current_atr * 0.25)

        entry_price = close_values[i]
        risk = abs(entry_price - stop_loss)
        if direction == 'long':
            target = entry_price + risk * self.risk_reward_ratio
        else:
            target = entry_price - risk * self.risk_reward_ratio

        return ICTPosition(
            symbol='',
            direction=direction,
            entry_price=entry_price,
            entry_time=current_time,
            stop_loss=stop_loss,
            target=target,
            order_block=nearby_ob,
            trade_type=trade_type_found,
            entry_bar_idx=i,
            initial_stop_loss=stop_loss,
            trailing_active=False,
            max_favorable_price=entry_price
        )

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
