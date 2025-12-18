"""
ICT/SMC (Smart Money Concepts) Liquidity-Based Indicators.

Provides calculations for:
- Swing point detection (HH, HL, LH, LL)
- Market structure classification (bullish, bearish, ranging)
- Order block identification (institutional entry zones)
- Liquidity level mapping (stop hunt zones)
- Liquidity sweep detection
- Switch candle pattern recognition (entry confirmation)
- Multi-timeframe bias calculation

Based on ICT (Inner Circle Trader) methodology for identifying
institutional order flow through liquidity analysis.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SwingPoint:
    """Represents a swing high or swing low in price action."""
    timestamp: datetime
    price: float
    swing_type: str  # 'HH' (higher high), 'HL' (higher low), 'LH' (lower high), 'LL' (lower low)
    index: int  # Bar index in DataFrame where swing occurred
    available_index: int = 0  # Bar index when swing is confirmed (index + lookback) - NO LOOKAHEAD


@dataclass
class OrderBlock:
    """
    Represents an institutional order block (supply/demand zone).

    Order blocks are the last opposing candle before an impulse move,
    where institutional orders are believed to reside.
    """
    timestamp: datetime
    high: float
    low: float
    direction: str  # 'bullish' (demand zone) or 'bearish' (supply zone)
    mitigated: bool
    mitigation_time: Optional[datetime]
    impulse_size: float  # Size of the impulse move that created this OB
    index: int  # Bar index in DataFrame where OB candle occurred
    available_index: int = 0  # Bar index when OB is confirmed (after impulse) - NO LOOKAHEAD


@dataclass
class LiquiditySweep:
    """
    Represents a liquidity sweep (stop hunt) event.

    Occurs when price exceeds a liquidity level (swing high/low)
    then reverses, triggering clustered stop losses.
    """
    timestamp: datetime
    swing_point: SwingPoint
    sweep_price: float  # How far price penetrated past the level
    direction: str  # 'bullish' (swept lows, setup for longs) or 'bearish' (swept highs)
    confirmed: bool
    index: int


@dataclass
class SwitchCandle:
    """
    Represents a switch candle pattern for entry confirmation.

    Switch candles are rejection patterns (hammer, engulfing, etc.)
    that confirm reversal after a liquidity sweep.
    """
    timestamp: datetime
    entry_price: float
    stop_loss: float
    direction: str  # 'long' or 'short'
    candle_type: str  # 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing'
    index: int


# =============================================================================
# ICT Indicators Class
# =============================================================================

class ICTIndicators:
    """
    ICT/SMC specific indicators for liquidity-based trading.

    Implements core Smart Money Concepts:
    - Market structure (swing highs/lows, trend direction)
    - Order blocks (institutional entry zones)
    - Liquidity (stop hunt zones and sweeps)
    - Entry confirmation (switch candle patterns)
    """

    # Market hours in Eastern Time
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    # ==========================================================================
    # Market Structure Detection
    # ==========================================================================

    @staticmethod
    def detect_swing_points(
        df: pd.DataFrame,
        lookback: int = 5,
        min_swing_size_pct: float = 0.002
    ) -> List[SwingPoint]:
        """
        Detect swing highs and swing lows in price data.

        Uses fractal method: a swing high/low is confirmed when price
        fails to make new high/low for 'lookback' bars on each side.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            lookback: Bars on each side for confirmation (default 5)
            min_swing_size_pct: Minimum swing size as % of price (default 0.2%)

        Returns:
            List of SwingPoint objects sorted by timestamp
        """
        if df.empty or len(df) < (2 * lookback + 1):
            return []

        highs = df['high'].values
        lows = df['low'].values
        timestamps = df.index.tolist()

        swing_points = []

        # Detect swing highs and lows using fractal method
        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

            # Check for swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_high:
                swing_points.append({
                    'timestamp': timestamps[i],
                    'price': highs[i],
                    'type': 'high',
                    'index': i,
                    'available_index': i + lookback  # Swing confirmed after lookback bars - NO LOOKAHEAD
                })

            if is_swing_low:
                swing_points.append({
                    'timestamp': timestamps[i],
                    'price': lows[i],
                    'type': 'low',
                    'index': i,
                    'available_index': i + lookback  # Swing confirmed after lookback bars - NO LOOKAHEAD
                })

        # Sort by index
        swing_points.sort(key=lambda x: x['index'])

        # Filter by minimum swing size
        filtered_swings = []
        for i, swing in enumerate(swing_points):
            if i == 0:
                filtered_swings.append(swing)
                continue

            prev_swing = filtered_swings[-1] if filtered_swings else None
            if prev_swing:
                swing_size = abs(swing['price'] - prev_swing['price']) / prev_swing['price']
                if swing_size >= min_swing_size_pct:
                    filtered_swings.append(swing)
            else:
                filtered_swings.append(swing)

        # Classify swing types (HH, HL, LH, LL)
        classified_swings = ICTIndicators._classify_swing_types(filtered_swings)

        return classified_swings

    @staticmethod
    def _classify_swing_types(swings: List[dict]) -> List[SwingPoint]:
        """
        Classify swing points as HH, HL, LH, or LL based on sequence.

        Args:
            swings: List of swing dictionaries with 'price', 'type', 'timestamp', 'index'

        Returns:
            List of SwingPoint objects with classified types
        """
        if not swings:
            return []

        classified = []
        prev_high = None
        prev_low = None

        for swing in swings:
            if swing['type'] == 'high':
                if prev_high is None:
                    swing_type = 'HH'  # First high, assume bullish
                elif swing['price'] > prev_high:
                    swing_type = 'HH'  # Higher high
                else:
                    swing_type = 'LH'  # Lower high
                prev_high = swing['price']
            else:  # low
                if prev_low is None:
                    swing_type = 'HL'  # First low, assume bullish
                elif swing['price'] > prev_low:
                    swing_type = 'HL'  # Higher low
                else:
                    swing_type = 'LL'  # Lower low
                prev_low = swing['price']

            classified.append(SwingPoint(
                timestamp=swing['timestamp'],
                price=swing['price'],
                swing_type=swing_type,
                index=swing['index'],
                available_index=swing.get('available_index', swing['index'])  # Use available_index for lookahead prevention
            ))

        return classified

    @staticmethod
    def classify_market_structure(
        swing_points: List[SwingPoint]
    ) -> Tuple[str, List[SwingPoint]]:
        """
        Classify market structure based on swing point sequence.

        Args:
            swing_points: List of SwingPoint objects

        Returns:
            Tuple of (trend, recent_swings)
            - trend: 'bullish' (HH/HL), 'bearish' (LH/LL), 'ranging'
            - recent_swings: Last 4 classified swing points
        """
        if len(swing_points) < 4:
            return 'ranging', swing_points[-4:] if len(swing_points) >= 1 else []

        recent = swing_points[-4:]

        # Count swing types
        hh_count = sum(1 for s in recent if s.swing_type == 'HH')
        hl_count = sum(1 for s in recent if s.swing_type == 'HL')
        lh_count = sum(1 for s in recent if s.swing_type == 'LH')
        ll_count = sum(1 for s in recent if s.swing_type == 'LL')

        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count

        if bullish_score >= 3:
            return 'bullish', recent
        elif bearish_score >= 3:
            return 'bearish', recent
        else:
            return 'ranging', recent

    @staticmethod
    def detect_bos(
        swing_points: List[SwingPoint],
        current_price: float,
        current_time: datetime
    ) -> Optional[Tuple[str, SwingPoint]]:
        """
        Detect Break of Structure (BOS).

        BOS occurs when price breaks a previous swing high (bullish BOS)
        or swing low (bearish BOS), confirming trend continuation.

        Args:
            swing_points: List of SwingPoint objects
            current_price: Current price
            current_time: Current timestamp

        Returns:
            Tuple of (direction, broken_swing) or None
        """
        if len(swing_points) < 2:
            return None

        # Find most recent swing high and swing low
        recent_high = None
        recent_low = None

        for swing in reversed(swing_points):
            if swing.swing_type in ['HH', 'LH'] and recent_high is None:
                recent_high = swing
            elif swing.swing_type in ['HL', 'LL'] and recent_low is None:
                recent_low = swing

            if recent_high and recent_low:
                break

        # Check for bullish BOS (break above recent high)
        if recent_high and current_price > recent_high.price:
            return ('bullish', recent_high)

        # Check for bearish BOS (break below recent low)
        if recent_low and current_price < recent_low.price:
            return ('bearish', recent_low)

        return None

    @staticmethod
    def detect_choch(
        swing_points: List[SwingPoint],
        current_price: float,
        current_time: datetime,
        current_trend: str
    ) -> Optional[Tuple[str, SwingPoint]]:
        """
        Detect Change of Character (CHoCH).

        CHoCH occurs when price breaks structure against the trend,
        signaling potential reversal. More significant than BOS.

        Args:
            swing_points: List of SwingPoint objects
            current_price: Current price
            current_time: Current timestamp
            current_trend: Current trend ('bullish', 'bearish', 'ranging')

        Returns:
            Tuple of (new_direction, broken_swing) or None
        """
        if len(swing_points) < 2 or current_trend == 'ranging':
            return None

        # Find key level to break for CHoCH
        if current_trend == 'bullish':
            # In uptrend, CHoCH = break below recent higher low
            for swing in reversed(swing_points):
                if swing.swing_type == 'HL':
                    if current_price < swing.price:
                        return ('bearish', swing)
                    break
        else:  # bearish
            # In downtrend, CHoCH = break above recent lower high
            for swing in reversed(swing_points):
                if swing.swing_type == 'LH':
                    if current_price > swing.price:
                        return ('bullish', swing)
                    break

        return None

    # ==========================================================================
    # Order Block Detection
    # ==========================================================================

    @staticmethod
    def identify_order_blocks(
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        min_impulse_move_pct: float = 0.01,
        max_age_bars: int = 50,
        impulse_bars: int = 3,
        use_atr_impulse: bool = False,
        atr_impulse_multiple: float = 1.5,
        atr_period: int = 14
    ) -> List[OrderBlock]:
        """
        Identify order blocks (institutional entry zones).

        An order block is the last opposing candle before an impulse move.
        - Bullish OB: Last bearish candle before bullish impulse
        - Bearish OB: Last bullish candle before bearish impulse

        Args:
            df: OHLCV DataFrame
            swing_points: Previously detected swing points
            min_impulse_move_pct: Minimum impulse size as percentage (default 1%)
                                  Used when use_atr_impulse=False
            max_age_bars: Maximum bars back to look for OBs (default 50)
            impulse_bars: Number of bars to measure impulse over (default 3)
                          For 1-minute data, use 15-30 for meaningful moves
            use_atr_impulse: If True, use ATR multiple instead of percentage (default False)
            atr_impulse_multiple: ATR multiple for valid impulse (default 1.5x)
                                  Original spec suggests 1.5x, 2x, 3x
            atr_period: Period for ATR calculation (default 14)

        Returns:
            List of valid (unmitigated) order blocks
        """
        if df.empty or len(df) < 5:
            return []

        order_blocks = []
        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df.index.tolist()

        # Calculate ATR if using ATR-based impulse detection
        atr_values = None
        if use_atr_impulse:
            atr_values = ICTIndicators.calculate_atr(df, period=atr_period)

        current_idx = len(df) - 1
        start_idx = max(0, current_idx - max_age_bars)

        # Look for impulse moves and identify OBs before them
        for i in range(start_idx + impulse_bars, current_idx):
            # Calculate impulse over next N bars
            impulse_start = closes[i]
            impulse_end = closes[min(i + impulse_bars, current_idx)]
            impulse_abs = abs(impulse_end - impulse_start)
            impulse_pct = impulse_abs / impulse_start

            # Check impulse threshold based on method
            if use_atr_impulse and atr_values is not None:
                # ATR-based: impulse must exceed ATR multiple
                atr_at_bar = atr_values[i] if i < len(atr_values) else atr_values[-1]
                min_impulse = atr_at_bar * atr_impulse_multiple
                if impulse_abs < min_impulse:
                    continue
            else:
                # Percentage-based (original method)
                if impulse_pct < min_impulse_move_pct:
                    continue

            # OB is confirmed when impulse completes - NO LOOKAHEAD
            impulse_end_idx = min(i + impulse_bars, current_idx)

            # Bullish impulse - look for bearish OB before it
            if impulse_end > impulse_start:
                # Find last bearish candle before impulse
                for j in range(i, max(start_idx, i - 5), -1):
                    if closes[j] < opens[j]:  # Bearish candle
                        ob = OrderBlock(
                            timestamp=timestamps[j],
                            high=max(opens[j], closes[j]),  # Use body, not wick
                            low=min(opens[j], closes[j]),
                            direction='bullish',
                            mitigated=False,
                            mitigation_time=None,
                            impulse_size=impulse_pct,
                            index=j,
                            available_index=impulse_end_idx  # OB available after impulse confirms - NO LOOKAHEAD
                        )
                        order_blocks.append(ob)
                        break

            # Bearish impulse - look for bullish OB before it
            else:
                # Find last bullish candle before impulse
                for j in range(i, max(start_idx, i - 5), -1):
                    if closes[j] > opens[j]:  # Bullish candle
                        ob = OrderBlock(
                            timestamp=timestamps[j],
                            high=max(opens[j], closes[j]),
                            low=min(opens[j], closes[j]),
                            direction='bearish',
                            mitigated=False,
                            mitigation_time=None,
                            impulse_size=impulse_pct,
                            index=j,
                            available_index=impulse_end_idx  # OB available after impulse confirms - NO LOOKAHEAD
                        )
                        order_blocks.append(ob)
                        break

        # Remove duplicates (keep highest impulse size)
        unique_obs = {}
        for ob in order_blocks:
            key = ob.index
            if key not in unique_obs or ob.impulse_size > unique_obs[key].impulse_size:
                unique_obs[key] = ob

        return list(unique_obs.values())

    @staticmethod
    def update_order_block_mitigation(
        order_blocks: List[OrderBlock],
        df: pd.DataFrame
    ) -> List[OrderBlock]:
        """
        Update order block mitigation status.

        An order block is mitigated when price returns and touches it.

        Args:
            order_blocks: List of OrderBlock objects
            df: Current OHLCV DataFrame

        Returns:
            Updated list with mitigation status and times
        """
        if not order_blocks or df.empty:
            return order_blocks

        highs = df['high'].values
        lows = df['low'].values
        timestamps = df.index.tolist()

        for ob in order_blocks:
            if ob.mitigated:
                continue

            # Check bars after OB creation for mitigation
            for i in range(ob.index + 1, len(df)):
                # Bullish OB mitigated when price drops into it
                if ob.direction == 'bullish':
                    if lows[i] <= ob.high:
                        ob.mitigated = True
                        ob.mitigation_time = timestamps[i]
                        break
                # Bearish OB mitigated when price rises into it
                else:
                    if highs[i] >= ob.low:
                        ob.mitigated = True
                        ob.mitigation_time = timestamps[i]
                        break

        return order_blocks

    @staticmethod
    def get_unmitigated_order_blocks(
        order_blocks: List[OrderBlock],
        current_price: float,
        direction: Optional[str] = None
    ) -> List[OrderBlock]:
        """
        Get unmitigated order blocks near current price.

        Args:
            order_blocks: List of OrderBlock objects
            current_price: Current price
            direction: Optional filter ('bullish' or 'bearish')

        Returns:
            List of unmitigated OBs, sorted by proximity to current price
        """
        unmitigated = [ob for ob in order_blocks if not ob.mitigated]

        if direction:
            unmitigated = [ob for ob in unmitigated if ob.direction == direction]

        # Sort by proximity to current price
        def proximity(ob):
            ob_mid = (ob.high + ob.low) / 2
            return abs(current_price - ob_mid)

        return sorted(unmitigated, key=proximity)

    # ==========================================================================
    # Liquidity Detection
    # ==========================================================================

    @staticmethod
    def identify_liquidity_levels(
        swing_points: List[SwingPoint],
        equal_highs_tolerance: float = 0.001,
        equal_lows_tolerance: float = 0.001
    ) -> Dict[str, List[float]]:
        """
        Identify liquidity pools (clusters of stop losses).

        Liquidity pools form at:
        - Swing highs (stop losses from shorts)
        - Swing lows (stop losses from longs)
        - Equal highs/lows (many stops at same level)

        Args:
            swing_points: List of SwingPoint objects
            equal_highs_tolerance: % tolerance for equal highs (default 0.1%)
            equal_lows_tolerance: % tolerance for equal lows (default 0.1%)

        Returns:
            Dict with 'buy_stops' (above highs) and 'sell_stops' (below lows)
        """
        if not swing_points:
            return {'buy_stops': [], 'sell_stops': []}

        # Separate highs and lows
        highs = [s.price for s in swing_points if s.swing_type in ['HH', 'LH']]
        lows = [s.price for s in swing_points if s.swing_type in ['HL', 'LL']]

        # Cluster equal highs/lows
        buy_stops = ICTIndicators._cluster_levels(highs, equal_highs_tolerance)
        sell_stops = ICTIndicators._cluster_levels(lows, equal_lows_tolerance)

        return {
            'buy_stops': sorted(buy_stops, reverse=True),  # Highest first
            'sell_stops': sorted(sell_stops)  # Lowest first
        }

    @staticmethod
    def _cluster_levels(prices: List[float], tolerance: float) -> List[float]:
        """Cluster nearby price levels."""
        if not prices:
            return []

        prices = sorted(prices)
        clusters = []
        current_cluster = [prices[0]]

        for price in prices[1:]:
            if abs(price - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(price)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [price]

        clusters.append(np.mean(current_cluster))
        return clusters

    @staticmethod
    def detect_liquidity_sweep(
        df: pd.DataFrame,
        liquidity_levels: Dict[str, List[float]],
        current_bar_idx: int,
        sweep_threshold_pct: float = 0.001
    ) -> Optional[LiquiditySweep]:
        """
        Detect liquidity sweep (stop hunt).

        A sweep occurs when price exceeds a liquidity level then
        reverses, triggering stops before moving in opposite direction.

        Args:
            df: OHLCV DataFrame
            liquidity_levels: From identify_liquidity_levels()
            current_bar_idx: Current bar index
            sweep_threshold_pct: Minimum penetration past level (default 0.1%)

        Returns:
            LiquiditySweep object if sweep detected, None otherwise
        """
        if df.empty or current_bar_idx < 1 or current_bar_idx >= len(df):
            return None

        high = df['high'].iloc[current_bar_idx]
        low = df['low'].iloc[current_bar_idx]
        close = df['close'].iloc[current_bar_idx]
        open_price = df['open'].iloc[current_bar_idx]
        timestamp = df.index[current_bar_idx]

        # Check for bearish sweep (price exceeded a high then reversed down)
        for level in liquidity_levels.get('buy_stops', []):
            if high > level * (1 + sweep_threshold_pct) and close < level:
                # Found bearish sweep - swept buy stops
                return LiquiditySweep(
                    timestamp=timestamp,
                    swing_point=SwingPoint(
                        timestamp=timestamp,
                        price=level,
                        swing_type='LH',  # Approximation
                        index=current_bar_idx
                    ),
                    sweep_price=high,
                    direction='bearish',
                    confirmed=close < open_price,  # Bearish candle confirms
                    index=current_bar_idx
                )

        # Check for bullish sweep (price exceeded a low then reversed up)
        for level in liquidity_levels.get('sell_stops', []):
            if low < level * (1 - sweep_threshold_pct) and close > level:
                # Found bullish sweep - swept sell stops
                return LiquiditySweep(
                    timestamp=timestamp,
                    swing_point=SwingPoint(
                        timestamp=timestamp,
                        price=level,
                        swing_type='HL',  # Approximation
                        index=current_bar_idx
                    ),
                    sweep_price=low,
                    direction='bullish',
                    confirmed=close > open_price,  # Bullish candle confirms
                    index=current_bar_idx
                )

        return None

    # ==========================================================================
    # Switch Candle Detection
    # ==========================================================================

    @staticmethod
    def detect_switch_candle(
        df: pd.DataFrame,
        liquidity_sweep: Optional[LiquiditySweep],
        order_block: Optional[OrderBlock],
        current_bar_idx: int,
        min_wick_ratio: float = 0.5,
        min_body_ratio: float = 0.3
    ) -> Optional[SwitchCandle]:
        """
        Detect switch candle pattern for entry confirmation.

        Switch candle patterns:
        - Bullish: Hammer, bullish engulfing
        - Bearish: Shooting star, bearish engulfing

        Requires: Rejection wick + close in opposite direction

        Args:
            df: OHLCV DataFrame
            liquidity_sweep: Previously detected sweep (or None)
            order_block: Optional order block for confluence
            current_bar_idx: Current bar index
            min_wick_ratio: Minimum wick as ratio of range (default 0.5)
            min_body_ratio: Minimum body as ratio of range (default 0.3)

        Returns:
            SwitchCandle for entry if pattern confirmed, None otherwise
        """
        if df.empty or current_bar_idx < 1 or current_bar_idx >= len(df):
            return None

        # Get current and previous bar data
        curr = df.iloc[current_bar_idx]
        prev = df.iloc[current_bar_idx - 1]

        curr_open = curr['open']
        curr_high = curr['high']
        curr_low = curr['low']
        curr_close = curr['close']

        prev_open = prev['open']
        prev_close = prev['close']

        timestamp = df.index[current_bar_idx]

        candle_range = curr_high - curr_low
        if candle_range <= 0:
            return None

        body = abs(curr_close - curr_open)
        lower_wick = min(curr_open, curr_close) - curr_low
        upper_wick = curr_high - max(curr_open, curr_close)

        body_ratio = body / candle_range
        lower_wick_ratio = lower_wick / candle_range
        upper_wick_ratio = upper_wick / candle_range

        is_bullish = curr_close > curr_open
        is_bearish = curr_close < curr_open

        # Detect bullish patterns
        if is_bullish:
            # Hammer pattern: Long lower wick, small upper wick
            is_hammer = (
                lower_wick_ratio >= min_wick_ratio and
                upper_wick_ratio < 0.2 and
                body_ratio >= min_body_ratio
            )

            # Bullish engulfing: Current body engulfs previous
            is_engulfing = (
                prev_close < prev_open and  # Previous was bearish
                curr_open < prev_close and  # Open below prev close
                curr_close > prev_open and  # Close above prev open
                body_ratio >= 0.5
            )

            if is_hammer or is_engulfing:
                candle_type = 'hammer' if is_hammer else 'bullish_engulfing'
                stop_loss = curr_low

                # Adjust stop if we have an order block
                if order_block and order_block.direction == 'bullish':
                    stop_loss = min(stop_loss, order_block.low)

                return SwitchCandle(
                    timestamp=timestamp,
                    entry_price=curr_close,
                    stop_loss=stop_loss,
                    direction='long',
                    candle_type=candle_type,
                    index=current_bar_idx
                )

        # Detect bearish patterns
        if is_bearish:
            # Shooting star: Long upper wick, small lower wick
            is_shooting_star = (
                upper_wick_ratio >= min_wick_ratio and
                lower_wick_ratio < 0.2 and
                body_ratio >= min_body_ratio
            )

            # Bearish engulfing: Current body engulfs previous
            is_engulfing = (
                prev_close > prev_open and  # Previous was bullish
                curr_open > prev_close and  # Open above prev close
                curr_close < prev_open and  # Close below prev open
                body_ratio >= 0.5
            )

            if is_shooting_star or is_engulfing:
                candle_type = 'shooting_star' if is_shooting_star else 'bearish_engulfing'
                stop_loss = curr_high

                # Adjust stop if we have an order block
                if order_block and order_block.direction == 'bearish':
                    stop_loss = max(stop_loss, order_block.high)

                return SwitchCandle(
                    timestamp=timestamp,
                    entry_price=curr_close,
                    stop_loss=stop_loss,
                    direction='short',
                    candle_type=candle_type,
                    index=current_bar_idx
                )

        return None

    # ==========================================================================
    # Entry/Exit Calculations
    # ==========================================================================

    @staticmethod
    def calculate_entry_targets(
        switch_candle: SwitchCandle,
        order_block: Optional[OrderBlock],
        atr: float,
        risk_reward_ratio: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate entry, stop loss, and take profit levels.

        Stop loss placement:
        - Below switch candle low (longs)
        - Above switch candle high (shorts)
        - Or below/above order block if present

        Args:
            switch_candle: Detected switch candle
            order_block: Optional order block for stop placement
            atr: Average True Range for stop adjustment
            risk_reward_ratio: Target RR ratio (default 2.0)

        Returns:
            Dict with 'entry', 'stop_loss', 'target', 'risk', 'reward'
        """
        entry = switch_candle.entry_price
        stop_loss = switch_candle.stop_loss

        # Add ATR buffer to stop
        if switch_candle.direction == 'long':
            stop_loss = stop_loss - (atr * 0.25)  # Small buffer below
            risk = entry - stop_loss
            target = entry + (risk * risk_reward_ratio)
        else:
            stop_loss = stop_loss + (atr * 0.25)  # Small buffer above
            risk = stop_loss - entry
            target = entry - (risk * risk_reward_ratio)

        reward = abs(target - entry)

        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'target': target,
            'risk': risk,
            'reward': reward,
            'rr_ratio': risk_reward_ratio
        }

    # ==========================================================================
    # Multi-Timeframe Analysis
    # ==========================================================================

    @staticmethod
    def get_htf_bias(
        htf_data: pd.DataFrame,
        htf_swing_points: Optional[List[SwingPoint]] = None,
        lookback: int = 20
    ) -> Tuple[str, float]:
        """
        Get higher timeframe directional bias.

        Uses market structure and recent price action to determine
        the dominant trend on a higher timeframe.

        Args:
            htf_data: Higher timeframe (15m/1H) OHLCV data
            htf_swing_points: Optional pre-computed swing points
            lookback: Bars to analyze (default 20)

        Returns:
            Tuple of (bias, confidence)
            - bias: 'bullish', 'bearish', 'neutral'
            - confidence: 0.0 to 1.0
        """
        if htf_data.empty or len(htf_data) < lookback:
            return 'neutral', 0.0

        recent = htf_data.tail(lookback)

        # Calculate trend strength
        close_start = recent['close'].iloc[0]
        close_end = recent['close'].iloc[-1]
        trend_pct = (close_end - close_start) / close_start

        # Simple trend classification
        if trend_pct > 0.02:  # 2% up
            bias = 'bullish'
            confidence = min(1.0, abs(trend_pct) / 0.05)
        elif trend_pct < -0.02:  # 2% down
            bias = 'bearish'
            confidence = min(1.0, abs(trend_pct) / 0.05)
        else:
            bias = 'neutral'
            confidence = 0.3

        # Enhance with swing point analysis if available
        if htf_swing_points:
            structure, _ = ICTIndicators.classify_market_structure(htf_swing_points)
            if structure == 'bullish' and bias == 'bullish':
                confidence = min(1.0, confidence + 0.2)
            elif structure == 'bearish' and bias == 'bearish':
                confidence = min(1.0, confidence + 0.2)
            elif structure != 'ranging' and structure != bias:
                confidence *= 0.5  # Reduce confidence if structure conflicts

        return bias, confidence

    # ==========================================================================
    # Utility Functions
    # ==========================================================================

    @staticmethod
    def is_price_in_zone(
        price: float,
        zone_high: float,
        zone_low: float,
        buffer_pct: float = 0.001
    ) -> bool:
        """Check if price is within a zone (with optional buffer)."""
        buffer = (zone_high - zone_low) * buffer_pct
        return (zone_low - buffer) <= price <= (zone_high + buffer)

    @staticmethod
    def calculate_atr(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        if df.empty or len(df) < period:
            return pd.Series(dtype=float)

        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()

        return atr

    @staticmethod
    def calculate_rvol(
        df: pd.DataFrame,
        lookback: int = 20
    ) -> pd.Series:
        """
        Calculate Relative Volume (RVOL).

        RVOL = Current Volume / Average Volume over lookback period.
        Values > 1.0 indicate above-average volume.

        Args:
            df: OHLCV DataFrame with 'volume' column
            lookback: Number of bars for average volume (default 20)

        Returns:
            Series of RVOL values
        """
        if df.empty or 'volume' not in df.columns:
            return pd.Series(dtype=float)

        volume = df['volume']
        avg_volume = volume.rolling(window=lookback, min_periods=1).mean()

        # Avoid division by zero
        rvol = volume / avg_volume.replace(0, np.nan)
        return rvol.fillna(1.0)

    @staticmethod
    def is_high_volume_bar(
        df: pd.DataFrame,
        bar_idx: int,
        rvol_threshold: float = 1.5,
        lookback: int = 20
    ) -> bool:
        """
        Check if a bar has above-average volume.

        Args:
            df: OHLCV DataFrame
            bar_idx: Index of the bar to check
            rvol_threshold: Minimum RVOL for "high volume" (default 1.5x)
            lookback: Bars for average volume calculation

        Returns:
            True if bar has high relative volume
        """
        if df.empty or 'volume' not in df.columns or bar_idx >= len(df):
            return False

        rvol = ICTIndicators.calculate_rvol(df, lookback)
        if bar_idx < len(rvol):
            return rvol.iloc[bar_idx] >= rvol_threshold
        return False

    # ==========================================================================
    # Session Filters (Original Spec: London/NY Overlap Focus)
    # ==========================================================================

    @staticmethod
    def is_optimal_session(
        timestamp: datetime,
        session_filter: str = 'us_regular'
    ) -> bool:
        """
        Check if timestamp falls within optimal trading session.

        Sessions are defined for US equities trading (in Eastern Time):
        - 'us_regular': Regular US market hours (9:30 AM - 4:00 PM ET)
        - 'us_core': Core hours with highest liquidity (10:00 AM - 3:00 PM ET)
        - 'ny_open': First 2 hours of NY session (9:30 AM - 11:30 AM ET)
        - 'power_hour': Last hour of trading (3:00 PM - 4:00 PM ET)
        - 'none': No session filter (always returns True)

        Crypto sessions (in UTC, converted from ET internally):
        - 'crypto_us': US hours for crypto (9:00 AM - 5:00 PM ET)
        - 'crypto_overlap': US/EU overlap highest volume (8:00 AM - 12:00 PM ET)
        - 'crypto_asia': Asian session (8:00 PM - 4:00 AM ET)

        Args:
            timestamp: Datetime to check (UTC timestamps will be converted to ET)
            session_filter: Session type to check against

        Returns:
            True if timestamp is within the specified session
        """
        if session_filter == 'none':
            return True

        # Convert UTC timestamp to ET for proper comparison
        # Data timestamps are in UTC but session windows are in Eastern Time
        try:
            from src.utils.timezone import tz
            if hasattr(timestamp, 'to_pydatetime'):
                et_ts = tz.from_utc(timestamp.to_pydatetime())
            else:
                et_ts = tz.from_utc(timestamp)
            t = et_ts.time()
        except Exception:
            # Fallback to raw time if conversion fails
            t = timestamp.time() if hasattr(timestamp, 'time') else timestamp

        session_windows = {
            # Equity sessions (ET)
            'us_regular': (time(9, 30), time(16, 0)),
            'us_core': (time(10, 0), time(15, 0)),
            'ny_open': (time(9, 30), time(11, 30)),
            'power_hour': (time(15, 0), time(16, 0)),
            # Crypto sessions (ET) - when institutional volume is highest
            'crypto_us': (time(9, 0), time(17, 0)),  # US business hours
            'crypto_overlap': (time(8, 0), time(12, 0)),  # US/EU overlap
        }

        # Handle overnight sessions separately (spans midnight)
        if session_filter == 'crypto_asia':
            # Asian session: 8 PM - 4 AM ET (overnight)
            return t >= time(20, 0) or t <= time(4, 0)

        if session_filter not in session_windows:
            return True

        start, end = session_windows[session_filter]
        return start <= t <= end

    # ==========================================================================
    # Zone Quality Scoring (Original Spec Feature)
    # ==========================================================================

    @staticmethod
    def calculate_zone_quality(
        order_block: OrderBlock,
        current_bar_idx: int,
        touch_count: int = 0,
        htf_aligned: bool = False,
        max_age_bars: int = 100
    ) -> float:
        """
        Calculate quality score for an order block zone.

        Scoring based on original spec factors:
        - Impulse strength (how strong was the move that created it)
        - Zone freshness (newer zones often stronger)
        - Touch count (first touch often best)
        - HTF confluence (alignment with higher timeframe)

        Args:
            order_block: The OrderBlock to score
            current_bar_idx: Current bar index for age calculation
            touch_count: Number of times price has visited this zone
            htf_aligned: Whether zone aligns with HTF bias
            max_age_bars: Maximum age for freshness calculation

        Returns:
            Quality score from 0.0 to 1.0
        """
        score = 0.0

        # Factor 1: Impulse strength (30% weight)
        # Normalize impulse size (0.5% = 0.5, 2% = 1.0, capped)
        impulse_score = min(order_block.impulse_size / 0.02, 1.0)
        score += 0.30 * impulse_score

        # Factor 2: Zone freshness (25% weight)
        # Newer zones score higher
        age = current_bar_idx - order_block.index
        freshness = max(0, 1 - (age / max_age_bars))
        score += 0.25 * freshness

        # Factor 3: Touch count (25% weight)
        # First touch = 1.0, second = 0.5, third+ = 0.0
        if touch_count == 0:
            touch_score = 1.0
        elif touch_count == 1:
            touch_score = 0.5
        else:
            touch_score = 0.0
        score += 0.25 * touch_score

        # Factor 4: HTF confluence (20% weight)
        if htf_aligned:
            score += 0.20

        return min(score, 1.0)

    @staticmethod
    def filter_zones_by_quality(
        order_blocks: List[OrderBlock],
        current_bar_idx: int,
        min_quality: float = 0.5,
        htf_bias: Optional[str] = None
    ) -> List[Tuple[OrderBlock, float]]:
        """
        Filter order blocks by quality score.

        Args:
            order_blocks: List of order blocks to filter
            current_bar_idx: Current bar index
            min_quality: Minimum quality score to include (default 0.5)
            htf_bias: Higher timeframe bias ('bullish', 'bearish', None)

        Returns:
            List of (OrderBlock, quality_score) tuples for blocks meeting threshold
        """
        scored_blocks = []

        for ob in order_blocks:
            if ob.mitigated:
                continue

            # Check HTF alignment
            htf_aligned = False
            if htf_bias is not None:
                htf_aligned = (
                    (ob.direction == 'bullish' and htf_bias == 'bullish') or
                    (ob.direction == 'bearish' and htf_bias == 'bearish')
                )

            quality = ICTIndicators.calculate_zone_quality(
                ob, current_bar_idx, touch_count=0, htf_aligned=htf_aligned
            )

            if quality >= min_quality:
                scored_blocks.append((ob, quality))

        # Sort by quality descending
        scored_blocks.sort(key=lambda x: x[1], reverse=True)
        return scored_blocks

    # ==========================================================================
    # VECTORIZED METHODS (High Performance)
    # ==========================================================================

    @staticmethod
    def detect_liquidity_sweeps_vectorized(
        df: pd.DataFrame,
        liquidity_levels: Dict[str, List[float]],
        sweep_threshold_pct: float = 0.001,
        min_sweep_depth_pct: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized detection of all liquidity sweeps across entire dataset.

        Instead of calling detect_liquidity_sweep() for each bar (O(n) calls),
        this pre-computes all sweeps in a single pass using numpy broadcasting.

        Args:
            df: OHLCV DataFrame
            liquidity_levels: Dict with 'buy_stops' and 'sell_stops' from identify_liquidity_levels()
            sweep_threshold_pct: Minimum penetration past level (default 0.1%)
            min_sweep_depth_pct: Minimum sweep depth % to count as valid (default 0.0)

        Returns:
            Tuple of (bullish_sweeps, bearish_sweeps, bullish_confirmed, bearish_confirmed,
                     bullish_sweep_depth, bearish_sweep_depth)
            Boolean arrays + float arrays of sweep depth percentages
        """
        n = len(df)
        if n == 0:
            empty_bool = np.zeros(0, dtype=bool)
            empty_float = np.zeros(0, dtype=float)
            return empty_bool, empty_bool, empty_bool, empty_bool, empty_float, empty_float

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_price = df['open'].values

        bullish_sweeps = np.zeros(n, dtype=bool)
        bearish_sweeps = np.zeros(n, dtype=bool)
        bullish_sweep_depth = np.zeros(n, dtype=float)
        bearish_sweep_depth = np.zeros(n, dtype=float)

        # Detect bearish sweeps (price exceeded a high then reversed down)
        for level in liquidity_levels.get('buy_stops', []):
            threshold = level * (1 + sweep_threshold_pct)
            sweep_mask = (high > threshold) & (close < level)
            # Calculate sweep depth = how far price penetrated past level
            depth = np.where(sweep_mask, (high - level) / level, 0.0)
            # Apply minimum sweep depth filter
            if min_sweep_depth_pct > 0:
                sweep_mask = sweep_mask & (depth >= min_sweep_depth_pct)
            bearish_sweeps |= sweep_mask
            bearish_sweep_depth = np.maximum(bearish_sweep_depth, depth)

        # Detect bullish sweeps (price exceeded a low then reversed up)
        for level in liquidity_levels.get('sell_stops', []):
            threshold = level * (1 - sweep_threshold_pct)
            sweep_mask = (low < threshold) & (close > level)
            # Calculate sweep depth = how far price penetrated past level
            depth = np.where(sweep_mask, (level - low) / level, 0.0)
            # Apply minimum sweep depth filter
            if min_sweep_depth_pct > 0:
                sweep_mask = sweep_mask & (depth >= min_sweep_depth_pct)
            bullish_sweeps |= sweep_mask
            bullish_sweep_depth = np.maximum(bullish_sweep_depth, depth)

        # Confirmation: candle closes in direction of sweep
        bullish_confirmed = bullish_sweeps & (close > open_price)
        bearish_confirmed = bearish_sweeps & (close < open_price)

        return bullish_sweeps, bearish_sweeps, bullish_confirmed, bearish_confirmed, bullish_sweep_depth, bearish_sweep_depth

    @staticmethod
    def detect_switch_candles_vectorized(
        df: pd.DataFrame,
        min_wick_ratio: float = 0.5,
        min_body_ratio: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized detection of switch candle patterns across entire dataset.

        Detects hammer/bullish engulfing (for longs) and shooting star/bearish engulfing (for shorts).

        Args:
            df: OHLCV DataFrame
            min_wick_ratio: Minimum wick as ratio of range (default 0.5)
            min_body_ratio: Minimum body as ratio of range (default 0.3)

        Returns:
            Tuple of (switch_long, switch_short) boolean numpy arrays
        """
        n = len(df)
        if n < 2:
            empty = np.zeros(max(0, n), dtype=bool)
            return empty, empty

        open_price = df['open'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)

        # Calculate candle metrics
        candle_range = high - low
        candle_range = np.where(candle_range <= 0, np.nan, candle_range)  # Avoid division by zero

        body = np.abs(close - open_price)
        lower_wick = np.minimum(open_price, close) - low
        upper_wick = high - np.maximum(open_price, close)

        body_ratio = body / candle_range
        lower_wick_ratio = lower_wick / candle_range
        upper_wick_ratio = upper_wick / candle_range

        is_bullish = close > open_price
        is_bearish = close < open_price

        # Previous bar values (shifted by 1)
        prev_open = np.roll(open_price, 1)
        prev_close = np.roll(close, 1)
        prev_open[0] = np.nan
        prev_close[0] = np.nan

        # Bullish patterns
        # Hammer: Long lower wick, small upper wick, decent body
        is_hammer = (
            (lower_wick_ratio >= min_wick_ratio) &
            (upper_wick_ratio < 0.2) &
            (body_ratio >= min_body_ratio) &
            is_bullish
        )

        # Bullish engulfing: Previous bearish, current body engulfs previous
        is_bullish_engulfing = (
            (prev_close < prev_open) &  # Previous was bearish
            (open_price < prev_close) &  # Open below prev close
            (close > prev_open) &  # Close above prev open
            (body_ratio >= 0.5) &
            is_bullish
        )

        switch_long = is_hammer | is_bullish_engulfing

        # Bearish patterns
        # Shooting star: Long upper wick, small lower wick, decent body
        is_shooting_star = (
            (upper_wick_ratio >= min_wick_ratio) &
            (lower_wick_ratio < 0.2) &
            (body_ratio >= min_body_ratio) &
            is_bearish
        )

        # Bearish engulfing: Previous bullish, current body engulfs previous
        is_bearish_engulfing = (
            (prev_close > prev_open) &  # Previous was bullish
            (open_price > prev_close) &  # Open above prev close
            (close < prev_open) &  # Close below prev open
            (body_ratio >= 0.5) &
            is_bearish
        )

        switch_short = is_shooting_star | is_bearish_engulfing

        # Handle NaN from division
        switch_long = np.nan_to_num(switch_long, nan=False).astype(bool)
        switch_short = np.nan_to_num(switch_short, nan=False).astype(bool)

        return switch_long, switch_short

    @staticmethod
    def precompute_order_block_mitigation(
        order_blocks: List[OrderBlock],
        df: pd.DataFrame
    ) -> Dict[int, int]:
        """
        Pre-compute mitigation indices for all order blocks.

        This fixes the O(n^2) bug in update_order_block_mitigation() which was
        called inside the main loop with growing data slices.

        An order block is mitigated when price first returns and touches it.
        This method computes that index once for all OBs.

        Args:
            order_blocks: List of OrderBlock objects
            df: Full OHLCV DataFrame

        Returns:
            Dict mapping OB index to its mitigation bar index (or -1 if never mitigated)
        """
        if not order_blocks or df.empty:
            return {}

        high = df['high'].values
        low = df['low'].values
        n = len(df)

        mitigation_map = {}

        for ob in order_blocks:
            ob_idx = ob.index
            mitigation_idx = -1  # -1 means never mitigated

            # Search for first bar after OB creation that touches it
            for i in range(ob_idx + 1, n):
                if ob.direction == 'bullish':
                    # Bullish OB mitigated when price drops into it
                    if low[i] <= ob.high:
                        mitigation_idx = i
                        break
                else:
                    # Bearish OB mitigated when price rises into it
                    if high[i] >= ob.low:
                        mitigation_idx = i
                        break

            mitigation_map[ob_idx] = mitigation_idx

        return mitigation_map

    @staticmethod
    def compute_entry_window_mask(
        df: pd.DataFrame,
        entry_start: time = time(9, 45),
        entry_cutoff: time = time(15, 30)
    ) -> np.ndarray:
        """
        Vectorized computation of entry time window mask.

        Args:
            df: DataFrame with DatetimeIndex (timestamps in UTC)
            entry_start: Start time for entries in ET (default 9:45 AM)
            entry_cutoff: End time for entries in ET (default 3:30 PM)

        Returns:
            Boolean numpy array where True = within entry window
        """
        if df.empty:
            return np.zeros(0, dtype=bool)

        # Extract time from index - CONVERT FROM UTC TO ET first
        # Data timestamps are in UTC but entry times are in Eastern Time
        try:
            from src.utils.timezone import tz
            # Convert index to ET and extract time
            et_times = []
            for ts in df.index:
                try:
                    et_ts = tz.from_utc(ts.to_pydatetime())
                    et_times.append(et_ts.time())
                except Exception:
                    # Fallback to raw time if conversion fails
                    et_times.append(ts.time())
            mask = np.array([(entry_start <= t <= entry_cutoff) for t in et_times])
        except (AttributeError, TypeError):
            # If index doesn't have time component, assume all valid
            mask = np.ones(len(df), dtype=bool)

        return mask

    @staticmethod
    def compute_high_volume_mask(
        df: pd.DataFrame,
        rvol_threshold: float = 1.5,
        lookback: int = 20
    ) -> np.ndarray:
        """
        Vectorized computation of high volume bars.

        Args:
            df: OHLCV DataFrame with 'volume' column
            rvol_threshold: Minimum RVOL for high volume (default 1.5)
            lookback: Rolling window for average volume

        Returns:
            Boolean numpy array where True = high volume bar
        """
        if df.empty or 'volume' not in df.columns:
            return np.ones(len(df), dtype=bool)  # No filter if no volume

        rvol = ICTIndicators.calculate_rvol(df, lookback)
        return (rvol >= rvol_threshold).values

    @staticmethod
    def compute_momentum_alignment(
        df: pd.DataFrame,
        ema_period: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute momentum alignment for directional filtering.

        Uses EMA crossover to determine short-term momentum direction.
        Aligns entries with prevailing momentum for higher win rate.

        Args:
            df: OHLCV DataFrame
            ema_period: EMA period for momentum (default 10)

        Returns:
            Tuple of (bullish_momentum, bearish_momentum) boolean arrays
            - bullish_momentum: True when close > EMA (favor longs)
            - bearish_momentum: True when close < EMA (favor shorts)
        """
        if df.empty or len(df) < ema_period:
            n = len(df) if not df.empty else 0
            return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

        close = df['close'].values

        # Calculate EMA
        ema = pd.Series(close).ewm(span=ema_period, adjust=False).mean().values

        # Momentum alignment
        bullish_momentum = close > ema
        bearish_momentum = close < ema

        return bullish_momentum, bearish_momentum

    @staticmethod
    def compute_price_momentum(
        df: pd.DataFrame,
        lookback: int = 5
    ) -> np.ndarray:
        """
        Compute simple price momentum over lookback period.

        Args:
            df: OHLCV DataFrame
            lookback: Bars to measure momentum over (default 5)

        Returns:
            Array of momentum values (positive = bullish, negative = bearish)
        """
        if df.empty or len(df) < lookback:
            return np.zeros(len(df) if not df.empty else 0, dtype=float)

        close = df['close'].values
        momentum = np.zeros(len(close), dtype=float)

        for i in range(lookback, len(close)):
            momentum[i] = (close[i] - close[i - lookback]) / close[i - lookback]

        return momentum
