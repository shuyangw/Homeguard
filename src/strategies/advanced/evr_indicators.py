"""
EVR (Effort vs. Result) Indicators.

A Volume Spread Analysis (VSA) based strategy that detects absorption:
when massive volume produces minimal price movement, indicating smart
money passively absorbing aggressive orders at trend exhaustion points.

Core Concept (Newton's Third Law Applied to Markets):
- Normal: Heavy volume -> Large price movement
- Anomaly: Heavy volume -> Tiny price movement (ABSORPTION)

Metrics:
- Relative Volume (Effort) = Volume / SMA(Volume, period)
- Relative Spread (Result) = (High - Low) / SMA(High - Low, period)
- Absorption = High Effort + Low Result
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EVRSignal:
    """Current state snapshot for EVR strategy."""
    timestamp: pd.Timestamp
    close: float
    volume: float
    rel_volume: float
    rel_spread: float
    is_absorption: bool
    context: str
    rsi: float
    bb_upper: float
    bb_middle: float
    bb_lower: float


class EVRIndicators:
    """
    Static indicator calculations for Effort vs. Result strategy.

    Provides volume spread analysis and absorption detection.
    """

    @staticmethod
    def calculate_relative_volume(
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate relative volume (Effort).

        Effort = Volume / SMA(Volume, period)

        Args:
            volume: Volume series
            period: Lookback period for SMA (default 20)

        Returns:
            Relative volume series (1.0 = average, 2.0 = 200% of average)
        """
        avg_volume = volume.rolling(window=period, min_periods=period).mean()
        rel_volume = volume / avg_volume
        rel_volume = rel_volume.replace([np.inf, -np.inf], np.nan)
        return rel_volume

    @staticmethod
    def calculate_relative_spread(
        high: pd.Series,
        low: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate relative spread (Result).

        Result = (High - Low) / SMA(High - Low, period)

        Args:
            high: High price series
            low: Low price series
            period: Lookback period for SMA (default 20)

        Returns:
            Relative spread series (1.0 = average, 0.5 = 50% of average)
        """
        spread = high - low
        avg_spread = spread.rolling(window=period, min_periods=period).mean()
        rel_spread = spread / avg_spread
        rel_spread = rel_spread.replace([np.inf, -np.inf], np.nan)
        return rel_spread

    @staticmethod
    def detect_absorption(
        rel_volume: pd.Series,
        rel_spread: pd.Series,
        vol_threshold: float = 2.0,
        spread_threshold: float = 0.8
    ) -> pd.Series:
        """
        Detect absorption (squat) candles.

        Absorption occurs when:
        - Relative Volume > vol_threshold (high effort)
        - Relative Spread < spread_threshold (low result)

        Args:
            rel_volume: Relative volume series
            rel_spread: Relative spread series
            vol_threshold: Minimum relative volume (default 2.0 = 200%)
            spread_threshold: Maximum relative spread (default 0.8 = 80%)

        Returns:
            Boolean series indicating absorption candles
        """
        is_high_effort = rel_volume > vol_threshold
        is_low_result = rel_spread < spread_threshold
        is_absorption = is_high_effort & is_low_result
        return is_absorption.fillna(False)

    @staticmethod
    def calculate_rsi(
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            close: Close price series
            period: RSI period (default 14)

        Returns:
            RSI series (0-100)
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger(
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            close: Close price series
            period: SMA period (default 20)
            std_dev: Number of standard deviations (default 2.0)

        Returns:
            Tuple of (upper, middle, lower) band series
        """
        middle = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    @staticmethod
    def calculate_context(
        close: pd.Series,
        rsi: pd.Series,
        bb_lower: pd.Series,
        bb_upper: pd.Series,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Determine market context: oversold or overbought.

        Oversold: Price below lower BB OR RSI < oversold threshold
        Overbought: Price above upper BB OR RSI > overbought threshold

        Args:
            close: Close price series
            rsi: RSI series
            bb_lower: Lower Bollinger Band
            bb_upper: Upper Bollinger Band
            rsi_oversold: RSI oversold threshold (default 30)
            rsi_overbought: RSI overbought threshold (default 70)

        Returns:
            Tuple of (is_oversold, is_overbought) boolean series
        """
        is_oversold = (close < bb_lower) | (rsi < rsi_oversold)
        is_overbought = (close > bb_upper) | (rsi > rsi_overbought)
        return is_oversold.fillna(False), is_overbought.fillna(False)

    @staticmethod
    def calculate_all_indicators(
        df: pd.DataFrame,
        vol_lookback: int = 20,
        spread_lookback: int = 20,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ) -> pd.DataFrame:
        """
        Calculate all EVR indicators.

        Args:
            df: OHLCV DataFrame
            vol_lookback: Volume lookback period
            spread_lookback: Spread lookback period
            rsi_period: RSI period
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviations

        Returns:
            DataFrame with all indicators
        """
        result = pd.DataFrame(index=df.index)

        # Relative volume and spread
        result['rel_volume'] = EVRIndicators.calculate_relative_volume(
            df['volume'], vol_lookback
        )
        result['rel_spread'] = EVRIndicators.calculate_relative_spread(
            df['high'], df['low'], spread_lookback
        )

        # RSI
        result['rsi'] = EVRIndicators.calculate_rsi(df['close'], rsi_period)

        # Bollinger Bands
        result['bb_upper'], result['bb_middle'], result['bb_lower'] = \
            EVRIndicators.calculate_bollinger(df['close'], bb_period, bb_std)

        # Add OHLCV for convenience
        result['open'] = df['open']
        result['high'] = df['high']
        result['low'] = df['low']
        result['close'] = df['close']
        result['volume'] = df['volume']

        return result

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        vol_lookback: int = 20,
        spread_lookback: int = 20,
        vol_threshold: float = 2.0,
        spread_threshold: float = 0.8,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_period: int = 20,
        bb_std: float = 2.0,
        require_confirmation: bool = True
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long/short entry/exit signals.

        Signal Logic:
        1. Calculate relative volume (effort) and spread (result)
        2. Detect absorption candles (high effort, low result)
        3. Check context: oversold for longs, overbought for shorts
        4. Require confirmation: next bar closes above/below absorption bar
        5. Exit at mean reversion target (middle Bollinger Band)

        Args:
            df: OHLCV DataFrame
            vol_lookback: Volume lookback period
            spread_lookback: Spread lookback period
            vol_threshold: Minimum relative volume for absorption
            spread_threshold: Maximum relative spread for absorption
            rsi_period: RSI period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviations
            require_confirmation: Require next bar confirmation

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        if df.empty:
            empty = pd.Series(dtype=bool)
            return empty, empty, empty, empty

        # Calculate indicators
        rel_volume = EVRIndicators.calculate_relative_volume(
            df['volume'], vol_lookback
        )
        rel_spread = EVRIndicators.calculate_relative_spread(
            df['high'], df['low'], spread_lookback
        )
        rsi = EVRIndicators.calculate_rsi(df['close'], rsi_period)
        bb_upper, bb_middle, bb_lower = EVRIndicators.calculate_bollinger(
            df['close'], bb_period, bb_std
        )

        # Detect absorption candles
        is_absorption = EVRIndicators.detect_absorption(
            rel_volume, rel_spread, vol_threshold, spread_threshold
        )

        # Context: Is market extended?
        is_oversold, is_overbought = EVRIndicators.calculate_context(
            df['close'], rsi, bb_lower, bb_upper, rsi_oversold, rsi_overbought
        )

        # Potential reversals (absorption in context)
        long_setup = is_absorption & is_oversold
        short_setup = is_absorption & is_overbought

        if require_confirmation:
            # Confirmation: Next bar closes above/below absorption bar
            long_confirm = df['close'] > df['high'].shift(1)
            short_confirm = df['close'] < df['low'].shift(1)

            # Entry signals (setup on prior bar + confirmation on current)
            long_entries = long_setup.shift(1).fillna(False) & long_confirm
            short_entries = short_setup.shift(1).fillna(False) & short_confirm
        else:
            # Direct entry on absorption candle (more aggressive)
            long_entries = long_setup
            short_entries = short_setup

        # Exit signals (mean reversion target - middle Bollinger Band)
        long_exits = df['close'] > bb_middle
        short_exits = df['close'] < bb_middle

        # Clean up NaN values
        long_entries = long_entries.fillna(False).astype(bool)
        long_exits = long_exits.fillna(False).astype(bool)
        short_entries = short_entries.fillna(False).astype(bool)
        short_exits = short_exits.fillna(False).astype(bool)

        return long_entries, long_exits, short_entries, short_exits

    @staticmethod
    def get_current_signal(
        df: pd.DataFrame,
        vol_lookback: int = 20,
        spread_lookback: int = 20,
        vol_threshold: float = 2.0,
        spread_threshold: float = 0.8,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ) -> EVRSignal:
        """
        Get current EVR signal state for live trading.

        Args:
            df: OHLCV DataFrame (must have enough history)
            vol_lookback: Volume lookback period
            spread_lookback: Spread lookback period
            vol_threshold: Minimum relative volume for absorption
            spread_threshold: Maximum relative spread for absorption
            rsi_period: RSI period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviations

        Returns:
            EVRSignal with current state
        """
        min_required = max(vol_lookback, spread_lookback, rsi_period, bb_period)
        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data: need {min_required} bars, got {len(df)}"
            )

        indicators = EVRIndicators.calculate_all_indicators(
            df,
            vol_lookback=vol_lookback,
            spread_lookback=spread_lookback,
            rsi_period=rsi_period,
            bb_period=bb_period,
            bb_std=bb_std,
        )

        latest = indicators.iloc[-1]

        # Determine context
        is_oversold = (latest['close'] < latest['bb_lower']) or (latest['rsi'] < rsi_oversold)
        is_overbought = (latest['close'] > latest['bb_upper']) or (latest['rsi'] > rsi_overbought)

        if is_oversold:
            context = 'oversold'
        elif is_overbought:
            context = 'overbought'
        else:
            context = 'neutral'

        # Check absorption
        is_absorption = (
            latest['rel_volume'] > vol_threshold and
            latest['rel_spread'] < spread_threshold
        )

        return EVRSignal(
            timestamp=df.index[-1],
            close=latest['close'],
            volume=latest['volume'],
            rel_volume=latest['rel_volume'],
            rel_spread=latest['rel_spread'],
            is_absorption=is_absorption,
            context=context,
            rsi=latest['rsi'],
            bb_upper=latest['bb_upper'],
            bb_middle=latest['bb_middle'],
            bb_lower=latest['bb_lower'],
        )

    # =========================================================================
    # IMPROVEMENT METHODS
    # =========================================================================

    @staticmethod
    def calculate_atr(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range for stop loss sizing.

        Args:
            df: OHLCV DataFrame
            period: ATR period (default 14)

        Returns:
            ATR series
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=period).mean()
        return atr

    @staticmethod
    def get_btc_regime(
        btc_close: pd.Series,
        sma_period: int = 50
    ) -> pd.Series:
        """
        Calculate BTC regime filter.

        Simple rule: BTC > SMA = bullish, else bearish.
        When BTC is weak, crypto mean reversion is riskier.

        Args:
            btc_close: BTC close price series
            sma_period: SMA period for regime detection (default 50)

        Returns:
            Series of 'bullish' or 'bearish' regime labels
        """
        sma = btc_close.rolling(window=sma_period, min_periods=sma_period).mean()
        regime = pd.Series('bearish', index=btc_close.index)
        regime[btc_close > sma] = 'bullish'
        return regime

    @staticmethod
    def calculate_position_size(
        close: pd.Series,
        target_vol: float = 0.20,
        lookback: int = 65,
        annualization_factor: float = 365.0,
        min_position: float = 0.25,
        max_position: float = 1.00,
    ) -> pd.Series:
        """
        Calculate volatility-scaled position size.

        Formula: position_size = target_vol / realized_vol
        Clipped to [min_position, max_position]

        Args:
            close: Close price series
            target_vol: Target annualized volatility (default 0.20 = 20%)
            lookback: Rolling window for volatility calculation (default 65)
            annualization_factor: Factor to annualize volatility
                                  (365 for daily, 8760 for hourly)
            min_position: Minimum position size (default 0.25 = 25%)
            max_position: Maximum position size (default 1.00 = 100%)

        Returns:
            Series of position sizes in [min_position, max_position] range
        """
        from src.features import close_to_close_rv
        returns = close.pct_change()
        realized_vol = close_to_close_rv(
            returns, window=lookback,
            annualization_factor=annualization_factor,
        )

        # Avoid division by zero
        realized_vol = realized_vol.replace(0, np.nan)

        raw_position = target_vol / realized_vol
        position_size = raw_position.clip(min_position, max_position)

        # Fill NaN with mid-range position
        position_size = position_size.fillna((min_position + max_position) / 2)

        return position_size

    @staticmethod
    def generate_signals_with_stops(
        df: pd.DataFrame,
        vol_lookback: int = 20,
        spread_lookback: int = 20,
        vol_threshold: float = 2.0,
        spread_threshold: float = 0.8,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        require_confirmation: bool = True
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate signals with ATR-based stop loss levels.

        Returns stop loss price levels in addition to entry/exit signals.

        Args:
            df: OHLCV DataFrame
            vol_lookback: Volume lookback period
            spread_lookback: Spread lookback period
            vol_threshold: Minimum relative volume for absorption
            spread_threshold: Maximum relative spread for absorption
            rsi_period: RSI period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviations
            atr_period: ATR period for stop calculation
            atr_multiplier: ATR multiplier for stop distance
            require_confirmation: Require next bar confirmation

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits,
                     long_stop_price, short_stop_price)
        """
        # Get base signals
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals(
                df=df,
                vol_lookback=vol_lookback,
                spread_lookback=spread_lookback,
                vol_threshold=vol_threshold,
                spread_threshold=spread_threshold,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                bb_period=bb_period,
                bb_std=bb_std,
                require_confirmation=require_confirmation,
            )

        # Calculate ATR for stop sizing
        atr = EVRIndicators.calculate_atr(df, atr_period)

        # Stop prices: entry price +/- ATR * multiplier
        # For long: stop below entry
        # For short: stop above entry
        long_stop_price = df['close'] - (atr * atr_multiplier)
        short_stop_price = df['close'] + (atr * atr_multiplier)

        # Add stop-based exits
        # Long exit if price hits stop
        long_stop_hit = df['low'] <= long_stop_price.shift(1)
        # Short exit if price hits stop
        short_stop_hit = df['high'] >= short_stop_price.shift(1)

        long_exits = long_exits | long_stop_hit.fillna(False)
        short_exits = short_exits | short_stop_hit.fillna(False)

        return (long_entries, long_exits, short_entries, short_exits,
                long_stop_price, short_stop_price)

    @staticmethod
    def generate_signals_with_btc_regime(
        df: pd.DataFrame,
        btc_close: pd.Series,
        vol_lookback: int = 20,
        spread_lookback: int = 20,
        vol_threshold: float = 2.0,
        spread_threshold: float = 0.8,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_period: int = 20,
        bb_std: float = 2.0,
        btc_sma_period: int = 50,
        regime_mode: str = 'force_exit',
        require_confirmation: bool = True
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate signals with BTC regime filter overlay.

        Two modes:
        - 'block_entry': Don't enter new positions when BTC bearish
        - 'force_exit': Exit all positions when BTC turns bearish

        Args:
            df: OHLCV DataFrame for the asset
            btc_close: BTC close price series
            vol_lookback: Volume lookback period
            spread_lookback: Spread lookback period
            vol_threshold: Minimum relative volume for absorption
            spread_threshold: Maximum relative spread for absorption
            rsi_period: RSI period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviations
            btc_sma_period: SMA period for BTC regime (default 50)
            regime_mode: 'block_entry' or 'force_exit' (default 'force_exit')
            require_confirmation: Require next bar confirmation

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits, btc_regime)
        """
        # Get base signals
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals(
                df=df,
                vol_lookback=vol_lookback,
                spread_lookback=spread_lookback,
                vol_threshold=vol_threshold,
                spread_threshold=spread_threshold,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                bb_period=bb_period,
                bb_std=bb_std,
                require_confirmation=require_confirmation,
            )

        # Align BTC close with asset data
        btc_aligned = btc_close.reindex(df.index, method='ffill')

        # Get BTC regime
        btc_regime = EVRIndicators.get_btc_regime(btc_aligned, btc_sma_period)
        is_btc_bearish = btc_regime == 'bearish'

        if regime_mode == 'block_entry':
            # Block new entries during bearish BTC
            long_entries = long_entries & ~is_btc_bearish
            short_entries = short_entries & ~is_btc_bearish

        elif regime_mode == 'force_exit':
            # Force exit when BTC turns bearish
            btc_turns_bearish = is_btc_bearish & (~is_btc_bearish.shift(1).fillna(False))
            long_exits = long_exits | btc_turns_bearish
            short_exits = short_exits | btc_turns_bearish
            # Also block new entries during bearish
            long_entries = long_entries & ~is_btc_bearish
            short_entries = short_entries & ~is_btc_bearish

        return long_entries, long_exits, short_entries, short_exits, btc_regime

    @staticmethod
    def generate_signals_improved(
        df: pd.DataFrame,
        btc_close: pd.Series = None,
        vol_lookback: int = 20,
        spread_lookback: int = 20,
        vol_threshold: float = 2.5,
        spread_threshold: float = 0.6,
        rsi_period: int = 14,
        rsi_oversold: float = 25,
        rsi_overbought: float = 75,
        bb_period: int = 20,
        bb_std: float = 2.5,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        btc_sma_period: int = 50,
        use_btc_filter: bool = True,
        use_atr_stops: bool = True,
        require_confirmation: bool = True
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate improved EVR signals with all enhancements.

        Improvements:
        1. Tighter thresholds (vol_threshold=2.5, spread_threshold=0.6)
        2. Stricter context (RSI 25/75 instead of 30/70)
        3. Wider Bollinger Bands (2.5 std)
        4. BTC regime filter (optional)
        5. ATR-based stop losses (optional)

        Args:
            df: OHLCV DataFrame
            btc_close: BTC close price series (optional, for regime filter)
            vol_lookback: Volume lookback period
            spread_lookback: Spread lookback period
            vol_threshold: Minimum relative volume (tighter: 2.5)
            spread_threshold: Maximum relative spread (tighter: 0.6)
            rsi_period: RSI period
            rsi_oversold: RSI oversold (stricter: 25)
            rsi_overbought: RSI overbought (stricter: 75)
            bb_period: Bollinger Band period
            bb_std: Bollinger Band std (wider: 2.5)
            atr_period: ATR period
            atr_multiplier: ATR multiplier for stops
            btc_sma_period: BTC SMA period for regime
            use_btc_filter: Whether to use BTC regime filter
            use_atr_stops: Whether to use ATR-based stops
            require_confirmation: Require next bar confirmation

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        # Start with base signals using tighter parameters
        long_entries, long_exits, short_entries, short_exits = \
            EVRIndicators.generate_signals(
                df=df,
                vol_lookback=vol_lookback,
                spread_lookback=spread_lookback,
                vol_threshold=vol_threshold,
                spread_threshold=spread_threshold,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                bb_period=bb_period,
                bb_std=bb_std,
                require_confirmation=require_confirmation,
            )

        # Apply BTC regime filter if enabled and BTC data provided
        if use_btc_filter and btc_close is not None:
            btc_aligned = btc_close.reindex(df.index, method='ffill')
            btc_regime = EVRIndicators.get_btc_regime(btc_aligned, btc_sma_period)
            is_btc_bearish = btc_regime == 'bearish'

            # Force exit on BTC bearish turn
            btc_turns_bearish = is_btc_bearish & (~is_btc_bearish.shift(1).fillna(False))
            long_exits = long_exits | btc_turns_bearish
            short_exits = short_exits | btc_turns_bearish

            # Block entries during bearish
            long_entries = long_entries & ~is_btc_bearish
            short_entries = short_entries & ~is_btc_bearish

        # Apply ATR-based stops if enabled
        if use_atr_stops:
            atr = EVRIndicators.calculate_atr(df, atr_period)
            long_stop_price = df['close'] - (atr * atr_multiplier)
            short_stop_price = df['close'] + (atr * atr_multiplier)

            long_stop_hit = df['low'] <= long_stop_price.shift(1)
            short_stop_hit = df['high'] >= short_stop_price.shift(1)

            long_exits = long_exits | long_stop_hit.fillna(False)
            short_exits = short_exits | short_stop_hit.fillna(False)

        return long_entries, long_exits, short_entries, short_exits
