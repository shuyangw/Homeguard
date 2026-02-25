"""
Fractal Regime Switching (FRS) Indicators.

A meta-strategy that uses the Hurst Exponent to detect market regimes
and routes to the appropriate sub-strategy:

- H > 0.60 (Trending): Deploy Donchian Breakout
- H < 0.40 (Mean-Reverting): Deploy Bollinger + RSI Mean Reversion
- 0.40 <= H <= 0.60 (Noise): Go to cash, do not trade

The Hurst Exponent measures long-term memory in a time series:
- H < 0.5: Anti-persistent (mean-reverting)
- H = 0.5: Random walk (Brownian motion)
- H > 0.5: Persistent (trending)
"""

from dataclasses import dataclass
from typing import Tuple
from enum import Enum

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional Hurst exponent import
try:
    from hurst import compute_Hc
    HAS_HURST = True
except ImportError:
    HAS_HURST = False
    logger.info("hurst package not installed - Hurst exponent will use fallback")


class FRSRegime(Enum):
    """Market regime classification based on Hurst exponent."""
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    NOISE = "NOISE"


@dataclass
class FRSSignal:
    """Current state snapshot for FRS strategy."""
    timestamp: pd.Timestamp
    close: float
    hurst: float
    hurst_smoothed: float
    regime: str
    donchian_upper: float
    donchian_lower: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    rsi: float


class FRSIndicators:
    """
    Static indicator calculations for Fractal Regime Switching strategy.

    Provides Hurst-based regime detection and sub-strategy indicators.
    """

    @staticmethod
    def calculate_hurst(series: pd.Series, window: int = 100) -> pd.Series:
        """
        Calculate rolling Hurst exponent using R/S analysis.

        The Hurst exponent measures long-term memory/persistence:
        - H < 0.5: Anti-persistent (mean-reverting)
        - H = 0.5: Random walk (Brownian motion)
        - H > 0.5: Persistent (trending)

        Args:
            series: Price series (will be converted to returns internally)
            window: Rolling window size (default 100 bars)

        Returns:
            Series of Hurst exponent values (0-1 range)
        """
        if not HAS_HURST:
            logger.warning("Hurst package not available, using 0.5 fallback")
            return pd.Series(0.5, index=series.index)

        # Calculate returns (Hurst is computed on returns, not prices)
        returns = series.pct_change().dropna()

        def calc_hurst_single(x: np.ndarray) -> float:
            """Calculate Hurst for a single window."""
            if len(x) < 20:
                return 0.5
            try:
                H, _, _ = compute_Hc(x, kind='change', simplified=True)
                return max(0.0, min(1.0, H))
            except Exception:
                return 0.5

        # Rolling Hurst calculation
        hurst_values = returns.rolling(
            window=window,
            min_periods=max(20, window // 2)
        ).apply(calc_hurst_single, raw=True)

        # Reindex to match original series
        result = pd.Series(0.5, index=series.index)
        result.loc[hurst_values.index] = hurst_values

        return result

    @staticmethod
    def smooth_hurst(hurst: pd.Series, span: int = 10) -> pd.Series:
        """
        Apply EMA smoothing to Hurst values to prevent flickering.

        Args:
            hurst: Raw Hurst exponent series
            span: EMA span for smoothing

        Returns:
            Smoothed Hurst series
        """
        return hurst.ewm(span=span, adjust=False).mean()

    @staticmethod
    def classify_regime(
        hurst: float,
        trend_threshold: float = 0.60,
        mr_threshold: float = 0.40
    ) -> str:
        """
        Classify market regime based on Hurst value.

        Args:
            hurst: Hurst exponent value
            trend_threshold: Above this = TRENDING
            mr_threshold: Below this = MEAN_REVERTING

        Returns:
            Regime string: 'TRENDING', 'MEAN_REVERTING', or 'NOISE'
        """
        if pd.isna(hurst):
            return FRSRegime.NOISE.value
        if hurst > trend_threshold:
            return FRSRegime.TRENDING.value
        elif hurst < mr_threshold:
            return FRSRegime.MEAN_REVERTING.value
        else:
            return FRSRegime.NOISE.value

    @staticmethod
    def classify_regime_series(
        hurst: pd.Series,
        trend_threshold: float = 0.60,
        mr_threshold: float = 0.40
    ) -> pd.Series:
        """
        Classify regime for entire series.

        Args:
            hurst: Hurst exponent series
            trend_threshold: Above this = TRENDING
            mr_threshold: Below this = MEAN_REVERTING

        Returns:
            Series of regime classifications
        """
        return hurst.apply(
            lambda h: FRSIndicators.classify_regime(h, trend_threshold, mr_threshold)
        )

    @staticmethod
    def calculate_donchian(
        df: pd.DataFrame,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Donchian Channel for trend breakout signals.

        Upper = highest high over period
        Lower = lowest low over period

        Entry signals:
        - Long: Close > Upper (breakout up)
        - Short: Close < Lower (breakout down)

        Args:
            df: OHLCV DataFrame with 'high' and 'low' columns
            period: Lookback period (default 20)

        Returns:
            Tuple of (upper, lower) Series
        """
        upper = df['high'].rolling(window=period, min_periods=period).max()
        lower = df['low'].rolling(window=period, min_periods=period).min()
        return upper, lower

    @staticmethod
    def calculate_bollinger(
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands for mean reversion signals.

        Middle = SMA(period)
        Upper = Middle + std_dev * StdDev
        Lower = Middle - std_dev * StdDev

        Entry signals:
        - Long: Close < Lower (oversold)
        - Short: Close > Upper (overbought)

        Args:
            close: Close price series
            period: SMA period (default 20)
            std_dev: Number of standard deviations (default 2.0)

        Returns:
            Tuple of (upper, middle, lower) Series
        """
        middle = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        RSI < 30 = oversold (confirms mean reversion long)
        RSI > 70 = overbought (confirms mean reversion short)

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
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for position sizing and stops.

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
    def calculate_all_indicators(
        df: pd.DataFrame,
        hurst_window: int = 100,
        hurst_smooth_span: int = 10,
        donchian_period: int = 20,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        rsi_period: int = 14,
        atr_period: int = 14,
    ) -> pd.DataFrame:
        """
        Calculate all FRS indicators.

        Args:
            df: OHLCV DataFrame
            hurst_window: Window for Hurst calculation
            hurst_smooth_span: EMA span for Hurst smoothing
            donchian_period: Donchian channel period
            bollinger_period: Bollinger band period
            bollinger_std: Bollinger standard deviations
            rsi_period: RSI period
            atr_period: ATR period

        Returns:
            DataFrame with all indicators
        """
        result = pd.DataFrame(index=df.index)

        # Hurst exponent
        result['hurst'] = FRSIndicators.calculate_hurst(df['close'], hurst_window)
        result['hurst_smoothed'] = FRSIndicators.smooth_hurst(
            result['hurst'], hurst_smooth_span
        )

        # Donchian channels
        result['donchian_upper'], result['donchian_lower'] = \
            FRSIndicators.calculate_donchian(df, donchian_period)

        # Bollinger bands
        result['bb_upper'], result['bb_middle'], result['bb_lower'] = \
            FRSIndicators.calculate_bollinger(df['close'], bollinger_period, bollinger_std)

        # RSI
        result['rsi'] = FRSIndicators.calculate_rsi(df['close'], rsi_period)

        # ATR
        result['atr'] = FRSIndicators.calculate_atr(df, atr_period)

        # Add OHLC for convenience
        result['open'] = df['open']
        result['high'] = df['high']
        result['low'] = df['low']
        result['close'] = df['close']

        return result

    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
        hurst_window: int = 100,
        hurst_smooth_span: int = 10,
        trend_threshold: float = 0.60,
        mr_threshold: float = 0.40,
        donchian_period: int = 20,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long/short entry/exit signals based on regime.

        Logic:
        - TRENDING regime: Donchian breakout signals
        - MEAN_REVERTING regime: Bollinger + RSI signals
        - NOISE regime: No signals (stay in cash)

        Args:
            df: OHLCV DataFrame
            hurst_window: Window for Hurst calculation
            hurst_smooth_span: EMA span for Hurst smoothing
            trend_threshold: Hurst above this = TRENDING
            mr_threshold: Hurst below this = MEAN_REVERTING
            donchian_period: Donchian channel period
            bollinger_period: Bollinger band period
            bollinger_std: Bollinger standard deviations
            rsi_period: RSI period
            rsi_overbought: RSI level for overbought
            rsi_oversold: RSI level for oversold

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits, regime)
        """
        # Calculate indicators
        indicators = FRSIndicators.calculate_all_indicators(
            df,
            hurst_window=hurst_window,
            hurst_smooth_span=hurst_smooth_span,
            donchian_period=donchian_period,
            bollinger_period=bollinger_period,
            bollinger_std=bollinger_std,
            rsi_period=rsi_period,
        )

        # Classify regime
        regime = FRSIndicators.classify_regime_series(
            indicators['hurst_smoothed'],
            trend_threshold=trend_threshold,
            mr_threshold=mr_threshold
        )

        close = df['close']

        # Initialize signals
        long_entries = pd.Series(False, index=df.index)
        long_exits = pd.Series(False, index=df.index)
        short_entries = pd.Series(False, index=df.index)
        short_exits = pd.Series(False, index=df.index)

        # TRENDING regime: Donchian Breakout
        is_trending = regime == FRSRegime.TRENDING.value
        donchian_upper = indicators['donchian_upper']
        donchian_lower = indicators['donchian_lower']

        # Entry on breakout (close exceeds previous channel)
        trend_long_entry = (close > donchian_upper.shift(1)) & is_trending
        trend_short_entry = (close < donchian_lower.shift(1)) & is_trending

        # Exit on opposite channel touch or regime change
        trend_long_exit = (close < donchian_lower.shift(1)) & is_trending
        trend_short_exit = (close > donchian_upper.shift(1)) & is_trending

        # MEAN_REVERTING regime: Bollinger + RSI
        is_mr = regime == FRSRegime.MEAN_REVERTING.value
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        bb_middle = indicators['bb_middle']
        rsi = indicators['rsi']

        # Entry at band extremes with RSI confirmation
        mr_long_entry = (close < bb_lower) & (rsi < rsi_oversold) & is_mr
        mr_short_entry = (close > bb_upper) & (rsi > rsi_overbought) & is_mr

        # Exit at middle band (mean reversion target)
        mr_long_exit = (close > bb_middle) & is_mr
        mr_short_exit = (close < bb_middle) & is_mr

        # Combine signals
        long_entries = trend_long_entry | mr_long_entry
        short_entries = trend_short_entry | mr_short_entry

        long_exits = trend_long_exit | mr_long_exit
        short_exits = trend_short_exit | mr_short_exit

        # Force exit on regime change to NOISE
        regime_to_noise = (regime == FRSRegime.NOISE.value) & (regime.shift(1) != FRSRegime.NOISE.value)
        long_exits = long_exits | regime_to_noise
        short_exits = short_exits | regime_to_noise

        return long_entries, long_exits, short_entries, short_exits, regime

    @staticmethod
    def get_current_signal(
        df: pd.DataFrame,
        hurst_window: int = 100,
        hurst_smooth_span: int = 10,
        trend_threshold: float = 0.60,
        mr_threshold: float = 0.40,
        donchian_period: int = 20,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        rsi_period: int = 14,
    ) -> FRSSignal:
        """
        Get current FRS signal state for live trading.

        Args:
            df: OHLCV DataFrame (must have enough history)
            hurst_window: Window for Hurst calculation
            hurst_smooth_span: EMA span for Hurst smoothing
            trend_threshold: Hurst above this = TRENDING
            mr_threshold: Hurst below this = MEAN_REVERTING
            donchian_period: Donchian channel period
            bollinger_period: Bollinger band period
            bollinger_std: Bollinger standard deviations
            rsi_period: RSI period

        Returns:
            FRSSignal with current state
        """
        min_required = max(hurst_window, donchian_period, bollinger_period, rsi_period)
        if len(df) < min_required:
            raise ValueError(f"Insufficient data: need {min_required} bars, got {len(df)}")

        indicators = FRSIndicators.calculate_all_indicators(
            df,
            hurst_window=hurst_window,
            hurst_smooth_span=hurst_smooth_span,
            donchian_period=donchian_period,
            bollinger_period=bollinger_period,
            bollinger_std=bollinger_std,
            rsi_period=rsi_period,
        )

        latest = indicators.iloc[-1]
        regime = FRSIndicators.classify_regime(
            latest['hurst_smoothed'],
            trend_threshold=trend_threshold,
            mr_threshold=mr_threshold
        )

        return FRSSignal(
            timestamp=df.index[-1],
            close=latest['close'],
            hurst=latest['hurst'],
            hurst_smoothed=latest['hurst_smoothed'],
            regime=regime,
            donchian_upper=latest['donchian_upper'],
            donchian_lower=latest['donchian_lower'],
            bb_upper=latest['bb_upper'],
            bb_middle=latest['bb_middle'],
            bb_lower=latest['bb_lower'],
            rsi=latest['rsi'],
        )

    # =========================================================================
    # IMPROVEMENT METHODS (Phase 1-3)
    # =========================================================================

    @staticmethod
    def get_btc_regime(
        btc_close: pd.Series,
        sma_period: int = 50
    ) -> pd.Series:
        """
        Calculate BTC regime filter.

        Simple rule: BTC > SMA = bullish, else bearish.
        When BTC is weak, altcoins typically suffer more.

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
    def generate_signals_with_regime(
        df: pd.DataFrame,
        btc_close: pd.Series,
        hurst_window: int = 100,
        hurst_smooth_span: int = 10,
        trend_threshold: float = 0.60,
        mr_threshold: float = 0.40,
        donchian_period: int = 20,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        btc_sma_period: int = 50,
        regime_mode: str = 'force_exit',
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate signals with BTC regime filter overlay.

        Combines Hurst-based regime detection with BTC market health filter.

        Two modes:
        - 'block_entry': Don't enter new positions when BTC bearish
        - 'force_exit': Exit all positions when BTC turns bearish

        Args:
            df: OHLCV DataFrame for the asset
            btc_close: BTC close price series (must align with df index)
            hurst_window: Window for Hurst calculation
            hurst_smooth_span: EMA span for Hurst smoothing
            trend_threshold: Hurst above this = TRENDING
            mr_threshold: Hurst below this = MEAN_REVERTING
            donchian_period: Donchian channel period
            bollinger_period: Bollinger band period
            bollinger_std: Bollinger standard deviations
            rsi_period: RSI period
            rsi_overbought: RSI level for overbought
            rsi_oversold: RSI level for oversold
            btc_sma_period: SMA period for BTC regime (default 50)
            regime_mode: 'block_entry' or 'force_exit' (default 'force_exit')

        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits,
                     hurst_regime, btc_regime)
        """
        # Get base FRS signals
        long_entries, long_exits, short_entries, short_exits, hurst_regime = \
            FRSIndicators.generate_signals(
                df=df,
                hurst_window=hurst_window,
                hurst_smooth_span=hurst_smooth_span,
                trend_threshold=trend_threshold,
                mr_threshold=mr_threshold,
                donchian_period=donchian_period,
                bollinger_period=bollinger_period,
                bollinger_std=bollinger_std,
                rsi_period=rsi_period,
                rsi_overbought=rsi_overbought,
                rsi_oversold=rsi_oversold,
            )

        # Align BTC close with asset data
        common_idx = df.index.intersection(btc_close.index)
        btc_aligned = btc_close.reindex(df.index, method='ffill')

        # Get BTC regime
        btc_regime = FRSIndicators.get_btc_regime(btc_aligned, btc_sma_period)
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

        return long_entries, long_exits, short_entries, short_exits, hurst_regime, btc_regime

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

        When realized volatility is high, position size decreases.
        When realized volatility is low, position size increases.

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
        returns = close.pct_change()
        realized_vol = returns.rolling(window=lookback, min_periods=lookback).std()
        realized_vol = realized_vol * np.sqrt(annualization_factor)

        # Avoid division by zero
        realized_vol = realized_vol.replace(0, np.nan)

        raw_position = target_vol / realized_vol
        position_size = raw_position.clip(min_position, max_position)

        # Fill NaN with mid-range position
        position_size = position_size.fillna((min_position + max_position) / 2)

        return position_size
