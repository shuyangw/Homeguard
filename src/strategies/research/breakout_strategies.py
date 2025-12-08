"""
Breakout Trading Strategies.

This module contains two breakout strategies:
1. Volume Breakout - Uses 4x volume spikes with price confirmation
2. ATR Squeeze Breakout - Volatility compression -> expansion breakouts

Both strategies use weekly rebalancing with top N stock selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.utils.logger import logger


@dataclass
class BreakoutSignal:
    """Signal for a breakout trade."""
    symbol: str
    signal_type: str  # 'volume' or 'atr_squeeze'
    score: float
    volume_ratio: float
    atr_percentile: float
    price_strength: float
    rank: int
    weight: float
    action: str  # 'buy', 'hold', 'sell'


class VolumeBreakoutSignals:
    """
    Volume spike breakout detection.

    Identifies stocks with abnormal volume (4x+ average) combined with
    bullish price action (close near daily high).

    Parameters:
        volume_multiplier: Required volume spike (default 4x)
        lookback: Days for average volume calculation (default 5)
        min_price_strength: Minimum (close-low)/(high-low) ratio (default 0.7)
        top_n: Number of stocks to select (default 10)
    """

    def __init__(
        self,
        volume_multiplier: float = 4.0,
        lookback: int = 5,
        min_price_strength: float = 0.7,
        top_n: int = 10
    ):
        self.volume_multiplier = volume_multiplier
        self.lookback = lookback
        self.min_price_strength = min_price_strength
        self.top_n = top_n

        logger.info("[VOL] Initialized Volume Breakout Signals")
        logger.info(f"[VOL]   Volume multiplier: {volume_multiplier}x")
        logger.info(f"[VOL]   Lookback: {lookback} days")
        logger.info(f"[VOL]   Min price strength: {min_price_strength}")
        logger.info(f"[VOL]   Top N: {top_n}")

    def calculate_signals(
        self,
        close_df: pd.DataFrame,
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        volume_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate volume breakout signals for all symbols.

        Args:
            close_df: Close prices (columns=symbols, index=dates)
            high_df: High prices
            low_df: Low prices
            volume_df: Volume data

        Returns:
            DataFrame with columns: volume_ratio, price_strength, score
        """
        # Average volume over lookback period (shifted to avoid lookahead)
        avg_volume = volume_df.rolling(window=self.lookback).mean().shift(1)

        # Volume ratio (current vs average)
        volume_ratio = volume_df / avg_volume

        # Price strength: where close is in daily range (0=low, 1=high)
        daily_range = high_df - low_df
        price_strength = (close_df - low_df) / daily_range.replace(0, np.nan)
        price_strength = price_strength.fillna(0.5)

        # Combined score: volume_ratio * price_strength
        score = volume_ratio * price_strength

        return {
            'volume_ratio': volume_ratio,
            'price_strength': price_strength,
            'score': score
        }

    def generate_signals(
        self,
        close_df: pd.DataFrame,
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        current_positions: Dict[str, float]
    ) -> List[BreakoutSignal]:
        """
        Generate trading signals for the latest date.

        Args:
            close_df: Close prices
            high_df: High prices
            low_df: Low prices
            volume_df: Volume data
            current_positions: Current holdings {symbol: shares}

        Returns:
            List of BreakoutSignal objects
        """
        metrics = self.calculate_signals(close_df, high_df, low_df, volume_df)

        # Get latest values
        latest_volume_ratio = metrics['volume_ratio'].iloc[-1].dropna()
        latest_price_strength = metrics['price_strength'].iloc[-1].dropna()
        latest_score = metrics['score'].iloc[-1].dropna()

        # Filter: volume spike AND bullish price action
        valid_mask = (
            (latest_volume_ratio >= self.volume_multiplier) &
            (latest_price_strength >= self.min_price_strength)
        )
        valid_symbols = valid_mask[valid_mask].index

        # Filter scores to valid symbols
        valid_scores = latest_score[valid_symbols]

        if len(valid_scores) == 0:
            logger.warning("[VOL] No stocks meet volume breakout criteria")
            # Return sell signals for current positions
            signals = []
            for symbol in current_positions.keys():
                signals.append(BreakoutSignal(
                    symbol=symbol,
                    signal_type='volume',
                    score=0,
                    volume_ratio=latest_volume_ratio.get(symbol, 0),
                    atr_percentile=0,
                    price_strength=latest_price_strength.get(symbol, 0),
                    rank=0,
                    weight=0,
                    action='sell'
                ))
            return signals

        # Rank by score (highest first)
        ranked = valid_scores.sort_values(ascending=False)
        top_stocks = ranked.head(self.top_n)

        # Calculate equal weight
        weight = 1.0 / self.top_n

        # Generate signals
        signals = []
        current_set = set(current_positions.keys())
        target_set = set(top_stocks.index)

        for rank, (symbol, score) in enumerate(top_stocks.items(), 1):
            action = 'hold' if symbol in current_set else 'buy'
            signals.append(BreakoutSignal(
                symbol=symbol,
                signal_type='volume',
                score=score,
                volume_ratio=latest_volume_ratio.get(symbol, 0),
                atr_percentile=0,
                price_strength=latest_price_strength.get(symbol, 0),
                rank=rank,
                weight=weight,
                action=action
            ))

        # Sell signals for positions not in top N
        for symbol in current_set - target_set:
            signals.append(BreakoutSignal(
                symbol=symbol,
                signal_type='volume',
                score=latest_score.get(symbol, 0),
                volume_ratio=latest_volume_ratio.get(symbol, 0),
                atr_percentile=0,
                price_strength=latest_price_strength.get(symbol, 0),
                rank=0,
                weight=0,
                action='sell'
            ))

        return signals


class ATRSqueezeSignals:
    """
    ATR Squeeze Breakout detection.

    Identifies stocks emerging from volatility compression (squeeze)
    with expanding ATR and price breaking above channel.

    Parameters:
        atr_period: ATR calculation period (default 14)
        squeeze_percentile: ATR percentile threshold for squeeze (default 20)
        squeeze_lookback: Lookback for percentile calculation (default 100)
        channel_period: Channel breakout period (default 20)
        top_n: Number of stocks to select (default 10)
    """

    def __init__(
        self,
        atr_period: int = 14,
        squeeze_percentile: float = 20,
        squeeze_lookback: int = 100,
        channel_period: int = 20,
        top_n: int = 10
    ):
        self.atr_period = atr_period
        self.squeeze_percentile = squeeze_percentile
        self.squeeze_lookback = squeeze_lookback
        self.channel_period = channel_period
        self.top_n = top_n

        logger.info("[ATR] Initialized ATR Squeeze Breakout Signals")
        logger.info(f"[ATR]   ATR period: {atr_period}")
        logger.info(f"[ATR]   Squeeze percentile: {squeeze_percentile}")
        logger.info(f"[ATR]   Channel period: {channel_period}")
        logger.info(f"[ATR]   Top N: {top_n}")

    def calculate_atr(
        self,
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        close_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Average True Range for all symbols."""
        # True Range components
        high_low = high_df - low_df
        high_close = abs(high_df - close_df.shift(1))
        low_close = abs(low_df - close_df.shift(1))

        # True Range = max of the three
        true_range = pd.concat([high_low, high_close, low_close]).groupby(level=0).max()

        # Handle MultiIndex if created
        if isinstance(true_range.index, pd.MultiIndex):
            true_range = high_low.combine(high_close, max).combine(low_close, max)

        # ATR = rolling mean of True Range
        atr = true_range.rolling(window=self.atr_period).mean()

        return atr

    def calculate_atr_percentile(self, atr: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR percentile (normalized by price) over lookback period."""
        # Normalize ATR by close price
        atr_normalized = atr / close_df

        # Calculate rolling percentile
        def percentile_rank(x):
            if len(x.dropna()) < 10:
                return np.nan
            return (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100

        atr_percentile = atr_normalized.rolling(window=self.squeeze_lookback).apply(
            percentile_rank, raw=False
        )

        return atr_percentile

    def calculate_signals(
        self,
        close_df: pd.DataFrame,
        high_df: pd.DataFrame,
        low_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate ATR squeeze breakout signals.

        Returns dict with: atr, atr_percentile, in_squeeze, channel_high, breakout, score
        """
        # Calculate ATR
        atr = self.calculate_atr(high_df, low_df, close_df)

        # Calculate ATR percentile
        atr_percentile = self.calculate_atr_percentile(atr, close_df)

        # Detect squeeze (low ATR percentile)
        in_squeeze = atr_percentile < self.squeeze_percentile

        # Channel high (for breakout detection)
        channel_high = high_df.rolling(window=self.channel_period).max()

        # Breakout: price > previous channel high AND was in squeeze
        breakout = (close_df > channel_high.shift(1)) & in_squeeze.shift(1)

        # ATR expanding (current > previous)
        atr_expanding = atr > atr.shift(1)

        # Momentum (10-day ROC > 0)
        roc = close_df.pct_change(10)
        momentum_positive = roc > 0

        # Valid breakout: all conditions met
        valid_breakout = breakout & atr_expanding & momentum_positive

        # Score: (100 - atr_percentile) * breakout_strength
        # breakout_strength = how much above channel
        breakout_strength = (close_df - channel_high.shift(1)) / channel_high.shift(1)
        breakout_strength = breakout_strength.clip(lower=0)

        score = (100 - atr_percentile) * breakout_strength
        score = score.where(valid_breakout, 0)

        return {
            'atr': atr,
            'atr_percentile': atr_percentile,
            'in_squeeze': in_squeeze,
            'channel_high': channel_high,
            'breakout': valid_breakout,
            'score': score
        }

    def generate_signals(
        self,
        close_df: pd.DataFrame,
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        current_positions: Dict[str, float]
    ) -> List[BreakoutSignal]:
        """
        Generate trading signals for the latest date.

        Args:
            close_df: Close prices
            high_df: High prices
            low_df: Low prices
            current_positions: Current holdings {symbol: shares}

        Returns:
            List of BreakoutSignal objects
        """
        metrics = self.calculate_signals(close_df, high_df, low_df)

        # Get latest values
        latest_score = metrics['score'].iloc[-1].dropna()
        latest_atr_percentile = metrics['atr_percentile'].iloc[-1].dropna()
        latest_breakout = metrics['breakout'].iloc[-1]

        # Filter to stocks with valid breakouts (score > 0)
        valid_scores = latest_score[latest_score > 0]

        if len(valid_scores) == 0:
            logger.warning("[ATR] No stocks meet ATR squeeze breakout criteria")
            # Return sell signals for current positions
            signals = []
            for symbol in current_positions.keys():
                signals.append(BreakoutSignal(
                    symbol=symbol,
                    signal_type='atr_squeeze',
                    score=0,
                    volume_ratio=0,
                    atr_percentile=latest_atr_percentile.get(symbol, 50),
                    price_strength=0,
                    rank=0,
                    weight=0,
                    action='sell'
                ))
            return signals

        # Rank by score (highest first)
        ranked = valid_scores.sort_values(ascending=False)
        top_stocks = ranked.head(self.top_n)

        # Calculate equal weight
        weight = 1.0 / self.top_n

        # Generate signals
        signals = []
        current_set = set(current_positions.keys())
        target_set = set(top_stocks.index)

        for rank, (symbol, score) in enumerate(top_stocks.items(), 1):
            action = 'hold' if symbol in current_set else 'buy'
            signals.append(BreakoutSignal(
                symbol=symbol,
                signal_type='atr_squeeze',
                score=score,
                volume_ratio=0,
                atr_percentile=latest_atr_percentile.get(symbol, 50),
                price_strength=0,
                rank=rank,
                weight=weight,
                action=action
            ))

        # Sell signals for positions not in top N
        for symbol in current_set - target_set:
            signals.append(BreakoutSignal(
                symbol=symbol,
                signal_type='atr_squeeze',
                score=latest_score.get(symbol, 0),
                volume_ratio=0,
                atr_percentile=latest_atr_percentile.get(symbol, 50),
                price_strength=0,
                rank=0,
                weight=0,
                action='sell'
            ))

        return signals
