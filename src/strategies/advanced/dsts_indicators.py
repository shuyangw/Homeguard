"""
Dual Signal Trend Sentinel (DSTS) Indicators.

Volatility-normalized trend following for Bitcoin using Z-score deviation from EMA.

Strategy Logic:
- Baseline: EMA(65)
- Volatility: StdDev(65)
- Signal: Z-Score = (Close - EMA) / StdDev
- Entry: Z-Score > bull_threshold (default 0.75)
- Exit: Z-Score < bear_threshold (default 0.0)
- Position: 100% long or 100% cash

Key Insight:
- Complexity hurts. Adding filters (RSI, MACD, volume) reduced performance
- Simple "eyeballed" thresholds outperformed grid search optimization
- State machine with hysteresis reduces whipsaws

Reference:
- Target: CAGR ~66%, Max DD ~27%, Win Rate ~47%, Win/Loss Ratio >8:1, ~4-5 trades/year
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DSTSSignal:
    """Current state snapshot for DSTS strategy."""
    timestamp: pd.Timestamp
    close: float
    ema: float
    std: float
    zscore: float
    state: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'


class DSTSIndicators:
    """
    Static indicator calculations for DSTS strategy.

    Provides Z-score based trend detection with volatility normalization.
    Designed for daily timeframe on BTC/USD.
    """

    @staticmethod
    def calculate_ema(close: pd.Series, period: int = 65) -> pd.Series:
        """
        Calculate exponential moving average.

        Args:
            close: Close price series
            period: EMA period (default 65)

        Returns:
            EMA series
        """
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_std(close: pd.Series, period: int = 65) -> pd.Series:
        """
        Calculate rolling standard deviation.

        Args:
            close: Close price series
            period: Rolling window period (default 65)

        Returns:
            Standard deviation series
        """
        return close.rolling(window=period).std()

    @staticmethod
    def calculate_zscore(close: pd.Series, period: int = 65) -> pd.Series:
        """
        Calculate Z-score: (Close - EMA) / StdDev.

        The Z-score measures how many standard deviations the price is
        from its exponential moving average, providing a volatility-normalized
        trend indicator.

        Args:
            close: Close price series
            period: Period for both EMA and StdDev (default 65)

        Returns:
            Z-score series
        """
        ema = DSTSIndicators.calculate_ema(close, period)
        std = DSTSIndicators.calculate_std(close, period)

        # Avoid division by zero
        zscore = (close - ema) / std.replace(0, np.nan)
        return zscore

    @staticmethod
    def calculate_all_indicators(
        close: pd.Series,
        period: int = 65
    ) -> pd.DataFrame:
        """
        Calculate all DSTS indicators.

        Args:
            close: Close price series
            period: Period for calculations

        Returns:
            DataFrame with columns: close, ema, std, zscore
        """
        ema = DSTSIndicators.calculate_ema(close, period)
        std = DSTSIndicators.calculate_std(close, period)
        zscore = (close - ema) / std.replace(0, np.nan)

        return pd.DataFrame({
            'close': close,
            'ema': ema,
            'std': std,
            'zscore': zscore
        }, index=close.index)

    @staticmethod
    def generate_signals(
        close: pd.Series,
        period: int = 65,
        bull_threshold: float = 0.75,
        bear_threshold: float = 0.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate entry/exit signals using state machine.

        The state machine implements hysteresis:
        - Enter long when Z-score crosses above bull_threshold
        - Exit long when Z-score crosses below bear_threshold

        This gap between thresholds reduces whipsaws.

        Args:
            close: Close price series
            period: Period for EMA and StdDev (default 65)
            bull_threshold: Z-score threshold to enter long (default 0.75)
            bear_threshold: Z-score threshold to exit long (default 0.0)

        Returns:
            Tuple of (entries, exits, zscore) as boolean Series
        """
        zscore = DSTSIndicators.calculate_zscore(close, period)

        # State machine: track position state across bars
        entries = pd.Series(False, index=close.index)
        exits = pd.Series(False, index=close.index)

        current_state = False  # False = cash, True = long

        for i in range(len(close)):
            if pd.isna(zscore.iloc[i]):
                continue

            z = zscore.iloc[i]

            if not current_state:  # Currently in cash
                if z > bull_threshold:
                    current_state = True
                    entries.iloc[i] = True
            else:  # Currently long
                if z < bear_threshold:
                    current_state = False
                    exits.iloc[i] = True

        return entries, exits, zscore

    @staticmethod
    def generate_signals_vectorized(
        close: pd.Series,
        period: int = 65,
        bull_threshold: float = 0.75,
        bear_threshold: float = 0.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate entry/exit signals using vectorized operations.

        This is faster than the iterative version but produces identical results.
        Uses forward-fill to track state transitions.

        Args:
            close: Close price series
            period: Period for EMA and StdDev
            bull_threshold: Z-score threshold to enter long
            bear_threshold: Z-score threshold to exit long

        Returns:
            Tuple of (entries, exits, zscore) as boolean Series
        """
        zscore = DSTSIndicators.calculate_zscore(close, period)

        # Identify potential entry/exit points
        potential_entries = zscore > bull_threshold
        potential_exits = zscore < bear_threshold

        # Build state machine using cumulative logic
        # Signal transitions: +1 for potential entry, -1 for potential exit
        signals = pd.Series(0, index=close.index)
        signals[potential_entries] = 1
        signals[potential_exits] = -1

        # Track state: 1 = long, 0 = cash
        # Forward-fill ensures we stay in state until opposite signal
        state = pd.Series(0, index=close.index)
        current = 0

        for i in range(len(signals)):
            if pd.isna(zscore.iloc[i]):
                state.iloc[i] = current
                continue

            sig = signals.iloc[i]
            if sig == 1 and current == 0:
                current = 1
            elif sig == -1 and current == 1:
                current = 0
            state.iloc[i] = current

        # Entries: transition from 0 to 1
        entries = (state == 1) & (state.shift(1).fillna(0) == 0)

        # Exits: transition from 1 to 0
        exits = (state == 0) & (state.shift(1).fillna(0) == 1)

        return entries, exits, zscore

    @staticmethod
    def get_current_signal(
        close: pd.Series,
        period: int = 65,
        bull_threshold: float = 0.75,
        bear_threshold: float = 0.0,
        is_long: bool = False
    ) -> DSTSSignal:
        """
        Get current DSTS signal state for live trading.

        Args:
            close: Close price series (must have at least `period` bars)
            period: Period for EMA and StdDev
            bull_threshold: Z-score threshold to enter long
            bear_threshold: Z-score threshold to exit long
            is_long: Current position state (from previous bar)

        Returns:
            DSTSSignal with current state
        """
        if len(close) < period:
            raise ValueError(f"Insufficient data: need {period} bars, got {len(close)}")

        indicators = DSTSIndicators.calculate_all_indicators(close, period)
        latest = indicators.iloc[-1]

        z = latest['zscore']

        # Determine state based on current position and z-score
        if not is_long:
            if z > bull_threshold:
                state = 'BULLISH'
            else:
                state = 'BEARISH'
        else:
            if z < bear_threshold:
                state = 'BEARISH'
            elif z > bull_threshold:
                state = 'BULLISH'
            else:
                state = 'NEUTRAL'  # In position but between thresholds

        return DSTSSignal(
            timestamp=close.index[-1],
            close=latest['close'],
            ema=latest['ema'],
            std=latest['std'],
            zscore=z,
            state=state
        )

    @staticmethod
    def generate_signals_with_trailing_stop(
        close: pd.Series,
        period: int = 65,
        bull_threshold: float = 0.75,
        bear_threshold: float = 0.0,
        trailing_stop_pct: float = 0.20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate entry/exit signals with trailing stop protection.

        Adds a trailing stop to the standard DSTS signals:
        - Tracks peak price during each trade
        - Exits if price drops more than trailing_stop_pct from peak

        Args:
            close: Close price series
            period: Period for EMA and StdDev (default 65)
            bull_threshold: Z-score threshold to enter long (default 0.75)
            bear_threshold: Z-score threshold to exit long (default 0.0)
            trailing_stop_pct: Max drawdown from peak before exit (default 0.20 = 20%)

        Returns:
            Tuple of (entries, exits, zscore) as boolean Series
        """
        zscore = DSTSIndicators.calculate_zscore(close, period)

        entries = pd.Series(False, index=close.index)
        exits = pd.Series(False, index=close.index)

        current_state = False  # False = cash, True = long
        peak_price = 0.0

        for i in range(len(close)):
            if pd.isna(zscore.iloc[i]):
                continue

            z = zscore.iloc[i]
            price = close.iloc[i]

            if not current_state:  # Currently in cash
                if z > bull_threshold:
                    current_state = True
                    entries.iloc[i] = True
                    peak_price = price
            else:  # Currently long
                # Update peak price
                if price > peak_price:
                    peak_price = price

                # Check trailing stop
                drawdown = (price - peak_price) / peak_price
                if drawdown < -trailing_stop_pct:
                    current_state = False
                    exits.iloc[i] = True
                    peak_price = 0.0
                # Check normal exit condition
                elif z < bear_threshold:
                    current_state = False
                    exits.iloc[i] = True
                    peak_price = 0.0

        return entries, exits, zscore

    @staticmethod
    def get_btc_regime(
        btc_close: pd.Series,
        sma_period: int = 20
    ) -> pd.Series:
        """
        Calculate BTC regime filter.

        Simple rule: BTC > SMA = bullish, else bearish.
        When BTC is weak, altcoins typically suffer more.

        Args:
            btc_close: BTC close price series
            sma_period: SMA period for regime detection (default 20)

        Returns:
            Series with 'bullish' or 'bearish' regime labels
        """
        sma = btc_close.rolling(window=sma_period, min_periods=sma_period).mean()
        regime = pd.Series('bearish', index=btc_close.index)
        regime[btc_close > sma] = 'bullish'
        return regime

    @staticmethod
    def generate_signals_with_regime(
        close: pd.Series,
        btc_close: pd.Series,
        period: int = 65,
        bull_threshold: float = 0.75,
        bear_threshold: float = 0.0,
        trailing_stop_pct: float = None,
        btc_sma_period: int = 20,
        regime_mode: str = 'block_entry',
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate signals with BTC regime filter.

        Args:
            close: Asset close price series
            btc_close: BTC close price series (for regime detection)
            period: Period for EMA and StdDev
            bull_threshold: Z-score threshold to enter long
            bear_threshold: Z-score threshold to exit long
            trailing_stop_pct: Optional trailing stop (None = disabled)
            btc_sma_period: SMA period for BTC regime
            regime_mode: 'block_entry' (no new entries in bearish) or
                        'force_exit' (exit all positions in bearish)

        Returns:
            Tuple of (entries, exits, zscore, regime)
        """
        zscore = DSTSIndicators.calculate_zscore(close, period)
        regime = DSTSIndicators.get_btc_regime(btc_close, btc_sma_period)

        entries = pd.Series(False, index=close.index)
        exits = pd.Series(False, index=close.index)

        current_state = False
        peak_price = 0.0

        for i in range(len(close)):
            if pd.isna(zscore.iloc[i]):
                continue

            z = zscore.iloc[i]
            price = close.iloc[i]
            current_regime = regime.iloc[i]

            if not current_state:  # Currently in cash
                # Only enter in bullish regime (or if regime_mode is not block_entry)
                if z > bull_threshold:
                    if regime_mode != 'block_entry' or current_regime == 'bullish':
                        current_state = True
                        entries.iloc[i] = True
                        peak_price = price
            else:  # Currently long
                # Update peak price
                if price > peak_price:
                    peak_price = price

                should_exit = False

                # Check force_exit mode
                if regime_mode == 'force_exit' and current_regime == 'bearish':
                    should_exit = True

                # Check trailing stop
                if trailing_stop_pct is not None:
                    drawdown = (price - peak_price) / peak_price
                    if drawdown < -trailing_stop_pct:
                        should_exit = True

                # Check normal exit condition
                if z < bear_threshold:
                    should_exit = True

                if should_exit:
                    current_state = False
                    exits.iloc[i] = True
                    peak_price = 0.0

        return entries, exits, zscore, regime

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

        Scales position inversely with realized volatility:
        position_size = target_vol / realized_vol

        Args:
            close: Close price series
            target_vol: Target annualized volatility (default 0.20 = 20%)
            lookback: Lookback period for volatility calculation
            annualization_factor: Factor to annualize volatility (365 for daily, 8760 for hourly)
            min_position: Minimum position size (default 0.25 = 25%)
            max_position: Maximum position size (default 1.00 = 100%)

        Returns:
            Series of position sizes between min_position and max_position
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

        return position_size

    @staticmethod
    def calculate_trade_statistics(
        entries: pd.Series,
        exits: pd.Series,
        close: pd.Series
    ) -> dict:
        """
        Calculate trade-level statistics.

        Args:
            entries: Boolean Series of entry signals
            exits: Boolean Series of exit signals
            close: Close price series

        Returns:
            Dictionary with trade statistics
        """
        entry_dates = entries[entries].index.tolist()
        exit_dates = exits[exits].index.tolist()

        if not entry_dates:
            return {
                'trade_count': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_loss_ratio': 0.0,
                'avg_trade_duration': 0.0,
            }

        trades = []
        entry_idx = 0
        exit_idx = 0

        while entry_idx < len(entry_dates) and exit_idx < len(exit_dates):
            entry_date = entry_dates[entry_idx]
            # Find next exit after this entry
            while exit_idx < len(exit_dates) and exit_dates[exit_idx] <= entry_date:
                exit_idx += 1

            if exit_idx >= len(exit_dates):
                break

            exit_date = exit_dates[exit_idx]
            entry_price = close.loc[entry_date]
            exit_price = close.loc[exit_date]
            pnl_pct = (exit_price - entry_price) / entry_price

            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'duration': (exit_date - entry_date).days
            })

            entry_idx += 1
            exit_idx += 1

        if not trades:
            return {
                'trade_count': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_loss_ratio': 0.0,
                'avg_trade_duration': 0.0,
            }

        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]

        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0.0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0.0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        return {
            'trade_count': len(trades),
            'win_count': len(wins),
            'loss_count': len(losses),
            'win_rate': len(wins) / len(trades) if trades else 0.0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'avg_trade_duration': np.mean([t['duration'] for t in trades]),
        }
