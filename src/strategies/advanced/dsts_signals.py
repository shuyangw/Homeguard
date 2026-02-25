"""
Dual Signal Trend Sentinel (DSTS) Signals for Live Trading.

Signal generator for live/paper trading that calculates real-time Z-score
and generates position targets based on state machine logic.

Usage:
    from src.strategies.advanced.dsts_signals import DSTSSignals

    signals = DSTSSignals(
        period=65,
        bull_threshold=0.75,
        bear_threshold=0.0,
    )

    # Update with latest data
    signals.update_historical_data(btc_ohlcv_df)

    # Get current position target
    risk_signals = signals.get_risk_signals()
    target_position = signals.get_target_position()
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.strategies.advanced.dsts_indicators import DSTSIndicators, DSTSSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DSTSRiskSignals:
    """Risk signals for live trading."""
    timestamp: pd.Timestamp
    zscore: float
    state: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    should_be_long: bool
    close: float
    ema: float
    std: float
    # Trailing stop fields
    trailing_stop_triggered: bool = False
    current_drawdown: float = 0.0
    peak_value: float = 0.0

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'zscore': self.zscore,
            'state': self.state,
            'should_be_long': self.should_be_long,
            'close': self.close,
            'ema': self.ema,
            'std': self.std,
            'trailing_stop_triggered': self.trailing_stop_triggered,
            'current_drawdown': self.current_drawdown,
            'peak_value': self.peak_value,
        }


class DSTSSignals:
    """
    Stateful signal generator for DSTS live trading.

    Maintains position state and generates entry/exit signals based on
    Z-score thresholds with hysteresis.

    Key Features:
    - State machine tracking (in/out of position)
    - Real-time Z-score calculation
    - Trailing stop protection
    - Configurable thresholds
    """

    def __init__(
        self,
        period: int = 65,
        bull_threshold: float = 0.75,
        bear_threshold: float = 0.0,
        trailing_stop_pct: float = 0.25,
    ):
        """
        Initialize DSTS signal generator.

        Args:
            period: Period for EMA and StdDev (default 65)
            bull_threshold: Z-score threshold to enter long (default 0.75)
            bear_threshold: Z-score threshold to exit long (default 0.0)
            trailing_stop_pct: Trailing stop percentage (0.25 = 25% drawdown)
        """
        self.period = period
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.trailing_stop_pct = trailing_stop_pct

        # Stateful tracking
        self._historical_data: Optional[pd.DataFrame] = None
        self._current_state: str = 'BEARISH'  # Start in cash
        self._is_long: bool = False

        # Trailing stop state
        self._peak_portfolio_value: float = 0.0
        self._current_portfolio_value: float = 0.0
        self._trailing_stop_triggered: bool = False

        logger.info("[DSTS] Initialized Dual Signal Trend Sentinel Signals")
        logger.info(f"[DSTS]   Period: {period}")
        logger.info(f"[DSTS]   Bull Threshold: {bull_threshold}")
        logger.info(f"[DSTS]   Bear Threshold: {bear_threshold}")
        logger.info(f"[DSTS]   Trailing Stop: {trailing_stop_pct:.0%}")

    def update_historical_data(self, data: pd.DataFrame) -> None:
        """
        Update with latest price data.

        Args:
            data: OHLCV DataFrame with at least 'close' column
        """
        self._historical_data = data

        if len(data) >= self.period:
            # Update state machine
            self._update_state()

        logger.debug(f"[DSTS] Updated historical data: {len(data)} bars")

    def _update_state(self) -> None:
        """Update internal state based on latest Z-score."""
        if self._historical_data is None or len(self._historical_data) < self.period:
            return

        close_col = 'close' if 'close' in self._historical_data.columns else 'Close'
        close = self._historical_data[close_col]

        zscore = DSTSIndicators.calculate_zscore(close, self.period).iloc[-1]

        if pd.isna(zscore):
            return

        old_state = self._is_long

        # State machine transition
        if not self._is_long:
            if zscore > self.bull_threshold:
                self._current_state = 'BULLISH'
                self._is_long = True
                logger.info(f"[DSTS] STATE CHANGE: BEARISH -> BULLISH (Z={zscore:.2f})")
        else:
            if zscore < self.bear_threshold:
                self._current_state = 'BEARISH'
                self._is_long = False
                logger.info(f"[DSTS] STATE CHANGE: BULLISH -> BEARISH (Z={zscore:.2f})")
            elif zscore > self.bull_threshold:
                self._current_state = 'BULLISH'
            else:
                self._current_state = 'NEUTRAL'  # In position but between thresholds

        # Override if trailing stop triggered
        if self._trailing_stop_triggered and self._is_long:
            self._current_state = 'BEARISH'
            self._is_long = False
            logger.info("[DSTS] Position closed due to trailing stop")

    def update_portfolio_value(self, current_value: float) -> None:
        """
        Update portfolio value for trailing stop calculation.

        Args:
            current_value: Current total portfolio value in USD
        """
        self._current_portfolio_value = current_value

        # Only track peak while in position
        if self._is_long:
            if current_value > self._peak_portfolio_value:
                self._peak_portfolio_value = current_value
                logger.debug(f"[DSTS] New peak value: ${current_value:,.2f}")

            # Check trailing stop
            if self._peak_portfolio_value > 0:
                drawdown = (self._peak_portfolio_value - current_value) / self._peak_portfolio_value

                if drawdown >= self.trailing_stop_pct and not self._trailing_stop_triggered:
                    self._trailing_stop_triggered = True
                    logger.warning(
                        f"[DSTS] TRAILING STOP TRIGGERED! "
                        f"DD={drawdown:.1%}, Peak=${self._peak_portfolio_value:,.0f}, "
                        f"Current=${current_value:,.0f}"
                    )
                    # Update state immediately
                    self._current_state = 'BEARISH'
                    self._is_long = False

    def reset_trailing_stop(self) -> None:
        """Reset trailing stop (allows re-entry after exit)."""
        if self._trailing_stop_triggered:
            logger.info("[DSTS] Resetting trailing stop")
        self._trailing_stop_triggered = False
        self._peak_portfolio_value = 0.0

    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        if self._peak_portfolio_value <= 0:
            return 0.0
        return (self._peak_portfolio_value - self._current_portfolio_value) / self._peak_portfolio_value

    def is_trailing_stop_active(self) -> bool:
        """Check if trailing stop is currently triggered."""
        return self._trailing_stop_triggered

    def get_risk_signals(self) -> DSTSRiskSignals:
        """
        Get current risk signals.

        Returns:
            DSTSRiskSignals with current state

        Raises:
            ValueError: If insufficient data
        """
        if self._historical_data is None or len(self._historical_data) < self.period:
            raise ValueError(
                f"Insufficient data: need {self.period} bars, "
                f"got {len(self._historical_data) if self._historical_data is not None else 0}"
            )

        close_col = 'close' if 'close' in self._historical_data.columns else 'Close'
        close = self._historical_data[close_col]

        indicators = DSTSIndicators.calculate_all_indicators(close, self.period)
        latest = indicators.iloc[-1]

        return DSTSRiskSignals(
            timestamp=self._historical_data.index[-1],
            zscore=latest['zscore'],
            state=self._current_state,
            should_be_long=self._is_long and not self._trailing_stop_triggered,
            close=latest['close'],
            ema=latest['ema'],
            std=latest['std'],
            trailing_stop_triggered=self._trailing_stop_triggered,
            current_drawdown=self.get_current_drawdown(),
            peak_value=self._peak_portfolio_value,
        )

    def get_target_position(self) -> float:
        """
        Return target position weight (0.0 or 1.0).

        Returns:
            1.0 if should be long, 0.0 if should be in cash
        """
        if self._trailing_stop_triggered:
            return 0.0
        return 1.0 if self._is_long else 0.0

    def should_enter(self) -> bool:
        """
        Check if strategy signals entry.

        Returns:
            True if should enter long position
        """
        if self._historical_data is None or len(self._historical_data) < self.period:
            return False

        if self._trailing_stop_triggered:
            return False

        close_col = 'close' if 'close' in self._historical_data.columns else 'Close'
        close = self._historical_data[close_col]

        zscore = DSTSIndicators.calculate_zscore(close, self.period).iloc[-1]
        return not self._is_long and zscore > self.bull_threshold

    def should_exit(self) -> bool:
        """
        Check if strategy signals exit.

        Returns:
            True if should exit long position
        """
        if self._historical_data is None or len(self._historical_data) < self.period:
            return False

        if self._trailing_stop_triggered and self._is_long:
            return True

        close_col = 'close' if 'close' in self._historical_data.columns else 'Close'
        close = self._historical_data[close_col]

        zscore = DSTSIndicators.calculate_zscore(close, self.period).iloc[-1]
        return self._is_long and zscore < self.bear_threshold

    def set_position_state(self, is_long: bool) -> None:
        """
        Manually set position state (for initialization from broker).

        Args:
            is_long: Whether currently holding a long position
        """
        self._is_long = is_long
        if is_long:
            self._current_state = 'BULLISH'
        else:
            self._current_state = 'BEARISH'
        logger.info(f"[DSTS] Position state set to: {'LONG' if is_long else 'CASH'}")

    def get_summary(self) -> Dict:
        """
        Get summary of current strategy state.

        Returns:
            Dictionary with state summary
        """
        if self._historical_data is None or len(self._historical_data) < self.period:
            return {
                'state': 'INSUFFICIENT_DATA',
                'is_long': False,
                'zscore': None,
                'target_position': 0.0,
                'bars_available': len(self._historical_data) if self._historical_data is not None else 0,
                'bars_required': self.period,
            }

        risk_signals = self.get_risk_signals()

        return {
            'state': self._current_state,
            'is_long': self._is_long,
            'zscore': risk_signals.zscore,
            'close': risk_signals.close,
            'ema': risk_signals.ema,
            'std': risk_signals.std,
            'target_position': self.get_target_position(),
            'trailing_stop_triggered': self._trailing_stop_triggered,
            'current_drawdown': self.get_current_drawdown(),
            'peak_value': self._peak_portfolio_value,
            'period': self.period,
            'bull_threshold': self.bull_threshold,
            'bear_threshold': self.bear_threshold,
            'trailing_stop_pct': self.trailing_stop_pct,
        }
