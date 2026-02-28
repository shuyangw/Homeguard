"""
ML Crypto Mean Reversion Strategy.

A mean reversion strategy for cryptocurrency markets enhanced with an ML
regime filter to identify favorable ranging conditions and avoid trend traps.

Entry Logic:
- Long: Z-score < -threshold AND ML filter predicts "ranging"
- Short: Z-score > +threshold AND ML filter predicts "ranging"

Exit Logic:
- Mean reversion exit: Z-score crosses toward zero
- Stop loss: ATR-based fixed stop
- Take profit: ATR-based target (~3:1 R:R)
- Time stop: Exit after max_hold_bars

The ML filter distinguishes ranging vs trending regimes:
- Ranging (~70-80% of time): Mean reversion works well
- Trending (~20-30% of time): Mean reversion fails, signals are avoided

Based on: Reddit r/algotrading "2 years building, 3 months live"
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

from src.strategies.advanced.zscore_mr_base import ZScoreMeanReversionBase
from src.strategies.advanced.ml_crypto_mr_indicators import (
    MLCryptoMRIndicators,
    MLRegimeFilter,
    MLCryptoMRSignal
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MLCryptoMRPosition:
    """Represents an active position in the ML Crypto MR strategy."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    entry_bar: int
    stop_loss: float
    take_profit: float
    atr_at_entry: float


class MLCryptoMRStrategy(ZScoreMeanReversionBase):
    """
    ML-filtered Mean Reversion Strategy for Crypto.

    A swing trading strategy that identifies overextended price conditions
    (via Z-score) and takes mean reversion trades when an ML model predicts
    the market is in a ranging regime.

    The strategy uses:
    - Z-score for mean reversion signal detection
    - RSI for momentum confirmation
    - ML classifier (GradientBoosting) for regime filtering
    - ATR-based stops and targets for risk management

    Parameters:
        zscore_window: Z-score calculation window (default 20)
        zscore_entry_threshold: Entry Z-score threshold (default 2.0)
        zscore_exit_threshold: Mean reversion exit threshold (default 0.5)
        rsi_period: RSI period (default 14)
        rsi_oversold: RSI oversold level (default 30)
        rsi_overbought: RSI overbought level (default 70)
        use_rsi_confirmation: Require RSI confirmation (default True)
        atr_period: ATR period (default 14)
        atr_stop_multiplier: Stop = ATR x mult (default 1.5)
        atr_target_multiplier: Target = ATR x mult (default 4.5)
        use_ml_filter: Enable ML regime filter (default True)
        ml_lookback_days: ML training lookback (default 252)
        ml_retrain_frequency: Retrain every N bars (default 20)
        adx_threshold: ADX trending threshold (default 25.0)
        choppiness_threshold: Choppiness ranging threshold (default 61.8)
        model_type: ML model type - 'gradient_boosting', 'xgboost',
                   'random_forest', or 'ensemble' (default 'gradient_boosting')
        long_only: Long-only mode (default False)
        max_hold_bars: Maximum holding period (default 10)
    """

    def __init__(
        self,
        # Z-score parameters
        zscore_window: int = 20,
        zscore_entry_threshold: float = 2.0,
        zscore_exit_threshold: float = 0.5,
        # RSI parameters
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        use_rsi_confirmation: bool = True,
        # ATR / Risk parameters
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        atr_target_multiplier: float = 4.5,
        # Fixed percentage exits (alternative to ATR-based)
        use_fixed_pct_exits: bool = False,
        fixed_stop_pct: float = 0.10,  # 10% stop loss
        fixed_target_pct: float = 0.318,  # 31.8% target (3.18:1 R:R)
        # ML filter parameters
        use_ml_filter: bool = True,
        ml_lookback_days: int = 252,
        ml_retrain_frequency: int = 20,
        adx_threshold: float = 25.0,
        choppiness_threshold: float = 61.8,
        model_type: str = 'gradient_boosting',
        # Trade management
        long_only: bool = False,
        max_hold_bars: int = 10,
        **kwargs
    ):
        # Strategy-specific parameters (set before super().__init__)
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.use_rsi_confirmation = use_rsi_confirmation
        self.use_ml_filter = use_ml_filter
        self.ml_lookback_days = ml_lookback_days
        self.ml_retrain_frequency = ml_retrain_frequency
        self.adx_threshold = adx_threshold
        self.choppiness_threshold = choppiness_threshold
        self.model_type = model_type

        # Create indicator config
        self._indicator_config = {
            'zscore_window': zscore_window,
            'rsi_period': rsi_period,
            'atr_period': atr_period,
            'adx_period': 14,
            'choppiness_period': 14,
            'efficiency_period': 10,
            'bollinger_window': 20,
            'volatility_window': 20,
        }

        # Initialize regime filter
        self._regime_filter = MLRegimeFilter(
            lookback_days=ml_lookback_days,
            retrain_frequency=ml_retrain_frequency,
            min_train_samples=100,
            use_ml=use_ml_filter,
            adx_threshold=adx_threshold,
            choppiness_threshold=choppiness_threshold,
            model_type=model_type,
        )

        # Position tracking
        self._current_position: Optional[MLCryptoMRPosition] = None
        self._features_cache: Optional[pd.DataFrame] = None

        # Pre-computed regime predictions (populated in _pre_loop_setup)
        self._is_ranging: Optional[pd.Series] = None

        # Call base init (sets shared params and calls super().__init__)
        super().__init__(
            zscore_window=zscore_window,
            zscore_entry_threshold=zscore_entry_threshold,
            zscore_exit_threshold=zscore_exit_threshold,
            atr_period=atr_period,
            atr_stop_multiplier=atr_stop_multiplier,
            atr_target_multiplier=atr_target_multiplier,
            use_fixed_pct_exits=use_fixed_pct_exits,
            fixed_stop_pct=fixed_stop_pct,
            fixed_target_pct=fixed_target_pct,
            long_only=long_only,
            max_hold_bars=max_hold_bars,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            use_rsi_confirmation=use_rsi_confirmation,
            use_ml_filter=use_ml_filter,
            ml_lookback_days=ml_lookback_days,
            ml_retrain_frequency=ml_retrain_frequency,
            adx_threshold=adx_threshold,
            choppiness_threshold=choppiness_threshold,
            model_type=model_type,
            **kwargs
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.zscore_window <= 0:
            raise ValueError("zscore_window must be positive")
        if self.zscore_entry_threshold <= 0:
            raise ValueError("zscore_entry_threshold must be positive")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.atr_stop_multiplier <= 0:
            raise ValueError("atr_stop_multiplier must be positive")
        if self.atr_target_multiplier <= 0:
            raise ValueError("atr_target_multiplier must be positive")
        if self.rsi_oversold < 0 or self.rsi_oversold > 100:
            raise ValueError("rsi_oversold must be between 0 and 100")
        if self.rsi_overbought < 0 or self.rsi_overbought > 100:
            raise ValueError("rsi_overbought must be between 0 and 100")
        if self.max_hold_bars <= 0:
            raise ValueError("max_hold_bars must be positive")

    def reset_state(self) -> None:
        """Reset strategy state for new backtest run."""
        self._current_position = None
        self._features_cache = None
        self._is_ranging = None
        self._regime_filter.reset()

    # ------------------------------------------------------------------
    # Base class hooks
    # ------------------------------------------------------------------

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features needed for signal generation."""
        return MLCryptoMRIndicators.calculate_all_features(
            data, self._indicator_config
        )

    def _pre_loop_setup(
        self, data: pd.DataFrame, indicators: pd.DataFrame
    ) -> None:
        """Train ML model and pre-compute regime predictions."""
        n = len(data)

        if self.use_ml_filter:
            min_train_idx = min(self.ml_lookback_days, n // 2)
            self._regime_filter.fit(indicators, end_idx=min_train_idx)

        is_ranging, _prob_ranging = self._regime_filter.predict(indicators)
        self._is_ranging = is_ranging

    def _get_regime_filter_result(
        self, i: int, indicators: pd.DataFrame
    ) -> bool:
        """Return True when ML filter predicts ranging regime (or ML off)."""
        if not self.use_ml_filter:
            return True

        ranging = self._is_ranging.iloc[i]
        if pd.isna(ranging):
            return False
        return bool(ranging)

    def _get_min_required_bars(self) -> int:
        """Minimum bars: zscore_window plus buffer."""
        return self.zscore_window + 10

    def _check_entry_confirmation(
        self, i: int, indicators: pd.DataFrame
    ) -> Tuple[bool, bool]:
        """
        RSI confirmation for entries.

        Returns (long_ok, short_ok). When use_rsi_confirmation is False,
        both are True.
        """
        if not self.use_rsi_confirmation:
            return True, True

        current_rsi = indicators['rsi'].iloc[i]
        if pd.isna(current_rsi):
            return False, False

        long_ok = current_rsi < self.rsi_oversold
        short_ok = current_rsi > self.rsi_overbought
        return long_ok, short_ok

    # ------------------------------------------------------------------
    # Live trading helpers (strategy-specific)
    # ------------------------------------------------------------------

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features needed for signal generation."""
        return MLCryptoMRIndicators.calculate_all_features(data, self._indicator_config)

    def get_current_signal(self, data: pd.DataFrame) -> MLCryptoMRSignal:
        """
        Get current signal state for live trading.

        Args:
            data: Recent OHLCV data (enough for indicators)

        Returns:
            MLCryptoMRSignal with current state
        """
        features = self._calculate_features(data)

        # Get latest values
        latest_idx = features.index[-1]
        latest = features.iloc[-1]

        # Get regime prediction
        is_ranging, prob_ranging = self._regime_filter.predict(features)

        # Calculate stop/target levels
        stop_dist = latest['atr'] * self.atr_stop_multiplier
        target_dist = latest['atr'] * self.atr_target_multiplier

        # Determine entry signals
        ranging = is_ranging.iloc[-1]
        zscore = latest['zscore']
        rsi = latest['rsi']

        # Long entry check
        long_cond = zscore <= -self.zscore_entry_threshold
        if self.use_rsi_confirmation:
            long_cond = long_cond and rsi < self.rsi_oversold
        if self.use_ml_filter:
            long_cond = long_cond and ranging

        # Short entry check
        short_cond = zscore >= self.zscore_entry_threshold
        if self.use_rsi_confirmation:
            short_cond = short_cond and rsi > self.rsi_overbought
        if self.use_ml_filter:
            short_cond = short_cond and ranging

        return MLCryptoMRSignal(
            timestamp=latest_idx,
            close=latest['close'],
            zscore=zscore,
            rsi=rsi,
            atr=latest['atr'],
            is_ranging=ranging,
            regime_prob=prob_ranging.iloc[-1],
            entry_long=long_cond,
            entry_short=short_cond and not self.long_only,
            stop_long=latest['close'] - stop_dist,
            stop_short=latest['close'] + stop_dist,
            target_long=latest['close'] + target_dist,
            target_short=latest['close'] - target_dist
        )

    def get_regime_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get regime statistics for analysis.

        Args:
            data: OHLCV DataFrame

        Returns:
            Dictionary with regime statistics
        """
        features = self._calculate_features(data)
        is_ranging, prob_ranging = self._regime_filter.predict(features)

        ranging_pct = is_ranging.mean() * 100

        return {
            'ranging_pct': ranging_pct,
            'trending_pct': 100 - ranging_pct,
            'avg_regime_prob': prob_ranging.mean(),
            'regime_changes': (is_ranging != is_ranging.shift(1)).sum(),
            'ml_trained': self._regime_filter._is_trained,
        }
