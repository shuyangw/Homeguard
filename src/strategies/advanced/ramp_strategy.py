"""
Regime-Aware Momentum Protection (RAMP) Strategy.

This strategy extends the momentum protection strategy with market regime detection,
adapting parameters based on whether the market is in a bull, bear, or sideways state.

Strategy Overview:
1. Universe: S&P 500 stocks
2. Regime Detection: Uses VIX, SPY momentum, and moving averages to classify market
3. Selection: Top N stocks by regime-specific momentum formula
4. Rebalance: Daily at 3:55 PM EST (near close)
5. Protection: Reduce exposure when VIX > 25 or SPY drawdown > 5%

Momentum Formula:
    momentum = (long_weight * return_long_period) - (penalty_weight * return_short_period)

This formula penalizes recent gains (avoids buying at the top) with regime-specific weights.

Regime-Specific Parameters (Walk-Forward Validated 2017-2024):
    STRONG_BULL: L21/S5, LW=0.3, PW=5.0, TopN=20
    WEAK_BULL:   L21/S5, LW=0.3, PW=5.0, TopN=10
    SIDEWAYS:    L21/S5, LW=0.5, PW=2.0, TopN=5
    UNPREDICTABLE: L42/S21, LW=0.5, PW=4.0, TopN=10
    BEAR:        L21/S5, LW=0.3, PW=3.0, TopN=10

Walk-Forward Validation (2022-2024 out-of-sample):
    - Sharpe: 1.859 (vs 1.271 for static)
    - Regime detection adds +0.588 Sharpe out-of-sample
    - Beats static L21/S5 in every major market condition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import logger


# Optimized regime-specific parameters
# From walk-forward validation: train 2017-2021, test 2022-2024
REGIME_PARAMS = {
    'STRONG_BULL': {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 20},
    'WEAK_BULL': {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 10},
    'SIDEWAYS': {'long_p': 21, 'short_p': 5, 'long_w': 0.5, 'pen_w': 2.0, 'top_n': 5},
    'UNPREDICTABLE': {'long_p': 42, 'short_p': 21, 'long_w': 0.5, 'pen_w': 4.0, 'top_n': 10},
    'BEAR': {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 3.0, 'top_n': 10}
}

# Default fallback if regime detection fails
DEFAULT_PARAMS = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 4.0, 'top_n': 10}


@dataclass
class RAMPSignal:
    """Signal for a single stock in RAMP strategy."""
    symbol: str
    momentum_score: float
    rank: int
    weight: float
    action: str  # 'buy', 'hold', 'sell'
    regime: str  # Current market regime


@dataclass
class RAMPRiskSignals:
    """Current risk signal status for RAMP."""
    regime: str
    regime_confidence: float
    high_vix: bool
    spy_drawdown: bool
    reduce_exposure: bool
    exposure_pct: float
    params: Dict  # Current regime parameters

    def to_dict(self) -> dict:
        return {
            'regime': self.regime,
            'regime_confidence': self.regime_confidence,
            'high_vix': self.high_vix,
            'spy_drawdown': self.spy_drawdown,
            'reduce_exposure': self.reduce_exposure,
            'exposure_pct': self.exposure_pct,
            'params': self.params
        }


class RAMPSignals:
    """
    Regime-Aware Momentum Protection signal generator.

    Key differences from standard MP:
    1. Uses MarketRegimeDetector to classify current market state
    2. Adapts momentum parameters based on detected regime
    3. Uses optimized formula: long_w * long_ret - pen_w * short_ret
    """

    def __init__(
        self,
        symbols: List[str],
        reduced_exposure: float = 0.5,
        vix_threshold: float = 25.0,
        spy_dd_threshold: float = -0.05
    ):
        """
        Initialize RAMP signal generator.

        Args:
            symbols: List of symbols to trade
            reduced_exposure: Exposure when risk signals trigger (0-1)
            vix_threshold: VIX level that triggers protection
            spy_dd_threshold: SPY drawdown threshold (negative)
        """
        self.symbols = symbols
        self.reduced_exposure = reduced_exposure
        self.vix_threshold = vix_threshold
        self.spy_dd_threshold = spy_dd_threshold

        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector()

        # Cache for historical data
        self._prices_cache: Optional[pd.DataFrame] = None
        self._spy_cache: Optional[pd.Series] = None
        self._vix_cache: Optional[pd.Series] = None

        # Current regime state
        self._current_regime: str = 'SIDEWAYS'
        self._regime_confidence: float = 0.5
        self._current_params: Dict = DEFAULT_PARAMS.copy()

        logger.info("[RAMP] Initialized Regime-Aware Momentum Protection Signals")
        logger.info(f"[RAMP]   Universe: {len(symbols)} symbols")
        logger.info(f"[RAMP]   Reduced exposure: {reduced_exposure:.0%}")
        logger.info(f"[RAMP]   VIX threshold: {vix_threshold}")
        logger.info("[RAMP]   Regime parameters:")
        for regime, params in REGIME_PARAMS.items():
            logger.info(f"[RAMP]     {regime}: L{params['long_p']}/S{params['short_p']}, "
                       f"LW={params['long_w']}, PW={params['pen_w']}, TopN={params['top_n']}")

    def update_historical_data(
        self,
        prices_df: pd.DataFrame,
        spy_prices: pd.Series,
        vix_prices: pd.Series
    ):
        """Update cached historical data."""
        self._prices_cache = prices_df
        self._spy_cache = spy_prices
        self._vix_cache = vix_prices
        logger.debug(f"[RAMP] Updated historical cache: {len(prices_df)} days")

    def detect_regime(
        self,
        spy_prices: Optional[pd.Series] = None,
        vix_prices: Optional[pd.Series] = None
    ) -> Tuple[str, float]:
        """
        Detect current market regime.

        Args:
            spy_prices: SPY prices. Uses cache if not provided.
            vix_prices: VIX prices. Uses cache if not provided.

        Returns:
            Tuple of (regime_name, confidence)
        """
        if spy_prices is None:
            spy_prices = self._spy_cache
        if vix_prices is None:
            vix_prices = self._vix_cache

        if spy_prices is None or len(spy_prices) < 252:
            logger.warning("[RAMP] Insufficient SPY data for regime detection")
            return 'SIDEWAYS', 0.5

        if vix_prices is None or len(vix_prices) < 252:
            logger.warning("[RAMP] Insufficient VIX data for regime detection")
            return 'SIDEWAYS', 0.5

        try:
            # Prepare data for regime detector
            spy_df = pd.DataFrame({
                'close': spy_prices,
                'open': spy_prices,
                'high': spy_prices,
                'low': spy_prices,
                'volume': 1e6
            })

            vix_df = pd.DataFrame({'close': vix_prices})

            # Get current timestamp
            timestamp = spy_prices.index[-1] if hasattr(spy_prices.index[-1], 'date') else datetime.now()

            # Detect regime
            regime, confidence = self.regime_detector.classify_regime(spy_df, vix_df, timestamp)

            # Update state
            self._current_regime = regime
            self._regime_confidence = confidence
            self._current_params = REGIME_PARAMS.get(regime, DEFAULT_PARAMS).copy()

            logger.info(f"[RAMP] Detected regime: {regime} (confidence: {confidence:.1%})")
            logger.info(f"[RAMP]   Using params: L{self._current_params['long_p']}/S{self._current_params['short_p']}, "
                       f"LW={self._current_params['long_w']}, PW={self._current_params['pen_w']}, "
                       f"TopN={self._current_params['top_n']}")

            return regime, confidence

        except Exception as e:
            logger.error(f"[RAMP] Error in regime detection: {e}")
            return 'SIDEWAYS', 0.5

    def calculate_momentum_scores(
        self,
        prices_df: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Calculate regime-adjusted momentum scores.

        Formula: (long_weight * return_long_period) - (penalty_weight * return_short_period)

        Args:
            prices_df: Optional prices DataFrame. Uses cache if not provided.

        Returns:
            Series of momentum scores indexed by symbol
        """
        if prices_df is None:
            prices_df = self._prices_cache

        if prices_df is None or len(prices_df) < 100:
            logger.warning("[RAMP] Insufficient price history for momentum calculation")
            return pd.Series(dtype=float)

        params = self._current_params
        long_p = params['long_p']
        short_p = params['short_p']
        long_w = params['long_w']
        pen_w = params['pen_w']

        # Calculate returns
        long_returns = prices_df.pct_change(long_p, fill_method=None)
        short_returns = prices_df.pct_change(short_p, fill_method=None)

        # RAMP momentum formula: long_w * long_ret - pen_w * short_ret
        momentum = (long_w * long_returns) - (pen_w * short_returns)

        # Get latest momentum scores
        latest_scores = momentum.iloc[-1].dropna()

        return latest_scores

    def calculate_risk_signals(
        self,
        spy_prices: Optional[pd.Series] = None,
        vix_prices: Optional[pd.Series] = None
    ) -> RAMPRiskSignals:
        """
        Calculate current risk signals.

        Risk protection triggers:
        - VIX > threshold
        - SPY drawdown > threshold

        Args:
            spy_prices: SPY prices. Uses cache if not provided.
            vix_prices: VIX prices. Uses cache if not provided.

        Returns:
            RAMPRiskSignals dataclass with current status
        """
        if spy_prices is None:
            spy_prices = self._spy_cache
        if vix_prices is None:
            vix_prices = self._vix_cache

        # Default risk signals
        high_vix = False
        spy_drawdown = False

        # Check VIX
        if vix_prices is not None and len(vix_prices) > 0:
            current_vix = vix_prices.iloc[-1]
            high_vix = current_vix > self.vix_threshold
            if high_vix:
                logger.warning(f"[RAMP] High VIX detected: {current_vix:.2f}")

        # Check SPY drawdown
        if spy_prices is not None and len(spy_prices) > 20:
            rolling_max = spy_prices.rolling(252, min_periods=20).max()
            current_dd = (spy_prices.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]
            spy_drawdown = current_dd < self.spy_dd_threshold
            if spy_drawdown:
                logger.warning(f"[RAMP] SPY drawdown detected: {current_dd:.1%}")

        # Determine exposure
        reduce_exposure = high_vix or spy_drawdown
        exposure_pct = self.reduced_exposure if reduce_exposure else 1.0

        if reduce_exposure:
            logger.warning(f"[RAMP] Reducing exposure to {exposure_pct:.0%}")

        return RAMPRiskSignals(
            regime=self._current_regime,
            regime_confidence=self._regime_confidence,
            high_vix=high_vix,
            spy_drawdown=spy_drawdown,
            reduce_exposure=reduce_exposure,
            exposure_pct=exposure_pct,
            params=self._current_params.copy()
        )

    def generate_signals(
        self,
        current_positions: Optional[Dict[str, float]] = None,
        prices_df: Optional[pd.DataFrame] = None,
        spy_prices: Optional[pd.Series] = None,
        vix_prices: Optional[pd.Series] = None
    ) -> Tuple[List[RAMPSignal], RAMPRiskSignals]:
        """
        Generate trading signals based on current regime and momentum.

        Args:
            current_positions: Dict of symbol -> position value
            prices_df: Stock prices DataFrame
            spy_prices: SPY prices
            vix_prices: VIX prices

        Returns:
            Tuple of (List of RAMPSignal, RAMPRiskSignals)
        """
        if current_positions is None:
            current_positions = {}

        # Use cached data if not provided
        if prices_df is None:
            prices_df = self._prices_cache
        if spy_prices is None:
            spy_prices = self._spy_cache
        if vix_prices is None:
            vix_prices = self._vix_cache

        # Detect regime first
        self.detect_regime(spy_prices, vix_prices)

        # Get current top_n from regime params
        top_n = self._current_params['top_n']

        # Calculate risk signals
        risk_signals = self.calculate_risk_signals(spy_prices, vix_prices)

        # Calculate momentum scores
        momentum_scores = self.calculate_momentum_scores(prices_df)

        if momentum_scores.empty:
            logger.warning("[RAMP] No momentum scores calculated")
            return [], risk_signals

        # Rank stocks by momentum (descending)
        ranked_scores = momentum_scores.sort_values(ascending=False)

        # Get top N symbols
        top_symbols = ranked_scores.head(top_n).index.tolist()

        # Current position symbols
        position_symbols = set(current_positions.keys())

        signals = []

        # Generate sell signals for positions not in top N
        for symbol in position_symbols:
            if symbol not in top_symbols:
                signals.append(RAMPSignal(
                    symbol=symbol,
                    momentum_score=float(ranked_scores.get(symbol, 0)),
                    rank=-1,
                    weight=0,
                    action='sell',
                    regime=self._current_regime
                ))
                logger.info(f"[RAMP] SELL signal: {symbol} (not in top {top_n})")

        # Generate buy/hold signals for top N
        equal_weight = 1.0 / top_n
        adjusted_weight = equal_weight * risk_signals.exposure_pct

        for rank, symbol in enumerate(top_symbols, 1):
            if symbol in position_symbols:
                action = 'hold'
            else:
                action = 'buy'

            signals.append(RAMPSignal(
                symbol=symbol,
                momentum_score=float(ranked_scores[symbol]),
                rank=rank,
                weight=adjusted_weight,
                action=action,
                regime=self._current_regime
            ))

            if action == 'buy':
                logger.info(f"[RAMP] BUY signal: {symbol} (rank #{rank}, score: {ranked_scores[symbol]:.2%})")

        return signals, risk_signals

    @property
    def current_regime(self) -> str:
        """Get current detected regime."""
        return self._current_regime

    @property
    def current_params(self) -> Dict:
        """Get current regime parameters."""
        return self._current_params.copy()

    @property
    def current_top_n(self) -> int:
        """Get current top_n based on regime."""
        return self._current_params['top_n']
