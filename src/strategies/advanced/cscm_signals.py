"""
Cross-Sectional Crypto Momentum (CSCM) Signals for Live Trading.

Signal generator for live/paper trading that calculates real-time momentum
rankings and generates position targets based on BTC regime.

Usage:
    from src.strategies.advanced.cscm_signals import CSCMSignals

    signals = CSCMSignals(
        symbols=['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'LINK/USD'],
        top_n=3,
    )

    # Update with latest data
    signals.update_historical_data(prices_dict)

    # Get current positions
    risk_signals = signals.get_risk_signals()
    target_positions = signals.get_target_positions()
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.advanced.cscm_indicators import CSCMIndicators, CSCMSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CSCMPositionSignal:
    """Position signal for a single crypto asset."""
    symbol: str
    momentum_score: float
    rank: int
    weight: float
    action: str  # 'buy', 'hold', 'sell'
    regime: str  # Current BTC regime


@dataclass
class CSCMRiskSignals:
    """Current risk/regime status for CSCM strategy."""
    regime: str  # 'bullish' or 'bearish'
    btc_price: float
    btc_sma: float
    is_rebalance_day: bool
    reduce_exposure: bool
    exposure_pct: float
    top_symbols: List[str]
    momentum_scores: Dict[str, float]
    # Trailing stop fields
    trailing_stop_triggered: bool = False
    current_drawdown: float = 0.0
    peak_value: float = 0.0

    def to_dict(self) -> dict:
        return {
            'regime': self.regime,
            'btc_price': self.btc_price,
            'btc_sma': self.btc_sma,
            'is_rebalance_day': self.is_rebalance_day,
            'reduce_exposure': self.reduce_exposure,
            'exposure_pct': self.exposure_pct,
            'top_symbols': self.top_symbols,
            'momentum_scores': self.momentum_scores,
            'trailing_stop_triggered': self.trailing_stop_triggered,
            'current_drawdown': self.current_drawdown,
            'peak_value': self.peak_value,
        }


class CSCMSignals:
    """
    Cross-Sectional Crypto Momentum signal generator for live trading.

    Generates position targets based on:
    1. Cross-sectional momentum ranking (14-day returns)
    2. BTC regime filter (BTC > 20-day SMA = bullish)
    3. Weekly rebalancing (Sunday 0:00 UTC)

    Key Features:
    - Real-time momentum calculation
    - BTC regime awareness (go to cash in bear markets)
    - Configurable universe and parameters
    """

    def __init__(
        self,
        symbols: List[str],
        top_n: int = 5,
        momentum_period: int = 14,
        btc_sma_period: int = 20,
        rebalance_day: str = 'sunday',
        go_to_cash_in_bear: bool = True,
        btc_symbol: str = 'BTC/USD',
        trailing_stop_pct: float = 0.25
    ):
        """
        Initialize CSCM signal generator.

        Args:
            symbols: List of crypto symbols to trade (e.g., ['BTC/USD', 'ETH/USD'])
            top_n: Number of top coins to hold
            momentum_period: Days for momentum calculation
            btc_sma_period: BTC SMA period for regime filter
            rebalance_day: Day of week to rebalance
            go_to_cash_in_bear: Exit all in bearish regime
            btc_symbol: Symbol for BTC regime filter
            trailing_stop_pct: Trailing stop percentage (0.25 = 25%)
        """
        self.symbols = symbols
        self.top_n = top_n
        self.momentum_period = momentum_period
        self.btc_sma_period = btc_sma_period
        self.rebalance_day = rebalance_day
        self.go_to_cash_in_bear = go_to_cash_in_bear
        self.btc_symbol = btc_symbol
        self.trailing_stop_pct = trailing_stop_pct

        # Cache for historical data
        self._prices_cache: Dict[str, pd.DataFrame] = {}
        self._btc_cache: Optional[pd.DataFrame] = None

        # Current state
        self._current_regime: str = 'bearish'  # Start defensive
        self._current_positions: Dict[str, float] = {}
        self._last_rebalance_date: Optional[datetime] = None

        # Trailing stop state
        self._peak_portfolio_value: float = 0.0
        self._current_portfolio_value: float = 0.0
        self._trailing_stop_triggered: bool = False

        logger.info("[CSCM] Initialized Cross-Sectional Crypto Momentum Signals")
        logger.info(f"[CSCM]   Universe: {len(symbols)} symbols")
        logger.info(f"[CSCM]   Top N: {top_n}")
        logger.info(f"[CSCM]   Momentum Period: {momentum_period} days")
        logger.info(f"[CSCM]   BTC SMA Period: {btc_sma_period}")
        logger.info(f"[CSCM]   Rebalance Day: {rebalance_day}")
        logger.info(f"[CSCM]   Go to Cash in Bear: {go_to_cash_in_bear}")
        logger.info(f"[CSCM]   Trailing Stop: {trailing_stop_pct:.0%}")

    def update_historical_data(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        btc_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Update cached historical data.

        Args:
            prices_dict: Dictionary of {symbol: DataFrame} with OHLCV data
            btc_data: BTC OHLCV data (if None, uses BTC from prices_dict)
        """
        self._prices_cache = prices_dict

        if btc_data is not None:
            self._btc_cache = btc_data
        elif self.btc_symbol in prices_dict:
            self._btc_cache = prices_dict[self.btc_symbol]

        # Update regime detection
        if self._btc_cache is not None and len(self._btc_cache) >= self.btc_sma_period:
            btc_close = self._btc_cache['close'] if 'close' in self._btc_cache.columns else self._btc_cache['Close']
            btc_sma = CSCMIndicators.calculate_sma(btc_close, self.btc_sma_period)
            self._current_regime = 'bullish' if btc_close.iloc[-1] > btc_sma.iloc[-1] else 'bearish'

        logger.debug(f"[CSCM] Updated historical cache: {len(prices_dict)} symbols")
        logger.debug(f"[CSCM] Current regime: {self._current_regime}")

    def update_portfolio_value(self, current_value: float) -> None:
        """
        Update portfolio value for trailing stop calculation.

        Args:
            current_value: Current total portfolio value in USD
        """
        self._current_portfolio_value = current_value

        # Update peak if new high
        if current_value > self._peak_portfolio_value:
            self._peak_portfolio_value = current_value
            logger.debug(f"[CSCM] New peak value: ${current_value:,.2f}")

        # Check trailing stop
        if self._peak_portfolio_value > 0:
            drawdown = (self._peak_portfolio_value - current_value) / self._peak_portfolio_value

            if drawdown >= self.trailing_stop_pct and not self._trailing_stop_triggered:
                self._trailing_stop_triggered = True
                logger.warning(f"[CSCM] TRAILING STOP TRIGGERED! "
                             f"DD={drawdown:.1%}, Peak=${self._peak_portfolio_value:,.0f}, "
                             f"Current=${current_value:,.0f}")

    def reset_trailing_stop(self) -> None:
        """Reset trailing stop after rebalance (allows re-entry)."""
        if self._trailing_stop_triggered:
            logger.info("[CSCM] Resetting trailing stop for new rebalance")
        self._trailing_stop_triggered = False
        # Keep peak value - don't reset to allow tracking overall drawdown

    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        if self._peak_portfolio_value <= 0:
            return 0.0
        return (self._peak_portfolio_value - self._current_portfolio_value) / self._peak_portfolio_value

    def is_trailing_stop_active(self) -> bool:
        """Check if trailing stop is currently triggered."""
        return self._trailing_stop_triggered

    def get_risk_signals(self) -> CSCMRiskSignals:
        """
        Get current risk/regime status.

        Returns:
            CSCMRiskSignals with current market state
        """
        if self._btc_cache is None or len(self._btc_cache) < self.btc_sma_period:
            logger.warning("[CSCM] Insufficient BTC data for risk signals")
            return CSCMRiskSignals(
                regime='bearish',
                btc_price=0.0,
                btc_sma=0.0,
                is_rebalance_day=False,
                reduce_exposure=True,
                exposure_pct=0.0,
                top_symbols=[],
                momentum_scores={}
            )

        # Get BTC data
        btc_close = self._btc_cache['close'] if 'close' in self._btc_cache.columns else self._btc_cache['Close']
        btc_sma = CSCMIndicators.calculate_sma(btc_close, self.btc_sma_period)
        current_btc = float(btc_close.iloc[-1])
        current_sma = float(btc_sma.iloc[-1])

        # Calculate momentum for all symbols
        momentum_scores = {}
        for symbol, df in self._prices_cache.items():
            if symbol not in self.symbols or df is None or df.empty:
                continue
            close = df['close'] if 'close' in df.columns else df['Close']
            mom = CSCMIndicators.calculate_momentum(close, self.momentum_period)
            if len(mom) > 0 and not pd.isna(mom.iloc[-1]):
                momentum_scores[symbol] = float(mom.iloc[-1])

        # Rank and get top symbols
        sorted_symbols = sorted(
            momentum_scores.keys(),
            key=lambda x: momentum_scores[x],
            reverse=True
        )
        top_symbols = sorted_symbols[:self.top_n]

        # Determine regime and exposure
        regime = 'bullish' if current_btc > current_sma else 'bearish'
        reduce_exposure = self.go_to_cash_in_bear and regime == 'bearish'

        # Also reduce exposure if trailing stop triggered
        if self._trailing_stop_triggered:
            reduce_exposure = True

        exposure_pct = 0.0 if reduce_exposure else 1.0

        # Check if today is rebalance day
        current_date = btc_close.index[-1]
        is_rebalance = CSCMIndicators.is_weekly_rebalance_day(current_date, self.rebalance_day)

        return CSCMRiskSignals(
            regime=regime,
            btc_price=current_btc,
            btc_sma=current_sma,
            is_rebalance_day=is_rebalance,
            reduce_exposure=reduce_exposure,
            exposure_pct=exposure_pct,
            top_symbols=top_symbols,
            momentum_scores=momentum_scores,
            trailing_stop_triggered=self._trailing_stop_triggered,
            current_drawdown=self.get_current_drawdown(),
            peak_value=self._peak_portfolio_value
        )

    def get_target_positions(self) -> Dict[str, float]:
        """
        Get target position weights for current state.

        Returns:
            Dictionary of {symbol: weight} where weight is 0.0 to 1.0
        """
        risk_signals = self.get_risk_signals()

        # If trailing stop triggered or in bearish regime
        if risk_signals.reduce_exposure:
            reason = "trailing stop" if risk_signals.trailing_stop_triggered else "bearish regime"
            logger.info(f"[CSCM] {reason.capitalize()} - targeting 0% exposure")
            return {sym: 0.0 for sym in self.symbols}

        # Calculate weights for top symbols
        positions = {}
        weight = 1.0 / self.top_n if risk_signals.top_symbols else 0.0

        for sym in self.symbols:
            if sym in risk_signals.top_symbols:
                positions[sym] = weight
            else:
                positions[sym] = 0.0

        return positions

    def generate_position_signals(
        self,
        current_positions: Dict[str, float]
    ) -> List[CSCMPositionSignal]:
        """
        Generate position signals comparing current vs target.

        Args:
            current_positions: Current position weights {symbol: weight}

        Returns:
            List of CSCMPositionSignal with action recommendations
        """
        risk_signals = self.get_risk_signals()
        target_positions = self.get_target_positions()

        signals = []
        for symbol in self.symbols:
            current = current_positions.get(symbol, 0.0)
            target = target_positions.get(symbol, 0.0)

            # Determine action
            if abs(target - current) < 0.01:
                action = 'hold'
            elif target > current:
                action = 'buy'
            else:
                action = 'sell'

            # Get rank and momentum
            rank = 0
            momentum = 0.0
            if symbol in risk_signals.momentum_scores:
                momentum = risk_signals.momentum_scores[symbol]
                sorted_symbols = sorted(
                    risk_signals.momentum_scores.keys(),
                    key=lambda x: risk_signals.momentum_scores[x],
                    reverse=True
                )
                if symbol in sorted_symbols:
                    rank = sorted_symbols.index(symbol) + 1

            signals.append(CSCMPositionSignal(
                symbol=symbol,
                momentum_score=momentum,
                rank=rank,
                weight=target,
                action=action,
                regime=risk_signals.regime
            ))

        return signals

    def should_rebalance(self, current_date: Optional[datetime] = None) -> bool:
        """
        Check if rebalancing should occur.

        Args:
            current_date: Date to check (defaults to latest data date)

        Returns:
            True if rebalancing should occur
        """
        if current_date is None:
            if self._btc_cache is None or len(self._btc_cache) == 0:
                return False
            current_date = self._btc_cache.index[-1]

        is_rebalance_day = CSCMIndicators.is_weekly_rebalance_day(
            current_date, self.rebalance_day
        )

        # Only rebalance once per day
        if self._last_rebalance_date is not None:
            if isinstance(current_date, pd.Timestamp):
                current_day = current_date.date()
            else:
                current_day = current_date.date() if hasattr(current_date, 'date') else current_date

            if isinstance(self._last_rebalance_date, pd.Timestamp):
                last_day = self._last_rebalance_date.date()
            else:
                last_day = self._last_rebalance_date.date() if hasattr(self._last_rebalance_date, 'date') else self._last_rebalance_date

            if current_day == last_day:
                return False

        return is_rebalance_day

    def mark_rebalanced(self, date: datetime) -> None:
        """Mark that rebalancing has occurred."""
        self._last_rebalance_date = date
        logger.info(f"[CSCM] Marked rebalance complete for {date}")

    def get_summary(self) -> Dict:
        """
        Get summary of current strategy state.

        Returns:
            Dictionary with state summary
        """
        risk_signals = self.get_risk_signals()
        target_positions = self.get_target_positions()

        return {
            'regime': risk_signals.regime,
            'btc_price': risk_signals.btc_price,
            'btc_sma': risk_signals.btc_sma,
            'is_rebalance_day': risk_signals.is_rebalance_day,
            'exposure_pct': risk_signals.exposure_pct,
            'top_symbols': risk_signals.top_symbols,
            'target_positions': target_positions,
            'universe_size': len(self.symbols),
            'top_n': self.top_n,
            'trailing_stop_triggered': risk_signals.trailing_stop_triggered,
            'current_drawdown': risk_signals.current_drawdown,
            'peak_value': risk_signals.peak_value,
            'trailing_stop_pct': self.trailing_stop_pct,
        }
