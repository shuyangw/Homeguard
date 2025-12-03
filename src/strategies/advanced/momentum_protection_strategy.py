"""
Momentum Strategy with Rule-Based Crash Protection.

This strategy trades top momentum stocks with rule-based crash protection
to reduce exposure during high-risk periods.

Strategy Overview:
1. Universe: S&P 500 stocks
2. Selection: Top N stocks by 12-1 month momentum
3. Rebalance: Daily at market close (3:55 PM EST)
4. Protection: Reduce exposure to 50% when risk signals trigger

Risk Signals (any triggers protection):
1. VIX > 25 (high fear)
2. VIX spike > 20% in 5 days
3. SPY drawdown > 5%
4. Momentum volatility in top 10% of past year
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from dataclasses import dataclass

from src.utils.logger import logger


@dataclass
class MomentumSignal:
    """Signal for a single stock."""
    symbol: str
    momentum_score: float
    rank: int
    weight: float
    action: str  # 'buy', 'hold', 'sell'


@dataclass
class RiskSignals:
    """Current risk signal status."""
    high_vix: bool
    vix_spike: bool
    spy_drawdown: bool
    high_mom_vol: bool
    reduce_exposure: bool
    exposure_pct: float

    def to_dict(self) -> dict:
        return {
            'high_vix': self.high_vix,
            'vix_spike': self.vix_spike,
            'spy_drawdown': self.spy_drawdown,
            'high_mom_vol': self.high_mom_vol,
            'reduce_exposure': self.reduce_exposure,
            'exposure_pct': self.exposure_pct
        }


class MomentumProtectionSignals:
    """
    Pure signal generator for momentum with crash protection.

    Can be used standalone or injected into live adapter.
    """

    def __init__(
        self,
        symbols: List[str],
        top_n: int = 10,
        reduced_exposure: float = 0.5,
        vix_threshold: float = 25.0,
        vix_spike_threshold: float = 0.20,
        spy_dd_threshold: float = -0.05,
        mom_vol_percentile: float = 0.90
    ):
        """
        Initialize momentum signal generator.

        Args:
            symbols: List of symbols to trade
            top_n: Number of top momentum stocks to hold
            reduced_exposure: Exposure when risk signals trigger (0-1)
            vix_threshold: VIX level that triggers protection
            vix_spike_threshold: VIX 5-day change that triggers protection
            spy_dd_threshold: SPY drawdown threshold (negative)
            mom_vol_percentile: Momentum volatility percentile threshold
        """
        self.symbols = symbols
        self.top_n = top_n
        self.reduced_exposure = reduced_exposure
        self.vix_threshold = vix_threshold
        self.vix_spike_threshold = vix_spike_threshold
        self.spy_dd_threshold = spy_dd_threshold
        self.mom_vol_percentile = mom_vol_percentile

        # Cache for historical data
        self._prices_cache: Optional[pd.DataFrame] = None
        self._spy_cache: Optional[pd.Series] = None
        self._vix_cache: Optional[pd.Series] = None

        logger.info("[MP] Initialized Momentum Protection Signals")
        logger.info(f"[MP]   Universe: {len(symbols)} symbols")
        logger.info(f"[MP]   Top N: {top_n}")
        logger.info(f"[MP]   Reduced exposure: {reduced_exposure:.0%}")
        logger.info(f"[MP]   VIX threshold: {vix_threshold}")

    def update_historical_data(
        self,
        prices_df: pd.DataFrame,
        spy_prices: pd.Series,
        vix_prices: pd.Series
    ):
        """
        Update cached historical data.

        Args:
            prices_df: DataFrame with stock prices (columns=symbols)
            spy_prices: SPY close prices series
            vix_prices: VIX close prices series
        """
        self._prices_cache = prices_df
        self._spy_cache = spy_prices
        self._vix_cache = vix_prices

        logger.debug(f"[MP] Updated historical cache: {len(prices_df)} days")

    def calculate_momentum_scores(
        self,
        prices_df: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Calculate 12-1 month momentum scores for all symbols.

        Args:
            prices_df: Optional prices DataFrame. Uses cache if not provided.

        Returns:
            Series of momentum scores indexed by symbol
        """
        if prices_df is None:
            prices_df = self._prices_cache

        if prices_df is None or len(prices_df) < 253:
            logger.warning("[MP] Insufficient price history for momentum calculation")
            return pd.Series(dtype=float)

        # 12-1 month momentum (skip most recent month)
        returns_12m = prices_df.pct_change(252, fill_method=None)
        returns_1m = prices_df.pct_change(21, fill_method=None)
        momentum = returns_12m - returns_1m

        # Get latest momentum scores
        latest_scores = momentum.iloc[-1].dropna()

        return latest_scores

    def calculate_risk_signals(
        self,
        spy_prices: Optional[pd.Series] = None,
        vix_prices: Optional[pd.Series] = None,
        prices_df: Optional[pd.DataFrame] = None
    ) -> RiskSignals:
        """
        Calculate current risk signals.

        Args:
            spy_prices: SPY prices. Uses cache if not provided.
            vix_prices: VIX prices. Uses cache if not provided.
            prices_df: Stock prices for momentum volatility calc.

        Returns:
            RiskSignals dataclass with current status
        """
        spy = spy_prices if spy_prices is not None else self._spy_cache
        vix = vix_prices if vix_prices is not None else self._vix_cache
        prices = prices_df if prices_df is not None else self._prices_cache

        # Ensure spy and vix are Series, not DataFrames
        if isinstance(spy, pd.DataFrame):
            spy = spy.iloc[:, 0] if len(spy.columns) > 0 else pd.Series()
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0] if len(vix.columns) > 0 else pd.Series()

        # Default to no risk if data missing
        if spy is None or vix is None or len(spy) < 252 or len(vix) < 5:
            logger.warning("[MP] Insufficient data for risk signals, defaulting to no risk")
            return RiskSignals(
                high_vix=False,
                vix_spike=False,
                spy_drawdown=False,
                high_mom_vol=False,
                reduce_exposure=False,
                exposure_pct=1.0
            )

        # Rule 1: High VIX
        current_vix = float(vix.iloc[-1])
        high_vix = current_vix > self.vix_threshold

        # Rule 2: VIX spike (5-day change)
        if len(vix) >= 5:
            vix_5d_ago = vix.iloc[-5]
            vix_change = (current_vix - vix_5d_ago) / vix_5d_ago if vix_5d_ago > 0 else 0
            vix_spike = vix_change > self.vix_spike_threshold
        else:
            vix_spike = False

        # Rule 3: SPY drawdown
        spy_max = spy.max()
        spy_current = spy.iloc[-1]
        spy_dd = (spy_current / spy_max) - 1
        spy_drawdown = spy_dd < self.spy_dd_threshold

        # Rule 4: High momentum volatility
        high_mom_vol = False
        if prices is not None and len(prices) >= 252:
            # Calculate momentum factor returns
            mom_factor_ret = self._calculate_momentum_factor_returns(prices)
            if len(mom_factor_ret) >= 252:
                mom_vol_21d = mom_factor_ret.rolling(21).std().iloc[-1] * np.sqrt(252)
                mom_vol_threshold = mom_factor_ret.rolling(21).std().rolling(252).quantile(
                    self.mom_vol_percentile
                ).iloc[-1] * np.sqrt(252)
                high_mom_vol = mom_vol_21d > mom_vol_threshold if pd.notna(mom_vol_threshold) else False

        # Combined signal
        reduce_exposure = high_vix or vix_spike or spy_drawdown or high_mom_vol
        exposure_pct = self.reduced_exposure if reduce_exposure else 1.0

        return RiskSignals(
            high_vix=high_vix,
            vix_spike=vix_spike,
            spy_drawdown=spy_drawdown,
            high_mom_vol=high_mom_vol,
            reduce_exposure=reduce_exposure,
            exposure_pct=exposure_pct
        )

    def _calculate_momentum_factor_returns(
        self,
        prices_df: pd.DataFrame
    ) -> pd.Series:
        """Calculate long-short momentum factor returns."""
        returns_12m = prices_df.pct_change(252, fill_method=None)
        returns_1m = prices_df.pct_change(21, fill_method=None)
        momentum = returns_12m - returns_1m
        daily_returns = prices_df.pct_change(fill_method=None)

        factor_returns = []

        for i in range(253, len(prices_df)):
            scores = momentum.iloc[i-1].dropna()
            if len(scores) < self.top_n * 2:
                continue

            top_stocks = scores.nlargest(self.top_n).index
            bottom_stocks = scores.nsmallest(self.top_n).index

            date = prices_df.index[i]
            long_ret = daily_returns.loc[date, top_stocks].mean()
            short_ret = daily_returns.loc[date, bottom_stocks].mean()
            factor_ret = long_ret - short_ret

            factor_returns.append({'date': date, 'return': factor_ret})

        if not factor_returns:
            return pd.Series(dtype=float)

        return pd.DataFrame(factor_returns).set_index('date')['return']

    def generate_signals(
        self,
        current_positions: Dict[str, float],
        prices_df: Optional[pd.DataFrame] = None,
        spy_prices: Optional[pd.Series] = None,
        vix_prices: Optional[pd.Series] = None
    ) -> Tuple[List[MomentumSignal], RiskSignals]:
        """
        Generate trading signals.

        Args:
            current_positions: Dict of symbol -> position value
            prices_df: Stock prices DataFrame
            spy_prices: SPY prices
            vix_prices: VIX prices

        Returns:
            Tuple of (list of MomentumSignal, RiskSignals)
        """
        # Use provided data or cache
        prices = prices_df if prices_df is not None else self._prices_cache
        spy = spy_prices if spy_prices is not None else self._spy_cache
        vix = vix_prices if vix_prices is not None else self._vix_cache

        if prices is None:
            logger.error("[MP] No price data available for signal generation")
            return [], RiskSignals(False, False, False, False, False, 1.0)

        # Calculate momentum scores
        momentum_scores = self.calculate_momentum_scores(prices)

        if len(momentum_scores) < self.top_n:
            logger.warning(f"[MP] Only {len(momentum_scores)} stocks with momentum scores")
            return [], RiskSignals(False, False, False, False, False, 1.0)

        # Get risk signals
        risk_signals = self.calculate_risk_signals(spy, vix, prices)

        # Select top N momentum stocks
        top_stocks = momentum_scores.nlargest(self.top_n)

        # Calculate target weights
        base_weight = 1.0 / self.top_n
        adjusted_weight = base_weight * risk_signals.exposure_pct

        signals = []
        current_symbols = set(current_positions.keys())
        target_symbols = set(top_stocks.index)

        # Generate signals
        for rank, (symbol, score) in enumerate(top_stocks.items(), 1):
            if symbol in current_symbols:
                action = 'hold'
            else:
                action = 'buy'

            signals.append(MomentumSignal(
                symbol=symbol,
                momentum_score=score,
                rank=rank,
                weight=adjusted_weight,
                action=action
            ))

        # Sell signals for positions not in top N
        for symbol in current_symbols - target_symbols:
            signals.append(MomentumSignal(
                symbol=symbol,
                momentum_score=momentum_scores.get(symbol, 0),
                rank=999,
                weight=0,
                action='sell'
            ))

        return signals, risk_signals


class MomentumProtectionStrategy:
    """
    Main strategy class for momentum with crash protection.

    Integrates:
    - Momentum signal generation
    - Rule-based crash protection
    - Position tracking
    - Performance metrics
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize momentum protection strategy.

        Args:
            params: Strategy parameters including:
                - symbols: List of symbols to trade
                - top_n: Number of positions (default 10)
                - position_size: Base position size (default 0.10)
                - reduced_exposure: Exposure when risk high (default 0.5)
                - vix_threshold: VIX level threshold (default 25)
                - slippage_per_share: Slippage in dollars (default 0.01)
        """
        params = params or {}

        self.symbols = params.get('symbols', [])
        self.top_n = params.get('top_n', 10)
        self.position_size = params.get('position_size', 0.10)
        self.reduced_exposure = params.get('reduced_exposure', 0.5)
        self.vix_threshold = params.get('vix_threshold', 25.0)
        self.vix_spike_threshold = params.get('vix_spike_threshold', 0.20)
        self.spy_dd_threshold = params.get('spy_dd_threshold', -0.05)
        self.mom_vol_percentile = params.get('mom_vol_percentile', 0.90)
        self.slippage_per_share = params.get('slippage_per_share', 0.01)

        # Initialize signal generator
        self.signal_generator = MomentumProtectionSignals(
            symbols=self.symbols,
            top_n=self.top_n,
            reduced_exposure=self.reduced_exposure,
            vix_threshold=self.vix_threshold,
            vix_spike_threshold=self.vix_spike_threshold,
            spy_dd_threshold=self.spy_dd_threshold,
            mom_vol_percentile=self.mom_vol_percentile
        )

        # Track positions and history
        self.positions: Dict[str, float] = {}
        self.trade_history: List[Dict] = []
        self.risk_history: List[Dict] = []

        logger.info("[MP] Initialized Momentum Protection Strategy")
        logger.info(f"[MP]   Top N: {self.top_n}")
        logger.info(f"[MP]   Position Size: {self.position_size:.0%}")
        logger.info(f"[MP]   Reduced Exposure: {self.reduced_exposure:.0%}")

    def update_data(
        self,
        prices_df: pd.DataFrame,
        spy_prices: pd.Series,
        vix_prices: pd.Series
    ):
        """Update historical data for signal generation."""
        self.signal_generator.update_historical_data(prices_df, spy_prices, vix_prices)

    def generate_signals(
        self,
        timestamp: Optional[datetime] = None
    ) -> Tuple[List[MomentumSignal], RiskSignals]:
        """
        Generate trading signals based on current market data.

        Args:
            timestamp: Current timestamp (for logging)

        Returns:
            Tuple of (signals list, risk signals)
        """
        signals, risk = self.signal_generator.generate_signals(
            current_positions=self.positions
        )

        # Log risk status
        if risk.reduce_exposure:
            triggers = []
            if risk.high_vix:
                triggers.append("VIX>25")
            if risk.vix_spike:
                triggers.append("VIX_SPIKE")
            if risk.spy_drawdown:
                triggers.append("SPY_DD")
            if risk.high_mom_vol:
                triggers.append("MOM_VOL")
            logger.warning(f"[MP] Risk signals triggered: {', '.join(triggers)} - Exposure: {risk.exposure_pct:.0%}")

        # Store risk history
        self.risk_history.append({
            'timestamp': timestamp or datetime.now(),
            **risk.to_dict()
        })

        return signals, risk

    def get_target_positions(
        self,
        portfolio_value: float,
        signals: List[MomentumSignal]
    ) -> Dict[str, float]:
        """
        Calculate target position sizes.

        Args:
            portfolio_value: Current portfolio value
            signals: List of momentum signals

        Returns:
            Dict of symbol -> target position value
        """
        targets = {}

        for signal in signals:
            if signal.action in ('buy', 'hold'):
                position_value = portfolio_value * self.position_size * signal.weight * self.top_n
                targets[signal.symbol] = position_value

        return targets

    def get_performance_stats(self) -> Dict:
        """Get strategy performance statistics."""
        if not self.trade_history:
            return {}

        trades_df = pd.DataFrame(self.trade_history)

        total_trades = len(trades_df)
        if 'pnl' in trades_df.columns:
            win_rate = (trades_df['pnl'] > 0).mean()
            total_pnl = trades_df['pnl'].sum()
            avg_pnl = trades_df['pnl'].mean()
        else:
            win_rate = 0
            total_pnl = 0
            avg_pnl = 0

        risk_df = pd.DataFrame(self.risk_history)
        if not risk_df.empty:
            pct_risk_on = risk_df['reduce_exposure'].mean()
            avg_exposure = risk_df['exposure_pct'].mean()
        else:
            pct_risk_on = 0
            avg_exposure = 1.0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'pct_risk_on': pct_risk_on,
            'avg_exposure': avg_exposure
        }
