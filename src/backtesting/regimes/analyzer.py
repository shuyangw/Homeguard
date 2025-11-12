"""
Regime-based performance analysis.

Analyzes strategy performance across different market regimes
to assess robustness and identify failure conditions.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.backtesting.regimes.detector import (
    TrendDetector,
    VolatilityDetector,
    DrawdownDetector,
    RegimeLabel,
    RegimePeriod
)
from src.utils import logger


@dataclass
class RegimePerformance:
    """Performance metrics for a specific regime."""
    regime: RegimeLabel
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    num_periods: int  # Number of time periods in this regime


@dataclass
class RegimeAnalysisResults:
    """Results from regime-based analysis."""

    # Performance by regime type
    trend_performance: Dict[RegimeLabel, RegimePerformance]
    volatility_performance: Dict[RegimeLabel, RegimePerformance]
    drawdown_performance: Dict[RegimeLabel, RegimePerformance]

    # Robustness metrics
    robustness_score: float  # 0-100, higher = more consistent across regimes
    worst_regime: str  # Which regime performed worst
    best_regime: str  # Which regime performed best

    # Overall stats
    overall_sharpe: float
    overall_return: float

    def print_summary(self):
        """Print regime analysis summary."""
        logger.blank()
        logger.separator()
        logger.header("REGIME-BASED PERFORMANCE ANALYSIS")
        logger.separator()
        logger.blank()

        logger.info(f"Overall Sharpe Ratio: {self.overall_sharpe:.2f}")
        logger.info(f"Overall Return: {self.overall_return:.1f}%")
        logger.blank()

        # Robustness score
        if self.robustness_score >= 70:
            logger.success(f"Robustness Score: {self.robustness_score:.1f}/100 (Excellent)")
            logger.success("Strategy is highly consistent across market conditions")
        elif self.robustness_score >= 50:
            logger.info(f"Robustness Score: {self.robustness_score:.1f}/100 (Good)")
            logger.info("Strategy shows reasonable consistency")
        else:
            logger.warning(f"Robustness Score: {self.robustness_score:.1f}/100 (Poor)")
            logger.warning("Strategy performance varies significantly by regime")

        logger.blank()
        logger.success(f"Best Regime: {self.best_regime}")
        logger.warning(f"Worst Regime: {self.worst_regime}")
        logger.blank()

        # Trend regimes
        logger.header("TREND REGIME PERFORMANCE")
        self._print_regime_table(self.trend_performance)
        logger.blank()

        # Volatility regimes
        logger.header("VOLATILITY REGIME PERFORMANCE")
        self._print_regime_table(self.volatility_performance)
        logger.blank()

        # Drawdown regimes
        logger.header("DRAWDOWN REGIME PERFORMANCE")
        self._print_regime_table(self.drawdown_performance)
        logger.blank()

        logger.separator()

    def _print_regime_table(self, performance: Dict[RegimeLabel, RegimePerformance]):
        """Print performance table for a regime type."""
        if not performance:
            logger.info("  No data available")
            return

        logger.info(f"{'Regime':<20} {'Sharpe':<10} {'Return':<12} {'Drawdown':<12} {'Trades':<8}")
        logger.info("-" * 70)

        for regime_label, perf in performance.items():
            regime_name = regime_label.value
            sharpe_str = f"{perf.sharpe_ratio:.2f}"
            return_str = f"{perf.total_return:.1f}%"
            dd_str = f"{perf.max_drawdown:.1f}%"
            trades_str = f"{perf.num_trades}"

            logger.info(f"{regime_name:<20} {sharpe_str:<10} {return_str:<12} {dd_str:<12} {trades_str:<8}")


class RegimeAnalyzer:
    """
    Analyzes strategy performance across market regimes.

    Breaks down backtest results by:
    - Trend regimes (Bull/Bear/Sideways)
    - Volatility regimes (High/Low)
    - Drawdown regimes (Drawdown/Recovery/Calm)
    """

    def __init__(
        self,
        trend_lookback: int = 60,
        vol_lookback: int = 20,
        drawdown_threshold: float = 10.0
    ):
        """
        Initialize regime analyzer.

        Args:
            trend_lookback: Days for trend detection
            vol_lookback: Days for volatility calculation
            drawdown_threshold: Minimum drawdown % for drawdown regime
        """
        self.trend_detector = TrendDetector(lookback_days=trend_lookback)
        self.volatility_detector = VolatilityDetector(lookback_days=vol_lookback)
        self.drawdown_detector = DrawdownDetector(drawdown_threshold=drawdown_threshold)

    def analyze(
        self,
        portfolio_returns: pd.Series,
        market_prices: pd.Series,
        trades: Optional[pd.DataFrame] = None
    ) -> RegimeAnalysisResults:
        """
        Analyze strategy performance by regime.

        Args:
            portfolio_returns: Strategy returns series
            market_prices: Market price series (for regime detection)
            trades: Optional trades dataframe for trade-level analysis

        Returns:
            RegimeAnalysisResults object
        """
        logger.info("Running regime-based analysis...")

        # Detect regimes
        trend_regimes = self.trend_detector.detect(market_prices)
        vol_regimes = self.volatility_detector.detect(market_prices)
        drawdown_regimes = self.drawdown_detector.detect(market_prices)

        logger.info(f"Detected {len(trend_regimes)} trend regime periods")
        logger.info(f"Detected {len(vol_regimes)} volatility regime periods")
        logger.info(f"Detected {len(drawdown_regimes)} drawdown regime periods")

        # Calculate performance for each regime type
        trend_perf = self._analyze_regime_type(
            portfolio_returns, trend_regimes, trades, market_prices
        )
        vol_perf = self._analyze_regime_type(
            portfolio_returns, vol_regimes, trades, market_prices
        )
        drawdown_perf = self._analyze_regime_type(
            portfolio_returns, drawdown_regimes, trades, market_prices
        )

        # Calculate robustness metrics
        robustness_score = self._calculate_robustness(
            trend_perf, vol_perf, drawdown_perf
        )

        worst_regime, best_regime = self._find_extremes(
            trend_perf, vol_perf, drawdown_perf
        )

        # Overall stats
        overall_sharpe = self._calculate_sharpe(portfolio_returns)
        overall_return = (1 + portfolio_returns).prod() - 1

        results = RegimeAnalysisResults(
            trend_performance=trend_perf,
            volatility_performance=vol_perf,
            drawdown_performance=drawdown_perf,
            robustness_score=robustness_score,
            worst_regime=worst_regime,
            best_regime=best_regime,
            overall_sharpe=overall_sharpe,
            overall_return=overall_return * 100
        )

        results.print_summary()

        return results

    def _analyze_regime_type(
        self,
        portfolio_returns: pd.Series,
        regime_periods: List[RegimePeriod],
        trades: Optional[pd.DataFrame],
        market_prices: pd.Series
    ) -> Dict[RegimeLabel, RegimePerformance]:
        """
        Analyze performance for a specific regime type.

        Args:
            portfolio_returns: Strategy returns
            regime_periods: List of regime periods
            trades: Optional trades dataframe
            market_prices: Market prices for drawdown calculation

        Returns:
            Dictionary mapping RegimeLabel to RegimePerformance
        """
        performance = {}

        for regime_period in regime_periods:
            start = pd.to_datetime(regime_period.start_date)
            end = pd.to_datetime(regime_period.end_date)
            regime = regime_period.regime

            # Handle timezone-aware indices
            if hasattr(portfolio_returns.index, 'tz') and portfolio_returns.index.tz is not None:
                start = start.tz_localize(portfolio_returns.index.tz)
                end = end.tz_localize(portfolio_returns.index.tz)

            # Filter returns for this period
            mask = (portfolio_returns.index >= start) & (portfolio_returns.index <= end)
            period_returns = portfolio_returns[mask]

            if len(period_returns) == 0:
                continue

            # Calculate metrics
            sharpe = self._calculate_sharpe(period_returns)
            total_return = (1 + period_returns).prod() - 1
            max_dd = self._calculate_max_drawdown(period_returns)

            # Trade-level metrics (if available)
            num_trades = 0
            win_rate = 0.0
            if trades is not None and len(trades) > 0:
                period_trades = trades[
                    (trades.index >= start) & (trades.index <= end)
                ]
                num_trades = len(period_trades)
                if num_trades > 0:
                    winning_trades = (period_trades['PnL'] > 0).sum()
                    win_rate = (winning_trades / num_trades) * 100

            # Aggregate by regime label
            if regime not in performance:
                performance[regime] = RegimePerformance(
                    regime=regime,
                    sharpe_ratio=sharpe,
                    total_return=total_return * 100,
                    max_drawdown=max_dd,
                    win_rate=win_rate,
                    num_trades=num_trades,
                    num_periods=1
                )
            else:
                # Average metrics across multiple periods of same regime
                existing = performance[regime]
                n = existing.num_periods
                performance[regime] = RegimePerformance(
                    regime=regime,
                    sharpe_ratio=(existing.sharpe_ratio * n + sharpe) / (n + 1),
                    total_return=(existing.total_return + total_return * 100) / (n + 1),
                    max_drawdown=(existing.max_drawdown * n + max_dd) / (n + 1),
                    win_rate=(existing.win_rate * n + win_rate) / (n + 1),
                    num_trades=existing.num_trades + num_trades,
                    num_periods=n + 1
                )

        return performance

    def _calculate_robustness(
        self,
        trend_perf: Dict[RegimeLabel, RegimePerformance],
        vol_perf: Dict[RegimeLabel, RegimePerformance],
        drawdown_perf: Dict[RegimeLabel, RegimePerformance]
    ) -> float:
        """
        Calculate robustness score (0-100).

        Higher score = more consistent performance across regimes.
        Based on standard deviation of Sharpe ratios across regimes.
        """
        all_sharpes = []

        for perf_dict in [trend_perf, vol_perf, drawdown_perf]:
            for perf in perf_dict.values():
                all_sharpes.append(perf.sharpe_ratio)

        if len(all_sharpes) < 2:
            return 50.0  # Default middle score

        # Calculate coefficient of variation (CV) of Sharpe ratios
        mean_sharpe = np.mean(all_sharpes)
        std_sharpe = np.std(all_sharpes)

        if mean_sharpe == 0:
            return 50.0

        cv = std_sharpe / abs(mean_sharpe)

        # Convert to 0-100 scale (lower CV = higher robustness)
        # CV of 0 = 100 robustness, CV of 1+ = 0 robustness
        robustness = max(0, min(100, 100 * (1 - cv)))

        return robustness

    def _find_extremes(
        self,
        trend_perf: Dict[RegimeLabel, RegimePerformance],
        vol_perf: Dict[RegimeLabel, RegimePerformance],
        drawdown_perf: Dict[RegimeLabel, RegimePerformance]
    ) -> tuple:
        """Find best and worst performing regimes."""
        all_regimes = []

        for perf_dict in [trend_perf, vol_perf, drawdown_perf]:
            for regime_label, perf in perf_dict.items():
                all_regimes.append((regime_label.value, perf.sharpe_ratio))

        if not all_regimes:
            return "Unknown", "Unknown"

        best = max(all_regimes, key=lambda x: x[1])
        worst = min(all_regimes, key=lambda x: x[1])

        return worst[0], best[0]

    def _calculate_sharpe(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio from returns series."""
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return float(sharpe)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()

        # Prevent division by zero (theoretical edge case if catastrophic loss)
        running_max = running_max.replace(0, 1e-10)

        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100

        return float(max_dd)
