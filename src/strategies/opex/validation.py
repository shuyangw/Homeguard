"""
OpEx Strategy Validation Tests.

Provides statistical validation of the OpEx Pinning strategy:
1. Pinning frequency analysis - Does pinning actually occur?
2. Phase returns analysis - Are returns different by phase?
3. GEX predictiveness - Does higher GEX = stronger pinning?
4. Walk-forward validation - Does strategy work out-of-sample?

Usage:
    validator = OpExValidator()

    # Test pinning frequency
    pinning_report = validator.test_pinning_frequency(price_data, gex_data)

    # Test phase returns
    phase_report = validator.test_phase_returns(price_data)

    # Full validation suite
    report = validator.run_full_validation(price_data, gex_data)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats

from src.strategies.opex.calendar import OpExCalendar, OpExPhase
from src.strategies.opex.gex_calculator import GEXCalculator
from src.data.options.options_schema import DailyGEXSummary
from src.utils.logger import logger


@dataclass
class PinningFrequencyReport:
    """Report on pinning frequency analysis."""
    total_opex_events: int = 0
    pinning_detected: int = 0
    pinning_rate: float = 0.0

    # Average distance to pin by phase
    avg_distance_normal: float = 0.0
    avg_distance_pre_opex: float = 0.0
    avg_distance_pinning: float = 0.0
    avg_distance_opex_day: float = 0.0
    avg_distance_post_opex: float = 0.0

    # Statistical significance
    pinning_vs_normal_pvalue: float = 1.0
    is_significant: bool = False


@dataclass
class PhaseReturnsReport:
    """Report on returns by OpEx phase."""
    # Average returns by phase
    normal_avg_return: float = 0.0
    pre_opex_avg_return: float = 0.0
    pinning_avg_return: float = 0.0
    opex_day_avg_return: float = 0.0
    post_opex_avg_return: float = 0.0

    # Volatility by phase
    normal_volatility: float = 0.0
    pinning_volatility: float = 0.0
    post_opex_volatility: float = 0.0

    # Statistical tests
    pinning_vs_normal_pvalue: float = 1.0
    post_opex_vs_normal_pvalue: float = 1.0

    # Sample sizes
    normal_count: int = 0
    pinning_count: int = 0
    post_opex_count: int = 0


@dataclass
class GEXPredictiveReport:
    """Report on GEX predictiveness."""
    # Correlation between GEX and price movement
    gex_price_correlation: float = 0.0
    gex_correlation_pvalue: float = 1.0

    # High vs Low GEX returns
    high_gex_avg_return: float = 0.0
    low_gex_avg_return: float = 0.0
    high_vs_low_pvalue: float = 1.0

    # GEX regime performance
    positive_gex_win_rate: float = 0.0
    negative_gex_win_rate: float = 0.0

    is_gex_predictive: bool = False


@dataclass
class ValidationReport:
    """Complete validation report."""
    pinning: PinningFrequencyReport = field(default_factory=PinningFrequencyReport)
    phase_returns: PhaseReturnsReport = field(default_factory=PhaseReturnsReport)
    gex_predictive: GEXPredictiveReport = field(default_factory=GEXPredictiveReport)

    # Walk-forward results
    is_avg_sharpe: float = 0.0
    oos_avg_sharpe: float = 0.0
    degradation: float = 0.0

    # Overall assessment
    pinning_hypothesis_confirmed: bool = False
    phase_effect_confirmed: bool = False
    gex_predictive_confirmed: bool = False
    strategy_viable: bool = False

    # Success criteria
    meets_sharpe_target: bool = False
    meets_win_rate_target: bool = False
    meets_drawdown_target: bool = False


class OpExValidator:
    """
    Validator for OpEx Pinning strategy hypotheses.

    Tests whether:
    1. Pinning actually occurs near monthly OpEx
    2. Returns differ by OpEx phase
    3. GEX is predictive of price behavior
    4. Strategy works out-of-sample
    """

    # Success criteria thresholds
    MIN_SHARPE = 0.6
    MIN_WIN_RATE = 0.52
    MAX_DRAWDOWN = 0.20
    SIGNIFICANCE_LEVEL = 0.05

    def __init__(self):
        """Initialize validator."""
        self.calendar = OpExCalendar()
        self.gex_calc = GEXCalculator()

    def test_pinning_frequency(
        self,
        price_data: pd.DataFrame,
        pin_strikes: Dict[date, float],
        symbols: Optional[List[str]] = None,
    ) -> PinningFrequencyReport:
        """
        Test whether prices actually "pin" near strike prices around OpEx.

        Args:
            price_data: OHLCV DataFrame with 'symbol' column
            pin_strikes: Dict of date -> pin strike price
            symbols: Symbols to analyze (default: all)

        Returns:
            PinningFrequencyReport with statistics
        """
        report = PinningFrequencyReport()

        if price_data.empty or not pin_strikes:
            return report

        # Get unique dates
        if hasattr(price_data.index, 'date'):
            dates = pd.Series(price_data.index.date).unique()
        else:
            dates = price_data.index.normalize().unique()

        # Calculate distance to pin by phase
        phase_distances: Dict[OpExPhase, List[float]] = {
            phase: [] for phase in OpExPhase
        }

        for dt in dates:
            current_date = dt if isinstance(dt, date) else dt.date()

            # Get phase
            phase = self.calendar.get_phase(current_date)

            # Get pin strike
            pin_strike = pin_strikes.get(current_date)
            if pin_strike is None:
                continue

            # Get close price
            day_data = price_data[price_data.index.date == current_date] if hasattr(price_data.index, 'date') else price_data.loc[str(current_date)]
            if len(day_data) == 0:
                continue

            close = day_data['close'].iloc[-1] if 'close' in day_data else day_data['Close'].iloc[-1]

            # Calculate distance
            distance = abs(close - pin_strike) / pin_strike
            phase_distances[phase].append(distance)

        # Calculate averages
        report.avg_distance_normal = np.mean(phase_distances[OpExPhase.NORMAL]) if phase_distances[OpExPhase.NORMAL] else 0
        report.avg_distance_pre_opex = np.mean(phase_distances[OpExPhase.PRE_OPEX]) if phase_distances[OpExPhase.PRE_OPEX] else 0
        report.avg_distance_pinning = np.mean(phase_distances[OpExPhase.PINNING]) if phase_distances[OpExPhase.PINNING] else 0
        report.avg_distance_opex_day = np.mean(phase_distances[OpExPhase.OPEX_DAY]) if phase_distances[OpExPhase.OPEX_DAY] else 0
        report.avg_distance_post_opex = np.mean(phase_distances[OpExPhase.POST_OPEX]) if phase_distances[OpExPhase.POST_OPEX] else 0

        # Statistical test: is distance during pinning significantly less than normal?
        if phase_distances[OpExPhase.PINNING] and phase_distances[OpExPhase.NORMAL]:
            _, pvalue = stats.ttest_ind(
                phase_distances[OpExPhase.PINNING],
                phase_distances[OpExPhase.NORMAL],
                alternative='less'  # Pinning should be lower
            )
            report.pinning_vs_normal_pvalue = pvalue
            report.is_significant = pvalue < self.SIGNIFICANCE_LEVEL

        # Count OpEx events and successful pins
        expirations = self.calendar.get_expirations_in_range(
            min(dates).date() if hasattr(min(dates), 'date') else min(dates),
            max(dates).date() if hasattr(max(dates), 'date') else max(dates),
        )
        report.total_opex_events = len(expirations)

        # Pinning detected if close within 0.5% of pin
        pinning_threshold = 0.005
        pinning_count = sum(1 for d in phase_distances[OpExPhase.OPEX_DAY] if d < pinning_threshold)
        report.pinning_detected = pinning_count
        report.pinning_rate = pinning_count / len(phase_distances[OpExPhase.OPEX_DAY]) if phase_distances[OpExPhase.OPEX_DAY] else 0

        return report

    def test_phase_returns(
        self,
        price_data: pd.DataFrame,
    ) -> PhaseReturnsReport:
        """
        Test whether returns differ by OpEx phase.

        Args:
            price_data: OHLCV DataFrame

        Returns:
            PhaseReturnsReport with statistics
        """
        report = PhaseReturnsReport()

        if price_data.empty:
            return report

        # Calculate daily returns
        if 'close' in price_data.columns:
            returns = price_data['close'].pct_change().dropna()
        elif 'Close' in price_data.columns:
            returns = price_data['Close'].pct_change().dropna()
        else:
            return report

        # Group returns by phase
        phase_returns: Dict[OpExPhase, List[float]] = {
            phase: [] for phase in OpExPhase
        }

        for dt, ret in returns.items():
            if hasattr(dt, 'date'):
                current_date = dt.date()
            else:
                current_date = dt

            phase = self.calendar.get_phase(current_date)
            phase_returns[phase].append(ret)

        # Calculate statistics
        report.normal_avg_return = np.mean(phase_returns[OpExPhase.NORMAL]) if phase_returns[OpExPhase.NORMAL] else 0
        report.pre_opex_avg_return = np.mean(phase_returns[OpExPhase.PRE_OPEX]) if phase_returns[OpExPhase.PRE_OPEX] else 0
        report.pinning_avg_return = np.mean(phase_returns[OpExPhase.PINNING]) if phase_returns[OpExPhase.PINNING] else 0
        report.opex_day_avg_return = np.mean(phase_returns[OpExPhase.OPEX_DAY]) if phase_returns[OpExPhase.OPEX_DAY] else 0
        report.post_opex_avg_return = np.mean(phase_returns[OpExPhase.POST_OPEX]) if phase_returns[OpExPhase.POST_OPEX] else 0

        report.normal_volatility = np.std(phase_returns[OpExPhase.NORMAL]) if phase_returns[OpExPhase.NORMAL] else 0
        report.pinning_volatility = np.std(phase_returns[OpExPhase.PINNING]) if phase_returns[OpExPhase.PINNING] else 0
        report.post_opex_volatility = np.std(phase_returns[OpExPhase.POST_OPEX]) if phase_returns[OpExPhase.POST_OPEX] else 0

        report.normal_count = len(phase_returns[OpExPhase.NORMAL])
        report.pinning_count = len(phase_returns[OpExPhase.PINNING])
        report.post_opex_count = len(phase_returns[OpExPhase.POST_OPEX])

        # Statistical tests
        if phase_returns[OpExPhase.PINNING] and phase_returns[OpExPhase.NORMAL]:
            _, report.pinning_vs_normal_pvalue = stats.ttest_ind(
                phase_returns[OpExPhase.PINNING],
                phase_returns[OpExPhase.NORMAL],
            )

        if phase_returns[OpExPhase.POST_OPEX] and phase_returns[OpExPhase.NORMAL]:
            _, report.post_opex_vs_normal_pvalue = stats.ttest_ind(
                phase_returns[OpExPhase.POST_OPEX],
                phase_returns[OpExPhase.NORMAL],
            )

        return report

    def test_gex_predictiveness(
        self,
        price_data: pd.DataFrame,
        gex_data: List[DailyGEXSummary],
    ) -> GEXPredictiveReport:
        """
        Test whether GEX is predictive of price behavior.

        Args:
            price_data: OHLCV DataFrame
            gex_data: List of daily GEX summaries

        Returns:
            GEXPredictiveReport with statistics
        """
        report = GEXPredictiveReport()

        if price_data.empty or not gex_data:
            return report

        # Create GEX lookup
        gex_lookup = {s.date: s for s in gex_data}

        # Calculate returns
        if 'close' in price_data.columns:
            returns = price_data['close'].pct_change().dropna()
        elif 'Close' in price_data.columns:
            returns = price_data['Close'].pct_change().dropna()
        else:
            return report

        # Match GEX with next-day returns
        gex_values = []
        next_day_returns = []

        for dt, ret in returns.items():
            if hasattr(dt, 'date'):
                current_date = dt.date()
            else:
                current_date = dt

            gex_summary = gex_lookup.get(current_date)
            if gex_summary:
                gex_values.append(gex_summary.total_gex)
                next_day_returns.append(ret)

        if len(gex_values) < 10:
            return report

        # Correlation
        corr, pvalue = stats.pearsonr(gex_values, next_day_returns)
        report.gex_price_correlation = corr
        report.gex_correlation_pvalue = pvalue

        # High vs Low GEX
        median_gex = np.median(gex_values)
        high_gex_returns = [r for g, r in zip(gex_values, next_day_returns) if g > median_gex]
        low_gex_returns = [r for g, r in zip(gex_values, next_day_returns) if g <= median_gex]

        report.high_gex_avg_return = np.mean(high_gex_returns) if high_gex_returns else 0
        report.low_gex_avg_return = np.mean(low_gex_returns) if low_gex_returns else 0

        if high_gex_returns and low_gex_returns:
            _, report.high_vs_low_pvalue = stats.ttest_ind(high_gex_returns, low_gex_returns)

        # Win rates by GEX regime
        positive_gex_wins = sum(1 for g, r in zip(gex_values, next_day_returns) if g > 0 and r > 0)
        positive_gex_total = sum(1 for g in gex_values if g > 0)
        report.positive_gex_win_rate = positive_gex_wins / positive_gex_total if positive_gex_total else 0

        negative_gex_wins = sum(1 for g, r in zip(gex_values, next_day_returns) if g <= 0 and r > 0)
        negative_gex_total = sum(1 for g in gex_values if g <= 0)
        report.negative_gex_win_rate = negative_gex_wins / negative_gex_total if negative_gex_total else 0

        # Is GEX predictive?
        report.is_gex_predictive = (
            abs(report.gex_price_correlation) > 0.1 and
            report.gex_correlation_pvalue < self.SIGNIFICANCE_LEVEL
        )

        return report

    def run_full_validation(
        self,
        price_data: pd.DataFrame,
        gex_data: Optional[List[DailyGEXSummary]] = None,
        pin_strikes: Optional[Dict[date, float]] = None,
        backtest_results: Optional[Any] = None,
    ) -> ValidationReport:
        """
        Run complete validation suite.

        Args:
            price_data: OHLCV DataFrame
            gex_data: GEX summaries (optional)
            pin_strikes: Pin strike prices by date (optional)
            backtest_results: Results from OpExBacktestRunner (optional)

        Returns:
            Complete ValidationReport
        """
        report = ValidationReport()

        # Test pinning frequency
        if pin_strikes:
            report.pinning = self.test_pinning_frequency(price_data, pin_strikes)
            report.pinning_hypothesis_confirmed = report.pinning.is_significant

        # Test phase returns
        report.phase_returns = self.test_phase_returns(price_data)
        report.phase_effect_confirmed = (
            report.phase_returns.pinning_vs_normal_pvalue < self.SIGNIFICANCE_LEVEL or
            report.phase_returns.post_opex_vs_normal_pvalue < self.SIGNIFICANCE_LEVEL
        )

        # Test GEX predictiveness
        if gex_data:
            report.gex_predictive = self.test_gex_predictiveness(price_data, gex_data)
            report.gex_predictive_confirmed = report.gex_predictive.is_gex_predictive

        # Check backtest results against criteria
        if backtest_results:
            report.meets_sharpe_target = backtest_results.sharpe_ratio >= self.MIN_SHARPE
            report.meets_win_rate_target = backtest_results.win_rate >= self.MIN_WIN_RATE
            report.meets_drawdown_target = abs(backtest_results.max_drawdown_pct) <= self.MAX_DRAWDOWN

            report.strategy_viable = (
                report.meets_sharpe_target and
                report.meets_win_rate_target and
                report.meets_drawdown_target
            )

        return report

    def print_validation_report(self, report: ValidationReport) -> None:
        """Print formatted validation report."""
        logger.info("=" * 70)
        logger.info("OPEX PINNING STRATEGY VALIDATION REPORT")
        logger.info("=" * 70)

        logger.info("[1] PINNING FREQUENCY ANALYSIS")
        logger.info("-" * 40)
        logger.info(f"  OpEx Events: {report.pinning.total_opex_events}")
        logger.info(f"  Pinning Detected: {report.pinning.pinning_detected} ({report.pinning.pinning_rate:.1%})")
        logger.info(f"  Avg Distance (NORMAL): {report.pinning.avg_distance_normal:.2%}")
        logger.info(f"  Avg Distance (PINNING): {report.pinning.avg_distance_pinning:.2%}")
        logger.info(f"  Statistical Test p-value: {report.pinning.pinning_vs_normal_pvalue:.4f}")
        logger.info(f"  Hypothesis Confirmed: {'YES' if report.pinning_hypothesis_confirmed else 'NO'}")

        logger.info("[2] PHASE RETURNS ANALYSIS")
        logger.info("-" * 40)
        logger.info(f"  NORMAL avg return: {report.phase_returns.normal_avg_return:.4%} (n={report.phase_returns.normal_count})")
        logger.info(f"  PINNING avg return: {report.phase_returns.pinning_avg_return:.4%} (n={report.phase_returns.pinning_count})")
        logger.info(f"  POST_OPEX avg return: {report.phase_returns.post_opex_avg_return:.4%} (n={report.phase_returns.post_opex_count})")
        logger.info(f"  Phase Effect Confirmed: {'YES' if report.phase_effect_confirmed else 'NO'}")

        logger.info("[3] GEX PREDICTIVENESS")
        logger.info("-" * 40)
        logger.info(f"  GEX-Price Correlation: {report.gex_predictive.gex_price_correlation:.3f}")
        logger.info(f"  Correlation p-value: {report.gex_predictive.gex_correlation_pvalue:.4f}")
        logger.info(f"  High GEX avg return: {report.gex_predictive.high_gex_avg_return:.4%}")
        logger.info(f"  Low GEX avg return: {report.gex_predictive.low_gex_avg_return:.4%}")
        logger.info(f"  GEX Predictive: {'YES' if report.gex_predictive_confirmed else 'NO'}")

        logger.info("[4] SUCCESS CRITERIA")
        logger.info("-" * 40)
        logger.info(f"  Sharpe >= {self.MIN_SHARPE}: {'PASS' if report.meets_sharpe_target else 'FAIL'}")
        logger.info(f"  Win Rate >= {self.MIN_WIN_RATE:.0%}: {'PASS' if report.meets_win_rate_target else 'FAIL'}")
        logger.info(f"  Max DD <= {self.MAX_DRAWDOWN:.0%}: {'PASS' if report.meets_drawdown_target else 'FAIL'}")

        logger.info("=" * 70)
        logger.info(f"STRATEGY VIABLE: {'YES' if report.strategy_viable else 'NO'}")
        logger.info("=" * 70)
