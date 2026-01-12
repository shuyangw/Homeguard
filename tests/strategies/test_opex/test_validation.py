"""Tests for OpEx Validation Module."""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from src.strategies.opex.validation import (
    OpExValidator,
    ValidationReport,
    PinningFrequencyReport,
    PhaseReturnsReport,
    GEXPredictiveReport,
)
from src.strategies.opex.calendar import OpExPhase
from src.strategies.opex.backtest import OpExBacktestResults
from src.data.options.options_schema import DailyGEXSummary


class TestPinningFrequencyReport:
    """Tests for PinningFrequencyReport dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        report = PinningFrequencyReport()

        assert report.total_opex_events == 0
        assert report.pinning_rate == 0.0
        assert report.is_significant is False


class TestPhaseReturnsReport:
    """Tests for PhaseReturnsReport dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        report = PhaseReturnsReport()

        assert report.normal_avg_return == 0.0
        assert report.pinning_avg_return == 0.0
        assert report.normal_count == 0


class TestGEXPredictiveReport:
    """Tests for GEXPredictiveReport dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        report = GEXPredictiveReport()

        assert report.gex_price_correlation == 0.0
        assert report.is_gex_predictive is False


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        report = ValidationReport()

        assert report.strategy_viable is False
        assert report.pinning_hypothesis_confirmed is False


class TestOpExValidator:
    """Tests for OpExValidator class."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = OpExValidator()

        assert validator.MIN_SHARPE == 0.6
        assert validator.MIN_WIN_RATE == 0.52
        assert validator.MAX_DRAWDOWN == 0.20

    def _create_price_data(self, start='2024-01-01', end='2024-03-31'):
        """Create mock price data spanning multiple OpEx cycles."""
        dates = pd.date_range(start, end, freq='B')  # Business days
        np.random.seed(42)

        # Generate random walk prices
        returns = np.random.normal(0.0005, 0.01, len(dates))
        prices = 475.0 * (1 + returns).cumprod()

        df = pd.DataFrame({
            'close': prices,
            'open': prices * 0.999,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'volume': np.random.randint(1000000, 5000000, len(dates)),
        }, index=dates)

        return df


class TestPinningFrequencyAnalysis:
    """Tests for pinning frequency analysis."""

    def _create_price_data(self):
        """Create mock price data."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        np.random.seed(42)
        prices = 475.0 + np.random.normal(0, 5, len(dates)).cumsum()

        return pd.DataFrame({
            'close': prices,
        }, index=dates)

    def test_pinning_frequency_empty_data(self):
        """Test with empty data."""
        validator = OpExValidator()

        report = validator.test_pinning_frequency(
            price_data=pd.DataFrame(),
            pin_strikes={},
        )

        assert report.total_opex_events == 0
        assert report.pinning_rate == 0.0

    def test_pinning_frequency_with_data(self):
        """Test pinning frequency calculation."""
        validator = OpExValidator()
        price_data = self._create_price_data()

        # Create pin strikes
        pin_strikes = {
            d.date(): round(price_data.loc[d, 'close'] / 5) * 5
            for d in price_data.index
        }

        report = validator.test_pinning_frequency(price_data, pin_strikes)

        # Should have calculated distances by phase
        assert report.avg_distance_normal >= 0
        assert report.avg_distance_pinning >= 0


class TestPhaseReturnsAnalysis:
    """Tests for phase returns analysis."""

    def _create_price_data(self):
        """Create mock price data."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        np.random.seed(42)
        prices = 475.0 * (1 + np.random.normal(0.0005, 0.01, len(dates))).cumprod()

        return pd.DataFrame({
            'close': prices,
        }, index=dates)

    def test_phase_returns_empty_data(self):
        """Test with empty data."""
        validator = OpExValidator()

        report = validator.test_phase_returns(pd.DataFrame())

        assert report.normal_count == 0
        assert report.pinning_count == 0

    def test_phase_returns_with_data(self):
        """Test phase returns calculation."""
        validator = OpExValidator()
        price_data = self._create_price_data()

        report = validator.test_phase_returns(price_data)

        # Should have returns grouped by phase
        assert report.normal_count > 0
        # Might have pinning days in the range
        assert report.normal_count >= report.pinning_count

    def test_phase_returns_volatility(self):
        """Test volatility calculation by phase."""
        validator = OpExValidator()
        price_data = self._create_price_data()

        report = validator.test_phase_returns(price_data)

        # Should have calculated volatility
        assert report.normal_volatility >= 0


class TestGEXPredictiveness:
    """Tests for GEX predictiveness analysis."""

    def _create_price_data(self):
        """Create mock price data."""
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='B')
        np.random.seed(42)
        prices = 475.0 * (1 + np.random.normal(0.0005, 0.01, len(dates))).cumprod()

        return pd.DataFrame({
            'close': prices,
        }, index=dates)

    def _create_gex_data(self, dates):
        """Create mock GEX data."""
        np.random.seed(42)
        return [
            DailyGEXSummary(
                symbol="SPY",
                date=d.date(),
                expiry=date(2024, 1, 19),
                underlying_price=475.0,
                total_gex=np.random.uniform(-5e6, 5e6),
                call_gex=np.random.uniform(-3e6, 3e6),
                put_gex=np.random.uniform(-2e6, 2e6),
                pin_strike=475.0,
                pin_strike_gex=1e6,
                distance_to_pin=0.01,
                total_oi=100000,
                put_call_ratio=1.0,
                gex_percentile=50.0,
            )
            for d in dates
        ]

    def test_gex_predictiveness_empty_data(self):
        """Test with empty data."""
        validator = OpExValidator()

        report = validator.test_gex_predictiveness(pd.DataFrame(), [])

        assert report.gex_price_correlation == 0.0
        assert report.is_gex_predictive is False

    def test_gex_predictiveness_with_data(self):
        """Test GEX predictiveness calculation."""
        validator = OpExValidator()
        price_data = self._create_price_data()
        gex_data = self._create_gex_data(price_data.index)

        report = validator.test_gex_predictiveness(price_data, gex_data)

        # Should have calculated correlation
        assert -1 <= report.gex_price_correlation <= 1
        assert 0 <= report.gex_correlation_pvalue <= 1

    def test_gex_high_vs_low(self):
        """Test high vs low GEX comparison."""
        validator = OpExValidator()
        price_data = self._create_price_data()
        gex_data = self._create_gex_data(price_data.index)

        report = validator.test_gex_predictiveness(price_data, gex_data)

        # Should have calculated high/low GEX returns
        # (actual values depend on random data)
        assert report.high_gex_avg_return != 0 or report.low_gex_avg_return != 0


class TestFullValidation:
    """Tests for full validation suite."""

    def _create_price_data(self):
        """Create mock price data."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        np.random.seed(42)
        prices = 475.0 * (1 + np.random.normal(0.0005, 0.01, len(dates))).cumprod()

        return pd.DataFrame({
            'close': prices,
        }, index=dates)

    def test_full_validation_empty(self):
        """Test full validation with minimal data."""
        validator = OpExValidator()

        report = validator.run_full_validation(pd.DataFrame())

        assert report.strategy_viable is False

    def test_full_validation_with_backtest_results(self):
        """Test full validation with backtest results."""
        validator = OpExValidator()
        price_data = self._create_price_data()

        # Mock backtest results that meet criteria
        backtest_results = OpExBacktestResults(
            sharpe_ratio=1.0,
            win_rate=0.60,
            max_drawdown=-5000,
            max_drawdown_pct=-0.05,
        )

        report = validator.run_full_validation(
            price_data=price_data,
            backtest_results=backtest_results,
        )

        # Should meet all criteria
        assert report.meets_sharpe_target is True  # 1.0 >= 0.6
        assert report.meets_win_rate_target is True  # 0.60 >= 0.52
        assert report.meets_drawdown_target is True  # 0.05 <= 0.20
        assert report.strategy_viable is True

    def test_full_validation_fails_sharpe(self):
        """Test validation fails on low Sharpe."""
        validator = OpExValidator()
        price_data = self._create_price_data()

        backtest_results = OpExBacktestResults(
            sharpe_ratio=0.3,  # Below 0.6 threshold
            win_rate=0.60,
            max_drawdown_pct=-0.05,
        )

        report = validator.run_full_validation(
            price_data=price_data,
            backtest_results=backtest_results,
        )

        assert report.meets_sharpe_target is False
        assert report.strategy_viable is False

    def test_full_validation_fails_win_rate(self):
        """Test validation fails on low win rate."""
        validator = OpExValidator()
        price_data = self._create_price_data()

        backtest_results = OpExBacktestResults(
            sharpe_ratio=1.0,
            win_rate=0.45,  # Below 0.52 threshold
            max_drawdown_pct=-0.05,
        )

        report = validator.run_full_validation(
            price_data=price_data,
            backtest_results=backtest_results,
        )

        assert report.meets_win_rate_target is False
        assert report.strategy_viable is False

    def test_full_validation_fails_drawdown(self):
        """Test validation fails on excessive drawdown."""
        validator = OpExValidator()
        price_data = self._create_price_data()

        backtest_results = OpExBacktestResults(
            sharpe_ratio=1.0,
            win_rate=0.60,
            max_drawdown_pct=-0.30,  # Above 0.20 threshold
        )

        report = validator.run_full_validation(
            price_data=price_data,
            backtest_results=backtest_results,
        )

        assert report.meets_drawdown_target is False
        assert report.strategy_viable is False
