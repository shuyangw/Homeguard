"""
Tests for regime analysis module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

import pytest
import pandas as pd
import numpy as np

from backtesting.regimes.analyzer import (
    RegimePerformance,
    RegimeAnalysisResults,
    RegimeAnalyzer
)
from backtesting.regimes.detector import RegimeLabel


@pytest.fixture
def sample_returns():
    """Create sample portfolio returns."""
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    return returns


@pytest.fixture
def sample_prices():
    """Create sample market prices."""
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(252) * 2), index=dates)
    return prices


@pytest.fixture
def trending_prices():
    """Create trending market prices for clearer regime detection."""
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    # Strong bull market
    prices = pd.Series(100 + np.linspace(0, 50, 252) + np.random.randn(252) * 1, index=dates)
    return prices


class TestRegimePerformance:
    """Tests for RegimePerformance dataclass."""

    def test_regime_performance_creation(self):
        """Test creating regime performance object."""
        perf = RegimePerformance(
            regime=RegimeLabel.BULL,
            sharpe_ratio=1.5,
            total_return=25.0,
            max_drawdown=-10.0,
            win_rate=60.0,
            num_trades=50,
            num_periods=3
        )

        assert perf.regime == RegimeLabel.BULL
        assert perf.sharpe_ratio == 1.5
        assert perf.total_return == 25.0
        assert perf.max_drawdown == -10.0
        assert perf.win_rate == 60.0
        assert perf.num_trades == 50
        assert perf.num_periods == 3


class TestRegimeAnalysisResults:
    """Tests for RegimeAnalysisResults dataclass."""

    def test_results_creation(self):
        """Test creating analysis results."""
        trend_perf = {
            RegimeLabel.BULL: RegimePerformance(
                regime=RegimeLabel.BULL,
                sharpe_ratio=1.5,
                total_return=20.0,
                max_drawdown=-5.0,
                win_rate=65.0,
                num_trades=30,
                num_periods=2
            )
        }

        results = RegimeAnalysisResults(
            trend_performance=trend_perf,
            volatility_performance={},
            drawdown_performance={},
            robustness_score=75.0,
            worst_regime="Bear Market",
            best_regime="Bull Market",
            overall_sharpe=1.2,
            overall_return=18.0
        )

        assert results.robustness_score == 75.0
        assert results.worst_regime == "Bear Market"
        assert results.best_regime == "Bull Market"
        assert results.overall_sharpe == 1.2
        assert results.overall_return == 18.0

    def test_print_summary(self):
        """Test printing summary."""
        results = RegimeAnalysisResults(
            trend_performance={},
            volatility_performance={},
            drawdown_performance={},
            robustness_score=75.0,
            worst_regime="Bear Market",
            best_regime="Bull Market",
            overall_sharpe=1.2,
            overall_return=18.0
        )

        # Should not raise error (logger outputs to custom logger, not stdout)
        results.print_summary()


class TestRegimeAnalyzer:
    """Tests for RegimeAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = RegimeAnalyzer(
            trend_lookback=60,
            vol_lookback=20,
            drawdown_threshold=10.0
        )

        assert analyzer.trend_detector.lookback_days == 60
        assert analyzer.volatility_detector.lookback_days == 20
        assert analyzer.drawdown_detector.drawdown_threshold == 10.0

    def test_analyze(self, sample_returns, trending_prices):
        """Test analyze method."""
        analyzer = RegimeAnalyzer()

        results = analyzer.analyze(
            portfolio_returns=sample_returns,
            market_prices=trending_prices,
            trades=None
        )

        # Check results structure
        assert isinstance(results, RegimeAnalysisResults)
        assert isinstance(results.trend_performance, dict)
        assert isinstance(results.volatility_performance, dict)
        assert isinstance(results.drawdown_performance, dict)
        assert isinstance(results.robustness_score, float)
        assert isinstance(results.worst_regime, str)
        assert isinstance(results.best_regime, str)
        assert isinstance(results.overall_sharpe, float)
        assert isinstance(results.overall_return, float)

        # Robustness score should be between 0 and 100
        assert 0 <= results.robustness_score <= 100

    def test_calculate_sharpe(self, sample_returns):
        """Test Sharpe ratio calculation."""
        analyzer = RegimeAnalyzer()
        sharpe = analyzer._calculate_sharpe(sample_returns)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_calculate_sharpe_zero_std(self):
        """Test Sharpe with zero standard deviation."""
        analyzer = RegimeAnalyzer()
        returns = pd.Series([0.0] * 100, index=pd.date_range('2020-01-01', periods=100))
        sharpe = analyzer._calculate_sharpe(returns)
        assert sharpe == 0.0

    def test_calculate_sharpe_empty(self):
        """Test Sharpe with empty series."""
        analyzer = RegimeAnalyzer()
        returns = pd.Series([])
        sharpe = analyzer._calculate_sharpe(returns)
        assert sharpe == 0.0

    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        analyzer = RegimeAnalyzer()

        # Create returns with known drawdown
        dates = pd.date_range('2020-01-01', periods=10)
        returns = pd.Series([0.1, 0.1, -0.2, -0.1, 0.0, 0.1, 0.1, 0.0, -0.05, 0.05], index=dates)

        max_dd = analyzer._calculate_max_drawdown(returns)

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero

    def test_calculate_max_drawdown_empty(self):
        """Test max drawdown with empty series."""
        analyzer = RegimeAnalyzer()
        returns = pd.Series([])
        max_dd = analyzer._calculate_max_drawdown(returns)
        assert max_dd == 0.0

    def test_calculate_robustness(self):
        """Test robustness score calculation."""
        analyzer = RegimeAnalyzer()

        # Create consistent performance across regimes
        consistent_perf = {
            RegimeLabel.BULL: RegimePerformance(
                regime=RegimeLabel.BULL,
                sharpe_ratio=1.5,
                total_return=20.0,
                max_drawdown=-5.0,
                win_rate=60.0,
                num_trades=30,
                num_periods=1
            ),
            RegimeLabel.BEAR: RegimePerformance(
                regime=RegimeLabel.BEAR,
                sharpe_ratio=1.4,
                total_return=18.0,
                max_drawdown=-6.0,
                win_rate=58.0,
                num_trades=25,
                num_periods=1
            )
        }

        robustness = analyzer._calculate_robustness(
            consistent_perf, {}, {}
        )

        assert isinstance(robustness, float)
        assert 0 <= robustness <= 100

        # Create inconsistent performance
        inconsistent_perf = {
            RegimeLabel.BULL: RegimePerformance(
                regime=RegimeLabel.BULL,
                sharpe_ratio=2.5,
                total_return=40.0,
                max_drawdown=-3.0,
                win_rate=70.0,
                num_trades=40,
                num_periods=1
            ),
            RegimeLabel.BEAR: RegimePerformance(
                regime=RegimeLabel.BEAR,
                sharpe_ratio=-0.5,
                total_return=-10.0,
                max_drawdown=-25.0,
                win_rate=35.0,
                num_trades=20,
                num_periods=1
            )
        }

        inconsistent_robustness = analyzer._calculate_robustness(
            inconsistent_perf, {}, {}
        )

        # Inconsistent performance should have lower robustness
        assert inconsistent_robustness < robustness

    def test_find_extremes(self):
        """Test finding best and worst regimes."""
        analyzer = RegimeAnalyzer()

        trend_perf = {
            RegimeLabel.BULL: RegimePerformance(
                regime=RegimeLabel.BULL,
                sharpe_ratio=2.0,
                total_return=30.0,
                max_drawdown=-3.0,
                win_rate=65.0,
                num_trades=40,
                num_periods=1
            ),
            RegimeLabel.BEAR: RegimePerformance(
                regime=RegimeLabel.BEAR,
                sharpe_ratio=0.5,
                total_return=5.0,
                max_drawdown=-15.0,
                win_rate=45.0,
                num_trades=20,
                num_periods=1
            )
        }

        worst, best = analyzer._find_extremes(trend_perf, {}, {})

        assert worst == "Bear Market"
        assert best == "Bull Market"

    def test_find_extremes_empty(self):
        """Test finding extremes with no data."""
        analyzer = RegimeAnalyzer()
        worst, best = analyzer._find_extremes({}, {}, {})

        assert worst == "Unknown"
        assert best == "Unknown"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
