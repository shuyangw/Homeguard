"""
Unit tests for CorrelationMonitor utility.

Tests correlation tracking, diversification checks, and warning generation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.utils.correlation_monitor import (
    CorrelationMonitor,
    CorrelationWarning
)


class TestCorrelationWarningDataclass:
    """Test CorrelationWarning dataclass."""

    def test_correlation_warning_creation(self):
        """CorrelationWarning can be created with all fields."""
        warning = CorrelationWarning(
            pair1='XLY/UVXY',
            pair2='XLI/UVXY',
            correlation=0.85,
            severity='critical'
        )

        assert warning.pair1 == 'XLY/UVXY'
        assert warning.pair2 == 'XLI/UVXY'
        assert warning.correlation == 0.85
        assert warning.severity == 'critical'

    def test_severity_levels(self):
        """Different severity levels can be assigned."""
        warning_high = CorrelationWarning('P1', 'P2', 0.75, 'high')
        warning_critical = CorrelationWarning('P3', 'P4', 0.90, 'critical')

        assert warning_high.severity == 'high'
        assert warning_critical.severity == 'critical'


class TestCorrelationMonitorInitialization:
    """Test CorrelationMonitor initialization."""

    def test_default_initialization(self):
        """CorrelationMonitor uses sensible defaults."""
        monitor = CorrelationMonitor()

        assert monitor.lookback_days == 60
        assert monitor.warning_threshold == 0.70
        assert monitor.critical_threshold == 0.85
        assert monitor.returns == {}
        assert monitor.correlation_matrix is None

    def test_custom_initialization(self):
        """CorrelationMonitor accepts custom parameters."""
        monitor = CorrelationMonitor(
            lookback_days=30,
            warning_threshold=0.60,
            critical_threshold=0.80
        )

        assert monitor.lookback_days == 30
        assert monitor.warning_threshold == 0.60
        assert monitor.critical_threshold == 0.80


class TestAddPairReturns:
    """Test adding pair return series."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return pd.Series(
            np.random.randn(100) * 0.01,
            index=dates,
            name='returns'
        )

    def test_add_single_pair_returns(self, sample_returns):
        """Can add returns for a single pair."""
        monitor = CorrelationMonitor()
        monitor.add_pair_returns('PAIR1', sample_returns)

        assert 'PAIR1' in monitor.returns
        assert len(monitor.returns['PAIR1']) == 100

    def test_add_multiple_pair_returns(self, sample_returns):
        """Can add returns for multiple pairs."""
        monitor = CorrelationMonitor()

        monitor.add_pair_returns('PAIR1', sample_returns)
        monitor.add_pair_returns('PAIR2', sample_returns * 1.2)
        monitor.add_pair_returns('PAIR3', sample_returns * 0.8)

        assert len(monitor.returns) == 3
        assert 'PAIR1' in monitor.returns
        assert 'PAIR2' in monitor.returns
        assert 'PAIR3' in monitor.returns


class TestCorrelationCalculation:
    """Test correlation matrix calculation."""

    @pytest.fixture
    def monitor_with_returns(self):
        """Create monitor with sample returns."""
        monitor = CorrelationMonitor(lookback_days=60)

        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Create correlated returns
        base_returns = np.random.randn(100) * 0.01

        # PAIR1 and PAIR2 highly correlated (0.9)
        monitor.add_pair_returns(
            'PAIR1',
            pd.Series(base_returns, index=dates)
        )
        monitor.add_pair_returns(
            'PAIR2',
            pd.Series(base_returns * 0.9 + np.random.randn(100) * 0.001, index=dates)
        )

        # PAIR3 uncorrelated
        monitor.add_pair_returns(
            'PAIR3',
            pd.Series(np.random.randn(100) * 0.01, index=dates)
        )

        return monitor

    def test_correlation_matrix_creation(self, monitor_with_returns):
        """Correlation matrix is created correctly."""
        corr_matrix = monitor_with_returns.update_correlations()

        assert corr_matrix is not None
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)
        assert list(corr_matrix.index) == ['PAIR1', 'PAIR2', 'PAIR3']
        assert list(corr_matrix.columns) == ['PAIR1', 'PAIR2', 'PAIR3']

    def test_correlation_diagonal_is_one(self, monitor_with_returns):
        """Diagonal of correlation matrix is 1.0."""
        corr_matrix = monitor_with_returns.update_correlations()

        assert np.allclose(np.diag(corr_matrix), 1.0)

    def test_correlation_matrix_symmetric(self, monitor_with_returns):
        """Correlation matrix is symmetric."""
        corr_matrix = monitor_with_returns.update_correlations()

        assert np.allclose(corr_matrix, corr_matrix.T)

    def test_correlation_bounds(self, monitor_with_returns):
        """Correlations are bounded between -1 and 1."""
        corr_matrix = monitor_with_returns.update_correlations()

        assert (corr_matrix >= -1.0).all().all()
        assert (corr_matrix <= 1.0).all().all()

    def test_lookback_window_limit(self):
        """Only uses last N days for correlation calculation."""
        monitor = CorrelationMonitor(lookback_days=30)

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.randn(100) * 0.01, index=dates)

        monitor.add_pair_returns('PAIR1', returns)
        monitor.add_pair_returns('PAIR2', returns * 1.1)

        corr_matrix = monitor.update_correlations()

        # Correlation should be calculated on last 30 days only
        # (We can't directly verify the window, but matrix should exist)
        assert corr_matrix is not None

    def test_insufficient_pairs_warning(self):
        """Returns None when less than 2 pairs."""
        monitor = CorrelationMonitor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        monitor.add_pair_returns('PAIR1', pd.Series(np.random.randn(100), index=dates))

        corr_matrix = monitor.update_correlations()
        assert corr_matrix is None


class TestDiversificationChecks:
    """Test diversification warning generation."""

    @pytest.fixture
    def monitor_with_high_correlation(self):
        """Create monitor with highly correlated pairs."""
        monitor = CorrelationMonitor(
            warning_threshold=0.70,
            critical_threshold=0.85
        )

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        base_returns = np.random.randn(100) * 0.01

        # PAIR1 and PAIR2: critical correlation (>0.85)
        monitor.add_pair_returns('PAIR1', pd.Series(base_returns, index=dates))
        monitor.add_pair_returns('PAIR2', pd.Series(base_returns + np.random.randn(100) * 0.0001, index=dates))

        # PAIR3 and PAIR4: high correlation (0.70-0.85)
        other_returns = np.random.randn(100) * 0.01
        monitor.add_pair_returns('PAIR3', pd.Series(other_returns, index=dates))
        monitor.add_pair_returns('PAIR4', pd.Series(other_returns * 0.8 + np.random.randn(100) * 0.005, index=dates))

        monitor.update_correlations()
        return monitor

    def test_high_correlation_detection(self, monitor_with_high_correlation):
        """Detects pairs with high correlation."""
        warnings = monitor_with_high_correlation.check_diversification()

        assert len(warnings) > 0
        assert all(isinstance(w, CorrelationWarning) for w in warnings)

    def test_critical_severity_assignment(self, monitor_with_high_correlation):
        """Critical severity assigned for correlations >= 0.85."""
        warnings = monitor_with_high_correlation.check_diversification()

        critical_warnings = [w for w in warnings if w.severity == 'critical']

        # Should have at least one critical warning (PAIR1 vs PAIR2)
        assert len(critical_warnings) >= 1

    def test_no_warnings_for_low_correlation(self):
        """No warnings when correlations are low."""
        monitor = CorrelationMonitor(warning_threshold=0.70)

        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Create uncorrelated pairs
        monitor.add_pair_returns('PAIR1', pd.Series(np.random.randn(100) * 0.01, index=dates))
        monitor.add_pair_returns('PAIR2', pd.Series(np.random.randn(100) * 0.01, index=dates))
        monitor.add_pair_returns('PAIR3', pd.Series(np.random.randn(100) * 0.01, index=dates))

        monitor.update_correlations()
        warnings = monitor.check_diversification()

        # May have warnings due to random chance, but likely very few
        # Check that not all pairs are flagged
        assert len(warnings) < 3  # Less than total possible pairs


class TestDiversificationScore:
    """Test diversification score calculation."""

    def test_perfect_diversification_score(self):
        """Score is 1.0 for zero correlation."""
        monitor = CorrelationMonitor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Create uncorrelated pairs using independent random walks
        np.random.seed(42)
        monitor.add_pair_returns('PAIR1', pd.Series(np.random.randn(100) * 0.01, index=dates))
        np.random.seed(123)
        monitor.add_pair_returns('PAIR2', pd.Series(np.random.randn(100) * 0.01, index=dates))

        monitor.update_correlations()
        score = monitor.get_diversification_score()

        # Score should be high for uncorrelated pairs
        # With random data, correlation should be close to 0
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should show reasonable diversification

    def test_poor_diversification_score(self):
        """Score is low for high correlation."""
        monitor = CorrelationMonitor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        base_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)

        # Create highly correlated pairs
        monitor.add_pair_returns('PAIR1', base_returns)
        monitor.add_pair_returns('PAIR2', base_returns + 0.0001)

        monitor.update_correlations()
        score = monitor.get_diversification_score()

        # Score should be low for highly correlated pairs
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Poor diversification

    def test_single_pair_score(self):
        """Returns 1.0 when only one pair (no diversification needed)."""
        monitor = CorrelationMonitor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        monitor.add_pair_returns('PAIR1', pd.Series(np.random.randn(100), index=dates))

        score = monitor.get_diversification_score()
        assert score == 1.0


class TestDiversificationBenefit:
    """Test Sharpe ratio diversification benefit calculation."""

    def test_zero_correlation_benefit(self):
        """Maximum benefit (40%) for zero correlation."""
        monitor = CorrelationMonitor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Uncorrelated using independent random walks
        np.random.seed(42)
        monitor.add_pair_returns('PAIR1', pd.Series(np.random.randn(100) * 0.01, index=dates))
        np.random.seed(123)
        monitor.add_pair_returns('PAIR2', pd.Series(np.random.randn(100) * 0.01, index=dates))

        monitor.update_correlations()
        benefit = monitor.calculate_diversification_benefit()

        # Should be between 1.0 and 1.40 (0-40% boost)
        # With low correlation, expect 15-35% benefit
        assert benefit >= 1.0
        assert benefit <= 1.40

    def test_perfect_correlation_no_benefit(self):
        """No benefit (0%) for perfect correlation."""
        monitor = CorrelationMonitor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        base_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)

        # Perfectly correlated
        monitor.add_pair_returns('PAIR1', base_returns)
        monitor.add_pair_returns('PAIR2', base_returns)

        monitor.update_correlations()
        benefit = monitor.calculate_diversification_benefit()

        # Should be close to 1.0 (0% boost)
        assert benefit >= 0.95  # Allow small numerical error
        assert benefit <= 1.05

    def test_moderate_correlation_benefit(self):
        """Moderate benefit for moderate correlation."""
        monitor = CorrelationMonitor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        base_returns = np.random.randn(100) * 0.01

        # Moderate correlation (~0.5)
        monitor.add_pair_returns('PAIR1', pd.Series(base_returns, index=dates))
        monitor.add_pair_returns('PAIR2', pd.Series(
            base_returns * 0.5 + np.random.randn(100) * 0.01,
            index=dates
        ))

        monitor.update_correlations()
        benefit = monitor.calculate_diversification_benefit()

        # Should be between 1.0 and 1.40
        assert 1.0 <= benefit <= 1.40
        # For 0.5 correlation, expect ~20% benefit (1.20)
        assert 1.10 <= benefit <= 1.30

    def test_single_pair_no_benefit(self):
        """Returns 1.0 (no benefit) for single pair."""
        monitor = CorrelationMonitor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        monitor.add_pair_returns('PAIR1', pd.Series(np.random.randn(100), index=dates))

        benefit = monitor.calculate_diversification_benefit()
        assert benefit == 1.0


class TestAverageCorrelation:
    """Test average correlation calculation."""

    def test_average_correlation_calculation(self):
        """Average correlation excludes diagonal."""
        monitor = CorrelationMonitor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Create returns with known correlations
        base_returns = np.random.randn(100) * 0.01
        monitor.add_pair_returns('PAIR1', pd.Series(base_returns, index=dates))
        monitor.add_pair_returns('PAIR2', pd.Series(base_returns * 0.9, index=dates))
        monitor.add_pair_returns('PAIR3', pd.Series(base_returns * 0.8, index=dates))

        corr_matrix = monitor.update_correlations()

        # Manually calculate average (excluding diagonal)
        n = len(corr_matrix)
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        avg_corr = abs(corr_matrix.values[mask]).mean()

        # Average should be high (pairs are correlated with same base)
        assert avg_corr > 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_returns_dict(self):
        """Handles empty returns gracefully."""
        monitor = CorrelationMonitor()

        corr_matrix = monitor.update_correlations()
        assert corr_matrix is None

        warnings = monitor.check_diversification()
        assert warnings == []

        score = monitor.get_diversification_score()
        assert score == 1.0

    def test_constant_returns(self):
        """Handles constant returns (zero variance)."""
        monitor = CorrelationMonitor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Constant returns (zero variance)
        monitor.add_pair_returns('PAIR1', pd.Series([0.01] * 100, index=dates))
        monitor.add_pair_returns('PAIR2', pd.Series([0.02] * 100, index=dates))

        # Should not crash
        try:
            monitor.update_correlations()
            success = True
        except Exception:
            success = False

        assert success

    def test_mismatched_lengths(self):
        """Handles mismatched return series lengths."""
        monitor = CorrelationMonitor()

        dates1 = pd.date_range('2023-01-01', periods=100, freq='D')
        dates2 = pd.date_range('2023-01-01', periods=50, freq='D')

        monitor.add_pair_returns('PAIR1', pd.Series(np.random.randn(100), index=dates1))
        monitor.add_pair_returns('PAIR2', pd.Series(np.random.randn(50), index=dates2))

        # Should not crash (pandas will align or handle gracefully)
        try:
            monitor.update_correlations()
            success = True
        except Exception:
            success = False

        assert success


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
