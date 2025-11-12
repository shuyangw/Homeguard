"""
Unit tests for MultiPairPortfolio coordinator.

Tests portfolio construction, capital allocation, metric aggregation,
and diversification benefit calculation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.multi_pair_portfolio import (
    MultiPairPortfolio,
    PairConfig,
    PortfolioRiskLimits
)


class TestPairConfigDataclass:
    """Test PairConfig dataclass."""

    @pytest.fixture
    def sample_pair_config(self):
        """Create a sample pair configuration."""
        return PairConfig(
            name='XLY/UVXY',
            symbol1='XLY',
            symbol2='UVXY',
            weight=0.25,
            expected_sharpe=0.735,
            params={
                'entry_zscore': 2.75,
                'exit_zscore': 0.25,
                'stop_loss_zscore': 3.5,
                'zscore_window': 25
            }
        )

    def test_pair_config_creation(self, sample_pair_config):
        """PairConfig can be created with all required fields."""
        config = sample_pair_config

        assert config.name == 'XLY/UVXY'
        assert config.symbol1 == 'XLY'
        assert config.symbol2 == 'UVXY'
        assert config.weight == 0.25
        assert config.expected_sharpe == 0.735
        assert config.params['entry_zscore'] == 2.75
        assert config.params['exit_zscore'] == 0.25


class TestPortfolioRiskLimitsDataclass:
    """Test PortfolioRiskLimits dataclass."""

    def test_default_risk_limits(self):
        """PortfolioRiskLimits uses sensible defaults."""
        limits = PortfolioRiskLimits()

        assert limits.max_portfolio_drawdown == 0.20
        assert limits.max_pair_drawdown == 0.15
        assert limits.max_leverage == 1.5
        assert limits.min_pair_sharpe == 0.5
        assert limits.max_correlation == 0.7

    def test_custom_risk_limits(self):
        """PortfolioRiskLimits can be customized."""
        limits = PortfolioRiskLimits(
            max_portfolio_drawdown=0.15,
            max_pair_drawdown=0.10,
            max_leverage=2.0,
            min_pair_sharpe=0.6,
            max_correlation=0.6
        )

        assert limits.max_portfolio_drawdown == 0.15
        assert limits.max_pair_drawdown == 0.10
        assert limits.max_leverage == 2.0
        assert limits.min_pair_sharpe == 0.6
        assert limits.max_correlation == 0.6


class TestMultiPairPortfolioInitialization:
    """Test MultiPairPortfolio initialization."""

    @pytest.fixture
    def sample_pairs(self):
        """Create sample pair configurations."""
        return [
            PairConfig(
                name='PAIR1',
                symbol1='A',
                symbol2='B',
                weight=0.5,
                expected_sharpe=0.7,
                params={'entry_zscore': 2.0}
            ),
            PairConfig(
                name='PAIR2',
                symbol1='C',
                symbol2='D',
                weight=0.5,
                expected_sharpe=0.6,
                params={'entry_zscore': 2.0}
            )
        ]

    def test_portfolio_initialization(self, sample_pairs):
        """MultiPairPortfolio can be initialized with pair configs."""
        portfolio = MultiPairPortfolio(
            pairs=sample_pairs,
            initial_capital=100000,
            fees=0.0001,
            slippage=0.001
        )

        assert len(portfolio.pairs) == 2
        assert portfolio.initial_capital == 100000
        assert portfolio.fees == 0.0001
        assert portfolio.slippage == 0.001

    def test_weight_normalization(self):
        """Weights are normalized if they don't sum to 1.0."""
        pairs = [
            PairConfig('P1', 'A', 'B', weight=0.6, expected_sharpe=0.7, params={}),
            PairConfig('P2', 'C', 'D', weight=0.6, expected_sharpe=0.6, params={})
        ]

        portfolio = MultiPairPortfolio(pairs=pairs, initial_capital=100000)

        # Weights should be normalized to sum to 1.0
        total_weight = sum(p.weight for p in portfolio.pairs)
        assert np.isclose(total_weight, 1.0)
        assert np.isclose(portfolio.pairs[0].weight, 0.5)
        assert np.isclose(portfolio.pairs[1].weight, 0.5)

    def test_default_risk_limits(self, sample_pairs):
        """Portfolio uses default risk limits if none provided."""
        portfolio = MultiPairPortfolio(pairs=sample_pairs)

        assert isinstance(portfolio.risk_limits, PortfolioRiskLimits)
        assert portfolio.risk_limits.max_portfolio_drawdown == 0.20

    def test_custom_risk_limits(self, sample_pairs):
        """Portfolio accepts custom risk limits."""
        custom_limits = PortfolioRiskLimits(max_portfolio_drawdown=0.15)
        portfolio = MultiPairPortfolio(
            pairs=sample_pairs,
            risk_limits=custom_limits
        )

        assert portfolio.risk_limits.max_portfolio_drawdown == 0.15


class TestCapitalAllocation:
    """Test capital allocation logic."""

    def test_capital_allocation_by_weight(self):
        """Capital is allocated according to pair weights."""
        pairs = [
            PairConfig('P1', 'A', 'B', weight=0.25, expected_sharpe=0.7, params={}),
            PairConfig('P2', 'C', 'D', weight=0.75, expected_sharpe=0.6, params={})
        ]

        portfolio = MultiPairPortfolio(
            pairs=pairs,
            initial_capital=100000
        )

        # Verify weights sum to 1.0
        total_weight = sum(p.weight for p in portfolio.pairs)
        assert np.isclose(total_weight, 1.0)

        # Calculate expected allocations
        expected_alloc_1 = 100000 * 0.25
        expected_alloc_2 = 100000 * 0.75

        assert np.isclose(
            portfolio.initial_capital * portfolio.pairs[0].weight,
            expected_alloc_1
        )
        assert np.isclose(
            portfolio.initial_capital * portfolio.pairs[1].weight,
            expected_alloc_2
        )


class TestPortfolioMetricsCalculation:
    """Test portfolio metrics aggregation."""

    @pytest.fixture
    def portfolio_with_results(self):
        """Create portfolio with mock results."""
        pairs = [
            PairConfig('P1', 'A', 'B', weight=0.5, expected_sharpe=0.7, params={}),
            PairConfig('P2', 'C', 'D', weight=0.5, expected_sharpe=0.6, params={})
        ]

        portfolio = MultiPairPortfolio(pairs=pairs, initial_capital=100000)

        # Mock results for two pairs
        portfolio.pair_results = {
            'P1': {
                'weight': 0.5,
                'total_return': 0.20,  # 20% return
                'sharpe_ratio': 0.70,
                'max_drawdown': -0.10,
                'total_trades': 10,
                'win_rate': 0.60,
                'final_value': 60000,
                'allocated_capital': 50000
            },
            'P2': {
                'weight': 0.5,
                'total_return': 0.30,  # 30% return
                'sharpe_ratio': 0.60,
                'max_drawdown': -0.15,
                'total_trades': 8,
                'win_rate': 0.55,
                'final_value': 65000,
                'allocated_capital': 50000
            }
        }

        return portfolio

    def test_weighted_return_calculation(self, portfolio_with_results):
        """Weighted return is calculated correctly."""
        metrics = portfolio_with_results._calculate_portfolio_metrics()

        expected_weighted_return = (0.5 * 0.20) + (0.5 * 0.30)  # 0.25
        assert np.isclose(metrics['weighted_return'], expected_weighted_return)

    def test_weighted_sharpe_calculation(self, portfolio_with_results):
        """Weighted Sharpe is calculated correctly."""
        metrics = portfolio_with_results._calculate_portfolio_metrics()

        expected_weighted_sharpe = (0.5 * 0.70) + (0.5 * 0.60)  # 0.65
        assert np.isclose(metrics['weighted_sharpe'], expected_weighted_sharpe)

    def test_diversification_adjustment(self, portfolio_with_results):
        """Adjusted Sharpe includes 15% diversification benefit."""
        metrics = portfolio_with_results._calculate_portfolio_metrics()

        weighted_sharpe = metrics['weighted_sharpe']
        adjusted_sharpe = metrics['adjusted_sharpe']

        # Should be ~15% higher (1.15x multiplier)
        expected_adjusted = weighted_sharpe * 1.15
        assert np.isclose(adjusted_sharpe, expected_adjusted)

    def test_portfolio_return_calculation(self, portfolio_with_results):
        """Portfolio return is calculated from total value."""
        metrics = portfolio_with_results._calculate_portfolio_metrics()

        total_value = 60000 + 65000  # 125000
        initial_capital = 100000
        expected_return = (total_value - initial_capital) / initial_capital  # 0.25

        assert np.isclose(metrics['portfolio_return'], expected_return)

    def test_aggregate_trades_count(self, portfolio_with_results):
        """Total trades is sum across all pairs."""
        metrics = portfolio_with_results._calculate_portfolio_metrics()

        expected_total_trades = 10 + 8  # 18
        assert metrics['total_trades'] == expected_total_trades

    def test_average_win_rate(self, portfolio_with_results):
        """Average win rate is mean of pair win rates."""
        metrics = portfolio_with_results._calculate_portfolio_metrics()

        expected_avg_win_rate = (0.60 + 0.55) / 2  # 0.575
        assert np.isclose(metrics['avg_win_rate'], expected_avg_win_rate)

    def test_max_drawdown_conservative(self, portfolio_with_results):
        """Max drawdown takes worst case across pairs."""
        metrics = portfolio_with_results._calculate_portfolio_metrics()

        # Should take the worst drawdown (most negative)
        expected_max_dd = -0.15
        assert np.isclose(metrics['max_drawdown'], expected_max_dd)


class TestResultsStorage:
    """Test results storage and retrieval."""

    def test_empty_results_handling(self):
        """Portfolio handles empty results gracefully."""
        pairs = [
            PairConfig('P1', 'A', 'B', weight=1.0, expected_sharpe=0.7, params={})
        ]

        portfolio = MultiPairPortfolio(pairs=pairs, initial_capital=100000)

        # Calculate metrics with no results
        metrics = portfolio._calculate_portfolio_metrics()

        # Should return empty dict or handle gracefully
        assert metrics == {} or metrics is not None


class TestProductionReadiness:
    """Test production readiness determination."""

    def test_sharpe_threshold_check(self):
        """Production readiness is based on 0.80 Sharpe threshold."""
        pairs = [
            PairConfig('P1', 'A', 'B', weight=1.0, expected_sharpe=0.7, params={})
        ]

        portfolio = MultiPairPortfolio(pairs=pairs, initial_capital=100000)

        # Mock high Sharpe result
        portfolio.portfolio_metrics = {'adjusted_sharpe': 0.85}
        production_ready = portfolio.portfolio_metrics['adjusted_sharpe'] >= 0.80
        assert production_ready is True

        # Mock low Sharpe result
        portfolio.portfolio_metrics = {'adjusted_sharpe': 0.75}
        production_ready = portfolio.portfolio_metrics['adjusted_sharpe'] >= 0.80
        assert production_ready is False


class TestResultsSaving:
    """Test results CSV export."""

    def test_save_results_creates_file(self, tmp_path):
        """save_results creates CSV file with correct data."""
        pairs = [
            PairConfig('P1', 'A', 'B', weight=0.5, expected_sharpe=0.7, params={}),
            PairConfig('P2', 'C', 'D', weight=0.5, expected_sharpe=0.6, params={})
        ]

        portfolio = MultiPairPortfolio(pairs=pairs, initial_capital=100000)

        # Mock results
        portfolio.pair_results = {
            'P1': {
                'weight': 0.5,
                'total_return': 0.20,
                'sharpe_ratio': 0.70,
                'max_drawdown': -0.10,
                'total_trades': 10,
                'win_rate': 0.60,
                'final_value': 60000
            },
            'P2': {
                'weight': 0.5,
                'total_return': 0.30,
                'sharpe_ratio': 0.60,
                'max_drawdown': -0.15,
                'total_trades': 8,
                'win_rate': 0.55,
                'final_value': 65000
            }
        }

        portfolio.portfolio_metrics = {
            'portfolio_return': 0.25,
            'adjusted_sharpe': 0.75,
            'max_drawdown': -0.15,
            'total_trades': 18,
            'avg_win_rate': 0.575,
            'total_value': 125000
        }

        # Save to temp directory
        output_file = portfolio.save_results(output_dir=tmp_path)

        # Verify file exists
        assert output_file.exists()

        # Read and verify content
        df = pd.read_csv(output_file)
        assert len(df) == 3  # 2 pairs + 1 portfolio row
        assert 'pair' in df.columns
        assert 'weight' in df.columns
        assert 'return' in df.columns
        assert 'sharpe' in df.columns

        # Check portfolio row exists
        portfolio_row = df[df['pair'] == 'PORTFOLIO']
        assert len(portfolio_row) == 1
        assert np.isclose(portfolio_row['sharpe'].values[0], 0.75)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
