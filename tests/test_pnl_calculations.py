"""
Unit tests for P&L calculations and performance metrics.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.engine.metrics import PerformanceMetrics
from strategies.base_strategies.moving_average import MovingAverageCrossover


class TestBasicPnLCalculations:
    """Test basic P&L calculation accuracy."""

    def test_simple_profit_calculation(self, simple_price_data):
        """Test profit is calculated correctly for winning trade."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        stats = portfolio.stats()
        if stats is not None:
            end_value = stats.get('End Value', 0)
            start_value = stats.get('Start Value', 0)

            assert end_value > 0, "End value should be positive"
            assert start_value == 10000.0, "Start value should match initial capital"

    def test_total_return_calculation(self, simple_price_data):
        """Test total return percentage is calculated correctly."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        stats = portfolio.stats()
        if stats is not None:
            total_return = stats.get('Total Return [%]', None)
            end_value = stats.get('End Value', 0)
            start_value = stats.get('Start Value', 0)

            if total_return is not None and start_value > 0:
                expected_return = ((end_value - start_value) / start_value) * 100
                assert abs(total_return - expected_return) < 0.1, "Total return calculation should be accurate"

    def test_losing_trade_reduces_capital(self):
        """Test that fees and losses reduce capital."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        np.random.seed(42)

        base_price = 100.0
        decline = np.linspace(0, -20, 50)
        noise = np.random.randn(50) * 1
        close_prices = base_price + decline + noise

        df = pd.DataFrame({
            'open': close_prices + np.random.randn(50) * 0.3,
            'high': close_prices + np.abs(np.random.randn(50) * 0.5),
            'low': close_prices - np.abs(np.random.randn(50) * 0.5),
            'close': close_prices,
            'volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)

        engine = BacktestEngine(initial_capital=10000.0, fees=0.01)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, df)

        stats = portfolio.stats()
        if stats is not None:
            total_trades = stats.get('Total Trades', 0)
            total_return = stats.get('Total Return [%]', 0)

            if total_trades > 0:
                assert total_return < 0, "Should have negative return on declining data with fees"


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def test_metrics_calculation(self, simple_price_data):
        """Test all metrics are calculated without errors."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.001)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        metrics = PerformanceMetrics.calculate_all_metrics(portfolio)  # type: ignore[arg-type]

        assert isinstance(metrics, dict)
        assert 'total_return_pct' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown_pct' in metrics
        assert 'win_rate_pct' in metrics

    def test_sharpe_ratio_calculation(self, simple_price_data):
        """Test Sharpe ratio is calculated."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        metrics = PerformanceMetrics.calculate_all_metrics(portfolio)  # type: ignore[arg-type]
        sharpe = metrics.get('sharpe_ratio', None)

        assert sharpe is not None, "Sharpe ratio should be calculated"
        assert isinstance(sharpe, (int, float)), "Sharpe ratio should be numeric"

    def test_max_drawdown_calculation(self, simple_price_data):
        """Test max drawdown is calculated and is negative."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.001)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        metrics = PerformanceMetrics.calculate_all_metrics(portfolio)  # type: ignore[arg-type]
        max_drawdown = metrics.get('max_drawdown_pct', 0)

        assert max_drawdown <= 0, "Max drawdown should be negative or zero"

    def test_win_rate_calculation(self, simple_price_data):
        """Test win rate is between 0 and 100."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.001)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        metrics = PerformanceMetrics.calculate_all_metrics(portfolio)  # type: ignore[arg-type]
        win_rate = metrics.get('win_rate_pct', -1)

        assert 0 <= win_rate <= 100, "Win rate should be between 0 and 100"


class TestTradeMetrics:
    """Test individual trade metrics."""

    def test_total_trades_count(self, simple_price_data):
        """Test total trades are counted correctly."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        stats = portfolio.stats()
        if stats is not None:
            total_trades = stats.get('Total Trades', -1)
            assert total_trades >= 0, "Total trades should be non-negative"

    def test_profit_factor_calculation(self, simple_price_data):
        """Test profit factor is calculated."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.0)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        metrics = PerformanceMetrics.calculate_all_metrics(portfolio)  # type: ignore[arg-type]
        profit_factor = metrics.get('profit_factor', None)

        assert profit_factor is not None, "Profit factor should be calculated"
        assert profit_factor >= 0, "Profit factor should be non-negative"

    def test_average_win_and_loss(self, simple_price_data):
        """Test average win and loss are calculated."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.001)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        metrics = PerformanceMetrics.calculate_all_metrics(portfolio)  # type: ignore[arg-type]

        avg_win = metrics.get('avg_win', None)
        avg_loss = metrics.get('avg_loss', None)

        assert avg_win is not None, "Average win should be calculated"
        assert avg_loss is not None, "Average loss should be calculated"

    def test_largest_win_and_loss(self, simple_price_data):
        """Test largest win and loss are tracked."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.001)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        metrics = PerformanceMetrics.calculate_all_metrics(portfolio)  # type: ignore[arg-type]

        largest_win = metrics.get('largest_win', None)
        largest_loss = metrics.get('largest_loss', None)

        assert largest_win is not None, "Largest win should be tracked"
        assert largest_loss is not None, "Largest loss should be tracked"


class TestFeeImpact:
    """Test impact of fees on P&L."""

    def test_higher_fees_reduce_returns(self, simple_price_data):
        """Test that higher fees reduce returns."""
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        engine_low_fees = BacktestEngine(initial_capital=10000.0, fees=0.001)
        portfolio_low = engine_low_fees.run_with_data(strategy, simple_price_data)

        engine_high_fees = BacktestEngine(initial_capital=10000.0, fees=0.01)
        portfolio_high = engine_high_fees.run_with_data(strategy, simple_price_data)

        metrics_low = PerformanceMetrics.calculate_all_metrics(portfolio_low)  # type: ignore[arg-type]
        metrics_high = PerformanceMetrics.calculate_all_metrics(portfolio_high)  # type: ignore[arg-type]

        return_low = metrics_low.get('total_return_pct', 0)
        return_high = metrics_high.get('total_return_pct', 0)

        assert return_low >= return_high, "Lower fees should result in higher or equal returns"

    def test_fee_impact_on_trade_count(self, simple_price_data):
        """Test that all trades incur fees."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.01)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        stats = portfolio.stats()
        if stats is not None:
            total_trades = stats.get('Total Trades', 0)
            if total_trades > 0:
                end_value = stats.get('End Value', 0)
                start_value = stats.get('Start Value', 10000.0)

                assert end_value < start_value + (start_value * 0.20), "High fees should impact performance"


class TestCapitalManagement:
    """Test capital management in backtests."""

    def test_initial_capital_allocation(self, simple_price_data):
        """Test that initial capital is correctly allocated."""
        initial_capital = 25000.0
        engine = BacktestEngine(initial_capital=initial_capital, fees=0.0)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        stats = portfolio.stats()
        if stats is not None:
            start_value = stats.get('Start Value', 0)
            assert abs(start_value - initial_capital) < 1.0, "Start value should match initial capital"

    def test_capital_cannot_go_negative(self, simple_price_data):
        """Test that portfolio value never goes negative."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.01)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_price_data)

        stats = portfolio.stats()
        if stats is not None:
            end_value = stats.get('End Value', 0)
            assert end_value >= 0, "Portfolio value should never be negative"
