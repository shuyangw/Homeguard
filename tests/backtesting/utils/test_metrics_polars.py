"""
Tests for Polars-native metrics calculations.

Includes parity tests comparing Polars vs pandas implementations.
"""

import pytest
import polars as pl
import pandas as pd
import numpy as np

from src.backtesting.utils.metrics_polars import (
    calculate_returns,
    calculate_total_return,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_all_metrics
)


class TestCalculateReturns:
    """Tests for calculate_returns function."""

    def test_simple_returns(self):
        """Test basic return calculation."""
        equity = pl.Series([100, 110, 105, 115])
        df = pl.DataFrame({"equity": equity})
        result = df.select(calculate_returns("equity").alias("returns"))
        returns = result["returns"]

        assert returns[0] == 0.0  # filled null
        assert abs(returns[1] - 0.10) < 1e-6  # 10% gain
        assert abs(returns[2] - (-0.0454545)) < 1e-4  # ~4.5% loss
        assert abs(returns[3] - 0.095238) < 1e-4  # ~9.5% gain


class TestCalculateTotalReturn:
    """Tests for calculate_total_return function."""

    def test_positive_return(self):
        """Test positive total return."""
        equity = pl.Series([100, 105, 110, 120])
        result = calculate_total_return(equity)
        assert abs(result - 20.0) < 1e-6

    def test_negative_return(self):
        """Test negative total return."""
        equity = pl.Series([100, 95, 90, 80])
        result = calculate_total_return(equity)
        assert abs(result - (-20.0)) < 1e-6

    def test_empty_series(self):
        """Test empty series returns 0."""
        equity = pl.Series([], dtype=pl.Float64)
        result = calculate_total_return(equity)
        assert result == 0.0

    def test_single_value(self):
        """Test single value returns 0."""
        equity = pl.Series([100])
        result = calculate_total_return(equity)
        assert result == 0.0


class TestCalculateSharpeRatio:
    """Tests for calculate_sharpe_ratio function."""

    def test_positive_sharpe(self):
        """Test positive Sharpe ratio."""
        # Steady upward trend = high Sharpe
        equity = pl.Series([100.0 + i * 1.0 for i in range(100)])
        result = calculate_sharpe_ratio(equity, periods_per_year=252)
        assert result > 0

    def test_zero_volatility(self):
        """Test zero volatility returns 0."""
        equity = pl.Series([100, 100, 100, 100])
        result = calculate_sharpe_ratio(equity)
        assert result == 0.0

    def test_empty_series(self):
        """Test empty series returns 0."""
        equity = pl.Series([], dtype=pl.Float64)
        result = calculate_sharpe_ratio(equity)
        assert result == 0.0


class TestCalculateMaxDrawdown:
    """Tests for calculate_max_drawdown function."""

    def test_simple_drawdown(self):
        """Test simple drawdown calculation."""
        equity = pl.Series([100, 110, 90, 95, 100])
        result = calculate_max_drawdown(equity)
        # Max was 110, dropped to 90 = -18.18% drawdown
        assert abs(result - (-18.18)) < 0.1

    def test_no_drawdown(self):
        """Test monotonically increasing equity."""
        equity = pl.Series([100, 110, 120, 130])
        result = calculate_max_drawdown(equity)
        assert result == 0.0

    def test_empty_series(self):
        """Test empty series returns 0."""
        equity = pl.Series([], dtype=pl.Float64)
        result = calculate_max_drawdown(equity)
        assert result == 0.0


class TestCalculateWinRate:
    """Tests for calculate_win_rate function."""

    def test_all_wins(self):
        """Test 100% win rate."""
        pnl = pl.Series([100, 200, 300])
        result = calculate_win_rate(pnl)
        assert result == 100.0

    def test_all_losses(self):
        """Test 0% win rate."""
        pnl = pl.Series([-100, -200, -300])
        result = calculate_win_rate(pnl)
        assert result == 0.0

    def test_mixed(self):
        """Test mixed win rate."""
        pnl = pl.Series([100, -50, 200, -100])
        result = calculate_win_rate(pnl)
        assert result == 50.0

    def test_empty(self):
        """Test empty series returns 0."""
        pnl = pl.Series([], dtype=pl.Float64)
        result = calculate_win_rate(pnl)
        assert result == 0.0


class TestCalculateProfitFactor:
    """Tests for calculate_profit_factor function."""

    def test_profit_factor(self):
        """Test profit factor calculation."""
        pnl = pl.Series([100, -50, 200, -50])
        result = calculate_profit_factor(pnl)
        # Gross profit = 300, gross loss = 100
        assert result == 3.0

    def test_no_losses(self):
        """Test all wins returns inf."""
        pnl = pl.Series([100, 200, 300])
        result = calculate_profit_factor(pnl)
        assert result == float('inf')

    def test_no_wins(self):
        """Test all losses returns 0."""
        pnl = pl.Series([-100, -200])
        result = calculate_profit_factor(pnl)
        assert result == 0.0


class TestCalculateAllMetrics:
    """Tests for calculate_all_metrics function."""

    def test_all_metrics_basic(self):
        """Test all metrics calculation."""
        equity = pl.Series([100, 105, 103, 110, 115])
        pnl = pl.Series([50, -20, 70, 50])

        metrics = calculate_all_metrics(equity, pnl)

        assert 'total_return_pct' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown_pct' in metrics
        assert 'win_rate_pct' in metrics
        assert 'profit_factor' in metrics
        assert 'total_trades' in metrics

        assert metrics['total_trades'] == 4
        assert metrics['win_rate_pct'] == 75.0

    def test_without_pnl(self):
        """Test metrics without P&L series."""
        equity = pl.Series([100, 110, 105, 120])
        metrics = calculate_all_metrics(equity)

        assert 'total_return_pct' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'win_rate_pct' not in metrics


class TestPolarsVsPandasParity:
    """Parity tests comparing Polars vs pandas implementations."""

    def test_total_return_parity(self):
        """Verify total return matches pandas calculation."""
        # Create test data
        data = [100000, 101000, 99500, 102000, 105000, 103000, 108000]

        # Polars
        equity_pl = pl.Series(data)
        result_pl = calculate_total_return(equity_pl)

        # Pandas equivalent
        equity_pd = pd.Series(data)
        result_pd = ((equity_pd.iloc[-1] - equity_pd.iloc[0]) / equity_pd.iloc[0]) * 100

        assert abs(result_pl - result_pd) < 1e-6

    def test_sharpe_ratio_parity(self):
        """Verify Sharpe ratio matches pandas calculation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
        equity = 100000 * np.cumprod(1 + returns)

        # Polars
        equity_pl = pl.Series(equity)
        result_pl = calculate_sharpe_ratio(equity_pl, periods_per_year=252)

        # Pandas equivalent
        equity_pd = pd.Series(equity)
        returns_pd = equity_pd.pct_change().dropna()
        mean_ret = returns_pd.mean()
        std_ret = returns_pd.std()
        result_pd = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

        # Allow small tolerance due to numerical differences
        assert abs(result_pl - result_pd) < 0.01

    def test_max_drawdown_parity(self):
        """Verify max drawdown matches pandas calculation."""
        data = [100, 110, 105, 95, 100, 90, 95, 100, 110]

        # Polars
        equity_pl = pl.Series(data)
        result_pl = calculate_max_drawdown(equity_pl)

        # Pandas equivalent
        equity_pd = pd.Series(data)
        cummax_pd = equity_pd.cummax()
        drawdown_pd = (equity_pd - cummax_pd) / cummax_pd * 100
        result_pd = drawdown_pd.min()

        assert abs(result_pl - result_pd) < 1e-6
