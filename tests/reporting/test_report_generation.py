"""
Comprehensive unit tests for report generation components.

Tests portfolio aggregation, chart generation, HTML reports, and tearsheets.
Includes edge cases like misaligned date ranges that caused the Sharpe bug.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.portfolio_aggregator import PortfolioAggregator
from backtesting.engine.results_aggregator import ResultsAggregator
from backtesting.engine.tearsheet_generator import TearsheetGenerator
from backtesting.engine.portfolio_simulator import Portfolio


@pytest.fixture
def aligned_portfolios():
    """
    Create portfolios with ALIGNED date ranges (same start/end dates).

    All symbols trade from Jan 1 to Jan 31 (31 days).
    This is the simple case - should give sensible aggregate metrics.
    """
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    np.random.seed(42)

    portfolios = {}
    for i, symbol in enumerate(['AAPL', 'MSFT', 'GOOGL']):
        # Generate upward trending equity curve with volatility
        base_value = 100000
        trend = np.linspace(0, 5000, len(dates))  # +5% trend
        volatility = np.random.randn(len(dates)) * 500  # Daily vol
        equity_curve = pd.Series(
            base_value + trend + volatility,
            index=dates,
            name=symbol
        )

        # Create mock portfolio with equity_curve attribute
        portfolio = type('Portfolio', (), {
            'equity_curve': equity_curve,
            'init_cash': base_value,
            'trades': []
        })()

        portfolios[symbol] = portfolio

    return portfolios


@pytest.fixture
def misaligned_portfolios():
    """
    Create portfolios with MISALIGNED date ranges (different end dates).

    AAPL: Jan 1 - Jan 31 (full month)
    MSFT: Jan 1 - Jan 28 (stops 3 days early)
    GOOGL: Jan 1 - Jan 25 (stops 6 days early)

    This tests the ffill bug - aggregate Sharpe should still be sensible.
    """
    dates_full = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    dates_28 = pd.date_range(start='2023-01-01', end='2023-01-28', freq='D')
    dates_25 = pd.date_range(start='2023-01-01', end='2023-01-25', freq='D')

    np.random.seed(42)

    portfolios = {}

    # AAPL: Full month, Sharpe ~1.5
    base_value = 100000
    trend = np.linspace(0, 6000, len(dates_full))  # +6% trend
    volatility = np.random.randn(len(dates_full)) * 400
    equity_aapl = pd.Series(base_value + trend + volatility, index=dates_full)
    portfolios['AAPL'] = type('Portfolio', (), {
        'equity_curve': equity_aapl,
        'init_cash': base_value,
        'trades': []
    })()

    # MSFT: Ends Jan 28, Sharpe ~1.3
    trend = np.linspace(0, 5000, len(dates_28))
    volatility = np.random.randn(len(dates_28)) * 400
    equity_msft = pd.Series(base_value + trend + volatility, index=dates_28)
    portfolios['MSFT'] = type('Portfolio', (), {
        'equity_curve': equity_msft,
        'init_cash': base_value,
        'trades': []
    })()

    # GOOGL: Ends Jan 25, Sharpe ~1.4
    trend = np.linspace(0, 5500, len(dates_25))
    volatility = np.random.randn(len(dates_25)) * 400
    equity_googl = pd.Series(base_value + trend + volatility, index=dates_25)
    portfolios['GOOGL'] = type('Portfolio', (), {
        'equity_curve': equity_googl,
        'init_cash': base_value,
        'trades': []
    })()

    return portfolios


@pytest.fixture
def high_sharpe_portfolios():
    """
    Create portfolios with HIGH Sharpe ratios (>2.0) for all symbols.

    Tests that aggregate Sharpe is also high (not dragged down to 0.1).
    """
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')  # 1 year
    np.random.seed(123)

    portfolios = {}
    for symbol in ['NFLX', 'AMZN', 'META', 'AAPL', 'GOOGL']:
        # High return, low volatility = high Sharpe
        base_value = 100000
        trend = np.linspace(0, 30000, len(dates))  # +30% annual return
        volatility = np.random.randn(len(dates)) * 200  # Low vol
        equity_curve = pd.Series(
            base_value + trend + volatility,
            index=dates
        )

        portfolios[symbol] = type('Portfolio', (), {
            'equity_curve': equity_curve,
            'init_cash': base_value,
            'trades': []
        })()

    return portfolios


class TestPortfolioAggregation:
    """
    Test PortfolioAggregator metrics calculation.

    Critical: These tests would have caught the Sharpe ratio ffill bug.
    """

    def test_aligned_sharpe_ratio(self, aligned_portfolios):
        """
        Test Sharpe calculation with aligned date ranges.

        Expected: Aggregate Sharpe should be positive and reasonable.
        """
        initial_capital = 100000
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            aligned_portfolios,
            initial_capital
        )

        sharpe = metrics.get('Sharpe Ratio', 0)

        # Sharpe should be positive for upward trending portfolios
        assert sharpe > 0, f"Sharpe should be positive, got {sharpe}"

        # Sharpe can be high for synthetic data with strong trends
        # Real-world Sharpe > 3 is exceptional, but synthetic data can be higher
        assert 0.5 <= sharpe <= 10.0, f"Sharpe should be reasonable, got {sharpe}"

    def test_misaligned_sharpe_ratio(self, misaligned_portfolios):
        """
        Test Sharpe calculation with MISALIGNED date ranges.

        CRITICAL: This is the bug we fixed - aggregate Sharpe was 0.1
        even though individual Sharpes were all >1.0.

        Expected: Aggregate Sharpe should still be >1.0, not dragged down by ffill.
        """
        initial_capital = 100000
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            misaligned_portfolios,
            initial_capital
        )

        sharpe = metrics.get('Sharpe Ratio', 0)

        # CRITICAL TEST: Sharpe should NOT be near zero
        assert sharpe > 0.5, f"Bug detected: Sharpe dragged down by ffill, got {sharpe}"

        # Should be consistent with individual Sharpes (all around 1.3-1.5)
        assert sharpe > 1.0, f"Aggregate Sharpe should be >1.0 when individuals are, got {sharpe}"

    def test_high_sharpe_portfolios(self, high_sharpe_portfolios):
        """
        Test that high individual Sharpes result in high aggregate Sharpe.

        Expected: If all symbols have Sharpe >2.0, aggregate should be >1.5.
        """
        initial_capital = 100000
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            high_sharpe_portfolios,
            initial_capital
        )

        sharpe = metrics.get('Sharpe Ratio', 0)

        # High Sharpe portfolios should give high aggregate Sharpe
        assert sharpe > 1.5, f"High Sharpe portfolios should give aggregate >1.5, got {sharpe}"

    def test_volatility_calculation(self, aligned_portfolios):
        """
        Test volatility calculation is annualized correctly.

        Expected: Volatility should be reasonable (5-30% annualized for stocks).
        """
        initial_capital = 100000
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            aligned_portfolios,
            initial_capital
        )

        vol = metrics.get('Volatility [%]', 0)

        # Volatility should be positive
        assert vol > 0, f"Volatility should be positive, got {vol}"

        # Should be reasonable for stock portfolios
        assert 1.0 <= vol <= 50.0, f"Volatility should be reasonable, got {vol}%"

    def test_total_return_calculation(self, aligned_portfolios):
        """
        Test total return sums correctly across symbols.

        Expected: Total return should be sum of all symbol returns.
        """
        initial_capital = 100000
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            aligned_portfolios,
            initial_capital
        )

        total_return = metrics.get('Total Return [%]', 0)
        total_initial = metrics.get('Total Initial Capital', 0)
        total_final = metrics.get('Total Final Value', 0)

        # Verify math: (final - initial) / initial * 100
        expected_return = ((total_final - total_initial) / total_initial) * 100

        assert abs(total_return - expected_return) < 0.01, \
            f"Total return calculation error: {total_return} vs {expected_return}"

    def test_max_drawdown_calculation(self, aligned_portfolios):
        """
        Test max drawdown calculation.

        Expected: Max drawdown should be negative and reasonable.
        """
        initial_capital = 100000
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            aligned_portfolios,
            initial_capital
        )

        max_dd = metrics.get('Max Drawdown [%]', 0)

        # Max drawdown should be negative (or zero if no drawdown)
        assert max_dd <= 0, f"Max drawdown should be <=0, got {max_dd}"

        # Should be reasonable (not -100% for moderate portfolios)
        assert max_dd >= -50, f"Max drawdown seems too large, got {max_dd}%"

    def test_win_rate_range(self, aligned_portfolios):
        """
        Test win rate is in valid range [0, 100].

        Expected: Win rate should be between 0% and 100%.
        """
        initial_capital = 100000
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            aligned_portfolios,
            initial_capital
        )

        win_rate = metrics.get('Win Rate [%]', 0)

        # Win rate must be between 0 and 100
        assert 0 <= win_rate <= 100, f"Win rate out of range: {win_rate}%"

    def test_symbol_contributions(self, aligned_portfolios):
        """
        Test symbol contribution breakdown.

        Expected: Sum of individual P&Ls should equal total P&L.
        """
        initial_capital = 100000
        contributions = PortfolioAggregator.calculate_symbol_contributions(
            aligned_portfolios,
            initial_capital
        )

        # Should have one row per symbol
        assert len(contributions) == len(aligned_portfolios), \
            f"Expected {len(aligned_portfolios)} symbols, got {len(contributions)}"

        # Sum of individual P&Ls
        total_pnl = contributions['P&L'].sum()

        # Calculate expected total P&L
        expected_pnl = sum(
            port.equity_curve.iloc[-1] - initial_capital
            for port in aligned_portfolios.values()
        )

        # Should match
        assert abs(total_pnl - expected_pnl) < 1.0, \
            f"P&L sum mismatch: {total_pnl} vs {expected_pnl}"


class TestChartDataGeneration:
    """
    Test chart data generation for HTML reports.
    """

    def test_equity_chart_data_structure(self, aligned_portfolios):
        """
        Test equity curve chart data has correct structure.

        Expected: Chart data should have 'labels' and 'datasets' keys.
        """
        initial_capital = 100000
        chart_data = PortfolioAggregator.generate_portfolio_composition_chart_data(
            aligned_portfolios,
            initial_capital
        )

        # Should have required keys
        assert 'labels' in chart_data, "Chart data missing 'labels' key"
        assert 'datasets' in chart_data, "Chart data missing 'datasets' key"

        # Should have one dataset per symbol
        assert len(chart_data['datasets']) == len(aligned_portfolios), \
            f"Expected {len(aligned_portfolios)} datasets, got {len(chart_data['datasets'])}"

    def test_equity_chart_overlaid_not_stacked(self, aligned_portfolios):
        """
        Test equity curve chart is overlaid (not stacked).

        Expected: Datasets should have transparent background (not filled areas).
        """
        initial_capital = 100000
        chart_data = PortfolioAggregator.generate_portfolio_composition_chart_data(
            aligned_portfolios,
            initial_capital
        )

        # Check all datasets have transparent background
        for dataset in chart_data['datasets']:
            bg_color = dataset.get('backgroundColor', '')
            assert bg_color == 'transparent', \
                f"Chart should be overlaid lines, not stacked area. Got backgroundColor: {bg_color}"

    def test_chart_data_valid_json(self, aligned_portfolios):
        """
        Test chart data can be serialized to JSON.

        Expected: No NaN or inf values that break JSON serialization.
        """
        import json

        initial_capital = 100000
        chart_data = PortfolioAggregator.generate_portfolio_composition_chart_data(
            aligned_portfolios,
            initial_capital
        )

        # Should serialize without errors
        try:
            json_str = json.dumps(chart_data)
            assert len(json_str) > 0, "JSON serialization produced empty string"
        except (TypeError, ValueError) as e:
            pytest.fail(f"Chart data cannot be serialized to JSON: {e}")


class TestResultsAggregation:
    """
    Test ResultsAggregator for HTML report generation.
    """

    def test_calculate_summary_stats(self):
        """
        Test summary statistics calculation from results DataFrame.
        """
        # Create sample results DataFrame
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Total Return [%]': [10.5, 8.2, 12.1],
            'Sharpe Ratio': [1.5, 1.3, 1.7],
            'Max Drawdown [%]': [-5.2, -6.1, -4.8],
            'Win Rate [%]': [55.0, 52.0, 58.0]
        })

        summary = ResultsAggregator.calculate_summary_stats(df)

        # Check expected keys exist
        expected_keys = [
            'Total Symbols',
            'Total Return [%] - Mean',
            'Sharpe Ratio - Mean',
            'Max Drawdown [%] - Mean',
            'Win Rate (Symbols)'
        ]

        for key in expected_keys:
            assert key in summary, f"Summary missing expected key: {key}"

        # Verify calculations
        assert summary['Total Symbols'] == 3
        assert abs(summary['Total Return [%] - Mean'] - 10.27) < 0.1
        assert abs(summary['Sharpe Ratio - Mean'] - 1.5) < 0.1

    def test_html_export_creates_file(self, tmp_path, aligned_portfolios):
        """
        Test HTML export creates a file.
        """
        # Create sample DataFrame
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Total Return [%]': [10.5, 8.2, 12.1],
            'Sharpe Ratio': [1.5, 1.3, 1.7],
            'Max Drawdown [%]': [-5.2, -6.1, -4.8],
            'Win Rate [%]': [55.0, 52.0, 58.0]
        })

        output_file = tmp_path / "test_report.html"

        # Export to HTML
        ResultsAggregator.export_to_html(
            df,
            output_file,
            title="Test Report",
            portfolios=aligned_portfolios
        )

        # File should exist
        assert output_file.exists(), "HTML file was not created"

        # File should have content
        content = output_file.read_text(encoding='utf-8')
        assert len(content) > 1000, "HTML file seems too small"

        # Should contain expected elements
        assert '<html' in content.lower(), "Not valid HTML"
        assert 'Test Report' in content, "Title missing from HTML"
        assert 'AAPL' in content, "Symbol data missing from HTML"

    def test_html_contains_all_charts(self, tmp_path, aligned_portfolios):
        """
        Test HTML report contains all expected chart canvases.
        """
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Total Return [%]': [10.5, 8.2, 12.1],
            'Sharpe Ratio': [1.5, 1.3, 1.7],
            'Max Drawdown [%]': [-5.2, -6.1, -4.8]
        })

        output_file = tmp_path / "test_report.html"

        ResultsAggregator.export_to_html(
            df,
            output_file,
            portfolios=aligned_portfolios
        )

        content = output_file.read_text(encoding='utf-8')

        # Check for chart canvas elements
        expected_charts = [
            'combinedEquityChart',  # Combined equity curve
            'returnsChart',         # Returns by symbol
            'sharpeChart',          # Sharpe ratio chart
            'drawdownChart',        # Drawdown chart
            'metricsRadar',         # Radar chart
            'returnsDistChart',     # Returns distribution
            'riskReturnScatter',    # Risk-return scatter
            'winRateChart'          # Win rate chart
        ]

        for chart_id in expected_charts:
            assert chart_id in content, f"Chart missing from HTML: {chart_id}"


class TestTearsheetGeneration:
    """
    Test TearsheetGenerator for QuantStats-style tearsheets.
    """

    def test_generate_returns_from_portfolio(self, aligned_portfolios):
        """
        Test returns extraction from portfolio.
        """
        portfolio = aligned_portfolios['AAPL']
        returns = TearsheetGenerator.generate_returns(portfolio)

        # Should be a pandas Series
        assert isinstance(returns, pd.Series), "Returns should be a pandas Series"

        # Should have same length as equity curve (fillna(0) preserves length)
        # The first value (NaN from pct_change) is filled with 0
        expected_len = len(portfolio.equity_curve)
        assert len(returns) == expected_len, \
            f"Returns length mismatch: expected {expected_len}, got {len(returns)}"

        # Should not contain NaN (filled with 0)
        assert not returns.isna().any(), "Returns should not contain NaN"

    def test_monthly_returns_table_html(self, aligned_portfolios):
        """
        Test monthly returns table HTML generation.
        """
        portfolio = aligned_portfolios['AAPL']
        returns = TearsheetGenerator.generate_returns(portfolio)

        html = TearsheetGenerator.generate_monthly_returns_table(returns)

        # Should contain HTML table elements
        assert '<table' in html.lower(), "Should generate HTML table"
        assert 'Jan' in html or 'January' in html, "Should have month labels"

    def test_yearly_returns_table_html(self, aligned_portfolios):
        """
        Test yearly returns table HTML generation.
        """
        portfolio = aligned_portfolios['AAPL']
        returns = TearsheetGenerator.generate_returns(portfolio)

        html = TearsheetGenerator.generate_yearly_returns_table(returns)

        # Should contain HTML table elements
        assert '<table' in html.lower(), "Should generate HTML table"
        assert '2023' in html, "Should have year 2023"


class TestEdgeCases:
    """
    Test edge cases and error conditions.
    """

    def test_empty_portfolios_dict(self):
        """
        Test handling of empty portfolios dictionary.
        """
        metrics = PortfolioAggregator.calculate_aggregate_metrics({}, 100000)

        # Should return empty dict, not crash
        assert metrics == {}, "Empty portfolios should return empty metrics"

    def test_single_symbol_portfolio(self):
        """
        Test aggregation with only one symbol.
        """
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        equity = pd.Series(
            100000 + np.linspace(0, 5000, 30) + np.random.randn(30) * 200,
            index=dates
        )

        portfolio = type('Portfolio', (), {
            'equity_curve': equity,
            'init_cash': 100000,
            'trades': []
        })()

        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            {'AAPL': portfolio},
            100000
        )

        # Should work with single symbol
        assert 'Sharpe Ratio' in metrics, "Should calculate metrics for single symbol"
        assert metrics['Sharpe Ratio'] is not None

    def test_portfolio_with_no_trades(self):
        """
        Test portfolio that made no trades (flat equity).
        """
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        equity = pd.Series([100000] * 30, index=dates)  # Perfectly flat

        portfolio = type('Portfolio', (), {
            'equity_curve': equity,
            'init_cash': 100000,
            'trades': []
        })()

        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            {'AAPL': portfolio},
            100000
        )

        # Sharpe should be 0 for flat equity (no returns)
        assert metrics['Sharpe Ratio'] == 0, "Flat equity should give Sharpe = 0"
        assert metrics['Volatility [%]'] == 0, "Flat equity should give Volatility = 0"
