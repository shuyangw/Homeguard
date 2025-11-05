"""
Unit tests for benchmark comparison feature.

Tests buy-and-hold calculations, outperformance metrics,
chart data generation, and HTML tearsheet integration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.benchmark_calculator import BenchmarkCalculator
from backtesting.engine.portfolio_aggregator import PortfolioAggregator
from backtesting.engine.results_aggregator import ResultsAggregator


# ============================================================================
# FIXTURES
# ============================================================================

class MockDataLoader:
    """Mock DataLoader for testing without database access."""

    def __init__(self, price_data_dict):
        """
        Initialize with price data dictionary.

        Args:
            price_data_dict: Dict of {symbol: DataFrame} with OHLCV data
        """
        self.price_data_dict = price_data_dict

    def load_single_symbol(self, symbol, start_date, end_date):
        """Load price data for a single symbol."""
        if symbol not in self.price_data_dict:
            raise ValueError(f"No data available for {symbol}")
        return self.price_data_dict[symbol].copy()


@pytest.fixture
def uptrend_price_data():
    """Generate uptrending price data for buy-and-hold testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    base_price = 100.0
    trend = np.linspace(0, 20, 100)  # +20% uptrend
    noise = np.random.randn(100) * 1
    close_prices = base_price + trend + noise

    df = pd.DataFrame({
        'open': close_prices + np.random.randn(100) * 0.2,
        'high': close_prices + np.abs(np.random.randn(100) * 0.3),
        'low': close_prices - np.abs(np.random.randn(100) * 0.3),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    return df


@pytest.fixture
def downtrend_price_data():
    """Generate downtrending price data for buy-and-hold testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(123)

    base_price = 100.0
    trend = np.linspace(0, -15, 100)  # -15% downtrend
    noise = np.random.randn(100) * 1
    close_prices = base_price + trend + noise

    df = pd.DataFrame({
        'open': close_prices + np.random.randn(100) * 0.2,
        'high': close_prices + np.abs(np.random.randn(100) * 0.3),
        'low': close_prices - np.abs(np.random.randn(100) * 0.3),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    return df


@pytest.fixture
def mock_data_loader_uptrend(uptrend_price_data):
    """Mock DataLoader with uptrending data for AAPL, MSFT, GOOGL."""
    price_data = {
        'AAPL': uptrend_price_data.copy(),
        'MSFT': uptrend_price_data.copy() * 1.2,  # Higher prices
        'GOOGL': uptrend_price_data.copy() * 0.8,  # Lower prices
        'SPY': uptrend_price_data.copy() * 1.5  # S&P 500 proxy
    }
    return MockDataLoader(price_data)


@pytest.fixture
def outperforming_portfolios():
    """
    Create portfolios where strategy outperforms buy-and-hold.

    Strategy returns: +15%, +12%, +18%
    Buy-hold would return: ~+20% (based on uptrend data)
    We'll use strategy equity curves that beat passive holding.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    portfolios = {}

    for i, symbol in enumerate(['AAPL', 'MSFT', 'GOOGL']):
        # Strategy with higher return than buy-and-hold
        base_value = 100000
        # Aggressive trend to beat buy-and-hold
        trend = np.linspace(0, 25000 + (i * 5000), 100)  # +25%, +30%, +35%
        volatility = np.random.randn(100) * 500
        equity_curve = pd.Series(
            base_value + trend + volatility,
            index=dates,
            name=symbol
        )

        portfolios[symbol] = type('Portfolio', (), {
            'equity_curve': equity_curve,
            'init_cash': base_value,
            'trades': []
        })()

    return portfolios


@pytest.fixture
def underperforming_portfolios():
    """
    Create portfolios where strategy underperforms buy-and-hold.

    Strategy returns: +3%, +5%, +2%
    Buy-hold would return: ~+20%
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(123)

    portfolios = {}

    for i, symbol in enumerate(['AAPL', 'MSFT', 'GOOGL']):
        # Strategy with lower return than buy-and-hold
        base_value = 100000
        # Modest trend - underperforms buy-and-hold
        trend = np.linspace(0, 3000 + (i * 1000), 100)  # +3%, +4%, +5%
        volatility = np.random.randn(100) * 300
        equity_curve = pd.Series(
            base_value + trend + volatility,
            index=dates,
            name=symbol
        )

        portfolios[symbol] = type('Portfolio', (), {
            'equity_curve': equity_curve,
            'init_cash': base_value,
            'trades': []
        })()

    return portfolios


@pytest.fixture
def mixed_performance_portfolios():
    """
    Create portfolios with mixed performance.

    AAPL: Outperforms (+30%)
    MSFT: Underperforms (+5%)
    GOOGL: Outperforms (+25%)
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(456)

    portfolios = {}

    # AAPL: Outperformer
    base_value = 100000
    trend = np.linspace(0, 30000, 100)  # +30%
    equity_aapl = pd.Series(base_value + trend + np.random.randn(100) * 500, index=dates)
    portfolios['AAPL'] = type('Portfolio', (), {
        'equity_curve': equity_aapl,
        'init_cash': base_value
    })()

    # MSFT: Underperformer
    trend = np.linspace(0, 5000, 100)  # +5%
    equity_msft = pd.Series(base_value + trend + np.random.randn(100) * 300, index=dates)
    portfolios['MSFT'] = type('Portfolio', (), {
        'equity_curve': equity_msft,
        'init_cash': base_value
    })()

    # GOOGL: Outperformer
    trend = np.linspace(0, 25000, 100)  # +25%
    equity_googl = pd.Series(base_value + trend + np.random.randn(100) * 500, index=dates)
    portfolios['GOOGL'] = type('Portfolio', (), {
        'equity_curve': equity_googl,
        'init_cash': base_value
    })()

    return portfolios


# ============================================================================
# TEST CLASS: BenchmarkCalculator
# ============================================================================

class TestBenchmarkCalculator:
    """Test BenchmarkCalculator buy-and-hold and outperformance calculations."""

    def test_buy_and_hold_with_uptrend(self, uptrend_price_data, mock_data_loader_uptrend):
        """Test buy-and-hold calculation with uptrending prices."""
        buy_hold = BenchmarkCalculator.calculate_buy_and_hold_equity(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_data_loader_uptrend
        )

        assert buy_hold is not None, "Buy-and-hold equity should be calculated"
        assert len(buy_hold) == 100, "Should have 100 data points"
        assert buy_hold.iloc[0] == pytest.approx(100000, rel=0.01), "First value should be initial capital"
        assert buy_hold.iloc[-1] > 100000, "Final value should be higher with uptrend"

        # Calculate expected return (~+20% from uptrend)
        return_pct = ((buy_hold.iloc[-1] - buy_hold.iloc[0]) / buy_hold.iloc[0]) * 100
        assert 15 < return_pct < 25, f"Return should be ~20%, got {return_pct:.2f}%"

    def test_buy_and_hold_with_downtrend(self, downtrend_price_data):
        """Test buy-and-hold calculation with downtrending prices."""
        mock_loader = MockDataLoader({'AAPL': downtrend_price_data})

        buy_hold = BenchmarkCalculator.calculate_buy_and_hold_equity(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_loader
        )

        assert buy_hold is not None
        assert buy_hold.iloc[-1] < 100000, "Final value should be lower with downtrend"

        # Calculate expected return (~-15% from downtrend)
        return_pct = ((buy_hold.iloc[-1] - buy_hold.iloc[0]) / buy_hold.iloc[0]) * 100
        assert -20 < return_pct < -10, f"Return should be ~-15%, got {return_pct:.2f}%"

    def test_buy_and_hold_missing_symbol(self, mock_data_loader_uptrend):
        """Test graceful handling when symbol data unavailable."""
        buy_hold = BenchmarkCalculator.calculate_buy_and_hold_equity(
            symbol='TSLA',  # Not in mock data
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_data_loader_uptrend
        )

        assert buy_hold is None, "Should return None when symbol data unavailable"

    def test_spy_benchmark_calculation(self, mock_data_loader_uptrend):
        """Test SPY benchmark calculation."""
        spy_equity = BenchmarkCalculator.calculate_spy_benchmark(
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=300000,  # 3 symbols * 100k
            data_loader=mock_data_loader_uptrend
        )

        assert spy_equity is not None, "SPY equity should be calculated"
        assert spy_equity.iloc[0] == pytest.approx(300000, rel=0.01)

    def test_outperformance_calculation_positive(self):
        """Test outperformance calculation when strategy beats benchmark."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        # Strategy: +20% return
        strategy_equity = pd.Series(
            np.linspace(100000, 120000, 100),
            index=dates
        )

        # Benchmark: +10% return
        benchmark_equity = pd.Series(
            np.linspace(100000, 110000, 100),
            index=dates
        )

        metrics = BenchmarkCalculator.calculate_outperformance(
            strategy_equity, benchmark_equity
        )

        assert 'strategy_return_pct' in metrics
        assert 'benchmark_return_pct' in metrics
        assert 'outperformance_pct' in metrics
        assert 'alpha' in metrics

        assert metrics['strategy_return_pct'] == pytest.approx(20.0, abs=0.1)
        assert metrics['benchmark_return_pct'] == pytest.approx(10.0, abs=0.1)
        assert metrics['outperformance_pct'] == pytest.approx(10.0, abs=0.1)
        assert metrics['alpha'] > 0, "Alpha should be positive when strategy outperforms"

    def test_outperformance_calculation_negative(self):
        """Test outperformance calculation when strategy underperforms."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        # Strategy: +5% return
        strategy_equity = pd.Series(
            np.linspace(100000, 105000, 100),
            index=dates
        )

        # Benchmark: +15% return
        benchmark_equity = pd.Series(
            np.linspace(100000, 115000, 100),
            index=dates
        )

        metrics = BenchmarkCalculator.calculate_outperformance(
            strategy_equity, benchmark_equity
        )

        assert metrics['outperformance_pct'] == pytest.approx(-10.0, abs=0.1)
        assert metrics['alpha'] < 0, "Alpha should be negative when strategy underperforms"

    def test_all_benchmarks_integration(self, outperforming_portfolios, mock_data_loader_uptrend):
        """Test calculate_all_benchmarks with multiple symbols."""
        benchmark_data = BenchmarkCalculator.calculate_all_benchmarks(
            portfolios=outperforming_portfolios,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_data_loader_uptrend,
            include_spy=True
        )

        assert 'per_symbol' in benchmark_data
        assert 'spy' in benchmark_data
        assert 'outperformers' in benchmark_data
        assert 'underperformers' in benchmark_data

        # Should have data for all 3 symbols
        assert len(benchmark_data['per_symbol']) == 3

        # Each symbol should have buy_hold_equity, strategy_equity, metrics
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            assert symbol in benchmark_data['per_symbol']
            assert 'buy_hold_equity' in benchmark_data['per_symbol'][symbol]
            assert 'strategy_equity' in benchmark_data['per_symbol'][symbol]
            assert 'metrics' in benchmark_data['per_symbol'][symbol]

        # SPY should be available
        assert benchmark_data['spy']['equity'] is not None


# ============================================================================
# TEST CLASS: PortfolioAggregator Benchmarks
# ============================================================================

class TestPortfolioAggregatorBenchmarks:
    """Test PortfolioAggregator benchmark chart data generation."""

    def test_benchmark_chart_data_structure(self, outperforming_portfolios, mock_data_loader_uptrend):
        """Test benchmark comparison chart data has correct structure."""
        benchmark_data = BenchmarkCalculator.calculate_all_benchmarks(
            portfolios=outperforming_portfolios,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_data_loader_uptrend
        )

        chart_data = PortfolioAggregator.generate_benchmark_comparison_chart_data(
            benchmark_data, 100000
        )

        assert 'labels' in chart_data, "Chart should have labels (dates)"
        assert 'datasets' in chart_data, "Chart should have datasets"
        assert 'outperformers' in chart_data
        assert 'underperformers' in chart_data

        # Should have 2 datasets per symbol (strategy + benchmark)
        assert len(chart_data['datasets']) == 6, "3 symbols * 2 lines = 6 datasets"

        # Check labels are dates
        assert len(chart_data['labels']) > 0
        assert isinstance(chart_data['labels'][0], str)

    def test_benchmark_chart_has_both_line_types(self, outperforming_portfolios, mock_data_loader_uptrend):
        """Test chart includes both strategy and benchmark lines."""
        benchmark_data = BenchmarkCalculator.calculate_all_benchmarks(
            portfolios=outperforming_portfolios,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_data_loader_uptrend
        )

        chart_data = PortfolioAggregator.generate_benchmark_comparison_chart_data(
            benchmark_data, 100000
        )

        strategy_lines = [ds for ds in chart_data['datasets'] if ds.get('type') == 'strategy']
        benchmark_lines = [ds for ds in chart_data['datasets'] if ds.get('type') == 'benchmark']

        assert len(strategy_lines) == 3, "Should have 3 strategy lines"
        assert len(benchmark_lines) == 3, "Should have 3 benchmark lines"

        # Benchmark lines should be dashed
        for ds in benchmark_lines:
            assert ds.get('borderDash') == [5, 5], "Benchmark lines should be dashed"

    def test_outperformers_categorization(self, mixed_performance_portfolios, mock_data_loader_uptrend):
        """Test correct categorization of outperformers vs underperformers."""
        benchmark_data = BenchmarkCalculator.calculate_all_benchmarks(
            portfolios=mixed_performance_portfolios,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_data_loader_uptrend
        )

        # AAPL and GOOGL should outperform (+30%, +25% vs ~+20% buy-hold)
        # MSFT should underperform (+5% vs ~+20% buy-hold)
        assert len(benchmark_data['outperformers']) == 2, "Should have 2 outperformers"
        assert len(benchmark_data['underperformers']) == 1, "Should have 1 underperformer"

        assert 'MSFT' in benchmark_data['underperformers'], "MSFT should underperform"

    def test_spy_chart_data_generation(self, outperforming_portfolios, mock_data_loader_uptrend):
        """Test SPY comparison chart data generation."""
        benchmark_data = BenchmarkCalculator.calculate_all_benchmarks(
            portfolios=outperforming_portfolios,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_data_loader_uptrend,
            include_spy=True
        )

        # Create aggregate equity curve
        equity_df = PortfolioAggregator.combine_equity_curves(outperforming_portfolios, 100000)

        spy_chart = PortfolioAggregator.generate_spy_comparison_chart_data(
            benchmark_data,
            equity_df['Total Portfolio'],
            300000
        )

        assert 'labels' in spy_chart
        assert 'datasets' in spy_chart
        assert len(spy_chart['datasets']) == 2, "Should have 2 lines: Strategy + SPY"

        # Check labels
        strategy_line = [ds for ds in spy_chart['datasets'] if 'Strategy' in ds['label']][0]
        spy_line = [ds for ds in spy_chart['datasets'] if 'SPY' in ds['label']][0]

        assert strategy_line is not None
        assert spy_line is not None
        assert spy_line.get('borderDash') == [5, 5], "SPY line should be dashed"

    def test_empty_portfolios_handling(self):
        """Test graceful handling of empty portfolios dict."""
        chart_data = PortfolioAggregator.generate_benchmark_comparison_chart_data({}, 100000)

        assert chart_data == {}, "Empty portfolios should return empty dict"

    def test_single_symbol_chart_data(self, outperforming_portfolios, mock_data_loader_uptrend):
        """Test chart generation with single symbol."""
        single_portfolio = {'AAPL': outperforming_portfolios['AAPL']}

        benchmark_data = BenchmarkCalculator.calculate_all_benchmarks(
            portfolios=single_portfolio,
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_data_loader_uptrend
        )

        chart_data = PortfolioAggregator.generate_benchmark_comparison_chart_data(
            benchmark_data, 100000
        )

        assert len(chart_data['datasets']) == 2, "Single symbol should have 2 lines"


# ============================================================================
# TEST CLASS: HTML Benchmark Integration
# ============================================================================

class TestHTMLBenchmarkIntegration:
    """Test HTML tearsheet integration with benchmarks."""

    def test_html_includes_benchmark_section(self, outperforming_portfolios, mock_data_loader_uptrend, tmp_path):
        """Test HTML contains benchmark comparison section."""
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Total Return [%]': [25.0, 30.0, 35.0],
            'Sharpe Ratio': [1.5, 1.8, 2.0],
            'Max Drawdown [%]': [-5.0, -6.0, -4.0]
        })

        html_path = tmp_path / "test_report.html"

        ResultsAggregator.export_to_html(
            df,
            html_path,
            portfolios=outperforming_portfolios,
            data_loader=mock_data_loader_uptrend,
            start_date='2023-01-01',
            end_date='2023-04-10',
            include_benchmarks=True
        )

        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()

        assert 'Strategy vs Buy-and-Hold Comparison' in html, "Should have benchmark section"
        assert 'benchmarkComparisonChart' in html, "Should have benchmark chart canvas"

    def test_html_benchmark_stats_cards(self, outperforming_portfolios, mock_data_loader_uptrend, tmp_path):
        """Test HTML contains outperformers/underperformers stats cards."""
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Total Return [%]': [25.0, 30.0, 35.0],
            'Sharpe Ratio': [1.5, 1.8, 2.0],
            'Max Drawdown [%]': [-5.0, -6.0, -4.0]
        })

        html_path = tmp_path / "test_report.html"

        ResultsAggregator.export_to_html(
            df,
            html_path,
            portfolios=outperforming_portfolios,
            data_loader=mock_data_loader_uptrend,
            start_date='2023-01-01',
            end_date='2023-04-10',
            include_benchmarks=True
        )

        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()

        assert 'Outperformers' in html
        assert 'Underperformers' in html
        assert 'Success Rate' in html

    def test_html_toggle_controls_present(self, outperforming_portfolios, mock_data_loader_uptrend, tmp_path):
        """Test HTML contains toggle controls."""
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Total Return [%]': [25.0, 30.0, 35.0],
            'Sharpe Ratio': [1.5, 1.8, 2.0],
            'Max Drawdown [%]': [-5.0, -6.0, -4.0]
        })

        html_path = tmp_path / "test_report.html"

        ResultsAggregator.export_to_html(
            df,
            html_path,
            portfolios=outperforming_portfolios,
            data_loader=mock_data_loader_uptrend,
            start_date='2023-01-01',
            end_date='2023-04-10',
            include_benchmarks=True
        )

        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()

        assert 'toggleAllSymbols' in html, "Should have toggleAllSymbols function"
        assert 'toggleBenchmarks' in html, "Should have toggleBenchmarks function"
        assert 'showOnlyOutperformers' in html, "Should have showOnlyOutperformers function"
        assert 'Show All' in html, "Should have Show All button"
        assert 'Hide All' in html, "Should have Hide All button"
        assert 'Only Outperformers' in html, "Should have Only Outperformers button"

    def test_html_backward_compatible_no_benchmarks(self, outperforming_portfolios, tmp_path):
        """Test HTML works when benchmarks disabled (backward compatibility)."""
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Total Return [%]': [25.0, 30.0, 35.0],
            'Sharpe Ratio': [1.5, 1.8, 2.0],
            'Max Drawdown [%]': [-5.0, -6.0, -4.0]
        })

        html_path = tmp_path / "test_report_no_benchmarks.html"

        # Don't pass data_loader or dates - benchmarks should be skipped
        ResultsAggregator.export_to_html(
            df,
            html_path,
            portfolios=outperforming_portfolios,
            include_benchmarks=False
        )

        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()

        # Verify the key user-visible HTML elements are NOT present
        # (JavaScript code may reference variables but that's harmless if elements don't exist)
        assert 'Strategy vs Buy-and-Hold Comparison' not in html, "Benchmark section header should not be present"
        assert '<canvas id="benchmarkComparisonChart">' not in html, "Benchmark chart canvas element should not be present"
        assert 'Only Outperformers</button>' not in html, "Outperformers button should not be present"

        # Verify HTML file was created and contains expected non-benchmark content
        assert html_path.exists(), "HTML file should be created"
        assert 'Backtest Sweep Results' in html, "Should have title"
        assert 'AAPL' in html, "Should contain symbol data"


# ============================================================================
# TEST CLASS: Edge Cases
# ============================================================================

class TestBenchmarkEdgeCases:
    """Test edge cases in benchmark calculations."""

    def test_all_symbols_outperform(self, outperforming_portfolios, mock_data_loader_uptrend):
        """Test when all symbols beat buy-and-hold."""
        benchmark_data = BenchmarkCalculator.calculate_all_benchmarks(
            portfolios=outperforming_portfolios,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_data_loader_uptrend
        )

        assert len(benchmark_data['outperformers']) == 3
        assert len(benchmark_data['underperformers']) == 0

    def test_all_symbols_underperform(self, underperforming_portfolios, mock_data_loader_uptrend):
        """Test when all symbols lose to buy-and-hold."""
        benchmark_data = BenchmarkCalculator.calculate_all_benchmarks(
            portfolios=underperforming_portfolios,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_data_loader_uptrend
        )

        assert len(benchmark_data['outperformers']) == 0
        assert len(benchmark_data['underperformers']) == 3

    def test_spy_missing_from_database(self, outperforming_portfolios):
        """Test graceful handling when SPY data not available."""
        # Mock loader without SPY (but with proper OHLCV data for AAPL)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        close_prices = np.linspace(100, 120, 100)

        mock_loader = MockDataLoader({
            'AAPL': pd.DataFrame({
                'open': close_prices + np.random.randn(100) * 0.1,
                'high': close_prices + np.abs(np.random.randn(100) * 0.2),
                'low': close_prices - np.abs(np.random.randn(100) * 0.2),
                'close': close_prices,
                'volume': np.random.randint(1000000, 5000000, 100)
            }, index=dates)
        })

        benchmark_data = BenchmarkCalculator.calculate_all_benchmarks(
            portfolios={'AAPL': outperforming_portfolios['AAPL']},
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-04-10',
            initial_capital=100000,
            data_loader=mock_loader,
            include_spy=True
        )

        # SPY should be None but per-symbol benchmarks should work
        assert benchmark_data['spy']['equity'] is None, "SPY should be None when unavailable"
        assert len(benchmark_data['per_symbol']) > 0, "Per-symbol benchmarks should still work"
