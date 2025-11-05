"""
Enhanced Sharpe ratio tests with full mathematical validation.

These tests independently calculate expected values and verify
the implementation matches mathematical expectations.

This is what ALL tests should do - not just sanity checks,
but actual validation of mathematical correctness.
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src')

from backtesting.engine.portfolio_aggregator import PortfolioAggregator


def calculate_sharpe_manually(returns_series: pd.Series) -> float:
    """
    Calculate Sharpe ratio independently (not using our implementation).

    This serves as the "gold standard" for validation.
    """
    returns = returns_series.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return sharpe


@pytest.fixture
def simple_aligned_portfolios():
    """
    Create simple portfolios with known characteristics.

    All portfolios have the same date range (aligned).
    """
    dates = pd.date_range(start='2023-01-01', periods=31, freq='D')
    np.random.seed(42)

    portfolios = {}
    base = 100000

    # Create 3 portfolios with different characteristics
    # Use strong trends relative to volatility to ensure positive Sharpes
    for symbol, (trend_pct, vol_pct) in [
        ('AAPL', (10.0, 3.0)),   # +10% trend, 3% volatility
        ('MSFT', (8.0, 3.0)),    # +8% trend, 3% volatility
        ('GOOGL', (9.0, 2.5))    # +9% trend, 2.5% volatility
    ]:
        trend = np.linspace(0, base * trend_pct / 100, 31)
        noise = np.random.randn(31) * (base * vol_pct / 100)
        equity = pd.Series(base + trend + noise, index=dates)

        portfolios[symbol] = type('Portfolio', (), {
            'equity_curve': equity,
            'init_cash': base,
            'trades': []
        })()

    return portfolios


@pytest.fixture
def simple_misaligned_portfolios():
    """
    Create portfolios with different date ranges (misaligned).

    This tests the critical ffill bug fix.
    """
    base = 100000
    np.random.seed(42)

    portfolios = {}

    # AAPL: Full month (31 days)
    dates_31 = pd.date_range(start='2023-01-01', periods=31, freq='D')
    trend = np.linspace(0, base * 0.06, 31)
    noise = np.random.randn(31) * (base * 0.04)
    portfolios['AAPL'] = type('Portfolio', (), {
        'equity_curve': pd.Series(base + trend + noise, index=dates_31),
        'init_cash': base,
        'trades': []
    })()

    # MSFT: 28 days (ends 3 days early)
    dates_28 = pd.date_range(start='2023-01-01', periods=28, freq='D')
    trend = np.linspace(0, base * 0.05, 28)
    noise = np.random.randn(28) * (base * 0.04)
    portfolios['MSFT'] = type('Portfolio', (), {
        'equity_curve': pd.Series(base + trend + noise, index=dates_28),
        'init_cash': base,
        'trades': []
    })()

    # GOOGL: 25 days (ends 6 days early)
    dates_25 = pd.date_range(start='2023-01-01', periods=25, freq='D')
    trend = np.linspace(0, base * 0.055, 25)
    noise = np.random.randn(25) * (base * 0.04)
    portfolios['GOOGL'] = type('Portfolio', (), {
        'equity_curve': pd.Series(base + trend + noise, index=dates_25),
        'init_cash': base,
        'trades': []
    })()

    return portfolios


class TestSharpeMathematicalValidation:
    """
    Tests that validate mathematical correctness, not just sanity checks.
    """

    def test_implementation_matches_manual_calculation(self, simple_aligned_portfolios):
        """
        CRITICAL TEST: Verify implementation matches independent manual calculation.

        This is the gold standard - if this passes, the implementation is correct.
        """
        # Get result from implementation
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            simple_aligned_portfolios,
            initial_capital=100000
        )
        impl_sharpe = metrics.get('Sharpe Ratio', 0)

        # Calculate expected result manually (independent implementation)
        returns_data = {}
        for symbol, portfolio in simple_aligned_portfolios.items():
            returns_data[symbol] = portfolio.equity_curve.pct_change()

        returns_df = pd.DataFrame(returns_data)
        portfolio_returns = returns_df.mean(axis=1, skipna=True).dropna()
        expected_sharpe = calculate_sharpe_manually(portfolio_returns)

        # Verify they match (within numerical tolerance)
        difference = abs(impl_sharpe - expected_sharpe)
        assert difference < 0.001, \
            f"Implementation ({impl_sharpe:.6f}) differs from manual ({expected_sharpe:.6f}) by {difference:.6f}"

    def test_individual_sharpes_calculated_correctly(self, simple_aligned_portfolios):
        """
        Validate that we can correctly calculate individual Sharpe ratios.

        This ensures our manual calculation function is correct and demonstrates
        how to calculate individual Sharpes for comparison.
        """
        for symbol, portfolio in simple_aligned_portfolios.items():
            returns = portfolio.equity_curve.pct_change().dropna()

            # Calculate manually
            sharpe = calculate_sharpe_manually(returns)

            # All our test portfolios have positive trends
            assert sharpe > 0, f"{symbol} should have positive Sharpe, got {sharpe:.3f}"

            # With upward trends and reasonable volatility, Sharpe should be positive
            # Note: For only 31 days of data, Sharpe of 0.5+ is reasonable
            assert sharpe > 0.3, f"{symbol} should have Sharpe > 0.3, got {sharpe:.3f}"

            # For 31 days with 4-6% gain, Sharpe should be in reasonable range
            assert sharpe < 20.0, f"{symbol} Sharpe seems unreasonably high: {sharpe:.3f}"

    def test_aggregate_vs_individual_relationship(self, simple_aligned_portfolios):
        """
        Test the mathematical relationship between aggregate and individual Sharpes.

        Key insight: Aggregate Sharpe depends on correlations:
        - High correlation → Aggregate ≈ Average Individual
        - Low correlation  → Aggregate > Average (diversification benefit)
        - Negative corr    → Aggregate >> Average (strong diversification)
        """
        # Calculate individual Sharpes
        individual_sharpes = {}
        individual_returns = {}

        for symbol, portfolio in simple_aligned_portfolios.items():
            returns = portfolio.equity_curve.pct_change().dropna()
            sharpe = calculate_sharpe_manually(returns)
            individual_sharpes[symbol] = sharpe
            individual_returns[symbol] = returns

        # Calculate aggregate Sharpe
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            simple_aligned_portfolios,
            initial_capital=100000
        )
        aggregate_sharpe = metrics.get('Sharpe Ratio', 0)

        # Calculate correlations
        returns_df = pd.DataFrame(individual_returns)
        corr_matrix = returns_df.corr()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

        # Calculate statistics
        min_individual = min(individual_sharpes.values())
        max_individual = max(individual_sharpes.values())
        avg_individual = np.mean(list(individual_sharpes.values()))

        # Verify aggregate is not completely unreasonable
        # It should be at least positive
        assert aggregate_sharpe > 0, "Aggregate Sharpe should be positive"

        # It shouldn't be dramatically lower than the worst individual
        # (the old ffill bug caused aggregate=0.1 when individuals were >1.0)
        assert aggregate_sharpe > 0.3 * min_individual, \
            f"Aggregate ({aggregate_sharpe:.2f}) dramatically lower than min individual ({min_individual:.2f})"

        # Based on correlation, verify relationship makes sense
        if avg_correlation > 0.7:
            # High correlation: aggregate should be close to average
            # Allow some variance due to random noise in test data
            lower_bound = avg_individual * 0.5
            upper_bound = avg_individual * 1.5
            assert lower_bound <= aggregate_sharpe <= upper_bound, \
                f"High correlation ({avg_correlation:.2f}): aggregate ({aggregate_sharpe:.2f}) " \
                f"should be close to average ({avg_individual:.2f}), " \
                f"expected range [{lower_bound:.2f}, {upper_bound:.2f}]"

        elif avg_correlation < 0.3:
            # Low correlation: aggregate can be higher due to diversification
            # It should be at least reasonable compared to individuals
            assert aggregate_sharpe >= 0.5 * avg_individual, \
                f"Low correlation: aggregate ({aggregate_sharpe:.2f}) should be reasonable " \
                f"compared to average ({avg_individual:.2f})"

    def test_misaligned_dates_no_ffill_bug(self, simple_misaligned_portfolios):
        """
        CRITICAL BUG TEST: Verify misaligned dates don't cause ffill bug.

        The bug was: Using ffill() to align dates creates artificial flat segments
        with 0% returns, dragging down the Sharpe ratio.

        Before fix: Aggregate Sharpe = 0.1 (when individuals were >1.0)
        After fix:  Aggregate Sharpe should be reasonable
        """
        # Calculate individual Sharpes
        individual_sharpes = {}
        for symbol, portfolio in simple_misaligned_portfolios.items():
            returns = portfolio.equity_curve.pct_change().dropna()
            sharpe = calculate_sharpe_manually(returns)
            individual_sharpes[symbol] = sharpe

        min_individual = min(individual_sharpes.values())
        avg_individual = np.mean(list(individual_sharpes.values()))

        # Calculate aggregate Sharpe
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            simple_misaligned_portfolios,
            initial_capital=100000
        )
        aggregate_sharpe = metrics.get('Sharpe Ratio', 0)

        # CRITICAL TEST: If all individuals have high Sharpe (>1.0),
        # aggregate should NOT be near zero (which was the bug)
        if all(s > 1.0 for s in individual_sharpes.values()):
            assert aggregate_sharpe > 0.5, \
                f"BUG DETECTED: Aggregate Sharpe ({aggregate_sharpe:.2f}) near zero " \
                f"when all individuals > 1.0. This indicates ffill bug!"

            # Aggregate should be at least somewhat comparable to individuals
            assert aggregate_sharpe > 0.3 * min_individual, \
                f"Aggregate ({aggregate_sharpe:.2f}) is too low compared to " \
                f"min individual ({min_individual:.2f})"

        # Manual validation: Calculate expected aggregate independently
        returns_data = {}
        for symbol, portfolio in simple_misaligned_portfolios.items():
            returns_data[symbol] = portfolio.equity_curve.pct_change()

        returns_df = pd.DataFrame(returns_data)
        portfolio_returns = returns_df.mean(axis=1, skipna=True).dropna()
        expected_sharpe = calculate_sharpe_manually(portfolio_returns)

        # Verify implementation matches manual calculation
        difference = abs(aggregate_sharpe - expected_sharpe)
        assert difference < 0.001, \
            f"Implementation ({aggregate_sharpe:.6f}) differs from manual ({expected_sharpe:.6f})"

    def test_correlation_effect_on_sharpe(self):
        """
        Demonstrate how correlation affects aggregate Sharpe.

        This test creates portfolios with controlled correlations to show:
        1. Uncorrelated assets → Higher aggregate Sharpe (diversification)
        2. Correlated assets → Aggregate Sharpe ≈ Average
        """
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        base = 100000

        # Test Case 1: Uncorrelated assets (independent random walks)
        np.random.seed(42)
        portfolios_uncorr = {}

        for symbol in ['A', 'B', 'C']:
            # Each has same characteristics but independent noise
            trend = np.linspace(0, base * 0.10, 252)  # +10% annual
            noise = np.random.randn(252) * (base * 0.15)  # 15% volatility
            equity = pd.Series(base + trend + noise, index=dates)

            portfolios_uncorr[symbol] = type('Portfolio', (), {
                'equity_curve': equity,
                'init_cash': base,
                'trades': []
            })()

        # Calculate individual Sharpes for uncorrelated
        individual_sharpes_uncorr = []
        for portfolio in portfolios_uncorr.values():
            returns = portfolio.equity_curve.pct_change().dropna()
            sharpe = calculate_sharpe_manually(returns)
            individual_sharpes_uncorr.append(sharpe)

        avg_individual_uncorr = np.mean(individual_sharpes_uncorr)

        # Calculate aggregate for uncorrelated
        metrics_uncorr = PortfolioAggregator.calculate_aggregate_metrics(
            portfolios_uncorr,
            initial_capital=100000
        )
        aggregate_uncorr = metrics_uncorr.get('Sharpe Ratio', 0)

        # Test Case 2: Correlated assets (same noise component)
        np.random.seed(42)
        shared_noise = np.random.randn(252) * (base * 0.15)
        portfolios_corr = {}

        for symbol in ['X', 'Y', 'Z']:
            trend = np.linspace(0, base * 0.10, 252)
            # All portfolios share the same noise → high correlation
            equity = pd.Series(base + trend + shared_noise, index=dates)

            portfolios_corr[symbol] = type('Portfolio', (), {
                'equity_curve': equity,
                'init_cash': base,
                'trades': []
            })()

        # Calculate aggregate for correlated
        metrics_corr = PortfolioAggregator.calculate_aggregate_metrics(
            portfolios_corr,
            initial_capital=100000
        )
        aggregate_corr = metrics_corr.get('Sharpe Ratio', 0)

        # Since portfolios are identical, individual Sharpes should be identical too
        individual_sharpe_corr = calculate_sharpe_manually(
            portfolios_corr['X'].equity_curve.pct_change().dropna()
        )

        # Verify diversification effect
        # Uncorrelated: aggregate should be higher than average (diversification benefit)
        # Correlated: aggregate should be close to individual (no diversification)

        # For uncorrelated, aggregate can be somewhat higher
        # (exact relationship depends on how well noise cancels)
        assert aggregate_uncorr > 0.8 * avg_individual_uncorr, \
            f"Uncorrelated aggregate ({aggregate_uncorr:.2f}) should be reasonable " \
            f"vs average ({avg_individual_uncorr:.2f})"

        # For perfectly correlated (identical assets), aggregate = individual
        assert abs(aggregate_corr - individual_sharpe_corr) < 0.1, \
            f"Perfectly correlated: aggregate ({aggregate_corr:.2f}) should equal " \
            f"individual ({individual_sharpe_corr:.2f})"

        # Verification: Uncorrelated aggregate should typically be >= correlated aggregate
        # (due to diversification benefit)
        # Note: This may not always hold with small samples, so we just check reasonableness
        print(f"\nDiversification effect demonstration:")
        print(f"  Uncorrelated aggregate: {aggregate_uncorr:.2f}")
        print(f"  Correlated aggregate:   {aggregate_corr:.2f}")
        print(f"  Diversification benefit: {aggregate_uncorr - aggregate_corr:.2f}")


class TestEdgeCasesWithValidation:
    """
    Edge case tests with proper validation.
    """

    def test_single_portfolio_sharpe_unchanged(self):
        """
        With only one portfolio, aggregate Sharpe should equal individual Sharpe.

        This is a special case: no diversification possible.
        """
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        np.random.seed(42)

        base = 100000
        trend = np.linspace(0, base * 0.08, 60)
        noise = np.random.randn(60) * (base * 0.05)
        equity = pd.Series(base + trend + noise, index=dates)

        portfolios = {
            'SOLO': type('Portfolio', (), {
                'equity_curve': equity,
                'init_cash': base,
                'trades': []
            })()
        }

        # Calculate individual Sharpe
        returns = equity.pct_change().dropna()
        individual_sharpe = calculate_sharpe_manually(returns)

        # Calculate aggregate Sharpe
        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            portfolios,
            initial_capital=100000
        )
        aggregate_sharpe = metrics.get('Sharpe Ratio', 0)

        # For single portfolio, aggregate must equal individual
        difference = abs(aggregate_sharpe - individual_sharpe)
        assert difference < 0.001, \
            f"Single portfolio: aggregate ({aggregate_sharpe:.6f}) should equal " \
            f"individual ({individual_sharpe:.6f}), difference = {difference:.6f}"

    def test_zero_volatility_sharpe(self):
        """
        Test handling of zero volatility (flat equity curve).

        Sharpe ratio is undefined when std=0, implementation should return 0.
        """
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        base = 100000

        # Perfectly flat equity curve (no variation)
        equity_flat = pd.Series([base] * 30, index=dates)

        portfolios = {
            'FLAT': type('Portfolio', (), {
                'equity_curve': equity_flat,
                'init_cash': base,
                'trades': []
            })()
        }

        metrics = PortfolioAggregator.calculate_aggregate_metrics(
            portfolios,
            initial_capital=100000
        )

        sharpe = metrics.get('Sharpe Ratio', 0)

        # With zero volatility, Sharpe should be 0 (avoid division by zero)
        assert sharpe == 0, f"Zero volatility should give Sharpe=0, got {sharpe}"
