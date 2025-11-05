"""
Independent validation of Sharpe ratio calculations.

This script manually calculates:
1. Individual Sharpe ratios for each symbol
2. Aggregate portfolio Sharpe ratio (equal-weighted)
3. Compares with PortfolioAggregator results

This validates the mathematical correctness of our implementation.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src')

from backtesting.engine.portfolio_aggregator import PortfolioAggregator


def calculate_sharpe(returns_series: pd.Series) -> float:
    """
    Calculate annualized Sharpe ratio from daily returns.

    Formula: (mean_return / std_return) * sqrt(252)
    """
    returns = returns_series.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return sharpe


def create_test_portfolios():
    """Create simple test portfolios with known characteristics."""
    dates = pd.date_range(start='2023-01-01', periods=31, freq='D')
    np.random.seed(42)

    portfolios = {}

    # AAPL: Strong uptrend, moderate volatility
    base = 100000
    trend = np.linspace(0, 6000, 31)  # +6% over 31 days
    noise = np.random.randn(31) * 400
    equity_aapl = pd.Series(base + trend + noise, index=dates)

    portfolios['AAPL'] = type('Portfolio', (), {
        'equity_curve': equity_aapl,
        'init_cash': base,
        'trades': []
    })()

    # MSFT: Moderate uptrend, moderate volatility
    trend = np.linspace(0, 4000, 31)  # +4% over 31 days
    noise = np.random.randn(31) * 500
    equity_msft = pd.Series(base + trend + noise, index=dates)

    portfolios['MSFT'] = type('Portfolio', (), {
        'equity_curve': equity_msft,
        'init_cash': base,
        'trades': []
    })()

    # GOOGL: Strong uptrend, low volatility
    trend = np.linspace(0, 5000, 31)  # +5% over 31 days
    noise = np.random.randn(31) * 300
    equity_googl = pd.Series(base + trend + noise, index=dates)

    portfolios['GOOGL'] = type('Portfolio', (), {
        'equity_curve': equity_googl,
        'init_cash': base,
        'trades': []
    })()

    return portfolios


def validate_sharpe_calculations():
    """
    Validate that our Sharpe calculations are mathematically correct.
    """
    print("=" * 80)
    print("SHARPE RATIO VALIDATION - Independent Calculation")
    print("=" * 80)
    print()

    portfolios = create_test_portfolios()

    # Step 1: Calculate individual Sharpe ratios manually
    print("STEP 1: Individual Sharpe Ratios (calculated independently)")
    print("-" * 80)

    individual_sharpes = {}
    individual_returns = {}

    for symbol, portfolio in portfolios.items():
        equity = portfolio.equity_curve
        returns = equity.pct_change().dropna()
        sharpe = calculate_sharpe(returns)

        individual_sharpes[symbol] = sharpe
        individual_returns[symbol] = returns

        print(f"{symbol:6s}: Sharpe = {sharpe:6.3f}  "
              f"(Mean = {returns.mean()*100:6.3f}%, Std = {returns.std()*100:6.3f}%)")

    print()
    print(f"Average Individual Sharpe: {np.mean(list(individual_sharpes.values())):.3f}")
    print(f"Min Individual Sharpe: {np.min(list(individual_sharpes.values())):.3f}")
    print(f"Max Individual Sharpe: {np.max(list(individual_sharpes.values())):.3f}")
    print()

    # Step 2: Calculate aggregate Sharpe manually (equal-weighted portfolio)
    print("STEP 2: Aggregate Portfolio Sharpe (calculated independently)")
    print("-" * 80)

    # Method: Equal-weighted average of returns
    returns_df = pd.DataFrame(individual_returns)
    portfolio_returns = returns_df.mean(axis=1)  # Equal-weighted

    manual_aggregate_sharpe = calculate_sharpe(portfolio_returns)

    print(f"Portfolio Returns Mean: {portfolio_returns.mean()*100:.4f}%")
    print(f"Portfolio Returns Std:  {portfolio_returns.std()*100:.4f}%")
    print(f"Manual Aggregate Sharpe: {manual_aggregate_sharpe:.3f}")
    print()

    # Step 3: Get aggregate Sharpe from PortfolioAggregator
    print("STEP 3: PortfolioAggregator Result (from our implementation)")
    print("-" * 80)

    metrics = PortfolioAggregator.calculate_aggregate_metrics(
        portfolios,
        initial_capital=100000
    )

    implementation_sharpe = metrics.get('Sharpe Ratio', 0)

    print(f"Implementation Aggregate Sharpe: {implementation_sharpe:.3f}")
    print()

    # Step 4: Compare and validate
    print("STEP 4: Validation")
    print("-" * 80)

    sharpe_diff = abs(manual_aggregate_sharpe - implementation_sharpe)

    print(f"Manual Calculation:         {manual_aggregate_sharpe:.6f}")
    print(f"Implementation Calculation: {implementation_sharpe:.6f}")
    print(f"Difference:                 {sharpe_diff:.6f}")
    print()

    if sharpe_diff < 0.001:
        print("✓ PASS: Implementation matches manual calculation (within 0.001)")
    else:
        print("✗ FAIL: Implementation differs from manual calculation!")
    print()

    # Step 5: Verify relationship between individual and aggregate Sharpes
    print("STEP 5: Relationship Analysis")
    print("-" * 80)

    avg_individual = np.mean(list(individual_sharpes.values()))

    print(f"Average Individual Sharpe:  {avg_individual:.3f}")
    print(f"Aggregate Portfolio Sharpe: {implementation_sharpe:.3f}")
    print(f"Difference:                 {implementation_sharpe - avg_individual:.3f}")
    print()

    # Calculate correlations between symbols
    print("Correlation Matrix:")
    corr_matrix = returns_df.corr()
    print(corr_matrix)
    print()

    avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    print(f"Average Pairwise Correlation: {avg_correlation:.3f}")
    print()

    print("INTERPRETATION:")
    print("-" * 80)
    print("For an equal-weighted portfolio:")
    print("• If correlations are HIGH (+1): Portfolio Sharpe ≈ Average Individual Sharpe")
    print("• If correlations are LOW (0):   Portfolio Sharpe > Average (diversification benefit)")
    print("• If correlations are NEGATIVE:  Portfolio Sharpe >> Average (strong diversification)")
    print()
    print(f"In this case: Correlation = {avg_correlation:.2f}")

    if avg_correlation > 0.7:
        print("→ High correlation: Portfolio Sharpe should be close to average individual")
        expected_range = (avg_individual * 0.8, avg_individual * 1.2)
    elif avg_correlation > 0.3:
        print("→ Moderate correlation: Portfolio Sharpe could be slightly higher than average")
        expected_range = (avg_individual * 0.8, avg_individual * 1.4)
    else:
        print("→ Low correlation: Portfolio Sharpe could be significantly higher (diversification)")
        expected_range = (avg_individual * 0.8, avg_individual * 2.0)

    print(f"Expected range: {expected_range[0]:.2f} to {expected_range[1]:.2f}")

    if expected_range[0] <= implementation_sharpe <= expected_range[1]:
        print(f"✓ PASS: Aggregate Sharpe ({implementation_sharpe:.2f}) is within expected range")
    else:
        print(f"✗ WARNING: Aggregate Sharpe ({implementation_sharpe:.2f}) is outside expected range")

    print()
    print("=" * 80)
    print()

    return {
        'individual_sharpes': individual_sharpes,
        'manual_aggregate': manual_aggregate_sharpe,
        'implementation_aggregate': implementation_sharpe,
        'difference': sharpe_diff,
        'correlations': corr_matrix,
        'passes': sharpe_diff < 0.001
    }


def validate_misaligned_sharpe():
    """
    Validate the critical bug fix: misaligned date ranges.
    """
    print("=" * 80)
    print("CRITICAL BUG VALIDATION - Misaligned Date Ranges")
    print("=" * 80)
    print()
    print("This tests the bug where aggregate Sharpe was 0.1 even though")
    print("all individual Sharpes were >1.0 due to ffill creating fake flat days.")
    print()

    # Create misaligned portfolios
    dates_full = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    dates_28 = pd.date_range(start='2023-01-01', end='2023-01-28', freq='D')
    dates_25 = pd.date_range(start='2023-01-01', end='2023-01-25', freq='D')

    np.random.seed(42)
    base = 100000

    portfolios = {}

    # AAPL: Full month (31 days)
    trend = np.linspace(0, 6000, len(dates_full))
    noise = np.random.randn(len(dates_full)) * 400
    portfolios['AAPL'] = type('Portfolio', (), {
        'equity_curve': pd.Series(base + trend + noise, index=dates_full),
        'init_cash': base,
        'trades': []
    })()

    # MSFT: Ends 3 days early (28 days)
    trend = np.linspace(0, 5000, len(dates_28))
    noise = np.random.randn(len(dates_28)) * 400
    portfolios['MSFT'] = type('Portfolio', (), {
        'equity_curve': pd.Series(base + trend + noise, index=dates_28),
        'init_cash': base,
        'trades': []
    })()

    # GOOGL: Ends 6 days early (25 days)
    trend = np.linspace(0, 5500, len(dates_25))
    noise = np.random.randn(len(dates_25)) * 400
    portfolios['GOOGL'] = type('Portfolio', (), {
        'equity_curve': pd.Series(base + trend + noise, index=dates_25),
        'init_cash': base,
        'trades': []
    })()

    # Calculate individual Sharpes
    print("Individual Sharpe Ratios:")
    print("-" * 80)

    individual_sharpes = {}
    for symbol, portfolio in portfolios.items():
        equity = portfolio.equity_curve
        returns = equity.pct_change().dropna()
        sharpe = calculate_sharpe(returns)
        individual_sharpes[symbol] = sharpe

        print(f"{symbol:6s}: Sharpe = {sharpe:.3f}  (Data points: {len(equity)})")

    min_individual = np.min(list(individual_sharpes.values()))
    avg_individual = np.mean(list(individual_sharpes.values()))

    print()
    print(f"Minimum Individual Sharpe: {min_individual:.3f}")
    print(f"Average Individual Sharpe: {avg_individual:.3f}")
    print()

    # Calculate aggregate Sharpe
    print("Aggregate Portfolio Sharpe:")
    print("-" * 80)

    metrics = PortfolioAggregator.calculate_aggregate_metrics(
        portfolios,
        initial_capital=100000
    )

    aggregate_sharpe = metrics.get('Sharpe Ratio', 0)

    print(f"Aggregate Sharpe: {aggregate_sharpe:.3f}")
    print()

    # Validation
    print("VALIDATION:")
    print("-" * 80)

    # The bug was: aggregate Sharpe dropped to 0.1 when individuals were >1.0
    # This was because ffill created artificial flat days with 0% returns

    bug_detected = aggregate_sharpe < 0.5 and min_individual > 1.0

    if bug_detected:
        print("✗ CRITICAL BUG DETECTED!")
        print(f"  Aggregate Sharpe ({aggregate_sharpe:.2f}) << Individual Sharpes (min {min_individual:.2f})")
        print("  This indicates the ffill bug is present!")
    else:
        print("✓ PASS: No ffill bug detected")
        print(f"  Aggregate Sharpe ({aggregate_sharpe:.2f}) is reasonable compared to individuals")

    # Additional checks
    reasonable_aggregate = aggregate_sharpe > 0.5 * min_individual

    print()
    print(f"Check 1: Aggregate > 0.5 * min(individuals)? "
          f"{aggregate_sharpe:.2f} > {0.5*min_individual:.2f} = {reasonable_aggregate}")

    if all(individual_sharpes.values()) and min_individual > 1.0:
        should_be_high = aggregate_sharpe > 1.0
        print(f"Check 2: If all individuals > 1.0, aggregate > 1.0? "
              f"{aggregate_sharpe:.2f} > 1.0 = {should_be_high}")

    print()
    print("=" * 80)
    print()

    return {
        'individual_sharpes': individual_sharpes,
        'aggregate_sharpe': aggregate_sharpe,
        'bug_detected': bug_detected,
        'reasonable': reasonable_aggregate
    }


if __name__ == '__main__':
    print("\n")
    print("=" * 80)
    print(" " * 20 + "SHARPE RATIO LOGIC VALIDATION")
    print("=" * 80)
    print()

    # Test 1: Aligned portfolios
    result1 = validate_sharpe_calculations()

    # Test 2: Misaligned portfolios (critical bug test)
    result2 = validate_misaligned_sharpe()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if result1['passes']:
        print("✓ Test 1 PASSED: Implementation correctly calculates aggregate Sharpe")
    else:
        print("✗ Test 1 FAILED: Implementation differs from manual calculation")

    print()

    if not result2['bug_detected'] and result2['reasonable']:
        print("✓ Test 2 PASSED: Misaligned dates handled correctly (no ffill bug)")
    else:
        print("✗ Test 2 FAILED: Misaligned dates show ffill bug")

    print()
    print("=" * 80)
    print()
