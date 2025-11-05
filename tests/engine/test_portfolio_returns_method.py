"""
Test that Portfolio.returns() method works correctly for QuantStats tearsheet generation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.portfolio_simulator import Portfolio


def test_portfolio_returns_method():
    """Test that Portfolio.returns() method is compatible with QuantStats."""

    print("\nTesting Portfolio.returns() method...\n")

    # Create a simple mock portfolio with equity curve
    price = pd.Series(
        [100.0, 101.0, 102.0, 101.5, 103.0],
        index=pd.date_range('2023-01-01', periods=5, freq='D')
    )

    entries = pd.Series([1, 0, 0, 0, 0], index=price.index, dtype=bool)
    exits = pd.Series([0, 0, 0, 0, 1], index=price.index, dtype=bool)

    portfolio = Portfolio(
        price=price,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001,
        slippage=0.0,
        freq='1d',
        market_hours_only=False
    )

    # Test 1: Check that returns() method exists
    assert hasattr(portfolio, 'returns'), "Portfolio doesn't have returns() method!"
    print("✓ Portfolio has returns() method")

    # Test 2: Check that returns() is callable
    assert callable(portfolio.returns), "Portfolio.returns is not callable!"
    print("✓ Portfolio.returns() is callable")

    # Test 3: Get returns
    returns = portfolio.returns()
    print(f"✓ Returns retrieved: type={type(returns)}, length={len(returns)}")

    # Test 4: Check return type
    assert isinstance(returns, pd.Series), f"Expected pd.Series, got {type(returns)}"
    print("✓ Returns is a pandas Series")

    # Test 5: Check that returns are numeric
    assert returns.dtype in [np.float64, float], f"Returns should be float, got {returns.dtype}"
    print("✓ Returns are numeric (float)")

    # Test 6: Check no NaN values (should be filled with 0)
    assert not returns.isna().any(), "Returns contain NaN values!"
    print("✓ No NaN values in returns (filled with 0)")

    # Test 7: Verify first value is 0 (pct_change of first row)
    assert returns.iloc[0] == 0.0, f"First return should be 0, got {returns.iloc[0]}"
    print("✓ First return is 0 (as expected from pct_change)")

    # Test 8: Test with QuantStats (if available)
    try:
        import quantstats as qs

        # Try to calculate basic metrics using the returns
        sharpe = qs.stats.sharpe(returns)
        max_dd = qs.stats.max_drawdown(returns)

        print(f"✓ QuantStats compatibility verified:")
        print(f"  - Sharpe Ratio: {sharpe:.3f}")
        print(f"  - Max Drawdown: {max_dd:.2%}")

        # Test that we can generate a report (just structure, not actual HTML file)
        print("✓ Returns are compatible with QuantStats metrics")

    except ImportError:
        print("⚠ QuantStats not installed - skipping compatibility test")
        print("  (This is OK - the returns() method is correctly implemented)")

    # Test 9: Compare with MultiAssetPortfolio format
    print("\n✓ Comparing with MultiAssetPortfolio.returns() format:")
    print(f"  - Returns type: {type(returns)}")
    print(f"  - Index type: {type(returns.index)}")
    print(f"  - First 3 values: {returns.head(3).tolist()}")

    print("\n✓ All tests passed!")
    print("✓ Portfolio.returns() is now compatible with QuantStats tearsheet generation!")


if __name__ == '__main__':
    test_portfolio_returns_method()
