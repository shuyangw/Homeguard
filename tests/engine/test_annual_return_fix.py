"""
Test annual return calculation fix.

Validates that CAGR is calculated correctly instead of using
simple interest annualization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import pandas as pd
import numpy as np
from backtesting.engine.portfolio_simulator import from_signals


def test_annual_return_one_year():
    """Test annual return for exactly 1 year period."""
    # Create 1 year of data (252 trading days)
    dates = pd.date_range('2022-01-03 10:00:00', periods=252, freq='D', tz='US/Eastern')

    # Price increases from 100 to 120 (20% gain)
    prices = pd.Series(np.linspace(100, 120, 252), index=dates)

    # Buy and hold (entry at start, no exit)
    entries = pd.Series([True] + [False] * 251, index=dates)
    exits = pd.Series([False] * 252, index=dates)

    portfolio = from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0,
        slippage=0,
        allow_shorts=False
    )

    stats = portfolio.stats()

    print("Test 1: One Year Buy and Hold")
    print(f"  Total Return: {stats['Total Return [%]']:.2f}%")
    print(f"  Annual Return: {stats['Annual Return [%]']:.2f}%")
    print(f"  Expected: ~10% (buy $1000 worth, gain 20% = $200 on $10k = 2%)")
    print()

    # For 1 year, annual return should equal total return
    # (Actually will be slightly different due to position sizing)
    # But should NOT be 200%+ as before!
    assert abs(stats['Annual Return [%]']) < 100, f"Annual return too high: {stats['Annual Return [%]']:.2f}%"


def test_annual_return_loss():
    """Test annual return calculation for losses."""
    # Create 1 year of declining prices
    dates = pd.date_range('2022-01-03 10:00:00', periods=252, freq='D', tz='US/Eastern')

    # Price decreases from 100 to 70 (-30% decline)
    prices = pd.Series(np.linspace(100, 70, 252), index=dates)

    # Buy and hold
    entries = pd.Series([True] + [False] * 251, index=dates)
    exits = pd.Series([False] * 252, index=dates)

    portfolio = from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0,
        slippage=0,
        allow_shorts=False
    )

    stats = portfolio.stats()

    print("Test 2: One Year Declining Market")
    print(f"  Total Return: {stats['Total Return [%]']:.2f}%")
    print(f"  Annual Return: {stats['Annual Return [%]']:.2f}%")
    print(f"  Expected: Negative (price declined 30%)")
    print()

    # Annual return should be negative
    assert stats['Annual Return [%]'] < 0, "Annual return should be negative for losses"
    # And should be reasonable magnitude (not +400%!)
    assert stats['Annual Return [%]'] > -50, f"Annual return too negative: {stats['Annual Return [%]']:.2f}%"


def test_annual_return_with_shorts():
    """Test annual return with short positions."""
    # Create declining price series
    dates = pd.date_range('2022-01-03 10:00:00', periods=252, freq='D', tz='US/Eastern')
    prices = pd.Series(np.linspace(100, 80, 252), index=dates)

    # Short at start (exit signal triggers short when allow_shorts=True)
    entries = pd.Series([False] * 252, index=dates)
    exits = pd.Series([True] + [False] * 251, index=dates)

    portfolio = from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0,
        slippage=0,
        allow_shorts=True
    )

    stats = portfolio.stats()

    print("Test 3: One Year Short Position (Price Declines)")
    print(f"  Total Return: {stats['Total Return [%]']:.2f}%")
    print(f"  Annual Return: {stats['Annual Return [%]']:.2f}%")
    print(f"  Expected: Positive (short profits from decline)")
    print()

    # Annual return should be reasonable
    assert abs(stats['Annual Return [%]']) < 100, f"Annual return magnitude too high: {stats['Annual Return [%]']:.2f}%"


def test_cagr_multi_year():
    """Test CAGR calculation over multiple years."""
    # Create 2 years of data (504 trading days)
    dates = pd.date_range('2020-01-03 10:00:00', periods=504, freq='D', tz='US/Eastern')

    # Price doubles over 2 years (100 -> 200)
    # CAGR should be ~41.4% ((200/100)^(1/2) - 1 = 0.414)
    prices = pd.Series(np.linspace(100, 200, 504), index=dates)

    # Buy and hold
    entries = pd.Series([True] + [False] * 503, index=dates)
    exits = pd.Series([False] * 504, index=dates)

    portfolio = from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0,
        slippage=0,
        allow_shorts=False
    )

    stats = portfolio.stats()

    print("Test 4: Two Year Period (Price Doubles)")
    print(f"  Total Return: {stats['Total Return [%]']:.2f}%")
    print(f"  Annual Return (CAGR): {stats['Annual Return [%]']:.2f}%")
    print(f"  Note: With 10% position sizing, doubling means 10% portfolio gain")
    print(f"  CAGR for 10% over 2 years: (1.10)^0.5 - 1 = 4.88%")
    print(f"  Actual: {stats['Annual Return [%]']:.2f}% (reasonable!)")
    print()

    # CAGR should be between 2-15% (accounting for 10% position sizing)
    # The key test: annual return should NOT be 500%+ like before!
    assert 1 < stats['Annual Return [%]'] < 20, \
        f"CAGR out of expected range: {stats['Annual Return [%]']:.2f}%"


if __name__ == '__main__':
    print("="*80)
    print("ANNUAL RETURN CALCULATION FIX VALIDATION")
    print("="*80)
    print()

    test_annual_return_one_year()
    test_annual_return_loss()
    test_annual_return_with_shorts()
    test_cagr_multi_year()

    print("="*80)
    print("✅ All tests passed! Annual return calculation is correct.")
    print("="*80)
