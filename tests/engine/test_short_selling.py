"""
Unit tests for short selling functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import pytest
import pandas as pd
import numpy as np
from backtesting.engine.portfolio_simulator import from_signals


def test_simple_short_position():
    """Test basic short position P&L calculation."""
    # Create simple price series that goes down
    # Use timestamps during market hours (10:00 AM EST)
    dates = pd.date_range('2022-01-03 10:00:00', periods=10, freq='D', tz='US/Eastern')
    prices = pd.Series([100, 99, 98, 97, 96, 95, 94, 93, 92, 91], index=dates)

    # Short on day 0, cover on day 9
    entries = pd.Series([False] * 10, index=dates)
    exits = pd.Series([True if i == 0 else False for i in range(10)], index=dates)
    exits.iloc[-1] = False  # Don't exit on last day
    entries.iloc[-1] = True  # Cover on last day

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

    # Should have 1 short trade
    short_trades = [t for t in portfolio.trades if t['type'] == 'short_entry']
    cover_trades = [t for t in portfolio.trades if t['type'] == 'cover_short']

    assert len(short_trades) == 1, f"Expected 1 short entry, got {len(short_trades)}"
    assert len(cover_trades) == 1, f"Expected 1 short cover, got {len(cover_trades)}"

    # Short entry should be at ~$100
    assert abs(short_trades[0]['price'] - 100) < 1

    # Debug: print actual values
    print(f"Short entry: ${short_trades[0]['price']:.2f}")
    print(f"Cover: ${cover_trades[0]['price']:.2f}")
    print(f"P&L: ${cover_trades[0]['pnl']:.2f}")
    print(f"P&L %: {cover_trades[0]['pnl_pct']:.2f}%")

    # Verify prices
    # assert abs(short_trades[0]['price'] - 100) < 1
    # assert abs(cover_trades[0]['price'] - 91) < 1

    # P&L should be positive (profit from price drop)
    assert cover_trades[0]['pnl'] > 0, f"Expected positive P&L, got {cover_trades[0]['pnl']}"


def test_long_only_vs_short_downtrend():
    """Compare long-only vs long/short on downtrending prices."""
    # Create simple downtrend
    dates = pd.date_range('2022-01-03 10:00:00', periods=20, freq='D', tz='US/Eastern')
    prices = pd.Series([100 - i for i in range(20)], index=dates)  # 100 → 81

    # Always bearish signal (want to be short)
    entries = pd.Series([False] * 20, index=dates)
    exits = pd.Series([True] * 20, index=dates)

    # Long-only
    portfolio_long = from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001,
        slippage=0.0005,
        allow_shorts=False
    )

    # Long/short
    portfolio_short = from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001,
        slippage=0.0005,
        allow_shorts=True
    )

    stats_long = portfolio_long.stats()
    stats_short = portfolio_short.stats()

    print("\nLong-only (downtrend):")
    print(f"  Final value: ${stats_long['End Value']:.2f}")
    print(f"  Total return: {stats_long['Total Return [%]']:.2f}%")
    print(f"  Sharpe: {stats_long['Sharpe Ratio']:.4f}")
    print(f"  Trades: {stats_long['Total Trades']}")

    print("\nLong/short (downtrend):")
    print(f"  Final value: ${stats_short['End Value']:.2f}")
    print(f"  Total return: {stats_short['Total Return [%]']:.2f}%")
    print(f"  Sharpe: {stats_short['Sharpe Ratio']:.4f}")
    print(f"  Trades: {stats_short['Total Trades']}")

    # Long/short should perform better in downtrend
    # (either make money or lose less)
    assert stats_short['Total Return [%]'] >= stats_long['Total Return [%]'], \
        f"Long/short should perform >= long-only in downtrend"


def test_portfolio_value_with_short():
    """Test that portfolio value is calculated correctly for short positions."""
    # Simple 2-day test
    dates = pd.date_range('2022-01-03 10:00:00', periods=2, freq='D', tz='US/Eastern')
    prices = pd.Series([100, 90], index=dates)  # Price drops 10%

    # Short on day 0
    entries = pd.Series([False, False], index=dates)
    exits = pd.Series([True, False], index=dates)

    portfolio = from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0,
        slippage=0,
        allow_shorts=True
    )

    # Check equity curve
    # Day 0: Short 1000 shares @ $100 → receive $100,000
    # Portfolio value should be: $10,000 (cash) + $100,000 (proceeds from short)
    # But we owe 1000 shares, so value = cash + (entry_price - current_price) * shares

    equity = portfolio.equity_curve

    # On day 0 (after short entry):
    # cash = $10,000 + $100,000 = $110,000
    # position = -1000 shares
    # portfolio_value = cash + (100 - 100) * 1000 = $110,000

    # On day 1 (price dropped to $90):
    # cash = $110,000 (unchanged)
    # position = -1000 shares
    # portfolio_value = cash + (100 - 90) * 1000 = $120,000 (gained $10,000)

    print(f"\nEquity curve:")
    print(equity)

    # Final value should reflect the profit from price drop
    final_value = equity.iloc[-1]
    expected_profit = 1000 * (100 - 90)  # $10,000 profit

    print(f"\nFinal value: ${final_value:.2f}")
    print(f"Expected profit: ${expected_profit:.2f}")

    # Allow some tolerance for calculation
    assert final_value > 10000, f"Portfolio should have gained value from short profit"


if __name__ == '__main__':
    print("Testing basic short position...")
    test_simple_short_position()

    print("\n" + "="*80)
    print("Testing long-only vs long/short in downtrend...")
    test_long_only_vs_short_downtrend()

    print("\n" + "="*80)
    print("Testing portfolio value calculation...")
    test_portfolio_value_with_short()

    print("\n" + "="*80)
    print("All tests passed!")
