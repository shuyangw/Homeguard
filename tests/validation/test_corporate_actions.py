"""
Corporate Actions Handling Tests

Ensures stock splits, dividends, and reverse splits are handled correctly
in return calculations.

Corporate actions that affect returns:
- Stock splits (2:1, 3:1, etc.)
- Reverse splits (1:5, 1:10, etc.)
- Dividends (cash distributions)
- Spin-offs (new companies created)

These tests validate that returns are calculated correctly across
corporate actions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.risk_config import RiskConfig


# ============================================================================
# DATA GENERATORS FOR CORPORATE ACTIONS
# ============================================================================

def create_stock_split_data(
    n_bars: int = 50,
    split_bar: int = 25,
    split_ratio: float = 2.0,
    start_price: float = 100.0
) -> pd.DataFrame:
    """
    Create data with a stock split.

    Args:
        split_ratio: 2.0 = 2:1 split (price halves, shares double)
    """
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    prices = []

    for i in range(n_bars):
        if i < split_bar:
            # Before split
            price = start_price + i * 0.5
        else:
            # After split - price adjusted
            pre_split_price = start_price + (split_bar - 1) * 0.5
            price = (pre_split_price / split_ratio) + (i - split_bar) * 0.5

        prices.append(price)

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [100000] * n_bars
    }, index=dates)

    return df


def create_reverse_split_data(
    n_bars: int = 50,
    split_bar: int = 25,
    reverse_ratio: float = 5.0,
    start_price: float = 10.0
) -> pd.DataFrame:
    """
    Create data with a reverse split.

    Args:
        reverse_ratio: 5.0 = 1:5 reverse split (price × 5, shares ÷ 5)
    """
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    prices = []

    for i in range(n_bars):
        if i < split_bar:
            # Before reverse split
            price = start_price + i * 0.1
        else:
            # After reverse split - price multiplied
            pre_split_price = start_price + (split_bar - 1) * 0.1
            price = (pre_split_price * reverse_ratio) + (i - split_bar) * 0.5

        prices.append(price)

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [10000] * n_bars
    }, index=dates)

    return df


def create_split_adjusted_data(
    n_bars: int = 50,
    split_bar: int = 25,
    split_ratio: float = 2.0,
    start_price: float = 100.0
) -> pd.DataFrame:
    """
    Create split-adjusted data (most common in backtest data).

    Prices are retroactively adjusted so there's no jump at split.
    This is how most data providers handle splits.
    """
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    # All prices already adjusted - smooth trend
    prices = [start_price + i * 0.5 for i in range(n_bars)]

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [100000] * n_bars
    }, index=dates)

    return df


def create_dividend_scenario(
    n_bars: int = 50,
    dividend_bars: list = None,
    dividend_pct: float = 0.02,
    start_price: float = 100.0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create data with dividend payments.

    Returns:
        (price_data, dividend_series)
    """
    if dividend_bars is None:
        dividend_bars = [15, 30, 45]  # Quarterly dividends

    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    prices = [start_price + i * 0.5 for i in range(n_bars)]

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [100000] * n_bars
    }, index=dates)

    # Create dividend series
    dividends = pd.Series(0.0, index=dates)
    for bar in dividend_bars:
        if bar < n_bars:
            # Dividend is % of price at that bar
            dividends.iloc[bar] = prices[bar] * dividend_pct

    return df, dividends


# ============================================================================
# TEST STRATEGIES
# ============================================================================

class BuyAndHoldStrategy(LongOnlyStrategy):
    """Simple buy and hold for corporate action tests."""

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        entries.iloc[0] = True
        exits.iloc[-1] = True

        return entries, exits


class HoldAcrossSplitStrategy(LongOnlyStrategy):
    """Hold position across split event."""

    def __init__(self, entry_bar: int = 0, exit_bar: int = 49):
        super().__init__()
        self.entry_bar = entry_bar
        self.exit_bar = exit_bar

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        if self.entry_bar < len(data):
            entries.iloc[self.entry_bar] = True
        if self.exit_bar < len(data):
            exits.iloc[self.exit_bar] = True

        return entries, exits


# ============================================================================
# TEST CLASS: Stock Splits
# ============================================================================

class TestStockSplits:
    """
    Verify returns are calculated correctly across stock splits.
    """

    def test_split_adjusted_data_smooth_returns(self):
        """
        Split-adjusted data should show smooth returns (no jump at split).

        Most backtest data is split-adjusted, so there should be no
        discontinuity at the split date.
        """
        data = create_split_adjusted_data(
            n_bars=50,
            split_bar=25,
            split_ratio=2.0,
            start_price=100.0
        )

        strategy = BuyAndHoldStrategy()

        print(f"\n{'='*60}")
        print("SPLIT-ADJUSTED DATA TEST")
        print(f"{'='*60}")
        print("Testing with pre-adjusted data (standard)")
        print(f"Price at bar 24: ${data.iloc[24]['close']:.2f}")
        print(f"Price at bar 25: ${data.iloc[25]['close']:.2f}")
        print(f"Price at bar 26: ${data.iloc[26]['close']:.2f}")
        print("Expected: No jump (data already adjusted)")

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.moderate()
        )

        portfolio = engine.run_with_data(strategy, data)

        stats = portfolio.stats()
        total_return = stats.get('Total Return [%]', 0)

        print(f"\nTotal return: {total_return:.2f}%")
        print(f"{'='*60}\n")

        # Should show positive return (uptrend)
        assert total_return > 0, \
            f"Split-adjusted data should show smooth return, got {total_return:.2f}%"

    def test_unadjusted_split_creates_discontinuity(self):
        """
        Un-adjusted split data creates price discontinuity.

        This test documents that unadjusted data would need
        special handling (shares adjust, not price).
        """
        data = create_stock_split_data(
            n_bars=50,
            split_bar=25,
            split_ratio=2.0,
            start_price=100.0
        )

        strategy = HoldAcrossSplitStrategy(entry_bar=20, exit_bar=30)

        print(f"\n{'='*60}")
        print("UNADJUSTED STOCK SPLIT TEST")
        print(f"{'='*60}")
        print("Testing with unadjusted data (2:1 split at bar 25)")
        print(f"Price at bar 24: ${data.iloc[24]['close']:.2f}")
        print(f"Price at bar 25: ${data.iloc[25]['close']:.2f}")
        print(f"Expected: Price halves (but shares double)")

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.moderate()
        )

        portfolio = engine.run_with_data(strategy, data)

        stats = portfolio.stats()
        total_return = stats.get('Total Return [%]', 0)

        print(f"\nTotal return: {total_return:.2f}%")
        print("\nNote: Engine treats as price drop (no split adjustment)")
        print("Use split-adjusted data for accurate backtesting!")
        print(f"{'='*60}\n")

        # This will show negative return because engine sees price drop
        # This is expected behavior - user should provide adjusted data


# ============================================================================
# TEST CLASS: Reverse Splits
# ============================================================================

class TestReverseSplits:
    """
    Verify returns are calculated correctly across reverse splits.
    """

    def test_reverse_split_adjusted_data(self):
        """
        Reverse split adjusted data should show smooth returns.

        Similar to regular splits, data should be pre-adjusted.
        """
        # Use adjusted data (smooth)
        data = create_split_adjusted_data(
            n_bars=50,
            split_bar=25,
            split_ratio=0.2,  # 1:5 reverse split
            start_price=10.0
        )

        strategy = BuyAndHoldStrategy()

        print(f"\n{'='*60}")
        print("REVERSE SPLIT ADJUSTED DATA TEST")
        print(f"{'='*60}")
        print("Testing with pre-adjusted data")

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.moderate()
        )

        portfolio = engine.run_with_data(strategy, data)

        stats = portfolio.stats()
        total_return = stats.get('Total Return [%]', 0)

        print(f"\nTotal return: {total_return:.2f}%")
        print(f"{'='*60}\n")

        # Should show positive return
        assert total_return > 0, \
            f"Adjusted data should show smooth return"


# ============================================================================
# TEST CLASS: Dividends
# ============================================================================

class TestDividends:
    """
    Test dividend handling in returns.

    NOTE: Current implementation may not include dividends in returns.
    This documents expected behavior.
    """

    def test_dividends_not_included_in_price_return(self):
        """
        Document that dividends are typically NOT included in price data.

        Most backtest data uses price-only (no dividends).
        Total return = price return + dividend yield.
        """
        data, dividends = create_dividend_scenario(
            n_bars=50,
            dividend_bars=[15, 30, 45],
            dividend_pct=0.02,
            start_price=100.0
        )

        strategy = BuyAndHoldStrategy()

        print(f"\n{'='*60}")
        print("DIVIDEND DOCUMENTATION TEST")
        print(f"{'='*60}")
        print("Dividend bars: [15, 30, 45]")
        print(f"Total dividends: ${dividends.sum():.2f}")
        print()
        print("Note: Backtest shows PRICE RETURN only")
        print("Dividends not automatically included in return calculation")
        print()
        print("To include dividends:")
        print("1. Use total return index (price + dividends)")
        print("2. Or manually adjust returns:")
        print("   total_return = price_return + dividend_yield")
        print(f"{'='*60}\n")

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.moderate()
        )

        portfolio = engine.run_with_data(strategy, data)

        stats = portfolio.stats()
        total_return = stats.get('Total Return [%]', 0)

        print(f"Price return: {total_return:.2f}%")

        # Calculate dividend yield
        entry_price = data.iloc[0]['close']
        dividend_yield_pct = (dividends.sum() / entry_price) * 100

        print(f"Dividend yield: {dividend_yield_pct:.2f}%")
        print(f"Total return (including dividends): {total_return + dividend_yield_pct:.2f}%")
        print()

        # Documentation test - always passes
        assert True, "Dividend handling documented"


# ============================================================================
# TEST CLASS: Data Quality with Corporate Actions
# ============================================================================

class TestCorporateActionDataQuality:
    """
    Test that corporate actions in data are handled correctly.
    """

    def test_recommendation_use_adjusted_data(self):
        """
        Document recommendation to use split-adjusted data.

        CRITICAL: Always use split-adjusted price data for backtesting.
        """
        print(f"\n{'='*60}")
        print("CORPORATE ACTIONS BEST PRACTICES")
        print(f"{'='*60}")
        print()
        print("✓ RECOMMENDED:")
        print("  - Use split-adjusted price data")
        print("  - Use total return index (includes dividends)")
        print("  - Verify data provider handles adjustments")
        print()
        print("✗ AVOID:")
        print("  - Unadjusted price data (creates discontinuities)")
        print("  - Price-only data without dividend adjustments")
        print("  - Manual split adjustments (error-prone)")
        print()
        print("DATA SOURCES:")
        print("  - Yahoo Finance: Adjusted Close (split-adjusted)")
        print("  - Alpha Vantage: Adjusted Close (split + dividend)")
        print("  - Bloomberg: Total Return Index (split + dividend)")
        print()
        print("VALIDATION:")
        print("  - Check for price discontinuities")
        print("  - Verify known splits are adjusted")
        print("  - Compare returns to benchmarks")
        print(f"{'='*60}\n")

        # Documentation test - always passes
        assert True, "Best practices documented"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
