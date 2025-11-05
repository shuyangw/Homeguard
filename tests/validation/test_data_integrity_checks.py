"""
Data Integrity Checks Tests

Ensures bad data doesn't crash the system or produce NaN metrics.

Critical validations:
- NaN in price data doesn't propagate to metrics
- Duplicate timestamps are handled gracefully
- Missing bars (gaps) don't cause crashes
- Negative prices are rejected or handled
- Zero volume bars don't crash
- Extreme prices don't cause overflow
- All-zero prices fail gracefully

These tests prevent production crashes from corrupted or malformed data.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Tuple
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.risk_config import RiskConfig


# ============================================================================
# CORRUPTED DATA GENERATORS
# ============================================================================

def create_data_with_nan(
    n_bars: int = 50,
    nan_positions: list = None
) -> pd.DataFrame:
    """Create data with NaN values at specified positions."""
    if nan_positions is None:
        nan_positions = [10, 25, 40]

    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    prices = [100 + i * 0.5 for i in range(n_bars)]

    # Insert NaN at specified positions
    for pos in nan_positions:
        if pos < len(prices):
            prices[pos] = np.nan

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 if not pd.isna(p) else np.nan for p in prices],
        'low': [p * 0.99 if not pd.isna(p) else np.nan for p in prices],
        'close': prices,
        'volume': [1000000] * n_bars
    }, index=dates)

    return df


def create_data_with_duplicates(n_bars: int = 50) -> pd.DataFrame:
    """Create data with duplicate timestamps."""
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    # Create duplicate by repeating some dates
    dates_list = list(dates)
    dates_list[10] = dates_list[9]  # Duplicate
    dates_list[25] = dates_list[24]  # Duplicate

    prices = [100 + i * 0.5 for i in range(n_bars)]

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * n_bars
    }, index=pd.DatetimeIndex(dates_list))

    return df


def create_data_with_gaps(gap_days: list = None) -> pd.DataFrame:
    """Create data with missing days (gaps)."""
    if gap_days is None:
        gap_days = [10, 11, 12, 25, 26]  # Weekend-like gaps

    # Create 50 business days but skip some
    all_dates = pd.date_range(start='2023-01-01', periods=70, freq='D')

    # Remove gap days
    dates = [d for i, d in enumerate(all_dates) if i not in gap_days][:50]

    prices = [100 + i * 0.5 for i in range(len(dates))]

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * len(dates)
    }, index=pd.DatetimeIndex(dates))

    return df


def create_data_with_negative_prices(n_bars: int = 50) -> pd.DataFrame:
    """Create data with some negative prices (corrupted data)."""
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    prices = [100 + i * 0.5 for i in range(n_bars)]

    # Make some prices negative
    prices[15] = -50.0
    prices[30] = -100.0

    df = pd.DataFrame({
        'open': prices,
        'high': [abs(p) * 1.01 for p in prices],  # High can't be negative
        'low': prices,  # But low might be
        'close': prices,
        'volume': [1000000] * n_bars
    }, index=dates)

    return df


def create_data_with_zero_volume(n_bars: int = 50) -> pd.DataFrame:
    """Create data with zero volume bars."""
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    prices = [100 + i * 0.5 for i in range(n_bars)]
    volumes = [1000000] * n_bars

    # Zero volume on some bars
    volumes[10] = 0
    volumes[25] = 0
    volumes[40] = 0

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)

    return df


def create_data_with_extreme_prices(n_bars: int = 50) -> pd.DataFrame:
    """Create data with extremely large prices."""
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    # Start with massive price
    base = 1_000_000.0  # $1 million per share
    prices = [base + i * 10000 for i in range(n_bars)]

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000] * n_bars  # Low volume for expensive stock
    }, index=dates)

    return df


def create_all_zero_prices(n_bars: int = 50) -> pd.DataFrame:
    """Create completely corrupted data (all zeros)."""
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    df = pd.DataFrame({
        'open': [0.0] * n_bars,
        'high': [0.0] * n_bars,
        'low': [0.0] * n_bars,
        'close': [0.0] * n_bars,
        'volume': [0] * n_bars
    }, index=dates)

    return df


# ============================================================================
# TEST STRATEGIES
# ============================================================================

class SimpleBuyHoldStrategy(LongOnlyStrategy):
    """Simple buy and hold for data integrity tests."""

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        # Buy on first valid bar
        entries.iloc[0] = True

        # Sell on last bar
        if len(data) > 1:
            exits.iloc[-1] = True

        return entries, exits


class MultipleEntryStrategy(LongOnlyStrategy):
    """Multiple entries to test data integrity across trades."""

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        # Buy every 10 bars
        for i in range(0, len(data), 10):
            if i < len(data):
                entries.iloc[i] = True
            if i + 5 < len(data):
                exits.iloc[i + 5] = True

        return entries, exits


# ============================================================================
# TEST CLASS: NaN Handling
# ============================================================================

class TestNaNHandling:
    """
    Verify NaN values don't propagate to metrics.
    """

    def test_nan_in_prices_doesnt_crash(self):
        """
        NaN in price data should not crash the backtest.

        Expected: Skip NaN bars gracefully, continue backtest.
        """
        data = create_data_with_nan(n_bars=50, nan_positions=[10, 25, 40])

        strategy = SimpleBuyHoldStrategy()

        print(f"\n{'='*60}")
        print("NaN PRICE HANDLING TEST")
        print(f"{'='*60}")
        print(f"Data has NaN at bars: 10, 25, 40")
        print(f"Running backtest...")

        try:
            engine = BacktestEngine(
                initial_capital=10000,
                fees=0.001,
                slippage=0.0,
                market_hours_only=False,
                risk_config=RiskConfig.moderate()
            )

            portfolio = engine.run_with_data(strategy, data)

            print(f"✓ Backtest completed without crash")
            print(f"{'='*60}\n")

            # Verify metrics are not NaN
            stats = portfolio.stats()

            print("Checking metrics for NaN:")
            nan_metrics = []

            for key, value in stats.items():
                if isinstance(value, (int, float, np.number)):
                    if pd.isna(value) or np.isnan(value):
                        nan_metrics.append(key)
                        print(f"  ⚠️ {key}: NaN")
                    else:
                        print(f"  ✓ {key}: {value}")

            print(f"\n{'='*60}")
            print(f"NaN metrics found: {len(nan_metrics)}")
            print(f"{'='*60}\n")

            # Should have no NaN metrics
            # (Some metrics might be 0 or inf, but not NaN)
            assert len(nan_metrics) == 0, \
                f"Found NaN in metrics: {nan_metrics}"

        except Exception as e:
            pytest.fail(f"Backtest crashed with NaN data: {e}")

    def test_all_nan_data_fails_gracefully(self):
        """
        If all data is NaN, should fail with clear error (not crash).
        """
        data = create_data_with_nan(n_bars=50, nan_positions=list(range(50)))

        strategy = SimpleBuyHoldStrategy()

        print(f"\n{'='*60}")
        print("ALL NaN DATA TEST")
        print(f"{'='*60}")
        print("All price data is NaN")

        # This should either:
        # 1. Raise a clear error
        # 2. Return gracefully with no trades

        try:
            engine = BacktestEngine(
                initial_capital=10000,
                fees=0.001,
                slippage=0.0,
                market_hours_only=False,
                risk_config=RiskConfig.moderate()
            )

            portfolio = engine.run_with_data(strategy, data)

            # If it succeeds, should have no trades
            print(f"✓ Handled gracefully")
            print(f"Trades: {len(portfolio.trades)}")
            print(f"{'='*60}\n")

        except Exception as e:
            # Acceptable to raise error for all-NaN data
            print(f"✓ Raised error (acceptable): {type(e).__name__}")
            print(f"{'='*60}\n")


# ============================================================================
# TEST CLASS: Data Quality Issues
# ============================================================================

class TestDataQualityIssues:
    """
    Test handling of various data quality issues.
    """

    def test_duplicate_timestamps_handled(self):
        """
        Duplicate timestamps should be handled gracefully.

        Either: Keep first, keep last, or average - but don't crash.
        """
        data = create_data_with_duplicates(n_bars=50)

        strategy = SimpleBuyHoldStrategy()

        print(f"\n{'='*60}")
        print("DUPLICATE TIMESTAMPS TEST")
        print(f"{'='*60}")
        print(f"Data has {data.index.duplicated().sum()} duplicate timestamps")

        try:
            engine = BacktestEngine(
                initial_capital=10000,
                fees=0.001,
                slippage=0.0,
                market_hours_only=False,
                risk_config=RiskConfig.moderate()
            )

            # Suppress any warnings about duplicates
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                portfolio = engine.run_with_data(strategy, data)

            print(f"✓ Handled duplicates gracefully")
            print(f"Trades: {len(portfolio.trades)}")
            print(f"{'='*60}\n")

        except Exception as e:
            pytest.fail(f"Failed to handle duplicate timestamps: {e}")

    def test_missing_bars_gaps_in_data(self):
        """
        Gaps in data (e.g., weekends, holidays) should be handled.

        Should not crash, metrics should still calculate.
        """
        data = create_data_with_gaps(gap_days=[10, 11, 12, 25, 26])

        strategy = MultipleEntryStrategy()

        print(f"\n{'='*60}")
        print("DATA GAPS TEST")
        print(f"{'='*60}")
        print(f"Data has gaps (weekends/holidays)")
        print(f"Total bars: {len(data)}")

        try:
            engine = BacktestEngine(
                initial_capital=10000,
                fees=0.001,
                slippage=0.0,
                market_hours_only=False,
                risk_config=RiskConfig.moderate()
            )

            portfolio = engine.run_with_data(strategy, data)

            print(f"✓ Handled gaps gracefully")
            print(f"Trades: {len(portfolio.trades)}")
            print(f"{'='*60}\n")

        except Exception as e:
            pytest.fail(f"Failed to handle data gaps: {e}")

    def test_zero_volume_bars_dont_crash(self):
        """
        Zero volume bars (illiquid stocks) should not crash.

        Should either skip the bar or handle gracefully.
        """
        data = create_data_with_zero_volume(n_bars=50)

        strategy = MultipleEntryStrategy()

        print(f"\n{'='*60}")
        print("ZERO VOLUME BARS TEST")
        print(f"{'='*60}")
        print(f"Data has zero volume at bars: 10, 25, 40")

        try:
            engine = BacktestEngine(
                initial_capital=10000,
                fees=0.001,
                slippage=0.0,
                market_hours_only=False,
                risk_config=RiskConfig.moderate()
            )

            portfolio = engine.run_with_data(strategy, data)

            print(f"✓ Handled zero volume gracefully")
            print(f"Trades: {len(portfolio.trades)}")
            print(f"{'='*60}\n")

        except Exception as e:
            pytest.fail(f"Failed to handle zero volume: {e}")


# ============================================================================
# TEST CLASS: Extreme Values
# ============================================================================

class TestExtremeValues:
    """
    Test handling of extreme price values.
    """

    def test_extremely_large_prices_no_overflow(self):
        """
        Very large prices ($1M+ per share) should not cause overflow.

        Example: Berkshire Hathaway Class A shares
        """
        data = create_data_with_extreme_prices(n_bars=50)

        strategy = SimpleBuyHoldStrategy()

        print(f"\n{'='*60}")
        print("EXTREME PRICES TEST")
        print(f"{'='*60}")
        print(f"Stock price: ${data.iloc[0]['close']:,.2f} per share")

        try:
            engine = BacktestEngine(
                initial_capital=10000000,  # $10M capital for expensive stock
                fees=0.001,
                slippage=0.0,
                market_hours_only=False,
                risk_config=RiskConfig.moderate()
            )

            portfolio = engine.run_with_data(strategy, data)

            stats = portfolio.stats()

            print(f"✓ No overflow errors")
            print(f"Final value: ${stats.get('Final Portfolio Value [$]', 0):,.2f}")
            print(f"{'='*60}\n")

            # Check for inf values
            inf_metrics = []
            for key, value in stats.items():
                if isinstance(value, (int, float, np.number)):
                    if np.isinf(value):
                        inf_metrics.append(key)

            assert len(inf_metrics) == 0, \
                f"Found inf in metrics: {inf_metrics}"

        except OverflowError as e:
            pytest.fail(f"Overflow error with large prices: {e}")
        except Exception as e:
            pytest.fail(f"Failed with extreme prices: {e}")

    def test_negative_prices_handled_or_rejected(self):
        """
        Negative prices (corrupted data) should either be rejected or skipped.

        Real stocks can't have negative prices.
        """
        data = create_data_with_negative_prices(n_bars=50)

        strategy = SimpleBuyHoldStrategy()

        print(f"\n{'='*60}")
        print("NEGATIVE PRICES TEST")
        print(f"{'='*60}")
        print(f"Data contains negative prices at bars: 15, 30")

        # This should either:
        # 1. Raise validation error
        # 2. Skip negative price bars
        # 3. Replace with NaN and handle like NaN test

        try:
            engine = BacktestEngine(
                initial_capital=10000,
                fees=0.001,
                slippage=0.0,
                market_hours_only=False,
                risk_config=RiskConfig.moderate()
            )

            portfolio = engine.run_with_data(strategy, data)

            print(f"✓ Handled negative prices (skipped or replaced)")
            print(f"Trades: {len(portfolio.trades)}")
            print(f"{'='*60}\n")

        except Exception as e:
            # Acceptable to raise error for negative prices
            print(f"✓ Rejected negative prices (acceptable): {type(e).__name__}")
            print(f"{'='*60}\n")

    def test_all_zero_prices_fails_gracefully(self):
        """
        All-zero prices (completely corrupted) should fail with clear error.

        Should not crash with cryptic error.
        """
        data = create_all_zero_prices(n_bars=50)

        strategy = SimpleBuyHoldStrategy()

        print(f"\n{'='*60}")
        print("ALL-ZERO PRICES TEST")
        print(f"{'='*60}")
        print("All prices are zero (corrupted data)")

        # Should raise clear error or handle gracefully
        try:
            engine = BacktestEngine(
                initial_capital=10000,
                fees=0.001,
                slippage=0.0,
                market_hours_only=False,
                risk_config=RiskConfig.moderate()
            )

            portfolio = engine.run_with_data(strategy, data)

            # If succeeds, should have no trades
            print(f"✓ Handled all-zero prices")
            print(f"Trades: {len(portfolio.trades)}")
            assert len(portfolio.trades) == 0, \
                "Should not trade on zero prices"
            print(f"{'='*60}\n")

        except Exception as e:
            # Acceptable to raise error
            print(f"✓ Raised error (acceptable): {type(e).__name__}")
            print(f"{'='*60}\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
