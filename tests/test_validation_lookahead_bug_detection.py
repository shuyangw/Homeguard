"""
Validation test: Create intentional lookahead bugs and verify they're detected.

This test file creates strategies with INTENTIONAL lookahead bias to verify
that our detection tests actually catch them.

These tests should FAIL - if they pass, our detection isn't working!
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.indicators import Indicators


# ============================================================================
# INTENTIONAL LOOKAHEAD BUGS (Should be detected by our tests)
# ============================================================================

class LookaheadBugStrategy1(LongOnlyStrategy):
    """
    BUG: Uses .shift(-1) to peek at next bar's close.

    This should be detected by signal timing consistency test.
    """

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        close = data['close']

        # INTENTIONAL BUG: Look at tomorrow's close
        future_close = close.shift(-1)

        # Buy if tomorrow's close will be higher
        entries = (future_close > close).fillna(False)

        # Exit if tomorrow's close will be lower
        exits = (future_close < close).fillna(False)

        return entries, exits


class LookaheadBugStrategy2(LongOnlyStrategy):
    """
    BUG: Uses future high to detect tops.

    This would allow "perfect" exits at market tops.
    """

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        close = data['close']
        high = data['high']

        # Calculate future maximum (next 10 bars)
        # INTENTIONAL BUG: This uses future data
        future_max = high.rolling(10).max().shift(-10)

        # Buy if we're near lows
        sma = Indicators.sma(close, 20)
        entries = (close < sma).fillna(False)

        # Exit if close is near future high (magical top-picking)
        exits = (close >= future_max * 0.95).fillna(False)

        return entries, exits


class LookaheadBugStrategy3(LongOnlyStrategy):
    """
    BUG: Uses rolling window that includes future bars.

    Subtle bug - rolling window with center=True uses future data.
    """

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        close = data['close']

        # INTENTIONAL BUG: centered window uses future data
        # This is a common mistake with pandas rolling operations
        centered_ma = close.rolling(window=10, center=True).mean()

        sma = Indicators.sma(close, 20)

        entries = (centered_ma > sma).fillna(False)
        exits = (centered_ma < sma).fillna(False)

        return entries, exits


# ============================================================================
# VALIDATION TESTS: Verify our detection tests catch these bugs
# ============================================================================

@pytest.fixture
def test_data():
    """Create test data for validation."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    trend = np.linspace(100, 150, 100)
    noise = np.random.randn(100) * 2
    close_prices = trend + noise

    df = pd.DataFrame({
        'open': close_prices - 0.2,
        'high': close_prices + np.abs(np.random.randn(100) * 0.5),
        'low': close_prices - np.abs(np.random.randn(100) * 0.5),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    return df


class TestLookaheadBugDetection:
    """
    These tests should FAIL when run with lookahead strategies.

    This validates that our detection tests actually work.
    """

    def test_detect_forward_shift_bug(self, test_data):
        """
        Verify signal timing test catches .shift(-1) bug.

        This test should FAIL with LookaheadBugStrategy1.
        """
        strategy = LookaheadBugStrategy1()

        # Generate signals on full dataset
        entries_full, _ = strategy.generate_signals(test_data)

        # Generate signals on partial dataset (51 bars)
        entries_partial, _ = strategy.generate_signals(test_data.iloc[:51])

        # With lookahead bug, signal at bar 50 will DIFFER
        # because .shift(-1) uses bar 51 which is missing in partial data

        # This assertion should FAIL (proving our test detects the bug)
        with pytest.raises(AssertionError, match="LOOKAHEAD BIAS"):
            assert entries_full.iloc[50] == entries_partial.iloc[50], \
                "Entry signal at bar 50 changed when given more data - LOOKAHEAD BIAS!"

    def test_detect_future_max_bug(self, test_data):
        """
        Verify signal timing test catches rolling max with negative shift.

        This test should FAIL with LookaheadBugStrategy2.
        """
        strategy = LookaheadBugStrategy2()

        # This strategy uses future_max which looks ahead 10 bars
        entries_full, exits_full = strategy.generate_signals(test_data)

        # Generate on partial data (60 bars)
        entries_partial, exits_partial = strategy.generate_signals(test_data.iloc[:60])

        # Signal at bar 50 will differ because future_max uses bars 51-60
        with pytest.raises(AssertionError, match="LOOKAHEAD BIAS"):
            assert exits_full.iloc[50] == exits_partial.iloc[50], \
                "Exit signal at bar 50 changed when given more data - LOOKAHEAD BIAS!"

    def test_detect_centered_rolling_bug(self, test_data):
        """
        Verify detection of centered rolling window (uses future data).

        This test should FAIL with LookaheadBugStrategy3.
        """
        strategy = LookaheadBugStrategy3()

        # Centered rolling uses future bars
        entries_full, _ = strategy.generate_signals(test_data)

        # Generate on partial data
        entries_partial, _ = strategy.generate_signals(test_data.iloc[:56])

        # Signal at bar 50 will differ because centered window needs bars 51-55
        with pytest.raises(AssertionError):
            assert entries_full.iloc[50] == entries_partial.iloc[50], \
                "Centered rolling window uses future data - LOOKAHEAD BIAS!"


class TestValidationWithCleanStrategy:
    """
    These tests should PASS when run with clean strategies.

    This proves our detection tests don't produce false positives.
    """

    def test_clean_strategy_passes_timing_check(self, test_data):
        """
        Clean strategy (from real codebase) should pass timing test.
        """
        from strategies.base_strategies.moving_average import MovingAverageCrossover

        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)

        # Generate on full data
        entries_full, _ = strategy.generate_signals(test_data)

        # Generate on partial data
        entries_partial, _ = strategy.generate_signals(test_data.iloc[:51])

        # Should be identical (no lookahead)
        assert entries_full.iloc[50] == entries_partial.iloc[50], \
            "Clean strategy failed timing test - false positive!"


# ============================================================================
# RUN VALIDATION
# ============================================================================

if __name__ == '__main__':
    # These tests verify that our lookahead detection works
    # The "bug" tests should fail, proving detection works
    # The "clean" test should pass, proving no false positives

    print("=" * 80)
    print("VALIDATION: Testing Lookahead Bug Detection")
    print("=" * 80)
    print()
    print("Running validation tests...")
    print("Expected: Bug detection tests should CATCH the intentional bugs")
    print()

    pytest.main([__file__, '-v', '--tb=short'])
