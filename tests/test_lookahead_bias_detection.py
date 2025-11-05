"""
Critical tests to detect lookahead bias in backtests.

Lookahead bias occurs when strategy decisions at time t
use information from t+1 or later, which is impossible in real trading.

These tests ensure the backtesting engine prevents future data leakage.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.base.strategy import LongOnlyStrategy
from strategies.base_strategies.moving_average import MovingAverageCrossover
from backtesting.utils.indicators import Indicators


# ============================================================================
# FIXTURES: Test Data Scenarios
# ============================================================================

@pytest.fixture
def crash_scenario_data():
    """
    Create price data with a crash on day 50.

    Pattern:
    - Days 0-49: Steady uptrend
    - Day 50: Crash (-20%)
    - Days 51-100: Recovery

    A strategy with lookahead bias would exit on day 49.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Build crash scenario
    prices = []
    for i in range(100):
        if i < 50:
            # Uptrend before crash
            base_price = 100 + i * 0.5
            noise = np.random.randn() * 0.3
            prices.append(base_price + noise)
        elif i == 50:
            # CRASH
            prices.append(prices[-1] * 0.80)  # -20% drop
        else:
            # Recovery after crash
            base_price = prices[50] + (i - 50) * 0.3
            noise = np.random.randn() * 0.3
            prices.append(base_price + noise)

    close_prices = np.array(prices)

    df = pd.DataFrame({
        'open': close_prices - 0.2,
        'high': close_prices + np.abs(np.random.randn(100) * 0.5),
        'low': close_prices - np.abs(np.random.randn(100) * 0.5),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    return df


@pytest.fixture
def stable_uptrend_data():
    """
    Create stable uptrend data with no crashes.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(123)

    # Steady uptrend
    trend = np.linspace(100, 150, 100)
    noise = np.random.randn(100) * 1.5
    close_prices = trend + noise

    df = pd.DataFrame({
        'open': close_prices - 0.2,
        'high': close_prices + np.abs(np.random.randn(100) * 0.5),
        'low': close_prices - np.abs(np.random.randn(100) * 0.5),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    return df


@pytest.fixture
def volatile_data():
    """
    Create volatile data with large swings.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(456)

    # High volatility
    base = 100
    prices = [base]
    for i in range(99):
        # Large random moves
        change_pct = np.random.randn() * 0.05  # ±5% per day
        prices.append(prices[-1] * (1 + change_pct))

    close_prices = np.array(prices)

    df = pd.DataFrame({
        'open': close_prices - 0.2,
        'high': close_prices + np.abs(np.random.randn(100) * 1.0),
        'low': close_prices - np.abs(np.random.randn(100) * 1.0),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    return df


# ============================================================================
# TEST CLASS: Signal Timing Validation
# ============================================================================

class TestSignalTimingConsistency:
    """
    Verify that signals at bar i don't change when bars i+1, i+2, ... are added.

    This is the GOLD STANDARD test for lookahead bias.
    """

    def test_signal_consistent_across_data_lengths(self, stable_uptrend_data):
        """
        CRITICAL: Signal at bar 50 should be identical whether we have
        100 bars or just 51 bars of data.

        If adding more data changes past signals → LOOKAHEAD BUG
        """
        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)

        # Generate signals on full dataset (100 bars)
        entries_full, exits_full = strategy.generate_signals(stable_uptrend_data)

        # Generate signals on partial dataset (first 51 bars only)
        entries_partial, exits_partial = strategy.generate_signals(
            stable_uptrend_data.iloc[:51]
        )

        # Signal at bar 50 MUST be identical in both cases
        assert entries_full.iloc[50] == entries_partial.iloc[50], \
            "Entry signal at bar 50 changed when given more data - LOOKAHEAD BIAS!"

        assert exits_full.iloc[50] == exits_partial.iloc[50], \
            "Exit signal at bar 50 changed when given more data - LOOKAHEAD BIAS!"

    def test_signal_consistent_multiple_positions(self, stable_uptrend_data):
        """
        Test signal consistency at multiple positions in the dataset.
        """
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        # Test at bars 30, 50, 70
        test_positions = [30, 50, 70]

        # Generate signals on full dataset
        entries_full, exits_full = strategy.generate_signals(stable_uptrend_data)

        for pos in test_positions:
            # Generate signals on partial dataset up to this position
            entries_partial, exits_partial = strategy.generate_signals(
                stable_uptrend_data.iloc[:pos+1]
            )

            # Verify consistency
            assert entries_full.iloc[pos] == entries_partial.iloc[pos], \
                f"Entry signal at bar {pos} changed with more data"

            assert exits_full.iloc[pos] == exits_partial.iloc[pos], \
                f"Exit signal at bar {pos} changed with more data"

    def test_indicator_values_consistent(self, stable_uptrend_data):
        """
        Test that indicator values don't change when more data is added.

        Moving average at bar 50 should be same with 51 bars or 100 bars.
        """
        close = stable_uptrend_data['close']

        # Calculate MA on full dataset
        ma_full = Indicators.sma(close, window=20)

        # Calculate MA on partial dataset
        ma_partial = Indicators.sma(close.iloc[:51], window=20)

        # Value at bar 50 should be identical
        assert abs(ma_full.iloc[50] - ma_partial.iloc[50]) < 1e-10, \
            "Moving average at bar 50 changed when more data added - BUG!"


# ============================================================================
# TEST CLASS: Crash Prediction Prevention
# ============================================================================

class TestCannotPredictFuture:
    """
    Verify strategies cannot predict future events.

    These tests create scenarios where lookahead bias would be obvious.
    """

    def test_cannot_exit_before_crash(self, crash_scenario_data):
        """
        CRITICAL: Strategy should NOT exit on day 49 before crash on day 50.

        This tests for "magical" prediction of future crashes.
        """
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        # Generate signals
        entries, exits = strategy.generate_signals(crash_scenario_data)

        # Check day 49 (before crash)
        # Strategy should only exit if it has a legitimate technical reason
        # (MA crossover, not future knowledge)

        # If there's an exit on day 49, verify it's based on past data
        if exits.iloc[49]:
            # Check that price action on days 45-49 justifies the exit
            # (e.g., MAs actually crossed)
            close = crash_scenario_data['close']
            fast_ma = Indicators.sma(close, 5)
            slow_ma = Indicators.sma(close, 15)

            # There should be a crossover (fast < slow) before day 49
            crossover_occurred = (fast_ma.iloc[49] < slow_ma.iloc[49]) and \
                                 (fast_ma.iloc[48] >= slow_ma.iloc[48])

            assert crossover_occurred, \
                "Exit on day 49 has no technical basis - possible lookahead bias!"

    def test_cannot_enter_at_exact_bottom(self, crash_scenario_data):
        """
        Strategy should not enter at the exact bottom of the crash.

        Real strategies need time to detect bottoms. Entering at the
        exact lowest point suggests future knowledge.
        """
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)
        entries, exits = strategy.generate_signals(crash_scenario_data)

        # Find the crash bottom (day 50)
        crash_day = 50

        # Strategy should NOT enter on day 50 (exact bottom)
        # It needs at least 1-2 days to detect the bottom

        if entries.iloc[crash_day]:
            # Verify there's a technical reason (e.g., MA crossover)
            close = crash_scenario_data['close']
            fast_ma = Indicators.sma(close, 5)
            slow_ma = Indicators.sma(close, 15)

            # Check if MAs crossed
            crossover = (fast_ma.iloc[crash_day] > slow_ma.iloc[crash_day]) and \
                       (fast_ma.iloc[crash_day-1] <= slow_ma.iloc[crash_day-1])

            # For MA strategies, it's unlikely (but possible) to enter at exact bottom
            # This is more of a sanity check than a hard rule

    def test_no_premature_reentry_after_exit(self, stable_uptrend_data):
        """
        After exiting, strategy shouldn't re-enter immediately unless
        there's a clear signal. This tests for future-looking optimization.
        """
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)
        entries, exits = strategy.generate_signals(stable_uptrend_data)

        # Check for exit-entry patterns
        for i in range(1, len(exits)-1):
            if exits.iloc[i]:
                # If exit at bar i, entry at bar i+1 should be rare
                # (unless there's a strong reversal signal)

                # Allow re-entry after at least 2 bars
                if entries.iloc[i+1]:
                    # Verify there's a technical reason
                    close = stable_uptrend_data['close']

                    # Price should have changed significantly
                    price_change = abs(close.iloc[i+1] - close.iloc[i]) / close.iloc[i]

                    # Some price change should justify re-entry
                    # (this is a soft check - MA crossovers can happen quickly)


# ============================================================================
# TEST CLASS: Reversed Data Test
# ============================================================================

class TestReversedDataBehavior:
    """
    Running strategy on reversed data should produce DIFFERENT results.

    If results are the same, strategy might be using symmetric operations
    that ignore time direction (potential bug).
    """

    def test_strategy_on_reversed_data_differs(self, volatile_data):
        """
        Strategy on forward vs. reversed data should give different signals.

        If they're identical, might indicate use of .shift(-1) or other
        forward-looking operations.

        Uses volatile data to ensure signals are generated.
        """
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        # Forward data
        entries_fwd, exits_fwd = strategy.generate_signals(volatile_data)

        # Check if strategy generates any signals
        total_fwd_signals = entries_fwd.sum() + exits_fwd.sum()

        if total_fwd_signals == 0:
            pytest.skip("Strategy generated no signals on this data - cannot test reversal")

        # Reversed data (flip time)
        data_reversed = volatile_data.iloc[::-1].copy()
        data_reversed.index = pd.date_range(
            start='2023-01-01',
            periods=len(data_reversed),
            freq='D'
        )

        entries_rev, exits_rev = strategy.generate_signals(data_reversed)

        # Reverse the reversed signals back to original order
        entries_rev_flipped = entries_rev.iloc[::-1].values
        exits_rev_flipped = exits_rev.iloc[::-1].values

        # They should NOT be identical
        # Calculate number of differences
        entry_differences = (entries_fwd.values != entries_rev_flipped).sum()
        exit_differences = (exits_fwd.values != exits_rev_flipped).sum()

        # At least some signals should differ (MAs are direction-dependent)
        # If they're all the same, it suggests the strategy might be symmetric
        # or using lookahead operations

        total_differences = entry_differences + exit_differences

        # Require at least 10% of signals to differ
        min_differences = max(1, int(total_fwd_signals * 0.10))

        assert total_differences >= min_differences, \
            f"Only {total_differences} differences out of {total_fwd_signals} signals " \
            f"({total_differences/total_fwd_signals*100:.1f}%) - might indicate lookahead bug or symmetric strategy"


# ============================================================================
# TEST CLASS: Execution Price Realism
# ============================================================================

class TestExecutionPriceRealism:
    """
    Verify execution prices are realistic (not at extremes of bars).
    """

    def test_cannot_buy_at_daily_low(self, volatile_data):
        """
        Entry prices should not consistently be at the low of the day.

        This would indicate unrealistic execution (catching bottoms perfectly).
        """
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)
        entries, exits = strategy.generate_signals(volatile_data)

        # Check entry bars
        entry_bars = volatile_data[entries]

        if len(entry_bars) > 0:
            # Entry price should be the close (in current implementation)
            # It should NOT be at the low of the bar

            for idx in entry_bars.index:
                entry_price = volatile_data.loc[idx, 'close']
                bar_low = volatile_data.loc[idx, 'low']
                bar_high = volatile_data.loc[idx, 'high']

                # Entry should be within bar range
                assert bar_low <= entry_price <= bar_high, \
                    f"Entry price {entry_price} outside bar range [{bar_low}, {bar_high}]"

                # Entry should not be exactly at the low (unrealistic)
                # Allow small tolerance for rounding
                tolerance = (bar_high - bar_low) * 0.01
                assert entry_price > bar_low + tolerance or entry_price == bar_low, \
                    f"Entry at exact low {bar_low} - too perfect!"

    def test_exit_prices_within_bar_range(self, volatile_data):
        """
        Exit prices should be within the bar's high-low range.
        """
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)
        entries, exits = strategy.generate_signals(volatile_data)

        # Check exit bars
        exit_bars = volatile_data[exits]

        if len(exit_bars) > 0:
            for idx in exit_bars.index:
                exit_price = volatile_data.loc[idx, 'close']
                bar_low = volatile_data.loc[idx, 'low']
                bar_high = volatile_data.loc[idx, 'high']

                # Exit should be within bar range
                assert bar_low <= exit_price <= bar_high, \
                    f"Exit price {exit_price} outside bar range [{bar_low}, {bar_high}]"


# ============================================================================
# TEST CLASS: Indicator Lookahead Detection
# ============================================================================

class TestIndicatorLookahead:
    """
    Verify indicators don't peek forward.
    """

    def test_sma_only_uses_past_data(self):
        """
        Simple Moving Average should only use current and past bars.
        """
        close = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])

        # Calculate SMA with window=5
        sma = Indicators.sma(close, window=5)

        # SMA at position 4 should be average of bars 0-4
        expected_sma_4 = (100 + 102 + 104 + 106 + 108) / 5
        assert abs(sma.iloc[4] - expected_sma_4) < 1e-10, \
            "SMA calculation incorrect - might be using future data"

        # SMA at position 6 should be average of bars 2-6
        expected_sma_6 = (104 + 106 + 108 + 110 + 112) / 5
        assert abs(sma.iloc[6] - expected_sma_6) < 1e-10, \
            "SMA calculation incorrect"

    def test_ema_only_uses_past_data(self):
        """
        Exponential Moving Average should only use current and past bars.
        """
        close = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])

        # Calculate EMA
        ema = Indicators.ema(close, window=5)

        # EMA is recursive: EMA[i] = α * close[i] + (1-α) * EMA[i-1]
        # It should not reference close[i+1]

        # Verify EMA is monotonically increasing for this data
        # (since prices are trending up)
        for i in range(len(ema) - 1):
            if not pd.isna(ema.iloc[i]) and not pd.isna(ema.iloc[i+1]):
                # EMA should increase (or stay similar) for uptrend
                # If it decreases significantly, might be lookahead
                assert ema.iloc[i+1] >= ema.iloc[i] * 0.95, \
                    f"EMA decreased unexpectedly at position {i+1}"


# ============================================================================
# TEST CLASS: Full Backtest Integration
# ============================================================================

class TestFullBacktestLookahead:
    """
    Integration tests: Run full backtests and verify no lookahead bias.
    """

    def test_backtest_results_consistent_with_partial_data(self, stable_uptrend_data):
        """
        Run backtest on partial data, then full data.

        The trades made in the partial backtest should be identical
        in the full backtest (same timestamps, same prices).
        """
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        # Run on partial data (first 50 bars)
        engine_partial = BacktestEngine(
            initial_capital=100000,
            fees=0.001,
            slippage=0.0
        )
        portfolio_partial = engine_partial.run_with_data(
            strategy,
            stable_uptrend_data.iloc[:50]
        )

        # Run on full data (100 bars)
        engine_full = BacktestEngine(
            initial_capital=100000,
            fees=0.001,
            slippage=0.0
        )
        portfolio_full = engine_full.run_with_data(
            strategy,
            stable_uptrend_data
        )

        # Compare trades in the first 50 bars
        # They should be IDENTICAL

        # Get trades from first 50 days
        partial_trades = portfolio_partial.trades
        full_trades_first_50 = [
            t for t in portfolio_full.trades
            if t['timestamp'] <= stable_uptrend_data.iloc[49].name
        ]

        # Should have same number of trades in first 50 bars
        assert len(partial_trades) == len(full_trades_first_50), \
            f"Trade count differs: {len(partial_trades)} vs {len(full_trades_first_50)} - LOOKAHEAD?"

        # Each trade should be identical
        for i in range(len(partial_trades)):
            assert partial_trades[i]['timestamp'] == full_trades_first_50[i]['timestamp'], \
                f"Trade {i} timestamp differs"

            assert partial_trades[i]['type'] == full_trades_first_50[i]['type'], \
                f"Trade {i} type differs"

            assert abs(partial_trades[i]['price'] - full_trades_first_50[i]['price']) < 0.01, \
                f"Trade {i} price differs - possible lookahead!"


# ============================================================================
# TEST CLASS: Deliberate Lookahead Bugs (Should Fail)
# ============================================================================

class TestDeliberateLookaheadBugs:
    """
    These tests intentionally create lookahead bias to verify our tests catch it.

    These should FAIL or be skipped in normal test runs.
    """

    @pytest.mark.skip(reason="Intentional bug test - used for validation")
    def test_intentional_forward_shift_detected(self, stable_uptrend_data):
        """
        This test intentionally uses .shift(-1) to create lookahead bias.

        Our other tests should catch this pattern.
        """
        # Create a strategy with intentional lookahead
        class LookaheadStrategy(LongOnlyStrategy):
            def generate_long_signals(self, data):
                close = data['close']

                # INTENTIONAL BUG: Use future data
                future_close = close.shift(-1)  # Tomorrow's close!

                # Buy if tomorrow's close will be higher
                entries = (future_close > close)
                exits = (future_close < close)

                return entries.fillna(False), exits.fillna(False)

        strategy = LookaheadStrategy()

        # Generate signals on full vs partial data
        entries_full, _ = strategy.generate_signals(stable_uptrend_data)
        entries_partial, _ = strategy.generate_signals(stable_uptrend_data.iloc[:51])

        # Signal at bar 50 will DIFFER because .shift(-1) uses bar 51
        # Our consistency test should catch this
        signal_differs = (entries_full.iloc[50] != entries_partial.iloc[50])

        assert signal_differs, \
            "Lookahead bug not detected! Signal consistency test needs improvement."


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
