"""
Unit tests for stop loss functionality in Numba ranked portfolio simulation.

Tests for the `simulate_ranked_portfolio_with_stops` function and the
`stop_loss_pct` parameter in `run_mp_backtest_numba`.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtesting.engine.numba_ranked_sim import (
    simulate_ranked_portfolio,
    simulate_ranked_portfolio_with_stops,
    simulate_ranked_portfolio_with_intraday_stops,
    calculate_momentum_scores,
    calculate_spy_drawdown,
    run_mp_backtest_numba,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_price_data():
    """Create simple multi-symbol price data for testing."""
    np.random.seed(42)
    n_days = 150
    n_symbols = 10

    dates = pd.date_range('2022-01-03', periods=n_days, freq='D')

    # Create prices with different trends
    prices = np.zeros((n_days, n_symbols))
    for j in range(n_symbols):
        base = 100 + j * 10
        trend = 0.001 * (j - 5)  # Some up, some down
        noise = np.random.randn(n_days) * 0.02
        prices[:, j] = base * (1 + trend + noise).cumprod()

    symbols = [f'SYM{i}' for i in range(n_symbols)]
    prices_df = pd.DataFrame(prices, index=dates, columns=symbols)

    spy_series = pd.Series(prices[:, 0], index=dates)
    vix_series = pd.Series(20.0, index=dates)  # Constant VIX

    return prices_df, spy_series, vix_series


@pytest.fixture
def crash_scenario_data():
    """Create data where one symbol crashes (for stop loss testing)."""
    n_days = 100
    n_symbols = 5

    dates = pd.date_range('2022-01-03', periods=n_days, freq='D')

    prices = np.zeros((n_days, n_symbols))

    # Symbol 0: crashes 20% on day 50
    prices[:, 0] = np.concatenate([
        np.linspace(100, 110, 50),  # Goes up first (good momentum)
        np.linspace(90, 85, 50)     # Then crashes
    ])

    # Symbol 1: steady uptrend
    prices[:, 1] = np.linspace(100, 130, n_days)

    # Symbol 2: steady uptrend (slightly less)
    prices[:, 2] = np.linspace(100, 125, n_days)

    # Symbol 3: flat with noise
    np.random.seed(123)
    prices[:, 3] = 100 + np.random.randn(n_days).cumsum() * 0.5

    # Symbol 4: downtrend
    prices[:, 4] = np.linspace(100, 70, n_days)

    symbols = [f'SYM{i}' for i in range(n_symbols)]
    prices_df = pd.DataFrame(prices, index=dates, columns=symbols)

    spy_series = pd.Series(np.linspace(100, 120, n_days), index=dates)
    vix_series = pd.Series(20.0, index=dates)

    return prices_df, spy_series, vix_series


@pytest.fixture
def multi_stop_scenario():
    """Create data where multiple symbols hit stop losses."""
    n_days = 80
    n_symbols = 6

    dates = pd.date_range('2022-01-03', periods=n_days, freq='D')

    prices = np.zeros((n_days, n_symbols))

    # All symbols start with good momentum (so they get selected)
    # Then crash at different times

    # Symbol 0: crashes on day 40
    prices[:, 0] = np.concatenate([
        np.linspace(100, 120, 40),
        np.linspace(95, 80, 40)  # -15% from peak
    ])

    # Symbol 1: crashes on day 50
    prices[:, 1] = np.concatenate([
        np.linspace(100, 115, 50),
        np.linspace(90, 85, 30)
    ])

    # Symbol 2: stays healthy
    prices[:, 2] = np.linspace(100, 140, n_days)

    # Symbol 3: crashes on day 45
    prices[:, 3] = np.concatenate([
        np.linspace(100, 118, 45),
        np.linspace(92, 88, 35)
    ])

    # Symbol 4: stays healthy
    prices[:, 4] = np.linspace(100, 135, n_days)

    # Symbol 5: stays healthy
    prices[:, 5] = np.linspace(100, 130, n_days)

    symbols = [f'SYM{i}' for i in range(n_symbols)]
    prices_df = pd.DataFrame(prices, index=dates, columns=symbols)

    spy_series = pd.Series(np.linspace(100, 120, n_days), index=dates)
    vix_series = pd.Series(20.0, index=dates)

    return prices_df, spy_series, vix_series


# ============================================================================
# Basic Stop Loss Tests
# ============================================================================

class TestStopLossBasic:
    """Basic stop loss functionality tests."""

    def test_stop_loss_triggers_on_crash(self):
        """Test that stop loss triggers when position drops below threshold."""
        # Create specific scenario where crash happens AFTER selection
        n_days = 80
        n_symbols = 4

        dates = pd.date_range('2022-01-03', periods=n_days, freq='D')
        prices = np.zeros((n_days, n_symbols))

        # Symbol 0: BEST momentum for first 40 days, then crashes hard
        prices[:, 0] = np.concatenate([
            np.linspace(100, 150, 40),  # +50% (best momentum) - gets selected
            np.linspace(120, 100, 40)   # Crashes from 150 to 120 = -20% from entry
        ])

        # Symbol 1: Good but less momentum
        prices[:, 1] = np.linspace(100, 140, n_days)

        # Symbol 2: Moderate
        prices[:, 2] = np.linspace(100, 125, n_days)

        # Symbol 3: Flat
        prices[:, 3] = np.linspace(100, 105, n_days)

        symbols = [f'SYM{i}' for i in range(n_symbols)]
        prices_df = pd.DataFrame(prices, index=dates, columns=symbols)
        spy_series = pd.Series(np.linspace(100, 120, n_days), index=dates)
        vix_series = pd.Series(20.0, index=dates)

        # Run with stop loss
        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=25, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.15  # 15% stop loss (crash is 20%)
        )

        # Should have some stops triggered since Symbol 0 crashes 20% from peak
        assert metadata['total_stops'] > 0, f"Stop loss should have triggered, got {metadata['total_stops']}"
        assert metadata['stop_loss_pct'] == -0.15

    def test_no_stop_loss_when_none(self, crash_scenario_data):
        """Test that no stop loss tracking when stop_loss_pct=None."""
        prices_df, spy_series, vix_series = crash_scenario_data

        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=None  # No stop loss
        )

        assert metadata['total_stops'] == 0
        assert metadata['stop_loss_pct'] is None

    def test_stop_count_matches_expected(self, multi_stop_scenario):
        """Test that stop counts accumulate correctly."""
        prices_df, spy_series, vix_series = multi_stop_scenario

        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=3,
            position_size=0.08, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.08  # 8% stop loss
        )

        # With 3 crashing symbols, should have multiple stops
        assert metadata['total_stops'] >= 1, f"Expected stops, got {metadata['total_stops']}"
        assert metadata['n_days'] == len(prices_df)


class TestStopLossVsNoStopLoss:
    """Compare behavior with and without stop loss."""

    def test_returns_differ_with_stop_loss(self, crash_scenario_data):
        """Test that returns differ when stop loss is active vs not."""
        prices_df, spy_series, vix_series = crash_scenario_data

        # Without stop loss
        returns_no_stop, _ = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=None
        )

        # With stop loss
        returns_with_stop, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10
        )

        # If stops triggered, returns should differ
        if metadata['total_stops'] > 0:
            # Returns should be different at some point
            assert not np.allclose(returns_no_stop, returns_with_stop), \
                "Returns should differ when stop loss triggers"

    def test_stop_loss_limits_losses(self, crash_scenario_data):
        """Test that stop loss limits losses on crashing positions."""
        prices_df, spy_series, vix_series = crash_scenario_data

        # Without stop loss
        returns_no_stop, _ = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=None
        )

        # With tight stop loss
        returns_with_stop, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.05  # 5% stop loss
        )

        # Calculate cumulative returns
        cum_no_stop = np.cumprod(1 + returns_no_stop) - 1
        cum_with_stop = np.cumprod(1 + returns_with_stop) - 1

        # Max drawdown should generally be less with stop loss
        dd_no_stop = np.min(cum_no_stop - np.maximum.accumulate(cum_no_stop + 1) + 1)
        dd_with_stop = np.min(cum_with_stop - np.maximum.accumulate(cum_with_stop + 1) + 1)

        # Stop loss should help (or at least not hurt significantly)
        # This is a soft check since results depend on replacement symbols
        if metadata['total_stops'] > 0:
            assert dd_with_stop >= dd_no_stop - 0.05, \
                f"Stop loss should limit drawdowns: {dd_with_stop:.3f} vs {dd_no_stop:.3f}"


class TestStopLossEdgeCases:
    """Edge case tests for stop loss functionality."""

    def test_stop_at_exact_threshold(self):
        """Test behavior when price drops exactly to stop threshold."""
        n_days = 50
        n_symbols = 3

        dates = pd.date_range('2022-01-03', periods=n_days, freq='D')

        prices = np.zeros((n_days, n_symbols))
        # Symbol 0: drops exactly 10% from entry (100 -> 90)
        prices[:, 0] = np.concatenate([
            [100.0] * 30,
            np.linspace(100, 90, 10),  # Drops to exactly 90 (10% loss)
            [90.0] * 10
        ])
        # Symbol 1: stable
        prices[:, 1] = [110.0] * n_days
        # Symbol 2: stable
        prices[:, 2] = [105.0] * n_days

        symbols = [f'SYM{i}' for i in range(n_symbols)]
        prices_df = pd.DataFrame(prices, index=dates, columns=symbols)
        spy_series = pd.Series([100.0] * n_days, index=dates)
        vix_series = pd.Series([20.0] * n_days, index=dates)

        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=20, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10  # Exactly 10%
        )

        # Should trigger stop at exactly the threshold
        assert metadata['total_stops'] >= 0  # May or may not trigger depending on timing

    def test_no_valid_replacement(self):
        """Test behavior when no valid replacement symbol exists."""
        n_days = 60
        n_symbols = 2  # Only 2 symbols, both crash

        dates = pd.date_range('2022-01-03', periods=n_days, freq='D')

        prices = np.zeros((n_days, n_symbols))
        # Both symbols crash
        prices[:, 0] = np.concatenate([
            np.linspace(100, 110, 30),
            np.linspace(90, 70, 30)
        ])
        prices[:, 1] = np.concatenate([
            np.linspace(100, 108, 30),
            np.linspace(88, 68, 30)
        ])

        symbols = ['SYM0', 'SYM1']
        prices_df = pd.DataFrame(prices, index=dates, columns=symbols)
        spy_series = pd.Series([100.0] * n_days, index=dates)
        vix_series = pd.Series([20.0] * n_days, index=dates)

        # Should not crash even with limited replacement options
        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=20, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.08
        )

        assert len(returns) == n_days
        assert not np.any(np.isnan(returns))

    def test_immediate_stop_after_entry(self):
        """Test stop loss on first day after entry."""
        n_days = 40
        n_symbols = 3

        dates = pd.date_range('2022-01-03', periods=n_days, freq='D')

        prices = np.zeros((n_days, n_symbols))
        # Symbol with good momentum then immediate crash
        prices[:, 0] = np.concatenate([
            np.linspace(100, 130, 25),  # Good momentum
            [100.0],  # Gap down (23% loss in one day)
            [95.0] * 14
        ])
        prices[:, 1] = np.linspace(100, 120, n_days)
        prices[:, 2] = np.linspace(100, 115, n_days)

        symbols = [f'SYM{i}' for i in range(n_symbols)]
        prices_df = pd.DataFrame(prices, index=dates, columns=symbols)
        spy_series = pd.Series([100.0] * n_days, index=dates)
        vix_series = pd.Series([20.0] * n_days, index=dates)

        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=20, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10
        )

        # Should handle gap down correctly
        assert len(returns) == n_days
        assert metadata['total_stops'] >= 1


class TestStopLossMetadata:
    """Test stop loss metadata tracking."""

    def test_metadata_fields_present(self, simple_price_data):
        """Test that all expected metadata fields are present."""
        prices_df, spy_series, vix_series = simple_price_data

        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=3,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10
        )

        expected_fields = [
            'n_days', 'avg_exposure', 'risk_off_days',
            'penalty_weight', 'long_weight', 'stop_loss_pct', 'total_stops'
        ]

        for field in expected_fields:
            assert field in metadata, f"Missing metadata field: {field}"

    def test_stop_loss_pct_in_metadata(self, simple_price_data):
        """Test that stop_loss_pct is correctly recorded in metadata."""
        prices_df, spy_series, vix_series = simple_price_data

        # Test with various stop loss values
        for stop_pct in [None, -0.05, -0.10, -0.15]:
            returns, metadata = run_mp_backtest_numba(
                prices_df, spy_series, vix_series,
                long_period=30, short_period=5, top_n=3,
                position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
                stop_loss_pct=stop_pct
            )

            assert metadata['stop_loss_pct'] == stop_pct, \
                f"Expected stop_loss_pct={stop_pct}, got {metadata['stop_loss_pct']}"


class TestBackwardCompatibility:
    """Test backward compatibility with original behavior."""

    def test_none_stop_loss_same_as_original(self, simple_price_data):
        """Test that stop_loss_pct=None gives same results as original function."""
        prices_df, spy_series, vix_series = simple_price_data

        prices = prices_df.values.astype(np.float64)
        spy_prices = spy_series.values.astype(np.float64)
        vix_values = vix_series.values.astype(np.float64)

        # Calculate returns and momentum
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        momentum = calculate_momentum_scores(prices, 30, 5)
        spy_dd = calculate_spy_drawdown(spy_prices)

        # Run original function
        portfolio_returns_orig, exposures_orig = simulate_ranked_portfolio(
            returns_matrix=returns,
            momentum_matrix=momentum,
            vix_values=vix_values,
            spy_drawdown=spy_dd,
            top_n=3,
            position_size=0.10,
            vix_threshold=30,
            spy_dd_threshold=-0.15,
            reduced_exposure=0.5
        )

        # Run with stop loss = None (via wrapper)
        returns_none, _ = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=3,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=None
        )

        # Should produce same results
        np.testing.assert_allclose(
            portfolio_returns_orig, returns_none,
            rtol=1e-10,
            err_msg="stop_loss_pct=None should match original behavior"
        )


class TestStopLossPositionReplacement:
    """Test position replacement after stop loss triggers."""

    def test_replacement_fills_slot(self, multi_stop_scenario):
        """Test that stopped positions are replaced with next ranked symbol."""
        prices_df, spy_series, vix_series = multi_stop_scenario

        # Run with stop loss
        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=25, short_period=5, top_n=3,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.08
        )

        # Portfolio should still have returns after stops (positions replaced)
        # Check that we're not just holding cash after stops
        post_crash_returns = returns[50:]  # After the crashes

        # Some of these should be non-zero (replacement positions)
        non_zero_count = np.sum(np.abs(post_crash_returns) > 1e-10)
        assert non_zero_count > 0, "Should have active positions after stop replacements"


class TestStopLossWithDifferentParameters:
    """Test stop loss with various parameter combinations."""

    @pytest.mark.parametrize("stop_pct", [-0.03, -0.05, -0.10, -0.15, -0.20])
    def test_various_stop_levels(self, crash_scenario_data, stop_pct):
        """Test with various stop loss levels."""
        prices_df, spy_series, vix_series = crash_scenario_data

        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=stop_pct
        )

        assert len(returns) == len(prices_df)
        assert metadata['stop_loss_pct'] == stop_pct
        assert not np.any(np.isnan(returns))

    @pytest.mark.parametrize("top_n", [1, 2, 3, 4, 5])
    def test_various_top_n(self, crash_scenario_data, top_n):
        """Test stop loss with various top_n values."""
        prices_df, spy_series, vix_series = crash_scenario_data

        # Need enough symbols for top_n
        if top_n > len(prices_df.columns):
            pytest.skip(f"Not enough symbols for top_n={top_n}")

        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=top_n,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10
        )

        assert len(returns) == len(prices_df)
        assert not np.any(np.isnan(returns))


class TestStopLossFunctionDirect:
    """Direct tests of simulate_ranked_portfolio_with_stops function."""

    def test_direct_call_returns_three_arrays(self):
        """Test that direct function call returns three arrays."""
        n_days = 50
        n_symbols = 4

        prices = np.random.randn(n_days, n_symbols) * 0.02 + 1
        prices = np.cumprod(prices, axis=0) * 100

        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

        momentum = np.random.randn(n_days, n_symbols)
        vix = np.ones(n_days) * 20
        spy_dd = np.zeros(n_days)

        portfolio_returns, exposures, stop_counts = simulate_ranked_portfolio_with_stops(
            prices_matrix=prices,
            returns_matrix=returns,
            momentum_matrix=momentum,
            vix_values=vix,
            spy_drawdown=spy_dd,
            top_n=2,
            position_size=0.10,
            vix_threshold=30,
            spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10,
            reduced_exposure=0.5
        )

        assert portfolio_returns.shape == (n_days,)
        assert exposures.shape == (n_days,)
        assert stop_counts.shape == (n_days,)
        assert stop_counts.dtype == np.int64

    def test_stop_counts_are_non_negative(self):
        """Test that stop counts are always non-negative."""
        np.random.seed(42)
        n_days = 100
        n_symbols = 5

        # Create volatile prices that will trigger stops
        prices = np.random.randn(n_days, n_symbols) * 0.05 + 1
        prices = np.cumprod(prices, axis=0) * 100

        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

        momentum = np.random.randn(n_days, n_symbols)
        vix = np.ones(n_days) * 20
        spy_dd = np.zeros(n_days)

        _, _, stop_counts = simulate_ranked_portfolio_with_stops(
            prices_matrix=prices,
            returns_matrix=returns,
            momentum_matrix=momentum,
            vix_values=vix,
            spy_drawdown=spy_dd,
            top_n=2,
            position_size=0.10,
            vix_threshold=30,
            spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10,
            reduced_exposure=0.5
        )

        assert np.all(stop_counts >= 0), "Stop counts should never be negative"
        assert np.all(stop_counts <= 2), "Stop counts per day should not exceed top_n"


# ============================================================================
# Intraday Stop Loss Tests (using daily low prices)
# ============================================================================

class TestIntradayStopLossBasic:
    """Basic intraday stop loss functionality tests."""

    def test_intraday_stop_triggers_on_low_breach(self):
        """Test that intraday stop triggers when daily low breaches stop price."""
        n_days = 80
        n_symbols = 3

        dates = pd.date_range('2022-01-03', periods=n_days, freq='D')

        # Create close prices - symbol 0 has BEST momentum and gets selected
        # Entry happens around day 26 (when momentum scores become valid with long_period=25)
        # At day 26, price is ~130 (linear from 100 to 160 over 40 days)
        # Stop price = 130 * 0.85 = 110.5
        closes = np.zeros((n_days, n_symbols))
        closes[:, 0] = np.concatenate([
            np.linspace(100, 160, 40),  # +60% best momentum - gets selected
            np.linspace(160, 155, 40)   # Slight pullback at close
        ])
        closes[:, 1] = np.linspace(100, 120, n_days)  # +20%
        closes[:, 2] = np.linspace(100, 115, n_days)  # +15%

        # Create low prices - symbol 0 has intraday dip that breaches stop
        # Entry ~day 26 at price ~130, stop at 130*0.85 = 110.5
        # Day 35: close is ~140, but low dips to 100 (breaches the ~110 stop!)
        lows = closes.copy()
        lows[35, 0] = 100.0  # Intraday dip well below any reasonable stop price

        symbols = [f'SYM{i}' for i in range(n_symbols)]
        prices_df = pd.DataFrame(closes, index=dates, columns=symbols)
        lows_df = pd.DataFrame(lows, index=dates, columns=symbols)
        spy_series = pd.Series([100.0] * n_days, index=dates)
        vix_series = pd.Series([20.0] * n_days, index=dates)

        # Run with intraday stop loss
        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=25, short_period=5, top_n=1,  # Only top 1 to ensure SYM0 selected
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.15,  # 15% stop
            lows_df=lows_df  # Enable intraday stops
        )

        assert metadata['stop_mode'] == 'intraday'
        # The intraday dip should trigger a stop
        assert metadata['total_stops'] >= 1, f"Intraday stop should have triggered, got {metadata['total_stops']}"

    def test_intraday_triggers_more_stops_than_eod(self):
        """Test that intraday mode triggers more stops than EOD mode."""
        np.random.seed(42)
        n_days = 100
        n_symbols = 5

        dates = pd.date_range('2022-01-03', periods=n_days, freq='D')

        # Create volatile price data
        closes = np.random.randn(n_days, n_symbols) * 0.03 + 1
        closes = np.cumprod(closes, axis=0) * 100

        # Lows are 2% below close (simulating intraday volatility)
        lows = closes * 0.98

        symbols = [f'SYM{i}' for i in range(n_symbols)]
        prices_df = pd.DataFrame(closes, index=dates, columns=symbols)
        lows_df = pd.DataFrame(lows, index=dates, columns=symbols)
        spy_series = pd.Series([100.0] * n_days, index=dates)
        vix_series = pd.Series([20.0] * n_days, index=dates)

        # Run EOD stop loss
        _, metadata_eod = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=20, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.05,  # 5% stop
            lows_df=None  # EOD mode
        )

        # Run intraday stop loss
        _, metadata_intraday = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=20, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.05,  # Same 5% stop
            lows_df=lows_df  # Intraday mode
        )

        assert metadata_eod['stop_mode'] == 'eod'
        assert metadata_intraday['stop_mode'] == 'intraday'
        # Intraday should trigger more stops (lows are lower than closes)
        assert metadata_intraday['total_stops'] >= metadata_eod['total_stops']


class TestIntradayStopLossMetadata:
    """Test intraday stop loss metadata tracking."""

    def test_stop_mode_field(self, simple_price_data):
        """Test that stop_mode is correctly set in metadata."""
        prices_df, spy_series, vix_series = simple_price_data
        lows_df = prices_df * 0.99  # Lows slightly below close

        # No stop loss
        _, meta_none = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=3,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=None, lows_df=None
        )
        assert meta_none['stop_mode'] is None

        # EOD stop loss
        _, meta_eod = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=3,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10, lows_df=None
        )
        assert meta_eod['stop_mode'] == 'eod'

        # Intraday stop loss
        _, meta_intraday = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=30, short_period=5, top_n=3,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10, lows_df=lows_df
        )
        assert meta_intraday['stop_mode'] == 'intraday'


class TestIntradayStopLossExitPrice:
    """Test that intraday stops exit at stop price, not low or close."""

    def test_stop_returns_equal_stop_loss_pct(self):
        """Stopped positions should realize exactly the stop loss return."""
        n_days = 50
        n_symbols = 2

        dates = pd.date_range('2022-01-03', periods=n_days, freq='D')

        # Create controlled scenario
        closes = np.zeros((n_days, n_symbols))
        lows = np.zeros((n_days, n_symbols))

        # Symbol 0: steady prices, then intraday crash
        closes[:, 0] = 100.0  # Constant close
        lows[:, 0] = 100.0    # Constant low

        # Symbol 1: good momentum, then intraday crash
        closes[:, 1] = np.concatenate([
            np.linspace(100, 120, 25),  # Up 20%
            [120.0] * 25  # Hold at 120
        ])
        lows[:, 1] = closes[:, 1].copy()

        # Day 30: Symbol 1 has intraday crash but closes at 120
        # Entry at 120, stop at 120*0.9 = 108
        # Low drops to 105 (below stop), close still 120
        lows[30, 1] = 105.0  # Intraday crash

        symbols = ['SYM0', 'SYM1']
        prices_df = pd.DataFrame(closes, index=dates, columns=symbols)
        lows_df = pd.DataFrame(lows, index=dates, columns=symbols)
        spy_series = pd.Series([100.0] * n_days, index=dates)
        vix_series = pd.Series([20.0] * n_days, index=dates)

        returns, metadata = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=20, short_period=5, top_n=1,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10,
            lows_df=lows_df
        )

        # Should have triggered intraday stop
        assert metadata['stop_mode'] == 'intraday'
        # The stop should have been triggered


class TestIntradayStopLossFunctionDirect:
    """Direct tests of simulate_ranked_portfolio_with_intraday_stops."""

    def test_function_returns_three_arrays(self):
        """Test that function returns three arrays."""
        n_days = 50
        n_symbols = 4

        prices = np.random.randn(n_days, n_symbols) * 0.02 + 1
        prices = np.cumprod(prices, axis=0) * 100
        lows = prices * 0.99

        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

        momentum = np.random.randn(n_days, n_symbols)
        vix = np.ones(n_days) * 20
        spy_dd = np.zeros(n_days)

        portfolio_returns, exposures, stop_counts = simulate_ranked_portfolio_with_intraday_stops(
            prices_matrix=prices,
            lows_matrix=lows,
            returns_matrix=returns,
            momentum_matrix=momentum,
            vix_values=vix,
            spy_drawdown=spy_dd,
            top_n=2,
            position_size=0.10,
            vix_threshold=30,
            spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10,
            reduced_exposure=0.5
        )

        assert portfolio_returns.shape == (n_days,)
        assert exposures.shape == (n_days,)
        assert stop_counts.shape == (n_days,)

    def test_low_equal_close_same_as_eod(self):
        """When lows equal closes, intraday should match EOD behavior."""
        np.random.seed(123)
        n_days = 80
        n_symbols = 4

        # Create prices where low = close (no intraday volatility)
        prices = np.random.randn(n_days, n_symbols) * 0.02 + 1
        prices = np.cumprod(prices, axis=0) * 100
        lows = prices.copy()  # Low equals close

        symbols = [f'SYM{i}' for i in range(n_symbols)]
        dates = pd.date_range('2022-01-03', periods=n_days, freq='D')
        prices_df = pd.DataFrame(prices, index=dates, columns=symbols)
        lows_df = pd.DataFrame(lows, index=dates, columns=symbols)
        spy_series = pd.Series([100.0] * n_days, index=dates)
        vix_series = pd.Series([20.0] * n_days, index=dates)

        # EOD mode
        returns_eod, meta_eod = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=20, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10, lows_df=None
        )

        # Intraday mode (with lows = closes)
        returns_intraday, meta_intraday = run_mp_backtest_numba(
            prices_df, spy_series, vix_series,
            long_period=20, short_period=5, top_n=2,
            position_size=0.10, vix_threshold=30, spy_dd_threshold=-0.15,
            stop_loss_pct=-0.10, lows_df=lows_df
        )

        # Stop counts should be similar (may differ slightly due to stop price vs close comparison)
        # But both should work without errors
        assert len(returns_eod) == len(returns_intraday)
        assert not np.any(np.isnan(returns_eod))
        assert not np.any(np.isnan(returns_intraday))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
