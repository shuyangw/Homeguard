"""
Parity tests for Numba JIT-compiled portfolio simulation.

These tests verify that the Numba simulation produces identical results
to the Python simulation across various configurations.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtesting.engine.portfolio_simulator import Portfolio, from_signals
from src.backtesting.utils.risk_config import RiskConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_price_data():
    """Simple ascending price series for basic tests."""
    dates = pd.date_range('2022-01-03 10:00:00', periods=100, freq='D', tz='US/Eastern')
    prices = pd.Series([100 + i * 0.5 for i in range(100)], index=dates)
    return prices


@pytest.fixture
def volatile_price_data():
    """Volatile price series with ups and downs."""
    np.random.seed(42)
    dates = pd.date_range('2022-01-03 10:00:00', periods=200, freq='D', tz='US/Eastern')
    base = 100
    returns = np.random.randn(200) * 0.02  # 2% daily vol
    prices = base * (1 + returns).cumprod()
    return pd.Series(prices, index=dates)


@pytest.fixture
def downtrend_price_data():
    """Downtrending price series for short selling tests."""
    dates = pd.date_range('2022-01-03 10:00:00', periods=50, freq='D', tz='US/Eastern')
    prices = pd.Series([100 - i * 0.3 for i in range(50)], index=dates)
    return prices


# ============================================================================
# Basic Parity Tests
# ============================================================================

class TestNumbaVsPythonParity:
    """Test that Numba and Python produce identical results."""

    def test_basic_long_only(self, simple_price_data):
        """Test basic long-only simulation parity."""
        prices = simple_price_data
        n = len(prices)

        # Entry on day 5, exit on day 20
        entries = pd.Series([i == 5 for i in range(n)], index=prices.index)
        exits = pd.Series([i == 20 for i in range(n)], index=prices.index)

        # Python simulation
        portfolio_py = Portfolio(
            price=prices,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.001,
            slippage=0.0005,
            market_hours_only=False,
            use_numba=False
        )

        # Numba simulation
        portfolio_nb = Portfolio(
            price=prices,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.001,
            slippage=0.0005,
            market_hours_only=False,
            use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="Equity curves differ between Python and Numba"
        )

        # Compare trade counts
        assert len(portfolio_py.trades) == len(portfolio_nb.trades), \
            f"Trade count differs: Python={len(portfolio_py.trades)}, Numba={len(portfolio_nb.trades)}"

        # Compare stats
        stats_py = portfolio_py.stats()
        stats_nb = portfolio_nb.stats()
        assert stats_py is not None and stats_nb is not None

        np.testing.assert_allclose(
            stats_py['Total Return [%]'],
            stats_nb['Total Return [%]'],
            rtol=1e-10,
            err_msg="Total returns differ"
        )

    @pytest.mark.parametrize("allow_shorts", [False, True])
    def test_with_shorts(self, volatile_price_data, allow_shorts):
        """Test parity with short selling enabled/disabled."""
        prices = volatile_price_data
        n = len(prices)

        # Generate alternating signals
        entries = pd.Series([i % 20 == 0 for i in range(n)], index=prices.index)
        exits = pd.Series([i % 20 == 10 for i in range(n)], index=prices.index)

        # Disable stop loss - Python RiskManager has bug with shorts
        risk_config = RiskConfig(use_stop_loss=False)

        # Python simulation
        portfolio_py = Portfolio(
            price=prices,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.001,
            slippage=0.0005,
            allow_shorts=allow_shorts,
            risk_config=risk_config,
            market_hours_only=False,
            use_numba=False
        )

        # Numba simulation
        portfolio_nb = Portfolio(
            price=prices,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.001,
            slippage=0.0005,
            allow_shorts=allow_shorts,
            risk_config=risk_config,
            market_hours_only=False,
            use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg=f"Equity curves differ (allow_shorts={allow_shorts})"
        )

        # Compare trade types
        for py_trade, nb_trade in zip(portfolio_py.trades, portfolio_nb.trades):
            assert py_trade['type'] == nb_trade['type'], \
                f"Trade type mismatch: Python={py_trade['type']}, Numba={nb_trade['type']}"

    @pytest.mark.parametrize("use_stop_loss", [False, True])
    def test_with_stop_loss(self, volatile_price_data, use_stop_loss):
        """Test parity with stop loss enabled/disabled."""
        prices = volatile_price_data
        n = len(prices)

        entries = pd.Series([i == 10 for i in range(n)], index=prices.index)
        exits = pd.Series([i == n - 1 for i in range(n)], index=prices.index)

        risk_config = RiskConfig(
            position_size_pct=0.10,
            use_stop_loss=use_stop_loss,
            stop_loss_pct=0.05,
            stop_loss_type='percentage'
        )

        # Python simulation
        portfolio_py = Portfolio(
            price=prices,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.001,
            slippage=0.0005,
            risk_config=risk_config,
            market_hours_only=False,
            use_numba=False
        )

        # Numba simulation
        portfolio_nb = Portfolio(
            price=prices,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.001,
            slippage=0.0005,
            risk_config=risk_config,
            market_hours_only=False,
            use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg=f"Equity curves differ (use_stop_loss={use_stop_loss})"
        )

    def test_short_pnl_calculation(self, downtrend_price_data):
        """Test that short P&L is calculated correctly in both implementations."""
        prices = downtrend_price_data
        n = len(prices)

        # Short entry on day 2, cover on day 40
        entries = pd.Series([i == 40 for i in range(n)], index=prices.index)
        exits = pd.Series([i == 2 for i in range(n)], index=prices.index)

        # Disable stop loss for this test - the Python RiskManager has a bug with shorts
        # where it applies long-position stop logic to short positions
        risk_config = RiskConfig(use_stop_loss=False)

        # Python simulation
        portfolio_py = Portfolio(
            price=prices,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0,
            slippage=0,
            allow_shorts=True,
            risk_config=risk_config,
            market_hours_only=False,
            use_numba=False
        )

        # Numba simulation
        portfolio_nb = Portfolio(
            price=prices,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0,
            slippage=0,
            allow_shorts=True,
            risk_config=risk_config,
            market_hours_only=False,
            use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="Short position equity curves differ"
        )

        # Both should have positive return (short in downtrend)
        stats_py = portfolio_py.stats()
        stats_nb = portfolio_nb.stats()
        assert stats_py['Total Return [%]'] > 0, "Python short didn't profit in downtrend"
        assert stats_nb['Total Return [%]'] > 0, "Numba short didn't profit in downtrend"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_trades(self, simple_price_data):
        """Test when no signals trigger trades."""
        prices = simple_price_data
        n = len(prices)

        # No entry or exit signals
        entries = pd.Series([False] * n, index=prices.index)
        exits = pd.Series([False] * n, index=prices.index)

        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0,
            market_hours_only=False, use_numba=False
        )

        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0,
            market_hours_only=False, use_numba=True
        )

        assert len(portfolio_py.trades) == 0
        assert len(portfolio_nb.trades) == 0
        np.testing.assert_array_equal(portfolio_py.equity_curve.values, portfolio_nb.equity_curve.values)

    def test_many_trades(self, volatile_price_data):
        """Test with many trades (stress test)."""
        prices = volatile_price_data
        n = len(prices)

        # Trade every 5 days
        entries = pd.Series([i % 10 < 5 for i in range(n)], index=prices.index)
        exits = pd.Series([i % 10 >= 5 for i in range(n)], index=prices.index)

        # Disable stop loss for cleaner parity comparison
        risk_config = RiskConfig(use_stop_loss=False)

        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config,
            market_hours_only=False, use_numba=False
        )

        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config,
            market_hours_only=False, use_numba=True
        )

        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10
        )


# ============================================================================
# Fallback Tests
# ============================================================================

class TestFallback:
    """Test that fallback to Python works correctly."""

    def test_fallback_when_disabled(self, simple_price_data):
        """Test that use_numba=False forces Python path."""
        prices = simple_price_data
        n = len(prices)
        entries = pd.Series([i == 5 for i in range(n)], index=prices.index)
        exits = pd.Series([i == 20 for i in range(n)], index=prices.index)

        portfolio = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0,
            market_hours_only=False, use_numba=False
        )

        # Should still work
        assert portfolio.equity_curve is not None
        assert len(portfolio.equity_curve) == n

    def test_fallback_for_atr_stops(self, simple_price_data):
        """Test fallback when ATR stops are used (not supported in Numba)."""
        prices = simple_price_data
        n = len(prices)
        entries = pd.Series([i == 5 for i in range(n)], index=prices.index)
        exits = pd.Series([i == 20 for i in range(n)], index=prices.index)

        risk_config = RiskConfig(
            position_size_pct=0.10,
            use_stop_loss=True,
            stop_loss_type='atr',
            atr_multiplier=2.0
        )

        portfolio = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0,
            risk_config=risk_config,
            market_hours_only=False,
            use_numba=True  # Will fallback to Python
        )

        # Should still work via Python fallback
        assert portfolio.equity_curve is not None


# ============================================================================
# Performance Benchmark
# ============================================================================

class TestPerformance:
    """Test that Numba provides expected speedup."""

    @pytest.mark.slow
    def test_numba_faster_than_python(self):
        """Verify Numba is significantly faster than Python."""
        # Generate large dataset
        n = 10000
        np.random.seed(42)
        dates = pd.date_range('2020-01-01 10:00:00', periods=n, freq='T', tz='US/Eastern')
        prices = pd.Series(100 * (1 + np.random.randn(n) * 0.001).cumprod(), index=dates)

        # Generate signals
        entries = pd.Series([i % 100 < 50 for i in range(n)], index=prices.index)
        exits = pd.Series([i % 100 >= 50 for i in range(n)], index=prices.index)

        # Time Python
        start = time.time()
        for _ in range(3):
            Portfolio(
                price=prices, entries=entries, exits=exits,
                init_cash=10000, fees=0.001, slippage=0,
                market_hours_only=False, use_numba=False
            )
        python_time = (time.time() - start) / 3

        # Warm up Numba (first call compiles)
        Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0,
            market_hours_only=False, use_numba=True
        )

        # Time Numba
        start = time.time()
        for _ in range(3):
            Portfolio(
                price=prices, entries=entries, exits=exits,
                init_cash=10000, fees=0.001, slippage=0,
                market_hours_only=False, use_numba=True
            )
        numba_time = (time.time() - start) / 3

        speedup = python_time / numba_time
        print(f"\nPython time: {python_time:.4f}s")
        print(f"Numba time: {numba_time:.4f}s")
        print(f"Speedup: {speedup:.1f}x")

        # Numba should be at least 5x faster
        assert speedup > 5, f"Expected >5x speedup, got {speedup:.1f}x"


# ============================================================================
# from_signals Parity
# ============================================================================

class TestFromSignalsParity:
    """Test that from_signals() works with both paths."""

    def test_from_signals_numba(self, simple_price_data):
        """Test from_signals with Numba."""
        prices = simple_price_data
        n = len(prices)
        entries = pd.Series([i == 5 for i in range(n)], index=prices.index)
        exits = pd.Series([i == 20 for i in range(n)], index=prices.index)

        portfolio = from_signals(
            close=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0,
            market_hours_only=False, use_numba=True
        )

        assert portfolio.equity_curve is not None
        stats = portfolio.stats()
        assert stats['Total Trades'] >= 1

    def test_from_signals_python(self, simple_price_data):
        """Test from_signals with Python."""
        prices = simple_price_data
        n = len(prices)
        entries = pd.Series([i == 5 for i in range(n)], index=prices.index)
        exits = pd.Series([i == 20 for i in range(n)], index=prices.index)

        portfolio = from_signals(
            close=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0,
            market_hours_only=False, use_numba=False
        )

        assert portfolio.equity_curve is not None
        stats = portfolio.stats()
        assert stats['Total Trades'] >= 1


class TestAdvancedFeatureParity:
    """Test parity for advanced features: profit target, time stop, market hours."""

    def test_profit_target_parity(self):
        """Test profit target exit produces identical results."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
        # Create uptrending prices so profit target can trigger
        prices = pd.Series(100 * (1 + np.random.randn(n) * 0.02).cumprod(), index=dates)

        entries = pd.Series([i % 30 == 0 for i in range(n)], index=dates)
        exits = pd.Series([i % 30 == 25 for i in range(n)], index=dates)

        # Configure with profit target (5% take profit)
        # Note: stop_loss_type must be 'profit_target' to activate take_profit_pct
        risk_config = RiskConfig(
            use_stop_loss=True,
            stop_loss_type='profit_target',
            stop_loss_pct=0.10,  # Also includes stop loss in profit_target mode
            take_profit_pct=0.05,  # 5% profit target
            position_size_pct=0.10
        )

        # Run Python
        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=False
        )

        # Run Numba
        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="Profit target: Equity curves differ"
        )

        # Compare trade counts
        assert len(portfolio_py.trades) == len(portfolio_nb.trades), \
            f"Profit target: Trade count mismatch - Python={len(portfolio_py.trades)}, Numba={len(portfolio_nb.trades)}"

    def test_time_stop_parity(self):
        """Test time-based stop exit produces identical results."""
        np.random.seed(123)
        n = 200
        dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
        prices = pd.Series(100 * (1 + np.random.randn(n) * 0.015).cumprod(), index=dates)

        # Entry every 30 bars, but exit signal at bar 50 (should be stopped out by time first)
        entries = pd.Series([i % 60 == 0 for i in range(n)], index=dates)
        exits = pd.Series([i % 60 == 50 for i in range(n)], index=dates)

        # Configure with time stop (max 10 bars holding)
        # Note: stop_loss_type must be 'time' to activate max_holding_bars
        risk_config = RiskConfig(
            use_stop_loss=True,
            stop_loss_type='time',
            max_holding_bars=10,  # Force exit after 10 bars
            position_size_pct=0.10
        )

        # Run Python
        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=False
        )

        # Run Numba
        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="Time stop: Equity curves differ"
        )

        # Compare trade counts
        assert len(portfolio_py.trades) == len(portfolio_nb.trades), \
            f"Time stop: Trade count mismatch - Python={len(portfolio_py.trades)}, Numba={len(portfolio_nb.trades)}"

    def test_market_hours_parity(self):
        """Test market hours filtering produces identical results."""
        np.random.seed(456)
        # Create intraday data spanning market hours and after-hours
        n = 500
        # Start at 8:00 AM, run through multiple days with minute frequency
        dates = pd.date_range('2022-01-03 08:00:00', periods=n, freq='min', tz='US/Eastern')
        prices = pd.Series(100 * (1 + np.random.randn(n) * 0.001).cumprod(), index=dates)

        # Signals that span market hours and after-hours
        entries = pd.Series([i % 100 == 10 for i in range(n)], index=dates)
        exits = pd.Series([i % 100 == 60 for i in range(n)], index=dates)

        risk_config = RiskConfig(use_stop_loss=False, position_size_pct=0.10)

        # Run Python with market hours filtering
        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=True, use_numba=False
        )

        # Run Numba with market hours filtering
        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=True, use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="Market hours: Equity curves differ"
        )

        # Compare trade counts
        assert len(portfolio_py.trades) == len(portfolio_nb.trades), \
            f"Market hours: Trade count mismatch - Python={len(portfolio_py.trades)}, Numba={len(portfolio_nb.trades)}"

    def test_short_with_stop_loss_parity(self):
        """Test short positions with stop loss produce identical results."""
        np.random.seed(789)
        n = 200
        dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
        # Create volatile uptrending prices (bad for shorts - will trigger stop loss)
        prices = pd.Series(100 * (1 + np.random.randn(n) * 0.025 + 0.002).cumprod(), index=dates)

        # Exit signals trigger short entry when allow_shorts=True
        entries = pd.Series([i % 40 == 30 for i in range(n)], index=dates)  # Long entry
        exits = pd.Series([i % 40 == 10 for i in range(n)], index=dates)    # Short entry

        # Configure with stop loss for shorts
        risk_config = RiskConfig(
            use_stop_loss=True,
            stop_loss_type='percentage',
            stop_loss_pct=0.05,  # 5% stop loss
            position_size_pct=0.10
        )

        # Run Python with shorts and stop loss
        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False,
            allow_shorts=True, use_numba=False
        )

        # Run Numba with shorts and stop loss
        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False,
            allow_shorts=True, use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="Short with stop loss: Equity curves differ"
        )

        # Compare trade counts
        assert len(portfolio_py.trades) == len(portfolio_nb.trades), \
            f"Short with stop loss: Trade count mismatch - Python={len(portfolio_py.trades)}, Numba={len(portfolio_nb.trades)}"

        # Verify some trades occurred
        assert len(portfolio_py.trades) > 0, "No trades executed"

    def test_short_with_profit_target_parity(self):
        """Test short positions with profit_target mode (both stop loss AND take profit)."""
        np.random.seed(321)
        n = 200
        dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
        # Create volatile downtrending prices (good for shorts - may hit profit target)
        prices = pd.Series(100 * (1 + np.random.randn(n) * 0.02 - 0.001).cumprod(), index=dates)

        # Exit signals trigger short entry when allow_shorts=True
        entries = pd.Series([i % 50 == 40 for i in range(n)], index=dates)  # Long entry
        exits = pd.Series([i % 50 == 10 for i in range(n)], index=dates)    # Short entry

        # Configure with profit_target mode (includes BOTH stop loss AND take profit)
        risk_config = RiskConfig(
            use_stop_loss=True,
            stop_loss_type='profit_target',
            stop_loss_pct=0.08,    # 8% stop loss
            take_profit_pct=0.05,  # 5% take profit
            position_size_pct=0.10
        )

        # Run Python
        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False,
            allow_shorts=True, use_numba=False
        )

        # Run Numba
        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False,
            allow_shorts=True, use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="Short with profit_target: Equity curves differ"
        )

        # Compare trade counts
        assert len(portfolio_py.trades) == len(portfolio_nb.trades), \
            f"Short with profit_target: Trade count mismatch"


class TestEdgeCases:
    """Test edge cases and boundary conditions for 100% confidence."""

    def test_same_bar_entry_exit(self):
        """Test when entry and exit signals fire on the same bar."""
        np.random.seed(111)
        n = 100
        dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
        prices = pd.Series(100 * (1 + np.random.randn(n) * 0.01).cumprod(), index=dates)

        # Create overlapping signals - both entry and exit on some bars
        entries = pd.Series([i % 10 == 0 for i in range(n)], index=dates)
        exits = pd.Series([i % 10 == 0 or i % 10 == 5 for i in range(n)], index=dates)

        risk_config = RiskConfig(use_stop_loss=False, position_size_pct=0.10)

        # Run Python
        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=False
        )

        # Run Numba
        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="Same bar entry/exit: Equity curves differ"
        )

        assert len(portfolio_py.trades) == len(portfolio_nb.trades)

    def test_price_exactly_at_stop(self):
        """Test when price lands exactly on stop loss price."""
        n = 50
        dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')

        # Create price series where price drops exactly 5% (stop loss level)
        # Entry at bar 5, price = 100
        # Stop at 5% = price 95
        # Bar 10 price drops to exactly 95
        prices_list = [100.0] * n
        prices_list[10] = 95.0  # Exactly at 5% stop loss from 100
        prices_list[11:] = [94.0] * (n - 11)  # Continue lower
        prices = pd.Series(prices_list, index=dates)

        entries = pd.Series([i == 5 for i in range(n)], index=dates)
        exits = pd.Series([i == n - 1 for i in range(n)], index=dates)

        risk_config = RiskConfig(
            use_stop_loss=True,
            stop_loss_type='percentage',
            stop_loss_pct=0.05,  # 5% stop
            position_size_pct=0.10
        )

        # Run Python
        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=False
        )

        # Run Numba
        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=True
        )

        # Compare equity curves
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="Price exactly at stop: Equity curves differ"
        )

        assert len(portfolio_py.trades) == len(portfolio_nb.trades)

    def test_fallback_for_volatility_sizing(self):
        """Test that volatility-based sizing falls back to Python correctly."""
        np.random.seed(222)
        n = 100
        dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
        prices = pd.Series(100 * (1 + np.random.randn(n) * 0.01).cumprod(), index=dates)

        entries = pd.Series([i == 10 for i in range(n)], index=dates)
        exits = pd.Series([i == 50 for i in range(n)], index=dates)

        # Volatility-based sizing should trigger fallback to Python
        risk_config = RiskConfig(
            position_sizing_method='volatility_based',
            use_stop_loss=False,
            position_size_pct=0.10
        )

        # This should use Python path (fallback) for both since volatility sizing
        # requires price history that Numba doesn't support
        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=False
        )

        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=True  # Should fallback
        )

        # Both should produce valid results
        assert portfolio_py.equity_curve is not None
        assert portfolio_nb.equity_curve is not None
        assert len(portfolio_py.equity_curve) == n
        assert len(portfolio_nb.equity_curve) == n

        # They should be identical since both use Python path
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="Volatility sizing fallback: Results should be identical"
        )

    def test_fallback_for_atr_stops(self):
        """Test that ATR-based stops fall back to Python correctly."""
        np.random.seed(333)
        n = 100
        dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
        prices = pd.Series(100 * (1 + np.random.randn(n) * 0.01).cumprod(), index=dates)

        # Create OHLC data for ATR calculation
        price_data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(n) * 0.002),
            'high': prices * (1 + np.abs(np.random.randn(n)) * 0.005),
            'low': prices * (1 - np.abs(np.random.randn(n)) * 0.005),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n)
        }, index=dates)

        entries = pd.Series([i == 20 for i in range(n)], index=dates)
        exits = pd.Series([i == 80 for i in range(n)], index=dates)

        # ATR-based stops should trigger fallback to Python
        risk_config = RiskConfig(
            use_stop_loss=True,
            stop_loss_type='atr',
            atr_multiplier=2.0,
            atr_lookback=14,
            position_size_pct=0.10
        )

        # Both should fall back to Python for ATR stops
        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, price_data=price_data,
            market_hours_only=False, use_numba=False
        )

        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, price_data=price_data,
            market_hours_only=False, use_numba=True  # Should fallback
        )

        # Both should produce valid results
        assert portfolio_py.equity_curve is not None
        assert portfolio_nb.equity_curve is not None

        # They should be identical since both use Python path
        np.testing.assert_allclose(
            portfolio_py.equity_curve.values,
            portfolio_nb.equity_curve.values,
            rtol=1e-10,
            err_msg="ATR stops fallback: Results should be identical"
        )

    def test_trade_fields_match(self):
        """Test that all trade fields match between Python and Numba."""
        np.random.seed(444)
        n = 100
        dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
        prices = pd.Series(100 * (1 + np.random.randn(n) * 0.015).cumprod(), index=dates)

        entries = pd.Series([i % 20 == 0 for i in range(n)], index=dates)
        exits = pd.Series([i % 20 == 10 for i in range(n)], index=dates)

        risk_config = RiskConfig(use_stop_loss=False, position_size_pct=0.10)

        portfolio_py = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=False
        )

        portfolio_nb = Portfolio(
            price=prices, entries=entries, exits=exits,
            init_cash=10000, fees=0.001, slippage=0.0005,
            risk_config=risk_config, market_hours_only=False, use_numba=True
        )

        assert len(portfolio_py.trades) == len(portfolio_nb.trades)
        assert len(portfolio_py.trades) > 0, "Need trades to compare"

        for i, (py_trade, nb_trade) in enumerate(zip(portfolio_py.trades, portfolio_nb.trades)):
            # Compare all common fields
            assert py_trade['type'] == nb_trade['type'], f"Trade {i}: type mismatch"
            np.testing.assert_allclose(
                py_trade['price'], nb_trade['price'], rtol=1e-10,
                err_msg=f"Trade {i}: price mismatch"
            )
            np.testing.assert_allclose(
                py_trade['shares'], nb_trade['shares'], rtol=1e-10,
                err_msg=f"Trade {i}: shares mismatch"
            )

            # For exit trades, compare P&L
            if 'pnl' in py_trade and 'pnl' in nb_trade:
                np.testing.assert_allclose(
                    py_trade['pnl'], nb_trade['pnl'], rtol=1e-6,
                    err_msg=f"Trade {i}: pnl mismatch"
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
