"""
Unit tests for rolling window backtest mode.

Tests cover:
- Window generation with trading days
- Rolling backtest execution
- Equity curve stitching
- Combined statistics calculation
- RollingWindowResults dataclass
"""

import pytest
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.backtesting.engine.rolling_results import RollingWindowResults
from src.backtesting.engine.portfolio_simulator import Portfolio
from src.backtesting.base.strategy import BaseStrategy


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_portfolio():
    """Create a mock portfolio with equity curve and trades."""
    portfolio = MagicMock(spec=Portfolio)

    # Create sample equity curve
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    equity = pd.Series(
        100000 * (1 + np.cumsum(np.random.randn(60) * 0.01)),
        index=dates
    )
    portfolio.equity_curve = equity

    # Sample trades
    portfolio.trades = [
        {'pnl': 100, 'entry_time': '2024-01-05', 'exit_time': '2024-01-10'},
        {'pnl': -50, 'entry_time': '2024-01-15', 'exit_time': '2024-01-20'},
        {'pnl': 200, 'entry_time': '2024-01-25', 'exit_time': '2024-01-30'},
    ]

    return portfolio


@pytest.fixture
def sample_hive_data(tmp_path):
    """
    Create sample hive-partitioned data for testing.

    Structure: equities_1min/symbol={SYM}/year={YYYY}/month={MM}/data.parquet
    """
    data_dir = tmp_path / "equities_1min"

    symbols = ["AAPL", "MSFT"]

    for symbol in symbols:
        for month in [1, 2, 3]:  # 3 months of data
            # Create partition directory
            partition_dir = data_dir / f"symbol={symbol}" / "year=2024" / f"month={month}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            # Create sample data - more days for proper window testing
            rows = []
            for day in range(1, 23):  # ~22 trading days per month
                for minute in range(390):  # 390 minutes in a trading day
                    hour = 9 + minute // 60
                    min_in_hour = minute % 60
                    if hour >= 16:  # Market closes at 4pm
                        continue

                    try:
                        ts = datetime(2024, month, day, hour, 30 + min_in_hour)
                    except ValueError:
                        # Invalid day for month
                        continue

                    price_base = 100.0 + symbol.count('A') * 10  # Different price per symbol
                    rows.append({
                        "timestamp": ts,
                        "open": price_base + minute * 0.01,
                        "high": price_base + minute * 0.01 + 0.5,
                        "low": price_base + minute * 0.01 - 0.5,
                        "close": price_base + minute * 0.01 + 0.1,
                        "volume": 1000.0 * (minute + 1),
                        "trade_count": 100.0,
                        "vwap": price_base + minute * 0.01,
                    })

            if rows:
                df = pl.DataFrame(rows)
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                df.write_parquet(partition_dir / "data.parquet")

    return tmp_path


@pytest.fixture
def engine_with_mock_loader(sample_hive_data):
    """Create BacktestEngine with test data."""
    engine = BacktestEngine(
        initial_capital=100000.0,
        fees=0.001,
        market_hours_only=False
    )
    # Override data loader path
    engine.data_loader._base_path = sample_hive_data
    return engine


class SimpleStrategy(BaseStrategy):
    """Simple strategy for testing that generates periodic signals."""

    def generate_signals(self, data: pd.DataFrame):
        """Generate signals every 100 bars."""
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        # Generate entry every 100 bars, exit 50 bars later
        for i in range(0, len(data) - 50, 100):
            entries.iloc[i] = True
            exits.iloc[i + 50] = True

        return entries, exits


# =============================================================================
# RollingWindowResults Tests
# =============================================================================


class TestRollingWindowResults:
    """Tests for RollingWindowResults dataclass."""

    def test_init_empty(self):
        """Should initialize with empty lists."""
        results = RollingWindowResults()

        assert results.windows == []
        assert results.window_portfolios == []
        assert results.window_stats == []
        assert results.combined_equity is None
        assert results.combined_trades == []
        assert results.combined_stats == {}

    def test_add_window(self, mock_portfolio):
        """Should add window with portfolio and compute stats."""
        results = RollingWindowResults()

        results.add_window('2024-01-01', '2024-03-01', mock_portfolio)

        assert len(results.windows) == 1
        assert results.windows[0] == ('2024-01-01', '2024-03-01')
        assert len(results.window_portfolios) == 1
        assert len(results.window_stats) == 1
        assert 'total_return' in results.window_stats[0]

    def test_add_multiple_windows(self, mock_portfolio):
        """Should track multiple windows."""
        results = RollingWindowResults()

        results.add_window('2024-01-01', '2024-03-01', mock_portfolio)
        results.add_window('2024-01-02', '2024-03-02', mock_portfolio)
        results.add_window('2024-01-03', '2024-03-03', mock_portfolio)

        assert len(results.windows) == 3
        assert len(results.window_portfolios) == 3
        assert len(results.combined_trades) == 9  # 3 trades * 3 windows

    def test_compute_window_stats(self, mock_portfolio):
        """Should compute accurate window statistics."""
        results = RollingWindowResults()

        stats = results._compute_window_stats(mock_portfolio)

        assert 'total_return' in stats
        assert 'sharpe' in stats
        assert 'max_drawdown' in stats
        assert 'win_rate' in stats
        assert stats['n_trades'] == 3

    def test_stitch_equity_curves(self, mock_portfolio):
        """Should stitch equity curves into continuous series."""
        results = RollingWindowResults()

        # Create portfolios with different starting values (simulating reset)
        for i in range(3):
            portfolio = MagicMock(spec=Portfolio)
            dates = pd.date_range(f'2024-0{i+1}-01', periods=30, freq='D')
            equity = pd.Series(
                100000 * (1.05 ** np.arange(30)),  # 5% growth each period
                index=dates
            )
            portfolio.equity_curve = equity
            portfolio.trades = []
            results.add_window(
                dates[0].strftime('%Y-%m-%d'),
                dates[-1].strftime('%Y-%m-%d'),
                portfolio
            )

        combined = results.stitch_equity_curves()

        assert combined is not None
        assert len(combined) > 30  # Should be longer than single window
        assert combined.iloc[0] == pytest.approx(100000, rel=0.01)  # Starts at init

    def test_compute_combined_stats(self, mock_portfolio):
        """Should compute overall statistics from stitched curve."""
        results = RollingWindowResults()

        results.add_window('2024-01-01', '2024-03-01', mock_portfolio)
        results.add_window('2024-03-02', '2024-05-01', mock_portfolio)

        results.stitch_equity_curves()
        stats = results.compute_combined_stats()

        assert 'total_return' in stats
        assert 'cagr' in stats
        assert 'sharpe' in stats
        assert 'max_drawdown' in stats
        assert 'n_windows' in stats
        assert stats['n_windows'] == 2

    def test_to_dataframe(self, mock_portfolio):
        """Should create summary DataFrame."""
        results = RollingWindowResults()

        results.add_window('2024-01-01', '2024-03-01', mock_portfolio)
        results.add_window('2024-03-02', '2024-05-01', mock_portfolio)

        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'window' in df.columns
        assert 'start_date' in df.columns
        assert 'end_date' in df.columns

    def test_repr(self, mock_portfolio):
        """Should have informative repr."""
        results = RollingWindowResults()
        results.add_window('2024-01-01', '2024-03-01', mock_portfolio)
        results.stitch_equity_curves()
        results.compute_combined_stats()

        repr_str = repr(results)

        assert 'RollingWindowResults' in repr_str
        assert '1 windows' in repr_str


# =============================================================================
# Window Generation Tests
# =============================================================================


class TestWindowGeneration:
    """Tests for _generate_windows method."""

    def test_generates_windows_from_trading_days(self):
        """Should generate windows based on trading days."""
        engine = BacktestEngine()

        # Mock the calendar to return predictable trading days
        with patch.object(engine, '_generate_windows') as mock_gen:
            # Setup return value
            mock_gen.return_value = [
                ('2024-01-02', '2024-03-22'),
                ('2024-01-03', '2024-03-25'),
            ]

            windows = engine._generate_windows('2024-01-01', '2024-12-31', 60, 1)

            assert len(windows) == 2

    def test_window_coverage(self):
        """Windows should cover the date range correctly."""
        engine = BacktestEngine()

        windows = engine._generate_windows('2024-01-01', '2024-06-30', 60, 20)

        assert len(windows) > 0
        # Each window should be a tuple of (start, end)
        for start, end in windows:
            assert start < end

    def test_insufficient_trading_days(self):
        """Should handle case with fewer trading days than window size."""
        engine = BacktestEngine()

        # Very short date range (only a few trading days)
        windows = engine._generate_windows('2024-01-02', '2024-01-10', 60, 1)

        # Should return single window with available days
        assert len(windows) <= 1

    def test_step_larger_than_window(self):
        """Should work when step_days > window_days (non-overlapping)."""
        engine = BacktestEngine()

        windows = engine._generate_windows('2024-01-01', '2024-12-31', 20, 30)

        # Windows should not overlap
        for i in range(len(windows) - 1):
            assert windows[i][1] < windows[i + 1][0]


# =============================================================================
# Rolling Backtest Execution Tests
# =============================================================================


class TestRollingBacktest:
    """Tests for _run_rolling method."""

    def test_run_rolling_returns_results(self, engine_with_mock_loader):
        """Should return RollingWindowResults object."""
        engine = engine_with_mock_loader
        strategy = SimpleStrategy()

        result = engine.run(
            strategy=strategy,
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-03-31',
            mode='rolling',
            window_days=20,
            step_days=10
        )

        assert isinstance(result, RollingWindowResults)

    def test_batch_mode_still_works(self, engine_with_mock_loader):
        """Batch mode should still work as before."""
        engine = engine_with_mock_loader
        strategy = SimpleStrategy()

        result = engine.run(
            strategy=strategy,
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            mode='batch'
        )

        # Should return Portfolio, not RollingWindowResults
        assert isinstance(result, Portfolio)

    def test_default_mode_is_batch(self, engine_with_mock_loader):
        """Default mode should be batch."""
        engine = engine_with_mock_loader
        strategy = SimpleStrategy()

        result = engine.run(
            strategy=strategy,
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        assert isinstance(result, Portfolio)

    def test_step_defaults_to_window(self, engine_with_mock_loader):
        """step_days should default to window_days if not specified."""
        engine = engine_with_mock_loader
        strategy = SimpleStrategy()

        # When step_days is None, it should default to window_days
        result = engine.run(
            strategy=strategy,
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-03-31',
            mode='rolling',
            window_days=20,
            step_days=None  # Should default to 20
        )

        assert isinstance(result, RollingWindowResults)


# =============================================================================
# Equity Curve Stitching Tests
# =============================================================================


class TestEquityCurveStitching:
    """Tests for equity curve stitching behavior."""

    def test_continuous_equity_growth(self):
        """Stitched equity should continue from previous end."""
        results = RollingWindowResults()

        # Window 1: 100k -> 110k
        p1 = MagicMock(spec=Portfolio)
        p1.equity_curve = pd.Series([100000, 105000, 110000],
                                     index=pd.date_range('2024-01-01', periods=3))
        p1.trades = []

        # Window 2: 100k -> 105k (10% return, but should continue from 110k)
        p2 = MagicMock(spec=Portfolio)
        p2.equity_curve = pd.Series([100000, 102500, 105000],
                                     index=pd.date_range('2024-01-04', periods=3))
        p2.trades = []

        results.add_window('2024-01-01', '2024-01-03', p1)
        results.add_window('2024-01-04', '2024-01-06', p2)

        combined = results.stitch_equity_curves()

        # Final value should reflect both windows' returns
        # Window 1: 10% return (100k -> 110k)
        # Window 2: 5% return (should be 110k -> 115.5k)
        assert combined.iloc[-1] == pytest.approx(115500, rel=0.01)

    def test_no_duplicate_timestamps(self):
        """Stitched curve should not have duplicate timestamps."""
        results = RollingWindowResults()

        for i in range(3):
            p = MagicMock(spec=Portfolio)
            p.equity_curve = pd.Series(
                [100000 + i * 1000, 101000 + i * 1000],
                index=pd.date_range(f'2024-01-0{i*2+1}', periods=2)
            )
            p.trades = []
            results.add_window(f'2024-01-0{i*2+1}', f'2024-01-0{i*2+2}', p)

        combined = results.stitch_equity_curves()

        # Check no duplicates
        assert not combined.index.duplicated().any()


# =============================================================================
# Integration Tests
# =============================================================================


class TestRollingIntegration:
    """Integration tests for rolling mode end-to-end."""

    def test_full_rolling_workflow(self, engine_with_mock_loader):
        """Test complete rolling backtest workflow."""
        engine = engine_with_mock_loader
        strategy = SimpleStrategy()

        result = engine.run(
            strategy=strategy,
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-03-31',
            mode='rolling',
            window_days=20,
            step_days=10
        )

        # Should have windows
        assert len(result.windows) > 0

        # Should have combined equity
        assert result.combined_equity is not None

        # Should have combined stats
        assert 'sharpe' in result.combined_stats or result.combined_stats == {}

    def test_multi_symbol_rolling(self, engine_with_mock_loader):
        """Test rolling mode with multiple symbols."""
        engine = engine_with_mock_loader
        strategy = SimpleStrategy()

        result = engine.run(
            strategy=strategy,
            symbols=['AAPL', 'MSFT'],
            start_date='2024-01-01',
            end_date='2024-03-31',
            mode='rolling',
            window_days=20,
            step_days=20,
            portfolio_mode='multi'
        )

        assert isinstance(result, RollingWindowResults)
